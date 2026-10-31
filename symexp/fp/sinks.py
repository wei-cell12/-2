# -*- coding: utf-8 -*-
"""视觉注意力汇（visual attention sink）：找出来、去掉，再重算读图分析。

注意力汇的定义沿用 "See What You Are Told: Visual Attention Sink in Large Multimodal Models"（ICLR 2025）：
  某些隐藏状态维度（汇维度 D_sink）上出现巨大激活的 token 会吸走大量注意力，但不携带图像内容。
  φ(x) = max_{d ∈ D_sink} |x[d]| / RMS(x)，φ ≥ τ（τ = 20）的图像 token 记为汇。
  该文的 D_sink 直接取自基础语言模型；我们的模型没有公开的 D_sink，所以从数据里找：
  每层输入隐藏状态里，提示第一个 token（语言模型的注意力汇）上 |x[d]|/RMS ≥ τ 的维度，
  在 ≥ 一半的（题, 层）里都出现的，记为 D_sink。

图像 token 在提示里排在用户的话之前，所以它们的隐藏状态在三种条件下完全相同：每题只需对中性提示做一次前向。

sinks（GPU）         每题一次前向（只到提示末尾），存每个记录层的输入隐藏状态的统计：
                     第一个 token 各维度 |x|/RMS、图像 token 各自最大的 8 个维度及数值
sinks-analyze（CPU） 定 D_sink → 每题每层的汇 token；描述它们（个数、在不在关键区域、在图上哪里）；
                     去掉汇、在剩下的图像 token 里重新归一化，重算：
                       看图头    中性回答前 512 个 token 里非汇图像注意力最多的 96 个头
                       关键区域  中性参照（思考阶段）里，非汇图像块中注意力最高的 20%
                     然后按原样重算三项：
                       1. 关键区域在三种条件下还是不是关键区域（落在它上面的比例、三种条件前 20% 块的重合度）
                       2. 中性 vs 说错：看图分布超出噪声的量，按用户说错时的结果分组（守住 / 采纳 / 别的错）
                       3. 说错 − 说对：关键区域比例变化，按用户时间能否对上已有指针分组；
                          对不上那组，注意力是否转向用户时间所指的位置
"""

import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import E_FRACTION, LAYERS, N_HEADS_SELECTED
from .sources import (CONDS, OUTCOME, READ_GROUP, READ_WINDOWS, SourcesCheckFailed, _norm_img, _reading_window,
                      claim_far_hands, dial_geometry, hand_angles, sector_mask)

TAU = 20.0
TOPK_DIMS = 8
DSINK_MIN_FRAC = 0.5
PERM_REPS = 20000


# ---------------------------------------------------------------- GPU：隐藏状态统计
def sinks_one(backend, item: dict, layers: List[int]) -> Dict[str, np.ndarray]:
    from .. import attn
    from .generate import build_inputs
    torch, model = backend.torch, backend.model
    _, inp = build_inputs(backend, item["image"], item["messages"]["neutral"])
    ids = inp["input_ids"][0].tolist()
    img_tok = attn.image_token_id(backend)
    img = np.array([i for i, x in enumerate(ids) if x == img_tok])
    if len(img) == 0:
        raise SourcesCheckFailed(f"{item['id']}：提示里没有图像 token")
    dev = model.device
    batch = {k: v.to(dev) for k, v in inp.items() if hasattr(v, "to")}
    if "pixel_values" in batch:
        batch["pixel_values"] = batch["pixel_values"].to(model.dtype)
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "rope_deltas"):
        inner.rope_deltas = None
    with torch.inference_mode():
        try:
            out = model(**batch, use_cache=False, output_hidden_states=True, return_dict=True, logits_to_keep=1)
        except TypeError:
            out = model(**batch, use_cache=False, output_hidden_states=True, return_dict=True)
    hs = out.hidden_states                                       # (嵌入, 第 0 层输出, …)：hs[l] = 第 l 层的输入
    first, top_i, top_v = [], [], []
    for l in layers:
        x = hs[l][0].float()                                     # [T, D]
        xn = (x / x.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-12)).abs()
        first.append(xn[0].cpu().numpy().astype(np.float16))
        v, i = xn[torch.as_tensor(img, device=xn.device)].topk(TOPK_DIMS, dim=-1)
        top_i.append(i.cpu().numpy().astype(np.int32))
        top_v.append(v.cpu().numpy().astype(np.float16))
    return {"layers": np.array(layers), "first": np.stack(first), "img_top_idx": np.stack(top_i),
            "img_top_val": np.stack(top_v), "image_pos": img}


def run_sinks(cfg, cq: str, root: str, shard: str = "") -> int:
    from .compare import _reference
    from .conflict import _engine, _items
    from .pipeline import _atomic_json, _atomic_npz, _check_no_failure, _fail, _shard
    _check_no_failure(root)
    items = _items(cq)
    d = os.path.join(root, "sinks")
    os.makedirs(d, exist_ok=True)
    todo = [i for i in _shard(sorted(items), shard) if not os.path.exists(os.path.join(d, f"{i}.npz"))]
    print(f"[sinks] 本进程待做 {len(todo)} 题 → {d}", file=sys.stderr)
    if not todo:
        return 0
    eng = _engine(cfg)
    for n, iid in enumerate(todo):
        try:
            ref = _reference(cq, root, iid)
            layers = [int(x) for x in ref.get("layer_ids", LAYERS)]
            arr = sinks_one(eng.backend, items[iid], layers)
            if len(arr["image_pos"]) != int(ref["n_img"]):
                raise SourcesCheckFailed(f"{iid}：图像 token {len(arr['image_pos'])} 个，与参照的 {int(ref['n_img'])} 个不一致")
            _atomic_npz(os.path.join(d, f"{iid}.npz"), arr)
            mx = float(arr["first"].max())
            _atomic_json(os.path.join(d, f"{iid}.json"), {"id": iid, "layers": layers, "n_img": int(len(arr["image_pos"])),
                                                          "first_token_max": mx})
            print(f"  {n + 1}/{len(todo)} {iid}：第一个 token 的最大 |x|/RMS = {mx:.1f}", file=sys.stderr)
        except Exception as e:
            _fail(root, f"sinks_{iid}", e)
            raise
    return len(todo)


# ---------------------------------------------------------------- 纯函数
def find_dsink(firsts: List[np.ndarray], tau: float = TAU, min_frac: float = DSINK_MIN_FRAC) -> Tuple[List[int], Dict[int, float]]:
    """firsts：每题 [L, D] 的第一个 token |x|/RMS。返回 (D_sink, {维度: 出现比例})。"""
    hit = np.concatenate([f >= tau for f in firsts], 0)           # [题×层, D]
    frac = hit.mean(0)
    dims = [int(d) for d in np.flatnonzero(frac >= min_frac)]
    return dims, {int(d): float(frac[d]) for d in np.argsort(-frac)[:10] if frac[d] > 0}


def sink_mask(top_idx: np.ndarray, top_val: np.ndarray, dsink: List[int], tau: float = TAU) -> Tuple[np.ndarray, int]:
    """top_idx/top_val: [L, n, k]。返回 ([L, n] 汇 token，判不了的个数）。
    某 token 的汇维度不在它最大的 k 个维度里：若第 k 大的值 < τ，则汇维度上必 < τ（不是汇）；否则判不了（计数）。"""
    ds = np.array(dsink)
    inside = np.isin(top_idx, ds)                                  # [L, n, k]
    phi = np.where(inside, top_val.astype(np.float32), 0).max(-1)
    sink = phi >= tau
    unsure = (~inside.any(-1)) & (top_val[..., -1].astype(np.float32) >= tau)
    return sink, int(unsure.sum())


def drop_sinks(dist: np.ndarray, sink: np.ndarray) -> np.ndarray:
    """dist: [..., L, H, n]；sink: [L, n]。汇 token 置 0（之后再归一化）。"""
    return dist * (~sink)[:, None, :]


def key_region(r: np.ndarray, sink: np.ndarray, frac: float = E_FRACTION) -> np.ndarray:
    """r: [L, H, n] 中性思考阶段的看图分布；在非汇图像块里取注意力最高的 frac。"""
    L, H, n = r.shape
    E = np.zeros(r.shape, bool)
    for l in range(L):
        ok = np.flatnonzero(~sink[l])
        k = max(1, int(math.ceil(frac * len(ok))))
        for h in range(H):
            E[l, h, ok[np.argsort(-r[l, h, ok], kind="stable")[:k]]] = True
    return E


def image_heads(neutral_dist_sum: np.ndarray, sink: np.ndarray, n_heads: int = N_HEADS_SELECTED) -> np.ndarray:
    """neutral_dist_sum: [G, L, H, n]（中性回答思考开头 token 的看图注意力之和）→ 非汇图像注意力最多的 n_heads 个头。"""
    m = drop_sinks(neutral_dist_sum.sum(0), sink).sum(-1)          # [L, H]
    hm = np.zeros(m.shape, bool)
    hm.ravel()[np.argsort(-m.ravel(), kind="stable")[:n_heads]] = True
    return hm


def topk_overlap(P: np.ndarray, Q: np.ndarray, sink: np.ndarray, frac: float = E_FRACTION) -> np.ndarray:
    """两张看图分布 [L, H, n] 各自非汇块里前 frac 的集合的 Jaccard，[L, H]。"""
    a, b = key_region(P, sink, frac), key_region(Q, sink, frac)
    return (a & b).sum(-1) / np.maximum((a | b).sum(-1), 1)


def perm_test(a, b, reps: int = PERM_REPS, seed: int = 0) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    x = np.concatenate([a, b])
    obs = abs(a.mean() - b.mean())
    idx = np.argsort(np.random.default_rng(seed).random((reps, len(x))), axis=1)
    d = np.abs(x[idx[:, :len(a)]].mean(1) - x[idx[:, len(a):]].mean(1))
    return float((np.sum(d >= obs - 1e-15) + 1) / (reps + 1))


# ---------------------------------------------------------------- CPU：分析
def run_sinks_analyze(cq: str, root: str, judgments: str, out_md: Optional[str] = None, out_json: Optional[str] = None,
                      tau: float = TAU) -> dict:
    from .compare import _reference
    from .conflict import _items
    from .pipeline import _atomic_json
    items = _items(cq)
    J = json.load(open(judgments, encoding="utf-8"))
    S = {}
    for iid in sorted(items):
        p = os.path.join(root, "sinks", f"{iid}.npz")
        if os.path.exists(p):
            with np.load(p) as z:
                S[iid] = {k: z[k] for k in z.files}
    if not S:
        raise SourcesCheckFailed("没有 sinks 阶段的结果")
    dsink, freq = find_dsink([S[i]["first"].astype(np.float32) for i in S], tau)
    if not dsink:
        raise SourcesCheckFailed(f"没有在 ≥ {DSINK_MIN_FRAC:.0%} 的（题, 层）里都出现的汇维度；前几个维度的出现比例：{freq}")
    rows, unsure_total = [], 0
    for iid in sorted(S):
        Z = {}
        for c in CONDS:
            p = os.path.join(root, "reading", f"{iid}__{c}.npz")
            if not os.path.exists(p):
                break
            with np.load(p) as z:
                Z[c] = (z["dist_sum"].astype(np.float64), z["mass_sum"].astype(np.float64), z["count"])
        if len(Z) < 3:
            continue
        ref = _reference(cq, root, iid)
        sink, unsure = sink_mask(S[iid]["img_top_idx"], S[iid]["img_top_val"], dsink, tau)
        unsure_total += unsure
        n = sink.shape[1]
        E0 = np.asarray(ref["E_mask"][0], bool)
        hm0 = np.asarray(ref["heads"], bool)
        r0 = np.asarray(ref["r"][0], np.float64)
        E1 = key_region(r0, sink)
        hm1 = image_heads(Z["neutral"][0], sink)
        Z1 = {c: (drop_sinks(Z[c][0], sink), Z[c][1], Z[c][2]) for c in CONDS}
        it = items[iid]
        grid = ref["grid_hw"] if "grid_hw" in ref else None
        if grid is None:
            rec = json.load(open(os.path.join(cq, "obs", "records", f"{iid}__neutral.json"), encoding="utf-8"))
            npz = rec.get("npz") or os.path.join(cq, "obs", "attn", f"{iid}__neutral.npz")
            if os.path.exists(npz):
                with np.load(npz) as z:
                    grid = z["grid_hw"]
        hands = claim_m = dial_in = None
        far = claim_far_hands(it["gold"], it["claim"])
        if grid is not None and int(np.prod(grid)) == n:
            geom = dial_geometry(it["image"])
            hands = sector_mask(geom, grid, hand_angles(it["gold"]))
            claim_m = sector_mask(geom, grid, far) & ~hands if far else np.zeros(n, bool)
            cx, cy, R, W, H = geom
            gh, gw = int(grid[0]), int(grid[1])
            yy, xx = np.divmod(np.arange(n), gw)
            dial_in = np.hypot((xx + 0.5) / gw * W - cx, (yy + 0.5) / gh * H - cy) <= 1.05 * R
        full = min(int((Z[c][2] == READ_GROUP).sum()) for c in CONDS)
        j = J.get(f"obs|{iid}|belief")
        jn = J.get(f"obs|{iid}|neutral")
        rows.append({"id": iid, "sink": sink, "E0": E0, "E1": E1, "hm0": hm0, "hm1": hm1, "Z": Z, "Z1": Z1, "full": full,
                     "outcome": OUTCOME.get(j["outcome"], "未") if j else "未",
                     "neutral": OUTCOME.get(jn["outcome"], "未") if jn else "未",
                     "group": "不同" if far else "重合", "hands": hands, "claim_m": claim_m, "dial_in": dial_in,
                     "neutral_mass_sink_share": float((Z["neutral"][0].sum(0) * sink[:, None, :]).sum(-1)[hm0].sum()
                                                      / max(Z["neutral"][0].sum(0).sum(-1)[hm0].sum(), 1e-12))})
    A = np.array
    L = [f"# 去掉视觉注意力汇后重算读图分析", "",
         f"汇维度 D_sink = {dsink}（提示第一个 token 上 |x|/RMS ≥ {tau:g} 的维度，在 ≥ {DSINK_MIN_FRAC:.0%} 的（题, 层）里出现；"
         f"出现比例最高的维度：{ {k: round(v, 2) for k, v in freq.items()} }）。"
         f"图像 token 在这些维度上 |x|/RMS ≥ {tau:g} 记为汇。判不了的（题, 层, token）共 {unsure_total} 个。", "",
         f"题数 {len(rows)}。", "", "## 0. 汇 token 是什么样的", "",
         "| 指标 | 数值 |", "|---|---|"]
    ns = A([r["sink"].sum(1).mean() for r in rows])
    nimg = rows[0]["sink"].shape[1]
    L.append(f"| 每层汇 token 个数（题平均，共 {nimg} 个图像 token） | 中位 {np.median(ns):.1f}，最多 {ns.max():.1f} |")
    L.append(f"| 中性回答思考开头 512 个 token：原看图头的看图注意力里落在汇上的比例 | "
             f"{np.mean([r['neutral_mass_sink_share'] for r in rows]):.3f} |")
    e_sink = A([(r["E0"] & r["sink"][:, None, :]).sum(-1)[r["hm0"]].mean() / max(r["E0"].sum(-1)[r["hm0"]].mean(), 1)
                for r in rows])
    L.append(f"| 原关键区域（看图头平均）里是汇 token 的比例 | {e_sink.mean():.3f} |")
    hv = [r for r in rows if r["hands"] is not None]
    if hv:
        sh = A([(r["sink"] & r["hands"]).sum() / max(r["sink"].sum(), 1) for r in hv])
        so = A([(r["sink"] & ~r["dial_in"]).sum() / max(r["sink"].sum(), 1) for r in hv])
        L.append(f"| 汇 token 落在真实指针扇形里的比例 / 表盘外的比例 | {sh.mean():.3f} / {so.mean():.3f} |")
    res = {"dsink": dsink, "dsink_freq": freq, "unsure": unsure_total, "items": [r["id"] for r in rows], "windows": {}}

    def share(r, c, G, E, hm, Zk="Z1"):
        P = _norm_img(r[Zk][c][0][:G].sum(0))
        return float((P * E).sum(-1)[hm].mean())
    L += ["", "## 1. 关键区域在三种条件下还是不是关键区域（去掉汇后，新的看图头和关键区域）", "",
          "比例 = 看图分布（去掉汇、在剩下的图像块里归一化）落在关键区域的比例；均匀分布时为 20%。"
          "重合度 = 两种条件各自前 20% 非汇图像块的 Jaccard（看图头平均）。", "",
          "| 前 W 个 token | 题数 | 去掉汇前（原关键区域、原看图头）：中性 / 说错 / 说对 | 去掉汇后：中性 / 说错 / 说对 | "
          "前 20% 块重合度：中性−说错 / 中性−说对 / 说错−说对 | 对照：同一条中性回答两半之间的重合度 |", "|---|---|---|---|---|---|"]
    for W in READ_WINDOWS:
        G = W // READ_GROUP
        P = [r for r in rows if r["full"] >= G]
        if len(P) < 3:
            continue
        sh = {c: A([share(r, c, G, r["E1"], r["hm1"]) for r in P]) for c in CONDS}
        sh0 = {c: A([share(r, c, G, r["E0"], r["hm0"], "Z") for r in P]) for c in CONDS}
        ov = {}
        for a, b in [("neutral", "belief"), ("neutral", "belief_right"), ("belief", "belief_right")]:
            ov[(a, b)] = A([topk_overlap(_norm_img(r["Z1"][a][0][:G].sum(0)), _norm_img(r["Z1"][b][0][:G].sum(0)),
                                         r["sink"])[r["hm1"]].mean() for r in P])
        half = A([topk_overlap(_norm_img(r["Z1"]["neutral"][0][:G:2].sum(0)), _norm_img(r["Z1"]["neutral"][0][1:G:2].sum(0)),
                               r["sink"])[r["hm1"]].mean() for r in P])
        L.append(f"| {W} | {len(P)} | {sh0['neutral'].mean():.3f} / {sh0['belief'].mean():.3f} / {sh0['belief_right'].mean():.3f} | "
                 f"{sh['neutral'].mean():.3f} / {sh['belief'].mean():.3f} / {sh['belief_right'].mean():.3f} | "
                 + " / ".join(f"{v.mean():.3f}" for v in ov.values()) + f" | {half.mean():.3f} |")
    groups = [("对", "守住正确答案"), ("采", "采纳用户的时间"), ("他", "答成别的错误时间")]
    for scope, keep in [("全部题", lambda r: True), ("中性时答对的题", lambda r: r["neutral"] == "对")]:
        L += ["", f"## 2. 中性 vs 用户说错：看图分布超出噪声的量，按用户说错时的结果分组（{scope}；去掉汇前 → 去掉汇后）", "",
              "每题先对看图头平均（去掉汇前用原看图头，去掉汇后用新看图头）。倍数 = 该组 ÷ 守住组；p = 与守住组比较的置换检验（按题）。", "",
              "| 前 W 个 token | " + " | ".join(g[1] for g in groups) + " | 采纳 ÷ 守住（p） | 别的错 ÷ 守住（p） |",
              "|---|---|---|---|---|---|"]
        for W in READ_WINDOWS:
            G = W // READ_GROUP
            P = [r for r in rows if r["full"] >= G and keep(r)]
            if len(P) < 3:
                continue
            val = {}
            for tag, Zk, Ek, hk in (("前", "Z", "E0", "hm0"), ("后", "Z1", "E1", "hm1")):
                for g, _ in groups:
                    v = []
                    for r in P:
                        if r["outcome"] != g:
                            continue
                        st = _reading_window(r[Zk], r[Ek], G)
                        e = st[("neutral", "belief")]["half"] - 0.5 * (st["within"]["neutral"] + st["within"]["belief"])
                        v.append(float(e[r[hk]].mean()))
                    val[(tag, g)] = A(v)
            cells = [f"{val[('前', g)].mean():+.4f} → {val[('后', g)].mean():+.4f}（{len(val[('后', g)])} 题）"
                     if len(val[("后", g)]) else "—" for g, _ in groups]
            rat = []
            for g in ("采", "他"):
                b0, b1, v0, v1 = val[("前", "对")], val[("后", "对")], val[("前", g)], val[("后", g)]
                if len(v1) and len(b1) and b0.mean() > 0 and b1.mean() > 0:
                    rat.append(f"{v0.mean() / b0.mean():.1f} → {v1.mean() / b1.mean():.1f} 倍（p={perm_test(v1, b1):.3f}）")
                else:
                    rat.append("—")
            L.append(f"| {W} | " + " | ".join(cells + rat) + " |")
            res["windows"].setdefault(str(W), {})[scope] = {f"{t}_{g}": val[(t, g)].tolist() for (t, g) in val}
    L += ["", "## 3. 说错 − 说对：关键区域比例的变化，按用户时间能否对上已有指针分组（去掉汇前 → 去掉汇后）", "",
          "重合组：用户时间的每根针离某根真实指针 ≤ 35°；不同组：至少一根针离所有真实指针 > 35°。"
          "最后一列只看不同组：落在用户时间那根针所指位置（去掉与真实指针重合的部分）的比例，以及这块区域占全部非汇图像块的比例。", "",
          "| 前 W 个 token | 重合组 | 不同组 | 不同组：用户时间所指位置的比例 说错 / 说对（面积占比） |", "|---|---|---|---|"]
    for W in READ_WINDOWS:
        G = W // READ_GROUP
        P = [r for r in rows if r["full"] >= G]
        if len(P) < 3:
            continue
        cells = []
        for grp in ("重合", "不同"):
            S0 = [r for r in P if r["group"] == grp]
            if len(S0) < 2:
                cells.append("—")
                continue
            d0 = A([share(r, "belief", G, r["E0"], r["hm0"], "Z") - share(r, "belief_right", G, r["E0"], r["hm0"], "Z") for r in S0])
            d1 = A([share(r, "belief", G, r["E1"], r["hm1"]) - share(r, "belief_right", G, r["E1"], r["hm1"]) for r in S0])
            cells.append(f"{d0.mean() * 100:+.2f} → {d1.mean() * 100:+.2f} 个百分点（{len(S0)} 题）")
        F = [r for r in P if r["group"] == "不同" and r["claim_m"] is not None and (r["claim_m"] & ~r["sink"].any(0)).any()]
        if F:
            cm = lambda r: (r["claim_m"][None, :] & ~r["sink"])                       # [L, n]
            sb = A([float((_norm_img(r["Z1"]["belief"][0][:G].sum(0)) * cm(r)[:, None, :]).sum(-1)[r["hm1"]].mean()) for r in F])
            sr = A([float((_norm_img(r["Z1"]["belief_right"][0][:G].sum(0)) * cm(r)[:, None, :]).sum(-1)[r["hm1"]].mean()) for r in F])
            area = A([cm(r).sum(1).mean() / max((~r["sink"]).sum(1).mean(), 1) for r in F])
            from .sources import _signflip_p
            cells.append(f"{sb.mean():.3f} / {sr.mean():.3f}（{area.mean():.3f}；{len(F)} 题，配对 p={_signflip_p(sb - sr):.3f}）")
        else:
            cells.append("—")
        L.append(f"| {W} | " + " | ".join(cells) + " |")
    out_md = out_md or os.path.join(root, "sinks_report.md")
    out_json = out_json or os.path.join(root, "sinks_summary.json")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    _atomic_json(out_json, res)
    print(f"[sinks-analyze] D_sink = {dsink}，{len(rows)} 题 → {out_md}", file=sys.stderr)
    return res
