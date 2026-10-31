# -*- coding: utf-8 -*-
"""思考开头 token 的看图分布：按用户说错时的最终结果分组比较（ClockQA 与 PathVQA 同一套算法）。

数据：fp-sources reading 存下的 $S/reading/<题>__<条件>.npz（思考前至多 512 个 token，每 8 个一组的看图注意力之和）
     和每题的中性参照 $S/refs/<题>.npz（看图头 96 个、每个头的关键区域 E = 中性思考阶段注意力最高的 20% 图像块）。

每个前 W 个 token、每题、每个看图头：
  看图分布 P        前 W 个 token 落在图像 token 上的注意力加总，再在图像内部归一化
  超出噪声的量      JSD(中性半份, 说错半份) − 两条回答各自两半之间 JSD 的平均（半份 = 按 8 个一组交替分）
  RRAR             相关区域的平均注意力 / 全图平均注意力 = P(区域) × n / 区域大小（与 CVPR 2026
                   "Deeper Thought, Weaker Aim" 的定义相同；> 1 表示比全图平均更集中在该区域）
                   区域有两种：E（中性时注意力最高的 20% 块，两个数据集都有）；
                   hands（真实指针方向 ±15° 的扇形，只有 ClockQA 有，相当于论文里的标注框）
然后对这道题的 96 个看图头取平均，得到每题一个数；组间比较用按题的置换检验（打乱分组，20000 次，双侧）。

fp-model2 生成的记录（PathVQA、Gemma 4 ClockQA）没有存生成时的注意力，所以 reading-gen 先把中性回答整条回放一次建参照，
再与 ClockQA 一样回放三种条件的思考开头 token。
"""

import json
import os
import sys
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .sources import (CONDS, READ_GROUP, READ_WINDOWS, SourcesCheckFailed, _norm_img, _reading_window,
                      reading_one)

PERM_REPS = 20000
CQ_GROUPS = [("对", "守住正确答案"), ("采", "采纳用户的答案"), ("他", "答成别的错误时间")]
PV_GROUPS = [("对", "守住正确答案"), ("采", "采纳用户的答案"), ("未", "不下结论")]


# ---------------------------------------------------------------- 结果
def outcomes_clock(judgments: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """ClockQA 时间判读 → ({题: 用户说错时的结果}, {题: 中性时的结果})，结果为 对 / 采 / 他 / 未。"""
    from .sources import OUTCOME
    J = json.load(open(judgments, encoding="utf-8"))
    bel, neu = {}, {}
    for k, v in J.items():
        src, iid, c = k.split("|")
        if src != "obs":
            continue
        o = OUTCOME.get(v.get("outcome"), "未")
        if c == "belief":
            bel[iid] = o
        elif c == "neutral":
            neu[iid] = o
    return bel, neu


def outcomes_yesno(yn_judgments: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """PathVQA 是/否判读（yn-summary 的 judgments.json）→ 同上；错 = 采纳用户的答案（是/否题只有两个选项）。"""
    rows = json.load(open(yn_judgments, encoding="utf-8"))
    m = {"对": "对", "错": "采", "未定": "未"}
    bel, neu = {}, {}
    for r in rows:
        if r["arm"] != "base":
            continue
        if r["condition"] == "belief":
            bel[r["id"]] = m[r["status"]]
        elif r["condition"] == "neutral":
            neu[r["id"]] = m[r["status"]]
    return bel, neu


# ---------------------------------------------------------------- 纯函数
def rrar(P: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """P: [..., n]（图像内部已归一化）；mask: 与 P 末维对齐（可按头不同）。返回 [...]。"""
    mask = np.broadcast_to(mask, P.shape)
    k = mask.sum(-1)
    if (k == 0).any():
        raise SourcesCheckFailed("RRAR 的区域为空")
    return (P * mask).sum(-1) * P.shape[-1] / k


def perm_test(a: np.ndarray, b: np.ndarray, reps: int = PERM_REPS, seed: int = 0) -> float:
    """两组均值差的置换检验（双侧）。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    x = np.concatenate([a, b])
    obs = abs(a.mean() - b.mean())
    g = np.random.default_rng(seed)
    idx = np.argsort(g.random((reps, len(x))), axis=1)
    sa = x[idx[:, :len(a)]].mean(1)
    sb = x[idx[:, len(a):]].mean(1)
    return float((np.sum(np.abs(sa - sb) >= obs - 1e-15) + 1) / (reps + 1))


def item_stats(Z: dict, E: np.ndarray, hm: np.ndarray, G: int, hands: Optional[np.ndarray] = None) -> dict:
    """一道题、前 G 组（G×8 个 token）：每个条件的 RRAR、中性 vs 说错的超出噪声量（都已对看图头平均）。"""
    st = _reading_window(Z, E, G)
    out = {"excess": float((st[("neutral", "belief")]["half"]
                            - 0.5 * (st["within"]["neutral"] + st["within"]["belief"]))[hm].mean())}
    for c in CONDS:
        P = _norm_img(Z[c][0][:G].sum(0))                      # [L, H, n]
        out[f"rrar_E_{c}"] = float(rrar(P, E)[hm].mean())
        if hands is not None and hands.any():
            out[f"rrar_hands_{c}"] = float(rrar(P, hands)[hm].mean())
    return out


# ---------------------------------------------------------------- 汇总
def _load(root: str, iid: str) -> Optional[dict]:
    Z = {}
    for c in CONDS:
        p = os.path.join(root, "reading", f"{iid}__{c}.npz")
        if not os.path.exists(p):
            return None
        with np.load(p) as z:
            Z[c] = (z["dist_sum"].astype(np.float64), z["mass_sum"].astype(np.float64), z["count"])
    return Z


def _hands_mask(cq: str, item: dict, ref: Optional[dict] = None) -> Optional[np.ndarray]:
    from .sources import dial_geometry, hand_angles, sector_mask
    if ref is not None and "grid_hw" in ref:
        grid_hw = ref["grid_hw"]
    else:
        rec = json.load(open(os.path.join(cq, "obs", "records", f"{item['id']}__neutral.json"), encoding="utf-8"))
        npz = rec.get("npz") or os.path.join(cq, "obs", "attn", f"{item['id']}__neutral.npz")
        if not os.path.exists(npz):
            return None
        with np.load(npz) as z:
            grid_hw = z["grid_hw"]
    return sector_mask(dial_geometry(item["image"]), grid_hw, hand_angles(item["gold"]))


def run_groups(cq: str, root: str, bel: Dict[str, str], neu: Dict[str, str], groups: List[Tuple[str, str]],
               dataset: str, out_md: Optional[str] = None, out_json: Optional[str] = None) -> dict:
    from .compare import _reference
    from .conflict import _items
    from .pipeline import _atomic_json
    items = _items(cq)
    per = {W: [] for W in READ_WINDOWS}
    has_hands = False
    for iid in sorted(items):
        Z = _load(root, iid)
        if Z is None or iid not in bel:
            continue
        ref = _reference(cq, root, iid)
        E = np.asarray(ref["E_mask"][0], bool)
        hm = np.asarray(ref["heads"], bool)
        hands = _hands_mask(cq, items[iid], ref) if dataset == "clockqa" else None
        if hands is not None:
            if hands.shape[0] != E.shape[-1]:
                raise SourcesCheckFailed(f"{iid}：指针区域 {hands.shape[0]} 块与图像 token {E.shape[-1]} 个不一致")
            has_hands = has_hands or bool(hands.any())
        full = min(int((Z[c][2] == READ_GROUP).sum()) for c in CONDS)
        for W in READ_WINDOWS:
            G = W // READ_GROUP
            if full >= G:
                per[W].append({"id": iid, "belief": bel[iid], "neutral": neu.get(iid, "未"),
                               **item_stats(Z, E, hm, G, hands)})
    if not any(per.values()):
        raise SourcesCheckFailed("没有三种条件都读到思考开头 token 的题")
    regions = ["E"] + (["hands"] if has_hands else [])
    rname = {"E": "关键区域 E（中性时注意力最高的 20% 块）", "hands": "真实指针区域（指针方向 ±15° 扇形）"}
    A = np.array
    L = [f"# 思考开头 token 的看图：按用户说错时的结果分组（{dataset}）", "",
         "每题一个数（先对这道题的 96 个看图头平均）。超出量 = 中性 vs 说错的半份 JSD − 两条回答各自两半之间 JSD 的平均；"
         "RRAR = 区域内平均注意力 / 全图平均注意力。组间 p 为与“守住正确答案”组比较的置换检验（按题，双侧）。", ""]
    res = {"dataset": dataset, "windows": {}}
    scopes = [("全部题", lambda x: True), ("中性时答对的题", lambda x: x["neutral"] == "对")]
    for scope, keep in scopes:
        L += [f"## {scope}", "", "### 1. 中性 vs 用户说错：看图分布超出噪声的量", "",
              "| 前 W 个 token | " + " | ".join(g[1] for g in groups) + " | " +
              " | ".join(f"{g[1]} ÷ 守住（p）" for g in groups[1:]) + " |",
              "|---" * (1 + 2 * len(groups) - 1) + "|"]
        for W in READ_WINDOWS:
            P = [x for x in per[W] if keep(x)]
            if not P:
                continue
            vals = {g: A([x["excess"] for x in P if x["belief"] == g]) for g, _ in groups}
            base = vals["对"]
            cells = [f"{v.mean():+.4f}（{len(v)} 题）" if len(v) else "—" for v in vals.values()]
            ratio = []
            for g, _ in groups[1:]:
                v = vals[g]
                if len(v) and len(base) and base.mean() > 0:
                    ratio.append(f"{v.mean() / base.mean():.1f} 倍（p={perm_test(v, base):.3f}）")
                elif len(v) and len(base):
                    ratio.append(f"守住组 ≤ 0（p={perm_test(v, base):.3f}）")
                else:
                    ratio.append("—")
            L.append(f"| {W} | " + " | ".join(cells + ratio) + " |")
            res["windows"].setdefault(str(W), {})[scope] = {"items": P}
        for reg in regions:
            L += ["", f"### 2. RRAR：{rname[reg]}", "",
                  "每格：中性 → 说错 → 说对；括号里是 说错 − 中性 的变化，以及这个变化与“守住”组相比的 p。", "",
                  "| 前 W 个 token | " + " | ".join(g[1] for g in groups) + " |", "|---" * (1 + len(groups)) + "|"]
            for W in READ_WINDOWS:
                P = [x for x in per[W] if keep(x) and f"rrar_{reg}_neutral" in x]
                if not P:
                    continue
                d0 = A([x[f"rrar_{reg}_belief"] - x[f"rrar_{reg}_neutral"] for x in P if x["belief"] == "对"])
                cells = []
                for g, _ in groups:
                    S = [x for x in P if x["belief"] == g]
                    if not S:
                        cells.append("—")
                        continue
                    n0, b, r = (A([x[f"rrar_{reg}_{c}"] for x in S]).mean() for c in CONDS)
                    d = A([x[f"rrar_{reg}_belief"] - x[f"rrar_{reg}_neutral"] for x in S])
                    p = "" if g == "对" else f"，p={perm_test(d, d0):.3f}"
                    cells.append(f"{n0:.2f} → {b:.2f} → {r:.2f}（{d.mean():+.3f}{p}；{len(S)} 题）")
                L.append(f"| {W} | " + " | ".join(cells) + " |")
        L.append("")
    out_md = out_md or os.path.join(root, "reading_groups_report.md")
    out_json = out_json or os.path.join(root, "reading_groups_summary.json")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    _atomic_json(out_json, res)
    print(f"[reading-groups] {dataset}：各长度的题数 {[len(per[W]) for W in READ_WINDOWS]} → {out_md}", file=sys.stderr)
    return res


# ---------------------------------------------------------------- fp-model2 记录的回放（PathVQA、Gemma 4 ClockQA 等）
def record_layers(model) -> Tuple[Optional[list], Optional[Callable]]:
    """要记录的层：有滑动窗口层的模型（Gemma 4）只用全局层——滑动窗口层看不到滑出窗口的图像 token，
    两种条件的提示长度不同，滑出的位置也不同，看图分布会因此不可比。没有滑动窗口层时用全部标准注意力层。"""
    from .. import attn
    mods = [m for _, m in attn.find_targets(model)]
    if any(getattr(m, "is_sliding", False) for m in mods):
        keep = lambda m: not getattr(m, "is_sliding", False)
        return sorted(int(m.layer_idx) for m in mods if keep(m)), keep
    return None, None


def think_skip(tok, cfg, output_ids: List[int]) -> int:
    """思考开头标记由模型自己写时（think_prefill: false），输出开头属于标记的 token 数（含紧跟的换行）。"""
    if cfg.think_prefill:
        return 0
    want = cfg.think_open.strip()
    for m in range(1, min(len(output_ids), 12) + 1):
        if tok.decode(output_ids[:m], skip_special_tokens=False).strip() == want:
            if m < len(output_ids) and tok.decode(output_ids[m:m + 1], skip_special_tokens=False).strip() == "":
                m += 1
            return m
    raise SourcesCheckFailed(f"输出开头不是思考标记 {want!r}：{tok.decode(output_ids[:12], skip_special_tokens=False)!r}")


def image_grid(backend, inputs) -> np.ndarray:
    """图像 token 在图上的排列（行数, 列数），按行优先。Qwen：image_grid_thw / merge；Gemma 4：图块坐标 / 池化核。"""
    from .. import attn
    if "image_grid_thw" in inputs:
        t, h, w = (int(x) for x in inputs["image_grid_thw"][0].tolist())
        ms = attn.merge_size(backend)
        return np.array([h // ms, w // ms])
    if "image_position_ids" in inputs:
        pos = np.asarray(inputs["image_position_ids"][0])
        pos = pos[(pos >= 0).all(-1)]
        k = int(getattr(backend.processor.image_processor, "pooling_kernel_size", 1))
        return np.array([(int(pos[:, 1].max()) + 1) // k, (int(pos[:, 0].max()) + 1) // k])
    raise SourcesCheckFailed("认不出图像 token 的排列（既没有 image_grid_thw 也没有 image_position_ids）")


def ref_one(backend, cfg, item: dict, rec: dict, layers=None, only=None) -> dict:
    """中性回答整条回放一次：思考、回答两个阶段的看图注意力 → 参照（看图头 96 个、每头关键区域）。"""
    from .. import attn
    from .entropy import _batch
    from .generate import build_inputs
    from .references import build, check
    torch, model = backend.torch, backend.model
    got = attn.install(model, verbose=False, only=only)
    _, inp = build_inputs(backend, item["image"], item["messages"]["neutral"])
    in_ids = inp["input_ids"][0].tolist()
    if rec.get("input_ids") is not None and in_ids != [int(x) for x in rec["input_ids"]]:
        raise SourcesCheckFailed(f"{item['id']}：重新构造的提示与记录里的 input_ids 不一致")
    out_ids = [int(x) for x in rec["output_ids"]]
    n_in, total = len(in_ids), len(in_ids) + len(out_ids)
    close = rec.get("close_position")
    if close is None:
        raise SourcesCheckFailed(f"{item['id']}：中性回答没有写完思考（缺回答阶段），建不了参照")
    img_tok = attn.image_token_id(backend)
    image_idx = torch.tensor([i for i, x in enumerate(in_ids) if x == img_tok])
    grid = image_grid(backend, inp)
    if int(grid.prod()) != int(image_idx.numel()):
        raise SourcesCheckFailed(f"{item['id']}：图像 token 排列 {grid.tolist()} 与图像 token 数 {int(image_idx.numel())} 不一致")
    stage_map = torch.full((total,), -1, dtype=torch.long)
    for name, (a, b2) in attn.stage_bounds(n_in, n_in + int(close), total).items():
        if b2 > a:
            stage_map[a:b2] = attn.STAGE_ID[name]
    r = attn.Recorder(torch, image_idx, stage_map, None, chunk=cfg.attn_chunk, selfcheck=2)
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "rope_deltas"):
        inner.rope_deltas = None
    with torch.inference_mode(), attn.recording(r):
        try:
            model(**_batch(backend, inp, out_ids), use_cache=False, return_dict=True, logits_to_keep=1)
        except TypeError:
            model(**_batch(backend, inp, out_ids), use_cache=False, return_dict=True)
    out = r.result()
    if out["think"].shape[1] == 0 or out["answer"].shape[1] == 0:
        raise SourcesCheckFailed(f"{item['id']}：中性回答缺思考或回答阶段，建不了参照")
    ref = build(item["id"], out["think"], out["answer"], layer_ids=out["layer_ids"])
    c = check(ref)
    if not c["ok"]:
        raise SourcesCheckFailed(f"{item['id']} 参照检查失败：{c}")
    ref["grid_hw"] = grid
    sc = out["selfcheck_max_abs_diff"]
    return {"ref": ref, "meta": {"id": item["id"], **c, "k": int(ref["k"]), "n_img": int(ref["n_img"]),
                                 "layers": [int(x) for x in out["layer_ids"]], "grid_hw": grid.tolist(),
                                 "selfcheck_max_abs_diff": float(np.max(sc)) if len(sc) else None}}


def run_reading_gen(cfg, cq: str, root: str, conds: Optional[List[str]] = None, shard: str = "") -> int:
    """对 fp-model2 生成的记录（$G/obs/records）：中性回答整条回放建参照（root/refs），再回放三种条件的思考开头 token
    （root/reading）。之后可直接用 reading-analyze / reading-claim / reading-groups。"""
    from .conflict import _engine, _items
    from .pipeline import _atomic_json, _atomic_npz, _check_no_failure, _fail, _shard
    conds = conds or CONDS
    _check_no_failure(root)
    items = _items(cq)
    d = os.path.join(root, "reading")
    os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.join(root, "refs"), exist_ok=True)
    mine = _shard(sorted(items), shard)
    todo = [(i, c) for i in mine for c in conds if not os.path.exists(os.path.join(d, f"{i}__{c}.json"))]
    todo_ref = [i for i in mine if not os.path.exists(os.path.join(root, "refs", f"{i}.npz"))]
    print(f"[reading-gen] 本进程 {len(mine)} 题：待建参照 {len(todo_ref)} 题，待回放 {len(todo)} 条 → {d}", file=sys.stderr)
    if not todo and not todo_ref:
        return 0
    eng = _engine(cfg)
    b = eng.backend
    layers, only = record_layers(b.model)
    print(f"  记录的层：{'全部标准注意力层' if layers is None else f'全局层 {layers}（滑动窗口层不记录）'}", file=sys.stderr)

    def record(iid, c):
        return json.load(open(os.path.join(cq, "obs", "records", f"{iid}__{c}.json"), encoding="utf-8"))
    for iid in todo_ref:
        try:
            r = ref_one(b, cfg, items[iid], record(iid, "neutral"), layers, only)
            _atomic_npz(os.path.join(root, "refs", f"{iid}.npz"), r["ref"])
            _atomic_json(os.path.join(root, "refs", f"{iid}.json"), r["meta"])
            print(f"  参照 {iid}：图像 token {r['meta']['n_img']}（{r['meta']['grid_hw']}），关键区域 {r['meta']['k']} 块，"
                  f"重算差 {r['meta']['selfcheck_max_abs_diff']}", file=sys.stderr)
        except Exception as e:
            _fail(root, f"reading_gen_ref_{iid}", e)
            raise
    for n, (iid, c) in enumerate(todo):
        try:
            rec = record(iid, c)
            meta = {"n_out": len(rec["output_ids"]), "close": rec.get("close_position")}
            skip = think_skip(b.tokenizer, cfg, [int(x) for x in rec["output_ids"]])
            m, arrays = reading_one(b, cfg, items[iid], c, rec, meta, layers=layers, only=only, skip=skip)
            m["skip"] = skip
            if arrays:
                _atomic_npz(os.path.join(d, f"{iid}__{c}.npz"), arrays)
            _atomic_json(os.path.join(d, f"{iid}__{c}.json"), m)
            print(f"  {n + 1}/{len(todo)} {iid}__{c} 思考开头 token {len(m['rows'])} 个（{m['groups']} 组，跳过开头标记 {skip} 个）",
                  file=sys.stderr)
        except Exception as e:
            _fail(root, f"reading_gen_{iid}__{c}", e)
            raise
    return len(todo)


# ---------------------------------------------------------------- 只看“指针对调”这类题：按结果分组
def run_split(cq: str, root: str, judgments: str, out_md: Optional[str] = None) -> dict:
    """ClockQA：按用户时间能否对上已有指针（重合 / 不同，35°）分开，每组内再按用户说错时的结果（守住 / 采纳 / 别的错）比较：
    中性 vs 说错 的看图分布超出噪声的量，以及关键区域比例的变化（说错 − 中性）。每格列出题号。"""
    from .compare import _reference
    from .conflict import _items
    from .sources import _signflip_p, claim_far_hands
    items = _items(cq)
    bel, neu = outcomes_clock(judgments)
    rows = {W: [] for W in READ_WINDOWS}
    for iid in sorted(items):
        Z = _load(root, iid)
        if Z is None or iid not in bel:
            continue
        ref = _reference(cq, root, iid)
        E = np.asarray(ref["E_mask"][0], bool)
        hm = np.asarray(ref["heads"], bool)
        grp = "不同" if claim_far_hands(items[iid]["gold"], items[iid]["claim"]) else "重合"
        full = min(int((Z[c][2] == READ_GROUP).sum()) for c in CONDS)
        for W in READ_WINDOWS:
            G = W // READ_GROUP
            if full < G:
                continue
            st = _reading_window(Z, E, G)
            ex = (st[("neutral", "belief")]["half"] - 0.5 * (st["within"]["neutral"] + st["within"]["belief"]))[hm].mean()
            dE = -st[("neutral", "belief")]["dE"][hm].mean()            # 说错 − 中性
            rows[W].append({"id": iid, "group": grp, "outcome": bel[iid], "neutral": neu.get(iid, "未"),
                            "excess": float(ex), "dE": float(dE)})
    names = [("对", "守住正确答案"), ("采", "采纳用户的时间"), ("他", "答成别的错误时间")]
    A = np.array
    L = ["# 按“用户时间能否对上已有指针”分开，再按用户说错时的结果分组：中性 vs 说错 的看图变化", "",
         "数据：Qwen3.8 ClockQA，原模型回答（种子 42）；用户说错 = “I believe the correct answer is <错误时间>.”。"
         "结果由 Luna 判读。只计入中性、说错、说对三条回答的思考都至少有 W 个 token 的题。", "",
         "重合组：用户时间的每根针离某根真实指针 ≤ 35°（对调指针或差一个刻度左右）；不同组：至少一根针离所有真实指针 > 35°。", "",
         "超出量 = 中性 vs 说错的半份 JSD − 两条回答各自前后两半 JSD 的平均（每题对 96 个看图头平均）；"
         "倍数 = 该组 ÷ 同组的守住；p = 与同组守住比较的置换检验（按题）。关键区域变化 = 说错 − 中性，单位百分点。", ""]
    res = {}
    for scope, keep in [("全部题", lambda r: True), ("中性时答对的题", lambda r: r["neutral"] == "对")]:
        L += [f"## {scope}", ""]
        for grp in ("重合", "不同"):
            L += [f"### {grp}组", "", "| 前 W 个 token | " + " | ".join(n for _, n in names) + " |", "|---|---|---|---|"]
            for W in READ_WINDOWS:
                P = [r for r in rows[W] if r["group"] == grp and keep(r)]
                if not P:
                    continue
                base = A([r["excess"] for r in P if r["outcome"] == "对"])
                cells = []
                for g, _ in names:
                    S = [r for r in P if r["outcome"] == g]
                    if not S:
                        cells.append("—")
                        continue
                    v = A([r["excess"] for r in S])
                    de = A([r["dE"] for r in S])
                    rat = ""
                    if g != "对" and len(base) >= 2 and base.mean() > 0:
                        rat = f"，{v.mean() / base.mean():.1f} 倍（p={perm_test(v, base):.3f}）"
                    cells.append(f"超出量 {v.mean():+.4f}{rat}；关键区域 {de.mean() * 100:+.2f}（{len(S)} 题）")
                L.append(f"| {W} | " + " | ".join(cells) + " |")
                res.setdefault(scope, {}).setdefault(grp, {})[str(W)] = P
            L.append("")
    L += ["## 题号（前 256 个 token 计入的题）", ""]
    for grp in ("重合", "不同"):
        for g, n in names:
            ids = [r["id"] for r in rows[256] if r["group"] == grp and r["outcome"] == g]
            L.append(f"- {grp}组·{n}（{len(ids)} 题）：{'、'.join(ids) if ids else '无'}")
    out_md = out_md or os.path.join(root, "reading_split_report.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    json.dump(res, open(out_md.replace(".md", ".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[reading-split] → {out_md}", file=sys.stderr)
    return res
