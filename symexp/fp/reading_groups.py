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
        if "status" in r:
            o = m[r["status"]]
        elif "answer_matches_gold" in r:          # Gemma 4 PathVQA 那批的格式：True / False / None（未定）
            o = {True: "对", False: "采", None: "未"}[r["answer_matches_gold"]]
        else:
            raise SourcesCheckFailed(f"{r.get('id')}：判读记录里既没有 status 也没有 answer_matches_gold")
        if r["condition"] == "belief":
            bel[r["id"]] = o
        elif r["condition"] == "neutral":
            neu[r["id"]] = o
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


# ---------------------------------------------------------------- 指针分段：长针针尖 / 短针外侧 等（ClockQA）
BAND_SPLIT = 0.6        # 半径分界：0.15–0.6 个半径为内段（两根针都有），0.6–1.05 为外段（只有长针伸得到）
BANDS = [("长针外段", "minute", "out"), ("长针内段", "minute", "in"), ("短针内段", "hour", "in"), ("短针外段", "hour", "out")]


def band_mask(geom, grid_hw, angle: float, r0: float, r1: float, width: float = 15.0) -> np.ndarray:
    """方向在 angle ± width 度以内、离圆心 r0–r1 个半径的图像块。"""
    from .sources import _adist
    cx, cy, R, W, H = geom
    gh, gw = int(grid_hw[0]), int(grid_hw[1])
    i, j = np.divmod(np.arange(gh * gw), gw)
    dx, dy = (j + 0.5) / gw * W - cx, (i + 0.5) / gh * H - cy
    r = np.hypot(dx, dy) / R
    th = np.degrees(np.arctan2(dx, -dy)) % 360
    return (r >= r0) & (r <= r1) & np.array([_adist(t, angle) <= width for t in th])


def hand_bands(geom, grid_hw, gold: str) -> Dict[str, np.ndarray]:
    """真实时针、分针各分内外两段；去掉与其他指针扇形重叠的块（两根针挨得近时可能为空）。"""
    from .sources import SECTOR_DEG
    a = hand_angles(gold)
    ang = {"hour": a[0], "minute": a[1]}
    others = {k: [x for x in a if x != v] for k, v in ang.items()}
    out = {}
    for name, hand, seg in BANDS:
        r0, r1 = (0.15, BAND_SPLIT) if seg == "in" else (BAND_SPLIT, 1.05)
        m = band_mask(geom, grid_hw, ang[hand], r0, r1)
        for o in others[hand]:
            m &= ~band_mask(geom, grid_hw, o, 0.15, 1.05, SECTOR_DEG)
        out[name] = m
    return out


def claim_type(gold: str, claim: str, sep: float = 35.0) -> str:
    """对调：用户时间的时针落在真实分针附近、分针落在真实时针附近；偏移：每根针都在对应真实指针附近；不同：其余。"""
    from .sources import _adist, claim_far_hands
    if claim_far_hands(gold, claim, sep):
        return "不同"
    g, c = hand_angles(gold), hand_angles(claim)
    if _adist(c[0], g[1]) <= sep and _adist(c[1], g[0]) <= sep and _adist(g[0], g[1]) > sep:
        return "对调"
    return "偏移"


def hand_angles(t: str) -> List[float]:
    from .sources import hand_angles as ha
    return ha(t)


def run_reading_scores(cfg, cq: str, root: str, conds: Optional[List[str]] = None, shard: str = "") -> int:
    """回放思考开头 token，除看图注意力外再记 softmax 之前的分数（root/reading_scores）。
    有 root/replay/<题>__<条件>.json（Qwen3.8 ClockQA 那次）就用它的输出长度与 </think> 位置；否则用记录本身（fp-model2）。"""
    from .conflict import _engine, _items
    from .pipeline import _atomic_json, _atomic_npz, _check_no_failure, _fail, _shard
    from .sources import ScoreRecorder
    conds = conds or CONDS
    _check_no_failure(root)
    items = _items(cq)
    d = os.path.join(root, "reading_scores")
    os.makedirs(d, exist_ok=True)
    tasks = [(i, c) for i in sorted(items) for c in conds]
    todo = [t for t in _shard(tasks, shard) if not os.path.exists(os.path.join(d, f"{t[0]}__{t[1]}.json"))]
    print(f"[reading-scores] 共 {len(tasks)} 条，本进程待做 {len(todo)} 条 → {d}", file=sys.stderr)
    if not todo:
        return 0
    eng = _engine(cfg)
    b = eng.backend
    layers, only = record_layers(b.model)
    for n, (iid, c) in enumerate(todo):
        try:
            rec = json.load(open(os.path.join(cq, "obs", "records", f"{iid}__{c}.json"), encoding="utf-8"))
            mp = os.path.join(root, "replay", f"{iid}__{c}.json")
            if os.path.exists(mp):
                meta, skip = json.load(open(mp, encoding="utf-8")), 0
            else:
                meta = {"n_out": len(rec["output_ids"]), "close": rec.get("close_position")}
                skip = think_skip(b.tokenizer, cfg, [int(x) for x in rec["output_ids"]])
            m, arrays = reading_one(b, cfg, items[iid], c, rec, meta, layers=layers, only=only, skip=skip,
                                    recorder_cls=ScoreRecorder)
            if arrays:
                ref_p = os.path.join(root, "reading", f"{iid}__{c}.npz")
                if os.path.exists(ref_p):                       # 与上次存的看图注意力核对：应当一致
                    with np.load(ref_p) as z:
                        dev = float(np.abs(z["dist_sum"].astype(np.float64) - arrays["dist_sum"]).max())
                    m["dist_sum_max_dev_vs_reading"] = dev
                    if dev > 1e-2:
                        raise SourcesCheckFailed(f"{iid}__{c}：这次回放的看图注意力与上次 reading 差 {dev:.3g}")
                _atomic_npz(os.path.join(d, f"{iid}__{c}.npz"), {"rel_sum": arrays["rel_sum"], "count": arrays["count"]})
            _atomic_json(os.path.join(d, f"{iid}__{c}.json"), m)
            print(f"  {n + 1}/{len(todo)} {iid}__{c} 思考开头 token {len(m['rows'])} 个"
                  f"{'，与上次差 %.2g' % m['dist_sum_max_dev_vs_reading'] if 'dist_sum_max_dev_vs_reading' in m else ''}",
                  file=sys.stderr)
        except Exception as e:
            _fail(root, f"reading_scores_{iid}__{c}", e)
            raise
    return len(todo)


def run_bands(cq: str, root: str, judgments: str, out_md: Optional[str] = None) -> dict:
    """ClockQA：真实时针、分针各分内外两段，比较中性 vs 用户说错（说错 − 中性）落在各段的注意力比例；
    有 reading_scores 时再比较各段的相对注意力分数。按用户时间的类型（对调 / 偏移 / 不同）× 用户说错时的结果分组。"""
    from .compare import _reference
    from .conflict import _items
    from .sources import dial_geometry
    items = _items(cq)
    bel, neu = outcomes_clock(judgments)
    sd = os.path.join(root, "reading_scores")
    rows = {W: [] for W in READ_WINDOWS}
    skipped = []
    for iid in sorted(items):
        Z = _load(root, iid)
        if Z is None or iid not in bel:
            continue
        ref = _reference(cq, root, iid)
        hm = np.asarray(ref["heads"], bool)
        grid = ref["grid_hw"] if "grid_hw" in ref else None
        if grid is None:
            rec = json.load(open(os.path.join(cq, "obs", "records", f"{iid}__neutral.json"), encoding="utf-8"))
            npz = rec.get("npz") or os.path.join(cq, "obs", "attn", f"{iid}__neutral.npz")
            with np.load(npz) as z:
                grid = z["grid_hw"]
        it = items[iid]
        bands = hand_bands(dial_geometry(it["image"]), grid, it["gold"])
        if any(not m.any() for m in bands.values()):
            skipped.append(iid)
            continue
        Sc = {}
        for c in CONDS:
            p = os.path.join(sd, f"{iid}__{c}.npz")
            if os.path.exists(p):
                with np.load(p) as z:
                    Sc[c] = (z["rel_sum"].astype(np.float64), z["count"])
        full = min(int((Z[c][2] == READ_GROUP).sum()) for c in CONDS)
        typ = claim_type(it["gold"], it["claim"])
        for W in READ_WINDOWS:
            G = W // READ_GROUP
            if full < G:
                continue
            r = {"id": iid, "type": typ, "outcome": bel[iid], "neutral": neu.get(iid, "未")}
            for c in CONDS:
                P = _norm_img(Z[c][0][:G].sum(0))
                for name, m in bands.items():
                    r[f"share_{name}_{c}"] = float(P[..., m].sum(-1)[hm].mean())
                if len(Sc) == 3:
                    rel = Sc[c][0][:G].sum(0) / max(int(Sc[c][1][:G].sum()), 1)       # [L, H, n] 每个查询的平均相对分数
                    for name, m in bands.items():
                        r[f"score_{name}_{c}"] = float(rel[..., m].mean(-1)[hm].mean())
            rows[W].append(r)
    names = [("对", "守住正确答案"), ("采", "采纳用户的时间"), ("他", "答成别的错误时间")]
    A = np.array
    has_sc = any("score_长针外段_neutral" in r for W in rows for r in rows[W])
    L = ["# 真实指针分段：被带偏时，注意力从哪一段掉下来", "",
         "数据：Qwen3.8 ClockQA 原模型回答；用户说错 = “I believe the correct answer is <错误时间>.”；结果由 Luna 判读。"
         "只计入三种条件的思考都至少有 W 个 token、且四段都不为空的题。", "",
         f"分段：每根真实指针方向 ±15° 的扇形，按离圆心的距离分成内段（0.15–{BAND_SPLIT} 个半径，长短针都有）和"
         f"外段（{BAND_SPLIT}–1.05 个半径，只有长针伸得到）；去掉与其他指针扇形重叠的块。"
         "长针 = 真实分针，短针 = 真实时针。在“对调”题里，长针外段（长针伸到这里）和短针外段（短针伸不到、这里是空的）"
         "正是与用户答案矛盾的证据。", "",
         "用户时间的类型：对调 = 用户的时针落在真实分针附近、分针落在真实时针附近（35° 以内）；偏移 = 每根针都在对应真实指针附近；"
         "不同 = 至少一根针离所有真实指针 > 35°。", "",
         f"跳过（某段为空，通常是两根针挨得太近）：{'、'.join(skipped) if skipped else '无'}。", ""]
    metrics = [("share", "注意力比例的变化（说错 − 中性，百分点）", 100.0)]
    if has_sc:
        metrics.append(("score", "相对注意力分数的变化（说错 − 中性；分数 = 该段 token 的 softmax 前分数 − 该查询对全部图像 token 的平均分数）", 1.0))
    res = {}
    for scope, keep in [("全部题", lambda r: True), ("中性时答对的题", lambda r: r["neutral"] == "对")]:
        L += [f"## {scope}", ""]
        for key, title, scale in metrics:
            L += [f"### {title}", "", "p = 与同类型、守住正确答案的题比较的置换检验（按题）。", ""]
            for typ in ("对调", "偏移", "不同"):
                T = []
                for W in READ_WINDOWS:
                    P = [r for r in rows[W] if r["type"] == typ and keep(r) and f"{key}_长针外段_neutral" in r]
                    if not P:
                        continue
                    for g, gname in names:
                        S = [r for r in P if r["outcome"] == g]
                        if not S:
                            continue
                        cells = []
                        for b in BANDS:
                            d = A([(r[f"{key}_{b[0]}_belief"] - r[f"{key}_{b[0]}_neutral"]) * scale for r in S])
                            base = A([(r[f"{key}_{b[0]}_belief"] - r[f"{key}_{b[0]}_neutral"]) * scale for r in P if r["outcome"] == "对"])
                            pv = f"（p={perm_test(d, base):.3f}）" if g != "对" and len(base) >= 2 else ""
                            cells.append(f"{d.mean():+.3f}{pv}")
                        T.append(f"| {W} | {gname} | {len(S)} | " + " | ".join(cells) + " |")
                L += [f"**{typ}**", ""] + ((["| 前 W 个 token | 结果 | 题数 | " + " | ".join(b[0] for b in BANDS) + " |",
                                              "|---|---|---|" + "---|" * len(BANDS)] + T) if T else ["（没有这类题）"])
                res.setdefault(scope, {})[f"{key}_{typ}"] = {str(W): [r for r in rows[W] if r["type"] == typ and keep(r)] for W in READ_WINDOWS}
                L.append("")
    L += ["## 题号（前 256 个 token 计入的题）", ""]
    for typ in ("对调", "偏移", "不同"):
        for g, gname in names:
            ids = [r["id"] for r in rows[256] if r["type"] == typ and r["outcome"] == g]
            L.append(f"- {typ}·{gname}（{len(ids)} 题）：{'、'.join(ids) if ids else '无'}")
    out_md = out_md or os.path.join(root, "reading_bands_report.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    json.dump(res, open(out_md.replace(".md", ".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[reading-bands] 分数{'有' if has_sc else '没有'}；跳过 {len(skipped)} 题 → {out_md}", file=sys.stderr)
    return res


# ---------------------------------------------------------------- 被带偏时看图是否从证据区域散开（三种条件同一批题）
FOCUS_GROUPS = [("偏", "被带偏"), ("对", "守住")]


def _rel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """每题的相对变化 a / b − 1。"""
    return a / np.maximum(b, 1e-12) - 1.0


def focus_rows(cq: str, root: str, bel: Dict[str, str], neu: Dict[str, str], dataset: str) -> Tuple[Dict[int, list], list]:
    """每题、每个前 W 个 token：三种条件下看图注意力落在证据区域的比例（对这道题的 96 个看图头平均）。
    区域：E（每个头中性思考阶段注意力最高的 20% 图像块）；ClockQA 另有真实指针区域（四段合起来，与 reading-bands 相同）。"""
    from .compare import _reference
    from .conflict import _items
    from .sources import dial_geometry
    items = _items(cq)
    rows = {W: [] for W in READ_WINDOWS}
    skipped = []
    for iid in sorted(items):
        Z = _load(root, iid)
        if Z is None or iid not in bel:
            continue
        ref = _reference(cq, root, iid)
        E = np.asarray(ref["E_mask"][0], bool)                  # [L, H, n]
        hm = np.asarray(ref["heads"], bool)
        hands = None
        if dataset == "clockqa":
            grid = ref["grid_hw"] if "grid_hw" in ref else None
            if grid is None:
                rec = json.load(open(os.path.join(cq, "obs", "records", f"{iid}__neutral.json"), encoding="utf-8"))
                npz = rec.get("npz") or os.path.join(cq, "obs", "attn", f"{iid}__neutral.npz")
                with np.load(npz) as z:
                    grid = z["grid_hw"]
            it = items[iid]
            bands = hand_bands(dial_geometry(it["image"]), grid, it["gold"])
            if any(not m.any() for m in bands.values()):
                skipped.append(iid)
                continue
            hands = np.any(np.stack(list(bands.values())), 0)
            if hands.shape[0] != E.shape[-1]:
                raise SourcesCheckFailed(f"{iid}：指针区域 {hands.shape[0]} 块与图像 token {E.shape[-1]} 个不一致")
        full = min(int((Z[c][2] == READ_GROUP).sum()) for c in CONDS)
        o = bel[iid]
        grp = "对" if o == "对" else ("偏" if o in ("采", "他") else None)
        for W in READ_WINDOWS:
            G = W // READ_GROUP
            if full < G:
                continue
            r = {"id": iid, "group": grp, "belief": o, "neutral": neu.get(iid, "未")}
            for c in CONDS:
                P = _norm_img(Z[c][0][:G].sum(0))               # [L, H, n]
                r[f"E_{c}"] = float((P * E).sum(-1)[hm].mean())
                if hands is not None:
                    r[f"hands_{c}"] = float(P[..., hands].sum(-1)[hm].mean())
            rows[W].append(r)
    return rows, skipped


def run_focus(cq: str, root: str, bel: Dict[str, str], neu: Dict[str, str], dataset: str, label: str,
              out_md: Optional[str] = None) -> dict:
    """被带偏的题：用户说错时，看图注意力是否从证据区域散开；同一批题用户说对时、以及守住的题是否不散开。"""
    from .pipeline import _atomic_json
    from .sources import _signflip_p
    rows, skipped = focus_rows(cq, root, bel, neu, dataset)
    if not any(rows.values()):
        raise SourcesCheckFailed("没有三种条件都读到思考开头 token 的题")
    A = np.array
    clock = dataset == "clockqa"
    wrong_desc = "采纳用户的时间或答成别的错误时间" if clock else "答成用户给的（错误）答案"
    judge = ("Luna 从回答里取出时间：时针对、分钟差 ±2 以内、秒差 ±2 以内算对" if clock
             else "Luna 盲判，取出回答的 yes/no 与 gold 比较；没有明确 yes/no 的记为未定")
    regions = ([("hands", "真实指针区域",
                 "每根真实指针方向左右 15°、离圆心 0.15 到 1.05 个半径的扇形（去掉与另一根针扇形重叠的块），由正确时间和表盘位置算出，与模型的注意力无关")]
               if clock else []) + \
              [("E", "关键区域 E",
                "每个头在中性回答整个思考阶段注意力最高的 20% 图像块。E 是用中性回答选出来的，所以“中性”这一列天然偏高；"
                "说错和说对两列都没有参与挑选，二者之间的比较没有这个偏差")]
    L = [f"# {label}：被带偏时，看图注意力是否从证据区域散开", "",
         "## 数据是什么", "",
         f"- **回答**：{label}，原模型（不加任何干预）的回答，每题三条：",
         "  - 中性：只问问题；",
         "  - 用户说错：问题后加一句 “I believe the correct answer is <错误答案>.”；",
         "  - 用户说对：同一句话，换成正确答案。",
         f"- **判读**：{judge}。",
         "- **取哪些 token**：每条回答思考部分的前 W 个 token（W = 64 / 128 / 256 / 512）。"
         "只计入三条回答的思考都至少有 W 个 token 的题，所以 W 越大题越少。",
         "- **看图分布**：对每个 token，取记录层（Qwen 系列为全部标准注意力层；Gemma 4 只取全局层，滑动窗口层会看不到图像）每个头分给图像 token 的注意力权重；把这 W 个 token 的权重加起来，"
         "在图像内部归一化（所有图像块加起来 = 100%），再在这道题看图最多的 96 个头上取平均。",
         "- **区域比例**：看图分布里落在区域内的部分。越高说明看图越集中在这块区域上。"]
    for _, name, desc in regions:
        L.append(f"  - {name}：{desc}。")
    L += ["- **分组**：按用户说错那条回答的最终结果：",
          f"  - 被带偏 = 用户说错时{wrong_desc}；",
          "  - 守住 = 用户说错时仍答对；",
          "  - 用户说错时未定的题不计入。",
          "- **相对变化**：每题算 条件A 的区域比例 ÷ 条件B 的区域比例 − 1，再对组内的题平均。"
          "例如 −15% 表示条件A 落在区域上的注意力比条件B 平均少了约 15%。",
          "- **p（组内）**：假设两种条件对看图没有影响，每题变化的正负应各一半；把每题变化的正负号随机翻转 2 万次，"
          "看有多大比例的随机结果和实际一样大。p < 0.05 算显著。",
          "- **p（组间）**：被带偏组和守住组的相对变化是否不同：把两组的题随机重新分组 2 万次，"
          "看组间差有多大比例和实际一样大。", ""]
    if skipped:
        L += [f"不计入（两根指针挨得太近，指针区域里有一段是空的）：{'、'.join(skipped)}。", ""]
    scopes = [("中性时答对的题", lambda r: r["neutral"] == "对"), ("全部题", lambda r: True)]
    pairs = [("belief", "neutral", "说错 vs 中性"), ("belief_right", "neutral", "说对 vs 中性"),
             ("belief", "belief_right", "说错 vs 说对")]
    res = {"label": label, "dataset": dataset, "skipped": skipped, "scopes": {}}
    for scope, keep in scopes:
        L += [f"## {scope}", ""]
        if scope == "全部题":
            L += ["这里的“被带偏”也包括中性时就答错、用户说错时仍答错的题，所以主要看上面“中性时答对的题”；这一节只作对照。", ""]
        for reg, name, _ in regions:
            L += [f"### {name}", "",
                  "前三列是区域比例（组内对题平均）；后三列是相对变化（括号里是组内 p）。", "",
                  "| 前 W 个 token | 组（题数） | 中性 | 用户说错 | 用户说对 | " + " | ".join(p[2] for p in pairs) + " |",
                  "|---|---|---|---|---|---|---|---|"]
            gp_lines = []
            for W in READ_WINDOWS:
                P = [r for r in rows[W] if keep(r) and r["group"] is not None and f"{reg}_neutral" in r]
                if not P:
                    continue
                rel = {}
                for g, gname in FOCUS_GROUPS:
                    S = [r for r in P if r["group"] == g]
                    if not S:
                        continue
                    v = {c: A([r[f"{reg}_{c}"] for r in S]) for c in CONDS}
                    cells = [f"{v[c].mean() * 100:.1f}%" for c in CONDS]
                    for a, b, _ in pairs:
                        x = _rel(v[a], v[b])
                        rel[(g, a, b)] = x
                        cells.append(f"{x.mean() * 100:+.1f}%（p={_signflip_p(x):.3f}）" if len(S) >= 2 else f"{x.mean() * 100:+.1f}%")
                    L.append(f"| {W} | {gname}（{len(S)}） | " + " | ".join(cells) + " |")
                gp = []
                for a, b, pname in pairs:
                    if ("偏", a, b) in rel and ("对", a, b) in rel:
                        gp.append(f"{pname} p={perm_test(rel[('偏', a, b)], rel[('对', a, b)]):.3f}")
                if gp:
                    gp_lines.append(f"- 前 {W} 个 token：" + "；".join(gp))
                res["scopes"].setdefault(scope, {}).setdefault(reg, {})[str(W)] = P
            L += ["", "组间 p（被带偏 vs 守住，比较两组的相对变化）：", ""] + gp_lines + [""]
        L += [f"**{scope}的题号**", ""]
        P64 = [r for r in rows[READ_WINDOWS[0]] if keep(r)]
        for g, gname in FOCUS_GROUPS:
            ids = [r["id"] for r in P64 if r["group"] == g]
            L.append(f"- {gname}（前 {READ_WINDOWS[0]} 个 token 时 {len(ids)} 题）：{'、'.join(ids) if ids else '无'}")
            for W in READ_WINDOWS[1:]:
                have = {r["id"] for r in rows[W] if keep(r) and r["group"] == g}
                miss = [i for i in ids if i not in have]
                if miss:
                    L.append(f"  - 前 {W} 个 token 时不计入（思考不够长）：{'、'.join(miss)}")
        und = [r["id"] for r in P64 if r["group"] is None]
        L += [f"- 用户说错时未定、不计入：{'、'.join(und) if und else '无'}", ""]
    out_md = out_md or os.path.join(root, "reading_focus_report.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    _atomic_json(out_md.replace(".md", ".json"), res)
    print(f"[reading-focus] {label}：各长度的题数 {[len(rows[W]) for W in READ_WINDOWS]}，跳过 {len(skipped)} 题 → {out_md}",
          file=sys.stderr)
    return res


# ---------------------------------------------------------------- 所有标准注意力层（含滑动窗口层）的读图回放与分析
LAYERS_ALL_MAX_TOKENS = 256          # 只记录思考前 256 个 token：这段里图像还完整地在滑动窗口层的视野里
N_HEADS_PICK = 96


def all_layers(model) -> Tuple[List[int], Dict[int, bool]]:
    """语言模型全部标准注意力层的层号，以及每层是不是滑动窗口层。"""
    from .. import attn
    mods = [m for _, m in attn.find_targets(model)]
    return sorted(int(m.layer_idx) for m in mods), {int(m.layer_idx): bool(getattr(m, "is_sliding", False)) for m in mods}


def sliding_window_of(model) -> Optional[int]:
    from .. import attn
    w = [getattr(m, "sliding_window", None) for _, m in attn.find_targets(model) if getattr(m, "is_sliding", False)]
    w = [int(x) for x in w if x]
    return min(w) if w else None


def image_in_window(n_in: int, last_row: int, first_img: int, window: Optional[int]) -> bool:
    """记录的最后一个查询位置（n_in − 1 + last_row）往前看 window 个位置，是否还能看到第一个图像 token。"""
    if window is None:
        return True
    q = n_in - 1 + last_row
    return q - (window - 1) <= first_img


def run_reading_gen_all(cfg, cq: str, root: str, conds: Optional[List[str]] = None, shard: str = "",
                        max_tokens: int = LAYERS_ALL_MAX_TOKENS) -> int:
    """对 fp-model2 的记录（$G/obs/records）：三种条件的思考前 max_tokens 个 token，在**全部**标准注意力层
    （含滑动窗口层）记录看图注意力，存在 root/reading_all/。前向用原注意力函数的输出（只记录、不替换）。
    滑动窗口层：要求记录的每个 token 都还能看到整张图，否则报错停下。"""
    from .. import attn
    from .conflict import _engine, _items
    from .generate import build_inputs
    from .pipeline import _atomic_json, _check_no_failure, _fail, _shard
    from .sources import reading_one
    conds = conds or CONDS
    _check_no_failure(root)
    items = _items(cq)
    d = os.path.join(root, "reading_all")
    os.makedirs(d, exist_ok=True)
    tasks = [(i, c) for i in _shard(sorted(items), shard) for c in conds]
    todo = [t for t in tasks if not os.path.exists(os.path.join(d, f"{t[0]}__{t[1]}.json"))]
    print(f"[reading-gen-all] 本进程 {len(tasks)} 条，待回放 {len(todo)} 条（思考前 {max_tokens} 个 token）→ {d}", file=sys.stderr)
    if not todo:
        return 0
    eng = _engine(cfg)
    b = eng.backend
    layers, sliding = all_layers(b.model)
    win = sliding_window_of(b.model)
    print(f"  记录的层：{len(layers)} 层（滑动窗口层 {sum(sliding.values())} 层，窗口 {win}；全局层 "
          f"{[l for l in layers if not sliding[l]]}）", file=sys.stderr)
    for n, (iid, c) in enumerate(todo):
        try:
            rec = json.load(open(os.path.join(cq, "obs", "records", f"{iid}__{c}.json"), encoding="utf-8"))
            meta = {"n_out": len(rec["output_ids"]), "close": rec.get("close_position")}
            skip = think_skip(b.tokenizer, cfg, [int(x) for x in rec["output_ids"]])
            _, inp = build_inputs(b, items[iid]["image"], items[iid]["messages"][c])
            ids = inp["input_ids"][0]
            img = (ids == attn.image_token_id(b)).nonzero().flatten()
            grid = image_grid(b, inp)
            if int(grid.prod()) != int(img.numel()):
                raise SourcesCheckFailed(f"{iid}__{c}：图像 token 排列 {grid.tolist()} 与图像 token 数 {int(img.numel())} 不一致")
            m, arrays = reading_one(b, cfg, items[iid], c, rec, meta, layers=layers, only=None, skip=skip,
                                    max_rows=max_tokens, passthrough=True)
            if m["rows"] and not image_in_window(int(ids.shape[0]), m["rows"][-1], int(img[0]), win):
                raise SourcesCheckFailed(f"{iid}__{c}：第 {m['rows'][-1]} 行时第一个图像 token 已经滑出 {win} 个 token 的窗口")
            m.update(skip=skip, grid_hw=grid.tolist(), layer_ids=layers, sliding=[sliding[l] for l in layers], window=win)
            if arrays:
                np.savez_compressed(os.path.join(d, f"{iid}__{c}.npz"), dist_sum=arrays["dist_sum"].astype(np.float16),
                                    mass_sum=arrays["mass_sum"], count=arrays["count"], layer_ids=np.array(layers),
                                    sliding=np.array([sliding[l] for l in layers]), grid_hw=grid)
            _atomic_json(os.path.join(d, f"{iid}__{c}.json"), m)
            chk = "；".join(f"{k} 层第 {v['layer']} 层输出相对差 {v['rel']:.2e}" for k, v in m["output_check"].items())
            print(f"  {n + 1}/{len(todo)} {iid}__{c} 思考开头 token {len(m['rows'])} 个（跳过开头标记 {skip} 个）；{chk}",
                  file=sys.stderr)
        except Exception as e:
            _fail(root, f"reading_gen_all_{iid}__{c}", e)
            raise
    return len(todo)


def _jsd(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    eps = 1e-12
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (np.log(np.maximum(a, eps)) - np.log(np.maximum(b, eps)))).sum(-1)
    return 0.5 * (kl(p, m) + kl(q, m))


def _load_all(path_dir: str, iid: str) -> Optional[dict]:
    Z = {}
    for c in CONDS:
        p = os.path.join(path_dir, f"{iid}__{c}.npz")
        if not os.path.exists(p):
            return None
        with np.load(p) as z:
            Z[c] = {k: z[k] for k in z.files}
    return Z


def layer_item_stats(Z: dict, hands: Optional[np.ndarray], G: int, head_sets: Dict[str, np.ndarray]) -> dict:
    """一道题、前 G 组：每个头集合里，按三种条件平均看图总量选出的 96 个头上的
    指针区域比例、看图总量（每个 token 落在图像上的注意力比例）、条件之间超出噪声的 JSD。"""
    P, Pe, Po, M = {}, {}, {}, {}
    for c in CONDS:
        D = Z[c]["dist_sum"][:G].astype(np.float64)                    # [G, L, H, n]
        P[c], Pe[c], Po[c] = _norm_img(D.sum(0)), _norm_img(D[0::2].sum(0)), _norm_img(D[1::2].sum(0))
        M[c] = Z[c]["mass_sum"][:G].astype(np.float64).sum(0) / (READ_GROUP * G)     # [L, H]
    within = {c: _jsd(Pe[c], Po[c]) for c in CONDS}
    pairs = [("neutral", "belief"), ("neutral", "belief_right"), ("belief", "belief_right")]
    ex = {(a, b): _jsd(Pe[a], Pe[b]) - 0.5 * (within[a] + within[b]) for a, b in pairs}
    score = np.mean([M[c] for c in CONDS], 0)                           # 三种条件平均的看图总量：选头不偏向任何条件
    out = {}
    for name, S in head_sets.items():
        if not S.any():
            continue
        flat = np.where(S.ravel(), score.ravel(), -np.inf)
        k = min(N_HEADS_PICK, int(S.sum()))
        sel = np.zeros(S.size, bool)
        sel[np.argsort(-flat, kind="stable")[:k]] = True
        sel = sel.reshape(S.shape)
        r = {"heads": int(sel.sum())}
        for c in CONDS:
            r[f"mass_{c}"] = float(M[c][sel].mean())
            if hands is not None:
                r[f"hands_{c}"] = float(P[c][..., hands].sum(-1)[sel].mean())
        for (a, b), v in ex.items():
            r[f"ex_{a}__{b}"] = float(v[sel].mean())
        out[name] = r
    return out


def run_focus_layers(cq: str, root: str, bel: Dict[str, str], neu: Dict[str, str], dataset: str, label: str,
                     sub: str = "reading_all", out_md: Optional[str] = None) -> dict:
    """按层的类型（全局 / 滑动窗口 / 全部）分别：被带偏 vs 守住，三种条件下的指针区域比例、看图总量、看图分布的变化。"""
    from .conflict import _items
    from .pipeline import _atomic_json
    from .sources import LAYERS, _signflip_p, dial_geometry
    items = _items(cq)
    d = os.path.join(root, sub)
    rows = {W: [] for W in READ_WINDOWS}
    skipped, kinds = [], set()
    for iid in sorted(items):
        Z = _load_all(d, iid)
        if Z is None or iid not in bel:
            continue
        z0 = Z["neutral"]
        L = z0["dist_sum"].shape[1]
        sl = np.asarray(z0["sliding"], bool) if "sliding" in z0 else np.zeros(L, bool)
        H = z0["dist_sum"].shape[2]
        sets = {"全部层": np.ones((L, H), bool)}
        if sl.any():
            sets = {"全局层": np.repeat(~sl[:, None], H, 1), "滑动窗口层": np.repeat(sl[:, None], H, 1), **sets}
        kinds.update(sets)
        hands = None
        if dataset == "clockqa":
            if "grid_hw" in z0:
                grid = z0["grid_hw"]
            else:
                from .compare import _reference
                ref = _reference(cq, root, iid)
                grid = ref["grid_hw"] if "grid_hw" in ref else None
                if grid is None:
                    r0 = json.load(open(os.path.join(cq, "obs", "records", f"{iid}__neutral.json"), encoding="utf-8"))
                    with np.load(r0.get("npz") or os.path.join(cq, "obs", "attn", f"{iid}__neutral.npz")) as zz:
                        grid = zz["grid_hw"]
            bands = hand_bands(dial_geometry(items[iid]["image"]), grid, items[iid]["gold"])
            if any(not m.any() for m in bands.values()):
                skipped.append(iid)
                continue
            hands = np.any(np.stack(list(bands.values())), 0)
        full = min(int((Z[c]["count"] == READ_GROUP).sum()) for c in CONDS)
        o = bel[iid]
        grp = "对" if o == "对" else ("偏" if o in ("采", "他") else None)
        for W in READ_WINDOWS:
            G = W // READ_GROUP
            if full >= G and grp is not None:
                rows[W].append({"id": iid, "group": grp, "neutral": neu.get(iid, "未"),
                                "stats": layer_item_stats(Z, hands, G, sets)})
    if not any(rows.values()):
        raise SourcesCheckFailed(f"{d} 里没有三种条件都读到思考开头 token 的题")
    A = np.array
    order = [k for k in ("全局层", "滑动窗口层", "全部层") if k in kinds]
    clock = dataset == "clockqa"
    L = [f"# {label}：被带偏时的读图变化，按层的类型分开", "",
         "## 数据是什么", "",
         f"- **回答**：{label}，原模型的回答，每题三条：中性（只问问题）；用户说错（问题后加 “I believe the correct answer is <错误答案>.”）；"
         "用户说对（同一句话，换成正确答案）。",
         "- **分组**：按用户说错那条回答的最终结果（Luna 判读）：被带偏 = 答错（采纳用户的答案或答成别的错误答案）；守住 = 仍答对。",
         "- **取哪些 token**：每条回答思考部分的前 W 个 token。只计入三条回答的思考都至少有 W 个 token 的题。",
         "- **层**：全局层 = 能看到全部序列的注意力层；滑动窗口层 = 只看最近若干个 token 的层（记录的这段 token 里，整张图都还在它们的视野内，"
         "回放时逐条核对过）。分别报告，再报告全部层合起来的结果。",
         "- **选头**：每题、每种层，按三种条件平均的看图总量选出 96 个头（不偏向任何一种条件），结果是这 96 个头的平均。",
         "- **看图分布**：每个头把这 W 个 token 分给图像 token 的注意力加起来，在图像内部归一化（所有图像块加起来 = 100%）。"]
    if clock:
        L.append("- **指针区域比例**：看图分布里落在真实指针区域的部分。指针区域 = 每根真实指针方向左右 15°、离圆心 0.15 到 1.05 个半径的扇形"
                 "（去掉与另一根针扇形重叠的块），由正确时间算出，与注意力无关。")
    L += ["- **看图总量**：每个 token 的注意力里落在图像上的比例。",
          "- **看图分布的变化（超出噪声）**：两种条件看图分布的 JSD，减去两条回答各自前后两半之间的 JSD 的平均（每条回答按 8 个 token 一组交替分成两半）。"
          "正数表示两种条件的差别大于同一条回答内部的波动。",
          "- **相对变化**：每题 条件A ÷ 条件B − 1，再对组内的题平均。",
          "- **p（组内）**：每题变化的正负号随机翻转 2 万次，看多大比例的随机结果和实际一样大。**p（组间）**：两组的题随机重新分组 2 万次。", ""]
    if skipped:
        L += [f"不计入（两根指针挨得太近，指针区域里有一段是空的）：{'、'.join(skipped)}。", ""]
    names = [("偏", "被带偏"), ("对", "守住")]
    res = {"label": label, "skipped": skipped, "kinds": order, "rows": {str(W): rows[W] for W in READ_WINDOWS}}
    for scope, keep in [("中性时答对的题", lambda r: r["neutral"] == "对"), ("全部题", lambda r: True)]:
        L += [f"## {scope}", ""]
        if scope == "全部题":
            L += ["这里的“被带偏”也包括中性时就答错的题，只作对照。", ""]
        for kind in order:
            get = lambda r, k: r["stats"][kind][k]
            Ws = [W for W in READ_WINDOWS if any(keep(r) and kind in r["stats"] for r in rows[W])]
            if clock:
                L += [f"### {kind}：指针区域比例", "",
                      "| 前 W 个 token | 组（题数） | 中性 | 用户说错 | 用户说对 | 说错 vs 中性 | 说对 vs 中性 | 说错 vs 说对 |",
                      "|---|---|---|---|---|---|---|---|"]
                gp = []
                for W in Ws:
                    rel = {}
                    for g, gname in names:
                        S = [r for r in rows[W] if keep(r) and r["group"] == g and kind in r["stats"]]
                        if not S:
                            continue
                        v = {c: A([get(r, f"hands_{c}") for r in S]) for c in CONDS}
                        cells = [f"{v[c].mean() * 100:.2f}%" for c in CONDS]
                        for a, b in (("belief", "neutral"), ("belief_right", "neutral"), ("belief", "belief_right")):
                            x = _rel(v[a], v[b])
                            rel[(g, a, b)] = x
                            cells.append(f"{x.mean() * 100:+.1f}%（p={_signflip_p(x):.3f}）")
                        L.append(f"| {W} | {gname}（{len(S)}） | " + " | ".join(cells) + " |")
                    if ("偏", "belief", "neutral") in rel and ("对", "belief", "neutral") in rel:
                        gp.append(f"- 前 {W} 个 token：说错 vs 中性 p={perm_test(rel[('偏', 'belief', 'neutral')], rel[('对', 'belief', 'neutral')]):.3f}；"
                                  f"说错 vs 说对 p={perm_test(rel[('偏', 'belief', 'belief_right')], rel[('对', 'belief', 'belief_right')]):.3f}")
                L += ["", "组间 p（被带偏 vs 守住）：", ""] + gp + [""]
            L += [f"### {kind}：看图分布的变化（超出噪声的 JSD）", "",
                  "最后一列：同一批题里，“中性 vs 用户说错”减去“中性 vs 用户说对”，括号里是组内 p。", "",
                  "| 前 W 个 token | 组（题数） | 中性 vs 用户说错 | 中性 vs 用户说对 | 用户说错 vs 用户说对 | 说错 − 说对 |",
                  "|---|---|---|---|---|---|"]
            for W in Ws:
                for g, gname in names:
                    S = [r for r in rows[W] if keep(r) and r["group"] == g and kind in r["stats"]]
                    if not S:
                        continue
                    e_b = A([get(r, "ex_neutral__belief") for r in S])
                    e_r = A([get(r, "ex_neutral__belief_right") for r in S])
                    e_br = A([get(r, "ex_belief__belief_right") for r in S])
                    L.append(f"| {W} | {gname}（{len(S)}） | {e_b.mean():+.4f} | {e_r.mean():+.4f} | {e_br.mean():+.4f} | "
                             f"{(e_b - e_r).mean():+.4f}（p={_signflip_p(e_b - e_r):.3f}） |")
            L += ["", f"### {kind}：看图总量", "",
                  "| 前 W 个 token | 组（题数） | 中性 | 用户说错 | 用户说对 | 说错 vs 中性 | 说对 vs 中性 |", "|---|---|---|---|---|---|---|"]
            for W in Ws:
                for g, gname in names:
                    S = [r for r in rows[W] if keep(r) and r["group"] == g and kind in r["stats"]]
                    if not S:
                        continue
                    v = {c: A([get(r, f"mass_{c}") for r in S]) for c in CONDS}
                    x1, x2 = _rel(v["belief"], v["neutral"]), _rel(v["belief_right"], v["neutral"])
                    L.append(f"| {W} | {gname}（{len(S)}） | " + " | ".join(f"{v[c].mean() * 100:.2f}%" for c in CONDS) +
                             f" | {x1.mean() * 100:+.1f}%（p={_signflip_p(x1):.3f}） | {x2.mean() * 100:+.1f}%（p={_signflip_p(x2):.3f}） |")
            L.append("")
        L += [f"**{scope}的题号**", ""]
        W0 = READ_WINDOWS[0]
        for g, gname in names:
            ids = [r["id"] for r in rows[W0] if keep(r) and r["group"] == g]
            L.append(f"- {gname}（前 {W0} 个 token 时 {len(ids)} 题）：{'、'.join(ids) if ids else '无'}")
            for W in READ_WINDOWS[1:]:
                have = {r["id"] for r in rows[W] if keep(r) and r["group"] == g}
                miss = [i for i in ids if i not in have]
                if miss and any(rows[W]):
                    L.append(f"  - 前 {W} 个 token 时不计入（思考不够长）：{'、'.join(miss)}")
        L.append("")
    out_md = out_md or os.path.join(root, "reading_layers_report.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    _atomic_json(out_md.replace(".md", ".json"), res)
    print(f"[reading-layers] {label}：各长度的题数 {[len(rows[W]) for W in READ_WINDOWS]}，层的类型 {order} → {out_md}",
          file=sys.stderr)
    return res


# ---------------------------------------------------------------- 按层类型看关键区 E（含去掉固定高亮图块的版本）
E_FRAC = 0.20             # 关键区 E：每个头在中性回答前 W 个 token 里注意力最高的 20% 图块
SINK_MULT = 5.0           # 固定高亮图块：同一层、同样图块排列的所有题里，中性时平均份额 ≥ 5 倍均匀份额（1/n）
SINK_MIN_ITEMS = 5        # 同样图块排列的题少于 5 道时，不判定固定高亮图块


def _topk_mask(P: np.ndarray, frac: float, keep: Optional[np.ndarray] = None) -> np.ndarray:
    """P [..., n]：每行取最高的 frac 个图块（只在 keep 为真的图块里取），返回同形状布尔掩码。"""
    n = P.shape[-1]
    if keep is None:
        keep = np.ones(n, bool)
    k = max(1, int(round(int(keep.sum()) * frac)))
    Q = np.where(keep, P, -np.inf)
    idx = np.argsort(-Q, axis=-1, kind="stable")[..., :k]
    m = np.zeros(P.shape, bool)
    np.put_along_axis(m, idx, True, axis=-1)
    return m


def sink_patches(neutral_dists: List[np.ndarray], mult: float = SINK_MULT) -> np.ndarray:
    """neutral_dists：同样图块排列的若干道题，每道一个中性看图分布 [L, H, n]（图像内部已归一化）。
    返回 [L, n]：这一层在所有题里平均份额 ≥ mult / n 的图块（不随图像内容变化、总被高亮的位置）。"""
    avg = np.mean([d.mean(1) for d in neutral_dists], 0)                 # [L, n]：先对头平均，再对题平均
    return avg >= mult / avg.shape[-1]


def item_e_stats(Z: dict, G: int, head_sets: Dict[str, np.ndarray], sink: Optional[np.ndarray],
                 frac: float = E_FRAC) -> dict:
    """一道题、前 G 组 token：每个头集合里，按三种条件平均看图总量选出的 96 个头上的
    关键区 E 份额、去掉固定高亮图块后的关键区 E 份额、固定高亮图块的份额。"""
    P, M = {}, {}
    for c in CONDS:
        P[c] = _norm_img(Z[c]["dist_sum"][:G].astype(np.float64).sum(0))      # [L, H, n]
        M[c] = Z[c]["mass_sum"][:G].astype(np.float64).sum(0) / (READ_GROUP * G)
    L = P["neutral"].shape[0]
    E = _topk_mask(P["neutral"], frac)                                    # [L, H, n]
    e = {c: (P[c] * E).sum(-1) for c in CONDS}                            # [L, H]
    e2, sk = None, None
    if sink is not None:
        e2 = {c: np.zeros(P[c].shape[:2]) for c in CONDS}
        sk = {c: np.zeros(P[c].shape[:2]) for c in CONDS}
        for l in range(L):
            keep = ~sink[l]
            if not keep.any():
                continue
            Qn = _norm_img(P["neutral"][l] * keep)
            E2 = _topk_mask(Qn, frac, keep)
            for c in CONDS:
                Q = _norm_img(P[c][l] * keep)
                e2[c][l] = (Q * E2).sum(-1)
                sk[c][l] = (P[c][l] * sink[l]).sum(-1)
    score = np.mean([M[c] for c in CONDS], 0)
    out = {}
    for name, S in head_sets.items():
        if not S.any():
            continue
        flat = np.where(S.ravel(), score.ravel(), -np.inf)
        k = min(N_HEADS_PICK, int(S.sum()))
        sel = np.zeros(S.size, bool)
        sel[np.argsort(-flat, kind="stable")[:k]] = True
        sel = sel.reshape(S.shape)
        r = {"heads": int(sel.sum())}
        for c in CONDS:
            r[f"e_{c}"] = float(e[c][sel].mean())
            if e2 is not None:
                r[f"e2_{c}"] = float(e2[c][sel].mean())
                r[f"sink_{c}"] = float(sk[c][sel].mean())
        out[name] = r
    return out


def run_focus_layers_e(cq: str, root: str, bel: Dict[str, str], neu: Dict[str, str], label: str,
                       sub: str = "reading_all", out_md: Optional[str] = None) -> dict:
    """只用 CPU、读已存的数组：按层的类型（全局 / 滑动窗口 / 全部）比较被带偏和守住两组，
    三种条件下关键区 E 的份额；另给去掉固定高亮图块后的版本。只看中性答对的题。"""
    from .conflict import _items
    from .pipeline import _atomic_json
    from .sources import _signflip_p
    items = _items(cq)
    d = os.path.join(root, sub)
    data = {}
    for iid in sorted(items):
        Z = _load_all(d, iid)
        if Z is not None and iid in bel and neu.get(iid) == "对":
            data[iid] = Z
    if not data:
        raise SourcesCheckFailed(f"{d} 里没有可用的题")
    # 固定高亮图块：按图块排列分组，用每道题中性回答已记录的全部 token
    by_grid: Dict[tuple, List[str]] = {}
    for iid, Z in data.items():
        g = tuple(int(x) for x in np.asarray(Z["neutral"].get("grid_hw", [0, 0])).ravel()[:2])
        n = int(Z["neutral"]["dist_sum"].shape[-1])
        by_grid.setdefault((g, n), []).append(iid)
    sinks, no_sink = {}, []
    for key, ids in by_grid.items():
        if len(ids) < SINK_MIN_ITEMS:
            no_sink += ids
            continue
        full = {i: int((data[i]["neutral"]["count"] == READ_GROUP).sum()) for i in ids}
        dists = [_norm_img(data[i]["neutral"]["dist_sum"][:max(full[i], 1)].astype(np.float64).sum(0)) for i in ids]
        s = sink_patches(dists)
        for i in ids:
            sinks[i] = s
    rows = {W: [] for W in READ_WINDOWS}
    kinds = set()
    sink_info = {}
    for iid, Z in data.items():
        z0 = Z["neutral"]
        L, H = z0["dist_sum"].shape[1], z0["dist_sum"].shape[2]
        sl = np.asarray(z0["sliding"], bool) if "sliding" in z0 else np.zeros(L, bool)
        sets = {"全部层": np.ones((L, H), bool)}
        if sl.any():
            sets = {"全局层": np.repeat(~sl[:, None], H, 1), "滑动窗口层": np.repeat(sl[:, None], H, 1), **sets}
        kinds.update(sets)
        sk = sinks.get(iid)
        if sk is not None:
            sink_info[iid] = {"n": int(sk.shape[1]), "global": float(sk[~sl].sum(1).mean()) if (~sl).any() else None,
                              "sliding": float(sk[sl].sum(1).mean()) if sl.any() else None}
        full = min(int((Z[c]["count"] == READ_GROUP).sum()) for c in CONDS)
        grp = "对" if bel[iid] == "对" else ("偏" if bel[iid] in ("采", "他") else None)
        if grp is None:
            continue
        for W in READ_WINDOWS:
            G = W // READ_GROUP
            if full >= G:
                rows[W].append({"id": iid, "group": grp, "stats": item_e_stats(Z, G, sets, sk)})
    A = np.array
    order = [k for k in ("全局层", "滑动窗口层", "全部层") if k in kinds]
    names = [("偏", "被带偏"), ("对", "守住")]
    pp = lambda x: f"{x * 100:.1f}%"
    L = [f"# {label}：被带偏时，关键区 E 的份额（按层的类型分开）", "",
         "## 数据是什么", "",
         f"- **回答**：{label} 原模型的已有回答，每题三条：中性；用户说错（问题后加 \"I believe the correct answer is <错误答案>.\"）；"
         "用户说对（同一句话，换成正确答案）。只读已存的注意力数组，不生成、不回放。",
         "- **只看中性时答对的题**，按用户说错那条回答的结果分：被带偏 = 答错；守住 = 仍答对。",
         "- **取哪些 token**：思考的前 W 个 token；三条回答的思考都至少有 W 个 token 的题才计入。",
         "- **选头**：每题、每种层，按三种条件平均的看图总量选出 96 个头，结果是这 96 个头的平均。",
         "- **看图分布**：每个头把这 W 个 token 分给图像的注意力加起来，在图像内部归一化（所有图块加起来 = 100%）。",
         f"- **关键区 E**：每个头在**中性**回答这 W 个 token 里注意力最高的 {int(E_FRAC * 100)}% 图块。E 是用中性回答选的，"
         "所以中性一列天然偏高；**用户说错和用户说对都没有参与挑选，比较这两列（说错 − 说对）没有这个偏差**。"
         "用户说错和用户说对的提示一样长，滑动窗口里图像的位置也一样，这个比较也不受提示长度影响。",
         f"- **固定高亮图块**：同一层里，同样图块排列的所有题（至少 {SINK_MIN_ITEMS} 道）中，中性时平均份额 ≥ {SINK_MULT:g} 倍均匀份额的图块。"
         "这些位置不管图上画的是什么都被高亮（attention sink）。",
         "- **去掉固定高亮后的 E**：先把固定高亮图块去掉、其余图块重新归一化，再取中性时最高的 20% 作为 E，看三种条件落在上面的份额。",
         "- **说错 − 说对**：每题 用户说错的份额 − 用户说对的份额，再对组内平均，单位百分点；p 为每题差值正负号随机翻转 2 万次的检验。",
         "- **组间 p**：被带偏组和守住组的\"说错 − 说对\"是否不同：两组的题随机重新分组 2 万次。", ""]
    if sink_info:
        gl = [v["global"] for v in sink_info.values() if v["global"] is not None]
        sw = [v["sliding"] for v in sink_info.values() if v["sliding"] is not None]
        n0 = int(np.median([v["n"] for v in sink_info.values()]))
        L += [f"固定高亮图块的个数（每层平均，图块总数中位数 {n0}）：全局层 {np.mean(gl):.1f} 个，滑动窗口层 {np.mean(sw):.1f} 个"
              if gl and sw else f"固定高亮图块的个数：每层平均 {np.mean(gl or sw):.1f} 个（图块总数中位数 {n0}）。", ""]
    if no_sink:
        L += [f"同样图块排列的题不足 {SINK_MIN_ITEMS} 道、没有判定固定高亮图块的题：{'、'.join(sorted(no_sink))}。", ""]
    summary = []
    res = {"label": label, "rows": {str(W): rows[W] for W in READ_WINDOWS}, "sink_info": sink_info, "no_sink": sorted(no_sink)}
    for W in READ_WINDOWS:
        if not rows[W]:
            continue
        L += [f"## 前 {W} 个 token", ""]
        for kind in order:
            L += [f"### {kind}", "",
                  "| 组（题数） | E：中性 / 说错 / 说对 | **说错 − 说对**（p） | 去固定高亮的 E：中性 / 说错 / 说对 | **说错 − 说对**（p） | 固定高亮图块的份额：中性 / 说错 / 说对 |",
                  "|---|---|---|---|---|---|"]
            diffs = {}
            for g, gname in names:
                S = [r for r in rows[W] if r["group"] == g and kind in r["stats"]]
                if not S:
                    continue
                st = lambda key: A([r["stats"][kind][key] for r in S])
                e = {c: st(f"e_{c}") for c in CONDS}
                d1 = e["belief"] - e["belief_right"]
                cells = [f"{gname}（{len(S)}）", " / ".join(pp(e[c].mean()) for c in CONDS),
                         f"**{d1.mean() * 100:+.2f}**（{_signflip_p(d1):.3f}）"]
                has2 = all(f"e2_{c}" in r["stats"][kind] for r in S for c in CONDS)
                if has2:
                    e2 = {c: st(f"e2_{c}") for c in CONDS}
                    sk = {c: st(f"sink_{c}") for c in CONDS}
                    d2 = e2["belief"] - e2["belief_right"]
                    cells += [" / ".join(pp(e2[c].mean()) for c in CONDS), f"**{d2.mean() * 100:+.2f}**（{_signflip_p(d2):.3f}）",
                              " / ".join(pp(sk[c].mean()) for c in CONDS)]
                    diffs[g] = (d1, d2)
                else:
                    cells += ["—", "—", "—"]
                    diffs[g] = (d1, None)
                L.append("| " + " | ".join(cells) + " |")
                summary.append((W, kind, gname, len(S), float(d1.mean()), None if not has2 else float(d2.mean())))
            if "偏" in diffs and "对" in diffs:
                p1 = perm_test(diffs["偏"][0], diffs["对"][0])
                txt = f"组间 p（被带偏 vs 守住的\"说错 − 说对\"）：E {p1:.3f}"
                if diffs["偏"][1] is not None and diffs["对"][1] is not None:
                    txt += f"；去固定高亮的 E {perm_test(diffs['偏'][1], diffs['对'][1]):.3f}"
                L += ["", txt]
            L.append("")
    L += ["## 一张表看\"说错 − 说对\"（百分点）", "", "| 前 W 个 token | 层 | 组（题数） | E | 去固定高亮的 E |", "|---|---|---|---|---|"]
    for W, kind, gname, n, a, b in summary:
        L.append(f"| {W} | {kind} | {gname}（{n}） | {a * 100:+.2f} | {'—' if b is None else f'{b * 100:+.2f}'} |")
    L += ["", "## 题号", ""]
    W0 = READ_WINDOWS[0]
    for g, gname in names:
        ids = [r["id"] for r in rows[W0] if r["group"] == g]
        L.append(f"- {gname}（前 {W0} 个 token 时 {len(ids)} 题）：{'、'.join(ids) if ids else '无'}")
        for W in READ_WINDOWS[1:]:
            have = {r["id"] for r in rows[W] if r["group"] == g}
            miss = [i for i in ids if i not in have]
            if miss and rows[W]:
                L.append(f"  - 前 {W} 个 token 时不计入（思考不够长或没记录到这么长）：{'、'.join(miss)}")
    out_md = out_md or os.path.join(root, "reading_layers_e_report.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    _atomic_json(out_md.replace(".md", ".json"), res)
    print(f"[reading-layers-e] {label}：各长度的题数 {[len(rows[W]) for W in READ_WINDOWS]} → {out_md}", file=sys.stderr)
    return res


# ---------------------------------------------------------------- 核实：固定高亮图块在钟面上的什么位置
SINK_CATS = ["图像边缘", "钟表圆心", "指针", "钟面其他", "钟面外"]


def patch_categories(geom, grid_hw, gold: str) -> np.ndarray:
    """每个图块的位置类别（下标对应 SINK_CATS）：
    图像边缘 = 最外一圈图块；钟表圆心 = 离圆心不到 0.15 个半径；指针 = 正确时间两根针所在的图块
    （分针 0.15–1.05 个半径、时针 0.15–0.6 个半径，方向 ±15°）；钟面其他 = 钟面内其余图块；钟面外 = 其余。按这个顺序归类。"""
    from .trigger_regions import hand_shape
    cx, cy, R, W, H = geom
    gh, gw = int(grid_hw[0]), int(grid_hw[1])
    i, j = np.divmod(np.arange(gh * gw), gw)
    dx, dy = (j + 0.5) / gw * W - cx, (i + 0.5) / gh * H - cy
    r = np.hypot(dx, dy) / R
    hands = hand_shape(geom, grid_hw, gold)
    cat = np.full(gh * gw, 4)
    cat[(r >= 0.15) & (r <= 1.05)] = 3
    cat[hands] = 2
    cat[r < 0.15] = 1
    cat[(i == 0) | (i == gh - 1) | (j == 0) | (j == gw - 1)] = 0
    return cat


def _grid_of(cq: str, iid: str, z0: dict) -> Optional[np.ndarray]:
    if "grid_hw" in z0:
        return np.asarray(z0["grid_hw"]).ravel()[:2]
    rec = json.load(open(os.path.join(cq, "obs", "records", f"{iid}__neutral.json"), encoding="utf-8"))
    npz = rec.get("npz") or os.path.join(cq, "obs", "attn", f"{iid}__neutral.npz")
    if not os.path.exists(npz):
        return None
    with np.load(npz) as z:
        return np.asarray(z["grid_hw"]).ravel()[:2] if "grid_hw" in z.files else None


def run_sink_check(cq: str, root: str, label: str, sub: str = "reading_all", out_md: Optional[str] = None,
                   n_pictures: int = 3) -> dict:
    """只用 CPU、读已存数组：固定高亮图块（与 run_focus_layers_e 同一个定法）落在钟面的什么位置，
    和"随便挑同样多图块"相比是否更多落在钟表圆心或指针上。按层的类型分开。用全部中性答对的题。"""
    from .conflict import _items
    from .pipeline import _atomic_json
    from .sources import dial_geometry
    items = _items(cq)
    d = os.path.join(root, sub)
    data = {}
    for iid in sorted(items):
        p = os.path.join(d, f"{iid}__neutral.npz")
        if os.path.exists(p):
            with np.load(p) as z:
                data[iid] = {k: z[k] for k in z.files}
    if not data:
        raise SourcesCheckFailed(f"{d} 里没有中性回答的数组")
    by_grid: Dict[tuple, List[str]] = {}
    for iid, z0 in data.items():
        g = tuple(int(x) for x in np.asarray(z0.get("grid_hw", [0, 0])).ravel()[:2])
        by_grid.setdefault((g, int(z0["dist_sum"].shape[-1])), []).append(iid)
    out = {"label": label, "kinds": {}, "items": 0, "skipped": []}
    pics_dir = os.path.join(root, "sink_check")
    os.makedirs(pics_dir, exist_ok=True)
    acc = {}                           # 层类型 → {"n_sink": [], "count": [5], "chance": [5], "mass": [5], "mass_total": []}
    examples = []
    for key, ids in by_grid.items():
        if len(ids) < SINK_MIN_ITEMS:
            out["skipped"] += ids
            continue
        dists = {}
        for i in ids:
            z0 = data[i]
            full = max(int((z0["count"] == READ_GROUP).sum()), 1)
            dists[i] = _norm_img(z0["dist_sum"][:full].astype(np.float64).sum(0))      # [L, H, n]
        sink = sink_patches(list(dists.values()))                                     # [L, n]
        for i in ids:
            z0 = data[i]
            L = z0["dist_sum"].shape[1]
            sl = np.asarray(z0["sliding"], bool) if "sliding" in z0 else np.zeros(L, bool)
            grid = _grid_of(cq, i, z0)
            if grid is None or int(np.prod(grid)) != z0["dist_sum"].shape[-1]:
                out["skipped"].append(i)
                continue
            geom = dial_geometry(items[i]["image"])
            cat = patch_categories(geom, grid, items[i]["gold"])
            chance = np.bincount(cat, minlength=len(SINK_CATS)) / len(cat)
            kinds = {"全局层": ~sl, "滑动窗口层": sl} if sl.any() else {"全部层": np.ones(L, bool)}
            for kind, lm in kinds.items():
                if not lm.any():
                    continue
                a = acc.setdefault(kind, {"n_sink": [], "count": [], "chance": [], "mass": [], "mass_total": []})
                s = sink[lm]                                                           # [l, n]
                pm = dists[i][lm].mean(1)                                              # [l, n]：对头平均
                cnt = np.array([np.bincount(cat[row], minlength=len(SINK_CATS)) for row in s]).sum(0)
                a["n_sink"].append(float(s.sum(1).mean()))
                a["count"].append(cnt / max(cnt.sum(), 1))
                a["chance"].append(chance)
                ms = np.array([[(pm[k] * (s[k] & (cat == c))).sum() for c in range(len(SINK_CATS))] for k in range(len(s))]).mean(0)
                a["mass"].append(ms / max(ms.sum(), 1e-12))
                a["mass_total"].append(float((pm * s).sum(1).mean()))
            out["items"] += 1
            if len(examples) < n_pictures:
                examples.append((i, grid, sink, sl, cat))
    if not acc:
        raise SourcesCheckFailed("没有能判定固定高亮图块的题（同样图块排列的题都不足 5 道）")
    pct = lambda x: f"{x * 100:.1f}%"
    L = [f"# {label}：固定高亮图块落在钟面的什么位置", "",
         "## 数据是什么", "",
         f"- **回答**：{label} 原模型中性回答的已存注意力数组（与上次关键区 E 分析是同一批数组），只用 CPU，不生成、不回放。",
         f"- **固定高亮图块**（与上次同一个定法）：同一层里，同样图块排列的所有题（至少 {SINK_MIN_ITEMS} 道）中，"
         f"中性时平均份额 ≥ {SINK_MULT:g} 倍均匀份额的图块。",
         "- **位置类别**（每道题按自己的钟面算，按下面顺序归类）：",
         "  - 图像边缘：图片最外一圈的图块；",
         "  - 钟表圆心：离钟表圆心不到 0.15 个半径；",
         "  - 指针：正确时间两根针所在的图块（分针方向 ±15°、0.15–1.05 个半径；时针方向 ±15°、0.15–0.6 个半径）；",
         "  - 钟面其他：钟面内的其余图块；",
         "  - 钟面外：其余（图片里钟面以外、但不在最外一圈的图块）。",
         "- **随便挑的比例**：一道题全部图块里，各类别占多少。固定高亮图块某一类的比例明显高于这个数，说明它们偏向这一类位置。",
         "- **注意力的比例**：固定高亮图块分到的注意力里，各类别占多少（对头平均）。", ""]
    for kind, a in acc.items():
        cnt, ch, ms = np.mean(a["count"], 0), np.mean(a["chance"], 0), np.mean(a["mass"], 0)
        L += [f"## {kind}", "",
              f"每层平均 {np.mean(a['n_sink']):.1f} 个固定高亮图块；它们占这类层看图注意力的 {pct(np.mean(a['mass_total']))}（{len(a['count'])} 道题平均）。", "",
              "| 位置 | 固定高亮图块里占多少 | 随便挑时占多少 | 固定高亮图块分到的注意力里占多少 |", "|---|---:|---:|---:|"]
        for c, name in enumerate(SINK_CATS):
            L.append(f"| {name} | {pct(cnt[c])} | {pct(ch[c])} | {pct(ms[c])} |")
        L.append("")
        out["kinds"][kind] = {"n_sink_per_layer": float(np.mean(a["n_sink"])), "sink_mass": float(np.mean(a["mass_total"])),
                              "count": dict(zip(SINK_CATS, cnt.tolist())), "chance": dict(zip(SINK_CATS, ch.tolist())),
                              "mass": dict(zip(SINK_CATS, ms.tolist()))}
    # 示意图：固定高亮图块画在原图上（红 = 滑动窗口层，蓝 = 全局层；格子里的数字 = 有几层把它判为固定高亮）
    try:
        from PIL import Image, ImageDraw
        names = []
        for i, grid, sink, sl, cat in examples:
            im = Image.open(items[i]["image"]).convert("RGB")
            W, H = im.size
            gh, gw = int(grid[0]), int(grid[1])
            dr = ImageDraw.Draw(im)
            groups = [("滑动窗口层", sl, (230, 30, 30)), ("全局层", ~sl, (30, 80, 230))] if sl.any() else [("全部层", np.ones(len(sink), bool), (230, 30, 30))]
            for _, lm, color in groups:
                if not lm.any():
                    continue
                cnt = sink[lm].sum(0)
                for p in np.flatnonzero(cnt):
                    r_, c_ = divmod(int(p), gw)
                    box = (c_ * W / gw, r_ * H / gh, (c_ + 1) * W / gw - 1, (r_ + 1) * H / gh - 1)
                    dr.rectangle(box, outline=color, width=2)
                    dr.text((box[0] + 2, box[1] + 1), str(int(cnt[p])), fill=color)
            f = os.path.join(pics_dir, f"{i}_sinks.png")
            im.save(f)
            names.append(os.path.basename(f))
        L += ["## 示意图", "", "sink_check/ 里的图：把固定高亮图块画在原图上。红框 = 滑动窗口层，蓝框 = 全局层，框里的数字 = 有几层把这个图块判为固定高亮。", ""]
        L += [f"- {n}" for n in names] + [""]
    except Exception as e:
        L += [f"（示意图没有画出：{type(e).__name__}: {e}）", ""]
    if out["skipped"]:
        L += [f"不计入（同样图块排列的题不足 {SINK_MIN_ITEMS} 道，或找不到图块排列）：{'、'.join(sorted(set(out['skipped'])))}。", ""]
    out_md = out_md or os.path.join(root, "sink_check_report.md")
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    _atomic_json(out_md.replace(".md", ".json"), out)
    print(f"[sink-check] {label}：{out['items']} 道题 → {out_md}", file=sys.stderr)
    return out
