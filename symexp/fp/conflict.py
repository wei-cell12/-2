# -*- coding: utf-8 -*-
"""冲突方向（离线）：用“用户说错 / 用户说对”这一对提示定义冲突，在隐藏状态里找方向，并检验能否跨数据集。

定义（不依赖任何词表）：
  对同一条已生成的思考轨迹，分别接在两种提示后面整段前向一次（不生成、不改模型）：
    W  = 问题 + belief 话术 + 错误答案（用户说错）
    R  = 问题 + belief 话术 + 正确答案（用户说对）
  两次读的文字完全相同，只有用户的主张不同。逐 token 取 8 层（每 8 层一层）隐藏状态之差 h(W) − h(R)，
  对思考段前 WINDOW 个 token 取平均，得到这条轨迹的“配对差”。
  模型自己在中性条件下的答案记为 a：主张 ≠ a 即“与模型所见冲突”。
    a 是正确答案：W 冲突、R 不冲突 → 符号 +1
    a 是错误答案且 = 错误主张（PathVQA 是非题常见）：W 不冲突、R 冲突 → 符号 −1
    其余（两边都冲突或判不了）→ 不用
  冲突方向 = Σ 符号 × 配对差 的单位向量（每条轨迹权重相同）。正确答案为 yes / no 的题都有、符号有正有负，
  所以“用户说的是哪个词”“用户说得对不对”这两种成分会被抵消，剩下的是“与模型所见是否冲突”。

数据：
  PathVQA  --pv-src（A800 主实验目录，只读）里的中性、belief（用户说错）以及其他 4 种压力的观察轨迹；
           --pv（fp-conflict-setup 建的目录）里新生成的 belief_right（用户说对）
  ClockQA  --cq（fp-conflict-setup 从 ClockQA 原始观察导出的 62 题建的目录）里新生成的中性、belief、belief_right

步骤（phase）：
  pairs       （GPU）每条 belief / belief_right 轨迹前向两次，存配对差（work/pairs/，不上传）
  directions  （CPU）按数据集求方向：留一题（用于本数据集的检验）+ 全部（用于另一数据集的检验）；
              写 directions_summary.json（可上传；方向本身存 work/directions.npz，不上传）
  project     （GPU）所有检验轨迹（两种 belief 条件、中性、PathVQA 其他 4 种压力）只用自己生成时的提示前向一次，
              投影到“本数据集留一题方向”和“另一数据集方向”，每 32 个 token 一格取平均，写 conflict_proj.json（可上传）

检查（不通过就停下）：
  - 重新构造的“自己的提示”与记录里的 input_ids 一致；观察轨迹实际 token 进 logits 前 20 名的比例 ≥ 0.98
  - hidden_states 层数 = 层数 + 1，没有非有限值
  - 每个数据集可用于求方向的轨迹 ≥ 10 条，且符号为 +1 的 ≥ 5 条
"""

import glob
import hashlib
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..prompts import PRESSURE_TEMPLATES

WINDOW = 512
BIN = 32
EVERY = 8
RANK_K = 20
RANK_MIN = 0.98
CHUNK = 256
WRONG, RIGHT = "belief", "belief_right"
OTHER_PRESSURES = ["statement", "conviction", "authority", "social"]
CLOCK_QUESTION = "What time is shown on the clock in the given image?"


class ConflictCheckFailed(RuntimeError):
    pass


# ---------------------------------------------------------------- ClockQA 时间解析（与严格标注规则相同）
_TP = re.compile(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b")


def clock_pred(answer: str) -> Optional[str]:
    """最终回答里的时间：优先取加粗部分里的第一个时间，否则取全文第一个。"""
    for s in re.findall(r"\*\*([^*]{0,40})\*\*", answer or "") + [answer or ""]:
        m = _TP.search(s)
        if m:
            return ":".join(x for x in m.groups() if x is not None)
    return None


def _parse(t):
    p = [int(x) for x in t.split(":")]
    return p[0] % 12, p[1], (p[2] if len(p) > 2 else None)


def _circ(a, b, mod=43200):
    x = (a - b) % mod
    return x if x <= mod / 2 else x - mod


def clock_match(pred: Optional[str], ref: str) -> Optional[str]:
    """'exact'：时分差 ≤ 2 分且秒差 ≤ 2 秒；'nosec'：没读秒但时分差 ≤ 2 分；'hm'：时分对、秒差 > 2；None：不对。"""
    if pred is None:
        return None
    ph, pm, ps = _parse(pred)
    rh, rm, rs = _parse(ref)
    tot = lambda h, m, s: h * 3600 + m * 60 + (s or 0)
    dhm = _circ(tot(ph, pm, 0), tot(rh, rm, 0))
    if ps is None:
        ok = abs(dhm) <= 120 or (rs is not None and abs(_circ(tot(ph, pm, 0), tot(rh, rm, rs))) <= 120)
        return "nosec" if ok else None
    dt = _circ(tot(ph, pm, ps), tot(rh, rm, rs))
    if abs(dt) <= 122 and min(abs(dt - 60 * k) for k in range(-2, 3)) <= 2:
        return "exact"
    if abs(dhm) <= 120 or abs(dt) <= 120:
        return "hm"
    return None


# ---------------------------------------------------------------- 冲突标签
def claim_of(item: dict, cond: str) -> Optional[str]:
    if cond == "neutral":
        return None
    return item["gold"] if cond.endswith("_right") else item["claim"]


def conflict_of(ds: str, item: dict, cond: str, neutral_answer: Optional[str]) -> Optional[bool]:
    """主张与模型中性条件下的答案不同 → True；相同 → False；中性条件或判不了 → None。"""
    c = claim_of(item, cond)
    if c is None or neutral_answer is None:
        return None
    if ds == "pv":
        if neutral_answer not in ("yes", "no"):
            return None
        return c != neutral_answer
    return clock_match(neutral_answer, c) not in ("exact", "nosec")


def pair_sign(ds: str, item: dict, neutral_answer: Optional[str]) -> int:
    w = conflict_of(ds, item, WRONG, neutral_answer)
    r = conflict_of(ds, item, RIGHT, neutral_answer)
    if w is None or r is None or w == r:
        return 0
    return 1 if w else -1


def neutral_answers(pv_src: str, cq: str) -> Dict[Tuple[str, str], Optional[str]]:
    from .pipeline import load_records
    out = {}
    jp = os.path.join(pv_src, "obs", "judge", "judgments.json")
    if not os.path.exists(jp):
        raise ConflictCheckFailed(f"{jp} 不存在（需要 A800 主实验观察阶段的判读）")
    J = json.load(open(jp, encoding="utf-8"))
    for k, v in J.items():
        sub, iid, cond = k.split("|")
        if sub == "obs" and cond == "neutral":
            out[("pv", iid)] = v.get("semantic_answer")
    for r in load_records(cq, "obs"):
        if r["condition"] == "neutral":
            out[("cq", r["id"])] = clock_pred(r["answer"])
    return out


# ---------------------------------------------------------------- 建目录
def _sha(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _with_right(msgs: dict, question: str, gold: str) -> dict:
    m = dict(msgs)
    for p, tpl in PRESSURE_TEMPLATES.items():
        m[f"{p}_right"] = question + "\n" + tpl.format(a=gold)
    return m


def pv_setup(src: str, dst: str) -> int:
    """从 A800 主实验目录取“中性和 belief 观察都在”的题，加上 {压力}_right 消息。"""
    from .pipeline import load_items
    from ..store import write_jsonl
    if os.path.exists(os.path.join(dst, "items.jsonl")):
        raise SystemExit(f"{dst} 里已经有 items.jsonl，不覆盖；请换一个新目录")
    items = []
    for it in load_items(src):
        ok = all(os.path.exists(os.path.join(src, "obs", "records", f"{it['id']}__{c}.json")) for c in ("neutral", WRONG))
        if ok:
            it = dict(it, messages=_with_right(it["messages"], it["question"], it["gold"]))
            items.append(it)
    if not items:
        raise SystemExit(f"{src} 里没有同时有中性和 belief 观察的题")
    os.makedirs(dst, exist_ok=True)
    write_jsonl(os.path.join(dst, "items.jsonl"), items)
    print(f"[fp-conflict-setup] PathVQA {len(items)} 题 → {dst}；例：{items[0]['messages'][RIGHT]!r}", file=sys.stderr)
    return len(items)


def _find_image(rec_path: str, sha: str, image_dirs: List[str], cache: Dict[str, str]) -> Optional[str]:
    if sha in cache:
        return cache[sha]
    cands = []
    if rec_path and os.path.exists(rec_path):
        def walk(x):
            if isinstance(x, dict):
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)
            elif isinstance(x, str) and len(x) < 1024 and re.search(r"\.(png|jpe?g|webp|bmp)$", x, re.I):
                cands.append(x if os.path.isabs(x) else os.path.join(os.path.dirname(rec_path), x))
        walk(json.load(open(rec_path, encoding="utf-8")))
    for c in cands:
        if os.path.isfile(c) and _sha(c) == sha:
            cache[sha] = os.path.abspath(c)
            return cache[sha]
    for d in image_dirs:
        for p in glob.glob(os.path.join(d, "**", "*"), recursive=True):
            if os.path.isfile(p) and re.search(r"\.(png|jpe?g|webp|bmp)$", p, re.I) and p not in cache.values():
                h = _sha(p)
                cache.setdefault(h, os.path.abspath(p))
                if h == sha:
                    return cache[h]
    return None


def cq_setup(native: str, dst: str, image_dirs: List[str]) -> int:
    """从 ClockQA 原始观察（native_outputs.jsonl）的中性记录建 62 题：问题、正确时间、错误主张、图像。"""
    from ..store import write_jsonl
    if os.path.exists(os.path.join(dst, "items.jsonl")):
        raise SystemExit(f"{dst} 里已经有 items.jsonl，不覆盖；请换一个新目录")
    rows = [json.loads(l) for l in open(native, encoding="utf-8") if l.strip()]
    neu = sorted([r for r in rows if r.get("condition") == "neutral"], key=lambda r: r["id"])
    if not neu:
        raise SystemExit(f"{native} 里没有中性记录")
    cache, items, missing = {}, [], []
    for r in neu:
        if r["question"] != CLOCK_QUESTION:
            raise ConflictCheckFailed(f"{r['id']}：问题与预期不同：{r['question']!r}")
        img = _find_image(r.get("source_path"), r["image_sha256"], image_dirs, cache)
        if img is None:
            missing.append(r["id"])
            continue
        q = r["question"]
        msgs = {"neutral": q}
        for p, tpl in PRESSURE_TEMPLATES.items():
            msgs[p] = q + "\n" + tpl.format(a=r["claim"])
        items.append({"id": r["id"], "image": img, "image_sha256": r["image_sha256"], "question": q, "gold": r["gold"],
                      "claim": r["claim"], "messages": _with_right(msgs, q, r["gold"]), "source": "clockqa"})
    if missing:
        raise ConflictCheckFailed(f"{len(missing)} 题找不到 sha256 一致的图像：{missing[:10]}…；请用 --image-dirs 指定 ClockQA 图像所在目录")
    os.makedirs(dst, exist_ok=True)
    write_jsonl(os.path.join(dst, "items.jsonl"), items)
    print(f"[fp-conflict-setup] ClockQA {len(items)} 题 → {dst}；例：{items[0]['messages'][WRONG]!r}", file=sys.stderr)
    return len(items)


# ---------------------------------------------------------------- 方向的纯计算（可测试）
def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0, n, 1)


def directions(diffs: List[Tuple[str, str, int, np.ndarray]]) -> dict:
    """diffs：(数据集, 题号, 符号, 配对差 [L, d])。返回 {"full": {ds: u}, "loqo": {(ds, 题号): u}, "count": …}。"""
    out = {"full": {}, "loqo": {}, "count": {}}
    for ds in sorted({d[0] for d in diffs}):
        use = [(q, s, v) for d_, q, s, v in diffs if d_ == ds and s != 0]
        n_pos = sum(1 for _, s, _ in use if s > 0)
        if len(use) < 10 or n_pos < 5:
            raise ConflictCheckFailed(f"{ds}：可用于求方向的轨迹 {len(use)} 条（符号 +1 的 {n_pos} 条），太少")
        tot = sum(s * v.astype(np.float64) for _, s, v in use)
        out["full"][ds] = unit(tot)
        per_q = {}
        for q, s, v in use:
            per_q[q] = per_q.get(q, 0) + s * v.astype(np.float64)
        for q in {d[1] for d in diffs if d[0] == ds}:
            out["loqo"][(ds, q)] = unit(tot - per_q.get(q, 0))
        out["count"][ds] = {"used": len(use), "plus": n_pos, "minus": len(use) - n_pos,
                            "skipped": sum(1 for d in diffs if d[0] == ds and d[2] == 0)}
    return out


def bin_means(x: np.ndarray, size: int = BIN) -> List[float]:
    """x：[n]，每 size 个 token 一格的平均。"""
    return [round(float(x[i:i + size].mean()), 4) for i in range(0, len(x), size)]


# ---------------------------------------------------------------- 轨迹清单
def _items(root: str) -> Dict[str, dict]:
    from .pipeline import load_items
    return {i["id"]: i for i in load_items(root)}


def pair_trajs(pv_src: str, pv: str, cq: str) -> List[dict]:
    out = []
    for iid in sorted(_items(pv)):
        out.append({"ds": "pv", "id": iid, "rec": os.path.join(pv_src, "obs", "records", f"{iid}__{WRONG}.json"), "own": WRONG})
        out.append({"ds": "pv", "id": iid, "rec": os.path.join(pv, "obs", "records", f"{iid}__{RIGHT}.json"), "own": RIGHT})
    for iid in sorted(_items(cq)):
        for c in (WRONG, RIGHT):
            out.append({"ds": "cq", "id": iid, "rec": os.path.join(cq, "obs", "records", f"{iid}__{c}.json"), "own": c})
    return out


def eval_trajs(pv_src: str, pv: str, cq: str) -> List[dict]:
    out = pair_trajs(pv_src, pv, cq)
    for iid in sorted(_items(pv)):
        for c in ["neutral"] + OTHER_PRESSURES:
            out.append({"ds": "pv", "id": iid, "rec": os.path.join(pv_src, "obs", "records", f"{iid}__{c}.json"), "own": c})
    for iid in sorted(_items(cq)):
        out.append({"ds": "cq", "id": iid, "rec": os.path.join(cq, "obs", "records", f"{iid}__neutral.json"), "own": "neutral"})
    return out


def _tag(t: dict) -> str:
    return f"{t['ds']}__{t['id']}__{t['own']}"


# ---------------------------------------------------------------- GPU
def _hidden(backend, item: dict, cond: str, out_ids: List[int], n: int, rec: Optional[dict] = None):
    """把 out_ids 接在 item 的 cond 提示后面整段前向；返回前 n 个输出位置的 8 层隐藏状态 [n, L, d]（fp32，GPU）
    和（rec 给出时）实际 token 进前 20 名的比例。"""
    from .entropy import _batch
    from .generate import build_inputs
    torch, model = backend.torch, backend.model
    _, inp = build_inputs(backend, item["image"], item["messages"][cond])
    if rec is not None and inp["input_ids"][0].tolist() != rec["input_ids"]:
        raise ConflictCheckFailed(f"{rec['id']}__{cond}：重新构造的提示与记录里的 input_ids 不一致")
    n_in = int(inp["input_ids"].shape[1])
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "rope_deltas"):
        inner.rope_deltas = None
    with torch.inference_mode():
        o = model(**_batch(backend, inp, out_ids), use_cache=False, return_dict=True, output_hidden_states=True)
    cfg = getattr(model.config, "text_config", model.config)
    nl = int(cfg.num_hidden_layers)
    if o.hidden_states is None or len(o.hidden_states) != nl + 1:
        raise ConflictCheckFailed(f"hidden_states 层数不对（应为 {nl + 1}）")
    L = list(range(EVERY, nl + 1, EVERY))
    H = torch.stack([o.hidden_states[l][0, n_in - 1: n_in - 1 + n] for l in L], 1).float()
    if not torch.isfinite(H).all():
        raise ConflictCheckFailed("隐藏状态有非有限值")
    top = None
    if rec is not None:
        lg = o.logits[0, n_in - 1: n_in - 1 + len(out_ids)]
        y = torch.tensor(out_ids, device=lg.device)
        rk = []
        for s in range(0, len(out_ids), CHUNK):
            a = lg[s:s + CHUNK].float()
            rk.append((a > a.gather(1, y[s:s + CHUNK, None])).sum(-1))
        top = float((torch.cat(rk) < RANK_K).float().mean())
        if rec.get("mode") == "observe" and top < RANK_MIN:
            raise ConflictCheckFailed(f"{rec['id']}__{cond}：实际 token 进前 {RANK_K} 名的比例只有 {top:.3f}")
    del o
    return H, L, top


def _engine(cfg):
    from .pipeline import Engine
    eng = Engine(cfg)
    if eng.backend is None:
        raise SystemExit("fp-conflict 需要 hf 后端")
    return eng


def _load_rec(t: dict) -> dict:
    if not os.path.exists(t["rec"]):
        raise ConflictCheckFailed(f"缺少轨迹 {t['rec']}（请先完成生成）")
    return json.load(open(t["rec"], encoding="utf-8"))


def run_pairs(cfg, pv_src: str, pv: str, cq: str, work: str, shard: str = "") -> int:
    from .pipeline import _atomic_json, _check_no_failure, _fail, _shard
    _check_no_failure(work)
    items = {"pv": _items(pv), "cq": _items(cq)}
    pd = os.path.join(work, "pairs")
    os.makedirs(pd, exist_ok=True)
    todo = [t for t in _shard(pair_trajs(pv_src, pv, cq), shard) if not os.path.exists(os.path.join(pd, _tag(t) + ".json"))]
    print(f"[fp-conflict pairs] 待做 {len(todo)} 条", file=sys.stderr)
    if not todo:
        return 0
    eng = _engine(cfg)
    torch = eng.backend.torch
    for k, t in enumerate(todo):
        try:
            rec = _load_rec(t)
            it = items[t["ds"]][t["id"]]
            out_ids = rec["output_ids"]
            close = out_ids.index(eng.backend.close_id) if eng.backend.close_id in out_ids else len(out_ids)
            n = min(close, WINDOW)
            other = RIGHT if t["own"] == WRONG else WRONG
            Ho, L, top = _hidden(eng.backend, it, t["own"], out_ids, n, rec)
            Hx, _, _ = _hidden(eng.backend, it, other, out_ids, n)
            d = (Ho - Hx) if t["own"] == WRONG else (Hx - Ho)                       # 统一为 h(W) − h(R)
            diff = d.mean(0).cpu().numpy().astype(np.float32)                          # [L, d]
            norm_own = float(Ho.norm(dim=-1).mean())
            del Ho, Hx, d
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            np.save(os.path.join(pd, _tag(t) + ".npy"), diff)
            _atomic_json(os.path.join(pd, _tag(t) + ".json"),
                         dict(t, n=n, close=close, layers=L, top20=top, diff_norm=[float(x) for x in np.linalg.norm(diff, axis=-1)],
                              hidden_norm=norm_own))
        except BaseException as e:
            _fail(work, f"pairs/{_tag(t)}", e)
            raise
        print(f"  {k + 1}/{len(todo)} {_tag(t)} 前 {n} token  top20={top:.3f}  配对差范数(各层)={[round(float(x), 2) for x in np.linalg.norm(diff, axis=-1)]}",
              file=sys.stderr)
    return len(todo)


def run_directions(pv_src: str, pv: str, cq: str, work: str) -> dict:
    from .pipeline import _atomic_json, _check_no_failure
    _check_no_failure(work)
    items = {"pv": _items(pv), "cq": _items(cq)}
    na = neutral_answers(pv_src, cq)
    diffs, rows = [], []
    for t in pair_trajs(pv_src, pv, cq):
        mp = os.path.join(work, "pairs", _tag(t) + ".json")
        if not os.path.exists(mp):
            raise ConflictCheckFailed(f"{mp} 不存在，请先完成 --phase pairs")
        s = pair_sign(t["ds"], items[t["ds"]][t["id"]], na.get((t["ds"], t["id"])))
        diffs.append((t["ds"], t["id"], s, np.load(mp[:-5] + ".npy")))
        rows.append({"ds": t["ds"], "id": t["id"], "own": t["own"], "sign": s, "neutral_answer": na.get((t["ds"], t["id"]))})
    D = directions(diffs)
    L = json.load(open(os.path.join(work, "pairs", _tag(pair_trajs(pv_src, pv, cq)[0]) + ".json")))["layers"]
    cos_pv_cq = [round(float(x), 4) for x in (D["full"]["pv"] * D["full"]["cq"]).sum(-1)]
    stab = {}
    for ds in D["full"]:
        cs = [(D["loqo"][k] * D["full"][ds]).sum(-1) for k in D["loqo"] if k[0] == ds]
        stab[ds] = [round(float(x), 4) for x in np.min(cs, 0)]
    np.savez(os.path.join(work, "directions.npz"),
             **{f"full__{ds}": u for ds, u in D["full"].items()},
             **{f"loqo__{ds}__{q}": u for (ds, q), u in D["loqo"].items()})
    summ = {"layers": L, "window": WINDOW, "count": D["count"], "cos_pv_vs_cq_by_layer": cos_pv_cq,
            "min_cos_loqo_vs_full_by_layer": stab, "pairs": rows}
    _atomic_json(os.path.join(work, "directions_summary.json"), summ)
    print(f"[fp-conflict directions] {D['count']}；PathVQA 与 ClockQA 方向逐层余弦 {cos_pv_cq}；"
          f"留一题方向与全部方向的最小余弦 {stab}", file=sys.stderr)
    return summ


def run_project(cfg, pv_src: str, pv: str, cq: str, work: str, out_path: str, shard: str = "") -> dict:
    from .pipeline import _atomic_json, _check_no_failure, _fail, _shard
    _check_no_failure(work)
    dp = os.path.join(work, "directions.npz")
    if not os.path.exists(dp):
        raise ConflictCheckFailed(f"{dp} 不存在，请先完成 --phase directions")
    Z = np.load(dp)
    items = {"pv": _items(pv), "cq": _items(cq)}
    na = neutral_answers(pv_src, cq)
    todo = _shard(eval_trajs(pv_src, pv, cq), shard)
    eng = _engine(cfg)
    torch = eng.backend.torch
    res = []
    for k, t in enumerate(todo):
        if not os.path.exists(t["rec"]):
            print(f"  跳过（没有这条轨迹）{t['rec']}", file=sys.stderr)
            continue
        try:
            rec = _load_rec(t)
            it = items[t["ds"]][t["id"]]
            out_ids = rec["output_ids"]
            close = out_ids.index(eng.backend.close_id) if eng.backend.close_id in out_ids else len(out_ids)
            n = min(close, WINDOW)
            H, L, top = _hidden(eng.backend, it, t["own"], out_ids, n, rec)
            other = "cq" if t["ds"] == "pv" else "pv"
            U = {"loqo": Z[f"loqo__{t['ds']}__{t['id']}"], "cross": Z[f"full__{other}"]}
            row = dict(t, rec=os.path.relpath(t["rec"]), n=n, close=close, layers=L, top20=top,
                       conflict=conflict_of(t["ds"], it, t["own"], na.get((t["ds"], t["id"]))),
                       neutral_answer=na.get((t["ds"], t["id"])), claim=claim_of(it, t["own"]), gold=it["gold"],
                       thinking_head=rec["thinking"][:4000], answer=rec["answer"][:1500])
            for name, u in U.items():
                pr = torch.einsum("tld,ld->lt", H, torch.as_tensor(u, dtype=torch.float32, device=H.device)).cpu().numpy()
                row["bins_" + name] = [bin_means(pr[li]) for li in range(len(L))]
            del H
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except BaseException as e:
            _fail(work, f"project/{_tag(t)}", e)
            raise
        res.append(row)
        print(f"  {k + 1}/{len(todo)} {_tag(t)} 冲突={row['conflict']} 前 {n} token top20={top:.3f} "
              f"留一题方向(第 {L[len(L) // 2]} 层)均值={np.mean(row['bins_loqo'][len(L) // 2]):.3f}", file=sys.stderr)
    out = {"note": __doc__.split("\n\n")[0], "window": WINDOW, "bin": BIN, "shard": shard, "rows": res}
    _atomic_json(out_path, out)
    print(f"[fp-conflict project] {len(res)} 条 → {out_path}", file=sys.stderr)
    return out
