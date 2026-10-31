# -*- coding: utf-8 -*-
"""ClockQA：模型写读数时，注意力有多少落在图像上、有多少落在用户的话上（只读回放，不生成）。

把原模型已有的回答（neutral / belief / belief_right，$CQ/obs/records）原样喂回模型做一次前向，
在 16 个标准注意力层、每个头上，记录“生成第 k 个输出 token 时”对下面 8 类 key 的注意力总量：

  0 image       图像 token
  1 question    题干文字（用户消息里断言那一行之前的部分）
  2 claim_time  断言里的时间本身（如 2:33:10）
  3 claim_rest  断言那一行的其余文字（I believe the correct answer is ... .）
  4 template    模板、角色标记、<think> 等其余提示 token
  5 gen_claim   模型自己写出的、与用户时间一致的时间串（复述用户）
  6 gen_gold    模型自己写出的、与正确时间一致的时间串
  7 gen_other   其余已生成的 token
每行 8 类之和 = 1。中性条件没有断言，2、3 为 0；gen_claim 用 belief 条件的错误时间来找。
用户说对时（belief_right）断言就是正确时间，模型写出的正确时间一律记为 gen_claim。
另外记录：每行在“看图头”（与锚点一相同的 96 个头，来自该题中性回答）上对每个图像块的平均注意力。

analyze（CPU）：把思考切成句子，按规则找出“读图句”（描述指针位置、或写出时间结论、且不提用户），
在这些句子的 token 上比较图像注意力与用户断言注意力的比值，按条件、按结果（坚持 / 采纳）、
按句子里的时间（正确 / 用户的）分组，并给出前 512 个 token 的时间曲线。
"""

import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import LAYERS

CATS = ["image", "question", "claim_time", "claim_rest", "template", "gen_claim", "gen_gold", "gen_other"]
CONDS = ["neutral", "belief", "belief_right"]
SUM_TOL = 2e-3
EPS = 1e-6
BIN = 32
WINDOW = 512


class SourcesCheckFailed(RuntimeError):
    pass


# ---------------------------------------------------------------- token 文本与类别（纯函数）
def token_pieces(decode, ids: List[int], window: int = 8) -> Tuple[List[str], bool]:
    """每个 token 对应的文字片段；拼起来应等于整段 decode。返回 (片段, 是否一致)。"""
    full = decode(ids)
    out = []
    for k in range(len(ids)):
        a = max(0, k - window)
        prev, cur = decode(ids[a:k]), decode(ids[a:k + 1])
        out.append(cur[len(prev):] if cur.startswith(prev) else None)
    if all(p is not None for p in out) and "".join(out) == full:
        return out, True
    out, prev = [], ""
    for k in range(len(ids)):
        cur = decode(ids[:k + 1])
        out.append(cur[len(prev):] if cur.startswith(prev) else "")
        prev = cur
    return out, "".join(out) == full


def offsets(pieces: List[str]) -> np.ndarray:
    o = np.zeros(len(pieces) + 1, np.int64)
    o[1:] = np.cumsum([len(p) for p in pieces])
    return o


def _ov(a0, a1, b0, b1) -> bool:
    return a0 < b1 and b0 < a1


_TIME = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")


def time_kind(t: str, gold: str, claim: Optional[str]) -> str:
    """'claim'（与用户时间一致，含时分一致秒不同）/ 'gold'（与正确时间一致）/ 'other'。用户时间优先。"""
    from .conflict import clock_match
    try:
        if claim and clock_match(t, claim) in ("exact", "nosec", "hm"):
            return "claim"
        if clock_match(t, gold) in ("exact", "nosec"):
            return "gold"
    except Exception:
        pass
    return "other"


def time_spans(text: str, gold: str, claim: Optional[str]) -> List[Tuple[int, int, str]]:
    return [(m.start(), m.end(), time_kind(m.group(0), gold, claim)) for m in _TIME.finditer(text)]


def prompt_categories(pieces: List[str], ids: List[int], img_tok: int, message: str, cond: str,
                      claim: Optional[str]) -> Tuple[np.ndarray, dict]:
    P = "".join(pieces)
    off = offsets(pieces)
    m0 = P.find(message)
    if m0 < 0:
        raise SourcesCheckFailed("提示文字里找不到用户消息")
    spans = {"message": [m0, m0 + len(message)]}
    if cond != "neutral":
        line = message.split("\n")[-1]
        c0 = m0 + message.rfind(line)
        c1 = c0 + len(line)
        t = line.find(claim or "\0")
        if t < 0:
            raise SourcesCheckFailed(f"断言那一行里找不到用户时间 {claim!r}：{line!r}")
        t0, t1 = c0 + t, c0 + t + len(claim)
        q = (m0, c0)
        spans.update(claim_line=[c0, c1], claim_time=[t0, t1], question=list(q))
    else:
        q = (m0, m0 + len(message))
        spans.update(question=list(q))
    cats = np.full(len(ids), 4, np.int8)
    for k in range(len(ids)):
        a, b = int(off[k]), int(off[k + 1])
        if ids[k] == img_tok:
            cats[k] = 0
        elif cond != "neutral" and _ov(a, b, t0, t1):
            cats[k] = 2
        elif cond != "neutral" and _ov(a, b, c0, c1):
            cats[k] = 3
        elif _ov(a, b, *q):
            cats[k] = 1
    return cats, spans


def output_categories(pieces: List[str], gold: str, claim: Optional[str]) -> np.ndarray:
    O = "".join(pieces)
    off = offsets(pieces)
    sp = [(a, b, k) for a, b, k in time_spans(O, gold, claim) if k != "other"]
    cats = np.full(len(pieces), 7, np.int8)
    for k in range(len(pieces)):
        a, b = int(off[k]), int(off[k + 1])
        hit = [kind for s0, s1, kind in sp if _ov(a, b, s0, s1)]
        if "claim" in hit:
            cats[k] = 5
        elif "gold" in hit:
            cats[k] = 6
    return cats


# ---------------------------------------------------------------- 句子与读图句（纯函数）
_HAND = re.compile(r"\b(hour|minute|second|short|long|thin|thick|red|black|white|gold|silver)[- ]hands?\b"
                   r"|\bhands?\b|\bpointers?\b|\bneedles?\b", re.I)
_POS = re.compile(r"\b(points?|pointing|pointed|at|near|past|between|toward|towards|just|slightly|exactly|on|before|"
                  r"after|reaching|close to)\b|\d|\b(?:II|III|IV|VI{0,3}|IX|XI{0,2}|X)\b|o'clock", re.I)
_USER = re.compile(r"\b(user|you|your|believes?|belief|claims?|claimed|said|says|stated|states|suggests?|suggested|"
                   r"proposed?|propose|assert\w*|insist\w*|expected answer|given answer|correct answer is|"
                   r"they think|their answer)\b", re.I)
_SEP = re.compile(r"(?<=[.!?])\s+|\n+")


def sentences(text: str) -> List[Tuple[int, int, str]]:
    out, s = [], 0
    for m in _SEP.finditer(text):
        if m.start() > s:
            out.append((s, m.start(), text[s:m.start()]))
        s = m.end()
    if s < len(text):
        out.append((s, len(text), text[s:]))
    return [x for x in out if x[2].strip()]


def classify(sent: str, gold: str, claim: Optional[str]) -> str:
    """read_gold / read_claim / read_mixed / read_other / read_notime：读图句及其中的时间；
    restate：提到用户且写出了用户的时间；other：其余。"""
    kinds = {time_kind(m.group(0), gold, claim) for m in _TIME.finditer(sent)}
    if _USER.search(sent):
        return "restate" if "claim" in kinds else "other"
    if not ((_HAND.search(sent) and _POS.search(sent)) or kinds):
        return "other"
    if "gold" in kinds and "claim" in kinds:
        return "read_mixed"
    if "gold" in kinds:
        return "read_gold"
    if "claim" in kinds:
        return "read_claim"
    if kinds:
        return "read_other"
    return "read_notime"


READ = ["read_gold", "read_claim", "read_mixed", "read_other", "read_notime"]


def sentence_rows(pieces: List[str], n_think: int, gold: str, claim: Optional[str]) -> List[dict]:
    """思考部分的句子：类别 + 覆盖的输出行（第 k 行 = 生成第 k 个输出 token）。"""
    off = offsets(pieces)
    text = "".join(pieces[:n_think])
    out = []
    for s0, s1, s in sentences(text):
        k0 = int(np.searchsorted(off, s0, side="right") - 1)
        k1 = int(np.searchsorted(off, s1, side="left"))
        rows = [k for k in range(max(k0, 0), min(k1, n_think)) if _ov(int(off[k]), int(off[k + 1]), s0, s1)]
        if rows:
            out.append({"kind": classify(s, gold, claim), "rows": rows, "text": s.strip()[:300]})
    return out


# ---------------------------------------------------------------- 记录器（与 symexp.attn 的包装函数配合）
class SourceRecorder:
    """symexp.attn._recording_attention 会调用 stages_for / accumulate，并读 chunk、selfcheck_*。"""

    def __init__(self, torch, cats_all: np.ndarray, n_in: int, n_out: int, image_idx, heads: np.ndarray,
                 chunk: int = 256, selfcheck: int = 2, layers=None):
        self.torch = torch
        self.chunk = chunk
        self.layers = list(LAYERS if layers is None else layers)
        self.selfcheck_left = selfcheck
        self.selfcheck_diffs: List[float] = []
        self.n_in, self.n_out = n_in, n_out
        self.cats = torch.as_tensor(np.asarray(cats_all, np.int64))
        self.image_idx = image_idx
        self.heads = np.asarray(heads, bool)
        self.onehot = {}
        self.mass: Dict[int, object] = {}
        self.filled: Dict[int, np.ndarray] = {}
        self.patch_sum = torch.zeros(n_out, int(image_idx.numel()), dtype=torch.float64)
        self.patch_cnt = np.zeros(n_out, np.int64)
        self.sum_dev = 0.0

    def stages_for(self, q_pos):
        return self.torch.full_like(q_pos, -1)

    def _oh(self, device, kL):
        key = (str(device), kL)
        if key not in self.onehot:
            t = self.torch
            self.onehot = {key: t.nn.functional.one_hot(self.cats[:kL].to(device), len(CATS)).float()}
        return self.onehot[key]

    def accumulate(self, layer_idx: int, probs, stages, q_pos):
        t = self.torch
        if layer_idx not in LAYERS:
            raise SourcesCheckFailed(f"第 {layer_idx} 层不在标准注意力层列表里")
        kL = probs.shape[-1]
        k = q_pos - (self.n_in - 1)
        sel = (k >= 0) & (k < self.n_out)
        if not bool(sel.any()):
            return
        sel_d = sel.to(probs.device)
        p = probs[0][:, sel_d, :].float()                                   # [H, c, kL]
        m = t.matmul(p, self._oh(p.device, kL))                             # [H, c, C]
        self.sum_dev = max(self.sum_dev, float((m.sum(-1) - 1.0).abs().max()))
        rows = k[sel].cpu().long()
        H = p.shape[0]
        if layer_idx not in self.mass:
            self.mass[layer_idx] = t.zeros(self.n_out, H, len(CATS), dtype=t.float32)
            self.filled[layer_idx] = np.zeros(self.n_out, np.int64)
        self.mass[layer_idx][rows] = m.transpose(0, 1).cpu()
        self.filled[layer_idx][rows.numpy()] += 1
        hs = np.flatnonzero(self.heads[LAYERS.index(layer_idx)])
        if len(hs):
            img = p[t.as_tensor(hs, device=p.device)][:, :, self.image_idx.to(p.device)]   # [h, c, n_img]
            self.patch_sum[rows] += img.sum(0).double().cpu()
            self.patch_cnt[rows.numpy()] += len(hs)

    def result(self) -> Dict[str, np.ndarray]:
        layers = sorted(self.mass)
        if layers != sorted(self.layers):
            raise SourcesCheckFailed(f"记录到的层 {layers} 与要记录的层 {sorted(self.layers)} 不一致")
        for l in layers:
            if not (self.filled[l] == 1).all():
                raise SourcesCheckFailed(f"第 {l} 层有输出行没记录到或记录了多次")
        if self.sum_dev > SUM_TOL:
            raise SourcesCheckFailed(f"8 类注意力之和偏离 1 达 {self.sum_dev:.4f}")
        if not (self.patch_cnt == int(self.heads.sum())).all():
            raise SourcesCheckFailed("看图头的图像块注意力累计次数不对")
        mass = np.stack([self.mass[l].numpy() for l in layers], 1)          # [T, L, H, C]
        patch = (self.patch_sum.numpy() / max(int(self.heads.sum()), 1))
        return {"mass": mass.astype(np.float16), "patch": patch.astype(np.float16)}


# ---------------------------------------------------------------- GPU：回放
def _neutral_heads(cq: str, root: str, item_id: str) -> np.ndarray:
    from .compare import _reference
    return np.asarray(_reference(cq, root, item_id)["heads"], bool)


def replay_one(backend, cfg, item: dict, cond: str, rec: dict, heads: np.ndarray) -> Tuple[dict, dict]:
    from .. import attn
    from .conflict import CHUNK, RANK_K, RANK_MIN, claim_of
    from .entropy import _batch
    from .generate import build_inputs
    torch, model, tok = backend.torch, backend.model, backend.tokenizer
    attn.install(model, verbose=False)
    _, inp = build_inputs(backend, item["image"], item["messages"][cond])
    in_ids = inp["input_ids"][0].tolist()
    if rec.get("input_ids") is not None and in_ids != rec["input_ids"]:
        raise SourcesCheckFailed(f"{item['id']}__{cond}：重新构造的提示与记录里的 input_ids 不一致")
    out_ids = [int(x) for x in rec["output_ids"]]
    n_in, n_out = len(in_ids), len(out_ids)
    img_tok = attn.image_token_id(backend)
    image_idx = torch.tensor([i for i, x in enumerate(in_ids) if x == img_tok])
    grid = inp["image_grid_thw"][0].tolist()
    ms = attn.merge_size(backend)
    if int(image_idx.numel()) != grid[0] * grid[1] * grid[2] // (ms * ms):
        raise SourcesCheckFailed("图像 token 数与 grid 推算的不一致")
    dec = lambda x: tok.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    p_pieces, p_ok = token_pieces(dec, in_ids)
    o_pieces, o_ok = token_pieces(dec, out_ids)
    claim = claim_of(item, cond)
    cats_p, spans = prompt_categories(p_pieces, in_ids, img_tok, item["messages"][cond], cond, claim)
    cats_o = output_categories(o_pieces, item["gold"], claim if cond != "neutral" else item["claim"])
    close = out_ids.index(backend.close_id) if backend.close_id in out_ids else None
    sr = SourceRecorder(torch, np.concatenate([cats_p, cats_o]), n_in, n_out, image_idx, heads,
                        chunk=cfg.attn_chunk, selfcheck=cfg.attn_selfcheck)
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "rope_deltas"):
        inner.rope_deltas = None
    t0 = time.time()
    with torch.inference_mode(), attn.recording(sr):
        o = model(**_batch(backend, inp, out_ids), use_cache=False, return_dict=True)
    lg = o.logits[0, n_in - 1: n_in - 1 + n_out]
    y = torch.tensor(out_ids, device=lg.device)
    rk = []
    for s in range(0, n_out, CHUNK):
        a = lg[s:s + CHUNK].float()
        rk.append((a > a.gather(1, y[s:s + CHUNK, None])).sum(-1))
    top = float((torch.cat(rk) < RANK_K).float().mean())
    del o, lg
    if top < RANK_MIN:
        raise SourcesCheckFailed(f"{item['id']}__{cond}：回放时实际 token 进前 {RANK_K} 名的比例只有 {top:.3f}")
    arrays = sr.result()
    arrays.update(heads=heads, cats_prompt=cats_p, cats_out=cats_o)
    meta = {"id": item["id"], "condition": cond, "n_in": n_in, "n_out": n_out, "close": close,
            "gold": item["gold"], "claim": claim, "gen_claim_time": claim if cond != "neutral" else item["claim"],
            "n_img": int(image_idx.numel()), "pieces_out": o_pieces, "pieces_ok": {"prompt": p_ok, "output": o_ok},
            "prompt_spans": spans, "prompt_counts": {CATS[c]: int((cats_p == c).sum()) for c in range(5)},
            "rank_top20": round(top, 4), "selfcheck_max_abs_diff": sr.selfcheck_diffs,
            "mass_sum_max_dev": sr.sum_dev, "seconds": round(time.time() - t0, 2)}
    return meta, arrays


def run_replay(cfg, cq: str, root: str, conds: Optional[List[str]] = None, shard: str = "") -> int:
    from .conflict import _engine, _items
    from .pipeline import _atomic_json, _atomic_npz, _check_no_failure, _fail, _shard
    conds = conds or CONDS
    _check_no_failure(root)
    items = _items(cq)
    d = os.path.join(root, "replay")
    os.makedirs(d, exist_ok=True)
    tasks = [(i, c) for i in sorted(items) for c in conds]
    todo = [t for t in _shard(tasks, shard) if not os.path.exists(os.path.join(d, f"{t[0]}__{t[1]}.json"))]
    print(f"[fp-sources replay] 共 {len(tasks)} 条，本进程待做 {len(todo)} 条 → {d}", file=sys.stderr)
    if not todo:
        return 0
    eng = _engine(cfg)
    for n, (iid, cond) in enumerate(todo):
        task = f"sources_{iid}__{cond}"
        try:
            rp = os.path.join(cq, "obs", "records", f"{iid}__{cond}.json")
            if not os.path.exists(rp):
                raise SourcesCheckFailed(f"缺少原模型记录 {rp}")
            rec = json.load(open(rp, encoding="utf-8"))
            heads = _neutral_heads(cq, root, iid)
            meta, arrays = replay_one(eng.backend, cfg, items[iid], cond, rec, heads)
            npz = os.path.join(d, f"{iid}__{cond}.npz")
            _atomic_npz(npz, arrays)
            meta["npz"] = npz
            _atomic_json(os.path.join(d, f"{iid}__{cond}.json"), meta)
            print(f"  {n + 1}/{len(todo)} {iid}__{cond} 输出 {meta['n_out']} token，{meta['seconds']}s，"
                  f"前20名 {meta['rank_top20']}，和偏差 {meta['mass_sum_max_dev']:.1e}，"
                  f"片段一致 {meta['pieces_ok']}", file=sys.stderr)
        except Exception as e:
            _fail(root, task, e)
            raise
    return len(todo)


# ---------------------------------------------------------------- CPU：分析
OUTCOME = {"坚持（答对）": "对", "采纳": "采", "其他": "他", "未解析": "未"}
SEGS = ["think", "first512"] + READ + ["read_all", "restate"]


def _group_means(mass: np.ndarray, heads: np.ndarray) -> Dict[str, np.ndarray]:
    m = mass.astype(np.float32)
    T = m.shape[0]
    return {"看图头": m[:, heads].mean(1), "全部头": m.reshape(T, -1, len(CATS)).mean(1)}


def row_metrics(G: np.ndarray) -> Dict[str, np.ndarray]:
    """G：[T, 8]（一组头的平均）。返回每行的各类注意力和三个对数比值。"""
    img, q = G[:, 0], G[:, 1]
    claim = G[:, 2] + G[:, 3]
    gen_c = G[:, 5]
    return {"img": img, "question": q, "claim": claim, "claim_time": G[:, 2], "gen_claim": gen_c,
            "gen_gold": G[:, 6], "template": G[:, 4], "gen_other": G[:, 7],
            "lr_claim": np.log((img + EPS) / (claim + EPS)),
            "lr_claim_all": np.log((img + EPS) / (claim + gen_c + EPS)),
            "lr_user": np.log((img + EPS) / (q + claim + EPS))}


def record_summary(meta: dict, mass: np.ndarray, heads: np.ndarray) -> dict:
    n_think = meta["close"] if meta["close"] is not None else meta["n_out"]
    sents = sentence_rows(meta["pieces_out"], n_think, meta["gold"], meta["claim"])
    seg_rows = defaultdict(list)
    seg_rows["think"] = list(range(n_think))
    seg_rows["first512"] = list(range(min(n_think, WINDOW)))
    for s in sents:
        seg_rows[s["kind"]].extend(s["rows"])
        if s["kind"] in READ:
            seg_rows["read_all"].extend(s["rows"])
    first_read = min((s["rows"][0] for s in sents if s["kind"] in READ), default=None)
    out = {"n_think": n_think, "first_read_row": first_read, "n_sent": {k: sum(s["kind"] == k for s in sents)
                                                                       for k in READ + ["restate", "other"]},
           "segs": {}, "bins": {}, "aligned": {}}
    for g, G in _group_means(mass, heads).items():
        M = row_metrics(G[:meta["n_out"]])
        out["segs"][g] = {seg: ({k: float(np.mean(v[sorted(set(rows))])) for k, v in M.items()} |
                               {"rows": len(set(rows))} if rows else None) for seg, rows in seg_rows.items()}
        out["bins"][g] = {k: [float(np.mean(M[k][i:min(i + BIN, n_think)])) for i in range(0, min(n_think, WINDOW), BIN)]
                          for k in ("img", "claim", "gen_claim", "lr_claim", "lr_claim_all")}
        if first_read is not None:
            al = {}
            for k in ("img", "claim", "lr_claim", "lr_claim_all"):
                vals = []
                for b in range(-64, 192, BIN):
                    a0, a1 = first_read + b, first_read + b + BIN
                    a0, a1 = max(a0, 0), min(a1, n_think)
                    vals.append(float(np.mean(M[k][a0:a1])) if a1 > a0 else None)
                al[k] = vals
            out["aligned"][g] = al
    out["examples"] = [{"kind": s["kind"], "text": s["text"]} for s in sents if s["kind"] in READ + ["restate"]][:40]
    return out


def auc(pos: List[float], neg: List[float]) -> float:
    """P(pos > neg)，平局算一半。"""
    if not pos or not neg:
        return float("nan")
    a = np.asarray(pos)[:, None]
    b = np.asarray(neg)[None, :]
    return float(((a > b).sum() + 0.5 * (a == b).sum()) / (a.size * b.size))


def perm_p(x: List[float], y: List[float], reps: int = 20000, seed: int = 0) -> float:
    """两组均值差的置换检验（双侧）。"""
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    z = np.asarray(list(x) + list(y))
    obs = abs(np.mean(x) - np.mean(y))
    rng = np.random.default_rng(seed)
    n = len(x)
    hit = 0
    for _ in range(reps):
        rng.shuffle(z)
        hit += abs(z[:n].mean() - z[n:].mean()) >= obs - 1e-12
    return float((hit + 1) / (reps + 1))


def _fmt(x, d=3):
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}f}"


def _val(s, g, seg, k):
    v = s["segs"][g].get(seg)
    return None if v is None else v[k]


def run_analyze(cq: str, root: str, judgments: str, out_md: Optional[str] = None,
                out_json: Optional[str] = None) -> dict:
    from .conflict import _items
    from .pipeline import _atomic_json
    from .stats import cluster_bootstrap_rate, cluster_signflip
    items = _items(cq)
    J = json.load(open(judgments, encoding="utf-8"))
    d = os.path.join(root, "replay")
    S, missing = {}, []
    for iid in sorted(items):
        for c in CONDS:
            jp = os.path.join(d, f"{iid}__{c}.json")
            if not os.path.exists(jp):
                missing.append(f"{iid}__{c}")
                continue
            meta = json.load(open(jp, encoding="utf-8"))
            with np.load(os.path.join(d, f"{iid}__{c}.npz")) as z:
                s = record_summary(meta, z["mass"], z["heads"])
            j = J.get(f"obs|{iid}|{c}")
            s["outcome"] = OUTCOME.get(j["outcome"], "未") if j else "未"
            s["pieces_ok"] = meta["pieces_ok"]
            S[(iid, c)] = s
    neu_ok = {i for i in items if (i, "neutral") in S and S[(i, "neutral")]["outcome"] == "对"}
    L = ["# 读时间时：注意力看图像还是看用户的话", "",
         f"原模型回放：{len(S)} 条（缺 {len(missing)} 条）；中性答对 {len(neu_ok)} 题。结果来自时间判读（严格规则）。",
         "比值都取对数：`ln(图像注意力 / 用户断言注意力)`，越小 = 相对图像越偏向用户的话。"
         "图像有 169 个 token，断言那一行约 10 个，所以比值本身大于 0 不说明什么，只比较组间差别。", "",
         "- `lr_claim`：图像 / 提示里断言那一行", "- `lr_claim_all`：图像 / （断言那一行 + 模型自己复述的用户时间）",
         "- `lr_user`：图像 / 用户整条消息（题干 + 断言）", "",
         "读图句：思考里描述指针位置、或写出时间结论、且不提用户的句子（规则见代码 `classify`）。"
         "读图句按句子里的时间分为：正确时间 read_gold、用户时间 read_claim、其他时间 read_other、没写时间 read_notime。", ""]
    bad = [f"{k[0]}__{k[1]}" for k, s in S.items() if not all(s["pieces_ok"].values())]
    if bad:
        L += [f"注意：{len(bad)} 条记录的 token 文字片段与整段解码不完全一致（句子定位可能略有偏差）：{bad[:10]}", ""]
    res = {"missing": missing, "neutral_ok": sorted(neu_ok), "tables": {}}

    # 1. 各条件总体
    L += ["## 1. 各条件、各段的平均注意力（看图头 / 全部头）", "",
          "| 条件 | 头 | 段 | 记录数 | 图像 | 题干 | 断言 | 复述用户时间 | lr_claim | lr_claim_all |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    t1 = []
    for c in CONDS:
        for g in ("看图头", "全部头"):
            for seg in ("think", "first512", "read_all", "read_gold", "read_claim", "restate"):
                vs = [s for (i, cc), s in S.items() if cc == c and s["segs"][g].get(seg)]
                if not vs:
                    continue
                f = lambda k: float(np.mean([s["segs"][g][seg][k] for s in vs]))
                row = {"cond": c, "group": g, "seg": seg, "n": len(vs), **{k: f(k) for k in
                       ("img", "question", "claim", "gen_claim", "lr_claim", "lr_claim_all")}}
                t1.append(row)
                L.append(f"| {c} | {g} | {seg} | {len(vs)} | {row['img']:.4f} | {row['question']:.4f} | {row['claim']:.4f} | "
                         f"{row['gen_claim']:.4f} | {_fmt(row['lr_claim'] if c != 'neutral' else None)} | "
                         f"{_fmt(row['lr_claim_all'] if c != 'neutral' else None)} |")
    res["tables"]["by_condition"] = t1

    # 2. 用户说错：坚持 vs 采纳
    L += ["", "## 2. 用户说错、中性答对的题：坚持 vs 采纳（读图句上）", "",
          "AUC = 坚持组比值高于采纳组的概率（> 0.5 表示采纳组读图时更偏向用户的话）；p 为置换检验。", "",
          "| 头 | 段 | 指标 | 坚持 n | 坚持均值 | 采纳 n | 采纳均值 | 其他错 均值 | AUC | p |", "|---|---|---|---|---|---|---|---|---|---|"]
    t2 = []
    for g in ("看图头", "全部头"):
        for seg in ("read_all", "first512", "think"):
            for k in ("lr_claim", "lr_claim_all", "lr_user", "img", "claim"):
                grp = {o: [_val(S[(i, "belief")], g, seg, k) for i in neu_ok if (i, "belief") in S
                           and S[(i, "belief")]["outcome"] == o] for o in ("对", "采", "他")}
                grp = {o: [x for x in v if x is not None] for o, v in grp.items()}
                a, p = auc(grp["对"], grp["采"]), perm_p(grp["对"], grp["采"])
                if k == "claim":
                    a = 1 - a if math.isfinite(a) else a
                row = {"group": g, "seg": seg, "metric": k, "n_hold": len(grp["对"]), "n_adopt": len(grp["采"]),
                       "hold": float(np.mean(grp["对"])) if grp["对"] else None,
                       "adopt": float(np.mean(grp["采"])) if grp["采"] else None,
                       "other": float(np.mean(grp["他"])) if grp["他"] else None, "auc": a, "p": p}
                t2.append(row)
                L.append(f"| {g} | {seg} | {k} | {row['n_hold']} | {_fmt(row['hold'], 4)} | {row['n_adopt']} | "
                         f"{_fmt(row['adopt'], 4)} | {_fmt(row['other'], 4)} | {_fmt(a)} | {_fmt(p, 4)} |")
    L += ["", "（`claim` 一行的 AUC 方向相反：采纳组断言注意力高于坚持组的概率。）"]
    res["tables"]["hold_vs_adopt"] = t2

    # 3. 配对：用户说错 vs 用户说对；用户说错 vs 中性（图像注意力）
    L += ["", "## 3. 同一题配对比较（读图句上，看图头）", "",
          "| 比较 | 指标 | 范围 | 题数 | 平均差 | 95% 区间 | p（符号翻转） |", "|---|---|---|---|---|---|---|"]
    t3 = []
    for name, c1, c2, metrics in (("说错 − 说对", "belief", "belief_right", ("lr_claim", "lr_claim_all", "img", "claim")),
                                  ("说错 − 中性", "belief", "neutral", ("img",)),
                                  ("说对 − 中性", "belief_right", "neutral", ("img",))):
        for k in metrics:
            for scope, ids in (("全部", list(items)), ("中性答对", sorted(neu_ok))):
                diffs = []
                for i in ids:
                    a = S.get((i, c1))
                    b = S.get((i, c2))
                    va = _val(a, "看图头", "read_all", k) if a else None
                    vb = _val(b, "看图头", "read_all", k) if b else None
                    if va is not None and vb is not None:
                        diffs.append((i, va - vb))
                if not diffs:
                    continue
                st = cluster_signflip(diffs)
                lo, hi = cluster_bootstrap_rate(diffs)
                row = {"compare": name, "metric": k, "scope": scope, "n": len(diffs),
                       "mean_diff": float(np.mean([x for _, x in diffs])), "ci": [lo, hi], "p": st["p"]}
                t3.append(row)
                L.append(f"| {name} | {k} | {scope} | {len(diffs)} | {row['mean_diff']:+.4f} | [{_fmt(lo, 4)}, {_fmt(hi, 4)}] | "
                         f"{_fmt(st['p'], 4)} |")
    res["tables"]["paired"] = t3

    # 4. 同一条回答里：写用户时间的读图句 vs 写正确时间的读图句
    L += ["", "## 4. 用户说错时，同一条回答里：写出用户时间的读图句 vs 写出正确时间的读图句（看图头）", "",
          "只用两种句子都有的回答；差 = read_claim − read_gold。", "",
          "| 指标 | 回答数 | 平均差 | 95% 区间 | p（符号翻转） |", "|---|---|---|---|---|"]
    t4 = []
    for k in ("lr_claim", "lr_claim_all", "img", "claim", "gen_claim"):
        diffs = []
        for (i, c), s in S.items():
            if c != "belief":
                continue
            a, b = _val(s, "看图头", "read_claim", k), _val(s, "看图头", "read_gold", k)
            if a is not None and b is not None:
                diffs.append((i, a - b))
        if not diffs:
            L.append(f"| {k} | 0 | — | — | — |")
            continue
        st = cluster_signflip(diffs)
        lo, hi = cluster_bootstrap_rate(diffs)
        row = {"metric": k, "n": len(diffs), "mean_diff": float(np.mean([x for _, x in diffs])), "ci": [lo, hi], "p": st["p"]}
        t4.append(row)
        L.append(f"| {k} | {len(diffs)} | {row['mean_diff']:+.4f} | [{_fmt(lo, 4)}, {_fmt(hi, 4)}] | {_fmt(st['p'], 4)} |")
    res["tables"]["claim_vs_gold_sentences"] = t4

    # 5. 时间曲线
    def curve(sel, key, kind):
        vs = [s[kind]["看图头"][key] for s in sel if "看图头" in s[kind] and key in s[kind]["看图头"]]
        if not vs:
            return []
        n = max(len(v) for v in vs)
        out = []
        for b in range(n):
            col = [v[b] for v in vs if b < len(v) and v[b] is not None]
            out.append((float(np.mean(col)) if col else None, len(col)))
        return out
    groups = {"说错·坚持": [S[(i, "belief")] for i in neu_ok if (i, "belief") in S and S[(i, "belief")]["outcome"] == "对"],
              "说错·采纳": [S[(i, "belief")] for i in neu_ok if (i, "belief") in S and S[(i, "belief")]["outcome"] == "采"],
              "说对": [s for (i, c), s in S.items() if c == "belief_right"],
              "中性": [s for (i, c), s in S.items() if c == "neutral"]}
    res["tables"]["curves"] = {}
    for kind, title, labels in (("bins", "## 5. 思考前 512 个 token 的时间曲线（每 32 个 token 一格，看图头）",
                                 [f"{b}–{b + BIN}" for b in range(0, WINDOW, BIN)]),
                                ("aligned", "## 6. 以第一句读图句为 0 点（每 32 个 token 一格，看图头）",
                                 [f"{b}–{b + BIN}" for b in range(-64, 192, BIN)])):
        L += ["", title, ""]
        for key in ("lr_claim_all", "img", "claim"):
            L += [f"**{key}**", "", "| 组 | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
            for gname, sel in groups.items():
                if gname == "中性" and key != "img":
                    continue
                cv = curve(sel, key, kind)
                res["tables"]["curves"][f"{kind}|{key}|{gname}"] = cv
                L.append(f"| {gname}（{len(sel)}） | " + " | ".join(_fmt(v, 3 if key.startswith('lr') else 4) for v, _ in cv)
                         + " |" if cv else f"| {gname} | 无数据 |")
            L.append("")

    # 7. 读图句统计与例子
    L += ["## 7. 句子分类统计（供核对规则）", "", "| 条件 | " + " | ".join(READ + ["restate", "other"]) + " |",
          "|---|" + "---|" * (len(READ) + 2)]
    for c in CONDS:
        cnt = defaultdict(int)
        for (i, cc), s in S.items():
            if cc == c:
                for k, v in s["n_sent"].items():
                    cnt[k] += v
        L.append(f"| {c} | " + " | ".join(str(cnt[k]) for k in READ + ["restate", "other"]) + " |")
    ex = {f"{i}__{c}": s["examples"] for (i, c), s in S.items() if c == "belief"}
    res["examples_belief"] = ex
    res["records"] = {f"{i}__{c}": {k: v for k, v in s.items() if k != "examples"} for (i, c), s in S.items()}
    out_md = out_md or os.path.join(root, "sources_report.md")
    out_json = out_json or os.path.join(root, "sources_summary.json")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    _atomic_json(out_json, res)
    print(f"[fp-sources analyze] {len(S)} 条记录 → {out_md}、{out_json}；缺 {len(missing)} 条", file=sys.stderr)
    return res


# ---------------------------------------------------------------- CPU：同一条回答内部，逐 token 比较
"""tokens 阶段：在读图句里找出“承载读数内容”的 token，按内容分类，在同一条回答内部比较。

  指针数字：句子里点明了某根针（hour/short → 时针，minute/long → 分针，second/red → 秒针）之后出现的数字
            （1–12、罗马数字，或 > 12 的分钟 / 秒数），按这根针在正确时间、用户时间下应指的位置判断：
              induced  只与用户时间相符（被诱导）    correct  只与正确时间相符
              wrong    两者都不符（其他读错）        both     两者都符（不用）
            只用句子里明说了是哪根针的数字（不沿用上一句）。
  时间数字：读图句里写出的时间串，与用户时间一致 → induced，与正确时间一致 → correct。
            写时间时模型会大量看自己前面写过的同一个时间（抄写），所以另列，并给“首次出现”的版本。
  其余：     读图句里的其他 token（filler）。
  排除：     提到用户的句子、假设句（if / should / would / expected …，例如“如果是 6:40，时针应该……”）、
            计算句（= * ° degree angle 分数 “of the way” “at 20 minutes,” 等，例如“时针在 12 到 1 的 1/3 处”）。
            小数（如 2.67）里的数字、否定后面的数字（“not III”“rather than 4”）不算。
中性回答没有用户时间，induced 用该题“用户说错”条件的时间来判，用作“没有压力时碰巧读成那个位置”的对照。
"""

_HAND_T = re.compile(r"\b(hour|short|minute|long|second|red)\b(?:\W+\w+){0,3}?\W+(?:hand|pointer|needle)s?\b", re.I)
_HAND_G = re.compile(r"\b(hand|pointer|needle|tip)s?\b", re.I)
_HYPO = re.compile(r"\b(if|unless|should|would|could|expected|supposed|must be|suppose|assume|assuming|hypothetically)\b"
                   r"|\bfor \d{1,2}:\d{2}", re.I)
_NEG = re.compile(r"\b(not|nor|rather than|instead of|isn't|aren't|wasn't|doesn't|no longer)\b[^,;.()]{0,25}$", re.I)
_NUMTOK = re.compile(r"\b(XII|XI|IX|X|VIII|VII|VI|IV|V|III|II)\b|(?<![\d.])(\d{1,2})(?![\d]|\.\d)")
_CALC = re.compile(r"[=*×°÷]|\bdegrees?\b|\bangles?\b|\d\s*/\s*\d|\bof the way\b|\balternative\w*\b|"
                   r"\b(?:at|for) \d{1,2}(?::\d{2})?\s*(?:min|minutes|hours?|seconds?)?\s*,", re.I)
_ROMAN = {"XII": 12, "XI": 11, "X": 10, "IX": 9, "VIII": 8, "VII": 7, "VI": 6, "V": 5, "IV": 4, "III": 3, "II": 2}
_HAND_OF = {"hour": "hour", "short": "hour", "minute": "minute", "long": "minute", "second": "second", "red": "second"}
TOKEN_CLASSES = ["hand_induced", "hand_correct", "hand_wrong", "hand_induced_obs", "hand_correct_obs", "hand_wrong_obs",
                 "time_induced", "time_correct", "time_induced_first", "time_correct_first", "filler"]
# 严格版（*_obs）：只用“观察句”——有 points / see / looks / appears / in the image 之类的观察词，
# 且没有“at 37 minutes”这种把时间换算成指针位置的说法（那是在推算用户时间意味着什么，不是在读图）。
_OBS = re.compile(r"\b(points?|pointing|pointed|see|seen|looks?|looking|appears?|shows?|in the image|there's|there is)\b", re.I)
_DERIV = re.compile(r"\b(?:at|is|=|of)\s*~?\d{1,2}\s*(?:min|mins|minutes?|sec|secs|seconds?)\b", re.I)


def _num12(x: int) -> int:
    return x % 12 or 12


def hand_match(hand: str, v: int, t: Optional[str]) -> bool:
    """数字 v（指针 hand 所指）与时间 t 下这根针的位置是否相符。"""
    if not t:
        return False
    from .conflict import _parse
    h, m, s = _parse(t)
    if hand == "hour":
        return v <= 12 and v in {_num12(h), _num12(h + 1)}
    x = m if hand == "minute" else s
    if x is None:
        return False
    if v <= 12:
        return v in {_num12(x // 5), _num12((x + 4) // 5)}
    return min((v - x) % 60, (x - v) % 60) <= 2


def content_tokens(pieces: List[str], n_think: int, gold: str, claim: Optional[str], return_details: bool = False):
    """返回 {类别: 行号列表} 和用于核对的例子。claim 为 None 时没有 induced。"""
    off = offsets(pieces)
    text = "".join(pieces[:n_think])

    def rows_of(a, b):
        k0 = max(int(np.searchsorted(off, a, side="right") - 1), 0)
        return [k for k in range(k0, min(n_think, len(pieces))) if int(off[k]) < b and a < int(off[k + 1])]

    out = {c: [] for c in TOKEN_CLASSES}
    examples = []
    details = []
    seen_first = set()
    for s0, s1, s in sentences(text):
        kind = classify(s, gold, claim)
        hyp = bool(_HYPO.search(s) or _CALC.search(s))
        if kind not in READ or hyp:
            continue
        content = set()
        tspans = [(m.start(), m.end(), time_kind(m.group(0), gold, claim)) for m in _TIME.finditer(s)]
        for a, b, k in tspans:
            if k in ("claim", "gold"):
                cls = "time_induced" if k == "claim" else "time_correct"
                rr = rows_of(s0 + a, s0 + b)
                out[cls].extend(rr)
                if cls not in seen_first and rr:
                    out[cls + "_first"].extend(rr)
                    seen_first.add(cls)
            content.update(rows_of(s0 + a, s0 + b))
        hands = [(m.start(), _HAND_OF[m.group(1).lower()]) for m in _HAND_T.finditer(s)]
        obs = bool(_OBS.search(s)) and not _DERIV.search(s)
        for m in _NUMTOK.finditer(s):
            if any(a <= m.start() < b for a, b, _ in tspans):
                continue
            prev = [h for p, h in hands if p < m.start()]
            if not prev or _NEG.search(s[max(0, m.start() - 40):m.start()]):
                continue
            hand = prev[-1]
            g = m.group(1) or m.group(2)
            v = _ROMAN[g] if m.group(1) else int(g)
            if v is None or v == 0 or v > 59:
                continue
            ig, ic = hand_match(hand, v, gold), hand_match(hand, v, claim)
            cls = ("hand_induced" if ic and not ig else "hand_correct" if ig and not ic else
                   "hand_wrong" if not ig and not ic else None)
            rr = rows_of(s0 + m.start(), s0 + m.end())
            content.update(rr)
            if cls:
                out[cls].extend(rr)
                if obs:
                    out[cls + "_obs"].extend(rr)
                for r in rr:
                    details.append({"row": int(r), "cls": cls.replace("hand_", ""), "hand": hand, "value": int(v), "obs": bool(obs)})
                if len(examples) < 60 and cls != "hand_correct":
                    examples.append({"class": cls, "hand": hand, "value": v, "sentence": s.strip()[:220]})
        out["filler"].extend(k for k in rows_of(s0, s1) if k not in content)
    if return_details:
        return {c: sorted(set(v)) for c, v in out.items()}, examples, details
    return {c: sorted(set(v)) for c, v in out.items()}, examples


def run_tokens(cq: str, root: str, judgments: str, out_md: Optional[str] = None,
               out_json: Optional[str] = None) -> dict:
    from .conflict import _items
    from .pipeline import _atomic_json
    from .stats import cluster_bootstrap_rate, cluster_signflip
    items = _items(cq)
    J = json.load(open(judgments, encoding="utf-8"))
    d = os.path.join(root, "replay")
    METRICS = ["lr_claim_all", "lr_claim", "img", "claim", "gen_claim", "question"]
    R, examples, missing = {}, {}, []
    for iid in sorted(items):
        for c in CONDS:
            jp = os.path.join(d, f"{iid}__{c}.json")
            if not os.path.exists(jp):
                missing.append(f"{iid}__{c}")
                continue
            meta = json.load(open(jp, encoding="utf-8"))
            n_think = meta["close"] if meta["close"] is not None else meta["n_out"]
            claim = meta["claim"] if c != "neutral" else items[iid]["claim"]
            cls, ex = content_tokens(meta["pieces_out"], n_think, meta["gold"], claim if c != "belief_right" else None)
            with np.load(os.path.join(d, f"{iid}__{c}.npz")) as z:
                G = _group_means(z["mass"], z["heads"])
            rec = {"n": {k: len(v) for k, v in cls.items()}, "m": {}}
            for g, Gm in G.items():
                M = row_metrics(Gm[:meta["n_out"]])
                rec["m"][g] = {k: ({m: float(np.mean(M[m][v])) for m in METRICS} if v else None) for k, v in cls.items()}
            j = J.get(f"obs|{iid}|{c}")
            rec["outcome"] = OUTCOME.get(j["outcome"], "未") if j else "未"
            R[(iid, c)] = rec
            if c == "belief":
                examples[iid] = ex[:15]
    neu_ok = {i for i in items if (i, "neutral") in R and R[(i, "neutral")]["outcome"] == "对"}

    def paired(cond, a, b, g, metric, sel=None):
        diffs = []
        for (i, c), r in R.items():
            if c != cond or (sel and not sel(i, r)):
                continue
            x, y = r["m"][g].get(a), r["m"][g].get(b)
            if x is not None and y is not None:
                diffs.append((i, x[metric] - y[metric]))
        if not diffs:
            return None
        st = cluster_signflip(diffs)
        lo, hi = cluster_bootstrap_rate(diffs)
        v = [x for _, x in diffs]
        return {"n": len(diffs), "mean": float(np.mean(v)), "ci": [lo, hi], "p": st["p"],
                "neg_share": float(np.mean([x < 0 for x in v]))}

    L = ["# 同一条回答内部：被诱导的读数 token 与正确的读数 token", "",
         f"原模型回放 {len(R)} 条（缺 {len(missing)}）。只看读图句，排除提到用户的句子和假设句（if / should / would …）。",
         "每个 token 用“生成它的那一步”的注意力。比值取对数，越小 = 相对图像越偏向用户信息：", "",
         "- `lr_claim_all` = ln(图像 / (提示里的断言 + 模型复述的用户时间))　← 主指标（“用户信息”）",
         "- `lr_claim` = ln(图像 / 提示里的断言)", "- `img`、`claim`、`gen_claim`、`question`：各自的注意力量", "",
         "指针数字：点明某根针后出现的数字，与用户时间下该针位置相符而与正确时间不符 = 被诱导（hand_induced），"
         "反之 = 正确（hand_correct），都不符 = 其他读错（hand_wrong）。时间数字另列（会抄写前面写过的同一个时间）。", ""]
    counts = defaultdict(lambda: defaultdict(int))
    for (i, c), r in R.items():
        for k, v in r["n"].items():
            counts[c][k] += v
    L += ["## 0. token 数（全部回答合计）", "", "| 条件 | " + " | ".join(TOKEN_CLASSES) + " |", "|---|" + "---|" * len(TOKEN_CLASSES)]
    for c in CONDS:
        L.append(f"| {c} | " + " | ".join(str(counts[c][k]) for k in TOKEN_CLASSES) + " |")
    res = {"missing": missing, "counts": {c: dict(v) for c, v in counts.items()}, "tables": {}}
    blocks = [
        ("1. 用户说错：被诱导的指针数字 − 正确的指针数字（同一条回答，两种都有的回答）", "belief", "hand_induced", "hand_correct", None),
        ("1a. 同上，只看最终采纳的回答", "belief", "hand_induced", "hand_correct", lambda i, r: r["outcome"] == "采"),
        ("1b. 同上，只看最终坚持的回答", "belief", "hand_induced", "hand_correct", lambda i, r: r["outcome"] == "对"),
        ("1s. 严格版（只用观察句）：被诱导的指针数字 − 正确的指针数字", "belief", "hand_induced_obs", "hand_correct_obs", None),
        ("1s-a. 严格版，只看最终采纳的回答", "belief", "hand_induced_obs", "hand_correct_obs", lambda i, r: r["outcome"] == "采"),
        ("1s-b. 严格版，只看最终坚持的回答", "belief", "hand_induced_obs", "hand_correct_obs", lambda i, r: r["outcome"] == "对"),
        ("2. 用户说错：被诱导的指针数字 − 同一读图句里的其他 token", "belief", "hand_induced", "filler", None),
        ("3. 用户说错：正确的指针数字 − 同一读图句里的其他 token", "belief", "hand_correct", "filler", None),
        ("4. 用户说错：其他读错的指针数字 − 正确的指针数字", "belief", "hand_wrong", "hand_correct", None),
        ("5. 中性（没有压力）：其他读错的指针数字 − 正确的指针数字", "neutral", "hand_wrong", "hand_correct", None),
        ("5s. 严格版，中性：其他读错的指针数字 − 正确的指针数字", "neutral", "hand_wrong_obs", "hand_correct_obs", None),
        ("6. 用户说错：写出用户时间的数字 − 写出正确时间的数字", "belief", "time_induced", "time_correct", None),
        ("7. 用户说错：同上，只用各自第一次出现", "belief", "time_induced_first", "time_correct_first", None)]
    for title, cond, a, b, sel in blocks:
        L += ["", f"## {title}", "", "| 头 | 指标 | 回答数 | 平均差 | 95% 区间 | 差 < 0 的比例 | p（符号翻转） |",
              "|---|---|---|---|---|---|---|"]
        tab = []
        for g in ("看图头", "全部头"):
            for metric in METRICS:
                if cond == "neutral" and metric in ("lr_claim_all", "lr_claim", "claim"):
                    continue
                r = paired(cond, a, b, g, metric, sel)
                if r is None:
                    L.append(f"| {g} | {metric} | 0 | — | — | — | — |")
                    continue
                tab.append({"group": g, "metric": metric, **r})
                L.append(f"| {g} | {metric} | {r['n']} | {r['mean']:+.4f} | [{_fmt(r['ci'][0], 4)}, {_fmt(r['ci'][1], 4)}] | "
                         f"{r['neg_share']:.2f} | {_fmt(r['p'], 4)} |")
        res["tables"][title] = tab
    L += ["", "## 8. 各类 token 的平均值（看图头，按回答平均后再平均）", "",
          "| 条件 | 类别 | 回答数 | " + " | ".join(METRICS) + " |", "|---|---|---|" + "---|" * len(METRICS)]
    for c in CONDS:
        for k in TOKEN_CLASSES:
            vs = [r["m"]["看图头"][k] for (i, cc), r in R.items() if cc == c and r["m"]["看图头"].get(k)]
            if vs:
                L.append(f"| {c} | {k} | {len(vs)} | " + " | ".join(f"{np.mean([v[m] for v in vs]):.4f}" for m in METRICS) + " |")
    L += ["", "## 9. 例子（用户说错，被诱导 / 其他读错的指针数字，供核对规则）", ""]
    for iid, ex in list(examples.items()):
        for e in ex[:3]:
            L.append(f"- {iid} [{e['class']}，{e['hand']}={e['value']}] {e['sentence']}")
    res["examples"] = examples
    res["records"] = {f"{i}__{c}": r for (i, c), r in R.items()}
    out_md = out_md or os.path.join(root, "tokens_report.md")
    out_json = out_json or os.path.join(root, "tokens_summary.json")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    _atomic_json(out_json, res)
    print(f"[fp-sources tokens] {len(R)} 条记录 → {out_md}、{out_json}；缺 {len(missing)} 条", file=sys.stderr)
    return res


# ---------------------------------------------------------------- 读提示阶段：题目 / 用户断言看图像
"""prompt / prompt-analyze：只把提示本身前向一遍（不带任何输出），记录用户消息里每个 token 读的时候
对 169 个图像块的注意力（16 层 × 24 头），以及对 5 类提示位置（图像、题干、断言时间、断言其余、模板）的总量。

比较：
  A. 同一条提示里：题目 token 看图 vs 用户断言 token 看图（看图总量、图像内部分布的 JS、落在关键区域 E 的比例）；
     用户断言分三种看：整句话、整句里时间以外的部分、时间本身。
  B. 同一题，用户说错 vs 用户说对：两条提示只有时间数字不同。比较时间数字 token 看图的位置
     （图像、题目完全相同；数字之前的 token 两边完全一样，注意力也必然一样，作为核对）。
     关键区域 E = 该题中性回答思考阶段看图头注意力最高的 20% 图像块（指针所在处）。
     如果数字 token 会按数字的内容去图上“找”，用户说错时它们落在真实指针处（E）的比例应该更低。
  C. 提示最后一个 token（模型开始写字前的那一刻）：同上。
"""
PCATS = CATS[:5]


class PromptRecorder:
    def __init__(self, torch, cats_prompt: np.ndarray, qpos: List[int], image_idx, chunk: int = 256):
        self.torch = torch
        self.chunk = chunk
        self.selfcheck_left = 0
        self.selfcheck_diffs: List[float] = []
        self.cats = torch.as_tensor(np.asarray(cats_prompt, np.int64))
        self.qpos = torch.as_tensor(qpos, dtype=torch.long)
        self.image_idx = image_idx
        self.patch, self.mass = {}, {}

    def stages_for(self, q_pos):
        return self.torch.full_like(q_pos, -1)

    def accumulate(self, layer_idx: int, probs, stages, q_pos):
        t = self.torch
        if layer_idx not in LAYERS:
            raise SourcesCheckFailed(f"第 {layer_idx} 层不在标准注意力层列表里")
        want = self.qpos.to(q_pos.device)
        sel = (q_pos[:, None] == want[None, :])                              # [c, Q]
        if not bool(sel.any()):
            return
        rows_c, rows_q = sel.nonzero(as_tuple=True)
        p = probs[0][:, rows_c.to(probs.device), :].float()                  # [H, r, kL]
        kL = p.shape[-1]
        oh = t.nn.functional.one_hot(self.cats[:kL].to(p.device), len(PCATS)).float()
        if layer_idx not in self.patch:
            H = p.shape[0]
            self.patch[layer_idx] = t.zeros(len(self.qpos), H, int(self.image_idx.numel()), dtype=t.float32)
            self.mass[layer_idx] = t.zeros(len(self.qpos), H, len(PCATS), dtype=t.float32)
        rq = rows_q.cpu()
        self.patch[layer_idx][rq] = p[:, :, self.image_idx.to(p.device)].transpose(0, 1).cpu()
        self.mass[layer_idx][rq] = t.matmul(p, oh).transpose(0, 1).cpu()

    def result(self) -> Dict[str, np.ndarray]:
        layers = sorted(self.patch)
        if layers != LAYERS:
            raise SourcesCheckFailed(f"记录到的层 {layers} 与标准注意力层 {LAYERS} 不一致")
        mass = np.stack([self.mass[l].numpy() for l in layers], 1)           # [Q, L, H, 5]
        if np.abs(mass.sum(-1) - 1).max() > SUM_TOL:
            raise SourcesCheckFailed("提示阶段 5 类注意力之和偏离 1")
        return {"patch": np.stack([self.patch[l].numpy() for l in layers], 1).astype(np.float16),
                "mass": mass.astype(np.float16)}


def prompt_one(backend, cfg, item: dict, cond: str, rec: dict) -> Tuple[dict, dict]:
    from .. import attn
    from .conflict import claim_of
    from .generate import build_inputs
    torch, model, tok = backend.torch, backend.model, backend.tokenizer
    attn.install(model, verbose=False)
    _, inp = build_inputs(backend, item["image"], item["messages"][cond])
    in_ids = inp["input_ids"][0].tolist()
    if rec.get("input_ids") is not None and in_ids != rec["input_ids"]:
        raise SourcesCheckFailed(f"{item['id']}__{cond}：重新构造的提示与记录里的 input_ids 不一致")
    img_tok = attn.image_token_id(backend)
    image_idx = torch.tensor([i for i, x in enumerate(in_ids) if x == img_tok])
    dec = lambda x: tok.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    pieces, ok = token_pieces(dec, in_ids)
    cats, spans = prompt_categories(pieces, in_ids, img_tok, item["messages"][cond], cond, claim_of(item, cond))
    qpos = [k for k in range(len(in_ids)) if cats[k] in (1, 2, 3)] + [len(in_ids) - 1]
    pr = PromptRecorder(torch, cats, qpos, image_idx, chunk=cfg.attn_chunk)
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "rope_deltas"):
        inner.rope_deltas = None
    dev = model.device
    b = {k: v.to(dev) for k, v in inp.items()}
    if "pixel_values" in b:
        b["pixel_values"] = b["pixel_values"].to(model.dtype)
    t0 = time.time()
    with torch.inference_mode(), attn.recording(pr):
        try:
            model(**b, use_cache=False, return_dict=True, logits_to_keep=1)
        except TypeError:
            model(**b, use_cache=False, return_dict=True)
    arrays = pr.result()
    arrays.update(qpos=np.array(qpos), qcats=np.array([cats[k] if k < len(in_ids) - 1 else 4 for k in qpos], np.int8),
                  cats_prompt=cats, input_ids=np.array(in_ids))
    meta = {"id": item["id"], "condition": cond, "n_in": len(in_ids), "n_img": int(image_idx.numel()),
            "pieces": [pieces[k] for k in qpos], "pieces_ok": ok, "spans": spans, "seconds": round(time.time() - t0, 2)}
    return meta, arrays


def run_prompt(cfg, cq: str, root: str, conds: Optional[List[str]] = None, shard: str = "") -> int:
    from .conflict import _engine, _items
    from .pipeline import _atomic_json, _atomic_npz, _check_no_failure, _fail, _shard
    conds = conds or CONDS
    _check_no_failure(root)
    items = _items(cq)
    d = os.path.join(root, "prompt")
    os.makedirs(d, exist_ok=True)
    tasks = [(i, c) for i in sorted(items) for c in conds]
    todo = [t for t in _shard(tasks, shard) if not os.path.exists(os.path.join(d, f"{t[0]}__{t[1]}.json"))]
    print(f"[fp-sources prompt] 共 {len(tasks)} 条，本进程待做 {len(todo)} 条 → {d}", file=sys.stderr)
    if not todo:
        return 0
    eng = _engine(cfg)
    for n, (iid, cond) in enumerate(todo):
        try:
            rec = json.load(open(os.path.join(cq, "obs", "records", f"{iid}__{cond}.json"), encoding="utf-8"))
            meta, arrays = prompt_one(eng.backend, cfg, items[iid], cond, rec)
            _atomic_npz(os.path.join(d, f"{iid}__{cond}.npz"), arrays)
            _atomic_json(os.path.join(d, f"{iid}__{cond}.json"), meta)
            print(f"  {n + 1}/{len(todo)} {iid}__{cond} 记录 {len(meta['pieces'])} 个提示 token，{meta['seconds']}s", file=sys.stderr)
        except Exception as e:
            _fail(root, f"prompt_{iid}__{cond}", e)
            raise
    return len(todo)


def _js_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a, b：[..., n]（各自和为 1）。返回 [...] 的 JS 散度（自然对数）。"""
    m = 0.5 * (a + b)
    f = lambda x: np.sum(np.where(x > 0, x * (np.log(np.maximum(x, 1e-12)) - np.log(np.maximum(m, 1e-12))), 0.0), -1)
    return 0.5 * (f(a) + f(b))


def prompt_group_stats(arrays: dict, heads: np.ndarray, E: np.ndarray) -> Dict[str, dict]:
    """每组提示 token（题干 / 断言其余 / 断言时间 / 最后一个 token）：看图总量、图像内部分布（每头）、落在 E 的比例。"""
    patch = arrays["patch"].astype(np.float32)                              # [Q, L, H, n]
    mass = arrays["mass"].astype(np.float32)                                # [Q, L, H, 5]
    qc = arrays["qcats"].astype(int)
    last = np.zeros(len(qc), bool)
    last[-1] = True
    groups = {"question": (qc == 1) & ~last, "claim_line": ((qc == 2) | (qc == 3)) & ~last,
              "claim_rest": (qc == 3) & ~last, "claim_time": (qc == 2) & ~last, "last": last}
    out = {}
    for g, sel in groups.items():
        if not sel.any():
            continue
        P = patch[sel]
        P = P / np.maximum(P.sum(-1, keepdims=True), 1e-12)                  # 图像内部归一化
        dist = P.mean(0)                                                    # [L, H, n]
        e = (P * E[None]).sum(-1).mean(0)                                   # [L, H]
        out[g] = {"dist": dist, "img": mass[sel][..., 0].mean(0), "E": e, "n_tok": int(sel.sum())}
    return out


def run_prompt_analyze(cq: str, root: str, out_md: Optional[str] = None, out_json: Optional[str] = None) -> dict:
    from .compare import _reference
    from .conflict import _items
    from .pipeline import _atomic_json
    from .stats import cluster_bootstrap_rate, cluster_signflip
    items = _items(cq)
    d = os.path.join(root, "prompt")
    S, missing = {}, []
    for iid in sorted(items):
        ref = _reference(cq, root, iid)
        heads = np.asarray(ref["heads"], bool)
        E = np.asarray(ref["E_mask"][0], np.float32)                        # 思考阶段的关键区域 [L, H, n]
        for c in CONDS:
            p = os.path.join(d, f"{iid}__{c}.npz")
            if not os.path.exists(p):
                missing.append(f"{iid}__{c}")
                continue
            with np.load(p) as z:
                S[(iid, c)] = (prompt_group_stats({k: z[k] for k in ("patch", "mass", "qcats")}, heads, E), heads)
    HG = {"看图头": lambda h: h, "全部头": lambda h: np.ones_like(h)}
    L = ["# 读提示阶段：题目 token 与用户断言 token 看图像的情况", "",
         f"提示前向 {len(S)} 条（缺 {len(missing)}）。每个 token 用“读到它那一步”的注意力；各组先在 token 上平均。", "",
         "- 组：question = 题目；claim_line = 用户整句话（I believe the correct answer is 8:50:59.）；"
         "claim_rest = 整句里除时间以外的部分；claim_time = 时间本身；last = 提示最后一个 token（开始写字前）",
         "- 看图总量：这一组 token 的注意力里落在 169 个图像块上的总比例",
         "- E 比例：看图的那部分里，落在该题关键区域（中性回答思考时看图头最集中的 20% 图像块，即指针附近）的比例",
         "- JS：两组 token 在图像内部的注意力分布之差（每个头算一次再平均；0 = 完全一样，上限 ln2≈0.69）", ""]
    res = {"missing": missing, "tables": {}}

    def summ(diffs):
        st = cluster_signflip(diffs)
        lo, hi = cluster_bootstrap_rate(diffs)
        v = [x for _, x in diffs]
        return {"n": len(v), "mean": float(np.mean(v)), "ci": [lo, hi], "p": st["p"], "pos_share": float(np.mean([x > 0 for x in v]))}

    def row(name, r, fmt="+.4f"):
        return (f"| {name} | {r['n']} | {format(r['mean'], fmt)} | [{_fmt(r['ci'][0], 4)}, {_fmt(r['ci'][1], 4)}] | "
                f"{r['pos_share']:.2f} | {_fmt(r['p'], 4)} |")

    head = ["| 比较 | 题数 | 平均差 | 95% 区间 | 差 > 0 的比例 | p（符号翻转） |", "|---|---|---|---|---|---|"]
    # 0. 各组平均值
    L += ["## 0. 各组 token 的平均值", "", "| 条件 | 头 | 组 | 题数 | 每题 token 数 | 看图总量 | E 比例 |", "|---|---|---|---|---|---|---|"]
    for c in CONDS:
        for hg, hf in HG.items():
            for g in ("question", "claim_line", "claim_rest", "claim_time", "last"):
                vs = [(st[g], h) for (i, cc), (st, h) in S.items() if cc == c and g in st]
                if not vs:
                    continue
                m = lambda key: float(np.mean([v[key][hf(h)].mean() for v, h in vs]))
                L.append(f"| {c} | {hg} | {g} | {len(vs)} | {np.mean([v['n_tok'] for v, _ in vs]):.1f} | {m('img'):.4f} | {m('E'):.4f} |")
    # A. 同一条提示：断言 vs 题目
    for c in ("belief", "belief_right"):
        L += ["", f"## A. {c}：同一条提示里，用户断言 token − 题目 token", ""] + head
        tab = {}
        for hg, hf in HG.items():
            for g in ("claim_line", "claim_time", "claim_rest"):
                dI, dE, js = [], [], []
                for (i, cc), (st, h) in S.items():
                    if cc != c or g not in st or "question" not in st:
                        continue
                    hm = hf(h)
                    dI.append((i, float(st[g]["img"][hm].mean() - st["question"]["img"][hm].mean())))
                    dE.append((i, float(st[g]["E"][hm].mean() - st["question"]["E"][hm].mean())))
                    js.append((i, float(_js_rows(st[g]["dist"], st["question"]["dist"])[hm].mean())))
                for name, dd in ((f"{hg} {g} 看图总量", dI), (f"{hg} {g} E 比例", dE)):
                    r = summ(dd)
                    tab[name] = r
                    L.append(row(name, r))
                r = summ(js)
                tab[f"{hg} {g} JS（与题目 token）"] = r
                L.append(f"| {hg} {g} 与题目 token 的 JS（不是差，是距离） | {r['n']} | {r['mean']:.4f} | "
                         f"[{_fmt(r['ci'][0], 4)}, {_fmt(r['ci'][1], 4)}] | — | — |")
        res["tables"][f"A_{c}"] = tab
    # B / C. 用户说错 − 用户说对
    L += ["", "## B / C. 同一题：用户说错 − 用户说对（两条提示只有时间数字不同）", "",
          "claim_rest 里有数字之前的 token（两边完全相同，差应为 0，作为核对）和数字之后的句号。", ""] + head
    tab = {}
    for hg, hf in HG.items():
        for g in ("question", "claim_line", "claim_rest", "claim_time", "last"):
            dI, dE, js = [], [], []
            for i in items:
                a, b = S.get((i, "belief")), S.get((i, "belief_right"))
                if not a or not b or g not in a[0] or g not in b[0]:
                    continue
                hm = hf(a[1])
                x, y = a[0][g], b[0][g]
                dI.append((i, float(x["img"][hm].mean() - y["img"][hm].mean())))
                dE.append((i, float(x["E"][hm].mean() - y["E"][hm].mean())))
                js.append((i, float(_js_rows(x["dist"], y["dist"])[hm].mean())))
            if not dI:
                continue
            for name, dd in ((f"{hg} {g} 看图总量", dI), (f"{hg} {g} E 比例", dE)):
                r = summ(dd)
                tab[name] = r
                L.append(row(name, r))
            r = summ(js)
            tab[f"{hg} {g} JS"] = r
            L.append(f"| {hg} {g} 说错与说对的 JS（距离） | {r['n']} | {r['mean']:.4f} | [{_fmt(r['ci'][0], 4)}, {_fmt(r['ci'][1], 4)}] | — | — |")
    res["tables"]["B_wrong_vs_right"] = tab
    out_md = out_md or os.path.join(root, "prompt_report.md")
    out_json = out_json or os.path.join(root, "prompt_summary.json")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    _atomic_json(out_json, res)
    print(f"[fp-sources prompt-analyze] {len(S)} 条 → {out_md}、{out_json}；缺 {len(missing)} 条", file=sys.stderr)
    return res


# ---------------------------------------------------------------- 切断 / 降低注意力的筛选（喂回已有回答，不生成）
"""cut / cut-analyze：把原模型已有回答（用户说错、用户说对）原样喂回模型，在 16 个标准注意力层里按下表改写注意力，
记录每个输出位置“写出原来那个字”的对数概率和模型最想写的字（top-1），与不改时比较。

改写方式：
  切断  谁在看的那些 token，对“不看谁”的那些 token 的注意力置 0，其余按比例放大，总和仍为 1。
  降低  谁在看的那些 token，落在图像上的注意力总量降到目标值；降下来的部分按比例分给其余 token。

  名称      谁在看                                不看谁 / 降低什么                         头
  none      —                                     不改                                      —
  A         所有输出位置的查询（写字时）          断言整句 + 模型自己写出的用户时间          全部
  B         所有输出位置的查询                    图像                                       全部
  Qctl      所有输出位置的查询                    题目                                       全部（对照：切掉一段相近长度的别的文字）
  A1        所有输出位置的查询                    用户这一轮全部文字（题目 + 断言整句）      96 个看图头
  D         断言整句之后的所有位置（含模板、写字）断言整句（相当于删掉它的 KV）              全部（上限参考）
  Cq        断言里的时间数字（读提示时）          图像注意力降到同一句里“I believe the correct answer is”那几个 token 的平均（每层每头）  全部
  Chalf     断言里的时间数字（读提示时）          图像注意力降到原来的一半                   全部
  C0        断言里的时间数字（读提示时）          图像（置 0）                               全部
  C1        用户这一轮全部文字（读提示时）        图像（置 0）                               全部
A、D、Cq、Chalf、C0 要用到断言在哪一句，只用于分析；A1、C1 只用聊天模板给出的“用户这一轮”边界。
"""
CUTS = ["none", "A", "B", "Qctl", "A1", "D", "Cq", "Chalf", "C0", "C1"]
CUT_CONDS = ["belief", "belief_right"]


class CutRecorder:
    def __init__(self, torch, spec: dict, total: int, heads_mask: np.ndarray, chunk: int = 256):
        self.torch = torch
        self.chunk = chunk
        self.selfcheck_left = 0
        self.selfcheck_diffs: List[float] = []
        t = torch
        self.kind = spec["kind"]
        self.q = t.as_tensor(spec["q"], dtype=t.bool)
        self.k = t.as_tensor(spec["k"], dtype=t.bool)
        self.frac = spec.get("frac")
        self.qref = t.as_tensor(spec["qref"], dtype=t.bool) if spec.get("qref") is not None else None
        self.heads = heads_mask                                              # [L, H] bool 或 None（全部头）
        self.rows = 0
        self.dose = defaultdict(lambda: [0.0, 0.0, 0])                      # 层 -> [改前图像量, 改后, 行数]
        self.dev = {}

    def stages_for(self, q_pos):
        return self.torch.full_like(q_pos, -1)

    def accumulate(self, *a, **kw):
        return

    def _d(self, x, device):
        key = (id(x), str(device))
        if key not in self.dev:
            self.dev[key] = x.to(device)
        return self.dev[key]

    def modify(self, layer_idx: int, p, q_pos):
        t = self.torch
        if self.kind == "none":
            return p
        qm = self._d(self.q, p.device)[q_pos]
        if not bool(qm.any()):
            return p
        kL = p.shape[-1]
        km = self._d(self.k, p.device)[:kL].float()
        idx = qm.nonzero().flatten()
        sub = p[0][:, idx, :]                                               # [H, r, kL]
        cur = (sub * km).sum(-1)                                            # [H, r]
        if self.kind == "block":
            new = sub * (1.0 - km)
            s = new.sum(-1, keepdim=True)
            new = t.where(s > 1e-9, new / s.clamp_min(1e-12), sub)
        else:
            if self.kind == "scale_frac":
                target = cur * self.frac
            else:                                                           # scale_ref：降到参照行（题目 token）的平均
                rm = self._d(self.qref, p.device)[q_pos]
                if not bool(rm.any()):
                    raise SourcesCheckFailed("同一块查询里没有参照 token，无法取参照（提示长度超过 attn_chunk）")
                ref = (p[0][:, rm.nonzero().flatten(), :] * km).sum(-1).mean(1, keepdim=True)   # [H, 1]
                target = t.minimum(cur, ref.expand_as(cur))
            fi = (target / cur.clamp_min(1e-12))[..., None]
            fo = ((1.0 - target) / (1.0 - cur).clamp_min(1e-12))[..., None]
            new = sub * (km * fi + (1.0 - km) * fo)
        if self.heads is not None:
            hm = t.as_tensor(self.heads[LAYERS.index(layer_idx)], device=p.device)[:, None, None]
            new = t.where(hm, new, sub)
        after = (new * km).sum(-1)
        d = self.dose[layer_idx]
        d[0] += float(cur.mean()) * len(idx)
        d[1] += float(after.mean()) * len(idx)
        d[2] += len(idx)
        self.rows += len(idx)
        p = p.clone()
        p[0][:, idx, :] = new.to(p.dtype)
        return p


def cut_specs(cats_p: np.ndarray, cats_o: np.ndarray, n_in: int, close: Optional[int]) -> Dict[str, dict]:
    n_out = len(cats_o)
    total = n_in + n_out
    pos = np.arange(total)
    catall = np.concatenate([cats_p, np.full(n_out, -1, np.int8)])
    img, ques = catall == 0, catall == 1
    ctime, cline = catall == 2, (catall == 2) | (catall == 3)
    user = ques | cline
    gclaim = np.zeros(total, bool)
    gclaim[n_in:] = cats_o == 5
    gen_q = (pos >= n_in - 1) & (pos < total - 1)                          # 产生输出 token 的查询
    if not cline.any():
        raise SourcesCheckFailed("提示里没有断言句")
    after_claim = pos > np.flatnonzero(cline).max()
    prefix = (catall == 3) & (pos < np.flatnonzero(ctime).min()) if ctime.any() else np.zeros(total, bool)
    if not prefix.any():
        raise SourcesCheckFailed("断言句里时间之前没有文字（找不到 I believe the correct answer is）")
    S = {"none": {"kind": "none", "q": np.zeros(total, bool), "k": np.zeros(total, bool)},
         "A": {"kind": "block", "q": gen_q, "k": cline | gclaim},
         "B": {"kind": "block", "q": gen_q, "k": img},
         "Qctl": {"kind": "block", "q": gen_q, "k": ques},
         "A1": {"kind": "block", "q": gen_q, "k": user, "image_heads": True},
         "D": {"kind": "block", "q": after_claim, "k": cline},
         "Cq": {"kind": "scale_ref", "q": ctime, "k": img, "qref": prefix},
         "Chalf": {"kind": "scale_frac", "q": ctime, "k": img, "frac": 0.5},
         "C0": {"kind": "block", "q": ctime, "k": img},
         "C1": {"kind": "block", "q": user, "k": img}}
    return S


def cut_one(backend, cfg, item: dict, cond: str, rec: dict, heads: np.ndarray) -> Tuple[dict, dict]:
    from .. import attn
    from .conflict import CHUNK, RANK_K, RANK_MIN, claim_of
    from .entropy import _batch
    from .generate import build_inputs
    torch, model, tok = backend.torch, backend.model, backend.tokenizer
    attn.install(model, verbose=False)
    _, inp = build_inputs(backend, item["image"], item["messages"][cond])
    in_ids = inp["input_ids"][0].tolist()
    if rec.get("input_ids") is not None and in_ids != rec["input_ids"]:
        raise SourcesCheckFailed(f"{item['id']}__{cond}：重新构造的提示与记录里的 input_ids 不一致")
    n_in = len(in_ids)
    if n_in > cfg.attn_chunk:
        raise SourcesCheckFailed(f"提示长度 {n_in} 超过 attn_chunk {cfg.attn_chunk}，Cq 取不到同一块里的参照")
    out_ids = [int(x) for x in rec["output_ids"]]
    n_out = len(out_ids)
    img_tok = attn.image_token_id(backend)
    dec = lambda x: tok.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    p_pieces, _ = token_pieces(dec, in_ids)
    o_pieces, _ = token_pieces(dec, out_ids)
    claim = claim_of(item, cond)
    cats_p, _ = prompt_categories(p_pieces, in_ids, img_tok, item["messages"][cond], cond, claim)
    cats_o = output_categories(o_pieces, item["gold"], claim)
    close = out_ids.index(backend.close_id) if backend.close_id in out_ids else None
    specs = cut_specs(cats_p, cats_o, n_in, close)
    y = torch.tensor(out_ids, device=model.device)
    LP, TOP, info = [], [], {}
    for name in CUTS:
        sp = specs[name]
        cr = CutRecorder(torch, sp, n_in + n_out, heads if sp.get("image_heads") else None, chunk=cfg.attn_chunk)
        inner = getattr(model, "model", None)
        if inner is not None and hasattr(inner, "rope_deltas"):
            inner.rope_deltas = None
        t0 = time.time()
        with torch.inference_mode(), attn.recording(cr):
            o = model(**_batch(backend, inp, out_ids), use_cache=False, return_dict=True)
        lg = o.logits[0, n_in - 1: n_in - 1 + n_out]
        lp, top, rk = [], [], []
        for s in range(0, n_out, CHUNK):
            a = lg[s:s + CHUNK].float()
            ls = torch.log_softmax(a, -1)
            lp.append(ls.gather(1, y[s:s + CHUNK, None])[:, 0])
            top.append(a.argmax(-1))
            rk.append((a > a.gather(1, y[s:s + CHUNK, None])).sum(-1))
        del o, lg
        LP.append(torch.cat(lp).cpu().numpy())
        TOP.append(torch.cat(top).cpu().numpy())
        frac20 = float((torch.cat(rk) < RANK_K).float().mean())
        if name == "none" and frac20 < RANK_MIN:
            raise SourcesCheckFailed(f"{item['id']}__{cond}：不改时实际 token 进前 {RANK_K} 名的比例只有 {frac20:.3f}")
        if name != "none" and cr.rows == 0:
            raise SourcesCheckFailed(f"{item['id']}__{cond}：{name} 没有改写任何一行注意力")
        dose = {int(l): [v[0] / max(v[2], 1), v[1] / max(v[2], 1)] for l, v in cr.dose.items()}
        info[name] = {"rows_modified": cr.rows, "top20": round(frac20, 4), "seconds": round(time.time() - t0, 2),
                      "dose_before_after": dose}
    arrays = {"logp": np.stack(LP).astype(np.float32), "top1": np.stack(TOP).astype(np.int32),
              "out_ids": np.array(out_ids, np.int32), "cats_out": cats_o}
    meta = {"id": item["id"], "condition": cond, "n_in": n_in, "n_out": n_out, "close": close, "cuts": CUTS,
            "gold": item["gold"], "claim": claim, "pieces_out": o_pieces, "info": info}
    return meta, arrays


def run_cut(cfg, cq: str, root: str, conds: Optional[List[str]] = None, shard: str = "") -> int:
    from .conflict import _engine, _items
    from .pipeline import _atomic_json, _atomic_npz, _check_no_failure, _fail, _shard
    conds = [c for c in (conds or CUT_CONDS) if c in CUT_CONDS]
    _check_no_failure(root)
    items = _items(cq)
    d = os.path.join(root, "cut")
    os.makedirs(d, exist_ok=True)
    tasks = [(i, c) for i in sorted(items) for c in conds]
    todo = [t for t in _shard(tasks, shard) if not os.path.exists(os.path.join(d, f"{t[0]}__{t[1]}.json"))]
    print(f"[fp-sources cut] 共 {len(tasks)} 条 × {len(CUTS)} 种改写，本进程待做 {len(todo)} 条 → {d}", file=sys.stderr)
    if not todo:
        return 0
    eng = _engine(cfg)
    for n, (iid, cond) in enumerate(todo):
        try:
            rec = json.load(open(os.path.join(cq, "obs", "records", f"{iid}__{cond}.json"), encoding="utf-8"))
            meta, arrays = cut_one(eng.backend, cfg, items[iid], cond, rec, _neutral_heads(cq, root, iid))
            _atomic_npz(os.path.join(d, f"{iid}__{cond}.npz"), arrays)
            _atomic_json(os.path.join(d, f"{iid}__{cond}.json"), meta)
            print(f"  {n + 1}/{len(todo)} {iid}__{cond} 输出 {meta['n_out']} token，"
                  f"{sum(v['seconds'] for v in meta['info'].values()):.0f}s，不改时前20名 {meta['info']['none']['top20']}",
                  file=sys.stderr)
        except Exception as e:
            _fail(root, f"cut_{iid}__{cond}", e)
            raise
    return len(todo)


CUT_CLASSES = ["hand_induced", "hand_correct", "hand_wrong", "hand_induced_obs", "hand_correct_obs",
               "time_induced_first", "time_correct_first", "time_induced", "time_correct", "filler"]


def run_cut_analyze(cq: str, root: str, judgments: str, out_md: Optional[str] = None, out_json: Optional[str] = None) -> dict:
    from .conflict import _items
    from .pipeline import _atomic_json
    from .stats import cluster_bootstrap_rate, cluster_signflip
    items = _items(cq)
    J = json.load(open(judgments, encoding="utf-8"))
    d = os.path.join(root, "cut")
    R, missing = {}, []
    for iid in sorted(items):
        for c in CUT_CONDS:
            jp = os.path.join(d, f"{iid}__{c}.json")
            if not os.path.exists(jp):
                missing.append(f"{iid}__{c}")
                continue
            meta = json.load(open(jp, encoding="utf-8"))
            with np.load(os.path.join(d, f"{iid}__{c}.npz")) as z:
                lp, top, oid = z["logp"], z["top1"], z["out_ids"]
            n_think = meta["close"] if meta["close"] is not None else meta["n_out"]
            cls, _ = content_tokens(meta["pieces_out"], n_think, meta["gold"], meta["claim"] if c == "belief" else None)
            r = {"outcome": OUTCOME.get((J.get(f"obs|{iid}|{c}") or {}).get("outcome"), "未"), "n": {}, "d": {}, "flip": {},
                 "dose": {k: v["dose_before_after"] for k, v in meta["info"].items()}}
            base = lp[0]
            for k, rows in cls.items():
                if k not in CUT_CLASSES or not rows:
                    continue
                rows = np.array(rows)
                r["n"][k] = len(rows)
                r["d"][k] = {name: float(np.mean(lp[j][rows] - base[rows])) for j, name in enumerate(meta["cuts"])}
                r["flip"][k] = {name: float(np.mean(top[j][rows] != oid[rows])) for j, name in enumerate(meta["cuts"])}
            R[(iid, c)] = r
    L = ["# 切断 / 降低注意力：被诱导的读数还写得出来吗", "",
         f"喂回原模型已有回答 {len(R)} 条（缺 {len(missing)}）。每个位置记录“写出原来那个字”的对数概率，"
         "下表的“信心变化”= 改写后 − 不改（负 = 更不想写原来那个字）。“不再首选”= 改写后模型最想写的字不再是原来那个字的比例。", "",
         "改写方式见代码 `symexp/fp/sources.py` 的 cut 部分说明。A、D、Cq、Chalf、C0 用到断言位置（只作分析）；A1、C1 不用。", ""]
    res = {"missing": missing, "tables": {}}

    def paired(cond, a, b, name, sel=None):
        diffs = []
        for (i, c), r in R.items():
            if c != cond or (sel and not sel(r)) or a not in r["d"] or b not in r["d"]:
                continue
            diffs.append((i, r["d"][a][name] - r["d"][b][name]))
        if not diffs:
            return None
        st = cluster_signflip(diffs)
        lo, hi = cluster_bootstrap_rate(diffs)
        v = [x for _, x in diffs]
        return {"n": len(v), "mean": float(np.mean(v)), "ci": [lo, hi], "p": st["p"], "neg": float(np.mean([x < 0 for x in v]))}

    def mean_over(cond, k, name, field="d"):
        vs = [r[field][k][name] for (i, c), r in R.items() if c == cond and k in r[field]]
        return (float(np.mean(vs)), len(vs)) if vs else (None, 0)

    for cond in CUT_CONDS:
        L += ["", f"## {cond}：各类字的平均信心变化（先在每条回答内平均，再在回答之间平均）", "",
              "| 改写 | " + " | ".join(CUT_CLASSES) + " |", "|---|" + "---|" * len(CUT_CLASSES)]
        for name in CUTS[1:]:
            cells = []
            for k in CUT_CLASSES:
                m, n = mean_over(cond, k, name)
                cells.append("—" if m is None else f"{m:+.3f}（{n}）")
            L.append(f"| {name} | " + " | ".join(cells) + " |")
        L += ["", f"**{cond}：不再首选的比例**（不改时的值在第一行）", "",
              "| 改写 | " + " | ".join(CUT_CLASSES) + " |", "|---|" + "---|" * len(CUT_CLASSES)]
        for name in CUTS:
            cells = []
            for k in CUT_CLASSES:
                m, n = mean_over(cond, k, name, "flip")
                cells.append("—" if m is None else f"{m:.3f}")
            L.append(f"| {name} | " + " | ".join(cells) + " |")
    L += ["", "## 用户说错：同一条回答里，被诱导的读数字 − 正确的读数字（信心变化之差；负 = 改写对被诱导的字打击更大）", "",
          "| 改写 | 字的类别 | 回答数 | 平均差 | 95% 区间 | 差 < 0 的比例 | p（符号翻转） |", "|---|---|---|---|---|---|---|"]
    tab = []
    for name in CUTS[1:]:
        for a, b, lab in (("hand_induced", "hand_correct", "指针数字"), ("hand_induced_obs", "hand_correct_obs", "指针数字（严格版）"),
                          ("time_induced_first", "time_correct_first", "时间数字（首次出现）")):
            r = paired("belief", a, b, name)
            if r is None:
                continue
            tab.append({"cut": name, "class": lab, **r})
            L.append(f"| {name} | {lab} | {r['n']} | {r['mean']:+.3f} | [{_fmt(r['ci'][0])}, {_fmt(r['ci'][1])}] | "
                     f"{r['neg']:.2f} | {_fmt(r['p'], 4)} |")
    res["tables"]["induced_minus_correct"] = tab
    L += ["", "## 读提示时降低时间数字看图的程度（Cq / Chalf / C0，用户说错，16 层平均）", "",
          "| 改写 | 改前看图总量 | 改后看图总量 |", "|---|---|---|"]
    for name in ("Cq", "Chalf", "C0", "C1"):
        v = [np.mean([x[0] for x in r["dose"][name].values()]) for (i, c), r in R.items() if c == "belief" and r["dose"].get(name)]
        w = [np.mean([x[1] for x in r["dose"][name].values()]) for (i, c), r in R.items() if c == "belief" and r["dose"].get(name)]
        if v:
            L.append(f"| {name} | {np.mean(v):.4f} | {np.mean(w):.4f} |")
    res["records"] = {f"{i}__{c}": r for (i, c), r in R.items()}
    out_md = out_md or os.path.join(root, "cut_report.md")
    out_json = out_json or os.path.join(root, "cut_summary.json")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    _atomic_json(out_json, res)
    print(f"[fp-sources cut-analyze] {len(R)} 条 → {out_md}、{out_json}；缺 {len(missing)} 条", file=sys.stderr)
    return res


# ---------------------------------------------------------------- 读提示阶段：逐头比较看图位置（不在头之间平均）
"""prompt-heads：用 prompt 阶段存下的注意力（每个 token × 16 层 × 24 头 × 169 图像块），在每个（层, 头）上分别算
两种条件之间“看图位置”的 JSD（图像内部归一化后的分布），不在头之间平均。

比较的 token 组（位置在不同条件之间是对齐的）：
  last        提示最后一个 token（模型开始写字前那一刻）：中性 vs 说错、中性 vs 说对、说错 vs 说对
  claim_time  断言里的时间数字（组内先平均）：说错 vs 说对
  claim_line  断言整句：说错 vs 说对
  question    题目 token：中性 vs 说错（题目在断言之前，按因果注意力应完全相同，差应为 0，作为核对）
  参照        同一条提示里，时间数字 vs 题目 token（两种完全不同的 token）的 JSD，用来给大小一个尺度

每个头还给出“关键区域（E，指针附近）比例”在两种条件之间的变化，以及这个变化在各题之间方向是否一致
（符号翻转检验，按 384 个头做 Benjamini–Hochberg 校正）。
"""


def _bh(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    thr = q * (np.arange(1, n + 1) / n)
    ok = p[order] <= thr
    k = int(np.max(np.nonzero(ok)[0]) + 1) if ok.any() else 0
    out = np.zeros(n, bool)
    out[order[:k]] = True
    return out


def _signflip_p(x: np.ndarray, reps: int = 20000, seed: int = 0) -> float:
    x = np.asarray(x, float)
    x = x[x != 0]
    if len(x) < 2:
        return 1.0
    obs = abs(x.sum())
    rng = np.random.default_rng(seed)
    s = rng.choice([-1.0, 1.0], size=(reps, len(x)))
    return float((1 + (np.abs(s @ x) >= obs - 1e-12).sum()) / (reps + 1))


def _signflip_many(X: np.ndarray, reps: int = 20000, seed: int = 0) -> np.ndarray:
    """X：[题数, 头数]。每一列一个符号翻转检验（双侧），所有列共用同一组随机符号。"""
    X = np.asarray(X, float)
    rng = np.random.default_rng(seed)
    S = rng.choice([-1.0, 1.0], size=(reps, X.shape[0]))
    obs = np.abs(X.sum(0))
    null = np.abs(S @ X)                                                   # [reps, 头数]
    return (1 + (null >= obs[None] - 1e-12).sum(0)) / (reps + 1)


def run_prompt_heads(cq: str, root: str, out_md: Optional[str] = None, out_json: Optional[str] = None) -> dict:
    from .compare import _reference
    from .conflict import _items
    from .pipeline import _atomic_json
    items = _items(cq)
    d = os.path.join(root, "prompt")
    comps = [("last", "neutral", "belief"), ("last", "neutral", "belief_right"), ("last", "belief", "belief_right"),
             ("claim_time", "belief", "belief_right"), ("claim_line", "belief", "belief_right"),
             ("question", "neutral", "belief")]
    JS = {c: [] for c in comps}
    DE = {c: [] for c in comps}
    REF = []                                                               # 时间数字 vs 题目（同一条说错提示里）
    used, heads_any = [], None
    for iid in sorted(items):
        st = {}
        for c in CONDS:
            p = os.path.join(d, f"{iid}__{c}.npz")
            if not os.path.exists(p):
                break
            with np.load(p) as z:
                st[c] = z["patch"].astype(np.float32), z["qcats"].astype(int)
        if len(st) < 3:
            continue
        ref = _reference(cq, root, iid)
        E = np.asarray(ref["E_mask"][0], np.float32)
        heads_any = np.asarray(ref["heads"], bool) if heads_any is None else heads_any | np.asarray(ref["heads"], bool)
        grp = {}
        for c, (P, qc) in st.items():
            P = P / np.maximum(P.sum(-1, keepdims=True), 1e-12)
            last = np.zeros(len(qc), bool)
            last[-1] = True
            sel = {"last": last, "question": (qc == 1) & ~last, "claim_time": (qc == 2) & ~last,
                   "claim_line": ((qc == 2) | (qc == 3)) & ~last}
            grp[c] = {g: P[s].mean(0) for g, s in sel.items() if s.any()}      # [L, H, n]
        for comp in comps:
            g, a, b = comp
            if g not in grp[a] or g not in grp[b]:
                continue
            x, y = grp[a][g], grp[b][g]
            JS[comp].append(_js_rows(x, y))                                # [L, H]
            DE[comp].append((x * E).sum(-1) - (y * E).sum(-1))             # [L, H]，a − b
        REF.append(_js_rows(grp["belief"]["claim_time"], grp["belief"]["question"]))
        used.append(iid)
    L = ["# 读提示阶段：逐头比较看图位置（每个层、每个头单独算，不在头之间平均）", "",
         f"用了 {len(used)} 题。JSD 在图像内部归一化后的分布上算（上限 ln2≈0.69）；每个头先在题之间平均。"
         "E 比例 = 看图部分里落在关键区域（该题中性回答思考时看图头最集中的 20% 图像块，指针附近）的比例。", "",
         "## 1. 384 个头的 JSD 分布（每个头先对题平均）", "",
         "| 比较 | 中位数 | 前 25% 的边界 | 前 5% 的边界 | 最大 | 最大的那个头（层, 头） | 看图头里的中位数 |",
         "|---|---|---|---|---|---|---|"]
    res = {"items": used, "comparisons": {}}
    ref_m = np.mean(REF, 0) if REF else None
    for comp in comps:
        if not JS[comp]:
            continue
        m = np.mean(JS[comp], 0)                                           # [L, H]
        flat = m.ravel()
        li, hi = np.unravel_index(int(np.argmax(m)), m.shape)
        name = f"{comp[0]}：{comp[1]} vs {comp[2]}"
        hv = m[heads_any] if heads_any is not None else flat
        L.append(f"| {name} | {np.median(flat):.4f} | {np.quantile(flat, .75):.4f} | {np.quantile(flat, .95):.4f} | "
                 f"{flat.max():.4f} | ({LAYERS[li]}, {hi}) | {np.median(hv):.4f} |")
        res["comparisons"][name] = {"js_mean_by_head": m.tolist()}
    if ref_m is not None:
        f = ref_m.ravel()
        L.append(f"| 参照：说错提示里 时间数字 vs 题目 token | {np.median(f):.4f} | {np.quantile(f, .75):.4f} | "
                 f"{np.quantile(f, .95):.4f} | {f.max():.4f} | — | — |")
        res["reference_js_by_head"] = ref_m.tolist()
    for comp in [c for c in comps if c[0] != "question" and JS[c]]:
        m = np.mean(JS[comp], 0)
        de = np.array(DE[comp])                                            # [items, L, H]
        ps = _signflip_many(de.reshape(de.shape[0], -1)).reshape(de.shape[1:])
        sig = _bh(ps.ravel()).reshape(ps.shape)
        name = f"{comp[0]}：{comp[1]} vs {comp[2]}"
        order = np.argsort(-m.ravel())[:10]
        L += ["", f"## 2. {name}：JSD 最大的 10 个头", "",
              f"E 比例变化 = {comp[1]} − {comp[2]}（负 = 前者落在指针附近更少）；“方向一致” = 各题里 E 比例变化为负的比例。", "",
              "| 层 | 头 | 是否看图头 | 平均 JSD | 参照 JSD（数字 vs 题目） | E 比例平均变化 | 方向一致（变化 < 0 的题） | p | BH 校正后显著 |",
              "|---|---|---|---|---|---|---|---|---|"]
        top = []
        for k in order:
            l, h = np.unravel_index(int(k), m.shape)
            x = de[:, l, h]
            row = {"layer": LAYERS[l], "head": int(h), "image_head": bool(heads_any[l, h]) if heads_any is not None else None,
                   "js": float(m[l, h]), "ref_js": float(ref_m[l, h]) if ref_m is not None else None,
                   "dE": float(x.mean()), "neg_share": float((x < 0).mean()), "p": float(ps[l, h]), "bh": bool(sig[l, h])}
            top.append(row)
            L.append(f"| {row['layer']} | {h} | {'是' if row['image_head'] else '否'} | {row['js']:.4f} | "
                     f"{_fmt(row['ref_js'], 4)} | {row['dE']:+.4f} | {row['neg_share']:.2f} | {_fmt(row['p'], 4)} | "
                     f"{'是' if row['bh'] else '否'} |")
        n_sig = int(sig.sum())
        neg = int((sig & (de.mean(0) < 0)).sum())
        L += ["", f"全部 384 个头里，E 比例变化在各题之间方向一致、BH 校正后显著的头：{n_sig} 个"
                  f"（其中 {neg} 个是 {comp[1]} 落在指针附近更少，{n_sig - neg} 个是更多）。"]
        res["comparisons"][name].update(top10=top, n_sig=n_sig, n_sig_less_E=neg,
                                        dE_mean_by_head=de.mean(0).tolist(), p_by_head=ps.tolist())
    out_md = out_md or os.path.join(root, "prompt_heads_report.md")
    out_json = out_json or os.path.join(root, "prompt_heads_summary.json")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    _atomic_json(out_json, res)
    print(f"[fp-sources prompt-heads] {len(used)} 题 → {out_md}、{out_json}", file=sys.stderr)
    return res


# ---------------------------------------------------------------- 思考开头 token 看图的分布：三种条件、相同长度、逐头比较
"""reading / reading-analyze：思考最前面的 token（从第 1 个开始，不挑句子）对 169 个图像 token 的注意力分布，
在三种条件之间逐头比较。

回放时取思考的前 512 个 token（思考更短就取到思考结束），每 8 个连续 token 存一组（每层每头、每个图像块的注意力之和）。
分析时分别看前 64 / 128 / 256 / 512 个 token：
  相同长度  三种条件都取思考的前 W 个 token（某题有一种条件的思考不足 W 个，这一题在这个长度下不用）
  条件之间  中性 vs 说错、中性 vs 说对、说错 vs 说对，每个（层, 头）算 JSD（平均后在图像内部归一化）
  噪声基准  同一条回答的前 W 个 token 按组交替分成两半，两半之间的 JSD；条件之间也用同样大小的半份（偶数组 vs 偶数组）比
"""
READ_MAX = 512
READ_GROUP = 8
READ_WINDOWS = [64, 128, 256, 512]
TOP_WINDOW = 256


def reading_rows(n_think: int) -> List[int]:
    """思考最前面的至多 READ_MAX 个 token（输出行号 0, 1, 2, …）。"""
    return list(range(min(n_think, READ_MAX)))


class ReadingRecorder:
    def __init__(self, torch, rows: List[int], n_in: int, image_idx, chunk: int = 256, layers=None):
        self.torch, self.chunk = torch, chunk
        self.layers = list(LAYERS if layers is None else layers)
        self.selfcheck_left, self.selfcheck_diffs = 0, []
        self.n_in = n_in
        self.image_idx = image_idx
        G = (len(rows) + READ_GROUP - 1) // READ_GROUP
        self.G = G
        self.group = {int(r): i // READ_GROUP for i, r in enumerate(rows)}
        self.rows_t = torch.as_tensor(sorted(self.group), dtype=torch.long)
        self.dist, self.mass, self.cnt = {}, {}, np.zeros(G, np.int64)
        self.filled = {}

    def stages_for(self, q_pos):
        return self.torch.full_like(q_pos, -1)

    def accumulate(self, layer_idx: int, probs, stages, q_pos):
        t = self.torch
        if layer_idx not in self.layers:
            raise SourcesCheckFailed(f"第 {layer_idx} 层不在要记录的层列表里")
        k = (q_pos - (self.n_in - 1)).cpu()
        hit = t.isin(k, self.rows_t)
        if not bool(hit.any()):
            return
        idx = hit.nonzero().flatten()
        p = probs[0][:, idx.to(probs.device), :][:, :, self.image_idx.to(probs.device)].float()   # [H, r, n]
        H, n = p.shape[0], p.shape[-1]
        if layer_idx not in self.dist:
            self.dist[layer_idx] = np.zeros((self.G, H, n), np.float64)
            self.mass[layer_idx] = np.zeros((self.G, H), np.float64)
            self.filled[layer_idx] = 0
        pc = p.cpu().numpy()
        for j, kk in enumerate(k[idx].tolist()):
            g = self.group[int(kk)]
            self.dist[layer_idx][g] += pc[:, j, :]
            self.mass[layer_idx][g] += pc[:, j, :].sum(-1)
            if layer_idx == self.layers[0]:
                self.cnt[g] += 1
            self.filled[layer_idx] += 1

    def result(self) -> Dict[str, np.ndarray]:
        layers = sorted(self.dist)
        if layers != sorted(self.layers):
            raise SourcesCheckFailed(f"记录到的层 {layers} 与标准注意力层 {self.layers} 不一致")
        n_rows = len(self.group)
        for l in layers:
            if self.filled[l] != n_rows:
                raise SourcesCheckFailed(f"第 {l} 层读图 token 记录了 {self.filled[l]} 行，应为 {n_rows}")
        return {"dist_sum": np.stack([self.dist[l] for l in layers], 1).astype(np.float32),   # [G, L, H, n]
                "mass_sum": np.stack([self.mass[l] for l in layers], 1).astype(np.float32),   # [G, L, H]
                "count": self.cnt}


class ScoreRecorder(ReadingRecorder):
    """在 ReadingRecorder 之外，再记 softmax 之前的注意力分数（query 与 key 的匹配程度）：
    每个查询对 n 个图像 token 的分数减去它对全部图像 token 分数的平均（softmax 对整体平移不变，只有相对值有意义），
    每 8 个 token 一组求和，存成 rel_sum [组, 层, 头, n]。"""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.rel = {}

    def accumulate_scores(self, layer_idx: int, sc, q_pos):
        t = self.torch
        k = (q_pos - (self.n_in - 1)).cpu()
        hit = t.isin(k, self.rows_t)
        if not bool(hit.any()):
            return
        idx = hit.nonzero().flatten()
        x = sc[0][:, idx.to(sc.device), :][:, :, self.image_idx.to(sc.device)].float()      # [H, r, n]
        if not bool(t.isfinite(x).all()):
            raise SourcesCheckFailed(f"第 {layer_idx} 层：图像 token 的注意力分数里有被遮挡的位置")
        x = (x - x.mean(-1, keepdim=True)).cpu().numpy()
        if layer_idx not in self.rel:
            self.rel[layer_idx] = np.zeros((self.G, x.shape[0], x.shape[-1]), np.float64)
        for j, kk in enumerate(k[idx].tolist()):
            self.rel[layer_idx][self.group[int(kk)]] += x[:, j, :]

    def result(self) -> Dict[str, np.ndarray]:
        out = super().result()
        layers = sorted(self.dist)
        if sorted(self.rel) != layers:
            raise SourcesCheckFailed(f"分数记录的层 {sorted(self.rel)} 与注意力记录的层 {layers} 不一致")
        out["rel_sum"] = np.stack([self.rel[l] for l in layers], 1).astype(np.float32)
        return out


def reading_one(backend, cfg, item: dict, cond: str, rec: dict, meta: dict, layers=None, only=None,
                skip: int = 0, recorder_cls=None) -> Tuple[dict, dict]:
    """layers / only：要记录的层与模块筛选（默认 Qwen3.8 的 16 个标准层）；skip：输出开头跳过的 token 数
    （思考开头标记由模型自己写的模型，如 Gemma 4 的 <|channel>thought，跳过这几个标记 token）。"""
    from .. import attn
    from .entropy import _batch
    from .generate import build_inputs
    torch, model = backend.torch, backend.model
    got = attn.install(model, verbose=False, only=only)
    if layers is not None and sorted(got) != sorted(layers):
        raise SourcesCheckFailed(f"装上的记录层 {got} 与要记录的层 {layers} 不一致")
    _, inp = build_inputs(backend, item["image"], item["messages"][cond])
    in_ids = inp["input_ids"][0].tolist()
    if rec.get("input_ids") is not None and in_ids != rec["input_ids"]:
        raise SourcesCheckFailed(f"{item['id']}__{cond}：重新构造的提示与记录里的 input_ids 不一致")
    out_ids = [int(x) for x in rec["output_ids"]]
    if len(out_ids) != meta["n_out"]:
        raise SourcesCheckFailed(f"{item['id']}__{cond}：记录与回放 meta 的输出长度不一致")
    n_think = meta["close"] if meta["close"] is not None else meta["n_out"]
    rows = [skip + r for r in reading_rows(max(n_think - skip, 0))]
    if not rows:
        return {"id": item["id"], "condition": cond, "rows": [], "groups": 0}, {}
    img_tok = attn.image_token_id(backend)
    image_idx = torch.tensor([i for i, x in enumerate(in_ids) if x == img_tok])
    rr = (recorder_cls or ReadingRecorder)(torch, rows, len(in_ids), image_idx, chunk=cfg.attn_chunk, layers=layers)
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "rope_deltas"):
        inner.rope_deltas = None
    t0 = time.time()
    with torch.inference_mode(), attn.recording(rr):
        try:
            model(**_batch(backend, inp, out_ids), use_cache=False, return_dict=True, logits_to_keep=1)
        except TypeError:
            model(**_batch(backend, inp, out_ids), use_cache=False, return_dict=True)
    arrays = rr.result()
    m = {"id": item["id"], "condition": cond, "rows": rows, "groups": rr.G, "seconds": round(time.time() - t0, 2),
         "first_rows": rows[:8], "last_row": rows[-1]}
    return m, arrays


def run_reading(cfg, cq: str, root: str, conds: Optional[List[str]] = None, shard: str = "") -> int:
    from .conflict import _engine, _items
    from .pipeline import _atomic_json, _atomic_npz, _check_no_failure, _fail, _shard
    conds = conds or CONDS
    _check_no_failure(root)
    items = _items(cq)
    d = os.path.join(root, "reading")
    os.makedirs(d, exist_ok=True)
    tasks = [(i, c) for i in sorted(items) for c in conds]
    todo = [t for t in _shard(tasks, shard) if not os.path.exists(os.path.join(d, f"{t[0]}__{t[1]}.json"))]
    print(f"[fp-sources reading] 共 {len(tasks)} 条，本进程待做 {len(todo)} 条 → {d}", file=sys.stderr)
    if not todo:
        return 0
    eng = _engine(cfg)
    for n, (iid, cond) in enumerate(todo):
        try:
            rec = json.load(open(os.path.join(cq, "obs", "records", f"{iid}__{cond}.json"), encoding="utf-8"))
            mp = os.path.join(root, "replay", f"{iid}__{cond}.json")
            if not os.path.exists(mp):
                raise SourcesCheckFailed(f"缺少之前回放的 {mp}（读图 token 的文字片段从这里取）")
            meta = json.load(open(mp, encoding="utf-8"))
            m, arrays = reading_one(eng.backend, cfg, items[iid], cond, rec, meta)
            if arrays:
                _atomic_npz(os.path.join(d, f"{iid}__{cond}.npz"), arrays)
            _atomic_json(os.path.join(d, f"{iid}__{cond}.json"), m)
            print(f"  {n + 1}/{len(todo)} {iid}__{cond} 思考开头 token {len(m['rows'])} 个（{m['groups']} 组）", file=sys.stderr)
        except Exception as e:
            _fail(root, f"reading_{iid}__{cond}", e)
            raise
    return len(todo)


def _norm_img(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(x.sum(-1, keepdims=True), 1e-12)


def _reading_window(Z: dict, E: np.ndarray, G: int) -> dict:
    avg = {c: _norm_img(Z[c][0][:G].sum(0)) for c in CONDS}
    ev = {c: _norm_img(Z[c][0][:G:2].sum(0)) for c in CONDS}
    od = {c: _norm_img(Z[c][0][1:G:2].sum(0)) for c in CONDS}
    mass = {c: Z[c][1][:G].sum(0) / max(int(Z[c][2][:G].sum()), 1) for c in CONDS}
    out = {"within": {c: _js_rows(ev[c], od[c]) for c in CONDS}}
    for a, b in [("neutral", "belief"), ("neutral", "belief_right"), ("belief", "belief_right")]:
        out[(a, b)] = {"full": _js_rows(avg[a], avg[b]), "half": _js_rows(ev[a], ev[b]),
                       "dE": (avg[a] * E).sum(-1) - (avg[b] * E).sum(-1), "dM": mass[a] - mass[b]}
    return out


def run_reading_analyze(cq: str, root: str, judgments: str, out_md: Optional[str] = None,
                        out_json: Optional[str] = None) -> dict:
    from .compare import _reference
    from .conflict import _items
    from .pipeline import _atomic_json
    items = _items(cq)
    J = json.load(open(judgments, encoding="utf-8"))
    d = os.path.join(root, "reading")
    pairs = [("neutral", "belief"), ("neutral", "belief_right"), ("belief", "belief_right")]
    per = {W: [] for W in READ_WINDOWS}                                   # (题号, 结果, 这个长度的统计)
    heads_any = None
    for iid in sorted(items):
        Z = {}
        for c in CONDS:
            p = os.path.join(d, f"{iid}__{c}.npz")
            if os.path.exists(p):
                with np.load(p) as z:
                    Z[c] = (z["dist_sum"].astype(np.float64), z["mass_sum"].astype(np.float64), z["count"])
        if len(Z) < 3:
            continue
        ref = _reference(cq, root, iid)
        E = np.asarray(ref["E_mask"][0], np.float64)
        hm = np.asarray(ref["heads"], bool)
        heads_any = hm if heads_any is None else heads_any | hm
        layer_ids = [int(x) for x in ref.get("layer_ids", LAYERS)]
        full_groups = min(int((Z[c][2] == READ_GROUP).sum()) for c in CONDS)
        j = J.get(f"obs|{iid}|belief")
        oc = OUTCOME.get(j["outcome"], "未") if j else "未"
        for W in READ_WINDOWS:
            G = W // READ_GROUP
            if full_groups >= G:
                per[W].append((iid, oc, _reading_window(Z, E, G)))
    if not any(per.values()):
        raise SourcesCheckFailed("没有三种条件都有足够思考 token 的题")
    A = np.array
    L = ["# 思考开头 token 看图的分布：三种条件、相同长度、逐头比较", "",
         "三种条件都取思考的前 W 个 token（不挑句子），平均后在图像内部归一化，每个（层, 头）算 JSD（上限 ln2≈0.69），"
         "每个头先对题平均。噪声基准 = 同一条回答的前 W 个 token 按 8 个一组交替分成两半，两半之间的 JSD；"
         "条件之间也用同样大小的半份比。", "",
         "## 1. 条件之间 vs 同一条回答内部（都用半份），384 个头的中位数（括号里是看图头的中位数）", "",
         "| 前 W 个 token | 题数 | 内部：中性 | 内部：说错 | 内部：说对 | 之间：中性 vs 说错 | 之间：中性 vs 说对 | 之间：说错 vs 说对 |",
         "|---|---|---|---|---|---|---|---|"]
    res = {"windows": {}}

    def med(m):
        return f"{np.median(m):.4f}（{np.median(m[heads_any]):.4f}）"
    for W in READ_WINDOWS:
        P = per[W]
        if not P:
            continue
        wi = {c: A([x[2]["within"][c] for x in P]).mean(0) for c in CONDS}
        bt = {pr: A([x[2][pr]["half"] for x in P]).mean(0) for pr in pairs}
        L.append(f"| {W} | {len(P)} | " + " | ".join(med(wi[c]) for c in CONDS) + " | "
                 + " | ".join(med(bt[pr]) for pr in pairs) + " |")
        res["windows"][str(W)] = {"items": [x[0] for x in P]}
    L += ["", "## 2. 条件之间超出噪声的量（每个头：之间的半份 JSD − 两种条件内部半份 JSD 的平均；正 = 超出噪声）", "",
          "p 为按题的符号翻转检验，BH 校正。", "",
          "| 前 W 个 token | 比较 | 头平均超出量 | 超出量 > 0 的头 | BH 校正后显著超出的头 | 其中看图头 |",
          "|---|---|---|---|---|---|"]
    for W in READ_WINDOWS:
        P = per[W]
        if len(P) < 4:
            continue
        for a, b in pairs:
            ex = A([x[2][(a, b)]["half"] - 0.5 * (x[2]["within"][a] + x[2]["within"][b]) for x in P])
            ps = _signflip_many(ex.reshape(len(P), -1)).reshape(ex.shape[1:])
            sig = _bh(ps.ravel()).reshape(ps.shape) & (ex.mean(0) > 0)
            L.append(f"| {W} | {a} vs {b} | {ex.mean():+.4f} | {int((ex.mean(0) > 0).sum())} / {ex.shape[1] * ex.shape[2]} | {int(sig.sum())} | "
                     f"{int((sig & heads_any).sum())} / {int(heads_any.sum())} |")
            res["windows"][str(W)][f"excess_{a}_{b}"] = {"mean_by_head": ex.mean(0).tolist(), "p_by_head": ps.tolist(),
                                                         "n_sig": int(sig.sum())}
    L += ["", "## 3. 完整长度的 JSD、关键区域比例、看图总量（看图头平均）", "",
          "E 比例 = 看图部分里落在关键区域（指针附近）的比例；差 = 前者 − 后者。看图总量 = 每个 token 的注意力里落在图像上的比例。", "",
          "| 前 W 个 token | 比较 | JSD 头中位数 | JSD 最大 | E 比例差 | E 比例差为负的题 | 看图总量差 | 看图总量差为负的题 |",
          "|---|---|---|---|---|---|---|---|"]
    for W in READ_WINDOWS:
        P = per[W]
        if not P:
            continue
        for a, b in pairs:
            m = A([x[2][(a, b)]["full"] for x in P]).mean(0)
            de = A([x[2][(a, b)]["dE"][heads_any].mean() for x in P])
            dm = A([x[2][(a, b)]["dM"][heads_any].mean() for x in P])
            L.append(f"| {W} | {a} vs {b} | {np.median(m):.4f} | {m.max():.4f} | {de.mean():+.4f} | {(de < 0).mean():.2f} | "
                     f"{dm.mean():+.4f} | {(dm < 0).mean():.2f} |")
    L += ["", "## 4. 用户说错时按最终结果分开：中性 vs 说错 超出噪声的量（头平均 / 看图头平均）", "",
          "| 前 W 个 token | 坚持 | 采纳 | 答成别的错 |", "|---|---|---|---|"]
    for W in READ_WINDOWS:
        P = per[W]
        if not P:
            continue
        cells = []
        for o in ("对", "采", "他"):
            ex = [x[2][("neutral", "belief")]["half"] - 0.5 * (x[2]["within"]["neutral"] + x[2]["within"]["belief"])
                  for x in P if x[1] == o]
            cells.append(f"{A(ex).mean():+.4f} / {A(ex)[:, heads_any].mean():+.4f}（{len(ex)} 题）" if ex else "—")
        L.append(f"| {W} | " + " | ".join(cells) + " |")
    TW = TOP_WINDOW if per[TOP_WINDOW] else max(W for W in READ_WINDOWS if per[W])
    P = per[TW]
    m = A([x[2][("neutral", "belief")]["full"] for x in P]).mean(0)
    noise = A([0.5 * (x[2]["within"]["neutral"] + x[2]["within"]["belief"]) for x in P]).mean(0)
    half = A([x[2][("neutral", "belief")]["half"] for x in P]).mean(0)
    de = A([x[2][("neutral", "belief")]["dE"] for x in P]).mean(0)
    order = np.argsort(-(half - noise).ravel())[:10]
    L += ["", f"## 5. 前 {TW} 个 token，中性 vs 说错：超出噪声最多的 10 个头", "",
          "| 层 | 头 | 是否看图头 | 完整长度 JSD | 之间（半份） | 噪声（半份） | E 比例差（中性 − 说错） |",
          "|---|---|---|---|---|---|---|"]
    for k in order:
        l, h = np.unravel_index(int(k), m.shape)
        L.append(f"| {layer_ids[l]} | {h} | {'是' if heads_any[l, h] else '否'} | {m[l, h]:.4f} | {half[l, h]:.4f} | "
                 f"{noise[l, h]:.4f} | {de[l, h]:+.4f} |")
    out_md = out_md or os.path.join(root, "reading_report.md")
    out_json = out_json or os.path.join(root, "reading_summary.json")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    _atomic_json(out_json, res)
    print(f"[fp-sources reading-analyze] 各长度的题数 {[len(per[W]) for W in READ_WINDOWS]} → {out_md}、{out_json}",
          file=sys.stderr)
    return res


# ---------------------------------------------------------------- 按“用户时间的指针位置是否和真实指针重合”分组
"""reading-claim：用 reading 阶段存好的数据（思考开头 token 看图的分布），按题分两组再比较：
  重合组  用户时间的每根针都离某根真实指针不超过 35°（只是对调指针，或差一个刻度左右）
  不同组  用户时间至少有一根针离所有真实指针都超过 35°（指向表盘上别的地方）
不同组里再看注意力有没有转向“用户时间的那根针该在的地方”：
  表盘几何  从图像自动估计：深色像素的重心作圆心，深色像素到圆心距离的第 95 百分位作半径（会输出核对图）
  区域      某根针的区域 = 与这根针方向相差 ≤ 15°、离圆心 0.15–1.05 个半径的图像块
            用户区域 = 用户时间里离所有真实指针 > 35° 的那几根针的区域，去掉与真实指针区域重叠的部分
            真实区域 = 正确时间三根针的区域
  比较      思考前 W 个 token，看图头上落在用户区域 / 真实区域的比例：说错 vs 说对（同一题配对），以及中性
"""
SEP_DEG = 35.0
SECTOR_DEG = 15.0


def hand_angles(t: str) -> List[float]:
    from .conflict import _parse
    h, m, s = _parse(t)
    a = [(h + m / 60 + (s or 0) / 3600) * 30 % 360, (m + (s or 0) / 60) * 6 % 360]
    if s is not None:
        a.append(s * 6 % 360)
    return a


def _adist(a: float, b: float) -> float:
    x = abs(a - b) % 360
    return min(x, 360 - x)


def claim_far_hands(gold: str, claim: str, sep: float = SEP_DEG) -> List[float]:
    g = hand_angles(gold)
    return [a for a in hand_angles(claim) if min(_adist(a, b) for b in g) > sep]


def dial_geometry(img_path: str) -> Tuple[float, float, float, int, int]:
    """(圆心 x, 圆心 y, 半径, 宽, 高)，像素坐标。"""
    from PIL import Image
    im = np.asarray(Image.open(img_path).convert("L"), np.float64)
    H, W = im.shape
    dark = im < 100
    if dark.sum() < 50:
        dark = im < np.quantile(im, 0.1)
    ys, xs = np.nonzero(dark)
    cx, cy = xs.mean(), ys.mean()
    r = np.quantile(np.hypot(xs - cx, ys - cy), 0.95)
    return float(cx), float(cy), float(r), W, H


def sector_mask(geom, grid_hw, angles: List[float], width: float = SECTOR_DEG) -> np.ndarray:
    cx, cy, R, W, H = geom
    gh, gw = int(grid_hw[0]), int(grid_hw[1])
    m = np.zeros(gh * gw, bool)
    for i in range(gh):
        for j in range(gw):
            x, y = (j + 0.5) / gw * W, (i + 0.5) / gh * H
            dx, dy = x - cx, y - cy
            r = np.hypot(dx, dy)
            if not 0.15 * R <= r <= 1.05 * R:
                continue
            th = np.degrees(np.arctan2(dx, -dy)) % 360
            if any(_adist(th, a) <= width for a in angles):
                m[i * gw + j] = True
    return m


def _check_image(img_path, geom, grid_hw, gold_m, claim_m, out_png, title):
    from PIL import Image, ImageDraw
    im = Image.open(img_path).convert("RGB")
    cx, cy, R, W, H = geom
    dr = ImageDraw.Draw(im, "RGBA")
    gh, gw = int(grid_hw[0]), int(grid_hw[1])
    for k in range(gh * gw):
        i, j = divmod(k, gw)
        box = [j / gw * W, i / gh * H, (j + 1) / gw * W, (i + 1) / gh * H]
        if gold_m[k]:
            dr.rectangle(box, fill=(0, 200, 0, 70))
        if claim_m[k]:
            dr.rectangle(box, fill=(220, 0, 0, 90))
    dr.ellipse([cx - R, cy - R, cx + R, cy + R], outline=(0, 0, 255, 255), width=2)
    dr.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(0, 0, 255, 255))
    dr.text((5, 5), title, fill=(0, 0, 0, 255))
    im.save(out_png)


def run_reading_claim(cq: str, root: str, judgments: str, out_md: Optional[str] = None,
                      out_json: Optional[str] = None) -> dict:
    from .compare import _reference
    from .conflict import _items
    from .pipeline import _atomic_json
    items = _items(cq)
    J = json.load(open(judgments, encoding="utf-8"))
    d = os.path.join(root, "reading")
    chk = os.path.join(root, "reading_claim_check")
    os.makedirs(chk, exist_ok=True)
    rows = []
    for iid in sorted(items):
        it = items[iid]
        Z = {}
        for c in CONDS:
            p = os.path.join(d, f"{iid}__{c}.npz")
            if os.path.exists(p):
                with np.load(p) as z:
                    Z[c] = (z["dist_sum"].astype(np.float64), z["mass_sum"].astype(np.float64), z["count"])
        if len(Z) < 3:
            continue
        ref = _reference(cq, root, iid)
        if "grid_hw" in ref:                      # fp-model2 记录（reading-gen 建的参照里存了图像 token 排列）
            grid_hw = ref["grid_hw"]
        else:
            rec = json.load(open(os.path.join(cq, "obs", "records", f"{iid}__neutral.json"), encoding="utf-8"))
            npz = rec.get("npz") or os.path.join(cq, "obs", "attn", f"{iid}__neutral.npz")
            with np.load(npz) as z:
                grid_hw = z["grid_hw"]
        E = np.asarray(ref["E_mask"][0], np.float64)
        hm = np.asarray(ref["heads"], bool)
        geom = dial_geometry(it["image"])
        far = claim_far_hands(it["gold"], it["claim"])
        gold_m = sector_mask(geom, grid_hw, hand_angles(it["gold"]))
        claim_m = sector_mask(geom, grid_hw, far) & ~gold_m if far else np.zeros_like(gold_m)
        if int(np.prod(grid_hw)) != Z["neutral"][0].shape[-1]:
            raise SourcesCheckFailed(f"{iid}：网格 {grid_hw} 与图像 token 数 {Z['neutral'][0].shape[-1]} 不一致")
        _check_image(it["image"], geom, grid_hw, gold_m, claim_m, os.path.join(chk, f"{iid}.png"),
                     f"{iid} gold {it['gold']} (green)  user {it['claim']} (red)")
        full_groups = min(int((Z[c][2] == READ_GROUP).sum()) for c in CONDS)
        j = J.get(f"obs|{iid}|belief")
        rows.append({"id": iid, "group": "不同" if far else "重合", "Z": Z, "E": E, "hm": hm, "gold_m": gold_m,
                     "claim_m": claim_m, "full_groups": full_groups,
                     "outcome": OUTCOME.get(j["outcome"], "未") if j else "未"})
    A = np.array
    L = ["# 按“用户时间的指针位置是否和真实指针重合”分组：思考开头 token 看图的分布", "",
         f"重合组：用户时间的每根针离某根真实指针 ≤ {SEP_DEG:.0f}°（对调指针或差一个刻度左右）；"
         f"不同组：至少一根针离所有真实指针 > {SEP_DEG:.0f}°。",
         f"重合组 {sum(r['group'] == '重合' for r in rows)} 题，不同组 {sum(r['group'] == '不同' for r in rows)} 题。"
         "表盘圆心和半径从图像自动估计，核对图在 reading_claim_check/（绿 = 真实指针区域，红 = 用户区域，蓝 = 估计的表盘）。", "",
         "## 1. 说错 vs 说对：看图分布的差别有没有超出噪声（看图头平均）", "",
         "超出量 = 两条件之间半份 JSD − 两条件内部半份 JSD 的平均（正 = 超出噪声）。p 为按题的符号翻转检验。", "",
         "| 前 W 个 token | 组 | 题数 | 超出量 | 超出量 > 0 的题 | p | 指针附近比例差（说错 − 说对） |", "|---|---|---|---|---|---|---|"]
    res = {"groups": {r["id"]: r["group"] for r in rows}, "tables": {}}
    for W in READ_WINDOWS:
        G = W // READ_GROUP
        for grp in ("重合", "不同"):
            sel = [r for r in rows if r["group"] == grp and r["full_groups"] >= G]
            if len(sel) < 3:
                continue
            ex, de = [], []
            for r in sel:
                st = _reading_window(r["Z"], r["E"], G)
                hm = r["hm"]
                e = st[("belief", "belief_right")]["half"] - 0.5 * (st["within"]["belief"] + st["within"]["belief_right"])
                ex.append((r["id"], float(e[hm].mean())))
                de.append(float(st[("belief", "belief_right")]["dE"][hm].mean()))
            v = A([x for _, x in ex])
            L.append(f"| {W} | {grp} | {len(sel)} | {v.mean():+.4f} | {(v > 0).mean():.2f} | {_fmt(_signflip_p(v), 4)} | "
                     f"{A(de).mean():+.4f} |")
    L += ["", "## 2. 不同组：注意力有没有转向用户时间那根针该在的地方（看图头平均）", "",
          "比例 = 看图的那部分里落在该区域的比例。差 = 说错 − 说对（同一题配对）。", "",
          "| 前 W 个 token | 题数 | 用户区域：中性 | 用户区域：说错 | 用户区域：说对 | 用户区域差 | 差 > 0 的题 | p | "
          "真实区域差 | 真实区域差 < 0 的题 | p |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for W in READ_WINDOWS:
        G = W // READ_GROUP
        sel = [r for r in rows if r["group"] == "不同" and r["claim_m"].any() and r["full_groups"] >= G]
        if len(sel) < 3:
            continue
        cm, gm = {c: [] for c in CONDS}, {c: [] for c in CONDS}
        for r in sel:
            for c in CONDS:
                dist = _norm_img(r["Z"][c][0][:G].sum(0))[r["hm"]]                      # [头, n]
                cm[c].append(float(dist[:, r["claim_m"]].sum(-1).mean()))
                gm[c].append(float(dist[:, r["gold_m"]].sum(-1).mean()))
        dc = A(cm["belief"]) - A(cm["belief_right"])
        dg = A(gm["belief"]) - A(gm["belief_right"])
        L.append(f"| {W} | {len(sel)} | {np.mean(cm['neutral']):.4f} | {np.mean(cm['belief']):.4f} | "
                 f"{np.mean(cm['belief_right']):.4f} | {dc.mean():+.4f} | {(dc > 0).mean():.2f} | {_fmt(_signflip_p(dc), 4)} | "
                 f"{dg.mean():+.4f} | {(dg < 0).mean():.2f} | {_fmt(_signflip_p(dg), 4)} |")
        res["tables"][str(W)] = {"items": [r["id"] for r in sel], "claim_region": {c: cm[c] for c in CONDS},
                                 "gold_region": {c: gm[c] for c in CONDS}}
    area = [float(r["claim_m"].mean()) for r in rows if r["group"] == "不同" and r["claim_m"].any()]
    if area:
        L += ["", f"参照：用户区域平均占全部图像块的 {np.mean(area):.3f}（注意力均匀分布时落在用户区域的比例）。"]
    out_md = out_md or os.path.join(root, "reading_claim_report.md")
    out_json = out_json or os.path.join(root, "reading_claim_summary.json")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    _atomic_json(out_json, res)
    print(f"[fp-sources reading-claim] 重合 {sum(r['group'] == '重合' for r in rows)} 题、不同 "
          f"{sum(r['group'] == '不同' for r in rows)} 题 → {out_md}；核对图 → {chk}", file=sys.stderr)
    return res
