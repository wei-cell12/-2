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
                 chunk: int = 256, selfcheck: int = 2):
        self.torch = torch
        self.chunk = chunk
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
        if layers != LAYERS:
            raise SourcesCheckFailed(f"记录到的层 {layers} 与标准注意力层 {LAYERS} 不一致")
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
