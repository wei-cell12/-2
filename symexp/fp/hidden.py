# -*- coding: utf-8 -*-
"""犹豫方向（离线，不生成）：从隐藏状态里找“快要写出犹豫词”的方向，看它是否比数词更早、更灵敏。

两步：
  forward  （GPU）对已保存的轨迹，用生成时那份提示把“提示 + 已生成内容”整段前向一次（模型不做修改），
           取 8 层（每 8 层一层）的隐藏状态，只保留思考段，fp16 存到 work/hidden/ （留在服务器，不上传）。
           第 t 行 = 写第 t 个输出 token 之前那一刻的状态（查询位置 n_in−1+t）。
  analyze  （CPU）留一题交叉验证：对每道题，用其余题的轨迹求方向，再投影到这道题的轨迹上：
             犹豫位置 H   下一个 token 开始一个犹豫词（but / however / wait / hmm / actually / although /
                          though / alternatively / hold on / on the other hand / then again / unless）
             句首位置 C   下一个 token 开始一个新句子，且不是犹豫词
             方向 vs_all  = mean(H) − mean(全部思考位置)       （与 arXiv 2506.18167 的做法相同）
             方向 vs_sent = mean(H) − mean(C)                  （去掉“句子边界”本身的成分）
           输出每条轨迹逐 token 的投影 (h − 中心)·方向/|方向|，以及 H、C 的位置（可以上传）。
           统计检验（提前多久升高、去掉犹豫词附近后能否区分冲突与用户说对、漏检的轨迹）在本地做。

检查（不通过就停下）：
  - 重新构造的提示与记录里的 input_ids 一致
  - 同一次前向的 logits：观察轨迹（mode = observe）实际 token 排在前 20 名的比例 ≥ 0.98（防止位置错位）
  - hidden_states 层数 = 层数 + 1，隐藏状态没有非有限值
  - analyze 时每一折的 H、C 位置都不少于 20 个
"""

import glob
import json
import os
import re
import sys
from typing import Dict, List, Tuple

import numpy as np

HES_WORDS = r"but|however|wait|hmm+|actually|although|though|alternatively|hold on|on the other hand|then again|unless"
HES = re.compile(rf"\b({HES_WORDS})\b", re.I)
EVERY = 8
RANK_K = 20
RANK_MIN = 0.98
MIN_POS = 20
CHUNK = 256


class HiddenCheckFailed(RuntimeError):
    pass


# ---------------------------------------------------------------- 纯函数
def token_starts(pieces: List[str]) -> List[int]:
    out, pos = [], 0
    for p in pieces:
        out.append(pos)
        pos += len(p)
    return out


def positions(pieces: List[str], close: int) -> Tuple[List[int], List[int]]:
    """思考段里的犹豫位置 H 和普通句首位置 C（都是 token 下标 j：第 j 行 = 写第 j 个 token 之前）。"""
    pieces = pieces[:close]
    text = "".join(pieces)
    st = token_starts(pieces)
    tok_of = np.zeros(len(text) + 1, np.int64)
    for j, s in enumerate(st):
        tok_of[s:s + len(pieces[j])] = j

    def word_tok(ch):          # 字符 ch 所在的 token；如果 ch 是该 token 去掉前导空白后的第一个字符才算“开始”
        j = int(tok_of[ch])
        lead = len(pieces[j]) - len(pieces[j].lstrip())
        return j if ch == st[j] + lead else None
    H = sorted({j for m in HES.finditer(text) if (j := word_tok(m.start())) is not None})
    Hs = set(H)
    C = []
    for m in re.finditer(r"(^|[.!?\n])\s*([A-Za-z\"'(])", text):
        j = word_tok(m.start(2))
        if j is not None and j not in Hs and j > 0:
            C.append(j)
    return H, sorted(set(C))


def layer_ids(n_layers: int) -> List[int]:
    return list(range(EVERY, n_layers + 1, EVERY))


def fold_directions(sums: Dict[str, Dict[str, np.ndarray]], counts: Dict[str, Dict[str, int]], held: str):
    """sums[题号][类别] = [L, d] 的和（类别：H / C / all）；返回留出 held 后的两种方向（单位向量）和中心。"""
    tot = {k: sum(sums[q][k] for q in sums if q != held) for k in ("H", "C", "all")}
    n = {k: sum(counts[q][k] for q in counts if q != held) for k in ("H", "C", "all")}
    if n["H"] < MIN_POS or n["C"] < MIN_POS:
        raise HiddenCheckFailed(f"留出 {held} 后犹豫位置 {n['H']} 个、句首位置 {n['C']} 个，太少")
    mu = {k: tot[k] / n[k] for k in tot}
    out = {}
    for name, ref in (("vs_all", "all"), ("vs_sent", "C")):
        v = mu["H"] - mu[ref]
        norm = np.linalg.norm(v, axis=-1, keepdims=True)
        out[name] = (v / np.where(norm > 0, norm, 1), mu[ref])
    return out, n


# ---------------------------------------------------------------- GPU：整段前向
def forward_one(backend, item: dict, rec: dict) -> Tuple[dict, np.ndarray]:
    from .entropy import _batch
    from .generate import build_inputs
    torch, tok, model = backend.torch, backend.tokenizer, backend.model
    _, inp = build_inputs(backend, item["image"], item["messages"][rec["condition"]])
    if inp["input_ids"][0].tolist() != rec["input_ids"]:
        raise HiddenCheckFailed(f"{rec['id']}：重新构造的提示与记录里的 input_ids 不一致")
    out_ids = rec["output_ids"]
    n_in, n_out = len(rec["input_ids"]), len(out_ids)
    close = out_ids.index(backend.close_id) if backend.close_id in out_ids else n_out
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "rope_deltas"):
        inner.rope_deltas = None
    with torch.inference_mode():
        o = model(**_batch(backend, inp, out_ids), use_cache=False, return_dict=True, output_hidden_states=True)
    hs = o.hidden_states
    cfg = getattr(model.config, "text_config", model.config)
    nl = int(cfg.num_hidden_layers)
    if hs is None or len(hs) != nl + 1:
        raise HiddenCheckFailed(f"hidden_states 层数 {None if hs is None else len(hs)}，应为 {nl + 1}")
    L = layer_ids(nl)
    rows = slice(n_in - 1, n_in - 1 + close)
    H = torch.stack([hs[l][0, rows] for l in L], 1).float()                                  # [close, L, d]
    if not torch.isfinite(H).all():
        raise HiddenCheckFailed(f"{rec['id']}：隐藏状态有非有限值")
    lg = o.logits[0, n_in - 1: n_in - 1 + n_out]
    y = torch.tensor(out_ids, device=lg.device)
    rk = []
    for s in range(0, n_out, CHUNK):
        a = lg[s:s + CHUNK].float()
        rk.append((a > a.gather(1, y[s:s + CHUNK, None])).sum(-1))
    top = float((torch.cat(rk) < RANK_K).float().mean())
    if rec.get("mode") == "observe" and top < RANK_MIN:
        raise HiddenCheckFailed(f"{rec['id']}__{rec['condition']}：实际 token 进前 {RANK_K} 名的比例只有 {top:.3f}")
    pieces = [tok.decode([x], skip_special_tokens=False, clean_up_tokenization_spaces=False) for x in out_ids]
    Hpos, Cpos = positions(pieces, close)
    meta = {"id": rec["id"], "condition": rec["condition"], "mode": rec.get("mode"), "n_in": n_in, "n_out": n_out,
            "close": close, "layers": L, "d": int(H.shape[-1]), "in_top20": top, "pieces": pieces[:close],
            "H": Hpos, "C": Cpos}
    arr = H.cpu().numpy().astype(np.float16)
    del o, hs, H, lg
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return meta, arr


def _records(root: str, subs: List[str]) -> List[Tuple[str, str]]:
    out = []
    for s in subs:
        pat = os.path.join(root, "obs", "records", "*.json") if s == "obs" else os.path.join(root, "int", "*", "records", "*.json")
        for p in sorted(glob.glob(pat)):
            out.append((os.path.relpath(os.path.dirname(os.path.dirname(p)), root), p))
    return out


def _tag(root: str, sub: str, p: str) -> str:
    return f"{os.path.basename(root.rstrip('/'))}__{sub.replace('/', '_')}__{os.path.splitext(os.path.basename(p))[0]}"


def run_forward(cfg, roots: List[str], work: str, subs: List[str] = ("obs", "int"), shard: str = "") -> int:
    from .pipeline import Engine, _atomic_json, _check_no_failure, _fail, _shard, load_items
    _check_no_failure(work)
    todo = []
    for root in roots:
        items = {i["id"]: i for i in load_items(root)}
        for sub, p in _records(root, list(subs)):
            todo.append((root, items, sub, p))
    if not todo:
        raise SystemExit("没有找到任何记录")
    todo = _shard(todo, shard)
    hd = os.path.join(work, "hidden")
    os.makedirs(hd, exist_ok=True)
    left = [t for t in todo if not os.path.exists(os.path.join(hd, _tag(t[0], t[2], t[3]) + ".json"))]
    print(f"[fp-hidden forward] 本 worker {len(todo)} 条，待做 {len(left)} 条", file=sys.stderr)
    if not left:
        return 0
    eng = Engine(cfg)
    if eng.backend is None:
        raise SystemExit("fp-hidden 需要 hf 后端")
    for k, (root, items, sub, p) in enumerate(left):
        tag = _tag(root, sub, p)
        try:
            rec = json.load(open(p, encoding="utf-8"))
            meta, arr = forward_one(eng.backend, items[rec["id"]], rec)
            meta.update(source=sub, root=os.path.basename(root.rstrip("/")))
            np.save(os.path.join(hd, tag + ".npy"), arr)
            _atomic_json(os.path.join(hd, tag + ".json"), meta)          # 最后写 json，表示这一条完成
        except BaseException as e:
            _fail(work, f"hidden/{tag}", e)
            raise
        print(f"  {k + 1}/{len(left)} {tag} 思考段 {meta['close']} token  犹豫位置 {len(meta['H'])}  句首 {len(meta['C'])}  "
              f"top20={meta['in_top20']:.3f}  d={meta['d']}", file=sys.stderr)
    return len(left)


# ---------------------------------------------------------------- CPU：留一题求方向、投影
def run_analyze(work: str, out_path: str) -> dict:
    from .pipeline import _atomic_json, _check_no_failure
    _check_no_failure(work)
    hd = os.path.join(work, "hidden")
    metas = []
    for p in sorted(glob.glob(os.path.join(hd, "*.json"))):
        m = json.load(open(p, encoding="utf-8"))
        npy = p[:-5] + ".npy"
        if not os.path.exists(npy):
            raise HiddenCheckFailed(f"{npy} 不存在")
        m["_npy"] = npy
        metas.append(m)
    if not metas:
        raise SystemExit(f"{hd} 里没有结果，请先运行 --phase forward")
    if len({tuple(m["layers"]) for m in metas}) != 1:
        raise HiddenCheckFailed("各条轨迹取的层不一致")
    qs = sorted({m["id"] for m in metas})
    sums, counts = {}, {}
    for m in metas:                                       # 第一遍：每道题的 H / C / 全部 位置之和
        a = np.load(m["_npy"]).astype(np.float64)        # [close, L, d]
        s, c = sums.setdefault(m["id"], {}), counts.setdefault(m["id"], {"H": 0, "C": 0, "all": 0})
        for k, idx in (("H", m["H"]), ("C", m["C"]), ("all", list(range(len(a))))):
            if idx:
                s[k] = s.get(k, 0) + a[idx].sum(0)
                c[k] += len(idx)
        for k in ("H", "C", "all"):
            s.setdefault(k, np.zeros(a.shape[1:]))
    folds, info = {}, {}
    for q in qs:
        folds[q], n = fold_directions(sums, counts, q)
        info[q] = n
    cos = {}
    for name in ("vs_all", "vs_sent"):                   # 各折方向之间的一致程度（逐层平均余弦）
        U = np.stack([folds[q][name][0] for q in qs])     # [Q, L, d]
        cs = np.einsum("ald,bld->abl", U, U)
        iu = np.triu_indices(len(qs), 1)
        cos[name] = [round(float(x), 4) for x in cs[iu].mean(0)]
    rows = []
    for m in metas:                                       # 第二遍：用留出这道题的方向投影
        a = np.load(m["_npy"]).astype(np.float32)
        r = {k: m[k] for k in ("id", "condition", "mode", "source", "root", "n_out", "close", "layers", "in_top20",
                               "pieces", "H", "C")}
        for name in ("vs_all", "vs_sent"):
            u, mu = folds[m["id"]][name]
            pr = np.einsum("tld,ld->lt", a - mu[None].astype(np.float32), u.astype(np.float32))
            r["proj_" + name] = [[round(float(x), 3) for x in row] for row in pr]
        rows.append(r)
    out = {"work": work, "note": __doc__.split("\n\n")[0], "hes_words": HES_WORDS, "questions": qs,
           "fold_counts": info, "fold_cosine_by_layer": cos, "rows": rows}
    _atomic_json(out_path, out)
    print(f"[fp-hidden analyze] {len(rows)} 条轨迹，{len(qs)} 道题；各折方向逐层平均余弦 "
          f"vs_all={cos['vs_all']} vs_sent={cos['vs_sent']} → {out_path}", file=sys.stderr)
    return out
