# -*- coding: utf-8 -*-
"""某一段思考对前文的注意力（离线回放一条已生成的回答，不生成）。

给定一条回答记录（例如 fp-seg 的 records/*.json）和这段话的开头、结尾文字，回放"完整提示 + 思考"，
记录这段话里每个词（作为 query）在每个标准注意力层、所有头平均后，分给前文各处的注意力，按来源分组：
  序列第一个 token（注意力汇点）、图像、提示里用户那句话、提示里的其他部分（系统提示、问题、模板）、
  思考里的每一段（按空行分段），每段再分成"被隔离的熵差片段"和"其余文字"。
如果记录是隔离组（arm = seg），回放时按生成时的规则隔离：熵差片段写完之后（query 位置 ≥ 片段最后一个词 + 1），
对它的注意力权重乘以 0、不重新归一化。报告里的数字是**置零之前**的注意力（模型注意到了那里，但读不到内容）。
"""

import json
import os
import sys
from typing import List, Optional

import numpy as np

TOP = 25


def paragraphs(pieces: List[str], start: int, stop: int) -> List[int]:
    """[start, stop) 每个词所在的思考段落编号（从 1 起）；含空行的词算在它所在的那一段，下一个词起换段。"""
    out, p = [], 1
    for i in range(start, stop):
        out.append(p)
        if "\n\n" in pieces[i]:
            p += 1
    return out


def locate(pieces: List[str], start_text: str, end_text: str, lo: int = 0) -> range:
    """开头文字所在的第一个词，到结尾文字所在的最后一个词。"""
    text, offs = "", []
    for p in pieces:
        offs.append(len(text))
        text += p
    offs.append(len(text))
    a = text.find(start_text, offs[lo])
    if a < 0:
        raise SystemExit(f"找不到开头文字 {start_text!r}")
    z = text.find(end_text, a)
    if z < 0:
        raise SystemExit(f"开头之后找不到结尾文字 {end_text!r}")
    z += len(end_text)
    toks = [i for i in range(len(pieces)) if offs[i] < z and offs[i + 1] > a]
    return range(toks[0], toks[-1] + 1)


class SpanRecorder:
    """在 modify 里记录（置零之前的）注意力，再按隔离规则置零。shares[行, 层, 组]；total[层, 键] 累加。"""

    def __init__(self, torch, n_in: int, q_out: range, labels: np.ndarray, n_groups: int, layers: List[int],
                 segs: List[List[int]], isolate: bool, chunk: int = 256):
        self.torch, self.n_in, self.q_out, self.layers, self.chunk = torch, n_in, q_out, list(layers), chunk
        self.labels, self.G = labels, n_groups
        self.segs, self.isolate = segs, isolate
        self.selfcheck_left, self.selfcheck_diffs = 0, []
        self.passthrough = not isolate
        self.rel_checked = {}
        self.shares = np.zeros((len(q_out), len(layers), n_groups))
        self.keyw = np.zeros((len(q_out), len(labels)))          # 各层平均后，每个词对每个键的注意力
        self._oh = {}

    def stages_for(self, q_pos):
        return self.torch.full_like(q_pos, -1)

    def accumulate(self, layer_idx, probs, stages, q_pos):
        return

    def _onehot(self, kL, device):
        key = (kL, str(device))
        if key not in self._oh:
            t = self.torch
            lab = t.as_tensor(self.labels[:kL], device=device, dtype=t.long)
            oh = t.zeros(kL, self.G, device=device)
            oh[t.arange(kL, device=device), lab] = 1.0
            self._oh[key] = oh
        return self._oh[key]

    def modify(self, layer_idx, p, q_pos):
        t = self.torch
        if layer_idx in self.layers:
            outi = (q_pos - self.n_in).cpu().numpy()                     # query 是第几个输出词
            hit = np.flatnonzero((outi >= self.q_out.start) & (outi < self.q_out.stop))
            if len(hit):
                m = p[0][:, hit, :].float().mean(0)                      # [r, kL]
                li = self.layers.index(layer_idx)
                rows = outi[hit] - self.q_out.start
                self.shares[rows, li] = (m @ self._onehot(m.shape[-1], m.device)).cpu().numpy()
                self.keyw[rows, :m.shape[-1]] += m.cpu().numpy() / len(self.layers)
        if not self.isolate or not self.segs:
            return p
        kL = p.shape[-1]
        qo = q_pos - self.n_in                                           # [c]
        mask = t.zeros(len(q_pos), kL, dtype=t.bool, device=p.device)
        for s in self.segs:
            keys = t.tensor([self.n_in + i for i in s if self.n_in + i < kL], device=p.device, dtype=t.long)
            if keys.numel() == 0:
                continue
            on = (qo >= s[-1] + 1)
            if bool(on.any()):
                mask[on.nonzero().flatten()[:, None], keys[None, :]] = True
        if not bool(mask.any()):
            return p
        return p.masked_fill(mask[None, None], 0.0)


def run(cfg, root: str, record: str, ent_dir: str, model_key: str, start_text: str, end_text: str,
        image_dirs: Optional[List[str]] = None, name: str = "") -> None:
    from .. import attn
    from .conflict import _engine
    from .entropy import _batch
    from .generate import build_inputs
    from .mask_query import claim_span
    from .trigger_regions import load_entropy, resolve_image, resolve_source
    os.makedirs(root, exist_ok=True)
    rec = json.load(open(record, encoding="utf-8"))
    iid, c = rec["id"], rec["condition"]
    ent = load_entropy(ent_dir, model_key, [rec.get("dataset", "ClockQA")])
    job = ent[(iid, c)]["job"]
    item = job["item"]
    orig = json.load(open(resolve_source(job), encoding="utf-8"))
    image = resolve_image(job, orig, image_dirs)
    eng = _engine(cfg)
    b = eng.backend
    torch, tok = b.torch, b.tokenizer
    targets = attn.find_targets(b.model)
    layers = sorted(int(m.layer_idx) for _, m in targets)
    attn.install(b.model, verbose=False)
    out = [int(x) for x in rec["output_ids"]]
    close = out.index(b.close_id) if b.close_id in out else len(out)
    out = out[:close]
    pieces = [tok.decode([x], skip_special_tokens=False) for x in out]
    q_out = locate(pieces, start_text, end_text)
    msg = item["messages"][c]
    _, inp = build_inputs(b, image, msg)
    ids = inp["input_ids"][0].tolist()
    n_in = len(ids)
    dec = lambda x: tok.decode(x, skip_special_tokens=False)
    q = item["messages"]["neutral"] if msg.startswith(item["messages"]["neutral"]) else item["question"]
    span = claim_span(dec, ids, msg, q)
    img = [i for i, x in enumerate(ids) if x == attn.image_token_id(b)]
    segs = [list(s) for s in rec.get("segment_positions", [])]
    isolate = rec.get("arm") == "seg"
    par = paragraphs(pieces, 0, q_out.stop)
    n_par = max(par)
    names = ["序列第一个 token（注意力汇点）", "图像", "提示：用户那句话", "提示：其他（系统提示、问题、模板）"]
    for k in range(1, n_par + 1):
        names += [f"思考第 {k} 段：被隔离的熵差片段", f"思考第 {k} 段：其余文字"]
    labels = np.full(n_in + q_out.stop, 3, np.int64)
    labels[img] = 1
    labels[span] = 2
    labels[0] = 0
    seg_set = {i for s in segs for i in s}
    for i in range(q_out.stop):
        labels[n_in + i] = 4 + 2 * (par[i] - 1) + (0 if i in seg_set else 1)
    R = SpanRecorder(torch, n_in, q_out, labels, len(names), layers, segs, isolate, chunk=cfg.attn_chunk)
    inner = getattr(b.model, "model", None)
    if inner is not None and hasattr(inner, "rope_deltas"):
        inner.rope_deltas = None
    with torch.inference_mode(), attn.recording(R):
        try:
            b.model(**_batch(b, inp, out[:q_out.stop]), use_cache=False, return_dict=True, logits_to_keep=1)
        except TypeError:
            b.model(**_batch(b, inp, out[:q_out.stop]), use_cache=False, return_dict=True)
    third = np.array_split(np.arange(len(layers)), 3)
    avg = R.shares.mean((0, 1))
    by_third = [R.shares[:, th].mean((0, 1)) for th in third]
    ptext = {}
    for i in range(q_out.stop):
        ptext[par[i]] = ptext.get(par[i], "") + pieces[i]
    def keytext(k):
        if k < n_in:
            return "（图像）" if labels[k] == 1 else dec([ids[k]])
        return pieces[k - n_in]
    def ctx(k):
        if k < n_in:
            if labels[k] == 1:
                return "（图像 token）"
            return dec(ids[max(0, k - 6):k]) + "⟪" + dec([ids[k]]) + "⟫" + dec(ids[k + 1:k + 4])
        i = k - n_in
        return "".join(pieces[max(0, i - 6):i]) + "⟪" + pieces[i] + "⟫" + "".join(pieces[i + 1:i + 4])
    mean_key = R.keyw.mean(0)
    order = [int(k) for k in np.argsort(-mean_key) if k != 0][:TOP]
    per_tok = []
    for r, i in enumerate(q_out):
        kw = R.keyw[r]
        top = [int(k) for k in np.argsort(-kw) if k != 0][:5]
        per_tok.append({"token": pieces[i], "groups": {names[g]: float(R.shares[r, :, g].mean()) for g in range(len(names))},
                        "top": [{"pos": k, "group": names[labels[k]], "text": keytext(k), "context": ctx(k), "w": float(kw[k])} for k in top]})
    res = {"record": record, "id": iid, "condition": c, "arm": rec.get("arm"), "isolated_in_replay": isolate, "name": name,
           "span_text": "".join(pieces[i] for i in q_out), "span_tokens": [q_out.start, q_out.stop], "n_in": n_in,
           "layers": layers, "groups": names, "avg": avg.tolist(), "by_third": [x.tolist() for x in by_third],
           "paragraphs": {k: v for k, v in ptext.items()},
           "top_keys": [{"pos": k, "group": names[labels[k]], "text": keytext(k), "context": ctx(k), "w": float(mean_key[k])} for k in order],
           "per_token": per_tok, "rel_checked": R.rel_checked}
    fn = f"span_attn_{iid}_{c}_{rec.get('arm')}{('_' + name) if name else ''}"
    with open(os.path.join(root, fn + ".json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    report(os.path.join(root, fn + ".json"))


def report(path: str) -> None:
    v = json.load(open(path, encoding="utf-8"))
    names, avg, th = v["groups"], v["avg"], v["by_third"]
    L = [f"# {v['id']} · {v['condition']} · {v['arm']}：这一段对前文的注意力", "",
         f"这一段（第 {v['span_tokens'][0]}–{v['span_tokens'][1] - 1} 个输出词）：{v['span_text']!r}", "",
         "数字：这一段每个词、所有头平均、所有标准注意力层平均后，分给各处的注意力占全部注意力的百分比；"
         + ("回放时按生成时的规则隔离了熵差片段，表里是置零之前的注意力（注意到了，但读不到内容）。" if v["isolated_in_replay"] else "回放时没有隔离。"),
         "", "| 前文 | 全部层 | 前 1/3 层 | 中 1/3 层 | 后 1/3 层 |", "|---|---:|---:|---:|---:|"]
    for g, n in enumerate(names):
        if max(avg[g], *(x[g] for x in th)) < 0.0005:
            continue
        L.append(f"| {n} | {avg[g] * 100:.2f}% | " + " | ".join(f"{x[g] * 100:.2f}%" for x in th) + " |")
    L += ["", "## 思考各段原文", ""]
    for k, t in sorted(v["paragraphs"].items(), key=lambda x: int(x[0])):
        L.append(f"- 第 {k} 段：{t.strip()[:160]!r}")
    L += ["", f"## 注意力最多的 {len(v['top_keys'])} 个位置（不含序列第一个 token；这一段所有词平均）", "",
          "| 位置 | 属于 | 上下文 | 注意力 |", "|---:|---|---|---:|"]
    for x in v["top_keys"]:
        L.append(f"| {x['pos']} | {x['group']} | {x['context']!r} | {x['w'] * 100:.2f}% |")
    L += ["", "## 每个词注意力最多的 5 个位置（不含序列第一个 token）", ""]
    for x in v["per_token"]:
        L.append(f"- {x['token']!r}：" + "；".join(f"{y['context']!r}（{y['group']}，{y['w'] * 100:.1f}%）" for y in x["top"]))
    out = path[:-5] + ".md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"[fp-spanattn] → {out}", file=sys.stderr)
