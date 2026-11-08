# -*- coding: utf-8 -*-
"""对照组：只在文本侧让思考看不见提示里的用户那句话，不碰图像注意力（ClockQA，20 题）。

mask 组：生成思考的前 MASK_W 个词时，每个标准注意力层里，当前词对"用户那句话"那几个 token 的注意力置零后重新归一化
（图像和其他位置按原比例分走这部分注意力，没有任何人工放大）。之后照常生成。
base 组：不改。两组同一随机种子。
注意：只能改标准注意力层。Qwen 的 48 个线性注意力层在读入提示时已经把那句话混进了状态；提示里那句话之后的模板 token
在读入时也看过它。这两条路都挡不住，记录里用"开头 128 个词有没有写出用户的时间"检查漏了多少。
记录：前 REC_W 个词的看图总量、看图分布（与中性回答前 REC_W 个词比较：关键区份额、JSD），第一次读数，最终答案。
关键区 = 中性回答（原模型中性条件的那条回答）前 REC_W 个词里注意力最高的 20% 图块（与 fp-refbranch 相同）。
"""

import json
import os
import re
import sys
import time
from typing import Dict, List, Optional

import numpy as np

MASK_W = 128
REC_W = 64
ARMS = ["base", "mask"]


def first_reading(text: str, gold: str, claim: Optional[str]) -> dict:
    """思考里模型自己写出的第一个时间（所在句子不是在说用户、不是假设句），以及它对不对。"""
    from .attrib import _T, context_of
    from .conflict import clock_match
    for m in _T.finditer(text or ""):
        if context_of(text, m.start()) != "own":
            continue
        t = m.group(0)
        if clock_match(t, gold) in ("exact", "nosec"):
            kind = "对"
        elif claim and clock_match(t, claim) in ("exact", "nosec", "hm"):
            kind = "用户的时间"
        else:
            kind = "其他"
        return {"time": t, "kind": kind, "char": m.start()}
    return {"time": None, "kind": "无读数", "char": None}


def claim_span(dec, in_ids: List[int], msg: str, question: str) -> List[int]:
    """用户那句话在提示里的 token 位置（与 attrib.py 的做法相同）。"""
    from .attrib import char_spans
    if not msg.startswith(question):
        raise RuntimeError("用户消息不是“问题 + 一句话”的格式，分不出用户那句话")
    sent = msg[len(question):].strip()
    ptext, ps, pe = char_spans(dec, in_ids)
    at = ptext.rfind(sent)
    if at < 0:
        raise RuntimeError(f"在提示里找不到用户那句话：{sent!r}")
    return [i for i in range(len(in_ids)) if ps[i] < at + len(sent) and pe[i] > at]


class MaskController:
    """attn 包装的记录器：mask 组在前 MASK_W 步把对 span 的注意力置零；两组都记录前 REC_W 步的看图注意力。"""

    def __init__(self, torch, arm: str, n_in: int, span: List[int], image_idx, layers: List[int], chunk: int = 256):
        self.torch, self.arm, self.n_in, self.layers, self.chunk = torch, arm, n_in, list(layers), chunk
        self.span = torch.as_tensor(span, dtype=torch.long)
        self.image_idx = image_idx
        self.selfcheck_left, self.selfcheck_diffs = 0, []
        self.k = 0
        self.prefilled = False
        self.dist = np.zeros((REC_W, len(self.layers), int(image_idx.numel())))
        self.mass = np.zeros((REC_W, len(self.layers)))
        self.span_before = np.zeros((MASK_W, len(self.layers)))      # 置零前，当前词分给用户那句话的注意力（头平均）
        self._rel = {}

    @property
    def passthrough(self):
        return self.arm == "base" or self.k > MASK_W

    @property
    def rel_checked(self):
        return self._rel

    def stages_for(self, q_pos):
        return self.torch.full_like(q_pos, -1)

    def _keys(self, idx, kL, device):
        off = (self.n_in + self.k - 1) + 1 - kL              # 当前查询的绝对位置 + 1 − 键的个数（滑动窗口层可能只留最近一段）
        k = idx.to(device) - off
        return k[k >= 0]

    def modify(self, layer_idx, p, q_pos):
        if p.shape[2] != 1 or self.k < 1 or self.k > MASK_W or layer_idx not in self.layers:
            return p
        s = self._keys(self.span, p.shape[-1], p.device)
        if s.numel() == 0:
            return p
        li = self.layers.index(layer_idx)
        self.span_before[self.k - 1, li] = float(p[0, :, 0].index_select(-1, s).sum(-1).mean())
        if self.arm != "mask":
            return p
        p = p.clone()
        p[..., s] = 0
        return p / p.sum(-1, keepdim=True).clamp_min(1e-12)

    def accumulate(self, layer_idx, probs, stages, q_pos):
        if probs.shape[2] != 1 or self.k < 1 or self.k > REC_W or layer_idx not in self.layers:
            return
        img = self._keys(self.image_idx, probs.shape[-1], probs.device)
        if img.numel() != self.image_idx.numel():
            return
        li = self.layers.index(layer_idx)
        P = probs[0, :, 0].index_select(-1, img).float()
        self.dist[self.k - 1, li] = P.sum(0).cpu().numpy()
        self.mass[self.k - 1, li] = float(P.sum(-1).mean())

    def pre_hook(self, module, args, kwargs):
        from .. import attn
        ids = kwargs.get("input_ids", args[0] if args else None)
        if not self.prefilled:
            self.prefilled = True
            attn._ACTIVE = None
            return
        self.k += 1                                        # 这一步预测第 k 个输出 token（输入是第 k−1 个）
        attn._ACTIVE = self if self.k <= MASK_W else None


def generate(backend, cfg, item: dict, cond: str, arm: str, seed: int = 42) -> dict:
    from .. import attn
    from .generate import build_inputs, make_budget
    torch, tf, model, tok = backend.torch, backend.transformers, backend.model, backend.tokenizer
    targets = attn.find_targets(model)
    layers = sorted(int(m.layer_idx) for _, m in targets)
    sliding = {int(m.layer_idx): bool(getattr(m, "is_sliding", False)) for _, m in targets}
    attn.install(model, verbose=False)
    prompt, inputs = build_inputs(backend, item["image"], item["messages"][cond])
    ids = inputs["input_ids"][0].tolist()
    dec = lambda x: tok.decode(x, skip_special_tokens=False)
    msg = item["messages"][cond]
    q = item["messages"]["neutral"] if msg.startswith(item["messages"]["neutral"]) else item["question"]
    span = claim_span(dec, ids, msg, q)
    img_idx = torch.tensor([i for i, x in enumerate(ids) if x == attn.image_token_id(backend)])
    dev = model.device
    batch = {k: v.to(dev) for k, v in inputs.items() if hasattr(v, "to")}
    if "pixel_values" in batch:
        batch["pixel_values"] = batch["pixel_values"].to(model.dtype)
    n_in = len(ids)
    gc = model.generation_config
    eos = gc.eos_token_id if isinstance(gc.eos_token_id, list) else [gc.eos_token_id]
    budget = make_budget(tf, torch, n_in, backend.close_id, int(eos[0]), cfg.think_budget, cfg.answer_budget)
    ctrl = MaskController(torch, arm, n_in, span, img_idx, layers, chunk=cfg.attn_chunk)
    hook = model.register_forward_pre_hook(ctrl.pre_hook, with_kwargs=True)
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "rope_deltas"):
        inner.rope_deltas = None
    tf.set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    t0 = time.time()
    try:
        with torch.inference_mode():
            out = model.generate(**batch, do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p, top_k=cfg.top_k,
                                 min_p=cfg.min_p, repetition_penalty=cfg.repetition_penalty,
                                 max_new_tokens=cfg.think_budget + cfg.answer_budget + 2, use_cache=True,
                                 logits_processor=tf.LogitsProcessorList([budget]))
    finally:
        hook.remove()
        attn._ACTIVE = None
    output = out[0, n_in:].tolist()
    close = output.index(backend.close_id) if backend.close_id in output else None
    thinking = tok.decode(output[:close] if close is not None else output, skip_special_tokens=True).strip()
    answer = tok.decode(output[close + 1:], skip_special_tokens=True).strip() if close is not None else ""
    head = tok.decode(output[:MASK_W], skip_special_tokens=True)
    return {"arm": arm, "seed": seed, "output_ids": output, "raw_output": tok.decode(output, skip_special_tokens=False),
            "thinking": thinking, "answer": answer, "n_in": n_in, "n_out": len(output), "seconds": round(time.time() - t0, 2),
            "span_tokens": len(span), "span_text": dec([ids[i] for i in span]), "layers": layers,
            "sliding": [sliding[l] for l in layers], "head_text": head,
            "_dist": ctrl.dist, "_mass": ctrl.mass, "span_attention_before": ctrl.span_before.mean(1).round(5).tolist(),
            "explicit_vs_original": ctrl._rel or None}


def neutral_reference(backend, cfg, item: dict, nrec: dict):
    """中性回答前 REC_W 个词的看图注意力（逐层，头合计）。"""
    from .. import attn
    from .entropy import _batch
    from .generate import build_inputs
    from .ref_branch import BranchRecorder
    torch, model = backend.torch, backend.model
    layers = sorted(int(m.layer_idx) for _, m in attn.find_targets(model))
    attn.install(model, verbose=False)
    _, inp = build_inputs(backend, item["image"], item["messages"]["neutral"])
    ids = inp["input_ids"][0].tolist()
    img_idx = torch.tensor([i for i, x in enumerate(ids) if x == attn.image_token_id(backend)])
    r = BranchRecorder(torch, REC_W, len(ids), img_idx, layers, chunk=cfg.attn_chunk)
    r.passthrough, r.rel_checked = True, {}
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "rope_deltas"):
        inner.rope_deltas = None
    out = [int(x) for x in nrec["output_ids"]][:REC_W]
    with torch.inference_mode(), attn.recording(r):
        try:
            model(**_batch(backend, inp, out), use_cache=False, return_dict=True, logits_to_keep=1)
        except TypeError:
            model(**_batch(backend, inp, out), use_cache=False, return_dict=True)
    return r.dist, r.mass


def stats(dist, mass, lm, ref_dist, frac: float = 0.2) -> dict:
    """前 REC_W 步、选定的层：看图总量、关键区份额（中性回答注意力最高的 20% 图块）、与中性回答的 JSD。"""
    from .ref_branch import _jsd
    d = dist[:, lm].sum((0, 1))
    d = d / max(d.sum(), 1e-12)
    r = ref_dist[:, lm].sum((0, 1))
    r = r / max(r.sum(), 1e-12)
    E = np.argsort(-r, kind="stable")[:max(1, int(round(len(r) * frac)))]
    return {"mass": float(mass[:, lm].mean()), "E": float(d[E].sum()), "E_neutral": float(r[E].sum()), "jsd": _jsd(d, r)}


def run(cfg, root: str, ent_dir: str, model_key: str, image_dirs: Optional[List[str]] = None, shard: str = "",
        seed: int = 42, ids: Optional[List[str]] = None) -> None:
    from .conflict import _engine, claim_of
    from .pipeline import _shard
    from .ref_branch import _sha
    from .trigger_regions import load_entropy, resolve_image, resolve_source
    from .valley_inject import CONDS, selection
    from .verify import final_time, outcome
    os.makedirs(os.path.join(root, "records"), exist_ok=True)
    ent = load_entropy(ent_dir, model_key, ["ClockQA"])
    sel = ids or selection(ent)["ClockQA"]
    with open(os.path.join(root, "selection.json"), "w", encoding="utf-8") as f:
        json.dump(sel, f, ensure_ascii=False, indent=1)
    tasks = _shard([(i, c, a) for i in sel for c in CONDS for a in ARMS], shard)
    todo = [x for x in tasks if not os.path.exists(os.path.join(root, "records", f"{x[0]}__{x[1]}__{x[2]}.json"))]
    print(f"[fp-maskq] 本进程 {len(tasks)} 项，待做 {len(todo)} 项", file=sys.stderr)
    eng, nref = None, {}
    for n, (iid, c, arm) in enumerate(todo):
        e = ent[(iid, c)]
        job = e["job"]
        item = dict(job["item"])
        orig = json.load(open(resolve_source(job), encoding="utf-8"))
        item["image"] = resolve_image(job, orig, image_dirs)
        if eng is None:
            eng = _engine(cfg)
        b = eng.backend
        if iid not in nref:
            npath = job.get("neutral_source")
            if not npath or not os.path.exists(npath) or _sha(npath) != job.get("neutral_sha256"):
                raise SystemExit(f"{iid}：中性回答 {npath} 不存在或 SHA 不一致")
            nref[iid] = neutral_reference(b, cfg, item, json.load(open(npath, encoding="utf-8")))
        rec = generate(b, cfg, item, c, arm, seed=seed)
        claim = claim_of(item, c)
        dist, mass = rec.pop("_dist"), rec.pop("_mass")
        sl = np.array(rec["sliding"])
        kinds = {"全局层": ~sl, "滑动窗口层": sl} if sl.any() else {"全部层": np.ones(len(sl), bool)}
        rec["attention"] = {k: stats(dist, mass, lm, nref[iid][0]) for k, lm in kinds.items()}
        rec.update(id=iid, dataset="ClockQA", condition=c, group=e["group"], gold=item["gold"], claim=claim,
                   outcome=outcome(rec["answer"], item["gold"], claim), final_time=final_time(rec["answer"], claim),
                   first_reading=first_reading(rec["thinking"], item["gold"], item.get("claim")),
                   claim_in_head=bool(claim and re.search(re.escape(":".join(claim.split(":")[:2])), rec["head_text"])))
        p = os.path.join(root, "records", f"{iid}__{c}__{arm}.json")
        with open(p + ".tmp", "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        os.replace(p + ".tmp", p)
        at = "；".join(f"{k} 看图 {v['mass'] * 100:.1f}% 关键区 {v['E'] * 100:.1f}%" for k, v in rec["attention"].items())
        print(f"  {n + 1}/{len(todo)} {iid}__{c} {arm}：{rec['n_out']} token，{rec['seconds']}s，用户那句话 {rec['span_tokens']} 个 token，"
              f"开头写出用户时间 {rec['claim_in_head']}，第一次读数 {rec['first_reading']['time']}（{rec['first_reading']['kind']}），"
              f"{rec['outcome']}；{at}", file=sys.stderr)


def summary(root: str, label: str = "") -> None:
    d = os.path.join(root, "records")
    R = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".json"):
            r = json.load(open(os.path.join(d, f), encoding="utf-8"))
            R[(r["id"], r["condition"], r["arm"])] = r
    cname = {"belief": "用户说错", "belief_right": "用户说对"}
    gname = {"flip": "被带偏", "hold": "守住"}
    L = [f"# 只让思考看不见用户那句话（不碰图像注意力）· {label}", "",
         f"- mask：生成思考的前 {MASK_W} 个词时，标准注意力层里当前词对用户那句话的注意力置零后重新归一化；之后照常生成；",
         "- base：不改；两组同一随机种子（42）；",
         f"- 看图总量、关键区份额：前 {REC_W} 个词的平均；关键区 = 中性回答前 {REC_W} 个词注意力最高的 20% 图块；",
         "- 第一次读数：思考里模型自己写出的第一个时间（不在说用户的句子里、不是假设句）；",
         f"- 开头写出用户的时间：前 {MASK_W} 个词里出现用户给的时:分（mask 组若出现，说明信息从挡不住的路漏了进来）。", "",
         "## 注意力（前 64 个词，按组平均）", "",
         "| 条件 | 组 | 层 | 题数 | 看图总量 base → mask | 关键区份额 base → mask（中性回答自己） | 与中性回答的 JSD base → mask |", "|---|---|---|---:|---|---|---|"]
    for c in ["belief", "belief_right"]:
        for g in ["flip", "hold"]:
            ks = [k for k in R if k[1] == c and k[2] == "base" and R[k]["group"] == g and (k[0], c, "mask") in R]
            if not ks:
                continue
            for kind in R[ks[0]]["attention"]:
                f = lambda arm, x: float(np.mean([R[(k[0], c, arm)]["attention"][kind][x] for k in ks]))
                L.append(f"| {cname[c]} | {gname[g]} | {kind} | {len(ks)} | {f('base', 'mass') * 100:.1f}% → {f('mask', 'mass') * 100:.1f}% | "
                         f"{f('base', 'E') * 100:.1f}% → {f('mask', 'E') * 100:.1f}%（{f('base', 'E_neutral') * 100:.1f}%） | "
                         f"{f('base', 'jsd'):.4f} → {f('mask', 'jsd'):.4f} |")
    L += ["", "## 读数与答案", "",
          "| 条件 | 组 | 题数 | 开头写出用户时间 base / mask | 第一次读数对 base / mask | 第一次读数是用户的时间 base / mask | 最终答对 base / mask | 最终照用户说的 base / mask |",
          "|---|---|---:|---|---|---|---|---|"]
    for c in ["belief", "belief_right"]:
        for g in ["flip", "hold"]:
            ks = [k for k in R if k[1] == c and k[2] == "base" and R[k]["group"] == g and (k[0], c, "mask") in R]
            if not ks:
                continue
            cnt = lambda arm, fn: sum(bool(fn(R[(k[0], c, arm)])) for k in ks)
            L.append(f"| {cname[c]} | {gname[g]} | {len(ks)} | {cnt('base', lambda r: r['claim_in_head'])} / {cnt('mask', lambda r: r['claim_in_head'])} | "
                     f"{cnt('base', lambda r: r['first_reading']['kind'] == '对')} / {cnt('mask', lambda r: r['first_reading']['kind'] == '对')} | "
                     f"{cnt('base', lambda r: r['first_reading']['kind'] == '用户的时间')} / {cnt('mask', lambda r: r['first_reading']['kind'] == '用户的时间')} | "
                     f"{cnt('base', lambda r: r['outcome'] == '坚持（答对）')} / {cnt('mask', lambda r: r['outcome'] == '坚持（答对）')} | "
                     f"{cnt('base', lambda r: r['outcome'] == '采纳')} / {cnt('mask', lambda r: r['outcome'] == '采纳')} |")
    L += ["", "## mask 组开头 200 个字（用户说错）", ""]
    for k in sorted(R):
        if k[1] == "belief" and k[2] == "mask":
            L.append(f"- {k[0]}（{gname.get(R[k]['group'], R[k]['group'])}）：{R[k]['head_text'][:200]!r}")
    with open(os.path.join(root, "maskq_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"[fp-maskq] {len(R)} 条 → {root}/maskq_report.md", file=sys.stderr)
