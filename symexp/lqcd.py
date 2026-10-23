# -*- coding: utf-8 -*-
"""LQCD / ITSM 核心复现：中性问题与诱导问题两条分支的对比解码。

论文：Sycophancy in Vision-Language Models: A Systematic Analysis and an Inference-Time Mitigation
Framework（arXiv 2408.11261）。每生成一个 token：

  新 logits = (1 + α) · 中性 logits − α · 诱导 logits
  α = α0 + λ · JSD(p_诱导 ‖ p_中性)                         （λ = 0 时就是固定 α）
  只保留 p_中性(y) ≥ β · max p_中性 的候选，β = β0 + μ · (1 − H(p_中性) / log|V|)

选出的 token 同时追加到两条分支。论文的情感缩放项（QSS）按其公式会被归一化抵消，不实现。
论文没有给出 α0、λ、β0、μ 的取值和去诱导提示，默认值沿用对比解码的常用设置（α = 1、β = 0.1），
请在论文里写明。中性问题默认用规则拆分得到的原问题（相当于理想的去诱导改写）。
"""

import math
from typing import Dict, List, Optional, Tuple


def contrast_numpy(ln, ll, alpha0: float, lam: float, beta0: float, mu: float):
    """numpy 参考实现（测试用）。返回 (新 logits, α, β, JSD)。"""
    import numpy as np
    ln = np.asarray(ln, dtype=np.float64)
    ll = np.asarray(ll, dtype=np.float64)

    def lsm(x):
        x = x - x.max()
        return x - np.log(np.exp(x).sum())

    lpn, lpl = lsm(ln), lsm(ll)
    pn, pl = np.exp(lpn), np.exp(lpl)
    m = 0.5 * (pn + pl)
    lm = np.log(np.maximum(m, 1e-300))
    jsd = 0.5 * (pn * (lpn - lm)).sum() + 0.5 * (pl * (lpl - lm)).sum()
    alpha = alpha0 + lam * jsd
    cd = (1 + alpha) * ln - alpha * ll
    ent = -(pn * lpn).sum()
    beta = min(1.0, max(0.0, beta0 + mu * (1 - ent / math.log(len(ln)))))
    cd = np.where(pn >= beta * pn.max(), cd, -np.inf)
    return cd, float(alpha), float(beta), float(jsd)


def contrast_torch(torch, ln, ll, alpha0: float, lam: float, beta0: float, mu: float):
    """与 contrast_numpy 相同的计算，输入为一维 logits 张量。"""
    ln = ln.float()
    ll = ll.float()
    lpn = torch.log_softmax(ln, -1)
    lpl = torch.log_softmax(ll, -1)
    pn, pl = lpn.exp(), lpl.exp()
    lm = torch.log((0.5 * (pn + pl)).clamp_min(1e-30))
    jsd = 0.5 * (pn * (lpn - lm)).sum() + 0.5 * (pl * (lpl - lm)).sum()
    alpha = alpha0 + lam * float(jsd)
    cd = (1 + alpha) * ln - alpha * ll
    ent = float(-(pn * lpn).sum())
    beta = min(1.0, max(0.0, beta0 + mu * (1 - ent / math.log(ln.numel()))))
    cd = cd.masked_fill(pn < beta * pn.max(), float("-inf"))
    return cd, alpha, beta, float(jsd)


def sample_torch(torch, scores, temperature: float, top_k: int, top_p: float, min_p: float, greedy: bool):
    """与 transformers 的采样设置一致：温度 → top-k → top-p → min-p → 多项式采样。"""
    if greedy:
        return int(torch.argmax(scores))
    s = scores / max(temperature, 1e-6)
    if top_k and top_k > 0:
        kth = torch.topk(s, min(top_k, s.numel())).values[-1]
        s = s.masked_fill(s < kth, float("-inf"))
    if top_p is not None and top_p < 1.0:
        sv, si = torch.sort(s, descending=True)
        cp = torch.softmax(sv, -1).cumsum(-1)
        remove = cp - torch.softmax(sv, -1) > top_p     # 保留累计概率首次超过 top_p 的那个
        s[si[remove]] = float("-inf")
    if min_p and min_p > 0:
        p = torch.softmax(s, -1)
        s = s.masked_fill(p < min_p * p.max(), float("-inf"))
    p = torch.softmax(s, -1)
    return int(torch.multinomial(p, 1))


# ---------------------------------------------------------------------------------------------
def _rope_holders(model):
    return [o for o in (model, getattr(model, "model", None)) if o is not None and hasattr(o, "rope_deltas")]


def _get_rope(model):
    return [getattr(o, "rope_deltas") for o in _rope_holders(model)]


def _set_rope(model, vals):
    for o, v in zip(_rope_holders(model), vals):
        o.rope_deltas = v


class _Branch:
    def __init__(self, name):
        self.name = name
        self.pkv = None
        self.length = 0
        self.prompt_len = 0
        self.rope = None
        self.logits = None
        self.rec = None
        self.grid = None


def generate_lqcd(backend, req_lead, req_neu, alpha0=1.0, lam=0.0, beta0=0.1, mu=0.0, record=False):
    """HF 后端的两分支对比解码。返回 (GenResult, info)。info["attn"] = {"lead": ..., "neutral": ...}（record=True 时）。"""
    import time
    from .backends import GenResult
    from . import attn as A
    torch = backend.torch
    model, c = backend.model, backend.cfg
    t0 = time.time()
    if backend.close_id is None:
        raise RuntimeError("LQCD 需要 </think> 是单个 token（单次生成）")
    if record:
        A.install(model, verbose=False)
    img_tok = A.image_token_id(backend) if record else None

    eos = set()
    gcfg = getattr(model, "generation_config", None)
    e = getattr(gcfg, "eos_token_id", None)
    if isinstance(e, int):
        eos.add(e)
    elif isinstance(e, (list, tuple)):
        eos.update(int(x) for x in e)
    for mk in c.end_markers:
        tid = backend.tokenizer.convert_tokens_to_ids(mk)
        if isinstance(tid, int) and tid >= 0 and tid != backend.tokenizer.unk_token_id:
            eos.add(tid)

    def forward(br, **kw):
        _set_rope(model, br.rope if br.rope is not None else _get_rope(model))
        with A.recording(br.rec if record else None):
            try:
                out = model(**kw, use_cache=True, logits_to_keep=1)
            except TypeError:
                out = model(**kw, use_cache=True)
        br.rope = _get_rope(model)
        return out

    branches = []
    prompts = {}
    for name, req in (("lead", req_lead), ("neutral", req_neu)):
        br = _Branch(name)
        base = backend._with_think_open(backend._template(req, thinking=True))
        prompts[name] = base
        inputs = backend.processor(text=[base], images=req.images or None, return_tensors="pt").to(model.device)
        br.prompt_len = int(inputs["input_ids"].shape[1])
        if "image_grid_thw" in inputs:
            br.grid = [int(x) for x in inputs["image_grid_thw"][0].tolist()]
        if record:
            ids = inputs["input_ids"][0]
            image_idx = (ids == img_tok).nonzero().flatten().cpu()
            sm = torch.full((br.prompt_len,), -1, dtype=torch.long)
            sm[br.prompt_len - 1] = A.STAGE_ID["prefill_end"]
            br.rec = A.Recorder(torch, image_idx, sm, None)
        br.rope = None
        with torch.inference_mode():
            for o in _rope_holders(model):
                o.rope_deltas = None
            out = forward(br, **inputs)
        br.pkv = out.past_key_values
        br.logits = out.logits[0, -1]
        br.length = br.prompt_len
        branches.append(br)
    lead, neu = branches

    torch.manual_seed(req_lead.seed)
    gen: List[int] = []
    closed_at = None
    forced = False
    alphas, jsds = [], []
    tb, ab = req_lead.think_budget, req_lead.answer_budget
    with torch.inference_mode():
        while True:
            if closed_at is None and len(gen) >= tb:
                tok = backend.close_id
                forced = True
            else:
                cd, a, b, j = contrast_torch(torch, neu.logits, lead.logits, alpha0, lam, beta0, mu)
                alphas.append(a)
                jsds.append(j)
                tok = sample_torch(torch, cd, c.temperature, c.top_k, c.top_p, c.min_p, req_lead.greedy)
            gen.append(tok)
            if tok == backend.close_id and closed_at is None:
                closed_at = len(gen) - 1
            if tok in eos:
                break
            if closed_at is not None and len(gen) - closed_at - 1 >= ab:
                break
            stage = "answer" if closed_at is not None else "think"
            ids = torch.tensor([[tok]], device=model.device)
            for br in branches:
                if record:
                    br.rec.step_stage = A.STAGE_ID[stage]
                out = forward(br, input_ids=ids, past_key_values=br.pkv,
                              attention_mask=torch.ones(1, br.length + 1, dtype=torch.long, device=model.device),
                              cache_position=torch.tensor([br.length], device=model.device))
                br.pkv = out.past_key_values
                br.logits = out.logits[0, -1]
                br.length += 1
    txt = backend.tokenizer.decode(gen, skip_special_tokens=False)
    txt = backend._cut_end(txt)
    if c.think_close in txt:
        thinking, answer = txt.split(c.think_close, 1)
    else:
        thinking, answer = txt, ""
    res = GenResult(thinking.strip(), answer.strip(), closed_at is not None and not forced,
                    lead.prompt_len + neu.prompt_len, len(gen), time.time() - t0, prompts["lead"])
    info = {"alpha_mean": sum(alphas) / max(1, len(alphas)), "jsd_mean": sum(jsds) / max(1, len(jsds)),
            "n_tokens": len(gen), "forced_close": forced, "neutral_prompt": prompts["neutral"]}
    if record:
        info["attn"] = {}
        for br in branches:
            r = br.rec.result()
            ms = A.merge_size(backend)
            r.update(prompt_len=br.prompt_len, close_pos=-1 if closed_at is None else br.prompt_len + closed_at,
                     total_len=br.prompt_len + len(gen), merge_size=ms)
            if br.grid:
                t, h, w = br.grid
                r.update(grid_thw=[t, h, w], grid_hw=[h // ms, w // ms])
            info["attn"][br.name] = r
    return res, info
