# -*- coding: utf-8 -*-
"""推力分析（离线，不生成）：用已保存的记录重读一遍，在决定答案的那个 Yes/No token 上，
把 16 个标准注意力层的输出按来源分组，算每组把“Yes 减 No”的分差推向哪边。

记号（与说明一致）：
  位置 t = 目标 token（Yes/No）前面那个位置；它的输出决定目标 token。
  Δ = s_Yes − s_No（目标 token 的写法对应的那一对，如 " No" ↔ " Yes"）。
  每层每头：o_G = Σ_{j∈G} a_tj v_j；该层这一组的贡献 c_G = W_O (γ ⊙ o_G)，γ 为注意力输出门控（没有则为 1）。
  方向 d' = s ⊙ (u_Yes − u_No) / rms(h_t)，s 为最终 RMSNorm 的逐维系数；推力 P_G = Σ_层 Σ_头 c_G · d'。
  总注意力 A_G = Σ_层 Σ_头 Σ_{j∈G} a_tj；立场 S_G = P_G / A_G。
分组：E 关键区域（答阶段参照、逐层逐头）、I 其余图像、U 用户那句话（受压提示比中性提示多出的部分）、
      Q 其余提示（问题、模板）、T 思考（含 </think>）、R 回答里目标之前的部分（含位置 t 本身）。
每份都检查：各组之和 = 注意力原输出；门控重建 = o_proj 输入；归一化线性化 = lm_head 输入；
            由 logits 得到的 Δ = 由归一化输出得到的 Δ。任何一项不通过就停下（写 failures/ 由调用方负责）。
"""

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import LAYERS
from .references import build, check

GROUPS = ["E", "I", "U", "Q", "T", "R"]
SWAP = {"No": "Yes", "no": "yes", "NO": "YES", "Yes": "No", "yes": "no", "YES": "NO"}


class PushCheckFailed(RuntimeError):
    pass


# ---------------------------------------------------------------- 纯 Python 部分（可在没有 torch 时测试）
def claim_span(pressure_ids: List[int], neutral_ids: List[int]) -> Tuple[int, int]:
    """受压提示比中性提示多出来的那一段 [lo, hi)：最长公共前缀之后、最长公共后缀之前。"""
    n = min(len(pressure_ids), len(neutral_ids))
    lo = 0
    while lo < n and pressure_ids[lo] == neutral_ids[lo]:
        lo += 1
    s = 0
    while s < n - lo and pressure_ids[-1 - s] == neutral_ids[-1 - s]:
        s += 1
    hi = len(pressure_ids) - s
    if not 0 < lo < hi:
        raise PushCheckFailed(f"找不到用户那句话的位置（lo={lo}, hi={hi}）")
    return lo, hi


def find_target(decode, output_ids: List[int], close: int, after: str) -> int:
    """在回答里找目标 token 的下标 k（output_ids 内）：after 之后的第一个 yes/no 词。after 为空取回答里第一个。"""
    full = decode(output_ids[close + 1:])
    start = 0
    if after:
        i = full.find(after)
        if i < 0:
            raise PushCheckFailed(f"回答里找不到定位文字 {after!r}")
        start = i + len(after)
    for k in range(close + 1, len(output_ids)):
        if len(decode(output_ids[close + 1:k + 1])) <= start:      # 这个 token 结束在定位文字之前
            continue
        w = decode([output_ids[k]]).strip().strip("*").strip()
        if w.lower() in ("yes", "no"):
            return k
    raise PushCheckFailed(f"定位文字 {after!r} 之后没有 yes/no token")


def pair_ids(tok, target_id: int) -> Tuple[int, int, str]:
    """目标 token 和同样写法的对立词：返回 (yes_id, no_id, 目标写法)。"""
    s = tok.decode([target_id])
    core = s.strip().strip("*").strip()
    if core not in SWAP:
        raise PushCheckFailed(f"目标 token {s!r} 不是 yes/no")
    other = s.replace(core, SWAP[core])
    ids = tok.encode(other, add_special_tokens=False)
    if len(ids) != 1:
        raise PushCheckFailed(f"对立写法 {other!r} 不是单个 token：{ids}")
    yes_id, no_id = (target_id, ids[0]) if core.lower() == "yes" else (ids[0], target_id)
    return yes_id, no_id, s


# ---------------------------------------------------------------- 模型部分
class Probe:
    """装在 controller 的注意力包装上：prefill 时只记录最后一个查询位置的分组输出。"""

    def __init__(self, torch, groups_pos: Dict[str, object], E_answer: Dict[int, object], img_idx):
        self.torch = torch
        self.groups_pos = groups_pos          # 组名 -> LongTensor（CPU），E/I 之外的组
        self.E = E_answer                     # layer -> bool ndarray [H, n_img]
        self.img_idx = img_idx                # LongTensor（CPU）
        self.rec: Dict[int, dict] = {}

    def attention(self, orig, module, query, key, value, attention_mask, *args, **kwargs):
        t = self.torch
        out = orig(module, query, key, value, attention_mask, *args, **kwargs)
        if query.shape[2] == 1:
            raise PushCheckFailed("推力分析只做一次 prefill，不应出现逐 token 解码")
        layer = int(module.layer_idx)
        B, H, Q, D = query.shape
        rep = H // key.shape[1]
        k = key.repeat_interleave(rep, dim=1) if rep > 1 else key
        v = value.repeat_interleave(rep, dim=1) if rep > 1 else value
        scale = kwargs.get("scaling", args[1] if len(args) > 1 else None)
        if scale is None:
            scale = getattr(module, "scaling", D ** -0.5)
        sc = t.matmul(query[:, :, -1:].float(), k.float().transpose(-1, -2)) * scale
        if attention_mask is not None and attention_mask.dim() == 4:
            m = attention_mask[:, :, -1:, : k.shape[2]]
            sc = sc.masked_fill(~m, float("-inf")) if m.dtype == t.bool else sc + m.float()
        a = t.softmax(sc, dim=-1)[0, :, 0]                 # [H, K]
        V = v[0].float()                                   # [H, K, D]
        dev = a.device
        groups = {}
        img = self.img_idx.to(dev)
        E = t.as_tensor(self.E[layer], device=dev)         # [H, n]
        a_img, V_img = a.index_select(1, img), V.index_select(1, img)
        for name, mask in (("E", E), ("I", ~E)):
            w = a_img * mask
            groups[name] = (t.einsum("hn,hnd->hd", w, V_img), w.sum(-1))
        for name, pos in self.groups_pos.items():
            p = pos.to(dev)
            if p.numel() == 0:
                groups[name] = (t.zeros(H, D, device=dev), t.zeros(H, device=dev))
                continue
            ag = a.index_select(1, p)
            groups[name] = (t.einsum("hk,hkd->hd", ag, V.index_select(1, p)), ag.sum(-1))
        R = out[0][0, -1].float()                           # [H, D]，原注意力输出
        total = sum(g[0] for g in groups.values())
        allowed = t.clamp(R.abs() * 2.0 ** -6, min=0.05)
        ratio = float(((total - R).abs() / allowed).max())
        cover = float(sum(g[1] for g in groups.values()).sub(1).abs().max())
        self.rec[layer] = {"groups": groups, "R": R, "ratio": ratio, "cover": cover}
        return out


def _final_norm(model):
    found = []
    for name, m in model.named_modules():
        low = name.lower()
        if "visual" in low or "vision" in low or ".layers." in name:
            continue
        if name.split(".")[-1] == "norm" and "rmsnorm" in type(m).__name__.lower():
            found.append((name, m))
    if len(found) != 1:
        raise PushCheckFailed(f"找不到唯一的最终 RMSNorm：{[n for n, _ in found]}")
    return found[0][1]


def _gate(torch, module, h_last, H, D):
    """注意力输出门控：q_proj 输出是 2·H·D 时按 Qwen3-Next 的布局取后一半做 sigmoid；否则为 1。"""
    out = module.q_proj.out_features
    if out == H * D:
        return None
    if out == 2 * H * D:
        q = module.q_proj(h_last.to(module.q_proj.weight.dtype)).float().view(H, 2 * D)
        return torch.sigmoid(q[:, D:]).reshape(-1)
    raise PushCheckFailed(f"q_proj 输出维度 {out} 既不是 H·D 也不是 2·H·D，无法确定门控")


def analyze_case(backend, rec: dict, neutral_rec: Optional[dict], ref: dict, after: str, claim: Optional[str],
                 gold: str) -> dict:
    from .controller import active, install
    from .generate import build_inputs
    from ..attn import image_token_id
    torch, model, tok = backend.torch, backend.model, backend.tokenizer
    install(model)
    out_ids = rec["output_ids"]
    close = out_ids.index(backend.close_id)
    decode = lambda ids: tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    k = find_target(decode, out_ids, close, after)
    yes_id, no_id, form = pair_ids(tok, out_ids[k])
    prompt, inputs = build_inputs(backend, rec["image"], rec["message"])
    ids = inputs["input_ids"][0].tolist()
    if ids != rec["input_ids"]:
        raise PushCheckFailed("重新构造的提示与记录里的 input_ids 不一致")
    n_in = len(ids)
    full = ids + out_ids[:k]                               # 最后一个位置 = 目标前一个
    t_pos = len(full) - 1
    img_idx = (torch.tensor(ids) == image_token_id(backend)).nonzero().flatten()
    if int(img_idx.numel()) != int(ref["n_img"]):
        raise PushCheckFailed("图像 token 数与参照不一致")
    if claim is not None:
        lo, hi = claim_span(ids, neutral_rec["input_ids"])
    else:
        lo = hi = 0
    img_set = set(img_idx.tolist())
    U = list(range(lo, hi))
    Q = [j for j in range(n_in) if j not in img_set and not lo <= j < hi]
    T = list(range(n_in, n_in + close + 1))
    R = list(range(n_in + close + 1, t_pos + 1))
    gp = {g: torch.tensor(v, dtype=torch.long) for g, v in (("U", U), ("Q", Q), ("T", T), ("R", R))}
    E_ans = {l: ref["E_mask"][1, i] for i, l in enumerate(LAYERS)}
    probe = Probe(torch, gp, E_ans, img_idx)

    dev = model.device
    batch = {}
    for kk, vv in inputs.items():
        batch[kk] = vv.to(dev)
    extra = torch.tensor([out_ids[:k]], dtype=batch["input_ids"].dtype, device=dev)
    batch["input_ids"] = torch.cat([batch["input_ids"], extra], 1)
    batch["attention_mask"] = torch.ones_like(batch["input_ids"])
    if "mm_token_type_ids" in batch:
        batch["mm_token_type_ids"] = torch.cat([batch["mm_token_type_ids"], torch.zeros_like(extra)], 1)
    if "pixel_values" in batch:
        batch["pixel_values"] = batch["pixel_values"].to(model.dtype)
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "rope_deltas"):
        inner.rope_deltas = None

    from .controller import _TARGETS
    mods = {int(m.layer_idx): m for m in model.modules() if id(m) in _TARGETS}
    caps: Dict[str, object] = {}
    hooks = []
    for l, m in mods.items():
        hooks.append(m.register_forward_pre_hook(
            lambda mod, a_, kw, l=l: caps.__setitem__(("h", l), (kw.get("hidden_states", a_[0] if a_ else None))[0, -1].detach()),
            with_kwargs=True))
        hooks.append(m.o_proj.register_forward_pre_hook(
            lambda mod, a_, l=l: caps.__setitem__(("x", l), a_[0][0, -1].detach())))
    norm = _final_norm(model)
    head = model.get_output_embeddings()
    hooks.append(norm.register_forward_pre_hook(lambda mod, a_: caps.__setitem__("hfin", a_[0][0, -1].detach())))
    hooks.append(head.register_forward_pre_hook(lambda mod, a_: caps.__setitem__("nfin", a_[0][0, -1].detach())))
    try:
        with torch.inference_mode(), active(probe):
            logits = model(**batch, use_cache=False).logits[0, -1].float()
    finally:
        for h in hooks:
            h.remove()
    if sorted(probe.rec) != LAYERS:
        raise PushCheckFailed(f"记录到的层 {sorted(probe.rec)} 与 16 个标准层不一致")

    # 最终归一化线性化：n ≈ h ⊙ s / rms(h)
    h = caps["hfin"].float()
    n_ = caps["nfin"].float()
    eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6)))
    with torch.inference_mode():
        s = norm(torch.ones(1, 1, h.numel(), dtype=caps["hfin"].dtype, device=caps["hfin"].device))[0, 0].float()
    s = s * (1.0 + eps) ** 0.5
    rms = float(h.pow(2).mean().add(eps).sqrt())
    lin = h * s / rms
    lin_ratio = float(((lin - n_).abs() / torch.clamp(n_.abs() * 2.0 ** -6, min=0.05)).max())
    if lin_ratio > 1.0:
        raise PushCheckFailed(f"最终归一化线性化与 lm_head 输入不一致（比值 {lin_ratio:.2f}）")
    W = head.weight
    u = (W[yes_id].float() - W[no_id].float()).to(n_.device)
    delta_logits = float(logits[yes_id] - logits[no_id])
    delta_norm = float(n_ @ u)
    if abs(delta_logits - delta_norm) > 0.1 + 0.02 * abs(delta_logits):
        raise PushCheckFailed(f"Δ 不一致：logits {delta_logits:.3f}，归一化输出 {delta_norm:.3f}")
    d = s.to(u.device) * u / rms
    probs = torch.softmax(logits, -1)

    P = {g: 0.0 for g in GROUPS}
    A = {g: 0.0 for g in GROUPS}
    per_layer = []
    checks = {"sum_vs_attn_ratio_max": 0.0, "weights_cover_max": 0.0, "gate_ratio_max": 0.0}
    attn_total = 0.0
    for l in LAYERS:
        m, r = mods[l], probe.rec[l]
        H, D = r["R"].shape
        dev_l = m.o_proj.weight.device
        g = _gate(torch, m, caps[("h", l)].to(dev_l), H, D)
        x = caps[("x", l)].float().to(dev_l)
        Rf = r["R"].reshape(-1).to(dev_l)
        recon = Rf if g is None else g * Rf
        gr = float(((recon - x).abs() / torch.clamp(x.abs() * 2.0 ** -6, min=0.05)).max())
        checks["gate_ratio_max"] = max(checks["gate_ratio_max"], gr)
        checks["sum_vs_attn_ratio_max"] = max(checks["sum_vs_attn_ratio_max"], r["ratio"])
        checks["weights_cover_max"] = max(checks["weights_cover_max"], r["cover"])
        w = (m.o_proj.weight.float().t() @ d.to(dev_l))      # [H·D]
        if g is not None:
            w = w * g
        w = w.view(H, D)
        row = {"layer": l}
        for name, (o, asum) in r["groups"].items():
            ph = (o.to(dev_l) * w).sum(-1)                   # 每头的推力 [H]
            row[name] = {"P": float(ph.sum()), "A": float(asum.sum())}
            P[name] += row[name]["P"]
            A[name] += row[name]["A"]
        attn_total += float(x @ (m.o_proj.weight.float().t() @ d.to(dev_l)))
        per_layer.append(row)
    if checks["sum_vs_attn_ratio_max"] > 1.0 or checks["gate_ratio_max"] > 1.0 or checks["weights_cover_max"] > 1e-3:
        raise PushCheckFailed(f"分组重建检查不通过：{checks}")
    gsum = sum(P.values())
    checks["groups_vs_direct_total"] = {"groups": gsum, "direct": attn_total}
    if abs(gsum - attn_total) > 0.1 + 0.05 * abs(attn_total):
        raise PushCheckFailed(f"各组推力之和 {gsum:.3f} 与 16 层注意力直接推力 {attn_total:.3f} 不一致")
    c = None if claim is None else (1 if claim == "yes" else -1)
    gdir = 1 if gold == "yes" else -1
    S = {g: (P[g] / A[g] if A[g] > 1e-9 else None) for g in GROUPS}
    out = {"target_index": k, "target_form": form, "target_is": decode([out_ids[k]]).strip(),
           "p_target": float(probs[out_ids[k]]), "p_yes": float(probs[yes_id]), "p_no": float(probs[no_id]),
           "delta": delta_logits, "delta_from_16_attn_direct": attn_total,
           "P": P, "A": A, "S": S,
           "toward_gold": {g: P[g] * gdir for g in GROUPS},
           "toward_claim": None if c is None else {g: P[g] * c for g in GROUPS},
           "user_pushes_claim": None if c is None else P["U"] * c > 0,
           "E_opposes_claim": None if c is None or S["E"] is None else S["E"] * c < 0,
           "T_opposes_claim": None if c is None or S["T"] is None else S["T"] * c < 0,
           "checks": checks, "per_layer": per_layer, "n_in": n_in, "t_pos": t_pos,
           "group_sizes": {"E_per_head": int(ref["k"]), "I_per_head": int(ref["n_img"]) - int(ref["k"]),
                           "U": len(U), "Q": len(Q), "T": len(T), "R": len(R)}}
    out["flag"] = None if c is None else bool(out["user_pushes_claim"] and (out["E_opposes_claim"] or out["T_opposes_claim"]))
    return out


def run(cfg, root: str, cases_path: str, out_path: str) -> dict:
    from .pipeline import Engine, _atomic_json, _fail, load_items, _check_no_failure
    work = os.path.dirname(os.path.abspath(out_path))      # 出错现场写在输出目录，主实验目录只读
    _check_no_failure(work)
    cases = json.load(open(cases_path, encoding="utf-8"))["cases"]
    items = {i["id"]: i for i in load_items(root)}
    eng = Engine(cfg)
    rows, refs = [], {}
    load = lambda iid, c: json.load(open(os.path.join(root, "obs", "records", f"{iid}__{c}.json"), encoding="utf-8"))
    for n, cs in enumerate(cases):
        iid, cond = cs["id"], cs["condition"]
        task = f"push/{iid}__{cond}"
        try:
            if iid not in refs:
                with np.load(os.path.join(root, "obs", "attn", f"{iid}__neutral.npz")) as z:
                    ref = build(iid, z["think"], z["answer"])
                if not check(ref)["ok"]:
                    raise PushCheckFailed(f"{iid} 参照检查失败")
                refs[iid] = ref
            it = items[iid]
            rec = load(iid, cond)
            neu = load(iid, "neutral")
            claim = None if cond == "neutral" else it["claim"]
            res = analyze_case(eng.backend, rec, neu, refs[iid], cs.get("after", ""), claim, it["gold"])
        except BaseException as e:
            _fail(work, task, e)
            raise
        row = {"id": iid, "condition": cond, "group": cs["group"], "gold": it["gold"], "claim": claim, **res}
        rows.append(row)
        tg = res["toward_gold"]
        print(f"  {n + 1}/{len(cases)} {iid}__{cond} [{cs['group']}] 目标 {res['target_is']!r} p={res['p_target']:.2f} "
              f"Δ={res['delta']:+.2f} | 推向正确答案：E {tg['E']:+.2f} I {tg['I']:+.2f} U {tg['U']:+.2f} "
              f"Q {tg['Q']:+.2f} T {tg['T']:+.2f} R {tg['R']:+.2f} | 标记 {res['flag']}", file=sys.stderr)
    summary = {}
    for g in sorted({r["group"] for r in rows}):
        sel = [r for r in rows if r["group"] == g]
        summary[g] = {"n": len(sel), "flagged": sum(bool(r["flag"]) for r in sel),
                      "mean_toward_gold": {k: float(np.mean([r["toward_gold"][k] for r in sel])) for k in GROUPS},
                      "user_pushes_claim": sum(bool(r["user_pushes_claim"]) for r in sel),
                      "E_opposes_claim": sum(bool(r["E_opposes_claim"]) for r in sel),
                      "T_opposes_claim": sum(bool(r["T_opposes_claim"]) for r in sel)}
    res = {"root": root, "cases_file": os.path.abspath(cases_path),
           "note": "toward_gold：正数 = 这一组把分差推向正确答案；flag = 用户推向主张，且关键区域或思考的立场反对主张",
           "summary": summary, "rows": rows}
    _atomic_json(out_path, res)
    print("组        n  标记  " + "  ".join(f"{g:>6}" for g in GROUPS) + "   （各组推向正确答案的平均推力）", file=sys.stderr)
    for g, s in summary.items():
        print(f"{g:<8}{s['n']:>3}{s['flagged']:>6}  " + "  ".join(f"{s['mean_toward_gold'][k]:>+6.2f}" for k in GROUPS),
              file=sys.stderr)
    return res
