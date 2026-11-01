# -*- coding: utf-8 -*-
"""ClockQA：模型自己写下读数的那个 token，是被谁推向那个数字的（直接 logit 归因，只读回放，不生成）。

目标位置：思考里模型自己写出的时间 “H:MM” 中，正确时间与用户时间第一处不同的那一位数字。在**预测这一位**的查询位置上，
把每个标准注意力层的输出按来源拆开：
  图像        图像 token
  用户那句话  用户消息里问题之后的那句话（中性条件没有）
  提示其他    系统提示、模板、问题本身
  自己前文    模型在这之前写的思考
每个来源的贡献 = 这一层从这些位置取回的信息（Σ 注意力 × value）经过这一层后续的线性变换（Qwen 的输出门、o_proj；
Gemma 的 o_proj、注意力后的 RMSNorm——按这个位置实际的缩放冻结）、之后各层的 layer_scalar、最后的 RMSNorm（按这个位置实际的
缩放冻结），投影到“正确数字 − 用户数字”的 logit 差方向。正数 = 推向正确数字，负数 = 推向用户的数字。
线性注意力层、MLP、词向量的贡献不拆，合在“其余”里（= 总 logit 差 − 各注意力层各来源之和）。
自检：每个目标位置，各来源之和与这一层实际加到残差上的量逐层核对；最终两个 token 的 logit 与模型实际输出核对。
"""

import json
import os
import re
import sys
from typing import Dict, List, Optional

import numpy as np

GROUPS = ["image", "claim", "prompt", "own"]
GROUP_ZH = {"image": "图像", "claim": "用户那句话", "prompt": "提示其他", "own": "自己前文", "rest": "其余", "total": "总 logit 差"}
MAX_TARGETS = 16
MAX_OUT = 1536                     # 只在思考前 1536 个 token 里找目标
CHECK_TOL = 2e-2
USER_WORDS = re.compile(r"\b(user|you|your|suggest\w*|claim\w*|propos\w*|stated?|says?|said|believe\w*|according)\b", re.I)
HYPO_WORDS = re.compile(r"\b(if|would|could|suppose|assum\w*|hypothetic\w*)\b", re.I)
_T = re.compile(r"(?<![\d:])(\d{1,2}):(\d{2})(?![\d])")


class AttribCheckFailed(RuntimeError):
    pass


# ---------------------------------------------------------------- 纯函数
def time_fields(t: str) -> Dict[str, str]:
    h, m = t.split(":")[:2]
    return {"hour": str(int(h)), "minute": m}


def first_diff(a: str, b: str) -> Optional[int]:
    """两个数字串第一处不同的位置；一个是另一个的前缀（如 1 与 12）时没有可比的一位，返回 None。"""
    for k in range(min(len(a), len(b))):
        if a[k] != b[k]:
            return k
    return None


def context_of(text: str, start: int) -> str:
    """这个时间所在句子的前半句：提到用户 = user；假设句 = hypo；否则 own（模型自己的读数）。"""
    seg = re.split(r"[.!?\n]", text[:start])[-1]
    if USER_WORDS.search(seg):
        return "user"
    if HYPO_WORDS.search(seg):
        return "hypo"
    return "own"


def find_targets(text: str, starts: List[int], ends: List[int], gold: str, claim: str, limit: int = MAX_TARGETS) -> List[dict]:
    """text：思考部分的文字；starts/ends：每个输出 token 在 text 里的字符范围。
    返回目标：哪个输出 token 写下了“正确与用户第一处不同”的那位数字，以及写的是哪个数字。"""
    G, C = time_fields(gold), time_fields(claim)
    out = []
    for m in _T.finditer(text):
        for field, gi in (("hour", 1), ("minute", 2)):
            w, g, c = m.group(gi), G[field], C[field]
            if g == c:
                continue
            k = first_diff(g, c)
            if k is None or len(w) <= k or w[:k] != g[:k]:
                continue
            ch = m.start(gi) + k
            o = next((i for i, (a, b) in enumerate(zip(starts, ends)) if a <= ch < b), None)
            if o is None:
                continue
            out.append({"o": o, "field": field, "char": ch, "digit": text[ch], "gold_digit": g[k], "claim_digit": c[k],
                        "written": "gold" if w == g else ("claim" if w == c else "other"),
                        "context": context_of(text, m.start()), "time_text": m.group(0)})
            if len(out) >= limit:
                return out
    return out


def swap_digit_token(raw: str, digit: str, new: str) -> Optional[str]:
    """token 的原始字符串里只有这一位数字时，把它换成另一个数字；否则返回 None。"""
    ds = [i for i, ch in enumerate(raw) if ch.isdigit()]
    if len(ds) != 1 or raw[ds[0]] != digit:
        return None
    return raw[:ds[0]] + new + raw[ds[0] + 1:]


def char_spans(decode, ids: List[int]) -> (str, List[int], List[int]):
    """逐个累加解码，得到每个 token 的字符范围（与整段解码对齐）。"""
    starts, ends, prev = [], [], ""
    for i in range(len(ids)):
        cur = decode(ids[: i + 1])
        if not cur.startswith(prev):
            prev = cur[: len(prev)]
        starts.append(len(prev))
        ends.append(len(cur))
        prev = cur
    return prev, starts, ends


def group_labels(n_total: int, n_in: int, image_pos: np.ndarray, claim_tok: List[int]) -> np.ndarray:
    g = np.full(n_total, GROUPS.index("prompt"), np.int64)
    g[n_in:] = GROUPS.index("own")
    g[np.asarray(claim_tok, np.int64)] = GROUPS.index("claim")
    g[np.asarray(image_pos, np.int64)] = GROUPS.index("image")
    return g


# ---------------------------------------------------------------- 回放时的拆分
class DLARecorder:
    """装在 symexp.attn 的包装上：只记录、不替换（前向用原注意力函数的输出）。"""
    passthrough = True

    def __init__(self, torch, model, labels: np.ndarray, targets: List[int], du: dict, chunk: int = 256):
        self.torch, self.chunk = torch, chunk
        self.selfcheck_left, self.selfcheck_diffs, self.rel_checked = 0, [], {}
        self.targets = list(targets)                         # 查询位置（绝对位置）
        self.tset = set(self.targets)
        self.du = du                                         # 查询位置 -> (u_gold, u_claim) 词向量
        dev = model.device
        self.masks = torch.stack([torch.as_tensor(labels == k, device=dev) for k in range(len(GROUPS))]).float()
        self.pending: Dict[int, Dict[int, object]] = {}
        self.hidden: Dict[int, object] = {}
        self.result: Dict[int, Dict[int, np.ndarray]] = {t: {} for t in self.targets}   # t -> {layer: [4] logit 差（未除最终 rms）}
        self.checks: List[float] = []
        self.final_h = {}
        self.layer_scale: Dict[int, float] = {}
        self.layer_info: Dict[int, dict] = {}

    def stages_for(self, q_pos):
        return self.torch.full_like(q_pos, -1)

    def accumulate(self, *a, **k):
        return None

    def attribute(self, module, p, vf, q_pos):
        t = self.torch
        sel = [i for i, q in enumerate(q_pos.tolist()) if q in self.tset]
        if not sel:
            return
        kL = p.shape[-1]
        layer = int(module.layer_idx)
        d = self.pending.setdefault(layer, {})
        M = self.masks[:, :kL].to(p.device)                          # [4, kL]
        for i in sel:
            pi = p[0, :, i, :].float()                               # [H, kL]
            a = t.einsum("gk,hk,hkd->ghd", M, pi, vf[0])            # [4, H, D]
            d[int(q_pos[i])] = a

    # 模块钩子
    def attn_pre(self, module, args, kwargs):
        h = kwargs.get("hidden_states", args[0] if args else None)
        if h is not None and h.shape[1] > max(self.targets):
            self.hidden[id(module)] = h[0, self.targets].detach()
        return None

    def attn_post(self, module, args, kwargs, output):
        """注意力模块算完：各来源 → o_proj 之后（Qwen 先乘输出门）；Gemma 的注意力后 RMSNorm 在 norm_post 里处理。"""
        t = self.torch
        layer = int(module.layer_idx)
        info = self.layer_info[layer]
        pend = self.pending.pop(layer, {})
        if not pend:
            return None
        if module.o_proj.bias is not None:
            raise AttribCheckFailed(f"第 {layer} 层 o_proj 有偏置，拆分不支持")
        W = module.o_proj.weight.float()
        y = output[0] if isinstance(output, tuple) else output
        for j, q in enumerate(self.targets):
            a = pend[q].reshape(len(GROUPS), -1)                       # [4, H*D]
            if info["gated"]:
                hq = module.q_proj(self.hidden[id(module)][j][None].to(module.q_proj.weight.dtype)).float()
                gate = t.chunk(hq.view(-1, module.head_dim * 2), 2, dim=-1)[1].reshape(-1)
                a = a * t.sigmoid(gate)[None]
            z = a @ W.T                                                # [4, d_model]
            info.setdefault("z", {})[q] = z
            if not info["postnorm"]:
                self._check(z.sum(0), y[0, q].float(), layer, q)
                self._project(layer, q, z)
        self.hidden.pop(id(module), None)
        return None

    def norm_post(self, layer: int):
        def hook(module, args, output):
            info = self.layer_info[layer]
            zs = info.pop("z", {})
            if not zs:
                return None
            t = self.torch
            f = module(t.ones(1, output.shape[-1], device=output.device, dtype=output.dtype)).float()[0]   # 权重因子
            for q, z in zs.items():
                tot = z.sum(0)
                rms = t.sqrt(tot.pow(2).mean() + float(getattr(module, "eps", 1e-6)))
                yv = z * f[None] / rms
                self._check(yv.sum(0), output[0, q].float(), layer, q)
                self._project(layer, q, yv)
            return None
        return hook

    def final_pre(self, module, args, kwargs):
        h = args[0] if args else kwargs.get("hidden_states")
        if h is not None and h.shape[1] > max(self.targets):
            self.final_h = {q: h[0, q].detach().float() for q in self.targets}
            self.final_factor = module(self.torch.ones(1, h.shape[-1], device=h.device, dtype=h.dtype)).float()[0]
            self.final_eps = float(getattr(module, "eps", getattr(module, "variance_epsilon", 1e-6)))
        return None

    def _check(self, got, want, layer, q):
        rel = float((got - want).abs().max() / want.abs().max().clamp_min(1e-6))
        self.checks.append(rel)
        if rel > CHECK_TOL:
            raise AttribCheckFailed(f"第 {layer} 层位置 {q}：各来源之和与实际输出的相对差 {rel:.3g} > {CHECK_TOL}")

    def _project(self, layer, q, v):
        ug, uc = self.du[q]
        s = self.layer_scale_after.get(layer, 1.0)
        self.result[q][layer] = ((v.to(ug.device) * s) @ (ug - uc)).detach().cpu().numpy()   # u 里已并入最终 RMSNorm 的权重

    def finalize(self, softcap: Optional[float]):
        """除以最终 RMSNorm 在这个位置的缩放（最终 RMSNorm 的权重已并进词向量 u）。"""
        t = self.torch
        out = {}
        for q in self.targets:
            h = self.final_h[q].to(self.du[q][0].device)
            rms = float(t.sqrt(h.pow(2).mean() + self.final_eps))
            ug, uc = self.du[q]
            tot = float(h @ (ug - uc)) / rms
            per = {l: v / rms for l, v in self.result[q].items()}
            out[q] = {"layers": per, "total": tot, "rms": rms,
                      "logit_gold": float(h @ ug) / rms, "logit_claim": float(h @ uc) / rms}
        return out


def model_parts(torch, model):
    """标准注意力模块所在的解码层、每层的类型（Qwen：输出门；Gemma：注意力后 RMSNorm）、之后各层的 layer_scalar、最终 RMSNorm、lm_head。"""
    from .. import attn
    targets = {id(m): m for _, m in attn.find_targets(model)}
    layers_list = None
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.ModuleList) and name.endswith("layers") and len(m) and hasattr(m[0], "mlp"):
            if "vision" in name or "visual" in name:
                continue
            layers_list = m
    if layers_list is None:
        raise AttribCheckFailed("找不到语言模型的解码层列表")
    dec = {}
    for k, lay in enumerate(layers_list):
        sa = getattr(lay, "self_attn", None)
        if sa is not None and id(sa) in targets:
            gated = getattr(sa.q_proj, "out_features", 0) == 2 * sa.config.num_attention_heads * sa.head_dim
            postnorm = hasattr(lay, "pre_feedforward_layernorm") and hasattr(lay, "post_attention_layernorm")
            dec[int(sa.layer_idx)] = {"k": k, "layer": lay, "attn": sa, "gated": gated, "postnorm": postnorm}
    scal = [float(getattr(lay, "layer_scalar", torch.ones(1)).reshape(-1)[0]) for lay in layers_list]
    after = {}
    for l, d in dec.items():
        after[l] = float(np.prod(scal[d["k"]:]))
    final = None
    for name, m in model.named_modules():
        if name.endswith("language_model.norm") or name == "model.norm":
            final = m
    if final is None:
        raise AttribCheckFailed("找不到最终的 RMSNorm")
    head = model.get_output_embeddings()
    cap = getattr(model.config.get_text_config(), "final_logit_softcapping", None) if hasattr(model.config, "get_text_config") \
        else getattr(model.config, "final_logit_softcapping", None)
    return dec, after, final, head, cap


def attrib_one(backend, cfg, item: dict, cond: str, rec: dict, skip: int) -> dict:
    """一条回答：找目标位置 → 一次只读前向 → 各目标、各层、各来源的贡献。"""
    from .. import attn
    from .entropy import _batch
    from .generate import build_inputs
    torch, model, tok = backend.torch, backend.model, backend.tokenizer
    attn.install(model, verbose=False)
    _, inp = build_inputs(backend, item["image"], item["messages"][cond])
    in_ids = inp["input_ids"][0].tolist()
    if rec.get("input_ids") is not None and in_ids != [int(x) for x in rec["input_ids"]]:
        raise AttribCheckFailed(f"{item['id']}__{cond}：重新构造的提示与记录里的 input_ids 不一致")
    n_in = len(in_ids)
    out_ids = [int(x) for x in rec["output_ids"]]
    close = rec.get("close_position")
    think = out_ids[: min(close if close is not None else len(out_ids), MAX_OUT)]
    dec = lambda x: tok.decode(x, skip_special_tokens=False)
    text, starts, ends = char_spans(dec, think)
    claim = item["claim"]
    tg = [x for x in find_targets(text, starts, ends, item["gold"], claim) if x["o"] >= max(skip, 1)]
    base = {"id": item["id"], "condition": cond, "n_in": n_in, "targets": []}
    if not tg:
        return base
    # 用户那句话在提示里的 token 范围
    ptext, ps, pe = char_spans(dec, in_ids)
    claim_tok = []
    msg, q = item["messages"][cond], item["question"]
    if cond != "neutral":
        if not msg.startswith(q):
            raise AttribCheckFailed(f"{item['id']}__{cond}：用户消息不是“问题 + 一句话”的格式，分不出用户那句话")
        sent = msg[len(q):].strip()
        at = ptext.rfind(sent)
        if at < 0:
            raise AttribCheckFailed(f"{item['id']}__{cond}：在提示里找不到用户那句话：{sent!r}")
        claim_tok = [i for i in range(n_in) if ps[i] < at + len(sent) and pe[i] > at]
    img = [i for i, x in enumerate(in_ids) if x == attn.image_token_id(backend)]
    last = max(x["o"] for x in tg)
    seq_out = out_ids[: last + 1]
    labels = group_labels(n_in + len(seq_out), n_in, np.array(img), claim_tok)
    dec_parts, after, final, head, cap = model_parts(torch, model)
    Wu = head.weight
    targets, du, keep = [], {}, []
    for x in tg:
        raw = tok.convert_ids_to_tokens(out_ids[x["o"]])
        rg, rc = swap_digit_token(raw, x["digit"], x["gold_digit"]), swap_digit_token(raw, x["digit"], x["claim_digit"])
        if rg is None or rc is None:
            continue
        ig, ic = tok.convert_tokens_to_ids(rg), tok.convert_tokens_to_ids(rc)
        if ig is None or ic is None or ig == tok.unk_token_id or ic == tok.unk_token_id or ig == ic:
            continue
        qpos = n_in + x["o"] - 1                       # 预测这一位数字的查询位置
        if qpos in du:
            continue
        x.update(qpos=qpos, tok_gold=int(ig), tok_claim=int(ic))
        targets.append(qpos)
        keep.append(x)
    if not targets:
        return base
    rr = DLARecorder(torch, model, labels, targets, {}, chunk=cfg.attn_chunk)
    rr.layer_info = {l: {"gated": d["gated"], "postnorm": d["postnorm"]} for l, d in dec_parts.items()}
    rr.layer_scale_after = after
    hooks = []
    for l, d in dec_parts.items():
        hooks.append(d["attn"].register_forward_pre_hook(rr.attn_pre, with_kwargs=True))
        hooks.append(d["attn"].register_forward_hook(rr.attn_post, with_kwargs=True))
        if d["postnorm"]:
            hooks.append(d["layer"].post_attention_layernorm.register_forward_hook(rr.norm_post(l)))
    hooks.append(final.register_forward_pre_hook(rr.final_pre, with_kwargs=True))
    # 最终 RMSNorm 的权重并进词向量：logit = (h / rms) · (w ⊙ u)
    with torch.inference_mode():
        f = final(torch.ones(1, Wu.shape[1], device=final.weight.device, dtype=model.dtype)).float()[0].to(Wu.device)
        for x in keep:
            rr.du[x["qpos"]] = (Wu[x["tok_gold"]].float() * f, Wu[x["tok_claim"]].float() * f)
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "rope_deltas"):
        inner.rope_deltas = None
    idx = torch.tensor(targets, device=model.device)
    try:
        with torch.inference_mode(), attn.recording(rr):
            try:
                o = model(**_batch(backend, inp, seq_out), use_cache=False, return_dict=True, logits_to_keep=idx)
                logits = o.logits[0].float()
            except TypeError:
                model(**_batch(backend, inp, seq_out), use_cache=False, return_dict=True)
                logits = None
    finally:
        for h in hooks:
            h.remove()
    res = rr.finalize(cap)
    for j, x in enumerate(keep):
        r = res[x["qpos"]]
        per = r["layers"]
        if set(per) != set(dec_parts):
            raise AttribCheckFailed(f"{item['id']}__{cond}：位置 {x['qpos']} 只拆到了 {len(per)} 层，应为 {len(dec_parts)} 层")
        lay = sorted(per)
        mat = np.stack([per[l] for l in lay])              # [L, 4]
        x["layers"] = lay
        x["by_layer"] = mat.round(5).tolist()
        x["by_group"] = {g: float(mat[:, k].sum()) for k, g in enumerate(GROUPS)}
        x["total"] = r["total"]
        x["rest"] = r["total"] - float(mat.sum())
        if logits is not None:
            lg, lc = r["logit_gold"], r["logit_claim"]
            if cap:
                lg, lc = cap * np.tanh(lg / cap), cap * np.tanh(lc / cap)
            ag, ac = float(logits[j, x["tok_gold"]]), float(logits[j, x["tok_claim"]])
            x["logit_check"] = {"ours": [lg, lc], "model": [ag, ac]}
            scale = max(abs(ag), abs(ac), 1.0)
            if max(abs(lg - ag), abs(lc - ac)) / scale > 5e-2:
                raise AttribCheckFailed(f"{item['id']}__{cond}：位置 {x['qpos']} 重算的 logit {lg:.3f}/{lc:.3f} 与模型 {ag:.3f}/{ac:.3f} 不一致")
    base["targets"] = keep
    base["check_max_rel"] = max(rr.checks) if rr.checks else None
    base["claim_tokens"] = len(claim_tok)
    return base


def run_attrib(cfg, cq: str, root: str, conds=None, shard: str = "") -> int:
    from .conflict import _engine, _items
    from .pipeline import _atomic_json, _check_no_failure, _fail, _shard
    from .reading_groups import think_skip
    conds = conds or ["neutral", "belief", "belief_right"]
    _check_no_failure(root)
    items = _items(cq)
    d = os.path.join(root, "attrib")
    os.makedirs(d, exist_ok=True)
    tasks = [(i, c) for i in _shard(sorted(items), shard) for c in conds]
    todo = [t for t in tasks if not os.path.exists(os.path.join(d, f"{t[0]}__{t[1]}.json"))]
    print(f"[attrib] 本进程 {len(tasks)} 条，待做 {len(todo)} 条 → {d}", file=sys.stderr)
    if not todo:
        return 0
    eng = _engine(cfg)
    b = eng.backend
    for n, (iid, c) in enumerate(todo):
        try:
            rec = json.load(open(os.path.join(cq, "obs", "records", f"{iid}__{c}.json"), encoding="utf-8"))
            skip = think_skip(b.tokenizer, cfg, [int(x) for x in rec["output_ids"]])
            r = attrib_one(b, cfg, items[iid], c, rec, skip)
            _atomic_json(os.path.join(d, f"{iid}__{c}.json"), r)
            t0 = r["targets"][0] if r["targets"] else None
            msg = (f"目标 {len(r['targets'])} 个；第一个：{t0['field']} 写的是 {t0['written']}（{t0['context']}），"
                   f"图像 {t0['by_group']['image']:+.2f} 用户那句话 {t0['by_group']['claim']:+.2f} 自己前文 {t0['by_group']['own']:+.2f} "
                   f"总 {t0['total']:+.2f}；拆分自检最大相对差 {r['check_max_rel']:.2e}") if t0 else "没有可用的目标位置"
            print(f"  {n + 1}/{len(todo)} {iid}__{c} {msg}", file=sys.stderr)
        except Exception as e:
            _fail(root, f"attrib_{iid}__{c}", e)
            raise
    return len(todo)


def run_attrib_summary(cq: str, root: str, judgments: str, label: str, out_md: Optional[str] = None) -> dict:
    """每条回答取第一个“模型自己的读数”（不是复述用户、不是假设句）上的目标位置，按 被带偏 / 守住 × 条件 汇总各来源的贡献。"""
    from .conflict import _items
    from .reading_groups import outcomes_clock
    from .sources import _signflip_p
    items = _items(cq)
    bel, neu = outcomes_clock(judgments)
    d = os.path.join(root, "attrib")
    nc = [i for i in sorted(items) if neu.get(i) == "对"]
    groups = [("被带偏", [i for i in nc if bel.get(i) in ("采", "他")]), ("守住", [i for i in nc if bel.get(i) == "对"])]
    cols = ["image", "claim", "prompt", "own", "rest", "total"]
    L = [f"# {label}：写下读数的那一位数字，被谁推向正确 / 用户的数字", "",
         "## 数据是什么", "",
         f"- **回答**：{label} 原模型的回答（中性 / 用户说错 / 用户说对），只读回放，不生成。",
         "- **目标位置**：思考里模型自己写出的时间 “H:MM” 中，正确时间与用户时间第一处不同的那一位数字（小时或分钟）。"
         "每条回答取第一个这样的位置，且所在句子不是在复述用户、不是假设句。用户说对时，“用户的数字”仍取那道题的错误时间，作为同一个对照方向。",
         "- **数值**：在预测这一位数字的位置上，“正确数字的 logit − 用户数字的 logit”被各来源推动了多少（logit 单位）。正数 = 推向正确数字，负数 = 推向用户的数字。",
         "  - 图像 / 用户那句话 / 提示其他 / 自己前文：所有标准注意力层从这些位置取回的信息的直接贡献之和；",
         "  - 其余：线性注意力层、MLP、词向量等没有拆开的部分；总 logit 差 = 以上之和。",
         "- **写下的数字**：这一位上模型实际写的是正确的数字、用户的数字，还是别的。",
         "- **分组**：中性时答对的题，按用户说错那条回答的最终结果分成被带偏 / 守住。p 为按题的符号翻转检验（检验均值是否为 0）。", ""]
    res = {"label": label, "cells": {}}
    for cond, cname in (("belief", "用户说错"), ("neutral", "中性"), ("belief_right", "用户说对")):
        L += [f"## {cname}", "", "| 组（题数） | 写下的数字 正确/用户/别的 | " + " | ".join(GROUP_ZH[c] for c in cols) + " |",
              "|---|---|" + "---|" * len(cols)]
        for gname, ids in groups:
            R = []
            for i in ids:
                p = os.path.join(d, f"{i}__{cond}.json")
                if not os.path.exists(p):
                    continue
                r = json.load(open(p, encoding="utf-8"))
                own = [x for x in r["targets"] if x["context"] == "own"]
                if own:
                    R.append((i, own[0]))
            if not R:
                L.append(f"| {gname}（0） | — |" + " — |" * len(cols))
                continue
            vals = {c: np.array([x["by_group"][c] if c in x["by_group"] else x[c] for _, x in R]) for c in cols}
            w = [x["written"] for _, x in R]
            L.append(f"| {gname}（{len(R)}） | {w.count('gold')}/{w.count('claim')}/{w.count('other')} | " +
                     " | ".join(f"{vals[c].mean():+.2f}（p={_signflip_p(vals[c]):.3f}）" for c in cols) + " |")
            res["cells"].setdefault(cond, {})[gname] = [{"id": i, **{k: x[k] for k in ("field", "written", "time_text", "by_group", "rest", "total")}}
                                                       for i, x in R]
        L.append("")
    L += ["## 题号", "", *[f"- {g}（{len(ids)} 题）：{'、'.join(ids) or '无'}" for g, ids in groups], ""]
    out_md = out_md or os.path.join(root, "attrib_report.md")
    open(out_md, "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump(res, open(out_md.replace(".md", ".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n".join(L), file=sys.stderr)
    return res
