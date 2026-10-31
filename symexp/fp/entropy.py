# -*- coding: utf-8 -*-
"""熵差分析（离线，不生成）：对已保存的生成轨迹，在同一段已生成内容上分别用“受压提示”和“中性提示”
各做一次前向，逐 token 比较两边对下一个 token 的预测。

  H^P_t   受压提示下的预测熵（完整词表、原始 logits，不做 top-k/top-p 截断）
  H^N_t   中性提示下的预测熵
  ΔH_t    = H^P_t − H^N_t：用户那句话让模型在这一步额外多出的不确定性
  JSD_t   两边预测分布的 JS 散度（自然对数，范围 [0, ln2]）
  Δlogp_t = log p^P(y_t) − log p^N(y_t)：实际写出的 token 在受压提示下比中性提示下更被看好多少

轨迹（root = fp-subset-setup 建的目录，例如 pathvqa-dual10）：
  obs/records/{id}__{压力}.json            原始受压生成
  obs/records/{id}__neutral.json           原始中性生成（对照：文字由中性条件写出）
  int/dual_mix|dual_shuffle/records/...    v2 干预后的生成
每条都用该题 candidates.json 里的那种压力作为“受压提示”。

检查（不通过就停下）：
  - 重新构造的提示与记录里的 input_ids 一致（受压、中性各自对应的那条）
  - 原始观察轨迹：实际 token 在“生成时用的那个提示”下排在前 20 名的比例 ≥ 0.98
    （生成用 top_k=20，比例低说明输入拼接或位置有误）
  - 熵在 [0, ln|V|]、JSD 在 [0, ln2] 内，没有非有限值
"""

import json
import math
import os
import sys
from typing import Dict, List

import numpy as np

SOURCES = [("obs", "pressure"), ("obs", "neutral"), ("int/dual_mix", "pressure"), ("int/dual_shuffle", "pressure")]
RANK_K = 20
RANK_MIN = 0.98
CHUNK = 256


class EntropyCheckFailed(RuntimeError):
    pass


def _batch(backend, inputs, out_ids):
    torch, model = backend.torch, backend.model
    dev = model.device
    b = {k: v.to(dev) for k, v in inputs.items() if hasattr(v, "to")}      # 非张量的字段（如图像 token 数的列表）不传
    extra = torch.tensor([out_ids], dtype=b["input_ids"].dtype, device=dev)
    b["input_ids"] = torch.cat([b["input_ids"], extra], 1)
    b["attention_mask"] = torch.ones_like(b["input_ids"])
    if "mm_token_type_ids" in b:
        b["mm_token_type_ids"] = torch.cat([b["mm_token_type_ids"], torch.zeros_like(extra)], 1)
    if "pixel_values" in b:
        b["pixel_values"] = b["pixel_values"].to(model.dtype)
    return b


def _logits(backend, inputs, out_ids):
    """返回预测 out_ids[0..T-1] 的那 T 行 logits（bf16，留在 GPU）。"""
    torch, model = backend.torch, backend.model
    n_in = int(inputs["input_ids"].shape[1])
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "rope_deltas"):
        inner.rope_deltas = None
    with torch.inference_mode():
        lg = model(**_batch(backend, inputs, out_ids), use_cache=False, return_dict=True).logits[0]
    return lg[n_in - 1: n_in - 1 + len(out_ids)]


def signals(torch, lgP, lgN, toks) -> Dict[str, np.ndarray]:
    """逐 token：H^P、H^N、JSD、log p^P(y)、log p^N(y)、y 在两边的名次。按块算，避免一次占满显存。"""
    out = {k: [] for k in ("H_P", "H_N", "JSD", "logp_P", "logp_N", "rank_P", "rank_N")}
    y = torch.tensor(toks, device=lgP.device)
    for s in range(0, lgP.shape[0], CHUNK):
        a = torch.log_softmax(lgP[s:s + CHUNK].float(), -1)
        b = torch.log_softmax(lgN[s:s + CHUNK].float(), -1)
        pa, pb = a.exp(), b.exp()
        out["H_P"].append(-(pa * a).sum(-1))
        out["H_N"].append(-(pb * b).sum(-1))
        lm = torch.log((0.5 * (pa + pb)).clamp_min(1e-30))
        out["JSD"].append(0.5 * (pa * (a - lm)).sum(-1) + 0.5 * (pb * (b - lm)).sum(-1))
        yy = y[s:s + CHUNK, None]
        ga, gb = a.gather(1, yy), b.gather(1, yy)
        out["logp_P"].append(ga[:, 0])
        out["logp_N"].append(gb[:, 0])
        out["rank_P"].append((a > ga).sum(-1))
        out["rank_N"].append((b > gb).sum(-1))
        del a, b, pa, pb, lm
    return {k: torch.cat(v).float().cpu().numpy() for k, v in out.items()}


def analyze_one(backend, item: dict, cond: str, rec: dict, written_by: str) -> dict:
    """rec 的文字由 written_by（pressure / neutral）条件写出；受压提示用 cond 这种压力。"""
    from .generate import build_inputs
    torch, tok = backend.torch, backend.tokenizer
    _, inP = build_inputs(backend, item["image"], item["messages"][cond])
    _, inN = build_inputs(backend, item["image"], item["messages"]["neutral"])
    own = inP if written_by == "pressure" else inN
    if own["input_ids"][0].tolist() != rec["input_ids"]:
        raise EntropyCheckFailed(f"{rec['id']}：重新构造的提示与记录里的 input_ids 不一致")
    out_ids = rec["output_ids"]
    lgP = _logits(backend, inP, out_ids)
    lgN = _logits(backend, inN, out_ids)
    sg = signals(torch, lgP, lgN, out_ids)
    vmax = math.log(int(lgP.shape[-1])) + 1e-3
    del lgP, lgN
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    for k in ("H_P", "H_N", "JSD"):
        v = sg[k]
        if not np.isfinite(v).all() or v.min() < -1e-4 or v.max() > (math.log(2) + 1e-4 if k == "JSD" else vmax):
            raise EntropyCheckFailed(f"{rec['id']}：{k} 超出范围 [{v.min()}, {v.max()}]")
    own_rank = sg["rank_P"] if written_by == "pressure" else sg["rank_N"]
    in_top = float((own_rank < RANK_K).mean())
    close = out_ids.index(backend.close_id) if backend.close_id in out_ids else len(out_ids)
    pieces = [tok.decode([t], skip_special_tokens=False, clean_up_tokenization_spaces=False) for t in out_ids]
    r4 = lambda a: [round(float(x), 4) for x in a]
    return {"n_out": len(out_ids), "close": close, "in_top20_own_prompt": in_top,
            "H_P": r4(sg["H_P"]), "H_N": r4(sg["H_N"]), "JSD": r4(sg["JSD"]),
            "dlogp": r4(sg["logp_P"] - sg["logp_N"]), "pieces": pieces,
            "think_mean": {k: float(np.mean(sg[k][:close])) if close else None for k in ("H_P", "H_N", "JSD")}}


def run(cfg, root: str, out_path: str) -> dict:
    from .pipeline import Engine, _atomic_json, _check_no_failure, _fail, load_items
    work = os.path.dirname(os.path.abspath(out_path))
    _check_no_failure(work)
    items = {i["id"]: i for i in load_items(root)}
    cands = json.load(open(os.path.join(root, "candidates.json"), encoding="utf-8"))["candidates"]
    eng = Engine(cfg)
    if eng.backend is None:
        raise SystemExit("fp-entropy 需要 hf 后端")
    res = []
    for k, c in enumerate(cands):
        iid, cond = c["id"], c["condition"]
        for sub, written in SOURCES:
            tag = f"{iid}__{cond if written == 'pressure' else 'neutral'}"
            p = os.path.join(root, sub, "records", tag + ".json")
            if not os.path.exists(p):
                continue
            task = f"entropy/{sub}/{tag}"
            try:
                rec = json.load(open(p, encoding="utf-8"))
                row = analyze_one(eng.backend, items[iid], cond, rec, written)
                if sub == "obs" and row["in_top20_own_prompt"] < RANK_MIN:
                    raise EntropyCheckFailed(f"{tag}：实际 token 在原提示下进前 {RANK_K} 名的比例只有 "
                                             f"{row['in_top20_own_prompt']:.3f}（输入拼接或位置可能有误）")
            except BaseException as e:
                _fail(work, task, e)
                raise
            row.update(id=iid, condition=cond, source=sub, written_by=written)
            res.append(row)
            tm = row["think_mean"]
            print(f"  {k + 1}/{len(cands)} {sub}/{tag} tokens={row['n_out']} top20={row['in_top20_own_prompt']:.3f} "
                  f"思考段 H^P={tm['H_P']:.3f} H^N={tm['H_N']:.3f} JSD={tm['JSD']:.4f}", file=sys.stderr)
    out = {"root": root, "note": __doc__.split("\n\n")[0], "rows": res}
    _atomic_json(out_path, out)
    print(f"[fp-entropy] {len(res)} 条轨迹 → {out_path}", file=sys.stderr)
    return out
