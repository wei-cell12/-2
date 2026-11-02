# -*- coding: utf-8 -*-
"""先读后听（preread）：两次调用。

  1. 初步读图：只给图和问题（模型自己改写出的中性提问，fp-model2 neutralize 的结果），开思考模式，生成 PRE_MIN–PRE_MAX
     个 token：满 PRE_MIN 个后在句末停，最多 PRE_MAX 个；写到思考结束 / 生成结束就停。
  2. 给图、问题和用户的消息（完整受压提示）。思考的开头先放入：
         [Preliminary reading of the image, made from the image and the question only:]
         <初步读图的文字>
         [End of preliminary reading.]
     然后让模型接着往下写（同一套解码设置、长度上限；预先放入的 token 计入思考长度）。
思考开头标记由模型自己写的模型（Gemma 4：<|channel>thought）在预置文字前补上这个标记。
"""

import re
import time
from typing import List

PRE_MIN, PRE_MAX = 64, 128
PRE_HEAD = "[Preliminary reading of the image, made from the image and the question only:]\n"
PRE_TAIL = "\n[End of preliminary reading.]\n\n"
_END = re.compile(r"[.!?]\s|\n")          # 句末：句号/问号/叹号后跟空白，或换行（不会停在 "3.5" 这样的数字中间）


class PrereadCheckFailed(RuntimeError):
    pass


def pre_should_stop(n_gen: int, tail: str, last_id: int, close_id: int, eos_ids, lo: int = PRE_MIN,
                    hi: int = PRE_MAX) -> bool:
    if last_id == close_id or last_id in eos_ids:
        return True
    if n_gen >= hi:
        return True
    return n_gen >= lo and bool(_END.search(tail))


def trim_lead(text: str) -> str:
    """停下时可能已多写了下一句的开头，或写满上限停在半句话中间：退回到最后一个句末（至少保留一半；找不到就原样保留）。"""
    text = text.strip()
    if not text or text[-1] in ".!?":
        return text
    ends = [m.start() + 1 for m in re.finditer(r"[.!?](?=\s)|\n", text)]
    ends = [e for e in ends if e >= len(text) // 2]
    return text[: ends[-1]].strip() if ends else text


def strip_open(text: str, think_open: str) -> str:
    """去掉初步读图输出开头模型自己写的思考开头标记（如 Gemma 4 的 <|channel>thought）。"""
    t = text.lstrip()
    head = think_open.split(">")[-1].strip() if ">" in think_open else ""
    for h in (think_open, head):
        if h and t.startswith(h):
            t = t[len(h):].lstrip()
    return t


def build_prefix(lead_text: str, think_open: str, think_prefill: bool) -> str:
    return ("" if think_prefill else think_open + "\n") + PRE_HEAD + lead_text.strip() + PRE_TAIL


def extend_inputs(inputs: dict, extra: List[int]) -> dict:
    """提示后面接上预置的 token：input_ids 接上，attention_mask 接 1，其他与 input_ids 同形状的逐 token 张量接 0。"""
    import torch
    n = int(inputs["input_ids"].shape[1])
    add = torch.tensor([extra], dtype=inputs["input_ids"].dtype)
    out = {}
    for k, v in inputs.items():
        if hasattr(v, "shape") and tuple(v.shape) == (1, n):
            if k == "input_ids":
                out[k] = torch.cat([v, add], 1)
            elif k == "attention_mask":
                out[k] = torch.cat([v, torch.ones_like(add, dtype=v.dtype)], 1)
            else:
                out[k] = torch.cat([v, torch.zeros_like(add, dtype=v.dtype)], 1)
        else:
            out[k] = v
    return out


def _seed(backend, seed: int):
    torch, tf = backend.torch, backend.transformers
    tf.set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    inner = getattr(backend.model, "model", None)
    if inner is not None and hasattr(inner, "rope_deltas"):
        inner.rope_deltas = None


def _to_dev(backend, inputs):
    dev = backend.model.device
    b = {k: v.to(dev) for k, v in inputs.items() if hasattr(v, "to")}
    if "pixel_values" in b:
        b["pixel_values"] = b["pixel_values"].to(backend.model.dtype)
    return b


def generate_preread(backend, cfg, item: dict, cond: str, seed: int, neutral_text: str) -> dict:
    from ..attn import image_token_id
    from .conflict import claim_of
    from .generate import build_inputs, make_budget
    from .model2 import split_output, task_of
    from .verify import final_time, outcome
    torch, tf, model, tok = backend.torch, backend.transformers, backend.model, backend.tokenizer
    if backend.close_id is None:
        raise PrereadCheckFailed(f"思考结束标记 {cfg.think_close!r} 不是单个 token")
    if not neutral_text:
        raise PrereadCheckFailed("先读需要模型自己改写的中性提问（先跑 --phase neutralize）")
    gc = model.generation_config
    eos = gc.eos_token_id if isinstance(gc.eos_token_id, list) else [gc.eos_token_id]
    eos_set = set(int(x) for x in eos)
    sample = dict(do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p, top_k=cfg.top_k, min_p=cfg.min_p,
                  repetition_penalty=cfg.repetition_penalty, use_cache=True)
    # ---- 1. 初步读图：只给图和问题
    _, inp1 = build_inputs(backend, item["image"], neutral_text)
    n1 = int(inp1["input_ids"].shape[1])

    class Stop(tf.StoppingCriteria):
        def __call__(self, input_ids, scores, **kw):
            gen = input_ids[0, n1:].tolist()
            tail = tok.decode(gen[-3:], skip_special_tokens=False) if gen else ""
            s = bool(gen) and pre_should_stop(len(gen), tail, gen[-1], backend.close_id, eos_set)
            return torch.full((input_ids.shape[0],), s, dtype=torch.bool, device=input_ids.device)
    _seed(backend, seed)
    t0 = time.time()
    with torch.inference_mode():
        o1 = model.generate(**_to_dev(backend, inp1), max_new_tokens=PRE_MAX, stopping_criteria=tf.StoppingCriteriaList([Stop()]),
                            **sample)
    lead_ids = o1[0, n1:].tolist()
    ended = bool(lead_ids) and (lead_ids[-1] == backend.close_id or lead_ids[-1] in eos_set)
    while lead_ids and (lead_ids[-1] == backend.close_id or lead_ids[-1] in eos_set):
        lead_ids.pop()
    hit_max = len(o1[0, n1:]) >= PRE_MAX and not ended
    raw_lead = tok.decode(lead_ids, skip_special_tokens=True)
    lead_text = trim_lead(strip_open(raw_lead, cfg.think_open))
    if not lead_text:
        raise PrereadCheckFailed(f"{item['id']}__{cond}：初步读图没有写出内容")
    s1 = round(time.time() - t0, 2)
    # ---- 2. 完整受压提示 + 思考开头放入初步读图
    prompt, inp2 = build_inputs(backend, item["image"], item["messages"][cond])
    n_in = int(inp2["input_ids"].shape[1])
    n_img = int((inp2["input_ids"][0] == image_token_id(backend)).sum())
    if n_img == 0:
        raise PrereadCheckFailed("提示里没有图像 token")
    prefix_text = build_prefix(lead_text, cfg.think_open, cfg.think_prefill)
    prefix_ids = tok(prefix_text, add_special_tokens=False)["input_ids"]
    if not cfg.think_prefill and tok.decode(prefix_ids, skip_special_tokens=False).strip()[: len(cfg.think_open)] != cfg.think_open:
        raise PrereadCheckFailed(f"思考开头标记 {cfg.think_open!r} 编码后对不上")
    batch2 = _to_dev(backend, extend_inputs(inp2, prefix_ids))
    tb, ab = cfg.think_budget, cfg.answer_budget
    budget = make_budget(tf, torch, n_in, backend.close_id, int(eos[0]), tb, ab)     # 预置的 token 计入思考长度
    _seed(backend, seed)
    t1 = time.time()
    with torch.inference_mode():
        o2 = model.generate(**batch2, max_new_tokens=max(tb + ab + 2 - len(prefix_ids), 16),
                            logits_processor=tf.LogitsProcessorList([budget]), **sample)
    gen = o2[0, n_in + len(prefix_ids):].tolist()
    s2 = round(time.time() - t1, 2)
    output = prefix_ids + gen
    dec = lambda x: tok.decode(x, skip_special_tokens=True)
    thinking, answer, close = split_output(dec, output, backend.close_id, cfg.think_open)
    claim = claim_of(item, cond)
    rec = {"arm": "preread", "id": item["id"], "condition": cond, "gold": item["gold"], "claim": claim,
           "message": item["messages"][cond], "image": item["image"], "image_sha256": item.get("image_sha256"),
           "seed": seed, "model": cfg.model_path, "neutral_text": neutral_text,
           "prompt_text": prompt, "input_ids": inp2["input_ids"][0].tolist(), "output_ids": output,
           "raw_output": tok.decode(output, skip_special_tokens=False), "thinking": thinking, "answer": answer,
           "close_position": close, "thinking_empty": not thinking.strip(), "n_in": n_in, "n_img": n_img, "n_out": len(output),
           "seconds": round(s1 + s2, 2), "budget": budget.metadata(),
           "preread": {"lead_tokens": len(lead_ids), "lead_hit_max": hit_max, "lead_ended_thinking": ended,
                       "lead_raw": raw_lead, "lead_text": lead_text, "prefix_text": prefix_text,
                       "prefix_tokens": len(prefix_ids), "seconds_lead": s1, "seconds_main": s2,
                       "min_max": [PRE_MIN, PRE_MAX]},
           "sampling": {"temperature": cfg.temperature, "top_p": cfg.top_p, "top_k": cfg.top_k, "min_p": cfg.min_p}}
    rec["task"] = task_of(item)
    if claim and rec["task"] == "clock":
        rec["outcome"] = outcome(answer, item["gold"], claim)
        rec["final_time"] = final_time(answer, claim)
    elif rec["task"] == "yesno":
        rec["semantic_judgment_status"] = "pending"
    return rec
