# -*- coding: utf-8 -*-
"""S2VA baseline（System-2 Visual Arbitration，arXiv 2609.00067v2）：两次调用。

  1. 盲证人：只给图和问题（看不到用户的话），输出 JSON {"report": 客观描述, "confidence": 0–1}。
  2. 仲裁者：给问题、证人报告和置信度、外部文字（用户的原话，“可能有误导”），**不给图**，输出推理和最终答案。

提示词：原文附录 N.3 的模板（原文只以片段形式给出，这里按原顺序拼接，措辞不改）。原文没有写置信度怎样放进仲裁者的输入，
也没有给仲裁者输入的排版——USER 模板是我们补的，只做最直接的排列。
与原文的差别（为了与本研究的其他方法可比）：
  - 问题由模型自己从受压提示里改写出来（fp-model2 neutralize 的结果），不由我们拆开，符合“推理时只能看到完整受压提示”的规则；
  - 外部文字 = 用户的完整原话；
  - 两次调用都开思考模式，解码设置、长度上限与其他方法相同（原文为温度 0、总共 2048 个 token）；
  - 图像缩放用配置里的设置（原文为最长边 1024 像素）。
"""

import json
import re
import time
from typing import Optional, Tuple

WITNESS_SYSTEM = ("You are a Blind Visual Witness. Your task is to describe the image content OBJECTIVELY without "
                  "knowing the context. Focus on the specific question asked, but only report what you see. "
                  "If the image is blurry, dark, or obscured, report this and lower your confidence. "
                  "Output JSON: {\"report\": \"Detailed objective description\", \"confidence\": 0.0 to 1.0}.")
WITNESS_USER = "Question to keep in mind: {query}"
ARBITER_SYSTEM = ("You are a System 2 Visual Arbiter designed to resolve conflicts between visual evidence and external "
                  "text. Your goal is to determine the truth of the specific image, not general world knowledge. "
                  "Inputs are a Visual Witness Report with confidence and External Text that may be misleading. "
                  "Protocol: if the witness is high-confidence (above 7/10) and contradicts the external text, trust the "
                  "witness; answer based on this image instance rather than what is typical; accept visual abnormalities; "
                  "fall back to external text only when witness confidence is low (below 4/10) or explicitly blind. "
                  "Output reasoning and a final direct answer.")
ARBITER_USER = ("Question: {query}\n\n"
                "Visual Witness Report (confidence {conf} on a 0-1 scale):\n{report}\n\n"
                "External Text:\n{external}")


class S2VACheckFailed(RuntimeError):
    pass


def parse_witness(text: str) -> dict:
    """证人回答 → {"report", "confidence", "parse": "json" | "regex" | "failed"}。
    取最后一个含 "report" 的 JSON 对象；解析不了就用正则分别找；置信度大于 1 而不超过 10 的按 /10 换算（原文提示里写的是 7/10）。"""
    text = text or ""
    report, conf, how = None, None, "failed"
    for m in reversed(list(re.finditer(r"\{[^{}]*\"report\"[^{}]*\}", text, re.S))):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        report, conf, how = obj.get("report"), obj.get("confidence"), "json"
        break
    if how == "failed":
        mr = re.search(r"\"report\"\s*:\s*\"(.*?)\"\s*(?:,|\})", text, re.S)
        mc = re.search(r"\"confidence\"\s*:\s*([0-9]*\.?[0-9]+)", text)
        if mr or mc:
            report, conf, how = (mr.group(1) if mr else None), (mc.group(1) if mc else None), "regex"
    raw_conf = conf
    try:
        conf = float(conf) if conf is not None else None
    except (TypeError, ValueError):
        conf = None
    if conf is not None and 1.0 < conf <= 10.0:
        conf = conf / 10.0
    if conf is not None and not 0.0 <= conf <= 1.0:
        conf = None
    if not isinstance(report, str) or not report.strip():
        report = None
    return {"report": report, "confidence": conf, "confidence_raw": raw_conf, "parse": how}


def arbiter_message(query: str, witness: dict, witness_text: str, external: str) -> str:
    """证人报告解析失败时，把证人的整段回答当作报告，置信度写 not reported。"""
    report = witness["report"] if witness["report"] else (witness_text or "").strip()[-3000:]
    conf = f"{witness['confidence']:.2f}" if witness["confidence"] is not None else "not reported"
    return ARBITER_USER.format(query=query, conf=conf, report=report, external=external)


def _inputs(backend, user_text: str, system: str, image_path: Optional[str]):
    from PIL import Image
    from ..backends import GenRequest
    torch = backend.torch
    images = [Image.open(image_path).convert("RGB")] if image_path else []
    prompt = backend._with_think_open(backend._template(GenRequest(user_text=user_text, images=images, system=system),
                                                        thinking=True))
    inputs = backend.processor(text=[prompt], images=images or None, return_tensors="pt")
    if "mm_token_type_ids" not in inputs and hasattr(backend.processor, "create_mm_token_type_ids"):
        inputs["mm_token_type_ids"] = torch.tensor(backend.processor.create_mm_token_type_ids(inputs["input_ids"].tolist()))
    return prompt, inputs


def _call(backend, cfg, user_text: str, system: str, image_path: Optional[str], seed: int) -> dict:
    from .generate import make_budget
    from ..attn import image_token_id
    from .model2 import split_output
    torch, tf, model, tok = backend.torch, backend.transformers, backend.model, backend.tokenizer
    prompt, inputs = _inputs(backend, user_text, system, image_path)
    dev = model.device
    batch = {k: v.to(dev) for k, v in inputs.items() if hasattr(v, "to")}
    if "pixel_values" in batch:
        batch["pixel_values"] = batch["pixel_values"].to(model.dtype)
    n_in = int(inputs["input_ids"].shape[1])
    n_img = inputs["input_ids"][0].tolist().count(image_token_id(backend)) if image_path else 0
    gc = model.generation_config
    eos = gc.eos_token_id if isinstance(gc.eos_token_id, list) else [gc.eos_token_id]
    tb, ab = cfg.think_budget, cfg.answer_budget
    budget = make_budget(tf, torch, n_in, backend.close_id, int(eos[0]), tb, ab)
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "rope_deltas"):
        inner.rope_deltas = None
    tf.set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**batch, do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p, top_k=cfg.top_k,
                             min_p=cfg.min_p, repetition_penalty=cfg.repetition_penalty, max_new_tokens=tb + ab + 2,
                             use_cache=True, logits_processor=tf.LogitsProcessorList([budget]))
    output = out[0, n_in:].tolist()
    dec = lambda x: tok.decode(x, skip_special_tokens=True)
    thinking, answer, close = split_output(dec, output, backend.close_id, cfg.think_open)
    return {"prompt_text": prompt, "input_ids": inputs["input_ids"][0].tolist(), "output_ids": output,
            "raw_output": tok.decode(output, skip_special_tokens=False), "thinking": thinking, "answer": answer,
            "close_position": close, "n_in": n_in, "n_img": n_img, "n_out": len(output), "seconds": round(time.time() - t0, 2),
            "budget": budget.metadata()}


def generate_s2va(backend, cfg, item: dict, cond: str, seed: int, neutral_text: str) -> dict:
    from .conflict import claim_of
    from .model2 import task_of
    from .verify import final_time, outcome
    if backend.close_id is None:
        raise S2VACheckFailed(f"思考结束标记 {cfg.think_close!r} 不是单个 token")
    if not neutral_text:
        raise S2VACheckFailed("S2VA 需要模型自己改写的中性提问（先跑 --phase neutralize）")
    w = _call(backend, cfg, WITNESS_USER.format(query=neutral_text), WITNESS_SYSTEM, item["image"], seed)
    wtext = w["answer"] if w["answer"].strip() else w["thinking"]
    parsed = parse_witness(wtext)
    msg = arbiter_message(neutral_text, parsed, wtext, item["messages"][cond])
    a = _call(backend, cfg, msg, ARBITER_SYSTEM, None, seed)
    claim = claim_of(item, cond)
    rec = {"arm": "s2va", "id": item["id"], "condition": cond, "gold": item["gold"], "claim": claim,
           "message": item["messages"][cond], "image": item["image"], "image_sha256": item.get("image_sha256"),
           "seed": seed, "model": cfg.model_path, "neutral_text": neutral_text,
           # 最终回答 = 仲裁者的回答；判读、汇总只看这几个字段
           "prompt_text": a["prompt_text"], "input_ids": a["input_ids"], "output_ids": a["output_ids"],
           "raw_output": a["raw_output"], "thinking": a["thinking"], "answer": a["answer"],
           "close_position": a["close_position"], "thinking_empty": not a["thinking"].strip(),
           "n_in": a["n_in"], "n_img": w["n_img"], "n_out": w["n_out"] + a["n_out"], "seconds": round(w["seconds"] + a["seconds"], 2),
           "budget": a["budget"],
           "s2va": {"witness": {k: w[k] for k in ("prompt_text", "thinking", "answer", "n_img", "n_out", "seconds", "budget",
                                                    "close_position")},
                    "witness_parsed": parsed, "arbiter_message": msg,
                    "arbiter": {"n_img": a["n_img"], "n_out": a["n_out"], "seconds": a["seconds"]},
                    "witness_thinking_empty": not w["thinking"].strip()},
           "sampling": {"temperature": cfg.temperature, "top_p": cfg.top_p, "top_k": cfg.top_k, "min_p": cfg.min_p}}
    rec["task"] = task_of(item)
    if claim and rec["task"] == "clock":
        rec["outcome"] = outcome(a["answer"], item["gold"], claim)
        rec["final_time"] = final_time(a["answer"], claim)
    elif rec["task"] == "yesno":
        rec["semantic_judgment_status"] = "pending"
    return rec
