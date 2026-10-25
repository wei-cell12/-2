# -*- coding: utf-8 -*-
"""一次生成：构造输入 → 安装控制器 → 一次 generate 完成思考和回答 → 拆分文本、返回记录与数组。

思考预算：生成了 think_budget 个 token 仍没有 </think> 时强制输出 </think>；
回答预算：</think> 之后最多 answer_budget 个 token，到了强制结束。与之前的预实验一致。
"""

import hashlib
import time
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

from ..backends import GenRequest
from .controller import FPController, active, install


def make_budget(tf, torch, prompt_len: int, close_id: int, eos_id: int, think_limit: int, answer_limit: int):
    class ThinkingBudget(tf.LogitsProcessor):
        def __init__(self):
            self.close_position = None
            self.forced_close = False
            self.forced_end = False

        def __call__(self, input_ids, scores):
            n = input_ids.shape[-1] - prompt_len
            if n and self.close_position is None and int(input_ids[0, -1]) == close_id:
                self.close_position = n - 1
            forced = None
            if self.close_position is None and n >= think_limit:
                forced, self.forced_close = close_id, True
            elif self.close_position is not None and n - self.close_position - 1 >= answer_limit:
                forced, self.forced_end = eos_id, True
            if forced is None:
                return scores
            out = torch.full_like(scores, float("-inf"))
            out[:, forced] = 0.0
            return out

        def metadata(self):
            return {"think_limit": think_limit, "answer_limit": answer_limit,
                    "thinking_budget_hit": self.forced_close, "answer_budget_hit": self.forced_end,
                    "close_position": self.close_position}

    return ThinkingBudget()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build_inputs(backend, image_path: str, user_text: str):
    torch = backend.torch
    image = Image.open(image_path).convert("RGB")
    prompt = backend._with_think_open(backend._template(GenRequest(user_text=user_text, images=[image]), thinking=True))
    inputs = backend.processor(text=[prompt], images=[image], return_tensors="pt")
    if "mm_token_type_ids" not in inputs and hasattr(backend.processor, "create_mm_token_type_ids"):
        inputs["mm_token_type_ids"] = torch.tensor(backend.processor.create_mm_token_type_ids(inputs["input_ids"].tolist()))
    return prompt, inputs


def generate_one(backend, cfg, image_path: str, user_text: str, mode: str, ref: Optional[Dict] = None,
                 seed: int = 42) -> Tuple[Dict, Dict[str, np.ndarray]]:
    from ..attn import image_token_id, merge_size
    torch, tf, model = backend.torch, backend.transformers, backend.model
    if backend.close_id is None:
        raise RuntimeError("</think> 不是单个 token，无法一次生成")
    install(model)
    prompt, inputs = build_inputs(backend, image_path, user_text)
    ids = inputs["input_ids"][0]
    img_idx = (ids == image_token_id(backend)).nonzero().flatten()
    grid = inputs["image_grid_thw"][0].tolist()
    ms = merge_size(backend)
    expect = grid[0] * grid[1] * grid[2] // (ms * ms)
    if int(img_idx.numel()) != expect:
        raise RuntimeError(f"图像 token 数 {int(img_idx.numel())} 与 grid 推算的 {expect} 不一致")
    if ref is not None and int(ref["n_img"]) != int(img_idx.numel()):
        raise RuntimeError("参照的图像 token 数与本次输入不一致")
    dev = model.device
    batch = {k: v.to(dev) for k, v in inputs.items()}
    if "pixel_values" in batch:
        batch["pixel_values"] = batch["pixel_values"].to(model.dtype)
    n_in = int(ids.shape[0])
    gc = model.generation_config
    eos = gc.eos_token_id if isinstance(gc.eos_token_id, list) else [gc.eos_token_id]
    tb, ab = cfg.think_budget, cfg.answer_budget
    budget = make_budget(tf, torch, n_in, backend.close_id, int(eos[0]), tb, ab)
    ctrl = FPController(torch, mode, img_idx, backend.close_id, ref=ref, lam=0.0 if mode in ("observe", "base") else 1.0)
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
        with torch.inference_mode(), active(ctrl):
            out = model.generate(**batch, do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p,
                                 top_k=cfg.top_k, min_p=cfg.min_p, repetition_penalty=cfg.repetition_penalty,
                                 max_new_tokens=tb + ab + 2, use_cache=True,
                                 logits_processor=tf.LogitsProcessorList([budget]))
    finally:
        hook.remove()
    output = out[0, n_in:].tolist()
    close = output.index(backend.close_id) if backend.close_id in output else None
    tok = backend.tokenizer
    raw = tok.decode(output, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    thinking = tok.decode(output[:close], skip_special_tokens=True).strip() if close is not None else \
        tok.decode(output, skip_special_tokens=True).strip()
    answer = tok.decode(output[close + 1:], skip_special_tokens=True).strip() if close is not None else ""
    arrays = ctrl.result()
    arrays.update(grid_hw=np.array([grid[1] // ms, grid[2] // ms], np.int32),
                  grid_thw=np.array(grid, np.int32), merge_size=np.array(ms),
                  image=np.array(str(image_path)), prompt_len=np.array(n_in),
                  close_pos=np.array(-1 if close is None else n_in + close), total_len=np.array(n_in + len(output)))
    summ = ctrl.summary()
    rows = arrays["pos_image_mass"].shape[0]
    if rows != len(output):
        raise RuntimeError(f"逐 token 记录行数 {rows} 与生成 token 数 {len(output)} 不一致")
    record = {"prompt_text": prompt, "input_ids": ids.tolist(), "output_ids": output, "raw_output": raw,
              "thinking": thinking, "answer": answer, "n_in": n_in, "n_out": len(output),
              "n_img": int(img_idx.numel()), "seconds": round(time.time() - t0, 2), "seed": seed,
              "budget": budget.metadata(), "controller": summ}
    return record, arrays
