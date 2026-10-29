# -*- coding: utf-8 -*-
"""推理后端。三种调用模式：
  think    开思考：先生成思考（最多 think_budget，遇到 </think> 停），再生成可见回答（最多 answer_budget）
  nothink  不开思考，直接生成可见回答
  prefill  把 prefill_thinking 放进思考位置并关闭思考，只生成可见回答（预实验 cut3 的结构）

如果你想直接用预实验那套推理代码，只要写一个 Backend 子类实现 generate() 即可，其余代码不用改。
"""

import base64
import io
import random
import re
import threading
import time
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from PIL import Image

from .config import Config


@dataclass
class GenRequest:
    user_text: str
    images: List[Image.Image] = field(default_factory=list)
    system: Optional[str] = None
    mode: str = "think"                  # think | nothink | prefill
    prefill_thinking: Optional[str] = None
    think_budget: int = 4096
    answer_budget: int = 1024
    greedy: bool = False
    seed: int = 0


@dataclass
class GenResult:
    thinking: str
    answer: str
    thinking_closed: bool               # 思考是否在预算内自然结束
    prompt_tokens: int
    gen_tokens: int
    seconds: float
    prompt_text: str = ""               # 送进模型的文字（不含图像像素），用于存档检查


class Backend:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def generate(self, req: GenRequest) -> GenResult:
        raise NotImplementedError

    # 去掉结束标记之后的内容
    def _cut_end(self, text: str) -> str:
        cut = len(text)
        for m in self.cfg.end_markers:
            i = text.find(m)
            if i >= 0:
                cut = min(cut, i)
        return text[:cut]


# =============================================================================================
# Hugging Face transformers（单卡或 device_map=auto）
# =============================================================================================
class HFBackend(Backend):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        import torch
        import transformers
        from transformers import AutoProcessor

        self.torch = torch
        self.transformers = transformers
        self.processor = AutoProcessor.from_pretrained(cfg.model_path, trust_remote_code=True,
                                                       **(cfg.processor_kwargs or {}))
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        cls = getattr(transformers, cfg.model_class)
        ver = tuple(int(x) for x in re.findall(r"\d+", transformers.__version__)[:2])
        dkey = "dtype" if ver >= (4, 56) else "torch_dtype"       # 新版 transformers 把 torch_dtype 改名为 dtype
        kw = {dkey: getattr(torch, cfg.torch_dtype), "device_map": cfg.device_map, "trust_remote_code": True}
        if cfg.image_max_soft_tokens:
            ip = getattr(self.processor, "image_processor", None)
            if ip is None or not hasattr(ip, "max_soft_tokens"):
                raise SystemExit("image_max_soft_tokens 只适用于有 max_soft_tokens 的 processor（Gemma 4）")
            ip.max_soft_tokens = int(cfg.image_max_soft_tokens)
        if cfg.attn_implementation:
            kw["attn_implementation"] = cfg.attn_implementation
        self.model = cls.from_pretrained(cfg.model_path, **kw).eval()
        self.lock = threading.Lock()
        ids = self.tokenizer.encode(cfg.think_close, add_special_tokens=False)
        # 一次生成要求 </think> 是单个 token（Qwen 系列是）；否则退回两段式
        self.close_id = ids[0] if (cfg.hf_single_pass and len(ids) == 1) else None

    def _template(self, req: GenRequest, thinking: bool) -> str:
        msgs = []
        if req.system:
            msgs.append({"role": "system", "content": [{"type": "text", "text": req.system}]})
        content = [{"type": "image"} for _ in req.images] + [{"type": "text", "text": req.user_text}]
        msgs.append({"role": "user", "content": content})
        kw = dict(self.cfg.thinking_template_kwargs if thinking else self.cfg.nothinking_template_kwargs)
        return self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)

    def _with_think_open(self, prompt: str) -> str:
        if not self.cfg.think_prefill:
            return prompt
        if prompt.rstrip().endswith(self.cfg.think_open):
            return prompt.rstrip() + "\n"
        return prompt + self.cfg.think_open + "\n"

    def _gen(self, text: str, images, max_new: int, greedy: bool, seed: int, stop: Optional[List[str]],
             extra: Optional[Any] = None):
        torch = self.torch
        inputs = self.processor(text=[text], images=images or None, return_tensors="pt").to(self.model.device)
        n_in = int(inputs["input_ids"].shape[1])
        kw: Dict[str, Any] = dict(max_new_tokens=max_new)
        if greedy:
            kw["do_sample"] = False
        else:
            kw.update(do_sample=True, temperature=self.cfg.temperature, top_p=self.cfg.top_p,
                      top_k=self.cfg.top_k, min_p=self.cfg.min_p,
                      repetition_penalty=self.cfg.repetition_penalty)
        if stop:
            kw.update(stop_strings=stop, tokenizer=self.tokenizer)
        if extra:
            kw.update(extra(n_in) if callable(extra) else extra)
        self.transformers.set_seed(seed)
        with torch.inference_mode():
            out = self.model.generate(**inputs, **kw)
        new = out[0, n_in:]
        return self.tokenizer.decode(new, skip_special_tokens=False), n_in, int(new.shape[0])

    def generate(self, req: GenRequest) -> GenResult:
        with self.lock:
            t0 = time.time()
            c = self.cfg
            if req.mode == "nothink":
                prompt = self._template(req, thinking=False)
                txt, n_in, n_out = self._gen(prompt, req.images, req.answer_budget, req.greedy, req.seed, None)
                return GenResult("", self._cut_end(txt).strip(), True, n_in, n_out, time.time() - t0, prompt)

            base = self._with_think_open(self._template(req, thinking=True))
            if req.mode == "prefill":
                prompt = base + (req.prefill_thinking or "").strip() + "\n" + c.think_close + "\n\n"
                txt, n_in, n_out = self._gen(prompt, req.images, req.answer_budget, req.greedy, req.seed, None)
                return GenResult(req.prefill_thinking or "", self._cut_end(txt).strip(), True, n_in, n_out,
                                 time.time() - t0, prompt)

            if self.close_id is not None:
                return self._think_single_pass(req, base, t0)

            # think：两段式生成（</think> 不是单个 token 时的后备）
            t1, n_in1, n_out1 = self._gen(base, req.images, req.think_budget, req.greedy, req.seed, [c.think_close])
            t1 = self._cut_end(t1)
            closed = c.think_close in t1
            thinking = t1.split(c.think_close, 1)[0].strip()
            prompt2 = base + thinking + "\n" + c.think_close + "\n\n"
            t2, n_in2, n_out2 = self._gen(prompt2, req.images, req.answer_budget, req.greedy, req.seed + 7919, None)
            return GenResult(thinking, self._cut_end(t2).strip(), closed, n_in1 + n_in2, n_out1 + n_out2,
                             time.time() - t0, base)


    def generate_lqcd(self, req_lead: GenRequest, req_neu: GenRequest, **kw):
        from .lqcd import generate_lqcd
        with self.lock:
            return generate_lqcd(self, req_lead, req_neu, **kw)

    def _think_single_pass(self, req: GenRequest, base: str, t0: float) -> GenResult:
        """与预实验相同：一次 generate 完成思考和回答，思考到回答之间沿用同一份缓存。
        思考达到 think_budget 仍未结束时强制输出 </think>；</think> 之后最多生成 answer_budget 个 token。"""
        torch, tf, c = self.torch, self.transformers, self.cfg
        close_id, tb, ab = self.close_id, req.think_budget, req.answer_budget
        state = {"closed_at": None, "forced": False}

        def find_close(gen):
            if state["closed_at"] is None:
                hit = (gen == close_id).nonzero()
                if hit.numel():
                    state["closed_at"] = int(hit[0, 0])
            return state["closed_at"]

        class ThinkBudget(tf.LogitsProcessor):
            def __init__(self, n_in):
                self.n_in = n_in

            def __call__(self, input_ids, scores):
                gen = input_ids[0, self.n_in:]
                if find_close(gen) is None and gen.shape[0] >= tb:
                    scores = torch.full_like(scores, float("-inf"))
                    scores[:, close_id] = 0.0
                    state["forced"] = True
                return scores

        class AnswerBudget(tf.StoppingCriteria):
            def __init__(self, n_in):
                self.n_in = n_in

            def __call__(self, input_ids, scores, **kw):
                gen = input_ids[0, self.n_in:]
                at = find_close(gen)
                done = at is not None and gen.shape[0] - at - 1 >= ab
                return torch.full((input_ids.shape[0],), done, dtype=torch.bool, device=input_ids.device)

        def extra(n_in):
            return {"logits_processor": tf.LogitsProcessorList([ThinkBudget(n_in)]),
                    "stopping_criteria": tf.StoppingCriteriaList([AnswerBudget(n_in)])}

        txt, n_in, n_out = self._gen(base, req.images, tb + 1 + ab, req.greedy, req.seed, None, extra)
        txt = self._cut_end(txt)
        if c.think_close in txt:
            thinking, answer = txt.split(c.think_close, 1)
        else:
            thinking, answer = txt, ""
        closed = state["closed_at"] is not None and not state["forced"]
        return GenResult(thinking.strip(), answer.strip(), closed, n_in, n_out, time.time() - t0, base)


# =============================================================================================
# vLLM OpenAI 兼容服务（服务端不要加 --reasoning-parser，让思考以 <think>…</think> 原样出现）
# =============================================================================================
class OpenAIBackend(Backend):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        from openai import OpenAI
        self.client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=3600)

    @staticmethod
    def _url(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def _call(self, req: GenRequest, thinking: bool, assistant_prefix: Optional[str], max_tokens: int,
              stop: Optional[List[str]], seed: int):
        msgs = []
        if req.system:
            msgs.append({"role": "system", "content": req.system})
        content = [{"type": "image_url", "image_url": {"url": self._url(im)}} for im in req.images]
        content.append({"type": "text", "text": req.user_text})
        msgs.append({"role": "user", "content": content})
        kw = dict(self.cfg.thinking_template_kwargs if thinking else self.cfg.nothinking_template_kwargs)
        extra: Dict[str, Any] = {"chat_template_kwargs": kw}
        if assistant_prefix is not None:
            msgs.append({"role": "assistant", "content": assistant_prefix})
            extra.update(continue_final_message=True, add_generation_prompt=False)
        args: Dict[str, Any] = dict(model=self.cfg.model_path, messages=msgs, max_tokens=max_tokens, seed=seed,
                                    extra_body=extra)
        if req.greedy:
            args["temperature"] = 0.0
        else:
            args.update(temperature=self.cfg.temperature, top_p=self.cfg.top_p)
            extra.update(top_k=self.cfg.top_k, min_p=self.cfg.min_p,
                         repetition_penalty=self.cfg.repetition_penalty)
        if stop:
            args["stop"] = stop
        r = self.client.chat.completions.create(**args)
        return (r.choices[0].message.content or ""), r.usage.prompt_tokens, r.usage.completion_tokens, \
            r.choices[0].finish_reason

    def generate(self, req: GenRequest) -> GenResult:
        t0 = time.time()
        c = self.cfg
        if req.mode == "nothink":
            txt, a, b, _ = self._call(req, False, None, req.answer_budget, None, req.seed)
            return GenResult("", txt.strip(), True, a, b, time.time() - t0, req.user_text)
        opener = c.think_open + "\n"
        if req.mode == "prefill":
            pre = opener + (req.prefill_thinking or "").strip() + "\n" + c.think_close + "\n\n"
            txt, a, b, _ = self._call(req, True, pre, req.answer_budget, None, req.seed)
            return GenResult(req.prefill_thinking or "", txt.strip(), True, a, b, time.time() - t0, req.user_text)
        t1, a1, b1, fin = self._call(req, True, opener, req.think_budget, [c.think_close], req.seed)
        closed = fin == "stop"
        thinking = t1.split(c.think_close, 1)[0].strip()
        pre = opener + thinking + "\n" + c.think_close + "\n\n"
        t2, a2, b2, _ = self._call(req, True, pre, req.answer_budget, None, req.seed + 7919)
        return GenResult(thinking, t2.strip(), closed, a1 + a2, b1 + b2, time.time() - t0, req.user_text)


# =============================================================================================
# Mock：不需要模型，只用来检查流程（数字没有意义）
# =============================================================================================
class MockBackend(Backend):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.lock = threading.Lock()

    @staticmethod
    def _rng(req: GenRequest) -> random.Random:
        key = "|".join([req.user_text, str(req.seed), req.mode, req.prefill_thinking or ""])
        return random.Random(zlib.crc32(key.encode("utf-8")))

    def generate_lqcd(self, req_lead: GenRequest, req_neu: GenRequest, record: bool = False, **kw):
        """mock：用中性分支的输出代替；记录注意力时给出形状正确的随机数组。"""
        import numpy as np
        r = self.generate(GenRequest(user_text=req_neu.user_text, images=req_neu.images, mode="think",
                                     think_budget=req_neu.think_budget, answer_budget=req_neu.answer_budget,
                                     seed=req_lead.seed))
        info = {"alpha_mean": kw.get("alpha0", 1.0), "jsd_mean": 0.1, "n_tokens": r.gen_tokens,
                "forced_close": False, "neutral_prompt": req_neu.user_text}
        if record:
            rng = np.random.default_rng(zlib.crc32(req_lead.user_text.encode()))
            info["attn"] = {}
            for b in ("lead", "neutral"):
                d = {"layer_ids": np.arange(3, 64, 4), "n_img": 16, "grid_hw": np.array([4, 4]),
                     "prompt_len": 100, "close_pos": 150, "total_len": 200}
                for st in ("prefill_end", "think", "answer"):
                    d[st] = rng.random((16, 4, 16)).astype(np.float32) * 0.01
                    d["count_" + st] = 1.0
                info["attn"][b] = d
        return r, info

    def generate(self, req: GenRequest) -> GenResult:
        r = self._rng(req)
        t = req.user_text
        yn = lambda: r.choice(["yes", "no"])
        if "Answer the question by comparing the possible answers fairly." in t:
            a = r.choice(["yes", "no", "yes", "no", "undecidable"])
            rej = "no" if a == "yes" else "yes"
            cell = r.choice(["A1", "B2", "C3", "A1, A2", "whole image"])
            block = (
                "[IDENTIFICATION]\n- tissue: liver | ruled out: kidney because no glomeruli; lung because no alveoli\n"
                "[CRITERIA]\n- necrosis means loss of nuclear detail with eosinophilic debris\n"
                f"[KEY OBSERVATIONS]\n- pink debris without nuclei | where: {cell} | supports: {a}\n"
                f"[CONCLUSION]\nanswer: {a} | confidence: {r.choice(['high', 'medium', 'low'])}\n"
                "[REJECTION REASONS]\n"
                f"- rejected: {rej} | type: missing expected feature | claim: there is no karyorrhexis in {cell} | "
                f"opposite: there is karyorrhexis in {cell} | where: {cell} | confidence: medium\n"
                f"- rejected: {rej} | type: criterion | claim: necrosis requires loss of nuclei | "
                "opposite: necrosis does not require loss of nuclei | where: whole image | confidence: low"
            )
            if r.random() < 0.05:
                block = "I could not decide."
            return GenResult("Comparing both answers with the same standard (mock).", block, True, 900, 600, 0.0, t)
        if "For each pair below" in t:
            n = len(re.findall(r"^P\d+ ", t, re.M))
            lines = [f"P{i + 1}: {r.choice(['a', 'b', 'cannot determine'])} - mock observation" for i in range(n)]
            return GenResult("", "\n".join(lines), True, 700, 40, 0.0, t)
        if "Observations from an earlier examination" in t:
            a = r.choice(["yes", "no", "undecidable"])
            return GenResult("Re-judging (mock).",
                             f"[KEY OBSERVATIONS]\n- re-checked debris | where: B2 | supports: {a}\n"
                             f"[CONCLUSION]\nanswer: {a} | confidence: medium", True, 1200, 300, 0.0, t)
        if "<Image Textualization>" in t:
            a = yn()
            m = re.search(r"answer is (yes|no)", t, re.I)
            if m and r.random() < 0.5:
                a = m.group(1).lower()
            return GenResult("mock", f"<Image Textualization> tissue </Image Textualization>\n<Reflection> ... "
                                     f"</Reflection>\n<Summarization> {a.capitalize()} </Summarization>",
                             True, 800, 500, 0.0, t)
        if req.mode == "prefill":
            m = re.search(r"Final conclusion: (yes|no)", req.prefill_thinking or "")
            if m:
                a = m.group(1)
                if r.random() < 0.1:
                    a = "no" if a == "yes" else "yes"
                return GenResult(req.prefill_thinking or "", f"{a.capitalize()}. The evidence is ... (mock)",
                                 True, 3000, 120, 0.0, t)
            return GenResult(req.prefill_thinking or "", "The image does not allow a decision. (mock)",
                             True, 3000, 60, 0.0, t)
        a = yn()
        m = re.search(r"answer is (yes|no)", t, re.I)
        if m and r.random() < 0.6:
            a = m.group(1).lower()
        return GenResult("mock thinking", f"{a.capitalize()}. (mock)", True, 600, 800, 0.0, t)


def make_backend(cfg: Config) -> Backend:
    if cfg.backend == "hf":
        return HFBackend(cfg)
    if cfg.backend == "openai":
        return OpenAIBackend(cfg)
    if cfg.backend == "mock":
        return MockBackend(cfg)
    raise ValueError(f"unknown backend {cfg.backend}")
