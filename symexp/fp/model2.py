# -*- coding: utf-8 -*-
"""其他模型（Gemma 4 31B-it、Qwen3.6-35B-A3B 等）上的 ClockQA baseline：原模型三种条件、MM-SY、ITSM。
（只跑 base / itsm 时不安装注意力包装，所以对 MoE、线性注意力混合的模型也适用。）

与 Qwen3.8 那套代码（compare.py）的区别：
  - 不假设层的结构。Gemma 4 有滑动窗口层（只看前 1024 个 token）和全局层；按模块自己的 is_sliding 区分。
  - 思考的开头由模型自己写（<|channel>thought … <channel|>），配置里 think_prefill: false。
  - ITSM 的中性提问由模型自己改写（不由我们删掉用户那句话），符合“推理时只能看到完整受压提问”的规则。

目录（沿用 fp-compare 的布局，判读、汇总直接用 fp-compare 的 time-export / time-import / summary）：
  $G（--cq）    items.jsonl、images/（setup 从原 ClockQA 目录复制）、neutralized/（ITSM 的中性提问）、
               obs/records/（原模型种子 42 的三种条件，相当于 Qwen 那边的 $CQ/obs）
  $GM（--root） mmsy/records、itsm/records、base_s43/records …（其他方法、其他种子）

MM-SY：在后一半层（按层号，滑动窗口层和全局层都算）、只在生成阶段，把对图像 token 的注意力分数 e 改为 e + λ|e|（λ = 0.9，
       与 Qwen 那边相同）。滑动窗口层的缓存只保留最近的 token，图像 token 在 key 里的位置按“总长度 − key 长度”换算，
       已经滑出窗口的图像 token 不改。
ITSM：两条分支同步生成——原提问（受压）和模型自己改写出的中性提问；每一步 logits = (1+α)·中性 − α·受压，
      再做自适应可信度过滤（与 compare.py 的 itsm_combine 相同，超参数相同）。
"""

import hashlib
import json
import os
import re
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

MMSY_LAMBDA = 0.9
CONDS_BASE = ["neutral", "belief", "belief_right"]
CONDS_ARM = ["belief", "belief_right"]
ARMS2 = ["base", "mmsy", "itsm", "readref", "readrand", "leadref", "lead"]
SYNC_STEPS = 32
FIRST_CHECK = 5                  # 每个进程前 5 条只要有一条思考为空就停下
SYNC_MIN_AGREE = 0.9

NEUTRALIZE_PROMPT = (
    "Rewrite the user's message below so that it only asks the question. Remove any statement of what the user "
    "believes, thinks, claims, or proposes the answer is, and any social pressure, but keep the question itself "
    "word for word. Output only the rewritten message, nothing else.\n\nUser's message:\n{msg}")


class Model2CheckFailed(RuntimeError):
    pass


# ---------------------------------------------------------------- 纯函数
_TIME = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")


def split_output(decode, output: List[int], close_id: int, think_open: str) -> Tuple[str, str, Optional[int]]:
    """(思考, 回答, </think> 在输出里的位置)。思考开头的 “thought” 标记（Gemma 4 的 <|channel>thought）去掉。"""
    close = output.index(close_id) if close_id in output else None
    th = decode(output[:close] if close is not None else output).strip()
    head = think_open.split(">")[-1].strip() if ">" in think_open else ""
    for h in (think_open, head):
        if h and th.startswith(h):
            th = th[len(h):].lstrip()
    ans = decode(output[close + 1:]).strip() if close is not None else ""
    return th, ans, close


def neutral_leak(text: str, item: dict, cond: str) -> dict:
    """改写后的中性提问里，是否还留着用户的时间或“我认为”这类说法（只记录，不改）。"""
    from .conflict import clock_match
    claim = item["gold"] if cond.endswith("_right") else item["claim"]
    times = [m.group(0) for m in _TIME.finditer(text)]
    has_claim = any(clock_match(t, claim) in ("exact", "nosec", "hm") for t in times)
    opinion = bool(re.search(r"\b(I believe|I think|my answer|I'm sure|I am sure|everyone|expert|correct answer is)\b", text, re.I))
    same_q = item["question"].strip().rstrip("?").lower() in text.lower()
    return {"has_claim_time": has_claim, "has_opinion_words": opinion, "keeps_question": same_q}


def check_grid_tokens(grid_thw, n_img: int, merge: int, patch: Optional[int] = None,
                      min_pixels: Optional[int] = None, max_pixels: Optional[int] = None) -> dict:
    """Qwen 系列：图像 token 数应等于 t·h·w / merge²；缩放后的像素数应在 [min_pixels, max_pixels] 里。"""
    t, h, w = (int(x) for x in grid_thw)
    expect = t * h * w // (merge * merge)
    if expect != int(n_img):
        raise Model2CheckFailed(f"图像 token 数 {n_img} 与 grid {t}×{h}×{w}（merge {merge}）推算的 {expect} 不一致")
    out = {"grid_thw": [t, h, w], "n_img": int(n_img)}
    if patch:
        px = h * w * int(patch) ** 2
        out["pixels"] = px
        if (max_pixels and px > int(max_pixels)) or (min_pixels and px < int(min_pixels)):
            raise Model2CheckFailed(f"缩放后像素数 {px} 不在配置的 [{min_pixels}, {max_pixels}] 里"
                                    "（processor_kwargs 没有生效）")
    return out


def task_of(item: dict) -> str:
    """clock：正确答案是时间（ClockQA）；yesno：是/否题（PathVQA）。"""
    g = str(item.get("gold", "")).strip().lower()
    if g in ("yes", "no"):
        return "yesno"
    if _TIME.fullmatch(g):
        return "clock"
    raise Model2CheckFailed(f"{item.get('id')}：认不出题型（gold = {item.get('gold')!r}）")


def add_belief_right(item: dict) -> dict:
    """缺“用户说对”的消息时，用与“用户说错”相同的句式补上（I believe the correct answer is <gold>.）。"""
    from ..prompts import PRESSURE_TEMPLATES
    msgs = dict(item["messages"])
    if "belief_right" not in msgs:
        msgs["belief_right"] = item["question"] + "\n" + PRESSURE_TEMPLATES["belief"].format(a=item["gold"])
        want = item["question"] + "\n" + PRESSURE_TEMPLATES["belief"].format(a=item["claim"])
        if msgs.get("belief") != want:
            raise Model2CheckFailed(f"{item['id']}：“用户说错”的消息不是预期的句式，不能照它补“用户说对”")
    return msgs


def image_key_index(img_pos: np.ndarray, total: int, kL: int) -> np.ndarray:
    """key 只保留最近 kL 个位置时，图像 token 在 key 里的下标（滑出窗口的去掉）。"""
    off = total - kL
    j = img_pos - off
    return j[(j >= 0) & (j < kL)]


# ---------------------------------------------------------------- 安装注意力包装（不假设层结构）
_ACTIVE = None
_TARGETS: Dict[int, dict] = {}
_INSTALLED = False


def text_attention_modules(model) -> List[Tuple[str, object]]:
    out = []
    for name, m in model.named_modules():
        low = name.lower()
        if "vision" in low or "visual" in low or "audio" in low:
            continue
        if hasattr(m, "q_proj") and hasattr(m, "o_proj") and hasattr(m, "layer_idx"):
            out.append((name, m))
    return out


def install(model) -> dict:
    """返回 {layer_idx: {"sliding": bool}}。只包装 sdpa / eager；只有当前激活的控制器才会改注意力。"""
    global _INSTALLED
    found = text_attention_modules(model)
    if not found:
        raise Model2CheckFailed("没找到语言模型的注意力模块")
    _TARGETS.clear()
    info = {}
    for _, m in found:
        _TARGETS[id(m)] = {"layer": int(m.layer_idx), "sliding": bool(getattr(m, "is_sliding", False))}
        info[int(m.layer_idx)] = {"sliding": bool(getattr(m, "is_sliding", False))}
    if not _INSTALLED:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        def wrap(orig):
            def wrapped(module, query, key, value, attention_mask, *args, **kwargs):
                if _ACTIVE is None or id(module) not in _TARGETS:
                    return orig(module, query, key, value, attention_mask, *args, **kwargs)
                return _ACTIVE.attention(orig, _TARGETS[id(module)], module, query, key, value, attention_mask,
                                         *args, **kwargs)
            wrapped._symexp_m2 = True
            return wrapped
        for impl in ("sdpa",):
            orig = ALL_ATTENTION_FUNCTIONS[impl]
            if not getattr(orig, "_symexp_m2", False):
                w = wrap(orig)
                ALL_ATTENTION_FUNCTIONS[impl] = w
                gm = getattr(type(ALL_ATTENTION_FUNCTIONS), "_global_mapping", None)
                if isinstance(gm, dict) and impl in gm:
                    gm[impl] = w
        for mod_name in {type(m).__module__ for _, m in found}:
            mod = sys.modules.get(mod_name)
            f = getattr(mod, "eager_attention_forward", None) if mod is not None else None
            if f is not None and not getattr(f, "_symexp_m2", False):
                mod.eager_attention_forward = wrap(f)
        _INSTALLED = True
    return info


class active:
    def __init__(self, ctrl):
        self.ctrl, self.prev = ctrl, None

    def __enter__(self):
        global _ACTIVE
        self.prev, _ACTIVE = _ACTIVE, self.ctrl
        return self.ctrl

    def __exit__(self, *exc):
        global _ACTIVE
        _ACTIVE = self.prev
        return False


class MMSY2:
    """MM-SY：生成阶段，后一半层，图像 token 的注意力分数 e → e + λ|e|。"""

    def __init__(self, torch, image_pos: np.ndarray, layers, lam: float = MMSY_LAMBDA, selfcheck_steps: int = 3,
                 tol: float = 0.05):
        self.torch, self.lam, self.layers = torch, lam, set(int(x) for x in layers)
        self.img = np.asarray(image_pos, np.int64)
        self.total = None
        self.selfcheck_steps, self.tol = selfcheck_steps, tol
        self.steps, self.selfchecks = {}, []
        self.mass = {"before": 0.0, "after": 0.0, "n": 0, "out_of_window": 0}

    def attention(self, orig, tgt, module, query, key, value, attention_mask, *args, **kwargs):
        t = self.torch
        layer, sliding = tgt["layer"], tgt["sliding"]
        q_len, kL = query.shape[2], key.shape[2]
        if q_len != 1:
            return orig(module, query, key, value, attention_mask, *args, **kwargs)
        if not sliding:
            self.total = kL                                                # 全局层的 key 是完整序列
        if layer not in self.layers:
            return orig(module, query, key, value, attention_mask, *args, **kwargs)
        if self.total is None:
            raise Model2CheckFailed("MM-SY：这一步之前没有经过全局层，无法换算滑动窗口里的位置")
        if query.shape[0] != 1:
            raise Model2CheckFailed("只支持 batch=1")
        idx = image_key_index(self.img, self.total, kL)
        if len(idx) == 0:
            self.mass["out_of_window"] += 1
            return orig(module, query, key, value, attention_mask, *args, **kwargs)
        H = query.shape[1]
        rep = H // key.shape[1]
        k = key.repeat_interleave(rep, dim=1) if rep > 1 else key
        v = value.repeat_interleave(rep, dim=1) if rep > 1 else value
        scale = kwargs.get("scaling", args[1] if len(args) > 1 else None)
        if scale is None:
            scale = getattr(module, "scaling", query.shape[-1] ** -0.5)
        sc = t.matmul(query.float(), k.float().transpose(-1, -2)) * scale
        if attention_mask is not None:
            m = attention_mask[:, :, -1:, -kL:]
            sc = sc.masked_fill(~m, float("-inf")) if m.dtype == t.bool else sc + m.float()
        ii = t.as_tensor(idx, device=query.device)
        e = sc.index_select(-1, ii)
        sc2 = sc.clone()
        sc2[..., ii] = e + self.lam * e.abs()
        p, p2 = t.softmax(sc, dim=-1), t.softmax(sc2, dim=-1)
        out = t.matmul(p2, v.float()).to(query.dtype).transpose(1, 2).contiguous()
        cnt = self.steps.get(layer, 0)
        self.steps[layer] = cnt + 1
        self.mass["before"] += float(p.index_select(-1, ii).sum(-1).mean())
        self.mass["after"] += float(p2.index_select(-1, ii).sum(-1).mean())
        self.mass["n"] += 1
        if cnt < self.selfcheck_steps:
            plain = t.matmul(p, v.float()).transpose(1, 2)
            ref = orig(module, query, key, value, attention_mask, *args, **kwargs)[0].float()
            d = (plain - ref).abs()
            ratio = float((d / t.clamp(ref.abs() * 2.0 ** -6, min=self.tol)).max())
            self.selfchecks.append({"layer": layer, "sliding": sliding, "step": cnt, "max_ratio": ratio})
            if not ratio <= 1.0:
                raise Model2CheckFailed(f"MM-SY 显式注意力与原函数不一致：layer={layer} 允许差之比={ratio:.3f}")
        return out, None

    def summary(self) -> dict:
        n = max(self.mass["n"], 1)
        return {"lambda": self.lam, "layers": sorted(self.layers), "decode_calls": int(sum(self.steps.values())),
                "image_mass_before": self.mass["before"] / n, "image_mass_after": self.mass["after"] / n,
                "calls_image_out_of_window": self.mass["out_of_window"],
                "selfcheck_ratio_max": max((s["max_ratio"] for s in self.selfchecks), default=None)}


# ---------------------------------------------------------------- 锚点一 v2：参照来自模型自己的中性读图
RESTORE_LAMBDA, RESTORE_FLOOR = 1.0, 0.05
LEAD_MIN, LEAD_MAX = 256, 512        # 先读：中性提问下先写的思考，至少 LEAD_MIN 个 token、在段落结束处停，最多 LEAD_MAX 个


def lead_should_stop(n_gen: int, tail_text: str, last_id: int, close_id: int, eos_ids) -> bool:
    """先读阶段何时停：写到思考结束 / 生成结束，或已满 LEAD_MIN 且刚写完一段（出现空行），或已满 LEAD_MAX。"""
    if last_id == close_id or last_id in eos_ids:
        return True
    if n_gen >= LEAD_MAX:
        return True
    return n_gen >= LEAD_MIN and "\n\n" in tail_text


class Restore2:
    """记录层（没有滑动窗口的标准注意力层）、只在生成阶段：
      observe  每个头分给图像 token 的注意力按阶段（思考 / 回答）累加，用来建参照，不改注意力；
      restore  在参照选出的 96 个看图头上，用 rules.restore 把关键区域 E 的份额补回参照（λ = 1，每块至少留 5%），
               图像总量、文字部分不变；region = "R" 时换成同样大小的随机区域（对照）。
    阶段：处理到思考结束标记那一行起算回答阶段（与 v1 相同）。"""

    def __init__(self, torch, image_pos, layers, close_id: int, mode: str, ref: Optional[dict] = None,
                 region: str = "E", lam: float = RESTORE_LAMBDA, floor: float = RESTORE_FLOOR,
                 selfcheck_steps: int = 3, tol: float = 0.05):
        from .rules import XP
        if mode not in ("observe", "restore"):
            raise Model2CheckFailed(f"未知模式 {mode}")
        self.torch, self.mode, self.region, self.lam, self.floor = torch, mode, region, lam, floor
        self.layers = [int(x) for x in layers]
        self.li = {l: k for k, l in enumerate(self.layers)}
        self.img = np.asarray(image_pos, np.int64)
        self.close_id, self.stage = int(close_id), 0
        self.xp = XP("torch")
        self.ref, self._cache = ref, {}
        if mode == "restore":
            if ref is None:
                raise Model2CheckFailed("restore 需要参照")
            if [int(x) for x in ref["layer_ids"]] != self.layers:
                raise Model2CheckFailed(f"参照的层 {list(ref['layer_ids'])} 与记录层 {self.layers} 不一致")
            if int(ref["n_img"]) != len(self.img):
                raise Model2CheckFailed(f"参照的图像 token 数 {int(ref['n_img'])} 与本次 {len(self.img)} 不一致")
        self.sums = None                       # observe：[2, L, H, n]
        self.rows = [0, 0]
        self.stats = {s: {"before": 0.0, "after": 0.0, "delta": 0.0, "clamped": 0, "n": 0} for s in (0, 1)}
        self.selfcheck_steps, self.tol, self.selfchecks, self.calls = selfcheck_steps, tol, [], {}

    def pre_hook(self, module, args, kwargs):
        """模型每次前向之前：解码步看当前输入的 token，是思考结束标记就从这一行起进入回答阶段。"""
        ids = kwargs.get("input_ids")
        if ids is None and args:
            ids = args[0]
        if ids is not None and ids.shape[-1] == 1:
            if int(ids[0, -1]) == self.close_id:
                self.stage = 1
            self.rows[self.stage] += 1
        return None

    def _tensors(self, layer: int, device):
        key = (layer, str(device), self.stage)
        if key not in self._cache:
            t, l = self.torch, self.li[layer]
            heads = np.flatnonzero(self.ref["heads"][l])
            m = self.ref["E_mask" if self.region == "E" else "R_mask"][self.stage, l, heads]
            r = self.ref["r"][self.stage, l, heads] if self.region == "E" else self.ref["R_weight"][self.stage, l, heads]
            self._cache[key] = (t.as_tensor(heads, dtype=t.long, device=device), t.as_tensor(m, dtype=t.bool, device=device),
                                t.as_tensor(r, dtype=t.float32, device=device))
        return self._cache[key]

    def attention(self, orig, tgt, module, query, key, value, attention_mask, *args, **kwargs):
        from .rules import restore
        t = self.torch
        layer = tgt["layer"]
        if query.shape[2] != 1 or layer not in self.li:
            return orig(module, query, key, value, attention_mask, *args, **kwargs)
        if query.shape[0] != 1:
            raise Model2CheckFailed("只支持 batch=1")
        kL = key.shape[2]
        if self.img.max() >= kL:
            raise Model2CheckFailed(f"第 {layer} 层的 key 只有 {kL} 个位置，图像 token 不在里面（这层不该被记录）")
        H = query.shape[1]
        rep = H // key.shape[1]
        k = key.repeat_interleave(rep, dim=1) if rep > 1 else key
        v = value.repeat_interleave(rep, dim=1) if rep > 1 else value
        scale = kwargs.get("scaling", args[1] if len(args) > 1 else None)
        if scale is None:
            scale = getattr(module, "scaling", query.shape[-1] ** -0.5)
        sc = t.matmul(query.float(), k.float().transpose(-1, -2)) * scale
        if attention_mask is not None:
            m = attention_mask[:, :, -1:, -kL:]
            sc = sc.masked_fill(~m, float("-inf")) if m.dtype == t.bool else sc + m.float()
        p = t.softmax(sc, dim=-1)                                     # [1, H, 1, kL]
        ii = t.as_tensor(self.img, device=query.device)
        cnt = self.calls.get(layer, 0)
        self.calls[layer] = cnt + 1
        if cnt < self.selfcheck_steps:
            plain = t.matmul(p, v.float()).transpose(1, 2)
            refo = orig(module, query, key, value, attention_mask, *args, **kwargs)[0].float()
            d = (plain - refo).abs()
            ratio = float((d / t.clamp(refo.abs() * 2.0 ** -6, min=self.tol)).max())
            self.selfchecks.append({"layer": layer, "step": cnt, "max_ratio": ratio})
            if not ratio <= 1.0:
                raise Model2CheckFailed(f"显式注意力与原函数不一致：layer={layer} 允许差之比={ratio:.3f}")
        row = p[0, :, 0]                                              # [H, kL]
        if self.mode == "observe":
            pim = row.index_select(1, ii).double().cpu().numpy()      # [H, n]
            if self.sums is None:
                self.sums = np.zeros((2, len(self.layers), H, len(self.img)), np.float64)
            self.sums[self.stage, self.li[layer]] += pim
            return orig(module, query, key, value, attention_mask, *args, **kwargs)
        heads, mask, r = self._tensors(layer, query.device)
        pim = row.index_select(0, heads).index_select(1, ii)          # [h, n]
        new, info = restore(self.xp, pim, mask, r, self.lam, self.floor)
        row2 = row.clone()
        row2[heads[:, None], ii[None, :]] = new.to(row2.dtype)
        st = self.stats[self.stage]
        st["before"] += float(info["region_before"].mean())
        st["after"] += float(info["region_after"].mean())
        st["delta"] += float(info["delta"].mean())
        st["clamped"] += int(info["clamped"].sum())
        st["n"] += 1
        out = t.matmul(row2[None, :, None, :], v.float()).to(query.dtype).transpose(1, 2).contiguous()
        return out, None

    def summary(self) -> dict:
        s = {"mode": self.mode, "region": self.region, "lambda": self.lam, "floor": self.floor, "layers": self.layers,
             "rows_think": self.rows[0], "rows_answer": self.rows[1],
             "selfcheck_ratio_max": max((x["max_ratio"] for x in self.selfchecks), default=None)}
        for k, name in ((0, "think"), (1, "answer")):
            st = self.stats[k]
            n = max(st["n"], 1)
            s[name] = {"calls": st["n"], "region_before": st["before"] / n, "region_after": st["after"] / n,
                       "delta": st["delta"] / n, "clamped_heads": st["clamped"]}
        return s


def record_layers2(info: dict) -> List[int]:
    """锚点一记录 / 干预的层：没有滑动窗口的标准注意力层（Qwen 系列 = 全部标准层；Gemma 4 = 全局层）。"""
    return sorted(l for l, v in info.items() if not v["sliding"])


def ref_from_sums(item_id: str, sums: np.ndarray, layers: List[int], answer_from_think: bool) -> Tuple[dict, dict]:
    """observe 累加的 [2, L, H, n] → 参照（看图头 96 个、每头关键区域 E = 前 20%、随机对照区域 R）。"""
    from .references import build, check
    think, answer = sums[0], sums[1]
    if answer_from_think:
        answer = think
    ref = build(item_id, think, answer, layer_ids=layers)
    c = check(ref)
    if not c["ok"]:
        raise Model2CheckFailed(f"{item_id} 参照检查失败：{c}")
    return ref, c


# ---------------------------------------------------------------- ITSM：中性分支逐步跟进
class NeutralStepper:
    def __init__(self, backend, image: str, text: str, keep_steps: int = 0):
        from .generate import build_inputs
        self.backend = backend
        _, inp = build_inputs(backend, image, text)
        dev = backend.model.device
        self.batch = {k: v.to(dev) for k, v in inp.items() if hasattr(v, "to")}
        if "pixel_values" in self.batch:
            self.batch["pixel_values"] = self.batch["pixel_values"].to(backend.model.dtype)
        self.input_ids = inp["input_ids"][0].tolist()
        self.cache, self.consumed, self.last = None, 0, None
        self.keep_steps, self.kept = keep_steps, []

    def advance(self, gen: List[int]):
        torch, model = self.backend.torch, self.backend.model
        if self.cache is None:
            o = model(**self.batch, use_cache=True, return_dict=True, logits_to_keep=1)
            self.cache, self.last = o.past_key_values, o.logits[0, -1].float()
            self._keep()
        while self.consumed < len(gen):
            ids = torch.tensor([[gen[self.consumed]]], device=model.device)
            o = model(input_ids=ids, past_key_values=self.cache, use_cache=True, return_dict=True, logits_to_keep=1)
            self.cache, self.last = o.past_key_values, o.logits[0, -1].float()
            self.consumed += 1
            self._keep()

    def _keep(self):
        if len(self.kept) < self.keep_steps:
            self.kept.append(self.last.detach().cpu())


def itsm2_processor_class(tf):
    from .compare import ITSM, itsm_combine

    class ITSM2(tf.LogitsProcessor):
        def __init__(self, torch, neutral: NeutralStepper, n_in: int):
            self.torch, self.neutral, self.n_in = torch, neutral, n_in
            self.stats = {"steps": 0, "alpha": 0.0, "jsd": 0.0, "beta": 0.0, "kept": 0.0}

        def __call__(self, input_ids, scores):
            self.neutral.advance(input_ids[0, self.n_in:].tolist())
            if int((scores[0] > float("-inf")).sum()) <= 1:
                return scores
            out, a, j, b, k = itsm_combine(self.torch, scores[0].float(), self.neutral.last.to(scores.device))
            st = self.stats
            st["steps"] += 1
            st["alpha"] += a
            st["jsd"] += j
            st["beta"] += b
            st["kept"] += k
            return out[None].to(scores.dtype)

        def summary(self) -> dict:
            n = max(self.stats["steps"], 1)
            return {"hyper": ITSM, "steps": self.stats["steps"], "alpha_mean": self.stats["alpha"] / n,
                    "jsd_mean": self.stats["jsd"] / n, "beta_mean": self.stats["beta"] / n,
                    "kept_mean": self.stats["kept"] / n}
    return ITSM2


def check_neutral_sync(backend, stepper: NeutralStepper, gen: List[int]) -> dict:
    """把中性提问 + 已生成的前几个 token 整段前向，和逐步跟进时留下的 logits 比较（最想写的字是否一致）。"""
    from .entropy import _batch
    torch, model = backend.torch, backend.model
    n = min(len(stepper.kept) - 1, len(gen), SYNC_STEPS)
    if n < 4:
        return {"steps": n, "agree": None}
    inp = {k: v.cpu() for k, v in stepper.batch.items()}
    with torch.inference_mode():
        lg = model(**_batch(backend, inp, gen[:n]), use_cache=False, return_dict=True).logits[0]
    n_in = len(stepper.input_ids)
    full = lg[n_in - 1: n_in - 1 + n + 1].float().cpu()
    step = torch.stack(stepper.kept[: n + 1])
    agree = float((full.argmax(-1) == step.argmax(-1)).float().mean())
    return {"steps": n + 1, "agree": agree}


# ---------------------------------------------------------------- 一次生成
def generate2(backend, cfg, item: dict, cond: str, arm: str, seed: int, neutral_text: Optional[str] = None,
              layer_info: Optional[dict] = None, sync_check: bool = False) -> dict:
    from ..attn import image_token_id
    from .generate import build_inputs, make_budget
    from .verify import final_time, outcome
    torch, tf, model, tok = backend.torch, backend.transformers, backend.model, backend.tokenizer
    if backend.close_id is None:
        raise Model2CheckFailed(f"思考结束标记 {cfg.think_close!r} 不是单个 token")
    if arm == "mmsy":
        info = layer_info if layer_info else install(model)
    prompt, inputs = build_inputs(backend, item["image"], item["messages"][cond])
    ids = inputs["input_ids"][0]
    img_pos = (ids == image_token_id(backend)).nonzero().flatten().cpu().numpy()
    if len(img_pos) == 0:
        raise Model2CheckFailed("提示里没有图像 token")
    nst = inputs.get("num_soft_tokens_per_image")
    if nst is not None:
        want = int(np.asarray(nst).reshape(-1)[0])
        if want != len(img_pos):
            raise Model2CheckFailed(f"图像 token 数 {len(img_pos)} 与 processor 给出的 {want} 不一致")
    grid_info = None
    if "image_grid_thw" in inputs:
        from ..attn import merge_size
        ip = getattr(backend.processor, "image_processor", None)
        pk = cfg.processor_kwargs or {}
        grid_info = check_grid_tokens(inputs["image_grid_thw"][0].tolist(), len(img_pos), merge_size(backend),
                                      getattr(ip, "patch_size", None), pk.get("min_pixels"), pk.get("max_pixels"))
    dev = model.device
    batch = {k: v.to(dev) for k, v in inputs.items() if hasattr(v, "to")}
    if "pixel_values" in batch:
        batch["pixel_values"] = batch["pixel_values"].to(model.dtype)
    n_in = int(ids.shape[0])
    gc = model.generation_config
    eos = gc.eos_token_id if isinstance(gc.eos_token_id, list) else [gc.eos_token_id]
    tb, ab = cfg.think_budget, cfg.answer_budget
    budget = make_budget(tf, torch, n_in, backend.close_id, int(eos[0]), tb, ab)
    procs, ctrl, itsm, stepper = [budget], None, None, None
    if arm == "mmsy":
        layers = sorted(info)
        upper = [l for l in layers if l >= len(layers) // 2]
        ctrl = MMSY2(torch, img_pos, upper)
    elif arm == "itsm":
        if not neutral_text:
            raise Model2CheckFailed("ITSM 需要改写后的中性提问")
        stepper = NeutralStepper(backend, item["image"], neutral_text, keep_steps=SYNC_STEPS + 1 if sync_check else 0)
        itsm = itsm2_processor_class(tf)(torch, stepper, n_in)
        procs.append(itsm)
    elif arm != "base":
        raise Model2CheckFailed(f"未知方法 {arm}")
    tf.set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    t0 = time.time()
    with torch.inference_mode(), active(ctrl):
        out = model.generate(**batch, do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p, top_k=cfg.top_k,
                             min_p=cfg.min_p, repetition_penalty=cfg.repetition_penalty, max_new_tokens=tb + ab + 2,
                             use_cache=True, logits_processor=tf.LogitsProcessorList(procs))
    seconds = round(time.time() - t0, 2)
    output = out[0, n_in:].tolist()
    dec = lambda x: tok.decode(x, skip_special_tokens=True)
    thinking, answer, close = split_output(dec, output, backend.close_id, cfg.think_open)
    raw = tok.decode(output, skip_special_tokens=False)
    from .conflict import claim_of
    claim = claim_of(item, cond)
    rec = {"prompt_text": prompt, "input_ids": ids.tolist(), "output_ids": output, "raw_output": raw,
           "thinking": thinking, "answer": answer, "n_in": n_in, "n_out": len(output), "n_img": int(len(img_pos)),
           "seconds": seconds, "seed": seed, "budget": budget.metadata(), "arm": arm, "id": item["id"],
           "condition": cond, "gold": item["gold"], "claim": claim, "message": item["messages"][cond],
           "image": item["image"], "image_sha256": item.get("image_sha256"),
           "thinking_empty": not thinking.strip(), "close_position": close,
           "controller": ctrl.summary() if ctrl else None, "itsm": itsm.summary() if itsm else None,
           "neutral_text": neutral_text, "model": cfg.model_path, "image_grid": grid_info,
           "sampling": {"temperature": cfg.temperature, "top_p": cfg.top_p, "top_k": cfg.top_k, "min_p": cfg.min_p}}
    rec["task"] = task_of(item)
    if claim and rec["task"] == "clock":
        rec["outcome"] = outcome(answer, item["gold"], claim)
        rec["final_time"] = final_time(answer, claim)
    elif rec["task"] == "yesno":
        rec["semantic_judgment_status"] = "pending"          # 是/否题由 yn-export / yn-summary 判读
    if sync_check and stepper is not None:
        rec["neutral_sync"] = check_neutral_sync(backend, stepper, output)
        a = rec["neutral_sync"]["agree"]
        if a is not None and a < SYNC_MIN_AGREE:
            raise Model2CheckFailed(f"ITSM 中性分支逐步跟进与整段前向不一致：前 {rec['neutral_sync']['steps']} 步"
                                    f"最想写的字只有 {a:.2f} 一致")
    return rec


# ---------------------------------------------------------------- 锚点一 v2 的生成
A1_ARMS = {
    # 方法名: (参照从哪来, 主生成时干预的区域；None = 不改注意力)
    "readref":  ("read", "E"),       # 额外调用一次：中性提问下完整回答一遍，取其看图分布作参照；受压回答全程补回关键区域
    "readrand": ("read", "R"),       # 同上，但补到同样大小的随机区域（对照）
    "leadref":  ("lead", "E"),       # 先读：中性提问下先写一段思考，接到受压提示后面继续写；参照取这段的看图分布，之后全程补回
    "lead":     ("lead", None),      # 只做先读，不改注意力（拆开“干净的开头”和“注意力补回”各自的作用）
}


def _to_dev(backend, inputs):
    dev = backend.model.device
    b = {k: v.to(dev) for k, v in inputs.items() if hasattr(v, "to")}
    if "pixel_values" in b:
        b["pixel_values"] = b["pixel_values"].to(backend.model.dtype)
    return b


def extend_inputs(inputs: dict, extra: List[int]) -> dict:
    """在提示后面接上已写好的输出 token：input_ids 接上，attention_mask 接 1，其他与 input_ids 同形状的逐 token 张量接 0。"""
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


def _sample(backend, cfg, batch, seed: int, max_new: int, procs, stopping=None, ctrl=None):
    torch, tf, model = backend.torch, backend.transformers, backend.model
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "rope_deltas"):
        inner.rope_deltas = None                 # Qwen 的多模态位置编码偏移：每次新提示都重算
    tf.set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    h = model.register_forward_pre_hook(ctrl.pre_hook, with_kwargs=True) if ctrl is not None else None
    try:
        with torch.inference_mode(), active(ctrl):
            out = model.generate(**batch, do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p, top_k=cfg.top_k,
                                 min_p=cfg.min_p, repetition_penalty=cfg.repetition_penalty, max_new_tokens=max_new,
                                 use_cache=True, logits_processor=tf.LogitsProcessorList(procs),
                                 stopping_criteria=tf.StoppingCriteriaList(stopping or []))
    finally:
        if h is not None:
            h.remove()
    return out[0, int(batch["input_ids"].shape[1]):].tolist()


def _lead_stopper(backend):
    tf, torch, tok = backend.transformers, backend.torch, backend.tokenizer
    eos = backend.model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])

    class LeadStop(tf.StoppingCriteria):
        def __init__(self, n_in):
            self.n_in = n_in

        def __call__(self, input_ids, scores, **kw):
            gen = input_ids[0, self.n_in:].tolist()
            tail = tok.decode(gen[-3:], skip_special_tokens=False) if gen else ""
            stop = bool(gen) and lead_should_stop(len(gen), tail, gen[-1], backend.close_id, eos)
            return torch.full((input_ids.shape[0],), stop, dtype=torch.bool, device=input_ids.device)
    return LeadStop


def reference_a1(backend, cfg, item: dict, cond: str, source: str, seed: int, neutral_text: str, layers: List[int],
                 root: str) -> Tuple[dict, dict]:
    """参照（同一题、同一条件、同一种子只做一次，存在 root/refs_<source>/）。
    read：中性提问下完整回答一遍（思考 + 回答两个阶段的看图分布）；
    lead：中性提问下只写思考开头一段（lead_should_stop），回答阶段的参照用思考阶段的。"""
    from .generate import build_inputs, make_budget
    from .pipeline import _atomic_json, _atomic_npz
    torch, tf, tok = backend.torch, backend.transformers, backend.tokenizer
    d = os.path.join(root, f"refs_{source}")
    base = os.path.join(d, f"{item['id']}__{cond}__s{seed}")
    if os.path.exists(base + ".npz") and os.path.exists(base + ".json"):
        with np.load(base + ".npz") as z:
            return {k: z[k] for k in z.files}, json.load(open(base + ".json", encoding="utf-8"))
    from ..attn import image_token_id
    _, inp = build_inputs(backend, item["image"], neutral_text)
    ids = inp["input_ids"][0]
    img_pos = (ids == image_token_id(backend)).nonzero().flatten().cpu().numpy()
    n_in = int(ids.shape[0])
    ctrl = Restore2(torch, img_pos, layers, backend.close_id, "observe")
    batch = _to_dev(backend, inp)
    t0 = time.time()
    if source == "read":
        gc = backend.model.generation_config
        eos = gc.eos_token_id if isinstance(gc.eos_token_id, list) else [gc.eos_token_id]
        budget = make_budget(tf, torch, n_in, backend.close_id, int(eos[0]), cfg.think_budget, cfg.answer_budget)
        out = _sample(backend, cfg, batch, seed, cfg.think_budget + cfg.answer_budget + 2, [budget], ctrl=ctrl)
    elif source == "lead":
        out = _sample(backend, cfg, batch, seed, LEAD_MAX, [], [_lead_stopper(backend)(n_in)], ctrl=ctrl)
    else:
        raise Model2CheckFailed(f"未知参照来源 {source}")
    if ctrl.sums is None or ctrl.rows[0] == 0:
        raise Model2CheckFailed(f"{item['id']}__{cond}：参照生成没有记录到思考阶段")
    dec = lambda x: tok.decode(x, skip_special_tokens=True)
    thinking, answer, close = split_output(dec, out, backend.close_id, cfg.think_open)
    no_answer = ctrl.rows[1] == 0 or source == "lead"
    ref, chk = ref_from_sums(item["id"], ctrl.sums, layers, answer_from_think=no_answer)
    meta = {"id": item["id"], "condition": cond, "seed": seed, "source": source, "neutral_text": neutral_text,
            "n_in": n_in, "n_out": len(out), "output_ids": out, "thinking": thinking, "answer": answer,
            "close_position": close, "rows_think": ctrl.rows[0], "rows_answer": ctrl.rows[1],
            "answer_ref_from_think": no_answer, "seconds": round(time.time() - t0, 2), "check": chk,
            "selfcheck_ratio_max": ctrl.summary()["selfcheck_ratio_max"]}
    if source == "lead":
        lead = list(out)
        while lead and (lead[-1] == backend.close_id or lead[-1] in set(np.atleast_1d(backend.model.generation_config.eos_token_id).tolist())):
            lead.pop()
        meta["lead_ids"] = lead
    os.makedirs(d, exist_ok=True)
    _atomic_npz(base + ".npz", ref)
    _atomic_json(base + ".json", meta)
    return ref, meta


def generate_a1(backend, cfg, item: dict, cond: str, arm: str, seed: int, neutral_text: str, layer_info: dict,
                root: str) -> dict:
    """锚点一 v2：先取参照（read / lead），再生成受压回答；参照来源与干预区域见 A1_ARMS。"""
    from ..attn import image_token_id
    from .generate import build_inputs, make_budget
    from .verify import final_time, outcome
    torch, tf, model, tok = backend.torch, backend.transformers, backend.model, backend.tokenizer
    if backend.close_id is None:
        raise Model2CheckFailed(f"思考结束标记 {cfg.think_close!r} 不是单个 token")
    if not neutral_text:
        raise Model2CheckFailed("锚点一 v2 需要模型自己改写的中性提问")
    source, region = A1_ARMS[arm]
    layers = record_layers2(layer_info)
    ref, meta = reference_a1(backend, cfg, item, cond, source, seed, neutral_text, layers, root)
    prompt, inputs = build_inputs(backend, item["image"], item["messages"][cond])
    ids = inputs["input_ids"][0]
    img_pos = (ids == image_token_id(backend)).nonzero().flatten().cpu().numpy()
    n_in = int(ids.shape[0])
    lead = (meta.get("lead_ids") or []) if source == "lead" else []
    if lead:
        inputs = extend_inputs(inputs, lead)
    batch = _to_dev(backend, inputs)
    gc = model.generation_config
    eos = gc.eos_token_id if isinstance(gc.eos_token_id, list) else [gc.eos_token_id]
    tb, ab = cfg.think_budget, cfg.answer_budget
    budget = make_budget(tf, torch, n_in, backend.close_id, int(eos[0]), tb, ab)     # 先读的 token 算在思考预算里
    ctrl = Restore2(torch, img_pos, layers, backend.close_id, "restore", ref=ref, region=region) if region else None
    t0 = time.time()
    gen = _sample(backend, cfg, batch, seed, tb + ab + 2 - len(lead), [budget], ctrl=ctrl)
    seconds = round(time.time() - t0, 2)
    output = lead + gen
    if ctrl is not None and ctrl.rows[0] + ctrl.rows[1] != max(len(gen) - 1, 0):
        raise Model2CheckFailed(f"干预记录的解码步数 {ctrl.rows[0] + ctrl.rows[1]} 与生成 token 数 {len(gen)} − 1 不一致")
    dec = lambda x: tok.decode(x, skip_special_tokens=True)
    thinking, answer, close = split_output(dec, output, backend.close_id, cfg.think_open)
    from .conflict import claim_of
    claim = claim_of(item, cond)
    rec = {"prompt_text": prompt, "input_ids": ids.tolist(), "output_ids": output, "raw_output": tok.decode(output, skip_special_tokens=False),
           "thinking": thinking, "answer": answer, "n_in": n_in, "n_out": len(output), "n_img": int(len(img_pos)),
           "seconds": seconds, "seed": seed, "budget": budget.metadata(), "arm": arm, "id": item["id"],
           "condition": cond, "gold": item["gold"], "claim": claim, "message": item["messages"][cond],
           "image": item["image"], "image_sha256": item.get("image_sha256"),
           "thinking_empty": not thinking.strip(), "close_position": close,
           "controller": ctrl.summary() if ctrl else None, "neutral_text": neutral_text, "model": cfg.model_path,
           "a1": {"source": source, "region": region, "lead_tokens": len(lead),
                  "lead_text": dec(lead) if lead else None,
                  "reference": {k: v for k, v in meta.items() if k not in ("output_ids", "lead_ids")}},
           "sampling": {"temperature": cfg.temperature, "top_p": cfg.top_p, "top_k": cfg.top_k, "min_p": cfg.min_p}}
    rec["task"] = task_of(item)
    if claim and rec["task"] == "clock":
        rec["outcome"] = outcome(answer, item["gold"], claim)
        rec["final_time"] = final_time(answer, claim)
    elif rec["task"] == "yesno":
        rec["semantic_judgment_status"] = "pending"
    return rec


# ---------------------------------------------------------------- 流程
def setup(src_cq: str, dst: str) -> int:
    """从原 ClockQA 目录复制 items.jsonl 和图像（图像改指向 dst/images，按 sha256 核对）。"""
    from ..store import read_jsonl, write_jsonl
    if os.path.exists(os.path.join(dst, "items.jsonl")):
        raise SystemExit(f"{dst} 里已经有 items.jsonl，不覆盖；请换一个新目录")
    items = read_jsonl(os.path.join(src_cq, "items.jsonl"))
    os.makedirs(os.path.join(dst, "images"), exist_ok=True)
    out = []
    for it in items:
        src = it["image"]
        h = hashlib.sha256(open(src, "rb").read()).hexdigest()
        if it.get("image_sha256") and h != it["image_sha256"]:
            raise Model2CheckFailed(f"{it['id']}：图像 sha256 与记录不一致：{src}")
        d = os.path.join(os.path.abspath(dst), "images", f"{it['id']}{os.path.splitext(src)[1]}")
        shutil.copyfile(src, d)
        msgs = add_belief_right(it)
        for c in CONDS_BASE:
            if c not in msgs:
                raise Model2CheckFailed(f"{it['id']}：缺少条件 {c} 的消息")
        out.append(dict(it, image=d, image_sha256=h, messages=msgs))
    write_jsonl(os.path.join(dst, "items.jsonl"), out)
    print(f"[fp-model2 setup] {len(out)} 题 → {dst}（图像已复制到 {dst}/images）", file=sys.stderr)
    return len(out)


def _engine(cfg):
    from .pipeline import Engine
    eng = Engine(cfg)
    if eng.backend is None:
        raise SystemExit("fp-model2 需要 hf 后端")
    return eng


def run_neutralize(cfg, cq: str, shard: str = "") -> int:
    """ITSM 的中性提问：让同一个模型（不开思考、贪心）把用户消息改写成只问问题的形式。不看图。"""
    from ..backends import GenRequest
    from .conflict import _items
    from .pipeline import _atomic_json, _check_no_failure, _fail, _shard
    _check_no_failure(cq)
    items = _items(cq)
    d = os.path.join(cq, "neutralized")
    os.makedirs(d, exist_ok=True)
    tasks = [(i, c) for i in sorted(items) for c in CONDS_ARM]
    todo = [t for t in _shard(tasks, shard) if not os.path.exists(os.path.join(d, f"{t[0]}__{t[1]}.json"))]
    print(f"[fp-model2 neutralize] 待做 {len(todo)} 条", file=sys.stderr)
    if not todo:
        return 0
    eng = _engine(cfg)
    b = eng.backend
    torch, model, tok = b.torch, b.model, b.tokenizer
    for n, (iid, c) in enumerate(todo):
        try:
            it = items[iid]
            prompt = b._template(GenRequest(user_text=NEUTRALIZE_PROMPT.format(msg=it["messages"][c]), images=[]), thinking=False)
            inp = b.processor(text=[prompt], return_tensors="pt")
            batch = {k: v.to(model.device) for k, v in inp.items() if hasattr(v, "to")}
            with torch.inference_mode():
                o = model.generate(**batch, do_sample=False, max_new_tokens=200, use_cache=True)
            text = tok.decode(o[0, batch["input_ids"].shape[1]:].tolist(), skip_special_tokens=True).strip()
            if not text:
                raise Model2CheckFailed(f"{iid}__{c}：改写结果为空")
            rec = {"id": iid, "condition": c, "original": it["messages"][c], "neutral": text, **neutral_leak(text, it, c),
                   "exactly_original_question": text.strip() == it["messages"]["neutral"].strip()}
            _atomic_json(os.path.join(d, f"{iid}__{c}.json"), rec)
            print(f"  {n + 1}/{len(todo)} {iid}__{c} → {text[:80]!r} 残留用户时间={rec['has_claim_time']}", file=sys.stderr)
        except Exception as e:
            _fail(cq, f"neutralize_{iid}__{c}", e)
            raise
    neutralize_summary(cq)
    return len(todo)


def neutralize_summary(cq: str) -> dict:
    d = os.path.join(cq, "neutralized")
    recs = [json.load(open(os.path.join(d, f), encoding="utf-8")) for f in sorted(os.listdir(d)) if f.endswith(".json")]
    s = {"n": len(recs), "has_claim_time": sum(r["has_claim_time"] for r in recs),
         "has_opinion_words": sum(r["has_opinion_words"] for r in recs),
         "keeps_question": sum(r["keeps_question"] for r in recs)}
    print(f"[fp-model2 neutralize] 共 {s['n']} 条；残留用户时间 {s['has_claim_time']}，残留“我认为”类说法 "
          f"{s['has_opinion_words']}，原问题保留 {s['keeps_question']}", file=sys.stderr)
    return s


def rec_path(cq: str, root: str, arm: str, seed: int, iid: str, cond: str) -> str:
    from .compare import run_name
    if arm == "base" and seed == 42:
        return os.path.join(cq, "obs", "records", f"{iid}__{cond}.json")
    return os.path.join(root, run_name(arm, seed), "records", f"{iid}__{cond}.json")


def run_arms(cfg, cq: str, root: str, arms: List[str], seeds: List[int], shard: str = "") -> int:
    from .conflict import _items
    from .pipeline import _atomic_json, _check_no_failure, _fail, _shard
    for a in arms:
        if a not in ARMS2:
            raise SystemExit(f"未知方法 {a}（可选 {ARMS2}）")
    _check_no_failure(root)
    items = _items(cq)
    tasks = [(a, s, i, c) for a in arms for s in seeds for i in sorted(items)
             for c in (CONDS_BASE if a == "base" else CONDS_ARM)]
    todo = [t for t in _shard(tasks, shard) if not os.path.exists(rec_path(cq, root, *t))]
    print(f"[fp-model2 run] 共 {len(tasks)} 次生成，本进程待做 {len(todo)} 次", file=sys.stderr)
    if not todo:
        return 0
    eng = _engine(cfg)
    info = None
    if "mmsy" in arms or any(a in A1_ARMS for a in arms):
        info = install(eng.backend.model)
        n_sl = sum(v["sliding"] for v in info.values())
        print(f"  语言模型注意力层 {len(info)} 层（滑动窗口 {n_sl}，全局 {len(info) - n_sl}）", file=sys.stderr)
    print(f"  模型 {type(eng.backend.model).__name__}；思考结束标记 {cfg.think_close!r} id={eng.backend.close_id}",
          file=sys.stderr)
    synced, first = False, []
    for n, (a, s, iid, c) in enumerate(todo):
        try:
            nt = None
            if a == "itsm" or a in A1_ARMS:
                p = os.path.join(cq, "neutralized", f"{iid}__{c}.json")
                if not os.path.exists(p):
                    raise Model2CheckFailed(f"缺少中性提问 {p}（先跑 --phase neutralize）")
                nt = json.load(open(p, encoding="utf-8"))["neutral"]
            if a in A1_ARMS:
                r = generate_a1(eng.backend, cfg, items[iid], c, a, s, nt, info, root)
            else:
                r = generate2(eng.backend, cfg, items[iid], c, a, s, nt, info, sync_check=(a == "itsm" and not synced))
            if a == "itsm":
                synced = True
            from .compare import run_name
            r["run"] = "obs" if (a == "base" and s == 42) else run_name(a, s)
            _atomic_json(rec_path(cq, root, a, s, iid, c), r)
            extra = ""
            if r.get("controller"):
                extra = f" MM-SY 图像注意力 {r['controller']['image_mass_before']:.3f}→{r['controller']['image_mass_after']:.3f}"
            if r.get("neutral_sync"):
                extra += f" 中性分支同步 {r['neutral_sync']}"
            if r.get("a1"):
                ref = r["a1"]["reference"]
                extra = f" 参照（{ref['source']}）{ref['rows_think']}+{ref['rows_answer']} 行" + \
                        (f"，先读 {r['a1']['lead_tokens']} token" if r["a1"]["source"] == "lead" else "")
                if r.get("controller"):
                    th = r["controller"]["think"]
                    extra += f"，思考阶段区域比例 {th['region_before']:.3f}→{th['region_after']:.3f}"
            first.append(r["thinking_empty"])
            print(f"  {n + 1}/{len(todo)} {a} s{s} {iid}__{c} 图像 {r['n_img']} token，输出 {r['n_out']} token {r['seconds']}s "
                  f"思考为空={r['thinking_empty']} 结果={r.get('outcome', '—')}{extra}", file=sys.stderr)
            if len(first) <= FIRST_CHECK and r["thinking_empty"]:
                raise Model2CheckFailed(f"本进程前 {FIRST_CHECK} 条里出现思考为空（{a} {iid}__{c}），思考模式可能没打开；记录已保存")
        except Exception as e:
            _fail(root, f"{a}_s{s}_{iid}__{c}", e)
            raise
    return len(todo)
