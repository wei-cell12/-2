# -*- coding: utf-8 -*-
"""生成时的注意力控制（只在服务器 GPU 上运行）。

所有条件走同一条计算路径：16 个标准注意力层在“逐 token 解码”时，自己显式计算
softmax(QK^T/√d)（fp32）再乘 V，用这个结果作为该层输出；prefill 原样调用原注意力函数；
视觉编码器和 48 个线性注意力层不动。

模式：
  observe       不修改，记录全部（层, 头）的阶段平均图像注意力和逐 token 图像总量
  base          同 observe（λ=0），用于确认重新生成与观察结果逐 token 一致
  restore_E     选中头上恢复关键区域（λ=1）
  rand_matched  选中头上恢复随机区域，抬高量逐 token、逐头等于 restore_E 在同一位置会抬高的量

阶段：prefill 最后一个查询 = prefill_end；解码时喂入的 token 是 </think> 那一步起算回答阶段，
之前是思考阶段（与 symexp/attn.py 的定义一致）。

自动检查（不通过就抛异常，由调用方停下汇报）：
  - 前 selfcheck_steps 个解码步，16 层显式计算（未修改）与原注意力函数输出逐元素比较：
    允许差 = max(tol_selfcheck, 2⁻⁶·|原输出|)，即 0.05 和 bf16 的 2 个最小精度单位取大者。
    两条路径都输出 bf16，量级为 x 的元素差 1 个最小精度单位就是 2⁻⁷·|x|（例如 |x|≈10 时为 0.0625），
    这属于正常舍入；实现写错（缩放、掩码、GQA 头顺序）时差会远大于这个量级
  - 每一行修改后：选中头的图像总量、文字总量、整行总和变化 < tol_mass，且没有负概率
  - restore_E：有抬高量时关键区域份额必须上升
"""

from typing import Dict, List, Optional

import numpy as np

from . import FLOOR, LAYERS
from .rules import XP, conservation, matched, restore

ACTIVE = None
_TARGETS = set()
_INSTALLED = False
LAYER_INDEX = {l: i for i, l in enumerate(LAYERS)}
STAGE_NAMES = ["prefill_end", "think", "answer"]


class CheckFailed(RuntimeError):
    pass


class FPController:
    def __init__(self, torch, mode: str, image_idx, close_id: int, ref: Optional[Dict[str, np.ndarray]] = None,
                 lam: float = 1.0, floor: float = FLOOR, selfcheck_steps: int = 3,
                 tol_selfcheck: float = 0.05, tol_mass: float = 1e-3):
        if mode not in ("observe", "base", "restore_E", "rand_matched"):
            raise ValueError(mode)
        if mode in ("restore_E", "rand_matched") and ref is None:
            raise ValueError("干预模式需要参照")
        self.torch, self.mode, self.close_id = torch, mode, close_id
        self.xp = XP("torch")
        self.image_idx_cpu = image_idx.cpu()
        self.n_img = int(image_idx.numel())
        self.ref, self.lam, self.floor = ref, lam, floor
        self.selfcheck_steps, self.tol_selfcheck, self.tol_mass = selfcheck_steps, tol_selfcheck, tol_mass
        self.step = -1             # 解码步编号；prefill 时为 -1
        self.stage = 0             # 0 = 思考，1 = 回答（只用于解码步）
        self.decode = False
        self.prefill_calls = 0
        self.decode_calls = 0
        self.cache = {}
        self.sums = {}             # layer -> Tensor[3, H, n]（未修改）
        self.sums_after = {}       # layer -> Tensor[3, H, n]（修改后；observe/base 时与 sums 相同）
        self.counts = np.zeros(3, dtype=np.int64)
        self.pos_mass: List[np.ndarray] = []   # 每个查询一行 [16, H]：该查询对整张图的注意力
        self.pos_stage: List[int] = []         # 每行的阶段：0 prefill_end，1 思考，2 回答
        self._row = None
        self._row_slot = 0
        self.trace: List[np.ndarray] = []
        self.selfchecks: List[dict] = []

    # ---------------------------------------------------------------- 阶段跟踪（模型 forward 前）
    def pre_hook(self, module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is None:
            raise CheckFailed("generate 的 forward 没有 input_ids")
        if ids.shape[-1] == 1 and self.step >= -1 and self.prefill_calls > 0:
            self._flush_row()
            self.decode = True
            self.step += 1
            if int(ids[0, -1]) == self.close_id:
                self.stage = 1
        else:
            if self.prefill_calls > 0:
                raise CheckFailed("出现第二次 prefill")
            self.decode = False
        self._row = None

    def _flush_row(self):
        if self._row is not None:
            self.pos_mass.append(self._row)
            self.pos_stage.append(self._row_slot)
            self._row = None

    # ---------------------------------------------------------------- 工具
    def _dev(self, device):
        key = ("img", str(device))
        if key not in self.cache:
            self.cache[key] = self.image_idx_cpu.to(device)
        return self.cache[key]

    def _ref_tensors(self, layer: int, stage: int, device):
        key = ("ref", layer, stage, str(device))
        if key not in self.cache:
            t = self.torch
            li = LAYER_INDEX[layer]
            hs = np.flatnonzero(self.ref["heads"][li])
            def g(name, dt):
                return t.as_tensor(np.ascontiguousarray(self.ref[name][stage, li, hs]), dtype=dt, device=device)
            self.cache[key] = (t.as_tensor(hs, dtype=t.long, device=device),
                               g("E_mask", t.bool), g("r", t.float32), g("R_mask", t.bool), g("R_weight", t.float32))
        return self.cache[key]

    def _probs(self, module, query, key, attention_mask, args, kwargs):
        """query: [1, H, 1, D] → 概率 [H, K]（fp32）。"""
        t = self.torch
        B, H, Q, D = query.shape
        rep = H // key.shape[1]
        k = key.repeat_interleave(rep, dim=1) if rep > 1 else key
        scale = kwargs.get("scaling", args[1] if len(args) > 1 else None)
        if scale is None:
            scale = getattr(module, "scaling", D ** -0.5)
        sc = t.matmul(query.float(), k.float().transpose(-1, -2)) * scale
        if attention_mask is not None:
            m = attention_mask[:, :, -1:, : k.shape[2]]
            sc = sc.masked_fill(~m, float("-inf")) if m.dtype == t.bool else sc + m.float()
        return t.softmax(sc, dim=-1)[0, :, 0]

    def _accumulate(self, layer: int, slot: int, before_img, after_img):
        t = self.torch
        if layer not in self.sums:
            H = before_img.shape[0]
            self.sums[layer] = t.zeros(3, H, self.n_img, dtype=t.float64, device=before_img.device)
            self.sums_after[layer] = t.zeros_like(self.sums[layer])
        self.sums[layer][slot] += before_img.double()
        self.sums_after[layer][slot] += after_img.double()
        if layer == LAYERS[0]:
            self.counts[slot] += 1
        if self._row is None:
            self._row = np.full((len(LAYERS), before_img.shape[0]), np.nan, dtype=np.float32)
            self._row_slot = slot
        self._row[LAYER_INDEX[layer]] = before_img.sum(-1).float().cpu().numpy()

    # ---------------------------------------------------------------- 注意力函数
    def attention(self, orig, module, query, key, value, attention_mask, *args, **kwargs):
        t = self.torch
        layer = int(module.layer_idx)
        img = self._dev(query.device)
        if query.shape[2] > 1:                      # prefill：原样计算，额外记录最后一个查询
            if self.decode:
                raise CheckFailed("解码阶段出现多查询调用")
            self.prefill_calls += 1 if layer == LAYERS[0] else 0
            out = orig(module, query, key, value, attention_mask, *args, **kwargs)
            m = attention_mask[:, :, -1:, :] if attention_mask is not None and attention_mask.dim() == 4 else None
            p = self._probs(module, query[:, :, -1:], key, m, args, kwargs)
            pi = p.index_select(1, img)
            self._accumulate(layer, 0, pi, pi)
            return out
        if query.shape[0] != 1:
            raise CheckFailed("只支持 batch=1")
        if layer == LAYERS[0]:
            self.decode_calls += 1
        p = self._probs(module, query, key, attention_mask, args, kwargs)      # [H, K]
        p_img = p.index_select(1, img)
        slot = 1 + self.stage
        p_new = p
        if self.mode in ("restore_E", "rand_matched"):
            hs, E, r, R, RW = self._ref_tensors(layer, self.stage, query.device)
            sub = p_img.index_select(0, hs)
            new_e, info_e = restore(self.xp, sub, E, r, self.lam, self.floor)
            if self.mode == "restore_E":
                new_sub, info = new_e, info_e
                if bool(((info["delta"] > 1e-6) & (info["region_after"] <= info["region_before"])).any()):
                    raise CheckFailed(f"关键区域份额没有随抬高而上升：layer={layer} step={self.step}")
            else:
                new_sub, info = matched(self.xp, sub, R, RW, info_e["delta"], self.floor)
            p_new = p.clone()
            p_new[hs[:, None], img[None, :]] = new_sub
            cons = conservation(self.xp, p, p_new, img, hs)
            if max(cons["image"], cons["text"], cons["row"]) >= self.tol_mass or cons["min_prob"] < -1e-7:
                raise CheckFailed(f"守恒检查失败：layer={layer} step={self.step} {cons}")
            n = hs.numel()
            col = lambda x: x.reshape(-1).float().cpu()
            rows = t.stack([t.full((n,), float(self.step)), t.full((n,), float(layer)), hs.float().cpu(),
                            t.full((n,), float(self.stage)), col(info["region_before"]), col(info["region_after"]),
                            col(info["P"]), col(info["delta"]), col(info["requested"]), col(info["clamped"]),
                            t.full((n,), cons["image"]), t.full((n,), cons["text"]), t.full((n,), cons["row"])], -1)
            self.trace.append(rows.numpy())
        self._accumulate(layer, slot, p_img, p_new.index_select(1, img))
        rep = p.shape[0] // value.shape[1]
        v = value.repeat_interleave(rep, dim=1) if rep > 1 else value
        result = t.matmul(p_new[None, :, None, :], v.float()).to(query.dtype).transpose(1, 2).contiguous()
        if self.step < self.selfcheck_steps:
            plain = result if p_new is p else t.matmul(p[None, :, None, :], v.float()).to(query.dtype).transpose(1, 2)
            ref = orig(module, query, key, value, attention_mask, *args, **kwargs)[0].float()
            d = (plain.float() - ref).abs()
            allowed = t.clamp(ref.abs() * 2.0 ** -6, min=self.tol_selfcheck)
            ratio = float((d / allowed).max())
            diff = float(d.max())
            self.selfchecks.append({"step": self.step, "layer": layer, "max_abs_diff": diff, "max_ratio": ratio,
                                    "ref_absmax": float(ref.abs().max())})
            if not ratio <= 1.0:
                raise CheckFailed(f"显式注意力与原函数不一致：step={self.step} layer={layer} "
                                  f"diff={diff} 允许差之比={ratio:.3f}")
        return result, None

    # ---------------------------------------------------------------- 结果
    def result(self) -> Dict[str, np.ndarray]:
        self._flush_row()
        L, H, n = len(LAYERS), None, self.n_img
        out: Dict[str, np.ndarray] = {}
        for tag, src in (("", self.sums), ("_after", self.sums_after)):
            for slot, name in enumerate(STAGE_NAMES):
                c = self.counts[slot]
                if not src:
                    out[name + tag] = np.zeros((L, 0, n), np.float32)
                    continue
                H = next(iter(src.values())).shape[1]
                arr = np.zeros((L, H, n), np.float32)
                if c > 0:
                    for layer, s in src.items():
                        arr[LAYER_INDEX[layer]] = (s[slot] / c).float().cpu().numpy()
                out[name + tag] = arr
        for slot, name in enumerate(STAGE_NAMES):
            out["count_" + name] = np.array(self.counts[slot])
        out["pos_image_mass"] = (np.stack(self.pos_mass) if self.pos_mass else np.zeros((0, L, H or 0))).astype(np.float16)
        out["pos_stage"] = np.array(self.pos_stage, dtype=np.int8)
        out["layer_ids"] = np.array(LAYERS)
        out["n_img"] = np.array(n)
        out["selfcheck_max_abs_diff"] = np.array([c["max_abs_diff"] for c in self.selfchecks], np.float32)
        cols = ["step", "layer", "head", "stage", "region_before", "region_after", "image_mass", "delta",
                "requested", "clamped", "cons_image", "cons_text", "cons_row"]
        out["trace"] = np.concatenate(self.trace).astype(np.float32) if self.trace else np.zeros((0, len(cols)), np.float32)
        out["trace_columns"] = np.array(cols)
        return out

    def summary(self) -> Dict[str, object]:
        tr = np.concatenate(self.trace) if self.trace else np.zeros((0, 13), np.float32)
        s = {"mode": self.mode, "decode_steps": self.step + 1, "prefill_calls": self.prefill_calls,
             "decode_calls": self.decode_calls, "stage_counts": {n: int(c) for n, c in zip(STAGE_NAMES, self.counts)},
             "selfcheck_max": float(max([c["max_abs_diff"] for c in self.selfchecks] or [0.0])),
             "selfcheck_ratio_max": float(max([c["max_ratio"] for c in self.selfchecks] or [0.0])),
             "selfchecks": len(self.selfchecks)}
        if len(tr):
            s.update(rows=int(len(tr)), mean_delta=float(tr[:, 7].mean()), mean_requested=float(tr[:, 8].mean()),
                     region_before=float(tr[:, 4].mean()), region_after=float(tr[:, 5].mean()),
                     clamped_rows=int(tr[:, 9].sum()), dose_gap_max=float(np.abs(tr[:, 8] - tr[:, 7]).max()),
                     cons_max=float(tr[:, 10:13].max()))
            for st, nm in ((0, "think"), (1, "answer")):
                z = tr[tr[:, 3] == st]
                if len(z):
                    s[nm] = {"rows": int(len(z)), "mean_delta": float(z[:, 7].mean()),
                             "region_before": float(z[:, 4].mean()), "region_after": float(z[:, 5].mean())}
        return s


# ---------------------------------------------------------------- 安装
def install(model) -> List[int]:
    global _INSTALLED
    from ..attn import find_targets
    found = find_targets(model)
    layers = sorted(int(m.layer_idx) for _, m in found)
    if layers != LAYERS:
        raise CheckFailed(f"标准注意力层与预期不同：{layers}")
    _TARGETS.clear()
    _TARGETS.update(id(m) for _, m in found)
    if not _INSTALLED:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        orig = ALL_ATTENTION_FUNCTIONS["sdpa"]

        def wrapped(module, query, key, value, attention_mask, *args, **kwargs):
            if ACTIVE is None or id(module) not in _TARGETS:
                return orig(module, query, key, value, attention_mask, *args, **kwargs)
            return ACTIVE.attention(orig, module, query, key, value, attention_mask, *args, **kwargs)

        wrapped._symexp_fp = True
        ALL_ATTENTION_FUNCTIONS["sdpa"] = wrapped
        gm = getattr(type(ALL_ATTENTION_FUNCTIONS), "_global_mapping", None)
        if isinstance(gm, dict) and "sdpa" in gm:
            gm["sdpa"] = wrapped
        _INSTALLED = True
    return layers


class active:
    def __init__(self, ctrl):
        self.ctrl, self.prev = ctrl, None

    def __enter__(self):
        global ACTIVE
        self.prev, ACTIVE = ACTIVE, self.ctrl
        return self.ctrl

    def __exit__(self, *exc):
        global ACTIVE
        ACTIVE = self.prev
        return False
