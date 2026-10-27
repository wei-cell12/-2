# -*- coding: utf-8 -*-
"""干预规则：只改图像内部的注意力分布，图像总量、文字部分、整行总和都不变。

同一份代码既能用 numpy（本地测试）也能用 torch（服务器上生成时），通过 XP 适配器切换。
输入都是“选中头”的图像部分概率 p_img：[h, n]（还没在图像内部归一化）。

restore（关键区域恢复，λ=1 时完全抬回参照）：
  q = p_img / P                      P = 图像总量（每个头一个数）
  对关键区域 E 中份额低于参照 r 的位置：加 λ·max(0, r − q)
  加的总量 Δ 从 E 以外的图像位置按比例扣除；扣除上限为它们原份额的 (1 − floor)
  新 p_img = P · q_new

matched（剂量匹配的随机区域）：
  给定每个头要加的总量 Δ（= 同一位置 restore 实际加的量），加在随机区域 R 上，
  按权重 w（随机区域的参照形状）分配；从 R 以外的图像位置按比例扣除，上限同上。
"""

from typing import Dict, Tuple


class XP:
    """numpy / torch 的最小公共接口。"""

    def __init__(self, lib: str):
        self.lib = lib
        if lib == "numpy":
            import numpy as np
            self.np = np
        else:
            import torch
            self.torch = torch

    def sum(self, x):
        return x.sum(-1, keepdims=True) if self.lib == "numpy" else x.sum(-1, keepdim=True)

    def maximum(self, a, b):
        return self.np.maximum(a, b) if self.lib == "numpy" else self.torch.maximum(a, b)

    def minimum(self, a, b):
        return self.np.minimum(a, b) if self.lib == "numpy" else self.torch.minimum(a, b)

    def where(self, c, a, b):
        return self.np.where(c, a, b) if self.lib == "numpy" else self.torch.where(c, a, b)

    def fl(self, x):
        """布尔掩码转成与 x 同精度的 0/1。"""
        return x.astype(self.np.float64) if self.lib == "numpy" else x.to(self.torch.float32)

    def tiny(self):
        return 1e-30

    def full_like(self, x, v):
        return self.np.full_like(x, v) if self.lib == "numpy" else self.torch.full_like(x, v)

    def abs(self, x):
        return self.np.abs(x) if self.lib == "numpy" else x.abs()

    def log(self, x):
        return self.np.log(x) if self.lib == "numpy" else self.torch.log(x)


def _js(xp: XP, a, b):
    """每个头一个数：图像内部分布 a、b（[h, n]，各自和为 1）的 JS 散度（自然对数）。"""
    eps = 1e-12
    m = 0.5 * (a + b)
    ka = xp.sum(a * (xp.log(xp.maximum(a, xp.full_like(a, eps))) - xp.log(xp.maximum(m, xp.full_like(m, eps)))))
    kb = xp.sum(b * (xp.log(xp.maximum(b, xp.full_like(b, eps))) - xp.log(xp.maximum(m, xp.full_like(m, eps)))))
    return 0.5 * (ka + kb)


def mix(xp: XP, p_img, ref, lam: float, ref_alt=None) -> Tuple[object, Dict[str, object]]:
    """v2：逐步参照的混合恢复。只改图像内部分布，图像总量、文字部分、整行总和都不变。

      q  = p_img / P                     受压分支此刻的图像内部分布（P = 图像总量）
      r  = 中性分支同一步、同一层、同一头的图像内部分布
      q' = (1 − λ)·q + λ·r
    ref_alt 不为空时是对照组：目标换成 ref_alt（r 的位置打乱），λ 按头调整为
      λ' = min(1, λ · |q − r|₁ / |q − ref_alt|₁)，使改动的量（½|q' − q|₁）与用 r 时相同。
    """
    P = xp.sum(p_img)
    q = p_img / xp.maximum(P, xp.full_like(P, xp.tiny()))
    r = ref / xp.maximum(xp.sum(ref), xp.full_like(P, xp.tiny()))
    d_r = 0.5 * xp.sum(xp.abs(q - r))
    if ref_alt is None:
        target = r
        lam_eff = xp.full_like(d_r, lam)
        want = lam * d_r
    else:
        target = ref_alt / xp.maximum(xp.sum(ref_alt), xp.full_like(P, xp.tiny()))
        d_t = 0.5 * xp.sum(xp.abs(q - target))
        want = lam * d_r
        lam_eff = xp.minimum(xp.full_like(d_r, 1.0), want / xp.maximum(d_t, xp.full_like(d_t, xp.tiny())))
    q_new = (1.0 - lam_eff) * q + lam_eff * target
    new = p_img if lam == 0 else P * q_new
    dose = 0.5 * xp.sum(xp.abs(q_new - q))
    info = {"P": P[..., 0], "js_before": _js(xp, q, r)[..., 0], "js_after": _js(xp, q_new, r)[..., 0],
            "dose": dose[..., 0], "requested": want[..., 0], "lam_eff": lam_eff[..., 0], "l1_q_r": d_r[..., 0]}
    return new, info


def restore(xp: XP, p_img, mask, ref, lam: float, floor: float) -> Tuple[object, Dict[str, object]]:
    m = xp.fl(mask)
    nm = 1.0 - m
    P = xp.sum(p_img)
    q = p_img / xp.maximum(P, xp.full_like(P, xp.tiny()))
    donor = xp.sum(q * nm)
    add = xp.maximum(ref - q, xp.full_like(q, 0.0)) * m * lam
    req = xp.sum(add)
    cap = donor * (1.0 - floor)
    ratio = xp.minimum(xp.full_like(req, 1.0), cap / xp.maximum(req, xp.full_like(req, xp.tiny())))
    add = add * ratio
    delta = xp.sum(add)
    q_new = q + add - q * nm * (delta / xp.maximum(donor, xp.full_like(donor, xp.tiny())))
    new = p_img if lam == 0 else P * q_new
    info = {"P": P[..., 0], "region_before": xp.sum(q * m)[..., 0], "region_after": xp.sum(q_new * m)[..., 0],
            "delta": delta[..., 0], "requested": req[..., 0], "clamped": (req > cap)[..., 0]}
    return new, info


def matched(xp: XP, p_img, rmask, weight, delta, floor: float) -> Tuple[object, Dict[str, object]]:
    """delta: [h] 或 [h, 1]，要加到随机区域上的总量。"""
    if delta.ndim == 1:
        delta = delta[:, None]
    m = xp.fl(rmask)
    nm = 1.0 - m
    P = xp.sum(p_img)
    q = p_img / xp.maximum(P, xp.full_like(P, xp.tiny()))
    donor = xp.sum(q * nm)
    w = weight * m
    ws = xp.sum(w)
    n_r = xp.sum(m)
    w = xp.where(ws > 0, w / xp.maximum(ws, xp.full_like(ws, xp.tiny())), m / xp.maximum(n_r, xp.full_like(n_r, 1.0)))
    cap = donor * (1.0 - floor)
    d = xp.minimum(delta, cap)
    add = w * d
    q_new = q + add - q * nm * (d / xp.maximum(donor, xp.full_like(donor, xp.tiny())))
    new = P * q_new
    info = {"P": P[..., 0], "region_before": xp.sum(q * m)[..., 0], "region_after": xp.sum(q_new * m)[..., 0],
            "delta": d[..., 0], "requested": delta[..., 0], "clamped": (delta > cap)[..., 0]}
    return new, info


def apply_rows(xp: XP, p_row, img_idx, heads, new_img):
    """把选中头的新图像部分写回整行注意力 p_row：[H, K]。返回新行（不改原数组）。"""
    if xp.lib == "numpy":
        out = p_row.copy()
        out[heads[:, None], img_idx[None, :]] = new_img
    else:
        out = p_row.clone()
        out[heads[:, None], img_idx[None, :]] = new_img.to(out.dtype)
    return out


def conservation(xp: XP, p_row, new_row, img_idx, heads) -> Dict[str, float]:
    """选中头的图像总量、文字总量、整行总和的最大绝对变化。"""
    if xp.lib == "numpy":
        a, b = p_row[heads], new_row[heads]
        ia, ib = a[:, img_idx].sum(-1), b[:, img_idx].sum(-1)
        ra, rb = a.sum(-1), b.sum(-1)
        f = lambda x: float(abs(x).max()) if x.size else 0.0
        neg = float(b.min()) if b.size else 0.0
    else:
        a, b = p_row.index_select(0, heads), new_row.index_select(0, heads)
        ia, ib = a.index_select(1, img_idx).sum(-1), b.index_select(1, img_idx).sum(-1)
        ra, rb = a.sum(-1), b.sum(-1)
        f = lambda x: float(x.abs().max()) if x.numel() else 0.0
        neg = float(b.min()) if b.numel() else 0.0
    return {"image": f(ib - ia), "text": f((rb - ib) - (ra - ia)), "row": f(rb - ra), "min_prob": neg}
