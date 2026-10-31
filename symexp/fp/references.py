# -*- coding: utf-8 -*-
"""从中性观察的阶段平均注意力构造干预参照（与 ClockQA 实验相同的规则）。

输入：中性 npz 里的 think / answer，形状 [16 层, 24 头, n_img]（阶段内逐 token 平均）。
输出（每题一份）：
  r        [2, 16, 24, n]   中性参照分布（图像内部归一化），阶段 0 = 思考，1 = 回答
  heads    [16, 24]          看图头：中性思考阶段图像总量最高的 96 个（层, 头）
  E_mask   [2, 16, 24, n]   关键区域：每个阶段、每个（层, 头）r 最高的 ceil(0.2·n) 个图像块
  R_mask   [2, 16, 24, n]   随机区域：从 E 以外随机取同样多个
  R_weight [2, 16, 24, n]   随机区域的分配权重：E 的参照值随机打乱后放到 R 上
随机数由 (种子, 题号, 阶段, 层, 头) 的 SHA256 派生，与模型采样无关，可复现。
"""

import hashlib
import math
from typing import Dict

import numpy as np

from . import E_FRACTION, LAYERS, N_HEADS_SELECTED, RANDOM_SEED


def _rng(item_id: str, s: int, layer: int, h: int) -> np.random.Generator:
    key = f"{RANDOM_SEED}|{item_id}|{s}|{layer}|{h}".encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "little"))


def build(item_id: str, think: np.ndarray, answer: np.ndarray, n_heads: int = N_HEADS_SELECTED,
          frac: float = E_FRACTION, layer_ids=None) -> Dict[str, np.ndarray]:
    """layer_ids：数组第 1 维对应的层号；默认 Qwen3.8 的 16 个标准注意力层（其他模型传入自己记录的层）。"""
    layers = list(LAYERS if layer_ids is None else [int(x) for x in layer_ids])
    w = np.stack([think, answer]).astype(np.float64)          # [2, L, H, n]
    if w.ndim != 4 or w.shape[1] != len(layers):
        raise ValueError(f"参照数组形状不对：{w.shape}")
    mass = w.sum(-1, keepdims=True)
    if not (np.isfinite(w).all() and (mass > 0).all()):
        raise ValueError(f"{item_id}：中性参照有非有限值或某阶段某头图像总量为 0")
    r = (w / mass).astype(np.float32)
    S, L, H, n = r.shape
    k = int(math.ceil(frac * n))
    if not 0 < k <= n - k:
        raise ValueError(f"{item_id}：图像 token 太少（n={n}）")
    heads = np.zeros((L, H), bool)
    order = np.argsort(-mass[0, :, :, 0].ravel(), kind="stable")[:n_heads]
    heads.ravel()[order] = True
    E = np.zeros(r.shape, bool)
    R = np.zeros(r.shape, bool)
    RW = np.zeros(r.shape, np.float32)
    for s in range(S):
        for l in range(L):
            for h in range(H):
                e = np.argsort(-r[s, l, h], kind="stable")[:k]
                E[s, l, h, e] = True
                g = _rng(item_id, s, layers[l], h)
                fake = g.choice(np.flatnonzero(~E[s, l, h]), size=k, replace=False)
                R[s, l, h, fake] = True
                RW[s, l, h, fake] = g.permutation(r[s, l, h, e])
    return {"layer_ids": np.array(layers), "r": r, "heads": heads, "E_mask": E, "R_mask": R, "R_weight": RW,
            "n_img": np.array(n), "k": np.array(k)}


def check(ref: Dict[str, np.ndarray]) -> Dict[str, object]:
    r, E, R = ref["r"], ref["E_mask"], ref["R_mask"]
    k = int(ref["k"])
    out = {"r_sums_to_one": bool(np.allclose(r.sum(-1), 1.0, atol=1e-4)),
           "E_size_ok": bool((E.sum(-1) == k).all()), "R_size_ok": bool((R.sum(-1) == k).all()),
           "E_R_disjoint": bool(not (E & R).any()), "heads": int(ref["heads"].sum())}
    out["ok"] = all(v for kk, v in out.items() if kk != "heads") and out["heads"] == N_HEADS_SELECTED
    return out
