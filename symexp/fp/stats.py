# -*- coding: utf-8 -*-
"""按图像分组的配对检验。PathVQA 有多道题共用一张图，同一张图的所有观测作为一组。"""

import itertools
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


def cluster_signflip(diffs: List[Tuple[str, float]], exact_max: int = 22, mc: int = 200000,
                     seed: int = 0) -> Dict[str, float]:
    """diffs：(组名, 配对差)。检验“组内差之和”的符号是否可以随机翻转。双侧。"""
    by = defaultdict(float)
    for g, d in diffs:
        by[g] += float(d)
    vals = np.array([v for v in by.values() if v != 0.0])
    obs = abs(float(np.sum(vals))) if len(vals) else 0.0
    out = {"clusters": len(by), "nonzero_clusters": int(len(vals)), "sum": float(np.sum(vals)) if len(vals) else 0.0}
    if not len(vals):
        out.update(p=1.0, method="none")
        return out
    if len(vals) <= exact_max:
        hits = tot = 0
        for signs in itertools.product((1.0, -1.0), repeat=len(vals)):
            tot += 1
            hits += abs(float(np.dot(signs, vals))) >= obs - 1e-12
        out.update(p=hits / tot, method="exact")
    else:
        rng = np.random.default_rng(seed)
        s = rng.choice([-1.0, 1.0], size=(mc, len(vals)))
        null = np.abs(s @ vals)
        out.update(p=float((1 + (null >= obs - 1e-12).sum()) / (mc + 1)), method=f"monte-carlo {mc}")
    return out


def cluster_bootstrap_rate(rows: List[Tuple[str, float]], reps: int = 5000, seed: int = 0) -> Tuple[float, float]:
    """按组重采样，返回均值的 95% 区间。"""
    by = defaultdict(list)
    for g, v in rows:
        by[g].append(float(v))
    groups = list(by.values())
    if len(groups) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(reps):
        pick = rng.integers(0, len(groups), len(groups))
        vals = [x for i in pick for x in groups[i]]
        stats.append(np.mean(vals))
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))
