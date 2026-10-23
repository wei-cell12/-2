"""LQCD 对比公式、阶段划分、注意力区域分析（numpy，不需要 torch）。"""

import math
import os

import numpy as np

from symexp.attn import stage_bounds
from symexp.attn_analysis import analyze, auc, item_metrics
from symexp.lqcd import contrast_numpy


def test_contrast_alpha_zero_is_neutral():
    ln = np.array([2.0, 1.0, 0.0, -1.0])
    ll = np.array([0.0, 3.0, 0.0, 0.0])
    cd, a, b, j = contrast_numpy(ln, ll, 0.0, 0.0, 0.0, 0.0)
    assert np.allclose(cd, ln) and a == 0.0 and j > 0


def test_contrast_pushes_away_from_leading():
    ln = np.array([1.0, 1.0, -5.0])
    ll = np.array([0.0, 2.0, -5.0])          # 诱导分支偏向第 1 个 token
    cd, a, _, _ = contrast_numpy(ln, ll, 1.0, 0.0, 0.1, 0.0)
    assert cd[0] > cd[1]                     # 对比后偏向第 0 个
    assert cd[2] == -np.inf                  # 中性概率太低的被过滤


def test_contrast_identical_branches():
    ln = np.array([0.5, 0.2, -0.3])
    cd, a, b, j = contrast_numpy(ln, ln, 1.0, 2.0, 0.0, 0.0)
    assert abs(j) < 1e-12 and math.isclose(a, 1.0) and np.allclose(cd, ln)


def test_stage_bounds():
    b = stage_bounds(prompt_len=10, close_pos=20, total_len=30)
    assert b == {"prefill_end": (9, 10), "think": (10, 20), "answer": (20, 29)}
    b = stage_bounds(prompt_len=10, close_pos=None, total_len=30)
    assert b["think"] == (10, 29) and b["answer"] == (29, 29)


def _maps(rng, hot, L=16, H=4, n=36, noise=0.2):
    x = rng.random((L, H, n)) * noise
    x[:, :, hot] += 1.0
    return x / x.sum(-1, keepdims=True) * 0.3      # 图像总量 0.3


def test_item_metrics_detects_shift_and_not_noise():
    rng = np.random.default_rng(0)
    ref = _maps(rng, [0, 1, 2, 3, 4, 5, 6])
    moved = _maps(rng, [20, 21, 22, 23, 24, 25, 26])
    same = _maps(rng, [0, 1, 2, 3, 4, 5, 6])
    m1 = item_metrics(ref, moved, None)
    m0 = item_metrics(ref, same, None)
    assert m1["dE"] < -0.3 and m1["dU"] > 1.0 and m1["retention"] < 0.2 and m1["dlogL"] > 1.0
    # 同一分布的两次独立噪声：拆半测量不会凭空测出“下降”
    assert abs(m0["dE"]) < 0.05 and m0["retention"] > 0.8


def test_auc():
    assert auc(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 1.0
    assert auc(np.array([1.0]), np.array([1.0])) == 0.5


def test_analyze_end_to_end(tmp_path):
    rng = np.random.default_rng(1)
    ref_dir, cmp_dir = tmp_path / "ref", tmp_path / "cmp"
    os.makedirs(ref_dir)
    os.makedirs(cmp_dir)
    labels = {}
    for k in range(12):
        i = f"X-{k:02d}"
        flip = k % 2 == 0
        labels[i] = flip
        r = {s: _maps(rng, [0, 1, 2, 3, 4, 5, 6]) for s in ("prefill_end", "think", "answer")}
        hot = [20, 21, 22, 23, 24, 25, 26] if flip else [0, 1, 2, 3, 4, 5, 6]
        c = {s: _maps(rng, hot) for s in ("prefill_end", "think", "answer")}
        for d, data in ((ref_dir, r), (cmp_dir, c)):
            np.savez(d / f"{i}__{'neutral' if d == ref_dir else 'user_wrong'}.npz",
                     grid_hw=np.array([6, 6]), image=np.array(""), **data)
    out = tmp_path / "out"
    rep = analyze(f"plain_neutral={ref_dir}@neutral", [f"plain_wrong={cmp_dir}@user_wrong"], str(out),
                  labels=labels, heatmap_ids=["X-00"], reps=200)
    assert "plain_wrong vs plain_neutral" in rep and "AUC" in rep
    assert os.path.exists(out / "heatmaps" / "X-00__plain_wrong.png")
    import json
    s = json.load(open(out / "summary.json", encoding="utf-8"))
    a = s["comparisons"]["plain_wrong"]["auc|think|全部"]
    assert a["dE"] > 0.9 and a["n_flip"] == 6
