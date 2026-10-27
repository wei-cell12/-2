# -*- coding: utf-8 -*-
"""不需要模型的假生成，只用来检查整条流程（数字没有意义）。数组形状与真实输出一致。"""

import zlib
from typing import Dict, Optional, Tuple

import numpy as np

from . import LAYERS

N_IMG = 36
GRID = (6, 6)
H = 24


def _rng(*parts) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32("|".join(map(str, parts)).encode()))


def generate_one(item: dict, condition: str, mode: str, ref: Optional[Dict] = None, seed: int = 42
                 ) -> Tuple[Dict, Dict[str, np.ndarray]]:
    same = "observe" if mode in ("base", "dual_base") else mode     # base 与观察同种子同路径 → 同样的结果
    g = _rng(item["id"], condition, same, seed)
    base = _rng(item["id"], "image").random((len(LAYERS), H, N_IMG)) + 0.05
    arrays: Dict[str, np.ndarray] = {}
    for name, scale in (("prefill_end", 1.0), ("think", 1.0), ("answer", 1.0)):
        noise = g.random(base.shape) * (0.6 if condition != "neutral" else 0.2)
        x = (base + noise) * 0.02 * scale
        arrays[name] = x.astype(np.float32)
        arrays[name + "_after"] = x.astype(np.float32)
    T = int(g.integers(40, 80))
    arrays["pos_image_mass"] = (g.random((T, len(LAYERS), H)) * 0.3).astype(np.float16)
    arrays["pos_stage"] = np.array([0] + [1] * (T // 2) + [2] * (T - 1 - T // 2), np.int8)
    for s in ("prefill_end", "think", "answer"):
        arrays["count_" + s] = np.array(int((arrays["pos_stage"] == {"prefill_end": 0, "think": 1, "answer": 2}[s]).sum()))
    arrays.update(layer_ids=np.array(LAYERS), n_img=np.array(N_IMG), grid_hw=np.array(GRID, np.int32),
                  image=np.array(item["image"]), selfcheck_max_abs_diff=np.zeros(48, np.float32),
                  trace=np.zeros((0, 13), np.float32), trace_columns=np.array(["step"] * 13))
    summ = {"mode": mode, "decode_steps": T - 1, "selfcheck_max": 0.0, "selfcheck_ratio_max": 0.0}
    if mode in ("restore_E", "rand_matched"):
        summ.update(mean_delta=0.2 if mode == "restore_E" else 0.2, region_before=0.66 if mode == "restore_E" else 0.08,
                    region_after=0.9 if mode == "restore_E" else 0.3, dose_gap_max=0.0, clamped_rows=0, cons_max=0.0,
                    rows=T * 96)
    if mode.startswith("dual_"):
        lam = 0.0 if mode == "dual_base" else 1.0
        summ.update(lam=lam, js_before=0.05, js_after=0.05 * (1 - lam), mean_dose=0.1 * lam, mean_requested=0.1 * lam,
                    dose_gap_max=0.0, cons_max=0.0, rows=T * 96,
                    neutral_branch={"prefill_vs_observation_maxabs": 0.0, "rope_deltas_equal": True,
                                    "position_check_k": 17, "top1_miss": 0, "logp_diff_max": 0.0,
                                    "logp_diff_median": 0.0})
    gold, claim = item["gold"], item["claim"]
    if condition == "neutral":
        p_correct = 0.8
    else:
        p_correct = {"observe": 0.2, "base": 0.2, "restore_E": 0.55, "rand_matched": 0.25,
                     "dual_base": 0.2, "dual_mix": 0.5, "dual_shuffle": 0.25}[mode]
        if condition == "statement":
            p_correct = 0.05
    g2 = _rng(item["id"], condition, same, seed, "ans")
    ans = gold if g2.random() < p_correct else claim
    out_ids = [int(x) for x in _rng(item["id"], condition, same, seed, "ids").integers(0, 1000, T)]
    record = {"prompt_text": "(mock)", "input_ids": [1, 2, 3], "output_ids": out_ids,
              "raw_output": f"mock thinking</think>{ans.capitalize()}. (mock)", "thinking": "mock thinking",
              "answer": f"{ans.capitalize()}. (mock)", "n_in": 3, "n_out": T, "n_img": N_IMG, "seconds": 0.0,
              "seed": seed, "budget": {"thinking_budget_hit": False, "answer_budget_hit": False},
              "controller": summ}
    return record, arrays
