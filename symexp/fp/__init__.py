# -*- coding: utf-8 -*-
"""五种压力的观察 + 关键区域恢复干预（PathVQA 重做）。

流程：prepare → observe（中性 + 5 种压力，生成时在线记录注意力）→ 判读 → candidates
      → intervene（restore_E / rand_matched，可选 base）→ 判读 → analyze。
说明见 docs/服务器任务_PathVQA五压力_重做.md。
"""

PRESSURES = ["statement", "belief", "conviction", "authority", "social"]
CONDITIONS = ["neutral"] + PRESSURES
ARMS = ["restore_E", "rand_matched", "base"]
LAYERS = list(range(3, 64, 4))      # Qwen3.8-27B 的 16 个标准注意力层
N_HEADS_SELECTED = 96               # 看图头：384 个（层, 头）里取前 96 个
E_FRACTION = 0.2                    # 关键区域：中性注意力最高的 20% 图像块
FLOOR = 0.05                        # 非目标区域至少保留原份额的 5%
RANDOM_SEED = 20261024              # 随机区域的种子（与模型采样无关）
