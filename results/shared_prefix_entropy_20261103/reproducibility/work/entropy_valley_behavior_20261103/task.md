# 波谷行为解释扩展

用户要求：除最大谷外，分析其他深浅局部谷，明确可检测的行为与触发用途。

输入为已完成600条shared-prefix entropy，保持原文/数值不变；不重新生成、不使用GPU。旧615深谷与152 Gemma一致性审核保留。全量扫描87341负差段，其中615深、5337中、29284浅，其余<.05仅数值存档。新语义核验701位置：161深、180中、180浅、180同轨迹普通对照；固定哈希分层选择，不看结果，盲化深度/模型/条件/gold。

同一 /root/pathvqa_luna_continuation，gpt-6-luna，reasoning max，自主写完29批并执行validate.py及finish.py。原标签不可覆盖，修正另写；程序仅校验/统计，不代替语义分类。无需服务器、无需监视或新agent。

当前已完成的次级深谷及全100条QwenClockQA注意力对照在 results/local_valleys/behavior_extension/CURRENT_FINDINGS.md；Luna完成后报告在同目录REPORT.md、index.html、summary.json。
