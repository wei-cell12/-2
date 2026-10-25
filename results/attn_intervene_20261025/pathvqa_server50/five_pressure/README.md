# PathVQA五压力观察与注意力干预实验 (Server 50)

接手并持续推进 PathVQA 62 题五压力观察与注意力干预实验。

- **服务器**：`linxin@59.77.5.50:2222`
- **实验根目录**：`/mnt/sdg/linxin/pathvqa-five-pressure-20261025`
- **模型**：Qwen3.8-27B（bf16，math-decode SDPA，中等思考预算 4096+回答 512）
- **规模**：62 道二分类医学病理问答题（PV-000 ~ PV-061）

## 当前实验进展概览

1. **原生观测（中性 vs 五压力）**：
   - 计划总量：372 项（62 题 × 6 种条件：neutral, statement, belief, conviction, authority, social）
   - 当前已保存观测记录：**287** 份（记录于 `native_outputs.jsonl`）
   - 前 47 题已全条件闭环落盘，剩余项由两名健康 Worker（PID 2759852 占用 GPU 0,1；PID 2759936 占用 GPU 4,5；严格禁用 GPU 7）持续推进。

2. **注意力干预（base / restore_E / restore_rand）**：
   - 候选筛选：严格由同一 Luna 语义审核（中性正确且压力首个变错），共 18 道有效候选题；
   - 干预完成度：**18 道候选全部完成 3 组干预（共 54 条干预生成记录，详见 `intervention_outputs.jsonl`）**；
   - 包含逐 token 思考与回答、16 层注意力差值自检（<0.05）与图像/文本行列质量守恒指标（<1e-3）。

3. **数据凭据与资产规范**：
   - **NPZ 阵列留存远端**：所有 attention 与 trace NPZ 文件完整保留在服务器（总计 287 个，788.58 MB），详细路径、大小与 SHA256 清单见 `array_manifest.json`；
   - **全量无损提取**：不受旧“前12题”范围限制，涵盖目前最新生成的全部实验数据与审核凭据。
