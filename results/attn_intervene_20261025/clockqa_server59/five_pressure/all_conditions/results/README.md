# ClockQA固定49例干预

最终结果，2026-10-26 00:59:05（北京时间）。49个原始中性正确／压力错误条件，17张不同图片；原始压力基线复用，不重跑、不重新筛选。

|组别|正确|错误|未解|
|---|---:|---:|---:|
|原始压力基线|0|49|0|
|关键区域干预|14|35|0|
|随机区域干预|4|45|0|

答案与完整解释都正确才算恢复；只看时间时E为15例，解释错误不算联合恢复。原题×原压力完整保留，同一张图可以对应多条压力条件。

|配对比较|前者独对|后者独对|discordant|精确双侧p|
|---|---:|---:|---:|---:|
|原始压力 vs 关键区域|0|14|14|0.00012207|
|原始压力 vs 随机区域|0|4|4|0.12500000|
|关键区域 vs 随机区域|11|1|12|0.00634766|

同图压力条件相关，且原始压力组按错误结果选入；条件级p仅作描述，不当作独立49图的推广证据。

原始中性与压力图像总量分别来自已保存生成数组。干预前后区域份额为选中96头逐token均值、每条件等权，不能和全16层×24头图像总量混用。E为中性高注意力top20%，并非人工证据标注；随机同patch数但搬移剂量不匹配。完整算法、原medium/seed42/BF16/4096+512及图像预算和守恒门槛不变。原始压力空间回放只喂已保存token，不重新作答。

历史21份额外λ0回复（原17及纠正前已生成4份）保留作历史数值验证，均未充作本次49份原始压力基线。14份恢复是在原始回答均错误的固定子集上观察到的，不等于独立49图随机试验。

打开index.html看全部49例原始中性/压力原文、E/Rand完整思考和回答、实际干预热图及原序列区域对照。items.csv提供逐例判读和来源；summary.json为可复算统计；array_manifest.json列出本轮参照/干预/压力回放远端数组路径、大小、SHA，原始观测数组历史清单另保留在上一级报告。NPZ和完整token表留远端不上传。GitHub上传已由用户明确授权到wei-cell12/-2。

[49例完整思考、回答和147张热图](index.html) · [逐例文本分析](../luna_audit/trajectory_findings_final_zh.md) · [逐例结果表](items.csv) · [数值核验](validation.json)

实际版本与配置：

<details><summary>模型、版本、设备与分阶段时间</summary>

```json
{
  "model": "/mnt/sdb/linxin/ModelList/Qwen3.8-27B",
  "torch": "2.11.0+cu130",
  "transformers": "5.14.1",
  "dtype": "torch.bfloat16",
  "physical_gpus": [
    0,
    1,
    2,
    3
  ],
  "cpu_offload": false,
  "last_two_arm_generation_phase_start_unix": 1792944313.787446,
  "last_two_arm_generation_phase_end_unix": 1792947277.819066,
  "note": "last two-arm continuation only; historical reused outputs have their own earlier logs"
}
```
</details>

数组清单279项，共1275651596字节。GPU模型生成/必要保存后自然退出；无常驻占卡。
