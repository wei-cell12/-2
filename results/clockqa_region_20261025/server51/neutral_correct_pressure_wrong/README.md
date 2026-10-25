# ClockQA 中性/压力 20 份序列：原始结果数据（server51）

服务器 `linxin@59.77.5.51:2222`，目录 `/mnt/sdb/linxin/clockqa-region-pilot10-20261025` 的原始产出。
本目录**只做搬运**，没有重新生成、没有重算、没有改动任何数值。

- 数据集：ClockQA（`mahzzz/Sycophancy-Dataset`，revision `e8c3b29be7dc264b682f2182f412d1646fcf457d`），62 题中 seed42 取 10 题
- 模型：`Qwen3.6-27B`（`/mnt/sdb/ljh/models/Qwen3.6-27B`）
- 规模：10 题 × 中性/压力 = **20 份序列**（`state/job_state.json`：`planned=20, saved=20`）

## 内容

```
answers.jsonl                     20 份回答 + 规则抽取的时间 + 压力类型 + 生成 token 数
answers.csv                       同上，表格版
items.jsonl                       10 道题的题面、标准答案、错误主张、压力话术、图像 SHA256
region_report/report.md           原区域对比报告（10 题配对）
region_report/summary.json        原区域指标（含 bootstrap CI 与 p 值）
region_report/items.csv           原逐题×阶段×头指标长表
region_report/heatmaps/           原 10 张四联热图（原图｜中性｜压力｜差值）
validation/replay_validation.json 原回放校验（20 份逐条 corr / MAD / selfcheck / 行数）
state/job_state.json              任务状态与计数
state/final_summary.json          原汇总（含 time_rule 计数与校验摘要）
state/launch_receipt.json         启动回执（GPU4/2/3、显存预算、资源前置检查）
state/runtime.json                运行环境（torch 2.9.1+cu128、transformers 5.15.1、16 层 [3,7,…,63]、24 头）
state/model_identity.json         模型文件 SHA256
state/generation_config_effective.json  实际生效的生成参数
state/length_audit.json           文本长度审计（未配平，逐题记录）
state/request.json                实验请求参数
server_README.md                  服务器上的原始 README（含跨卡调整与中断接续记录）
```

## 未上传的内容与原因

- `attn/*.npz`、`results/*/attention.npz`：20 份共 586,406,991 字节，留在服务器原位。
- `results/*/record.json`：每份含 `input_ids` / `output_ids` 完整 token 列表，按约定不公开。
- `env/`、`images/`（图像字节可由 `items.jsonl` 里的 SHA256 校验）、`attempts/`、`live/`。

## 两点必须注意的口径

1. `state/final_summary.json` 里的 `time_rule`（中性 6 对 4 错、压力 2 对 6 错 2 待定）是**正则抽取精确时间**的
   次要口径，文件自身也写明 `not joint answer+explanation correctness`；它不是“答案与解释都正确”的判读结果，
   20 份在远端均为 `unreviewed`（未做 Luna 判读）。
2. `region_report/` 是**全部 10 题配对**的区域对比，没有按“中性答对/压力答错”筛题；
   其中 `report.md` 的 AUC 表用的是该报告自己的规则标签，不能当作严格筛选后的结论。
