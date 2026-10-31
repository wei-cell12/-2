# 服务器任务：Gemma 4 ClockQA 思考开头 token 的看图分析（与 Qwen3.8 那次同一套分析）

交给 51 上的 agent（Gemma 4 ClockQA 基线跑的那台）。代码已写好。

**规则**
- 不要修改任何代码、参数、阈值或清单，也不要额外运行检查命令（下面列出的命令除外）。
- 报错就停下，汇报错误信息和 `failures/` 里的内容。不要自己打补丁，不要自动重试，不要自己清理 `failures/`。
- 停进程用 `kill -TERM`，不要用 Ctrl-C；不要停止或抢占别人的进程。
- 用 Gemma 那次的环境（gemma4）和仓库 `$R`。不要上传 npz。

## 做什么

把 Gemma 4 ClockQA 已有的 186 条原模型回答（62 题 × 中性 / 用户说错 / 用户说对）原样喂回模型，只做前向，不生成。分两步：
1. 每题的中性回答整条回放一次，建参照：96 个看图头，每个头的关键区域（中性思考阶段注意力最高的 20% 图像块）；
2. 三种条件的回答各回放一次，记录思考最前面 512 个 token 对图像 token 的注意力（每 8 个 token 存一组）。

Gemma 4 有滑动窗口层，只看最近 1024 个 token，图像会滑出视野。代码自动只记录全局层，滑动窗口层不记录。

然后在 CPU 上跑与 Qwen3.8 那次完全相同的两个分析：`reading-analyze` 和 `reading-claim`。

## 0. 准备

```bash
R=/mnt/sdb/linxin/repo-gemma4
conda activate gemma4
git -C $R pull
cd $R && python -m pytest tests/test_fp.py -q -rs   # 应全部通过，没有 skipped
C=$R/configs/gemma4_31b.yaml
G=/mnt/sdb/linxin/gemma4-cq-20261029          # Gemma 4 ClockQA 目录（原模型记录在 $G/obs/records）
GM=/mnt/sdb/linxin/gemma4-compare-20261029    # 判读结果在 $GM/time_judge/judgments.json
S=/mnt/sdb/linxin/gemma4-cq-sources-20261031  # 新目录
mkdir -p $S
```

如果 pytest 有 skipped 或失败，或者上面两个已有目录不存在，停下汇报。

## 1. 回放（GPU，两张卡一个进程，开两个进程）

```bash
cd $R
CUDA_VISIBLE_DEVICES=<两张空闲卡> python run.py fp-sources --phase reading-gen --config $C --cq $G --root $S --shard 0/2 2>&1 | tee $S/reading_gen_0.log
CUDA_VISIBLE_DEVICES=<另两张空闲卡> python run.py fp-sources --phase reading-gen --config $C --cq $G --root $S --shard 1/2 2>&1 | tee $S/reading_gen_1.log
```

- 只用空闲的卡，没有空闲就等。
- 日志开头会打印"记录的层：全局层 […]"；每题参照会打印图像 token 数、排列（行 × 列）、关键区域块数、重算差。
- 如果某题的中性回答没有写完思考，程序会报错停下。把题号汇报给我，不要自己跳过。

## 2. 分析（CPU）

```bash
cd $R
python run.py fp-sources --phase reading-analyze --cq $G --root $S --judgments $GM/time_judge/judgments.json 2>&1 | tee $S/reading_analyze.log
python run.py fp-sources --phase reading-claim --cq $G --root $S --judgments $GM/time_judge/judgments.json 2>&1 | tee $S/reading_claim.log
```

## 上传

```bash
D=$R/results/gemma4_cq_sources_20261031
mkdir -p $D/reading $D/refs $D/reading_claim_check
cp $S/reading_report.md $S/reading_summary.json $S/reading_claim_report.md $S/reading_claim_summary.json $S/*.log $D/
cp $S/reading/*.json $D/reading/
cp $S/refs/*.json $D/refs/
cp $S/reading_claim_check/*.png $D/reading_claim_check/
```

不要上传 npz。在 `$R` 里提交并推送；推送被拒就先 `git pull --rebase` 再推。

## 汇报

1. pytest 最后一行。
2. 每个回放进程日志的前 5 行和最后 3 行。
3. `reading_report.md` 全文。
4. `reading_claim_report.md` 全文。
