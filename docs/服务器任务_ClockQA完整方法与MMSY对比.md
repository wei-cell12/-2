# 服务器任务：ClockQA 完整方法（锚点一 + 锚点二）与 MM-SY 对比

交给服务器上的 agent。代码已写好（`run.py fp-compare`），按步骤运行。

**规则**
- 不要修改任何代码、参数、阈值或清单，也不要额外运行检查命令。
- 报错或预检不通过就停下，汇报错误信息和 `$M/failures/` 里的内容。不要自己打补丁，不要自动重试，不要自己清理 `failures/`。
- 停进程用 `kill -TERM`，不要用 Ctrl-C。不要停止或抢占别人的进程。
- 卡号由我指定。

## 做什么

对 ClockQA 全部 62 题，在两种条件下各生成一次（种子 42）：
- 用户说错（belief）；
- 用户说对（belief_right）。

两种方法：
- **mmsy**：MM-SY（ICLR 2025）。在后 8 个标准注意力层，把图像 token 的注意力打分 e 改为 e + 0.9·|e|。
- **full**：完整方法。
  - 锚点一：v1 restore_E，参照来自同一题的中性回答。
  - 锚点二：生成时实时计算冲突分数，超过阈值就插入 `<verify>`，由裁判核对后接回 `<result>`。

原模型的结果直接用已有的 `$CQ/obs` 记录，不重新生成。

## 步骤

```bash
git pull                                       # 应包含 fp-compare
python -m pytest tests/test_fp.py -q           # 应全部通过（31 项，包括 test_compare_*）
CQ=/mnt/sdb/linxin/conflict-cq-20261028        # 只读：题目、原模型记录、中性注意力
W=/mnt/sdb/linxin/conflict-work-20261028       # 只读：directions.npz（两个数据集一起求出的版本）
M=/mnt/sdb/linxin/compare-cq-20261029          # 新目录
C=<A800 主实验用的配置文件>
mkdir -p $M
```

### 1. 预检（GPU，约 15 分钟）

```bash
CUDA_VISIBLE_DEVICES=<卡号> python run.py fp-compare --phase gate --config $C --cq $CQ --work $W --root $M 2>&1 | tee $M/gate.log
```

预检会做四件事：
1. 用新代码以原模型设置生成 2 题，看能否和已有记录逐 token 一致。
2. 检查生成时实时算的冲突分数，和整段重算的结果是否一致。
3. full 跑 1 题。
4. mmsy 跑 1 题。

日志最后一行如果是 `ok=False`，**停下汇报** `$M/compare_gate.json`。如果是 `ok=True`，继续下一步；这时即使 `deterministic=False` 也继续，但要在汇报里说明。

### 2. 正式运行（GPU）

```bash
CUDA_VISIBLE_DEVICES=<卡号> python run.py fp-compare --phase run --config $C --cq $CQ --work $W --root $M --arms mmsy full 2>&1 | tee $M/run.log
```

- 共 248 次生成，预检里已经做完 2 次。full 每次触发后还要多跑 2 次裁判。单卡约 10 小时。
- 多张卡时：每个进程加 `--shard k/n`，日志写 `run_k.log`，所有进程用同一个 `--root $M`。
- 中途停下后，重新运行同一条命令，会跳过已完成的。

### 3. 汇总（CPU，等所有进程结束）

```bash
python run.py fp-compare --phase summary --cq $CQ --root $M --out $M/compare_summary.json 2>&1 | tee $M/summary.log
```

## 上传

```bash
D=<仓库>/results/compare_cq_20261029
mkdir -p $D
cp -r $M/mmsy $M/full $M/loop $D/
cp $M/compare_gate.json $M/compare_summary.json $M/*.log $D/
```

**不要上传** `$M/refs/`（npz）。提交并推送。

## 汇报

1. 预检日志的最后 5 行。
2. 正式运行是否全部完成。
3. `summary.log` 全文。
