# 服务器任务：ClockQA 切断 / 降低注意力的筛选（喂回已有回答，不生成）

交给服务器上的 agent。把原模型已有的 124 条回答（62 题 × 用户说错 / 用户说对）原样喂回模型，每条在 10 种注意力改写下各前向一次，记录每个位置写出原来那个字的概率。不生成新回答。

**规则**
- 不要修改任何代码、参数、阈值或清单，也不要额外运行检查命令。
- 报错就停下，汇报错误信息和 `$S/failures/` 里的内容。不要自己打补丁，不要自动重试，不要自己清理 `failures/`。
- 停进程用 `kill -TERM`。不要停止或抢占别人的进程。
- 全部在标注用的那份仓库 `$L` 里做。

**如果已经按上一版开始跑了**：用 `kill -TERM` 停掉正在跑的 cut 进程，把 `$S/cut` 改名为 `$S/cut_v1`（不要删除），再按下面步骤从头跑。这一版改了其中一种改写（Cq 的参照），旧结果不能混用。

## 步骤

```bash
L=/mnt/sdb/linxin/repo-label
git -C $L pull
cd $L && python -m pytest tests/test_fp.py -q -rs   # 应全部通过，没有 skipped（含 test_sources_cut_modify）
CQ=/mnt/sdb/linxin/conflict-cq-20261028
M=/mnt/sdb/linxin/compare-cq-20261029-a100
S=/mnt/sdb/linxin/sources-cq-20261029             # 和之前回放同一个目录
C=/mnt/sdb/linxin/pathvqa-fp-a800-20261027/repo/configs/qwen38_fp_local.yaml
```

如果 pytest 有 skipped 或失败，停下汇报。

### 1. 喂回（GPU）

共 124 条 × 10 种改写 ≈ 1240 次前向。单卡约 2 小时，4 张卡约 30 分钟。只用空闲的 A100，没有空闲就等。

```bash
cd $L
# 有 n 张空闲卡时，第 k 张（k = 0 … n-1）：
CUDA_VISIBLE_DEVICES=<卡号> python run.py fp-sources --phase cut --config $C --cq $CQ --root $S --shard k/n 2>&1 | tee $S/cut_k.log
```

只有一张卡时去掉 `--shard`，日志写 `cut.log`。中途停下后重新运行同一条命令，会跳过已完成的。

### 2. 分析（CPU）

```bash
cd $L
python run.py fp-sources --phase cut-analyze --cq $CQ --root $S --judgments $M/time_judge/judgments.json 2>&1 | tee $S/cut_analyze.log
```

## 上传

```bash
D=$L/results/sources_cq_20261029
cp $S/cut_report.md $S/cut_summary.json $S/cut*.log $D/
```

不要上传 `$S/cut/*.npz`。在 `$L` 里提交并推送。

## 汇报

1. pytest 最后一行和 `-rs` 列出的跳过项（应为没有）。
2. 是否 124 条全部完成；各 `cut*.log` 的最后 3 行。
3. `cut_analyze.log` 全文。
4. `cut_report.md` 全文。
