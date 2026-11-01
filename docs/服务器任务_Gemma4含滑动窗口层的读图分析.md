# 服务器任务：Gemma 4 ClockQA 读图分析，加上滑动窗口层；Qwen3.8 用同一方法重算一遍作对照

**为什么要做**：上次 Gemma 只记录了 10 个全局层，滑动窗口层没有记录。但在思考的前 256 个 token 里，整张图都还在滑动窗口层的视野里，这些层也在读图。这次全部层都记录，并把全局层、滑动窗口层分开报告。

**和上次相比的三处改动**
- 前向时用原注意力函数的输出，只记录、不替换。上次那个 0.125 的重算差不会再影响前向。每类层各核对一次重算的相对差，并打印出来。
- 每个头的入选按三种条件平均的看图总量来定，不偏向任何一种条件。
- 只回放到思考的第 256 个 token，后面不算，所以比上次快。

**规则**
- 不要修改任何代码、参数、阈值或清单，也不要额外运行检查命令（下面列出的命令除外）。
- 报错就停下，汇报错误信息和 `failures/` 里的内容。不要自己打补丁，不要自动重试，不要自己清理 `failures/`。
- 停进程用 `kill -TERM`，不要用 Ctrl-C。不要停止或抢占别人的进程。只用空闲的卡，没有空闲就等。
- 不要上传 npz。

## A. 51：Gemma 4（GPU 回放 186 条，然后用 CPU）

```bash
R=/mnt/sdb/linxin/repo-gemma4
conda activate gemma4
git -C $R pull
cd $R && python -m pytest tests/test_fp.py -q -rs   # 应全部通过，没有 skipped
C=$R/configs/gemma4_31b.yaml
G=/mnt/sdb/linxin/gemma4-cq-20261029
GM=/mnt/sdb/linxin/gemma4-compare-20261029
S2=/mnt/sdb/linxin/gemma4-cq-sources-all-20261102     # 新目录
mkdir -p $S2
df -h $S2                                             # 存下来的数组约 10 GB；剩余空间不到 20 GB 就停下汇报
# 两张卡一个进程，开两个进程：
CUDA_VISIBLE_DEVICES=<两张空闲卡> python run.py fp-sources --phase reading-gen-all --config $C --cq $G --root $S2 --shard 0/2 2>&1 | tee $S2/reading_gen_all_0.log
CUDA_VISIBLE_DEVICES=<另两张空闲卡> python run.py fp-sources --phase reading-gen-all --config $C --cq $G --root $S2 --shard 1/2 2>&1 | tee $S2/reading_gen_all_1.log
# 两个都跑完后（CPU）：
python run.py fp-sources --phase reading-layers --cq $G --root $S2 --judgments $GM/time_judge/judgments.json --label "Gemma 4 31B · ClockQA" 2>&1 | tee $S2/reading_layers.log
```

- 日志开头会打印记录的层数、滑动窗口层数、窗口大小和全局层的层号。
- 每条会打印两类层重算输出的相对差。
- 如果某条记录到的 token 已经让图像滑出窗口，程序会报错停下。

上传：

```bash
D=$R/results/gemma4_cq_20261029/reading_layers_20261102
mkdir -p $D/reading_all
cp $S2/reading_layers_report.md $S2/reading_layers_report.json $S2/*.log $D/
cp $S2/reading_all/*.json $D/reading_all/
```

在 `$R` 里提交并推送；推送被拒就先 `git pull --rebase` 再推。

## B. 59：Qwen3.8 用同一方法重算（只用 CPU，用上次 reading 存好的数据）

```bash
L=/mnt/sdb/linxin/repo-label
git -C $L pull
cd $L && python -m pytest tests/test_fp.py -q -rs
CQ=/mnt/sdb/linxin/conflict-cq-20261028
M=/mnt/sdb/linxin/compare-cq-20261029-a100
S=/mnt/sdb/linxin/sources-cq-20261029
python run.py fp-sources --phase reading-layers --cq $CQ --root $S --judgments $M/time_judge/judgments.json --label "Qwen3.8-27B · ClockQA" 2>&1 | tee $S/reading_layers.log
D=$L/results/sources_cq_20261029
cp $S/reading_layers_report.md $S/reading_layers_report.json $S/reading_layers.log $D/
```

在 `$L` 里提交并推送。

## 汇报

1. 两台服务器 pytest 的最后一行；A 的 `df -h` 输出。
2. 每个回放进程日志的前 5 行和最后 3 行。
3. 两份 `reading_layers_report.md` 的全文。
