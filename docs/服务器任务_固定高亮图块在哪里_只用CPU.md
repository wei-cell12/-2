# 服务器任务：固定高亮图块落在钟面的什么位置（只用 CPU，读已存数组）

**目的**：上次"按层类型看关键区 E"用到了"固定高亮图块"：同一层里，所有题都被高度关注的图块。Gemma 滑动层有 70% 到 80% 的看图注意力落在这些图块上。
要确认它们是不是其实落在钟表圆心或指针上；如果是，就混进了关键位置。

**做什么**：用和上次同一个定法找出固定高亮图块，看它们落在哪类位置：图像边缘、钟表圆心、指针、钟面其他、钟面外。再和"随便挑同样多图块"的比例对比，并把它们画在 3 张钟表图上。

**不生成、不回放、不用 GPU**：只读已存的中性回答注意力数组，几分钟完成。

## 规则

- 不要修改任何代码、参数、阈值或清单，也不要额外运行检查命令（下面列出的命令除外）。
- 报错就停下，汇报错误信息；不要自己打补丁，不要自动重试。
- 全部在标注用的那份仓库里做。不要上传 npz。
- 不要使用 GPU；不要碰 50 号服务器上卡住的 GPU4、GPU6 和那个挂起的进程。

## A. 51：Gemma 4 31B · ClockQA

```bash
R=/mnt/sdb/linxin/repo-gemma4
conda activate gemma4
git -C $R pull
cd $R && python -m pytest tests/test_fp.py -q -rs   # 应全部通过，没有 skipped
G=/mnt/sdb/linxin/gemma4-cq-20261029
S2=/mnt/sdb/linxin/gemma4-cq-sources-all-20261102
python run.py fp-sources --phase sink-check --cq $G --root $S2 --label "Gemma 4 31B · ClockQA" 2>&1 | tee $S2/sink_check.log
D=$R/results/gemma4_cq_20261029/reading_layers_20261102
mkdir -p $D/sink_check
cp $S2/sink_check_report.md $S2/sink_check_report.json $S2/sink_check.log $D/
cp $S2/sink_check/*.png $D/sink_check/
```

在 `$R` 里提交并推送；推送被拒就先 `git pull --rebase` 再推。

## B. 59：Qwen3.8-27B · ClockQA

```bash
L=/mnt/sdb/linxin/repo-label
git -C $L pull
cd $L && python -m pytest tests/test_fp.py -q -rs
CQ=/mnt/sdb/linxin/conflict-cq-20261028
S=/mnt/sdb/linxin/sources-cq-20261029
python run.py fp-sources --phase sink-check --cq $CQ --root $S --label "Qwen3.8-27B · ClockQA" 2>&1 | tee $S/sink_check.log
D=$L/results/sources_cq_20261029
mkdir -p $D/sink_check
cp $S/sink_check_report.md $S/sink_check_report.json $S/sink_check.log $D/
cp $S/sink_check/*.png $D/sink_check/
```

在 `$L` 里提交并推送。

## 汇报

1. 每台的 pytest 最后一行；
2. 两份 `sink_check_report.md` 全文。
