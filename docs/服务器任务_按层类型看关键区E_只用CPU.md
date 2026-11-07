# 服务器任务：按层类型看关键区 E（只用 CPU，读已存数组）

**目的**：之前 ClockQA 上，Gemma 滑动窗口层用"指针区域比例"看不出变化。原因是滑动层看图的注意力大多堆在少数固定图块上（attention sink），落在指针上的只有 2% 到 6%。
这次换用 PENDULUM 那次用的指标"关键区 E"（每个头在中性回答里最常看的 20% 图块），并另做一版"先去掉固定高亮图块再取 E"。全局层、滑动窗口层分开算，比较被带偏组和守住组的"用户说错 − 用户说对"。

**不生成、不回放、不用 GPU**：只读 11 月 2 日已经存好的注意力数组。数组在 `reading_all/`：Gemma 全部 60 层，前 256 个思考 token。

## 规则

- 不要修改任何代码、参数、阈值或清单，也不要额外运行检查命令（下面列出的命令除外）。
- 报错就停下，汇报错误信息；不要自己打补丁，不要自动重试。
- 全部在标注用的那份仓库里做。不要上传 npz。
- 不要使用 GPU。不要碰 50 号服务器上卡住的 GPU4、GPU6 和那个挂起的进程。

## A. 51：Gemma 4 31B · ClockQA

```bash
R=/mnt/sdb/linxin/repo-gemma4
conda activate gemma4
git -C $R pull
cd $R && python -m pytest tests/test_fp.py -q -rs   # 应全部通过，没有 skipped
G=/mnt/sdb/linxin/gemma4-cq-20261029
GM=/mnt/sdb/linxin/gemma4-compare-20261029
S2=/mnt/sdb/linxin/gemma4-cq-sources-all-20261102
python run.py fp-sources --phase reading-layers-e --cq $G --root $S2 --judgments $GM/time_judge/judgments.json --label "Gemma 4 31B · ClockQA" 2>&1 | tee $S2/reading_layers_e.log
```

上传：

```bash
D=$R/results/gemma4_cq_20261029/reading_layers_20261102
cp $S2/reading_layers_e_report.md $S2/reading_layers_e_report.json $S2/reading_layers_e.log $D/
```

在 `$R` 里提交并推送；推送被拒就先 `git pull --rebase` 再推。

## B. 59：Qwen3.8-27B · ClockQA（同样的指标，作对照）

```bash
L=/mnt/sdb/linxin/repo-label
git -C $L pull
cd $L && python -m pytest tests/test_fp.py -q -rs
CQ=/mnt/sdb/linxin/conflict-cq-20261028
M=/mnt/sdb/linxin/compare-cq-20261029-a100
S=/mnt/sdb/linxin/sources-cq-20261029
python run.py fp-sources --phase reading-layers-e --cq $CQ --root $S --judgments $M/time_judge/judgments.json --label "Qwen3.8-27B · ClockQA" 2>&1 | tee $S/reading_layers_e.log
D=$L/results/sources_cq_20261029
cp $S/reading_layers_e_report.md $S/reading_layers_e_report.json $S/reading_layers_e.log $D/
```

在 `$L` 里提交并推送。

## 汇报

1. 每台的 pytest 最后一行；
2. 两份 `reading_layers_e_report.md` 全文。
