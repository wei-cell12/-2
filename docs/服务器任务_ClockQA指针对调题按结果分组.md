# 服务器任务：ClockQA 只看"用户时间能对上已有指针"（指针对调）这类题，按结果分组比较看图变化

交给 59 上的 agent。只用 CPU，读上次 reading 存好的数据，不跑模型，几分钟。

**规则**
- 不要修改任何代码、参数、阈值或清单，也不要额外运行检查命令（下面列出的命令除外）。
- 报错就停下汇报。不要自己打补丁，不要自动重试。
- 全部在标注用的那份仓库 `$L` 里做。

```bash
L=/mnt/sdb/linxin/repo-label
git -C $L pull
cd $L && python -m pytest tests/test_fp.py -q -rs   # 应全部通过，没有 skipped
CQ=/mnt/sdb/linxin/conflict-cq-20261028
M=/mnt/sdb/linxin/compare-cq-20261029-a100
S=/mnt/sdb/linxin/sources-cq-20261029
python run.py fp-sources --phase reading-split --cq $CQ --root $S --judgments $M/time_judge/judgments.json 2>&1 | tee $S/reading_split.log
D=$L/results/sources_cq_20261029
cp $S/reading_split_report.md $S/reading_split_report.json $S/reading_split.log $D/
```

在 `$L` 里提交并推送。

**汇报**：pytest 最后一行；`reading_split_report.md` 全文。
