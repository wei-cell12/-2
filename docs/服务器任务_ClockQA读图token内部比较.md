# 服务器任务：ClockQA 同一条回答内部，被诱导的读数 token 与正确读数 token 的注意力比较

交给服务器上的 agent。只用 CPU，用上一次回放已经存好的数据（`$S/replay/*.npz`），不重新回放、不生成。

**规则**
- 不要修改任何代码、参数、阈值或清单，也不要额外运行检查命令。
- 报错就停下，汇报错误信息。不要自己打补丁，不要自动重试。
- 全部在标注用的那份仓库 `$L` 里做。

## 步骤

```bash
L=/mnt/sdb/linxin/repo-label
git -C $L pull
cd $L && python -m pytest tests/test_fp.py -q -rs   # 应全部通过，没有 skipped
CQ=/mnt/sdb/linxin/conflict-cq-20261028
M=/mnt/sdb/linxin/compare-cq-20261029-a100
S=/mnt/sdb/linxin/sources-cq-20261029             # 上一次回放的目录（已有 replay/*.npz）
python run.py fp-sources --phase tokens --cq $CQ --root $S --judgments $M/time_judge/judgments.json 2>&1 | tee $S/tokens.log
```

输出 `$S/tokens_report.md` 和 `$S/tokens_summary.json`。

## 上传

```bash
D=$L/results/sources_cq_20261029
cp $S/tokens_report.md $S/tokens_summary.json $S/tokens.log $D/
```

在 `$L` 里提交并推送。不要上传 npz。

## 汇报

1. pytest 最后一行。
2. `tokens.log` 全文。
3. `tokens_report.md` 全文。
