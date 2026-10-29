# 服务器任务：ClockQA 读提示阶段，逐头比较看图位置（JSD）

交给服务器上的 agent。只用 CPU，读之前提示前向存好的数据（`$S/prompt/*.npz`），不跑模型。几分钟。

**规则**
- 不要修改任何代码、参数、阈值或清单，也不要额外运行检查命令。
- 报错就停下汇报。不要自己打补丁，不要自动重试。
- 全部在标注用的那份仓库 `$L` 里做。

## 步骤

```bash
L=/mnt/sdb/linxin/repo-label
git -C $L pull
cd $L && python -m pytest tests/test_fp.py -q -rs   # 应全部通过，没有 skipped
CQ=/mnt/sdb/linxin/conflict-cq-20261028
S=/mnt/sdb/linxin/sources-cq-20261029             # 之前提示前向用的同一个目录（已有 prompt/*.npz）
python run.py fp-sources --phase prompt-heads --cq $CQ --root $S 2>&1 | tee $S/prompt_heads.log
```

输出 `$S/prompt_heads_report.md` 和 `$S/prompt_heads_summary.json`。

## 上传

```bash
D=$L/results/sources_cq_20261029
cp $S/prompt_heads_report.md $S/prompt_heads_summary.json $S/prompt_heads.log $D/
```

在 `$L` 里提交并推送。

## 汇报

1. pytest 最后一行。
2. `prompt_heads.log` 全文。
3. `prompt_heads_report.md` 全文。
