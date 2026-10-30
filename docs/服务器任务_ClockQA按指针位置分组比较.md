# 服务器任务：ClockQA 按“用户时间的指针位置是否和真实指针重合”分组，比较思考开头看图的分布

交给服务器上的 agent。只用 CPU，读上一步 reading 存好的数据（`$S/reading/*.npz`）和 62 张钟面图像，不跑模型。几分钟。

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
M=/mnt/sdb/linxin/compare-cq-20261029-a100
S=/mnt/sdb/linxin/sources-cq-20261029
python run.py fp-sources --phase reading-claim --cq $CQ --root $S --judgments $M/time_judge/judgments.json 2>&1 | tee $S/reading_claim.log
```

输出 `$S/reading_claim_report.md`、`$S/reading_claim_summary.json`，以及每题一张核对图 `$S/reading_claim_check/*.png`
（绿 = 真实指针区域，红 = 用户时间的指针区域，蓝圈 = 自动估计的表盘）。

## 上传

```bash
D=$L/results/sources_cq_20261029
cp $S/reading_claim_report.md $S/reading_claim_summary.json $S/reading_claim.log $D/
mkdir -p $D/reading_claim_check && cp $S/reading_claim_check/*.png $D/reading_claim_check/
```

在 `$L` 里提交并推送。

## 汇报

1. pytest 最后一行。
2. `reading_claim.log` 全文。
3. `reading_claim_report.md` 全文。
