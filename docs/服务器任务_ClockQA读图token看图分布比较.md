# 服务器任务：ClockQA 思考里的读图 token 看图的分布，三种条件相同长度逐头比较

交给服务器上的 agent。把原模型已有的 186 条回答（62 题 × 中性 / 用户说错 / 用户说对）原样喂回模型一次，只记录思考里“读图 token”在每层每头对 169 个图像 token 的注意力。不生成新回答。

**规则**
- 不要修改任何代码、参数、阈值或清单，也不要额外运行检查命令。
- 报错就停下，汇报错误信息和 `$S/failures/` 里的内容。不要自己打补丁，不要自动重试，不要自己清理 `failures/`。
- 停进程用 `kill -TERM`。不要停止或抢占别人的进程。
- 全部在标注用的那份仓库 `$L` 里做。

## 步骤

```bash
L=/mnt/sdb/linxin/repo-label
git -C $L pull
cd $L && python -m pytest tests/test_fp.py -q -rs   # 应全部通过，没有 skipped
CQ=/mnt/sdb/linxin/conflict-cq-20261028
M=/mnt/sdb/linxin/compare-cq-20261029-a100
S=/mnt/sdb/linxin/sources-cq-20261029             # 之前回放用的同一个目录（要用到 replay/*.json）
C=/mnt/sdb/linxin/pathvqa-fp-a800-20261027/repo/configs/qwen38_fp_local.yaml
```

如果 pytest 有 skipped 或失败，停下汇报。

### 1. 回放（GPU）

186 条，每条一次前向。4 张卡约 5 分钟。只用空闲的 A100，没有空闲就等。

```bash
cd $L
# 有 n 张空闲卡时，第 k 张（k = 0 … n-1）：
CUDA_VISIBLE_DEVICES=<卡号> python run.py fp-sources --phase reading --config $C --cq $CQ --root $S --shard k/n 2>&1 | tee $S/reading_k.log
```

只有一张卡时去掉 `--shard`，日志写 `reading.log`。

### 2. 分析（CPU）

```bash
cd $L
python run.py fp-sources --phase reading-analyze --cq $CQ --root $S --judgments $M/time_judge/judgments.json 2>&1 | tee $S/reading_analyze.log
```

## 上传

```bash
D=$L/results/sources_cq_20261029
cp $S/reading_report.md $S/reading_summary.json $S/reading*.log $D/
mkdir -p $D/reading && cp $S/reading/*.json $D/reading/
```

不要上传 `$S/reading/*.npz`。在 `$L` 里提交并推送。

## 汇报

1. pytest 最后一行。
2. 是否 186 条全部完成；各 `reading*.log` 最后 3 行。
3. `reading_analyze.log` 全文。
4. `reading_report.md` 全文。
