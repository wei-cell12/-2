# 服务器任务：ClockQA 被带偏时注意力从指针的哪一段掉下来；softmax 之前的注意力分数（Qwen3.8）

交给 59 上的 agent。第 1 步用 GPU 回放（只前向，不生成，约 5 分钟），第 2 步只用 CPU。

**规则**
- 不要修改任何代码、参数、阈值或清单，也不要额外运行检查命令（下面列出的命令除外）。
- 报错就停下，汇报错误信息和 `failures/` 里的内容。不要自己打补丁，不要自动重试，不要自己清理 `failures/`。
- 停进程用 `kill -TERM`，不要用 Ctrl-C；不要停止或抢占别人的进程。
- 全部在标注用的那份仓库 `$L` 里做。不要上传 npz。

## 做什么

- **分段**：每根真实指针方向左右 15° 的扇形，按离圆心的距离分成两段：
  - 内段：0.15 到 0.6 个半径，长针、短针都有；
  - 外段：0.6 到 1.05 个半径，只有长针伸得到。

  比较中性和用户说错两条回答（思考前 64、128、256、512 个 token），落在每一段的注意力比例变了多少。
- **注意力分数**：再回放一次 186 条回答，额外记录 softmax 之前的分数，也就是 query 和 key 的匹配程度。每个查询对每个图像 token 的分数，减去它对全部图像 token 分数的平均。然后比较各段的分数在中性和用户说错之间怎么变。
- **分组**：按用户时间的类型（对调 / 偏移 / 不同）× 用户说错时的结果（守住 / 采纳 / 答成别的错误时间）。

## 0. 准备

```bash
L=/mnt/sdb/linxin/repo-label
git -C $L pull
cd $L && python -m pytest tests/test_fp.py -q -rs   # 应全部通过，没有 skipped
CQ=/mnt/sdb/linxin/conflict-cq-20261028
M=/mnt/sdb/linxin/compare-cq-20261029-a100
S=/mnt/sdb/linxin/sources-cq-20261029
C=/mnt/sdb/linxin/pathvqa-fp-a800-20261027/repo/configs/qwen38_fp_local.yaml
```

## 1. 回放并记录注意力分数（GPU）

186 条，每条一次前向。4 张卡约 5 分钟。只用空闲的 A100，没有空闲就等。

```bash
cd $L
# 有 n 张空闲卡时，第 k 张（k = 0 … n-1）：
CUDA_VISIBLE_DEVICES=<卡号> python run.py fp-sources --phase reading-scores --config $C --cq $CQ --root $S --shard k/n 2>&1 | tee $S/reading_scores_k.log
```

每条会和上次 reading 存的看图注意力核对，差太大会自动报错停下。

## 2. 分段分析（CPU）

```bash
python run.py fp-sources --phase reading-bands --cq $CQ --root $S --judgments $M/time_judge/judgments.json 2>&1 | tee $S/reading_bands.log
```

## 上传

```bash
D=$L/results/sources_cq_20261029
mkdir -p $D/reading_scores
cp $S/reading_bands_report.md $S/reading_bands_report.json $S/reading_bands.log $S/reading_scores_*.log $D/
cp $S/reading_scores/*.json $D/reading_scores/
```

不要上传 npz。在 `$L` 里提交并推送。

## 汇报

1. pytest 最后一行。
2. 每个回放进程日志的最后 3 行。
3. `reading_bands_report.md` 全文。
