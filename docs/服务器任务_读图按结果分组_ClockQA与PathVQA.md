# 服务器任务：思考开头 token 的看图，按"用户说错时的最终结果"分组比较（ClockQA 现在做；PathVQA 补齐后做）

交给 59 上的 agent。代码已写好（`run.py fp-sources --phase reading-groups / reading-pv`）。

**规则**
- 不要修改任何代码、参数、阈值或清单，也不要额外运行检查命令（下面列出的命令除外）。
- 报错就停下，汇报错误信息和 `failures/` 里的内容。不要自己打补丁，不要自动重试，不要自己清理 `failures/`。
- 停进程用 `kill -TERM`，不要用 Ctrl-C；不要停止或抢占别人的进程。
- 全部在标注用的那份仓库 `$L` 里做。不要上传 npz。

## 算什么

每道题、思考的前 W 个 token（W = 64 / 128 / 256 / 512），对 96 个看图头分别算，再对这道题的看图头取平均：
- **超出噪声的量**：中性回答和用户说错回答的看图分布差别（JSD），减去同一条回答前后两半之间的差别；
- **RRAR**：相关区域的平均注意力 ÷ 全图平均注意力（与 CVPR 2026 那篇 "Deeper Thought, Weaker Aim" 的定义相同）。区域分两种：
  - 关键区域 E，即中性时注意力最高的 20% 图像块；
  - 真实指针区域，即指针方向左右 15° 的扇形，只有 ClockQA 有。

然后按用户说错时的最终结果分组：守住正确答案 / 采纳用户的答案 / 答成别的错误时间（PathVQA 第三组是"不下结论"）。各组之间用按题的置换检验比较。全部题和"中性时答对的题"分别报。

## A. ClockQA（现在做，只用 CPU，几分钟）

用上次 reading 存好的数据，不跑模型。

```bash
L=/mnt/sdb/linxin/repo-label
git -C $L pull
cd $L && python -m pytest tests/test_fp.py -q -rs   # 应全部通过，没有 skipped
CQ=/mnt/sdb/linxin/conflict-cq-20261028
M=/mnt/sdb/linxin/compare-cq-20261029-a100
S=/mnt/sdb/linxin/sources-cq-20261029
python run.py fp-sources --phase reading-groups --cq $CQ --root $S --judgments $M/time_judge/judgments.json 2>&1 | tee $S/reading_groups.log
```

输出 `$S/reading_groups_report.md`、`$S/reading_groups_summary.json`。

上传：

```bash
D=$L/results/sources_cq_20261029
cp $S/reading_groups_report.md $S/reading_groups_summary.json $S/reading_groups.log $D/
```

在 `$L` 里提交并推送。

**汇报 A**：pytest 最后一行；`reading_groups.log` 全文；`reading_groups_report.md` 全文。

## B. PathVQA（等"Qwen3.8 PathVQA 补齐"那个任务全部做完、`yn-summary` 也跑完以后再做）

沿用补齐任务里的目录：

```bash
C=/mnt/sdb/linxin/pathvqa-fp-a800-20261027/repo/configs/qwen38_fp_local.yaml
P=/mnt/sdb/linxin/qwen38-pv-20261031
PM=/mnt/sdb/linxin/qwen38-pathvqa-compare-20261031
PS=/mnt/sdb/linxin/qwen38-pv-sources-20261031      # 新目录
mkdir -p $PS
```

### B1. 回放（GPU）

先把每题的中性回答整条回放一次，建参照（看图头和关键区域）；再把三种条件回答的思考开头 token 各回放一次。只做前向，不生成。原模型 300 条，4 张卡约 15 分钟。只用空闲的 A100，没有空闲就等。

```bash
cd $L
# 有 n 张空闲卡时，第 k 张（k = 0 … n-1）：
CUDA_VISIBLE_DEVICES=<卡号> python run.py fp-sources --phase reading-pv --config $C --cq $P --root $PS --shard k/n 2>&1 | tee $PS/reading_pv_k.log
```

如果某题的中性回答没有写完思考（缺回答阶段），程序会报错停下，把那题的题号汇报给我，不要自己跳过。

### B2. 分组分析（CPU）

```bash
python run.py fp-sources --phase reading-groups --cq $P --root $PS --yn-judgments $PM/yesno_judge/judgments.json 2>&1 | tee $PS/reading_groups.log
```

上传：

```bash
D=$L/results/qwen38_pathvqa_20261031
mkdir -p $D/reading $D/refs
cp $PS/reading_groups_report.md $PS/reading_groups_summary.json $PS/*.log $D/
cp $PS/reading/*.json $D/reading/
cp $PS/refs/*.json $D/refs/
```

不要上传 npz。在 `$L` 里提交并推送。

**汇报 B**：每个回放进程日志的最后 3 行；`reading_groups.log` 全文；`reading_groups_report.md` 全文。
