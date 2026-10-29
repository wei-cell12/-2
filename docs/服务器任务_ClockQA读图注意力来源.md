# 服务器任务：ClockQA 读时间时，注意力看图像还是看用户的话

交给服务器上的 agent。代码已写好（`run.py fp-sources`），按步骤运行。

**规则**
- 不要修改任何代码、参数、阈值或清单，也不要额外运行检查命令。
- 报错就停下，汇报错误信息和 `$S/failures/` 里的内容。不要自己打补丁，不要自动重试，不要自己清理 `failures/`。
- 停进程用 `kill -TERM`，不要用 Ctrl-C。不要停止或抢占别人的进程，包括正在跑的第二轮（full3 / base / itsm）。
- **全部在标注用的那份仓库 `$L` 里做**，不要在第二轮正在用的仓库里 `git pull`。

## 做什么

不生成新回答。把原模型已有的 62 题 × 3 种条件（中性、用户说错、用户说对，共 186 条，`$CQ/obs/records`）原样喂回模型做一次前向。在 16 个标准注意力层的每个头上，记录模型生成每个 token 时，注意力分别落在这几类位置上的总量：图像、题干、用户断言（时间本身、其余文字）、模板、模型自己复述的用户时间、模型自己写出的正确时间、其余已生成文字。

然后在 CPU 上把思考切成句子，找出“读图句”，比较这些句子上“图像注意力 / 用户断言注意力”的比值：
- 坚持组 vs 采纳组；
- 用户说错 vs 用户说对；
- 同一条回答里，写出用户时间的读图句 vs 写出正确时间的读图句；
- 另外给出前 512 个 token 的时间曲线。

## 步骤

```bash
L=/mnt/sdb/linxin/repo-label                     # 标注用的那份仓库
git -C $L pull
cd $L && python -m pytest tests/test_fp.py -q -rs  # 应全部通过；test_sources_recorder_matches_direct 必须是通过，不能是 skipped
CQ=/mnt/sdb/linxin/conflict-cq-20261028          # 只读：题目、原模型记录、中性注意力
M=/mnt/sdb/linxin/compare-cq-20261029-a100       # 只读：用其中的 time_judge/judgments.json（原模型的判读结果）
S=/mnt/sdb/linxin/sources-cq-20261029            # 新目录
C=<第一轮用的配置文件>
mkdir -p $S
```

如果 pytest 里 `test_sources_recorder_matches_direct` 显示 skipped 或失败，停下汇报。

### 1. 回放（GPU）

需要一整张 A100 80GB（模型约 55GB）。用 59 上空闲的卡；没有空闲的卡，就等第二轮结束后再跑，不要抢占。

```bash
cd $L
CUDA_VISIBLE_DEVICES=<空闲的卡号> python run.py fp-sources --phase replay --config $C --cq $CQ --root $S 2>&1 | tee $S/replay.log
```

- 共 186 条，单卡预计 1 小时左右。有多张空闲卡时，每个进程加 `--shard k/n`，日志写 `replay_k.log`，所有进程用同一个 `--root $S`。
- 每条会打印：输出 token 数、耗时、“前20名”（回放时实际 token 进模型前 20 名的比例，低于 0.98 会自动报错）、“和偏差”、“片段一致”。
- 中途停下后，重新运行同一条命令会跳过已完成的。

### 2. 分析（CPU）

```bash
cd $L
python run.py fp-sources --phase analyze --cq $CQ --root $S --judgments $M/time_judge/judgments.json 2>&1 | tee $S/analyze.log
```

输出 `$S/sources_report.md` 和 `$S/sources_summary.json`。

## 上传

```bash
D=$L/results/sources_cq_20261029
mkdir -p $D/replay
cp $S/sources_report.md $S/sources_summary.json $S/*.log $D/
cp $S/replay/*.json $D/replay/
```

**不要上传** `$S/replay/*.npz` 和 `$S/refs/` 里的 npz。在 `$L` 里提交并推送。

## 汇报

1. pytest 的最后一行，以及 `-rs` 列出的跳过项（应为没有）。
2. 回放是否 186 条全部完成；`replay*.log` 里“前20名”的最小值、“和偏差”的最大值，以及“片段一致”不是全 True 的条目。
3. `analyze.log` 全文。
4. `sources_report.md` 全文。
