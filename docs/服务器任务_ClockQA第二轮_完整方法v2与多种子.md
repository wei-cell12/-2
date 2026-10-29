# 服务器任务：ClockQA 第二轮——完整方法第三版（full3）+ 多种子

> 2026-10-29 更新：主方法从 full2 换成 **full3**。full3 的核查器是只看图和问题的“盲证人”，完全不看候选。
> full2 的候选是主模型在用户压力下写的，经常含有用户的答案，会成为暗示。
> 如果第二轮已经开始：
> 1. 用 `kill -TERM` 停掉 4 个 run 进程；
> 2. 已经生成的 base、itsm、full2 记录都保留，不要删；
> 3. 更新代码，按下面的步骤做 full3 的预检，再重新启动。base 和 itsm 已完成的会自动跳过。

交给 59 上的 agent。代码已写好，按步骤运行。

**规则**
- 不要修改任何代码、参数、阈值或说明，也不要额外运行检查命令。
- 报错或预检不通过就停下，汇报错误信息和 `$M/failures/` 里的内容。不要自己打补丁，不要自动重试，不要自己清理 `failures/`。
- 停进程用 `kill -TERM`，不要用 Ctrl-C。不要停止或抢占别人的进程。

## 做什么

结果目录沿用第一轮的 `$M`。第一轮的结果和时间判读都保留，新结果放在新的子目录里。

| 方法 | 种子 | 新生成 | 说明 |
|---|---|---|---|
| full3 | 42、43、44 | 372 | 完整方法第三版，见下 |
| base（原模型） | 43、44 | 248 | 种子 42 就是已有的原模型记录，不重跑 |
| itsm | 43、44 | 248 | 种子 42 第一轮已有 |

**full3 和第一轮 full 的区别：**
1. **锚点一**：仍只在两段窗口里干预；参照改回 ClockQA 上验证过的 v1 参照，即中性回答整个阶段的平均，存在 `$M/refs/`。
2. **核查器是盲证人**：只给图和问题，完全不给候选，也不给用户的说法。它独立读图两次（不同种子），每次都给出自己的答案：
   - 两次答案相同，就接回第一份读数，并注明两次一致；
   - 两次答案不同，就把两份都接回，由主模型自己对比。
3. **核查只能由检测触发**：模型自己写的 `<verify>`、任何 `<result>`，以及思考结束后出现的这两种标记，都会被屏蔽。第一轮有 10 条回答在最终回答里写了 `<verify>`。

**耗时**：共 868 次生成，约 45 GPU·小时，4 张 A100 约 11–12 小时。

## 步骤

```bash
cd <原来跑实验的那份仓库>        # 第一轮的进程都已结束，可以直接更新
git pull                          # 应包含 full2
python -m pytest tests/test_fp.py -q   # 应全部通过（36 项）
CQ=/mnt/sdb/linxin/conflict-cq-20261028
W=/mnt/sdb/linxin/conflict-work-20261028
M=/mnt/sdb/linxin/compare-cq-20261029-a100
C=<第一轮用的配置文件>
```

### 1. 预检（GPU 0，约 10 分钟）

```bash
CUDA_VISIBLE_DEVICES=0 python run.py fp-compare --phase gate --config $C --cq $CQ --work $W --root $M --arms full3 base 2>&1 | tee $M/gate3.log
```

最后一行是 `ok=False` 就停下汇报；是 `ok=True` 就继续。

### 2. 正式运行（4 张卡同时，各一个进程）

```bash
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$k nohup python run.py fp-compare --phase run --config $C --cq $CQ --work $W --root $M \
    --arms full3 base itsm --seeds 42 43 44 --shard $k/4 > $M/run3_$k.log 2>&1 &
done
```

任务已经在 4 个分片之间交错分配，各卡的工作量大致相同。中途停下后，重新运行同一条命令，会跳过已完成的。

### 3. 时间判读（和生成并行，与第一轮相同的流程）

在标注用的那份仓库 `$L` 里先 `git -C $L pull`，然后按第一轮的做法每 2 小时左右做一批：
1. `time-export`；
2. 判读 `requests_new.jsonl`，写到新文件 `judgments_raw_<批次号>.jsonl`，批次号接着第一轮往下编；
3. `time-import`；
4. `summary`。

**判读者要和第一轮一样，用 GPT-6 Luna、max**，保证前后两轮判读口径一致。

### 4. 上传

每批判读后都上传一次进度，生成全部结束后再上传最后一次：

```bash
D=$L/results/compare_cq_20261029/a100
cp -r $M/full3* $M/base_s* $M/itsm_s* $M/time_judge $D/
cp -r $M/full2* $D/ 2>/dev/null       # full2 已有的部分也上传，作为对照
cp $M/compare_summary.json $M/*.log $D/
mkdir -p $D/refs && cp $M/refs/*.json $D/refs/      # 不传 npz
```

先 `git pull`，再提交并推送。

## 汇报

1. 预检最后 5 行。
2. 每批的 `time_import.log` 最后一行。
3. 最后一次的 `summary.log` 里 `by_method` 那一段。
