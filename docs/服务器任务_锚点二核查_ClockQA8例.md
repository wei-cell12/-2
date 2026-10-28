# 服务器任务：锚点二（冲突触发的独立核查）续写实验，ClockQA 8 例

交给服务器上的 agent。代码已写好（`run.py fp-verify`），按步骤运行。**不要修改任何代码、参数、阈值或清单**；报错就停下汇报错误信息和 `$W/failures/` 里的内容，不要自己打补丁、不要自动重试。停进程用 `kill -TERM`，不要用 Ctrl-C。不要停止或抢占别人的进程。卡号由我指定。

## 做什么

`results/conflict_direction_20261028/verify_cases_cq8.json` 里有 8 条 ClockQA 轨迹，都来自锚点一干预（restore_E）后的结果，属于"看清仍谄媚"：模型在思考里明确写出了正确读数，但最后还是采纳了用户的说法。

对每条轨迹：

1. 用冲突方向找到触发点：第 48 层，阈值 −9.28。
2. 从触发点开始，按三种条件各续写 3 次：
   - **A 基线**：不加任何东西。
   - **B 只有工具说明**：系统提示里加入核查工具的说明，模型自己决定要不要调用。
   - **C 完整方法**：在 B 的基础上，在触发点插入 `<verify>`。模型自己写出要核对的候选答案；裁判（同一个模型、单独的对话，不知道候选答案是谁提出的）核对后，结果放进 `<result>` 接回，模型接着写完。
3. 比较三种条件下最终答案是"采纳"还是"坚持（答对）"。

## 步骤

```bash
git pull                                       # 应包含 fp-verify
python -m pytest tests/test_fp.py -q           # 应全部通过（包括 test_verify_*）
CQ=/mnt/sdb/linxin/conflict-cq-20261028
W=/mnt/sdb/linxin/conflict-work-20261028       # 用其中两个数据集一起求出的 directions.npz
C=<A800 主实验用的配置文件>
K=<仓库>/results/conflict_direction_20261028/verify_cases_cq8.json
CUDA_VISIBLE_DEVICES=<卡号> python run.py fp-verify --config $C --cq $CQ --work $W --cases $K --out $W/verify_cq8.json 2>&1 | tee $W/verify_cq8.log
```

**耗时**：共 72 次续写，加上裁判调用，单卡约 3–4 小时。

**多卡并行**：
- 加 `--shard k/n`，按题目分片；
- `--out` 改为 `$W/verify_cq8_k.json`，日志写 `verify_cq8_k.log`。

**中断后重跑**：已完成的续写存在 `$W/verify/` 里，重新运行同一命令会跳过它们。

## 上传

```bash
cp $W/verify_cq8*.json $W/verify_cq8*.log <仓库>/results/conflict_direction_20261028/
```

提交并推送。不要上传 NPZ。

## 汇报

日志最后一行（三种条件的汇总），以及每条题目的触发位置。
