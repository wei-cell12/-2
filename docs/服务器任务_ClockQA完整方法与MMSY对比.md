# 服务器任务：ClockQA 完整方法（锚点一 + 锚点二）与 MM-SY、ITSM 对比

交给服务器上的 agent。代码已写好（`run.py fp-compare`），按步骤运行。

**规则**
- 不要修改任何代码、参数、阈值或清单，也不要额外运行检查命令。
- 报错或预检不通过就停下，汇报错误信息和 `$M/failures/` 里的内容。不要自己打补丁，不要自动重试，不要自己清理 `failures/`。
- 停进程用 `kill -TERM`，不要用 Ctrl-C。不要停止或抢占别人的进程。
- 卡号由我指定。

## 做什么

对 ClockQA 全部 62 题，在两种条件下各生成一次（种子 42）：
- 用户说错（belief）；
- 用户说对（belief_right）。

三种方法：
- **mmsy**：MM-SY（ICLR 2025）。在后 8 个标准注意力层，把图像 token 的注意力打分 e 改为 e + 0.9·|e|。
- **full**：完整方法。
  - 锚点一：v1 restore_E，只在两段里干预：思考开头 256 个 token；锚点二接回 `<result>` 起，到结果文字结束后 256 个 token。其余位置用原注意力函数。参照来自同一题中性回答的前 256 个 token，程序会先用种子 42 只生成中性回答的前 256 个 token，存在 `$M/refs_prefix/`，每题约半分钟。
  - 锚点二：生成时实时计算冲突分数，超过阈值就插入 `<verify>`，由裁判核对后接回 `<result>`。
- **itsm**：ITSM（arXiv 2408.11261）的谄媚感知对比解码。中性提问（去掉用户那句话）和原提问两条分支同步生成，每一步 logits = (1+α)·中性 − α·原提问，再做自适应可信度过滤。每个 token 要多一次前向，所以每次生成大约慢一倍。

原模型的结果直接用已有的 `$CQ/obs` 记录，不重新生成。

## 步骤

```bash
git pull                                       # 应包含 fp-compare
python -m pytest tests/test_fp.py -q           # 应全部通过（33 项，包括 test_compare_*）
CQ=/mnt/sdb/linxin/conflict-cq-20261028        # 只读：题目、原模型记录、中性注意力
W=/mnt/sdb/linxin/conflict-work-20261028       # 只读：directions.npz（两个数据集一起求出的版本）
M=/mnt/sdb/linxin/compare-cq-20261029          # 新目录
C=<A800 主实验用的配置文件>
mkdir -p $M
```

### 1. 预检（GPU，约 15 分钟）

```bash
CUDA_VISIBLE_DEVICES=<卡号> python run.py fp-compare --phase gate --config $C --cq $CQ --work $W --root $M 2>&1 | tee $M/gate.log
```

预检会做四件事：
1. 用新代码以原模型设置生成 2 题，看能否和已有记录逐 token 一致。
2. 检查生成时实时算的冲突分数，和整段重算的结果是否一致。
3. full 跑 1 题，包括生成这道题的前缀参照，并检查它是否和已有的中性记录逐 token 一致。
4. mmsy 跑 1 题。
5. itsm 跑 1 题，并检查中性分支逐步计算和整段计算是否一致。

预检可以重复运行，已经生成过的会直接复用。通过的方法会累计记在 `compare_gate.json` 的 `arms_ok` 里，正式运行只接受已经通过预检的方法。

日志最后一行如果是 `ok=False`，**停下汇报** `$M/compare_gate.json`。如果是 `ok=True`，继续下一步；这时即使 `deterministic=False` 也继续，但要在汇报里说明。

### 2. 正式运行（GPU）

```bash
CUDA_VISIBLE_DEVICES=<卡号> python run.py fp-compare --phase run --config $C --cq $CQ --work $W --root $M --arms mmsy full itsm 2>&1 | tee $M/run.log
```

- 共 372 次生成（每种方法 124 次），预检里已经做完 3 次。full 每次触发后还要多跑 2 次裁判；itsm 每次大约慢一倍。单卡约 20 小时。
- 多张卡时：每个进程加 `--shard k/n`，日志写 `run_k.log`，所有进程用同一个 `--root $M`。
- 中途停下后，重新运行同一条命令，会跳过已完成的。

### 3. 取出时间（判读子 agent，等所有生成进程结束）

答案对错不再用正则取时间，改成由判读模型取出“最终断言的时间”，再由程序按严格规则判定。

```bash
git pull                                       # 应包含 time-export / time-import（commit 见对话）
python run.py fp-compare --phase time-export --cq $CQ --root $M 2>&1 | tee $M/time_export.log
```

用之前判读 PathVQA 的同一个判读者（Gemini 3.8 Flash，medium，子 agent），按 `$M/time_judge/提取说明.md` 逐行处理 `$M/time_judge/requests.jsonl`：
- 每行输出一个 JSON，写到 `$M/time_judge/judgments_raw.jsonl`；
- 不要把正确答案、方法名或条件告诉判读者；
- 不要改说明。

```bash
python run.py fp-compare --phase time-import --cq $CQ --root $M --judgments $M/time_judge/judgments_raw.jsonl 2>&1 | tee $M/time_import.log
```

日志里“缺 N 条”或“格式不对 N 条”不为 0 时，只把缺的或格式不对的那几条重新交给判读者，再运行一次 time-import。

### 4. 汇总（CPU）

```bash
python run.py fp-compare --phase summary --cq $CQ --root $M --out $M/compare_summary.json 2>&1 | tee $M/summary.log
```

## 上传

```bash
D=<仓库>/results/compare_cq_20261029
mkdir -p $D
cp -r $M/mmsy $M/full $M/itsm $M/loop $D/
cp $M/compare_gate.json $M/compare_summary.json $M/*.log $D/
cp -r $M/time_judge $D/
```

**不要上传** `$M/refs_prefix/` 里的 npz；同目录下的 json 要上传：`mkdir -p $D/refs_prefix && cp $M/refs_prefix/*.json $D/refs_prefix/`。提交并推送。

## 汇报

1. 预检日志的最后 5 行。
2. 正式运行是否全部完成。
3. `summary.log` 全文。
