# 服务器任务：冲突方向（PathVQA 75 题 + ClockQA 62 题）

交给服务器上的 agent。代码已写好（`run.py fp-conflict-setup`、`fp-observe`、`fp-conflict`），按步骤运行。**不要修改任何代码、参数、阈值或清单**；报错或检查不通过就停下汇报错误信息和对应目录下 `failures/` 里的内容，不要自己打补丁、不要自动重试、不要自己清理 `failures/`。停进程用 `kill -TERM`，不要用 Ctrl-C。不要停止或抢占别人的进程。卡号由我指定。

## 做什么

1. 补生成“用户说对”的轨迹：
   - PathVQA：75 题各生成一条 `belief_right`。
   - ClockQA：62 题各生成三条：中性、`belief`（用户说错）、`belief_right`（用户说对）。
2. 对每条 `belief` / `belief_right` 轨迹，把同一段文字分别接在“用户说错”和“用户说对”两种提示后面，各整段前向一次，记录隐藏状态之差。
3. 在 CPU 上按数据集求“冲突方向”。
4. 把所有轨迹投影到冲突方向上，看它能否区分有冲突和没冲突，并且能否从 PathVQA 用到 ClockQA，或者反过来。

## 步骤

```bash
git pull                                       # 应包含 fp-conflict
python -m pytest tests/test_fp.py -q           # 应全部通过（包括 test_conflict_*）
R=/mnt/sdb/linxin/pathvqa-fp-a800-20261027     # A800 主实验目录，只读
PV=/mnt/sdb/linxin/conflict-pv-20261028        # 新目录
CQ=/mnt/sdb/linxin/conflict-cq-20261028        # 新目录
W=/mnt/sdb/linxin/conflict-work-20261028       # 新目录
N=<仓库>/results/attn_intervene_20261025/clockqa_server59/five_pressure/native_outputs.jsonl
C=<A800 主实验用的配置文件>                     # 两个数据集都用它，不改任何设置
mkdir -p $W
```

### 1. 建目录（CPU）

```bash
python run.py fp-conflict-setup --pv-src $R --pv $PV --native $N --cq $CQ 2>&1 | tee $W/setup.log
```

PathVQA 会选出 A800 里中性和 belief 观察都有的题，预计 75 题。ClockQA 会从原始观察记录里找到每题的图像，并核对 sha256。

如果报“找不到 sha256 一致的图像”，**停下汇报**，由我们给出 ClockQA 图像目录后再加 `--image-dirs <目录>` 重跑。

### 2. 生成（GPU）

```bash
CUDA_VISIBLE_DEVICES=<卡号> python run.py fp-observe --config $C --root $PV --conditions belief_right 2>&1 | tee $W/gen_pv.log
CUDA_VISIBLE_DEVICES=<卡号> python run.py fp-observe --config $C --root $CQ --conditions neutral belief belief_right 2>&1 | tee $W/gen_cq.log
```

**耗时：** 单卡跑，PathVQA 75 条约 2 小时，ClockQA 186 条约 8 小时。

**多张卡并行：** 每条命令都可以加 `--shard k/n`，同一个 root 下各进程用不同的 k，日志分别写 `gen_pv_k.log`、`gen_cq_k.log`。两个数据集也可以同时在不同卡上跑。

**中途停下后：** 重新运行同一条命令，会跳过已完成的。

### 3. 配对前向（GPU，等第 2 步全部完成）

```bash
CUDA_VISIBLE_DEVICES=<卡号> python run.py fp-conflict --phase pairs --config $C --pv-src $R --pv $PV --cq $CQ --work $W 2>&1 | tee $W/pairs.log
```

共约 274 条轨迹，每条前向两次。可以加 `--shard k/n` 并行，日志写 `pairs_k.log`。

### 4. 求方向（CPU，等第 3 步全部完成）

```bash
python run.py fp-conflict --phase directions --pv-src $R --pv $PV --cq $CQ --work $W 2>&1 | tee $W/directions.log
```

### 5. 投影（GPU）

```bash
CUDA_VISIBLE_DEVICES=<卡号> python run.py fp-conflict --phase project --config $C --pv-src $R --pv $PV --cq $CQ --work $W --out $W/conflict_proj.json 2>&1 | tee $W/project.log
```

共约 700 条，每条前向一次。多张卡时：
- 加 `--shard k/n`，`--out` 改成 `$W/conflict_proj_k.json`，日志写 `project_k.log`；
- 不同分片的输出文件名不能相同。

## 上传

```bash
D=<仓库>/results/conflict_direction_20261028
mkdir -p $D
python run.py fp-public --root $PV --out $D/pv
python run.py fp-public --root $CQ --out $D/cq
cp $W/directions_summary.json $W/conflict_proj*.json $W/*.log $D/
```

**不要上传：**
- `$W/pairs/`、`$W/directions.npz`；
- 任何 NPZ，包括 `$PV`、`$CQ` 下 `obs/attn/` 里的文件。

提交并推送。

## 汇报

1. 第 1 步：两个数据集各多少题。
2. 第 2 步：每个数据集生成了多少条，是否全部完成。
3. 第 3 步：是否全部通过；`top20` 的最小值。
4. 第 4 步：日志最后一行，包括各数据集可用条数、PathVQA 与 ClockQA 方向的逐层余弦。
5. 第 5 步：一共多少条，是否全部通过。

## 补充：给 ClockQA 以前实验的轨迹打冲突分数（GPU，不生成）

`results/conflict_direction_20261028/external_cq_trajs.jsonl` 里是 ClockQA 以前实验留下的 591 条思考文字，这些实验都不是本次生成的：
- 原始观察：中性加 5 种压力，共 372 条；
- 锚点一干预：restore_E 99 条、restore_rand 99 条、base 21 条。

做法：每条重新切成 token，在原来的受压提示下前向一次，投影到第 4 步求出的方向上。需要先完成两个数据集一起的第 4 步，即 `$W/directions.npz` 里要同时有 PathVQA 和 ClockQA。

```bash
X=<仓库>/results/conflict_direction_20261028/external_cq_trajs.jsonl
CUDA_VISIBLE_DEVICES=<卡号> python run.py fp-conflict --phase external --config $C --pv-src $R --pv $PV --cq $CQ --work $W --external $X --out $W/external_proj.json 2>&1 | tee $W/external.log
```

多张卡时：
- 加 `--shard k/n`；
- `--out` 改成 `$W/external_proj_k.json`，日志写 `external_k.log`。

上传：`cp $W/external_proj*.json $W/external*.log <仓库>/results/conflict_direction_20261028/`。
