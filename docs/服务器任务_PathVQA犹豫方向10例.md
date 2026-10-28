# 服务器任务：PathVQA 犹豫方向（隐藏状态，10 例 + “用户说对”10 例，离线，不生成）

交给服务器上的 agent。代码已写好（`run.py fp-hidden`），按步骤运行。**不要修改任何代码、参数、阈值**；报错或检查不通过就停下汇报错误信息和 `$W/failures/` 里的内容，不要自己打补丁、不要自动重试。

## 做什么

数据与上一次 fp-spans 相同，共 52 条轨迹。

**第一步 forward（用 GPU）：**
- 每条轨迹用生成时那份提示，把“提示 + 已生成内容”整段前向一次。不生成，也不改模型。
- 取 8 层（第 8、16、…、64 层）的隐藏状态，只保留思考段，存到 `$W/hidden/`。

**第二步 analyze（只用 CPU）：** 采用留一题交叉验证。对每道题：
- 用其余 9 题的轨迹求“犹豫方向”，即“即将写出 but / however / wait / hmm 等词时的隐藏状态平均”减去对照平均；
- 再把这道题的轨迹投影到这个方向上，输出每个 token 的投影值。

统计检验由我在本地做。

## 自动检查（不通过会自动停下）

- 重新构造的提示与记录一致。
- 同一次前向中，观察轨迹实际写出的 token 在 logits 里排前 20 名的比例 ≥ 0.98，用来防止位置错位。
- 隐藏状态层数正确，没有非有限值。
- 每一折的犹豫位置和句首位置都不少于 20 个。

## 步骤

```bash
git pull                                       # 应包含 fp-hidden
python -m pytest tests/test_fp.py -q           # 应全部通过（包括 test_hidden_positions_and_folds、test_hidden_requires_hf）
T=/mnt/sdb/linxin/pathvqa-dual10-20261027      # 只读
U=/mnt/sdb/linxin/pathvqa-agree10-20261027     # 只读
W=/mnt/sdb/linxin/pathvqa-hidden10-20261028    # 新目录
C=<A800 主实验用的配置文件>                     # 不改任何设置
mkdir -p $W
df -h $W                                       # $W/hidden/ 约需 5 GB，空间不够就停下汇报

# 1. forward（GPU）
CUDA_VISIBLE_DEVICES=<卡号> python run.py fp-hidden --phase forward --config $C --roots $T $U --work $W 2>&1 | tee $W/forward.log

# 2. analyze（CPU，等 forward 全部完成后再运行）
python run.py fp-hidden --phase analyze --work $W --out $W/hidden_proj.json 2>&1 | tee $W/analyze.log
```

**两张卡各跑一半 forward 的写法：** 两个进程用同一个 `--work $W`：
- 第一张卡加 `--shard 0/2`，日志写 `forward_0.log`；
- 第二张卡加 `--shard 1/2`，日志写 `forward_1.log`。

两个都结束后再运行 analyze。

**中途停下后：** 重新运行 forward，会跳过已完成的轨迹。前提是 `failures/` 为空；出错时要先汇报，不要自己清理。

## 上传

```bash
mkdir -p <仓库>/results/pathvqa_hidden10_20261028
cp $W/hidden_proj.json $W/*.log <仓库>/results/pathvqa_hidden10_20261028/
```

`hidden_proj.json` 只有投影值，约 10 MB，可以上传。**不要上传 `$W/hidden/`**，里面是隐藏状态的 .npy 文件，约 5 GB。提交并推送。

## 汇报

1. forward：52 条是否全部完成，每条的 `top20`。
2. analyze：日志最后一行（各折方向的逐层平均余弦）。
