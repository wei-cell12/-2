# 服务器任务：锚点一（读图阶段把看图注意力往参照分支拉），PathVQA 和 PENDULUM 各 20 题

**做什么**：和 ClockQA 试跑（`results/anchor1_read_20261108/`）相同的方法，换成另外两个数据集。

- **参照分支**：提示去掉用户那句话。思考开头第一段（跳过 "Thinking Process:" 这类短标题行）提到 user 时，它是复述段，参照里删掉；没提 user 时什么都不删。
- **调整窗口**：开头一段之后的 128 个 token（思考提前结束就到思考结束为止）。
- **调整方式**：
  - 看图总量只补不压：m' = max(m, m*)；
  - 图像内部分配向参照靠一半，固定位置不动。
- **两组**：base（参照分支照跑、不改）和 pull（按上式改），用同一个随机种子。

**题目**：与熵谷注入实验相同的选题（被带偏、守住各取一半，不够时互相补），代码会写进 `selection.json`。每个模型 2 个数据集 × 20 题 × 2 种条件 × 2 组 = 160 次生成。

| 模型 | PathVQA | PENDULUM |
|---|---|---|
| Qwen3.8-27B | 被带偏 15、守住 5 | 被带偏 4、守住 16 |
| Gemma 4 31B-it | 被带偏 3、守住 13、中性未解 4 | 被带偏 3、守住 17 |

**预计耗时**：按 ClockQA 试跑的速度估算，每个模型约 3 卡时（PENDULUM 回答很短，PathVQA 中等）。可以用 `--shard k/n` 分到多份并行。

## 规则

- 不要修改任何代码、参数、阈值或清单，也不要额外运行检查命令（下面列出的命令除外）。运行前确认 `git status` 干净。
- 报错就停下汇报错误信息；不要自己打补丁，不要自动重试。
- 停进程用 `kill -TERM`，不要用 Ctrl-C；不要停止或抢占别人的进程。只用我指定的服务器和卡。
- 全部在标注用的那份仓库里做。不要上传 npz。
- 不要碰 50 上卡住的 GPU4、GPU6 和那个挂起的进程。
- 判读时不告诉判读者模型名、方法、条件或正确答案。

## 1. 生成

用能找到 PathVQA、PENDULUM 原回答和图像的那份本地化配置和熵差目录（与熵谷注入实验相同）。

```bash
cd <代码目录> && git pull && git status
python -m pytest tests/test_fp.py -q -rs        # 应全部通过，没有 skipped
AQ=<结果目录>/anchor1-pv-pd-qwen38-20261108
AG=<结果目录>/anchor1-pv-pd-gemma4-20261108
mkdir -p $AQ $AG
# 每份一组卡；n 份时第 k 份（k = 0 … n−1）
CUDA_VISIBLE_DEVICES=<卡> nohup python run.py fp-anchor1 --config <Qwen 本地化配置> --root $AQ --entropy-dir <Qwen 本地化熵差目录> --model-key qwen38 --image-dirs <图像目录> --datasets PathVQA PENDULUM --shard k/n --label "Qwen3.8-27B" > $AQ/run_k.log 2>&1 &
CUDA_VISIBLE_DEVICES=<卡> nohup python run.py fp-anchor1 --config <Gemma 本地化配置> --root $AG --entropy-dir <Gemma 本地化熵差目录> --model-key gemma4 --image-dirs <图像目录> --datasets PathVQA PENDULUM --shard k/n --label "Gemma 4 31B-it" > $AG/run_k.log 2>&1 &
```

中途断了用同一条命令重跑，已完成的会自动跳过。

## 2. 判读（与熵谷注入实验相同的两步盲判读）

每个模型分别做，以 Qwen 为例（Gemma 把 `$AQ` 换成 `$AG`）：

```bash
python run.py fp-anchor1 --phase judge-export --root $AQ
```

- **第一步：盲抽取答案。** 判读者按 `$AQ/judge_extract/INSTRUCTIONS.md`，只读 `queue.json` 和 `requests/`，写 `labels/<key>.json`。

```bash
python run.py fp-anchor1 --phase match-export --root $AQ
```

- **第二步：PENDULUM 语义匹配。** 判读者按 `$AQ/judge_match/INSTRUCTIONS.md`，只读 `queue.json` 和 `requests/`，写 `labels/<key>.json`。

```bash
python run.py fp-anchor1 --phase summary --root $AQ --label "Qwen3.8-27B"    # 生成 $AQ/summary_report.md
```

判读者与之前 PathVQA、PENDULUM 判读用的是同一个。不要让判读者读 `private_keymap.json`、`private_order.json`、records 或任何报告。

## 上传

两个目录里的以下内容，放到 `results/anchor1_pv_pd_20261108/{qwen38,gemma4}/`：
- `records/*.json`、`selection.json`、全部日志；
- `judge_extract/`、`judge_match/`；
- `read_anchor_report.md`、`summary_report.md`。

## 汇报

- pytest 最后一行；
- 生成是否全部完成，以及每个日志的最后 3 行；
- 两份 `summary_report.md` 全文。
