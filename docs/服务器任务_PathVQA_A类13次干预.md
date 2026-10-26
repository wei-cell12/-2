# 服务器任务：PathVQA A 类 13 次观测的注意力干预

这份说明交给服务器上的 agent。代码已经写好（`run.py fp-subset-setup`、`fp-region`，以及原有的 `fp-intervene` 等），按步骤运行。检查不通过就停下汇报，不要自己改代码、参数或阈值绕过；出错不自动重试。

## 背景

主实验目录 `R = /mnt/sdg/linxin/pathvqa-fp-20261026` 的观察阶段已完成 46 份（PV-000～PV-007），目前暂停。人工读完思考后，26 次受压采纳里有 13 次属于 **A 类**：模型从第一句描述起就按用户的方向解释图像，没有先看对再改口。这一轮只对这 13 次做干预，检验：

1. A 类受压时，关键区域的注意力份额是否真的下降（只用 CPU，读已有 npz）；
2. 把关键区域注意力恢复后（`restore_E`），以及剂量相同的随机区域（`rand_matched`），回答会不会改变。

13 次观测和分组写在 `docs/pathvqa_typeA_cases.json`，里面另有 B 类 11 次、C 类 2 次、守住 7 次，只用于第 1 步的对比，不干预。

## 总要求

- **不改主实验目录 `R`**：`fp-subset-setup` 和 `fp-region` 只读 `R`；所有新文件写在新目录 `T`。主实验保持暂停，这一轮跑完、汇报后再由用户决定是否恢复。
- **GPU**：只用 (0,1) 和 (6,7) 两对卡，**GPU 4、5 不用**。启动前在这 4 张卡上各做一次小的 CUDA 测试（分配约 1GB、做一次矩阵乘、释放），失败就停下汇报。两个 worker 错开启动：第一个开始生成后再启动第二个。不停止、不抢占任何别人的进程，不建常驻占卡进程。
- 模型、设置、种子与主实验完全相同：用主实验那份配置 `$C`（只改过 `model_path` 的 `qwen38_fp` 副本），种子 42，λ = 1。
- 出错时代码在 `T/failures/` 留现场并退出，之后的命令会拒绝运行；停进程用 `kill -TERM`，不要用 Ctrl-C。
- NPZ 留在服务器；GitHub 上只放 `fp-public` 的输出和 `region_check.json`。

## 第 0 步：准备

```bash
git pull                                       # 最新 main
python -m pytest tests/test_fp.py -q           # 应全部通过（包括 test_fp_subset_and_region）
R=/mnt/sdg/linxin/pathvqa-fp-20261026
T=/mnt/sdg/linxin/pathvqa-fp-typeA-20261026    # 新目录，不能已存在 candidates.json
C=<主实验用的配置文件>
python run.py fp-subset-setup --src $R --dst $T --cases docs/pathvqa_typeA_cases.json
```

应打印：13 次待干预、复制了 18 份观察记录（5 题中性 + 13 次受压），预检用 `PV-000__statement`、`PV-000__belief`。

## 第 1 步：关键区域份额检查（只用 CPU）

```bash
python run.py fp-region --src $R --cases docs/pathvqa_typeA_cases.json --out $T/region_check.json
```

会打印每组（A / B / C / held）、每阶段（think / answer）的：中性和受压时关键区域份额的均值、份额差 dE 的均值、dE < 0 的次数、图像总注意力的变化。把这张表原样贴进汇报。

## 第 2 步：预检（单个 worker）

```bash
CUDA_VISIBLE_DEVICES=0,1 python run.py fp-intervene --config $C --root $T --phase pilot
```

跑 2 次观测 × 3 组（`restore_E`、`rand_matched`、`base`）共 6 次，写 `T/pilot_gate.json`：

- `ok` 必须为 true（自检比值 ≤ 1、守恒误差 < 1e-3、`restore_E` 关键区域份额上升、`rand_matched` 剂量差 < 1e-3、回复非空无乱码）；
- `deterministic`：`base` 是否与观察阶段逐 token 一致。这两次观察是凌晨用旧版控制器生成的，现在用的是去掉 CPU 同步的新版，所以这一项同时检验新旧代码是否一致。**如实记录，false 也不要处理**，代码会自动给全部 13 次加跑 `base`。

`ok` 为 true 后，完整阅读这 6 份回复的 `raw_output`（`T/int/*/records/*.json`），确认通顺，再写：

```json
{"ok": true, "pilot_gate_sha256": "<pilot_gate.json 的 SHA256>", "read": ["PV-000__statement/restore_E", "..."]}
```

保存为 `T/pilot_review.json`。

## 第 3 步：干预全部 13 次（两个 worker，错开启动）

```bash
CUDA_VISIBLE_DEVICES=0,1 nohup python run.py fp-intervene --config $C --root $T --phase full --shard 0/2 > $T/int_w0.log 2>&1 &
# 等 w0 开始生成后
CUDA_VISIBLE_DEVICES=6,7 nohup python run.py fp-intervene --config $C --root $T --phase full --shard 1/2 > $T/int_w1.log 2>&1 &
```

共 13 × 2 次（若 `deterministic` 为 false，再加 13 次 `base`），已完成的自动跳过。按每次约 2.5 分钟估计，两个 worker 约 20～40 分钟，加上模型加载。

## 第 4 步：判读（Gemini 3.8 Flash，medium，子 agent）

干预结果和对应的原始观察**一起判读**，用同一个判读者、同样的说明，判读者看不到条件和组别：

```bash
python run.py fp-export --root $T --phase obs    # 5 份中性 + 13 份原始受压
python run.py fp-export --root $T --phase int    # 26 份（或 39 份）干预
```

按 `T/obs/judge/判读说明.md`（两份说明相同）逐行判读 `requests.jsonl`，输出 `{"key", "semantic_answer", "explanation_correct", "note"}`，然后：

```bash
python run.py fp-import --root $T --phase obs --judgments <obs 判读结果.jsonl>
python run.py fp-import --root $T --phase int --judgments <int 判读结果.jsonl>
```

不要运行 `fp-candidates`（会覆盖手写的 candidates.json），也不需要 `fp-analyze`。

## 第 5 步：上传

```bash
python run.py fp-public --root $T --out <仓库>/results/pathvqa_fp_typeA_20261026/
cp $T/region_check.json <仓库>/results/pathvqa_fp_typeA_20261026/
```

再补一个 README：使用的 GPU、各步耗时、预检的 `ok` 和 `deterministic`、判读设置、任何异常和没按说明做的地方。提交并推送。

## 汇报

1. 第 1 步的表（A / B / C / held × think / answer）。
2. 预检 `ok`、`deterministic`。
3. 13 次逐条：原始受压、`restore_E`、`rand_matched` 各自的判读结果（对 / 采纳 / 其他错 / 未解）。
4. 两组的平均抬高量 `mean_delta` 是否相同，以及 `restore_E` 关键区域份额干预前后（`region_before → region_after`）。
