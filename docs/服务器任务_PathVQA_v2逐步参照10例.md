# 服务器任务：PathVQA 锚点一 v2（逐步参照）10 例验证

交给服务器上的 agent。代码已写好（`run.py fp-subset-setup`、`fp-dual`，以及原有的 `fp-export` / `fp-import` / `fp-public`），按步骤运行。**不要修改任何代码、参数、阈值或清单**；任何一步报错或检查不通过，就停下汇报错误信息和 `failures/` 里的内容，不要自己打补丁、不要自动重试。

## 做什么

v1 用的是中性回答的“阶段平均图”做参照，每一步取最大值，结果把注意力压成一张固定、过度集中的图，在 PathVQA 上和随机扰动没有区别。v2 改成逐步参照：

- 同时运行两条分支：受压分支（带用户那句话，正常采样、输出被保存）和中性分支（不带那句话）。每一步把受压分支采样到的 token 同样喂给中性分支，两条分支的已生成内容完全相同。
- 每一步、16 个标准层、选中的 96 个看图头上，把受压分支的图像内部分布 q 换成 `(1 − λ)·q + λ·r_t`，其中 r_t 是中性分支同一步的分布；λ = 1。图像总量、文字部分不变；prefill、线性层、视觉编码器不动。
- 三组：
  - `dual_base`：λ = 0，只在预检里跑，必须与观察阶段逐 token 一致；
  - `dual_mix`：换成中性分支此刻的分布；
  - `dual_shuffle`：对照组，目标是 r_t 的图像位置随机打乱，每个头的改动量与 `dual_mix` 相同。

题目：A800 主实验里中性答对、受压答错的 10 次，10 张不同的图，清单在 `docs/pathvqa_dual10_cases.json`。

每生成一次约是原来的 2 倍时间（每步多一次中性分支的前向）。

## 第 0 步：准备

```bash
git pull
python -m pytest tests/test_fp.py -q          # 应全部通过（包括 test_mix_rules、test_dual_perm_and_mock_pipeline）
R=/mnt/sdb/linxin/pathvqa-fp-a800-20261027    # A800 主实验目录，只读
T=/mnt/sdb/linxin/pathvqa-dual10-20261027     # 新目录
C=<A800 主实验用的配置文件>                    # 不改任何设置
python run.py fp-subset-setup --src $R --dst $T --cases docs/pathvqa_dual10_cases.json
```

应打印：10 次待干预，复制了 20 份观察记录（10 题中性 + 10 次受压）。

## 第 1 步：预检（一个进程）

```bash
CUDA_VISIBLE_DEVICES=<卡号> python run.py fp-dual --config $C --root $T --phase pilot 2>&1 | tee $T/pilot.log
```

前 2 次 × 三组，共 6 次生成，写 `T/dual_gate.json`。生成过程中会自动检查：

- 中性分支 prefill 的图像注意力与观察阶段中性记录一致（最大差 < 1e-3）；
- 中性分支的位置编码正确：生成结束后整段重算一次，和逐步算出的结果比较；
- 守恒误差、显式注意力自检。

`dual_gate.json` 里 `ok` 必须为 true，其中 `dual_base` 的 `same_as_observation` 必须为 true。`ok` 为 true 后，完整阅读这 6 份回复的 `raw_output`（`T/int/dual_*/records/*.json`），确认通顺，再写 `T/dual_review.json`：

```json
{"ok": true, "dual_gate_sha256": "<dual_gate.json 的 SHA256>", "read": ["PV-000__conviction/dual_mix", "..."]}
```

## 第 2 步：全部 10 次

```bash
CUDA_VISIBLE_DEVICES=<卡号> nohup python run.py fp-dual --config $C --root $T --phase full --shard 0/2 > $T/full_w0.log 2>&1 &
CUDA_VISIBLE_DEVICES=<卡号> nohup python run.py fp-dual --config $C --root $T --phase full --shard 1/2 > $T/full_w1.log 2>&1 &
```

共 10 × 2 次（`dual_mix`、`dual_shuffle`），预检做过的自动跳过。停进程用 `kill -TERM`，不要用 Ctrl-C。

## 第 3 步：判读

```bash
python run.py fp-export --root $T --phase int
```

用 A800 主实验同一个判读者（Gemini 3.8 Flash，medium，子 agent）、同一份说明，逐行判读 `T/int/judge/requests.jsonl`，输出 `{"key", "semantic_answer", "explanation_correct", "note"}`，然后：

```bash
python run.py fp-import --root $T --phase int --judgments <判读结果.jsonl>
```

不要运行 `fp-candidates`。

## 第 4 步：上传

```bash
python run.py fp-public --root $T --out <仓库>/results/pathvqa_dual10_20261027/
cp $T/pilot.log $T/full_w0.log $T/full_w1.log <仓库>/results/pathvqa_dual10_20261027/
```

补一个 README：使用的 GPU、各步耗时、每次生成的秒数，以及任何异常和没按说明做的地方。提交并推送。不要上传 NPZ。

## 汇报

1. 预检 `ok`，以及 6 份的检查项（尤其是 `same_as_observation`、`prefill_vs_observation_maxabs`、`top1_miss`、`logp_diff_max`）。
2. 10 次逐条：`dual_mix` 和 `dual_shuffle` 各自的判读结果（对 / 采纳 / 其他错 / 未解）。
3. 每次的 `js_before → js_after`（受压分支与中性分支看图分布的差距，干预前后）和 `mean_dose`，两组各一列。
