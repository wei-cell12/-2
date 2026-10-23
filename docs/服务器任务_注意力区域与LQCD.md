# 服务器任务：注意力区域比较 + LQCD 复现

这份文件是交给服务器上的 agent 执行的任务说明。按顺序做，每一步的检查不通过就停下来汇报，不要自行换设置绕过。

## 背景

我们要观测一个现象：**加入用户的错误主张后，模型对图像里“本来看的位置”的注意力下降，对另一些位置的注意力上升。** 预实验只存了“图像整体”的注意力份额，没有存图像内部各个图像块的注意力，所以要把已保存的序列原样喂回模型，做一次只读前向计算重新取出来（不生成、不采样、不改动任何已有结果）。

另外复现 LQCD（中性问题 / 诱导问题两条分支的对比解码），在生成时同时记录两条分支的图像注意力，看它把答案纠正过来时，模型看图的位置有没有跟着恢复。

代码都在仓库里：

- `symexp/attn.py`：记录标准注意力层对每个图像 token 的注意力，按三个阶段（prefill 末尾、思考、回答）平均
- `symexp/attn_replay.py`：只读前向回放
- `symexp/attn_analysis.py`：区域比较、和预实验 `generation_masses` 的校验、热图
- `symexp/lqcd.py`：LQCD 两分支解码

## 总要求

1. 拉取仓库最新的 main。
2. 用预实验同一份权重、同一套 transformers / torch 环境。配置用 `configs/qwen38_hf.yaml`（不加 system、思考 4096 / 回答 512、seed 42、medium 思考强度、像素范围 65536～1048576），并且**设置 `attn_implementation: sdpa`**（不要用 eager 或 flash）。可以用 `--set attn_implementation=sdpa`，也可以复制一份配置文件改。
3. 只用一张空闲的 GPU。开始前确认没有别的实验在用这张卡；如果 SRT / 本方法的正式实验还在跑，排在它后面，**不要停掉或抢占任何已有进程**。
4. 不覆盖、不修改任何已有的运行目录和判读文件。新结果放到新目录。
5. **不要把 npz 数组推到 GitHub**（300 份约 1～2GB）。数组留在服务器上，GitHub 上只放报告、汇总、日志和热图，并在 README 里写清数组在服务器上的路径和总大小。
6. 任何一步失败：保留现场和日志，停下来汇报，不要自动重试或改参数重跑。

## 任务 A：plain 中性 vs 用户说错的图像注意力（150 题 × 2）

### A1. 整理序列清单

写一个转换脚本（放在结果目录里，和结果一起提交），从预实验保存的无干预 plain 记录生成清单 `manifest_plain.jsonl`，共 300 行：150 题中性 + 150 题用户说错。每行：

```json
{"id": "PV-000", "condition": "neutral", "image": "<原实验用的那张图的绝对路径>", "input_ids": [...], "output_ids": [...]}
```

- `condition` 只能是 `neutral` 或 `user_wrong`。
- `input_ids`：当初送进模型的完整提示 token（模板、图像占位 token、以 `<think>` 结尾的生成起点），**直接用保存的 token，不要从文字重新分词**。
- `output_ids`：当初生成的全部 token（思考、`</think>`、回答）。
- `image`：当初用的同一个图像文件，字节不变。

检查：每行 `len(output_ids)` 应等于保存的 `generation_masses` 的第一维（例如 PV-000 中性是 815，用户说错是 1355）。对不上就停下来汇报。

### A2. 先跑 2 题试一下

```bash
C=configs/qwen38_hf.yaml
python run.py attn-replay --config $C --set attn_implementation=sdpa \
  --manifest <结果目录>/manifest_plain.jsonl --out <服务器数组目录>/attn_plain --limit 2
```

开头会打印找到的记录目标，应该是 **16 个模块，layer_idx = [3, 7, 11, …, 63]**。不是的话停下来汇报。

看 `<服务器数组目录>/attn_plain/replay_log.jsonl`：

- 每行 `ok: true`。
- `selfcheck_max_abs_diff`：重算的注意力输出和原 sdpa 输出的最大差，bf16 下应该在 0.05 以内。明显更大就停下来汇报。
- `counts`：`prefill_end` 是 1；`think` 加 `answer` 加 1 等于 `len(output_ids)`。
- `n_img` 和原实验的图像 token 数一致。如果报“图像 token 数不一致”，说明像素范围或 processor 设置和原实验不同，停下来汇报。

### A3. 和预实验保存的数据校验

```bash
python run.py attn-validate --replay <服务器数组目录>/attn_plain \
  --saved "<保存 generation_masses 的路径模板，用 {id} 和 {condition} 占位>" \
  [--array <npz 里的数组名>] --class-index 0
```

这一步把我们重算的“每个生成位置、每层、每头对整个图像的注意力”和保存的 `generation_masses[..., 0]`（图像那一类）对比。`--class-index` 填图像类在最后一维的序号。`{condition}` 会被替换成 `neutral` 和 `user_wrong`；文件名如果用的是别的写法（比如 `pressure`），就先做一个符号链接或改模板。

通过标准（看打印的 summary）：

- `rows_mismatch` 为 0；
- `raw_corr_min` 或 `normalized_corr_min` ≥ 0.99，对应的 `mean_abs_diff_max` ≤ 0.01。

不通过就停下来汇报，附上 `validate.json`。常见原因：注意力实现不同、层的顺序不同、类别序号不对、模板或像素设置不同。

### A4. 全部 300 条

```bash
python run.py attn-replay --config $C --set attn_implementation=sdpa \
  --manifest <结果目录>/manifest_plain.jsonl --out <服务器数组目录>/attn_plain
python run.py attn-validate --replay <服务器数组目录>/attn_plain --saved "..." --class-index 0
```

中断后用同一条命令重跑会跳过已完成的。

### A5. 分析

```bash
python run.py attn-analyze \
  --ref plain_neutral=<服务器数组目录>/attn_plain@neutral \
  --cmp plain_wrong=<服务器数组目录>/attn_plain@user_wrong \
  --runs <复用目录>/runs/plain <复用目录>/runs/plain_pressure_reference \
  --heatmaps PV-000 PV-001 PV-003 PV-009 PV-013 <再加 5 题受压后改口的题> \
  --out <结果目录>/attn_plain_report
```

输出 `report.md`、`summary.json`、`items.csv`、`heatmaps/*.png`。

## 任务 B：LQCD（150 题，只跑用户说错）

设置：中性问题用规则拆分出的原问题（`lqcd_neutral=rule`）；α0 = 1、β0 = 0.1，不做自适应（`lqcd_lambda=0`、`lqcd_mu=0`）。论文没给这些值，这是对比解码的常用设置。采样设置和 plain 相同。每个 token 要跑两条分支，耗时约为 plain 的 2 倍。

### B1. 先跑 2 题

```bash
D=<复用目录>/data/conditions.jsonl
python run.py run --method lqcd --config $C --data $D --out <复用目录>/runs/lqcd \
  --conditions user_wrong --limit 2 \
  --set attn_implementation=sdpa lqcd_record_attention=true
```

检查：

- `runs/lqcd/final.jsonl`：`reply` 不为空，`lqcd_forced_close` 为 false（或说明原因），`lqcd_alpha_mean` 约为 1，`attn_files` 有 `lead` 和 `neutral` 两个文件。
- `runs/lqcd/calls.jsonl`：`neutral_text` 是不带压力话术的原问题。
- 回复内容正常，不是乱码或重复。

### B2. 全部 150 题

去掉 `--limit 2`，其余不变。

### B3. 判读与评测

```bash
python run.py judge-export --runs <复用目录>/runs/lqcd --out <复用目录>/judge/lqcd_requests.jsonl
# 按 judge/lqcd_requests_判读说明.md 交给 Luna 判读，得到 judgments，再导回：
python run.py judge-import --runs <复用目录>/runs/lqcd --judgments <判读结果>
python run.py eval --runs <复用目录>/runs/plain <复用目录>/runs/plain_pressure_reference <复用目录>/runs/lqcd \
  --out <结果目录>/lqcd_eval
```

Luna 暂时不可用就先跳过判读，评测会用规则抽取并在报告里注明。

### B4. LQCD 的注意力分析

```bash
python run.py attn-analyze \
  --ref plain_neutral=<服务器数组目录>/attn_plain@neutral \
  --cmp plain_wrong=<服务器数组目录>/attn_plain@user_wrong \
        lqcd_lead=<复用目录>/runs/lqcd/attn@user_wrong__*__lead \
        lqcd_neutral=<复用目录>/runs/lqcd/attn@user_wrong__*__neutral \
  --runs <复用目录>/runs/plain <复用目录>/runs/plain_pressure_reference <复用目录>/runs/lqcd \
  --label-method lqcd \
  --heatmaps <同 A5 的题号> \
  --out <结果目录>/attn_lqcd_report
```

- `lqcd_lead`：LQCD 里带诱导问题的那条分支。核心问题是：答案被纠正时，这条分支看图的位置是否仍然偏离中性。
- `lqcd_neutral`：LQCD 里中性问题那条分支，用来估计“生成内容不同”本身带来的变化。
- `--label-method lqcd` 用 LQCD 自己的结果标出哪些题仍然改口；判读还没做时会用规则抽取。

## 汇报与上传

在 GitHub 上新建 `results/attn_lqcd_<日期>/`，放：

- 转换脚本、`manifest_plain.jsonl` 的前 3 行样例（不放全文件）
- `replay_config.json`、`replay_log.jsonl`、`validate.json`
- `attn_plain_report/`、`attn_lqcd_report/`（report、summary、items.csv、heatmaps）
- `runs/lqcd` 的 `config.json`、`final.jsonl`、`calls.jsonl`（不含 attn 目录）、`lqcd_eval/`
- 一个 README：数组在服务器上的路径和总大小、GPU 型号、各步耗时、遇到的问题

汇报时直接给出：

1. A3 的校验结果（相关系数、平均绝对差、行数是否一致）。
2. `attn_plain_report/report.md` 里思考阶段的那一行：dE、dU、留存率和 p 值，以及 AUC 表。
3. LQCD 的谄媚率和中性准确率，以及 `lqcd_lead` 相对中性的 dE、dU。
4. 任何异常和你没按说明做的地方。
