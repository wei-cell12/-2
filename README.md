# 抗谄媚实验 v1：SRT 提示版 vs 本方法

实现《抗谄媚实验方案 v1》：在 PathVQA 是非题上，用 Qwen3.8-27B 比较

- **plain**：原始消息直接回答。只跑中性条件，同时作为谄媚率、更正率的参照，以及“忽略用户”baseline。
- **SRT 提示版**：SRT 的三阶段格式（图像文字化、反思、总结），只用提示词、不训练。
- **本方法**：对称推理（用户不可见）→ 否定理由核查（只在分歧时）→ 思考前缀接续写出完整回复。

所有提示词集中在 `symexp/prompts.py`，与方案文件一致。

## 目录

```
run.py                 命令行入口
configs/               qwen38_hf.yaml（transformers）、qwen38_vllm.yaml（vLLM 服务）、mock.yaml
symexp/
  prompts.py           全部提示词与压力话术
  data.py              数据准备与条件构造
  backends.py          HF / vLLM / mock 三种后端
  methods.py           plain、SRT、本方法（第 1–4 步）
  parsing.py           结构化区块与答案解析
  crops.py             3×3 网格裁切
  runner.py            批量运行与续跑
  judge.py             导出给 Luna 判读、导回判读
  evaluate.py          指标、配对检验、bootstrap、阶段诊断、成本
tests/                 单元测试与 mock 端到端测试
```

## 安装

```bash
pip install -r requirements.txt
```

transformers、torch 请用与预实验相同的版本。

## 数据

`mahzzz/Sycophancy-Dataset`（预实验用的 revision `e8c3b29b…`）一共只有 462 行、4 个子数据集，PathVQA 部分约 150 题，**凑不到方案里的 300 题**。两种做法：

```bash
# 做法一：用预实验的数据集，取 PathVQA 部分的全部是非题（与预实验同源）
python run.py prepare --source sycophancy --out data/syco

# 做法二：从原始 PathVQA（flaviagiammarino/path-vqa）的 test split 按答案分层抽 300 题
python run.py prepare --source pathvqa --n 300 --out data/pvqa
```

两种都会生成 `base.jsonl`（每题一行）、`images/`、`conditions.jsonl`。

**压力方式的分配**（默认 `--ptype-mode assign`）：每题只分配一种压力方式，按标准答案分层后轮流分配，150 题时 5 种方式各 30 题、每种 yes/no 约各一半。每题 3 个条件：中性、用户说错、用户说对，后两者用同一种压力方式。150 题共 450 条。分配结果会打印出来，也记在每行的 `ptype` 字段里；`--ptype-seed` 改分配用的随机种子。

要每题 5 种压力全跑（每题 11 个条件），加 `--ptype-mode all`。
`prepare --source sycophancy` 用 `dataset` 字段里包含 `path` 的行，名称不同时用 `--subset-filter` 改。
更正率的分母是 plain 中性答错的题，少于 30 题时结果不可靠。

## 运行顺序

先改 `configs/qwen38_hf.yaml` 里的 `model_path`、`crop_target_long_side`，以及思考强度参数（见下文）。

```bash
D=data/syco/conditions.jsonl
C=configs/qwen38_hf.yaml

# 0. 先挑 5 题跑通，逐项看方案里的检查清单（输出在 runs/*/calls.jsonl）
python run.py run --method ours  --config $C --data $D --out runs/try_ours --limit 5
python run.py run --method srt   --config $C --data $D --out runs/try_srt  --limit 5

# 1. 正式跑（1 个 seed）
python run.py run --method plain --config $C --data $D --out runs/plain --conditions neutral
python run.py run --method srt   --config $C --data $D --out runs/srt
python run.py run --method ours  --config $C --data $D --out runs/ours

# 2. 导出给 Luna 判读，判完再导回
python run.py judge-export --runs runs/plain runs/srt runs/ours --out judge/requests.jsonl
#    …… 用 Luna 判读，产出 judge/judgments.jsonl ……
python run.py judge-import --runs runs/plain runs/srt runs/ours --judgments judge/judgments.jsonl

# 3. 评测
python run.py eval --runs runs/plain runs/srt runs/ours --out reports/v1
```

中断后用同一条命令重跑即可续跑，已完成的单元会跳过。临时改配置用 `--set`，例如 `--set workers=8`。

### Luna 判读的格式

`judge/requests.jsonl` 每行有 `key`、`question`、`message`（用户实际发的消息）、`claim`、`gold`、`image`（绝对路径）、`reply`。判读者看不到本方法内部的结论。

`judge/judgments.jsonl` 每行至少要有：

```json
{"key": "<原样带回>", "semantic_answer": "yes | no | undecidable | null", "addresses_user": true}
```

`addresses_user` 是“回复有没有正面回应用户的说法”，可选。还没导回判读时，评测会用规则抽取答案，报告里会注明。

## 本方法怎么跑

| 步骤 | 调用 | 共用方式 |
|---|---|---|
| 1 拆分 | 规则，不调用模型 | — |
| 2 对称推理 | 开思考；用户不可见 | 每题每个 seed 一次，该题所有条件共用 |
| 3a 否定理由核查 | 不开思考、贪心 | 每题每个 seed 每个主张一次（`all` 模式下 5 种压力共用） |
| 3c 重新判断 | 开思考（2048） | 同上，只在理由被推翻或全部不确定时 |
| 4 最终回复 | 预填思考前缀并关闭思考，只生成回复 | 每种条件一次 |

第 3 步只在主张和第 2 步结论不同时触发。结论是 yes 或 no 的题最多触发一次；结论是 undecidable 的题，yes、no 两个主张会各触发一次。

输出文件（`runs/ours/`）：`step2.jsonl`、`step3.jsonl`、`final.jsonl`，以及记录每一次调用完整输入输出、token 数、耗时的 `calls.jsonl`。

## 要按你的环境确认的地方

1. **思考强度**：预实验用的是 medium 思考强度。如果它是聊天模板参数，加进配置的 `thinking_template_kwargs`；如果是别的机制，在 `HFBackend._template` 里改。
2. **聊天模板与图像占位**：`HFBackend` 用 `{"type": "image"}` 占位，再把 PIL 图像交给 processor。你的 processor 如果需要别的写法（例如 `qwen_vl_utils`），改 `HFBackend._template` 和 `_gen`。
3. **思考预算**：HF 后端分两段生成：先生成思考，遇到 `</think>` 或达到预算就停；再补上 `</think>` 生成回答。第二段会重新预填，和预实验的做法相同，但比一次生成慢。
4. **裁切放大**：`crop_target_long_side` 请设成模型实际输入的最大边长。原图分辨率不比模型输入高时，放大不会增加细节。
5. **想直接用预实验的推理代码**：写一个 `symexp.backends.Backend` 的子类，实现 `generate()` 的三种模式（think / nothink / prefill），在 `make_backend` 里注册，其余代码不用改。

## 测试

```bash
python -m pytest -q
```

`tests/test_smoke.py` 用 mock 后端在合成数据上跑完整流程（三种方法、判读导出导入、评测），并检查第 2、3 步的共用和续跑。mock 的数字没有意义。
