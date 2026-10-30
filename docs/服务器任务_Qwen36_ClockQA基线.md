# 服务器任务：Qwen3.6-35B-A3B 上的 ClockQA 基线（原模型三种条件、ITSM）

交给服务器上的 agent。代码已写好（`run.py fp-model2`，与 Gemma 4 那次同一套），按步骤运行。

**规则**
- 不要修改任何代码、参数、阈值或清单，也不要额外运行检查命令（下面列出的命令除外）。
- 报错或检查不通过就停下，汇报错误信息和 `failures/` 里的内容。不要自己打补丁，不要自动重试，不要自己清理 `failures/`。
- 停进程用 `kill -TERM`，不要用 Ctrl-C。不要停止或抢占别人的进程。
- 不要改动现有的 conda 环境。

## 做什么

模型：Qwen3.6-35B-A3B（bf16，MoE，与 Qwen3.8 同一家族），开思考模式。图像处理、解码参数、长度上限都与 Qwen3.8 那边相同。ClockQA 全部 62 题，种子 42。

| 方法 | 条件 | 次数 |
|---|---|---|
| 原模型 | 中性、用户说错、用户说对 | 186 |
| ITSM（中性提问由模型自己改写，两条分支同步生成，每一步慢约一倍） | 用户说错、用户说对 | 124 |

这次不跑 MM-SY。

## 0. 准备

### 0.1 环境和模型（只做一次）

```bash
MP=<Qwen3.6-35B-A3B-bf16 的模型目录>
conda activate <现在跑 Qwen3.8 的环境名>
python -c "import transformers; from transformers import Qwen3_5MoeForConditionalGeneration; print(transformers.__version__)"
cat $MP/generation_config.json
```

如果第二条报错（这个版本的 transformers 不认识这个模型），不要在现有环境里升级。新建一个环境：

```bash
conda create -n qwen36 --clone <现在跑 Qwen3.8 的环境名> -y
conda activate qwen36
pip install -U transformers accelerate
python -c "import transformers; from transformers import Qwen3_5MoeForConditionalGeneration; print(transformers.__version__)"
```

还报错就停下汇报。transformers 版本号和 `generation_config.json` 全文都放进汇报。

### 0.2 代码和数据

```bash
R=<这台服务器上放仓库的目录>/repo-qwen36
git clone https://github.com/wei-cell12/-2 $R      # 已有就 git -C $R pull
cd $R && python -m pytest tests/test_fp.py -q -rs   # 应全部通过，没有 skipped
Q=<这台服务器上的新目录>/qwen36-cq-20261031        # 这个模型的 ClockQA 目录（新建，不要和其他模型的混用）
QM=<这台服务器上的新目录>/qwen36-compare-20261031  # ITSM 等的结果目录（新建）
C=$R/configs/qwen36_35b_a3b.yaml
mkdir -p $QM
```

如果 pytest 有 skipped 或失败，停下汇报。

建 `$Q`（复制题目和 62 张图像）。ClockQA 原目录在 59 上（`/mnt/sdb/linxin/conflict-cq-20261028`）：

- 如果这台服务器就是 59：
  ```bash
  python run.py fp-model2 --phase setup --src-cq /mnt/sdb/linxin/conflict-cq-20261028 --cq $Q
  ```
- 如果不是：这台服务器上已经有 Gemma 那次建好的目录（`gemma4-cq-20261029`），直接用它作为来源：
  ```bash
  python run.py fp-model2 --phase setup --src-cq <gemma4-cq-20261029 的完整路径> --cq $Q
  ```

## 1. 改写中性提问（GPU，约 10 分钟）

ITSM 需要。让模型自己（不开思考、贪心）把用户消息改写成只问问题的形式，不看图。

```bash
cd $R
CUDA_VISIBLE_DEVICES=0,1 python run.py fp-model2 --phase neutralize --config $C --set model_path=$MP --cq $Q 2>&1 | tee $Q/neutralize.log
```

日志最后一行会给出：残留用户时间几条、残留"我认为"类说法几条、原问题保留几条。**只记录、不处理**，放进汇报。

## 2. 生成（GPU，两张卡一个进程，开两个进程）

bf16 权重约 70GB。A100 80GB 和 A6000 48GB 都用两张卡跑一个进程。

```bash
cd $R
CUDA_VISIBLE_DEVICES=0,1 python run.py fp-model2 --phase run --config $C --set model_path=$MP --cq $Q --root $QM --arms base itsm --seeds 42 --shard 0/2 2>&1 | tee $QM/run_0.log
CUDA_VISIBLE_DEVICES=2,3 python run.py fp-model2 --phase run --config $C --set model_path=$MP --cq $Q --root $QM --arms base itsm --seeds 42 --shard 1/2 2>&1 | tee $QM/run_1.log
```

两个进程同时在后台跑。只用空闲的卡，没有空闲就等。

- 原模型的结果写在 `$Q/obs/records/`，ITSM 的写在 `$QM/itsm/records/`。
- 每条日志会打印：图像 token 数、输出 token 数、秒数、**思考为空**、结果。第一条 ITSM 会打印"中性分支同步"的核对结果，不通过会自动报错停下。
- 代码会自动核对：图像 token 数是否与图像网格一致，缩放后的像素数是否在 65536～1048576 之间（不在说明图像设置没生效，会自动报错停下）。
- **两个进程最开始的各 5 条里，如果有任何一条"思考为空=True"，停下汇报**（说明思考模式没打开）。
- 第一条原模型结果出来后运行下面这一条，把输出放进汇报（看提示末尾是否以 `<think>` 开头的思考区结束、图像网格是多少）：
  ```bash
  python -c "import json,glob; f=sorted(glob.glob('$Q/obs/records/*.json'))[0]; r=json.load(open(f)); print(f); print(repr(r['prompt_text'][-150:])); print(r['image_grid']); print(repr(r['raw_output'][:300]))"
  ```
- 先跑原模型，再跑 ITSM。中途停下后，重新运行同一条命令会跳过已完成的。
- 每个进程跑出约 10 条原模型结果后，在汇报里给出每条大约多少秒，方便估计总时间。

## 3. 取出时间（判读子 agent，可以边跑边做）

与 Qwen3.8、Gemma 4 那两次完全相同的流程和判读者（GPT-6 Luna max），只是 `--cq` 和 `--root` 换成这次的目录：

```bash
cd $R
python run.py fp-compare --phase time-export --cq $Q --root $QM 2>&1 | tee -a $QM/time_export.log
# 按 $QM/time_judge/提取说明.md 逐行判读 $QM/time_judge/requests_new.jsonl，写到新的 judgments_raw_<批次号>.jsonl
python run.py fp-compare --phase time-import --cq $Q --root $QM 2>&1 | tee -a $QM/time_import.log
python run.py fp-compare --phase summary --cq $Q --root $QM --out $QM/compare_summary.json 2>&1 | tee $QM/summary.log
```

不要把正确答案、方法名或条件告诉判读者，也不要改说明。

## 上传

```bash
D=$R/results/qwen36_cq_20261031
mkdir -p $D/obs $D/neutralized
cp -r $Q/obs/records $D/obs/
cp $Q/neutralized/*.json $D/neutralized/
cp $Q/items.jsonl $Q/*.log $D/
cp -r $QM/itsm $QM/time_judge $D/
cp $QM/*.log $QM/compare_summary.json $D/
cp $C $D/config_used.yaml
```

不要上传图像。在 `$R` 里提交并推送；推送被拒就先 `git pull --rebase` 再推。

## 汇报

1. 第 0.1 步的 transformers 版本、用的是哪个环境、`generation_config.json` 全文。
2. pytest 最后一行。
3. `neutralize.log` 最后一行。
4. 每个进程最开始 5 条的日志行；第 2 步那条检查命令的输出；第一条 ITSM 的"中性分支同步"结果；每条原模型大约多少秒。
5. 生成是否全部完成。
6. `summary.log` 全文。
