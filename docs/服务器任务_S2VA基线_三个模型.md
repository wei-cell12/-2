# 服务器任务：S2VA 基线（Qwen3.8、Gemma 4、Qwen3.6-35B-A3B；ClockQA 与 PathVQA）

代码已写好：`run.py fp-model2 --arms s2va`。

**S2VA**（arXiv 2609.00067）每条回答调用模型两次：
1. **盲证人**：只给图和问题，看不到用户的话，输出读图报告和置信度（JSON）。
2. **仲裁者**：给问题、证人报告和置信度、用户的原话，不给图，给出最终答案。

问题由模型自己从受压提示里改写出来（neutralize 的结果）。两次调用都开思考模式，解码设置与其他方法相同。

**规则**
- 不要修改任何代码、参数、阈值或清单，也不要额外运行检查命令（下面列出的命令除外）。
- 报错就停下，汇报错误信息和 `failures/` 里的内容。不要自己打补丁，不要自动重试，不要自己清理 `failures/`。
- 停进程用 `kill -TERM`，不要用 Ctrl-C。不要停止或抢占别人的进程。只用空闲的卡，没有空闲就等。
- 判读时不要把模型名、方法名、条件或正确答案告诉判读者，也不要改说明。不要让判读者读 `private_keymap.json` 或 records。

**生成量**：每题两种条件（用户说错、用户说对），每种条件 2 次调用。每条大约是原模型的 2 倍时间。

| 模型 · 数据集 | 条数 |
|---|---|
| Qwen3.8 · ClockQA | 124 |
| Qwen3.8 · PathVQA | 200 |
| Gemma 4 · ClockQA | 124 |
| Gemma 4 · PathVQA | 200 |
| Qwen3.6 · ClockQA | 124 |

**每条日志会打印**：证人和仲裁者各写了多少 token，证人报告的解析方式（json / regex / failed）和置信度。每个进程前 5 条里只要有一条思考为空，程序会自动停下。中途停下后，重新运行同一条命令会跳过已完成的。

## A. 59：Qwen3.8-27B

```bash
L=/mnt/sdb/linxin/repo-label
git -C $L pull
cd $L && python -m pytest tests/test_fp.py -q -rs   # 应全部通过，没有 skipped
C=/mnt/sdb/linxin/pathvqa-fp-a800-20261027/repo/configs/qwen38_fp_local.yaml
```

### A1. ClockQA

```bash
CQ=/mnt/sdb/linxin/conflict-cq-20261028
M=/mnt/sdb/linxin/compare-cq-20261029-a100
cd $L
# 1) 让模型自己改写出中性提问（这个目录之前没有做过；写到 $CQ/neutralized/，124 条，一张卡约 10 分钟）
CUDA_VISIBLE_DEVICES=<一张空闲卡> python run.py fp-model2 --phase neutralize --config $C --cq $CQ 2>&1 | tee $M/s2va_neutralize.log
# 2) 生成：有 n 张空闲 A100 时，第 k 张（k = 0 … n-1）各跑一份
CUDA_VISIBLE_DEVICES=<卡号> python run.py fp-model2 --phase run --config $C --cq $CQ --root $M --arms s2va --seeds 42 --shard k/n 2>&1 | tee $M/s2va_run_k.log
# 3) 全部生成完后，时间判读（只会导出还没判读的 s2va 回答；与之前同一个判读者 GPT-6 Luna max、同一份说明）
python run.py fp-compare --phase time-export --cq $CQ --root $M 2>&1 | tee -a $M/time_export.log
#    按 $M/time_judge/提取说明.md 逐行判读 $M/time_judge/requests_new.jsonl，写到新的 judgments_raw_<批次号>.jsonl
python run.py fp-compare --phase time-import --cq $CQ --root $M 2>&1 | tee -a $M/time_import.log
python run.py fp-compare --phase summary --cq $CQ --root $M --out $M/compare_summary_s2va.json 2>&1 | tee $M/summary_s2va.log
```

### A2. PathVQA

```bash
P=/mnt/sdb/linxin/qwen38-pv-20261031
PM=/mnt/sdb/linxin/qwen38-pathvqa-compare-20261031
cd $L
# 中性提问上次已改写好（$P/neutralized），直接生成：
CUDA_VISIBLE_DEVICES=<卡号> python run.py fp-model2 --phase run --config $C --cq $P --root $PM --arms s2va --seeds 42 --shard k/n 2>&1 | tee $PM/s2va_run_k.log
# 全部生成完后，是/否判读（只导出 s2va 的 200 条，导出到新目录）：
python run.py fp-model2 --phase yn-export --cq $P --root $PM --yn-out $PM/yesno_judge_s2va --yn-arms s2va 2>&1 | tee $PM/s2va_yn_export.log
#    把 $PM/yesno_judge_s2va/instructions.md 和 queue.json 交给判读子 agent（GPT-6 Luna max），标签写到 $PM/yesno_judge_s2va/labels/<key>.json
python run.py fp-model2 --phase yn-summary --cq $P --root $PM --yn-out $PM/yesno_judge_s2va --yn-extra $PM/yesno_judge/judgments.json 2>&1 | tee $PM/s2va_yn_summary.log
```

### 上传 A

```bash
D=$L/results/compare_cq_20261029/a100
mkdir -p $D/s2va $D/time_judge
cp -r $M/s2va/records $D/s2va/
cp $M/s2va_*.log $M/summary_s2va.log $M/compare_summary_s2va.json $D/
cp $M/time_judge/judgments.json $D/time_judge/
mkdir -p $D/neutralized && cp $CQ/neutralized/*.json $D/neutralized/
D=$L/results/qwen38_pathvqa_20261031
mkdir -p $D/s2va $D/yesno_judge_s2va
cp -r $PM/s2va/records $D/s2va/
cp $PM/s2va_*.log $D/
cp -r $PM/yesno_judge_s2va/labels $D/yesno_judge_s2va/
cp $PM/yesno_judge_s2va/judgments.json $PM/yesno_judge_s2va/summary.json $PM/yesno_judge_s2va/report.md $PM/yesno_judge_s2va/private_keymap.json $D/yesno_judge_s2va/
```

在 `$L` 里提交并推送；推送被拒就先 `git pull --rebase` 再推。

## B. 51：Gemma 4 31B（两张卡一个进程，可以开两个进程：`--shard 0/2`、`--shard 1/2`）

```bash
R=/mnt/sdb/linxin/repo-gemma4
conda activate gemma4
git -C $R pull
cd $R && python -m pytest tests/test_fp.py -q -rs   # 应全部通过，没有 skipped
C=$R/configs/gemma4_31b.yaml
```

### B1. ClockQA

```bash
G=/mnt/sdb/linxin/gemma4-cq-20261029
GM=/mnt/sdb/linxin/gemma4-compare-20261029
CUDA_VISIBLE_DEVICES=<两张空闲卡> python run.py fp-model2 --phase run --config $C --cq $G --root $GM --arms s2va --seeds 42 --shard 0/2 2>&1 | tee $GM/s2va_run_0.log
CUDA_VISIBLE_DEVICES=<另两张空闲卡> python run.py fp-model2 --phase run --config $C --cq $G --root $GM --arms s2va --seeds 42 --shard 1/2 2>&1 | tee $GM/s2va_run_1.log
# 全部生成完后，时间判读（同 A1 第 3 步，--cq $G --root $GM）：
python run.py fp-compare --phase time-export --cq $G --root $GM 2>&1 | tee -a $GM/time_export.log
python run.py fp-compare --phase time-import --cq $G --root $GM 2>&1 | tee -a $GM/time_import.log
python run.py fp-compare --phase summary --cq $G --root $GM --out $GM/compare_summary_s2va.json 2>&1 | tee $GM/summary_s2va.log
```

### B2. PathVQA

```bash
GP=/mnt/sdb/linxin/gemma4-pv-20261030
GPM=/mnt/sdb/linxin/gemma4-pathvqa-compare-20261030
CUDA_VISIBLE_DEVICES=<两张空闲卡> python run.py fp-model2 --phase run --config $C --cq $GP --root $GPM --arms s2va --seeds 42 --shard 0/2 2>&1 | tee $GPM/s2va_run_0.log
CUDA_VISIBLE_DEVICES=<另两张空闲卡> python run.py fp-model2 --phase run --config $C --cq $GP --root $GPM --arms s2va --seeds 42 --shard 1/2 2>&1 | tee $GPM/s2va_run_1.log
# 全部生成完后，是/否判读（同 A2）：
python run.py fp-model2 --phase yn-export --cq $GP --root $GPM --yn-out $GPM/yesno_judge_s2va --yn-arms s2va 2>&1 | tee $GPM/s2va_yn_export.log
python run.py fp-model2 --phase yn-summary --cq $GP --root $GPM --yn-out $GPM/yesno_judge_s2va --yn-extra $R/results/gemma4_pathvqa_20261030/judgments.json 2>&1 | tee $GPM/s2va_yn_summary.log
```

### 上传 B

```bash
D=$R/results/gemma4_cq_20261029
mkdir -p $D/s2va $D/time_judge
cp -r $GM/s2va/records $D/s2va/
cp $GM/s2va_*.log $GM/summary_s2va.log $GM/compare_summary_s2va.json $D/
cp $GM/time_judge/judgments.json $D/time_judge/
D=$R/results/gemma4_pathvqa_20261030
mkdir -p $D/s2va $D/yesno_judge_s2va
cp -r $GPM/s2va/records $D/s2va/
cp $GPM/s2va_*.log $D/
cp -r $GPM/yesno_judge_s2va/labels $D/yesno_judge_s2va/
cp $GPM/yesno_judge_s2va/judgments.json $GPM/yesno_judge_s2va/summary.json $GPM/yesno_judge_s2va/report.md $GPM/yesno_judge_s2va/private_keymap.json $D/yesno_judge_s2va/
```

不要上传图像。在 `$R` 里提交并推送。

## C. Qwen3.6-35B-A3B ClockQA（跑基线的那台；两张卡一个进程）

用《服务器任务_Qwen36_ClockQA基线》里的同一组 `$R`、`$Q`、`$QM`、`$MP`、`$C`。中性提问那次已改写好（`$Q/neutralized`）。

```bash
git -C $R pull
cd $R && python -m pytest tests/test_fp.py -q -rs   # 应全部通过，没有 skipped
CUDA_VISIBLE_DEVICES=<两张空闲卡> python run.py fp-model2 --phase run --config $C --set model_path=$MP --cq $Q --root $QM --arms s2va --seeds 42 --shard 0/2 2>&1 | tee $QM/s2va_run_0.log
CUDA_VISIBLE_DEVICES=<另两张空闲卡> python run.py fp-model2 --phase run --config $C --set model_path=$MP --cq $Q --root $QM --arms s2va --seeds 42 --shard 1/2 2>&1 | tee $QM/s2va_run_1.log
```

那次基线的时间判读全部做完以后，再做 s2va 的判读（同 A1 第 3 步，`--cq $Q --root $QM`），然后：

```bash
D=$R/results/qwen36_cq_20261031
mkdir -p $D/s2va $D/time_judge
cp -r $QM/s2va/records $D/s2va/
cp $QM/s2va_*.log $QM/summary_s2va.log $QM/compare_summary_s2va.json $D/
cp $QM/time_judge/judgments.json $D/time_judge/
```

在 `$R` 里提交并推送。

## 汇报

1. 每台服务器 pytest 的最后一行。
2. A1 第 1 步 neutralize 日志的最后一行。
3. 每个生成进程最开始 5 条的日志行，以及每条大约多少秒。
4. 每个生成日志里证人报告解析失败的条数：`grep -c "报告解析=failed" <日志>`。
5. 生成是否全部完成。
6. 每个 `summary_s2va.log` 和 `s2va_yn_summary.log` 的全文。
