# 服务器任务：PathVQA 补"先读后听（preread）"基线（Qwen3.8、Gemma 4），与已有的原模型 / ITSM / S2VA 同一流程

**不需要新代码**：用跑原模型、ITSM、S2VA 时同一套 `run.py fp-model2`，只把 `--arms` 换成 `preread`；是/否判读也是同一套（`yn-export` 已支持 preread）。

**做什么**：PathVQA 原来的 100 题（与已有基线同一份 items.jsonl、同 100 张图），用户说错、用户说对，种子 42，每个模型 200 次生成。中性提问用上次已改写好的（`$P/neutralized`、`$GP/neutralized`），不重新改写。

**MM-SY 不跑**：它在 ClockQA 上输出严重退化（Gemma 124 份里 121 份没有可判读的时间；Qwen 用户说对时答对从 58 降到 30），PathVQA 不再补。

## 规则

- 不要修改任何代码、参数、阈值或清单，也不要额外运行检查命令（下面列出的命令除外）。运行前确认 `git status` 干净。
- 用之前跑通的同一个 Python 环境（51 上是 `conda activate gemma4`），不要安装或升级任何库。
- 报错或检查不通过就停下，汇报错误信息和 `failures/` 里的内容；不要自己打补丁，不要自动重试，不要自己清理 `failures/`。
- 停进程用 `kill -TERM`，不要用 Ctrl-C；不要停止或抢占别人的进程。只用我指定的服务器和卡。
- 不要上传 npz 和图像。不要碰 50 上卡住的 GPU4、GPU6 和那个挂起的进程。
- 判读者与之前相同（GPT-6 Luna，max）；不要把模型名、方法名、条件或正确答案告诉判读者，不要让它读 `private_keymap.json` 或 records。

## A. 59：Qwen3.8-27B（一张卡一个进程）

```bash
L=/mnt/sdb/linxin/repo-label
git -C $L pull
cd $L && git status && python -m pytest tests/test_fp.py -q -rs   # 应全部通过，没有 skipped
C=/mnt/sdb/linxin/pathvqa-fp-a800-20261027/repo/configs/qwen38_fp_local.yaml
P=/mnt/sdb/linxin/qwen38-pv-20261031
PM=/mnt/sdb/linxin/qwen38-pathvqa-compare-20261031
# 生成：有 n 张空闲卡时，第 k 张（k = 0 … n-1）各跑一份
CUDA_VISIBLE_DEVICES=<卡号> python run.py fp-model2 --phase run --config $C --cq $P --root $PM --arms preread --seeds 42 --shard k/n 2>&1 | tee $PM/preread_run_k.log
# 全部生成完后，是/否判读（只导出 preread 的 200 条，导出到新目录）
python run.py fp-model2 --phase yn-export --cq $P --root $PM --yn-out $PM/yesno_judge_preread --yn-arms preread 2>&1 | tee $PM/preread_yn_export.log
#    把 $PM/yesno_judge_preread/instructions.md 和 queue.json 交给判读子 agent，标签写到 $PM/yesno_judge_preread/labels/<key>.json
python run.py fp-model2 --phase yn-summary --cq $P --root $PM --yn-out $PM/yesno_judge_preread --yn-extra $PM/yesno_judge/judgments.json $PM/yesno_judge_s2va/judgments.json 2>&1 | tee $PM/preread_yn_summary.log
```

上传：

```bash
D=$L/results/qwen38_pathvqa_20261031
mkdir -p $D/preread $D/yesno_judge_preread
cp -r $PM/preread/records $D/preread/
cp $PM/preread_*.log $D/
cp -r $PM/yesno_judge_preread/labels $D/yesno_judge_preread/
cp $PM/yesno_judge_preread/judgments.json $PM/yesno_judge_preread/summary.json $PM/yesno_judge_preread/report.md $PM/yesno_judge_preread/private_keymap.json $D/yesno_judge_preread/
```

## B. 51：Gemma 4 31B（两张卡一个进程）

```bash
R=/mnt/sdb/linxin/repo-gemma4
conda activate gemma4
git -C $R pull
cd $R && git status && python -m pytest tests/test_fp.py -q -rs   # 应全部通过，没有 skipped
C=$R/configs/gemma4_31b.yaml
GP=/mnt/sdb/linxin/gemma4-pv-20261030
GPM=/mnt/sdb/linxin/gemma4-pathvqa-compare-20261030
CUDA_VISIBLE_DEVICES=<两张空闲卡> python run.py fp-model2 --phase run --config $C --cq $GP --root $GPM --arms preread --seeds 42 --shard 0/2 2>&1 | tee $GPM/preread_run_0.log
CUDA_VISIBLE_DEVICES=<另两张空闲卡> python run.py fp-model2 --phase run --config $C --cq $GP --root $GPM --arms preread --seeds 42 --shard 1/2 2>&1 | tee $GPM/preread_run_1.log
# 全部生成完后
python run.py fp-model2 --phase yn-export --cq $GP --root $GPM --yn-out $GPM/yesno_judge_preread --yn-arms preread 2>&1 | tee $GPM/preread_yn_export.log
python run.py fp-model2 --phase yn-summary --cq $GP --root $GPM --yn-out $GPM/yesno_judge_preread --yn-extra $R/results/gemma4_pathvqa_20261030/judgments.json $GPM/yesno_judge_s2va/judgments.json 2>&1 | tee $GPM/preread_yn_summary.log
```

上传：

```bash
D=$R/results/gemma4_pathvqa_20261030
mkdir -p $D/preread $D/yesno_judge_preread
cp -r $GPM/preread/records $D/preread/
cp $GPM/preread_*.log $D/
cp -r $GPM/yesno_judge_preread/labels $D/yesno_judge_preread/
cp $GPM/yesno_judge_preread/judgments.json $GPM/yesno_judge_preread/summary.json $GPM/yesno_judge_preread/report.md $GPM/yesno_judge_preread/private_keymap.json $D/yesno_judge_preread/
```

在各自仓库里提交并推送；推送被拒就先 `git pull --rebase` 再推，推送成功后汇报 `git ls-remote origin main` 的输出。

## 汇报

- pytest 最后一行；
- 每个生成日志的最后 3 行，以及每条大约多少秒；
- 两份 `preread_yn_summary.log` 全文。
