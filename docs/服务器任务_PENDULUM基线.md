# 服务器任务：PENDULUM 补 ITSM、S2VA、先读后听（preread）三个基线（Qwen3.8、Gemma 4）

**做什么**：PENDULUM 那 200 题（与原模型那批完全相同：易混淆视角、视觉谜题、复用图像、自拍图像各 50 题；题目清单在仓库 `results/pendulum_attention_20261103/items.jsonl`），用户说错、用户说对，种子 42，三种方法，每个模型 1200 次生成。MM-SY 不跑（它在 ClockQA 上输出严重退化）。

- 生成用跑 ClockQA、PathVQA 基线时同一套 `run.py fp-model2`（同样的解码设置和长度上限）。
- 中性提问由模型自己改写（`--phase neutralize`），与其他数据集相同。
- 判读与原模型那 1200 份**同一个判读者（GPT-6 Luna，max）、同一份说明原文、同样两阶段**：先盲化抽取答案并封存，再给参考答案做语义比较。说明由程序原样复制，不要改。

## 规则

- 不要修改任何代码、参数、阈值或清单，也不要额外运行检查命令（下面列出的命令除外）。运行前确认 `git status` 干净。
- 用之前跑通的同一个 Python 环境（51 上是 `conda activate gemma4`），不要安装或升级任何库。
- 报错或检查不通过就停下，汇报错误信息和 `failures/` 里的内容；不要自己打补丁，不要自动重试，不要自己清理 `failures/`。
- 停进程用 `kill -TERM`，不要用 Ctrl-C；不要停止或抢占别人的进程。只用我指定的服务器和卡。
- 不要上传 npz 和图像。不要碰 50 上卡住的 GPU4、GPU6 和那个挂起的进程。
- 判读者只读判读目录里的 INSTRUCTIONS.md、requests_NN.jsonl 和 traces/；不要让它读 private_keymap.json、records 或正确答案；不要把模型名、方法名、条件告诉它。第一阶段全部写完、`open-compare` 封存之前，不给它任何参考答案。

## A. 59：Qwen3.8-27B（一张卡一个进程）

```bash
L=/mnt/sdb/linxin/repo-label
git -C $L pull
cd $L && git status && python -m pytest tests/test_fp.py -q -rs   # 应全部通过，没有 skipped
C=/mnt/sdb/linxin/pathvqa-fp-a800-20261027/repo/configs/qwen38_fp_local.yaml   # 与 ClockQA、PathVQA 基线同一份配置
P=/mnt/sdb/linxin/qwen38-pd-20261109            # 新目录：题目和图像
PM=/mnt/sdb/linxin/qwen38-pendulum-compare-20261109   # 新目录：三种方法的记录、判读
mkdir -p $PM
# 0. 建工作目录（核对每张图的 sha256；日志应显示 200 题）
python run.py fp-model2 --phase setup --src-cq $L/results/pendulum_attention_20261103 --cq $P 2>&1 | tee $PM/setup.log
# 1. 改写中性提问（有 n 张空闲卡时第 k 张跑一份）
CUDA_VISIBLE_DEVICES=<卡号> python run.py fp-model2 --phase neutralize --config $C --cq $P --shard k/n 2>&1 | tee $PM/neutralize_k.log
# 2. 生成（一张卡一个进程，有 n 张卡就分 n 份）
CUDA_VISIBLE_DEVICES=<卡号> python run.py fp-model2 --phase run --config $C --cq $P --root $PM --arms itsm s2va preread --seeds 42 --shard k/n 2>&1 | tee $PM/run_k.log
# 3. 全部生成完后：第一阶段导出（1200 条，24 批）
python run.py fp-model2 --phase open-export --cq $P --root $PM 2>&1 | tee $PM/open_export.log
#    判读子 agent 按 $PM/open_judge/INSTRUCTIONS.md 写 labels_NN.jsonl 和 completion.json
python run.py fp-model2 --phase open-compare --cq $P --root $PM 2>&1 | tee $PM/open_compare.log
#    判读子 agent 按 $PM/open_judge/comparison/INSTRUCTIONS.md 写 comparison/comparison_NN.jsonl 和 completion.json
python run.py fp-model2 --phase open-summary --cq $P --root $PM --base-judgments $L/results/pendulum_attention_20261103/qwen38_judgments.json --label "Qwen3.8-27B" 2>&1 | tee $PM/open_summary.log
```

## B. 51：Gemma 4 31B（两张卡一个进程）

```bash
R=/mnt/sdb/linxin/repo-gemma4
conda activate gemma4
git -C $R pull
cd $R && git status && python -m pytest tests/test_fp.py -q -rs   # 应全部通过，没有 skipped
C=$R/configs/gemma4_31b.yaml
GP=/mnt/sdb/linxin/gemma4-pd-20261109
GPM=/mnt/sdb/linxin/gemma4-pendulum-compare-20261109
mkdir -p $GPM
python run.py fp-model2 --phase setup --src-cq $R/results/pendulum_attention_20261103 --cq $GP 2>&1 | tee $GPM/setup.log
CUDA_VISIBLE_DEVICES=<两张空闲卡> python run.py fp-model2 --phase neutralize --config $C --cq $GP 2>&1 | tee $GPM/neutralize.log
CUDA_VISIBLE_DEVICES=<两张空闲卡> python run.py fp-model2 --phase run --config $C --cq $GP --root $GPM --arms itsm s2va preread --seeds 42 --shard k/n 2>&1 | tee $GPM/run_k.log
python run.py fp-model2 --phase open-export --cq $GP --root $GPM 2>&1 | tee $GPM/open_export.log
python run.py fp-model2 --phase open-compare --cq $GP --root $GPM 2>&1 | tee $GPM/open_compare.log
python run.py fp-model2 --phase open-summary --cq $GP --root $GPM --base-judgments $R/results/pendulum_attention_20261103/gemma4_judgments.json --label "Gemma 4 31B-it" 2>&1 | tee $GPM/open_summary.log
```

中途断了用同一条命令重跑，已完成的会自动跳过。

## 上传（每个模型）

放到 `results/pendulum_baselines_20261109/{qwen38,gemma4}/`：`$P/neutralized/`、`$PM/{itsm,s2va,preread}/records/`、`$PM/open_judge/` 里除 `traces/` 以外的全部文件（含 labels、comparison、SEALED.json、private_keymap.json、report.md、summary.json、judgments.json）、全部日志。不要上传图像。生成完成、判读之前可以先上传一次 records 和日志。推送成功后汇报 `git ls-remote origin main` 的输出。

## 汇报

- pytest 最后一行；setup 日志最后一行（应为 200 题）；
- 每个生成日志的最后 3 行，以及每条大约多少秒；
- 两份 `open_summary.log` 全文。
