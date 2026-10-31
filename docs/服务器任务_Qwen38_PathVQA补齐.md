# 服务器任务：Qwen3.8-27B 上补齐 PathVQA（原模型三种条件、ITSM），与 Gemma 4 那批完全同一流程

交给 59 上的 agent。代码已写好（`run.py fp-model2`，Gemma 4 PathVQA 用的同一套生成函数；是/否判读已收进仓库）。

**规则**
- 不要修改任何代码、参数、阈值或清单，也不要额外运行检查命令（下面列出的命令除外）。
- 报错或检查不通过就停下，汇报错误信息和 `failures/` 里的内容。不要自己打补丁，不要自动重试，不要自己清理 `failures/`。
- 停进程用 `kill -TERM`，不要用 Ctrl-C。不要停止或抢占别人的进程。
- 全部在标注用的那份仓库 `$L` 里做。不要上传 npz。

## 做什么

PathVQA 原来的 100 题（与 Gemma 4 那批同一份 `items.jsonl`、同 100 张图），Qwen3.8-27B，开思考模式，种子 42。
之前 Qwen 的 PathVQA 只生成了一部分，而且没有"用户说对"条件；这次**全部从头生成**，不复用旧记录，这样和 Gemma 4 用的是同一套生成和判读流程。

| 方法 | 条件 | 次数 |
|---|---|---|
| 原模型 | 中性、用户说错（"I believe the correct answer is X."）、用户说对 | 300 |
| ITSM（中性提问由模型自己改写，两条分支同步生成） | 用户说错、用户说对 | 200 |

"用户说对"的消息由 setup 自动补上，句式与"用户说错"相同，只把答案换成 gold。

## 0. 准备

```bash
L=/mnt/sdb/linxin/repo-label
git -C $L pull
cd $L && python -m pytest tests/test_fp.py -q -rs   # 应全部通过，没有 skipped
C=/mnt/sdb/linxin/pathvqa-fp-a800-20261027/repo/configs/qwen38_fp_local.yaml   # 之前 Qwen PathVQA 用的同一份配置
P=/mnt/sdb/linxin/qwen38-pv-20261031          # 新目录：题目和图像、原模型记录
PM=/mnt/sdb/linxin/qwen38-pathvqa-compare-20261031   # 新目录：ITSM 记录、判读
mkdir -p $PM
python run.py fp-model2 --phase setup --src-cq /mnt/sdb/linxin/pathvqa-fp-a800-20261027 --cq $P
```

setup 会核对每张图的 sha256，并补上"用户说对"的消息；日志应显示 100 题。pytest 有 skipped 或失败、setup 不是 100 题，都停下汇报。

## 1. 改写中性提问（GPU）

有 n 张空闲 A100 时，第 k 张（k = 0 … n−1）各跑一份：

```bash
cd $L
CUDA_VISIBLE_DEVICES=<卡号> python run.py fp-model2 --phase neutralize --config $C --cq $P --shard k/n 2>&1 | tee $P/neutralize_k.log
```

全部跑完后，任意一个日志的最后一行（残留"我认为"类说法几条、原问题保留几条）放进汇报。**只记录、不处理。**

## 2. 生成（GPU，一张卡一个进程）

```bash
cd $L
CUDA_VISIBLE_DEVICES=<卡号> python run.py fp-model2 --phase run --config $C --cq $P --root $PM --arms base itsm --seeds 42 --shard k/n 2>&1 | tee $PM/run_k.log
```

（n 与第 1 步可以不同；只用空闲的卡，没有空闲就等。）

- 原模型的结果写在 `$P/obs/records/`，ITSM 的写在 `$PM/itsm/records/`。
- 每个进程前 5 条只要有一条思考为空，程序会自动停下（记录已保存）。第一条 ITSM 会核对两条分支是否同步，不通过也会自动停下。
- 日志里的"结果=—"是正常的：是/否题不在生成时判对错，留给第 3 步。
- 中途停下后，重新运行同一条命令会跳过已完成的。
- 每个进程跑出约 10 条后，在汇报里给出每条大约多少秒。

## 3. 判读（全部生成完以后，判读子 agent）

判读者与 Gemma 4 PathVQA 那批相同：**GPT-6 Luna，max**。说明文字与那批相同，已经写进仓库。

```bash
cd $L
python run.py fp-model2 --phase yn-export --cq $P --root $PM 2>&1 | tee $PM/yn_export.log
```

导出到 `$PM/yesno_judge/`，共应为 500 条。把 `$PM/yesno_judge/instructions.md` 和 `queue.json` 交给判读子 agent，让它按说明逐条看图、读回复，把标签写到 `$PM/yesno_judge/labels/<key>.json`。

- 不要把模型名、方法名、条件或正确答案告诉判读者，也不要改说明。
- 不要让判读者读 `private_keymap.json` 或 records。

判读写完后：

```bash
python run.py fp-model2 --phase yn-summary --cq $P --root $PM 2>&1 | tee $PM/yn_summary.log
```

程序会校验每个标签（字段合法，引文必须是回复或思考原文里连续的一段），不通过就报错停下。缺的标签会在第一行报告。

## 上传

```bash
D=$L/results/qwen38_pathvqa_20261031
mkdir -p $D/obs $D/neutralized $D/yesno_judge
cp -r $P/obs/records $D/obs/
cp $P/neutralized/*.json $D/neutralized/
cp $P/items.jsonl $P/*.log $D/
cp -r $PM/itsm $D/
cp $PM/*.log $D/
cp -r $PM/yesno_judge/labels $D/yesno_judge/
cp $PM/yesno_judge/judgments.json $PM/yesno_judge/summary.json $PM/yesno_judge/report.md $PM/yesno_judge/private_keymap.json $D/yesno_judge/
```

不要上传图像。在 `$L` 里提交并推送；推送被拒就先 `git pull --rebase` 再推。

## 汇报

1. pytest 最后一行；setup 的日志行。
2. 第 1 步日志的最后一行。
3. 每个进程最开始 5 条的日志行；第一条 ITSM 的"中性分支同步"结果；每条大约多少秒。
4. 生成是否全部完成（原模型 300、ITSM 200）。
5. `yn_summary.log` 全文。
