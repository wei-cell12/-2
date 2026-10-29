# 服务器任务：Gemma 4 31B-it 上的 ClockQA 基线（原模型三种条件、MM-SY、ITSM）

交给 A6000 服务器上的 agent。代码已写好（`run.py fp-model2`），按步骤运行。

**规则**
- 不要修改任何代码、参数、阈值或清单，也不要额外运行检查命令（下面列出的命令除外）。
- 报错或检查不通过就停下，汇报错误信息和 `failures/` 里的内容。不要自己打补丁，不要自动重试，不要自己清理 `failures/`。
- 停进程用 `kill -TERM`，不要用 Ctrl-C。不要停止或抢占别人的进程。
- 不要改动现有的 conda 环境（Gemma 4 需要新版 transformers，单独建一个环境）。

## 做什么

模型：Gemma 4 31B-it（`/mnt/sdb/ljh/models/gemma-4-31B-it`），开思考模式，每张图最多 560 个图像 token。ClockQA 全部 62 题，种子 42。

| 方法 | 条件 | 次数 |
|---|---|---|
| 原模型 | 中性、用户说错、用户说对 | 186 |
| MM-SY（后一半层、只在生成阶段，图像注意力分数 e → e + 0.9·\|e\|） | 用户说错、用户说对 | 124 |
| ITSM（中性提问由模型自己改写，两条分支同步生成，每一步慢约一倍） | 用户说错、用户说对 | 124 |

## 0. 准备

### 0.1 环境（只做一次）

```bash
conda create -n gemma4 --clone <现在跑实验用的环境名> -y
conda activate gemma4
pip install -U transformers accelerate
python -c "import transformers; from transformers import Gemma4ForConditionalGeneration; print(transformers.__version__)"
cat /mnt/sdb/ljh/models/gemma-4-31B-it/generation_config.json
```

最后两条的输出都要放进汇报。第三条报错就停下汇报。

### 0.2 代码和数据

```bash
R=<这台服务器上放仓库的目录>/repo-gemma4
git clone https://github.com/wei-cell12/-2 $R      # 已有就 git -C $R pull
cd $R && python -m pytest tests/test_fp.py -q -rs   # 应全部通过，没有 skipped
G=<这台服务器上的新目录>/gemma4-cq-20261029        # 这个模型的 ClockQA 目录（新建，不要和 Qwen 的混用）
GM=<这台服务器上的新目录>/gemma4-compare-20261029  # 其他方法的结果目录（新建）
C=$R/configs/gemma4_31b.yaml
```

如果 pytest 有 skipped 或失败，停下汇报。

建 `$G`（复制题目和 62 张图像）。ClockQA 原目录在 59 上（`/mnt/sdb/linxin/conflict-cq-20261028`）：

- 如果这台服务器就是 59：
  ```bash
  python run.py fp-model2 --phase setup --src-cq /mnt/sdb/linxin/conflict-cq-20261028 --cq $G
  ```
- 如果不是：在 59 上用同一条命令把 `--cq` 换成 59 上的一个新目录，然后把整个目录（含 `items.jsonl` 和 `images/`）原样拷到这台服务器的 `$G`。拷过来以后 `items.jsonl` 里的图像路径还指向 59 上的位置，所以要在这台服务器上用拷过来的目录作为 `--src-cq` 再跑一次 setup，生成最终的 `$G`：
  ```bash
  python run.py fp-model2 --phase setup --src-cq <拷过来的目录> --cq $G
  ```

## 1. 改写中性提问（GPU，两张卡，约 10 分钟）

ITSM 需要。让 Gemma 自己（不开思考、贪心）把用户消息改写成只问问题的形式，不看图。

```bash
cd $R
CUDA_VISIBLE_DEVICES=0,1 python run.py fp-model2 --phase neutralize --config $C --cq $G 2>&1 | tee $G/neutralize.log
```

日志最后一行会给出：残留用户时间几条、残留“我认为”类说法几条、原问题保留几条。**只记录、不处理**，放进汇报。

## 2. 生成（GPU，两张卡一个进程，开两个进程）

```bash
cd $R
CUDA_VISIBLE_DEVICES=0,1 python run.py fp-model2 --phase run --config $C --cq $G --root $GM --arms base mmsy itsm --seeds 42 --shard 0/2 2>&1 | tee $GM/run_0.log
CUDA_VISIBLE_DEVICES=2,3 python run.py fp-model2 --phase run --config $C --cq $G --root $GM --arms base mmsy itsm --seeds 42 --shard 1/2 2>&1 | tee $GM/run_1.log
```

（`$GM` 要先 `mkdir -p`。两个进程同时在后台跑；第 5 张卡空着。）

- 原模型的结果写在 `$G/obs/records/`，MM-SY 和 ITSM 写在 `$GM/mmsy/records/`、`$GM/itsm/records/`。
- 每条日志会打印：token 数、秒数、**思考为空**、结果；MM-SY 会打印图像注意力改前改后；第一条 ITSM 会打印“中性分支同步”的核对结果（不通过会自动报错停下）。
- **两个进程最开始的各 5 条里，如果有任何一条“思考为空=True”，停下汇报**（说明思考模式没打开）。
- 先跑完原模型，再跑 MM-SY、ITSM。中途停下后，重新运行同一条命令会跳过已完成的。
- 第一批原模型结果出来后（每个进程约 10 条），在汇报里给出每条大约多少秒，方便估计总时间。

## 3. 取出时间（判读子 agent，可以边跑边做）

与 Qwen 那边完全相同的流程和判读者（GPT-6 Luna max，与之前两轮一致），只是 `--cq` 和 `--root` 换成这次的目录：

```bash
cd $R
python run.py fp-compare --phase time-export --cq $G --root $GM 2>&1 | tee -a $GM/time_export.log
# 按 $GM/time_judge/提取说明.md 逐行判读 $GM/time_judge/requests_new.jsonl，写到新的 judgments_raw_<批次号>.jsonl
python run.py fp-compare --phase time-import --cq $G --root $GM 2>&1 | tee -a $GM/time_import.log
python run.py fp-compare --phase summary --cq $G --root $GM --out $GM/compare_summary.json 2>&1 | tee $GM/summary.log
```

不要把正确答案、方法名或条件告诉判读者，也不要改说明。

## 上传

```bash
D=$R/results/gemma4_cq_20261029
mkdir -p $D/obs $D/neutralized
cp -r $G/obs/records $D/obs/
cp $G/neutralized/*.json $D/neutralized/
cp $G/items.jsonl $G/*.log $D/
cp -r $GM/mmsy $GM/itsm $GM/time_judge $D/
cp $GM/*.log $GM/compare_summary.json $D/
```

不要上传图像。在 `$R` 里提交并推送。

## 汇报

1. 第 0.1 步的 transformers 版本和 `generation_config.json` 全文。
2. pytest 最后一行。
3. `neutralize.log` 最后一行。
4. 每个进程最开始 5 条的日志行；第一条 ITSM 的“中性分支同步”结果；每条原模型大约多少秒。
5. 生成是否全部完成。
6. `summary.log` 全文。
