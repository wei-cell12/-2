# 服务器任务：PathVQA “用户说对”对照（10 例）+ 熵差分析

交给服务器上的 agent。代码已写好（`run.py fp-agree-setup`、`fp-agree`，以及已有的 `fp-export` / `fp-import` / `fp-entropy`），按步骤运行。**不要修改任何代码、参数、阈值或清单**；报错或检查不通过就停下汇报错误信息和 `failures/` 里的内容，不要自己打补丁、不要自动重试。

## 做什么

之前 10 例（`docs/pathvqa_dual10_cases.json`）都是“用户说错”。这次给同样 10 题、同一种压力话术，但把用户的主张换成**正确答案**（例如 `I believe the correct answer is yes.`），各生成一次，设置与 A800 观察阶段完全相同（不干预）。然后对这 10 条轨迹做与上次相同的熵差分析，用来测：没有冲突时，两种提示的分歧是否保持很低（即冲突信号的误报率）。

## 步骤

```bash
git pull                                       # 应包含 fp-agree
python -m pytest tests/test_fp.py -q           # 应全部通过（包括 test_agree_setup_and_mock_run）
R=/mnt/sdb/linxin/pathvqa-fp-a800-20261027     # A800 主实验目录，只读
U=/mnt/sdb/linxin/pathvqa-agree10-20261027     # 新目录
C=<A800 主实验用的配置文件>                     # 不改任何设置

# 1. 建目录（打印一条消息示例，确认主张是正确答案）
python run.py fp-agree-setup --src $R --dst $U --cases docs/pathvqa_dual10_cases.json

# 2. 生成 10 次（可用两张卡各跑一半：--shard 0/2、--shard 1/2）
CUDA_VISIBLE_DEVICES=<卡号> python run.py fp-agree --config $C --root $U 2>&1 | tee $U/agree.log

# 3. 判读：同一个 Gemini 3.8 Flash（medium）子 agent、同一份说明
python run.py fp-export --root $U --phase obs
#    逐行判读 $U/obs/judge/requests.jsonl，输出 {"key","semantic_answer","explanation_correct","note"}
python run.py fp-import --root $U --phase obs --judgments <判读结果.jsonl>

# 4. 熵差分析（与上次相同）
CUDA_VISIBLE_DEVICES=<卡号> python run.py fp-entropy --config $C --root $U --out $U/entropy.json 2>&1 | tee $U/entropy.log
```

不要运行 `fp-candidates`。停进程用 `kill -TERM`，不要用 Ctrl-C。

## 上传

```bash
python run.py fp-public --root $U --out <仓库>/results/pathvqa_agree10_20261027/
cp $U/agree.log $U/entropy.json $U/entropy.log <仓库>/results/pathvqa_agree10_20261027/
```

不要上传 NPZ。提交并推送。

## 汇报

1. 10 次生成的判读结果（模型最后答 yes / no）。
2. `fp-entropy` 是否 10 条都通过检查，每条的 `in_top20_own_prompt`。
