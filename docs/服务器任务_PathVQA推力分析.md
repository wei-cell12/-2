# 服务器任务：PathVQA 推力分析（第一轮，不生成）

交给服务器上的 agent。代码已写好（`run.py fp-push`），按步骤运行；检查不通过就停下汇报，不要改代码、参数或阈值绕过，出错不自动重试。

## 做什么

对主实验目录 `R = /mnt/sdg/linxin/pathvqa-fp-20261026` 里已保存的 23 份观察记录（10 次受压采纳、6 次守住、7 份中性），把“提示 + 原来的输出”整段喂给模型做**一次前向（不采样、不生成）**，在回答里决定答案的那个 Yes/No token 前一个位置上，把 16 个标准注意力层的输出按来源分组（关键区域 E、其余图像 I、用户那句话 U、其余提示 Q、思考 T、回答前文 R），算每组把 “Yes 减 No” 的分差推向哪边。清单在 `docs/pathvqa_push10_cases.json`。

每份都会自动检查：各组之和等于注意力原输出、门控重建等于 `o_proj` 输入、最终归一化线性化等于 `lm_head` 输入、两种方式算出的分差一致、各组推力之和等于 16 层注意力的直接推力。任何一项不通过都会报错停下。

## 要求

- **主实验目录 `R` 只读**：`fp-push` 只读 `R`；输出和出错现场都写在输出目录 `P`。
- **GPU**：一个模型实例、两张卡。不用 GPU 4、5。如果 A 类 13 次干预还在跑，等它跑完，或者用空闲的那一对卡；不抢占任何进程。启动前做一次小的 CUDA 测试。
- 配置用主实验那份 `$C`（只改过 `model_path` 的 `qwen38_fp` 副本），不改其他设置。
- 停进程用 `kill -TERM`，不要用 Ctrl-C。

## 步骤

```bash
git pull
python -m pytest tests/test_fp.py -q           # 应全部通过（包括 test_push_helpers）
R=/mnt/sdg/linxin/pathvqa-fp-20261026
P=/mnt/sdg/linxin/pathvqa-push-20261026        # 新目录
C=<主实验用的配置文件>
mkdir -p $P
CUDA_VISIBLE_DEVICES=<两张空闲卡> python run.py fp-push --config $C --root $R \
    --cases docs/pathvqa_push10_cases.json --out $P/push.json 2>&1 | tee $P/push.log
```

每份打印一行：目标 token、它的概率、分差 Δ、各组“推向正确答案”的推力（正数 = 推向正确答案）、是否被标记。最后打印按组（adopted_A / adopted_B / adopted_C / held / neutral）汇总的表。预计每份几秒到十几秒，加上模型加载约十几分钟。

## 上传

```bash
mkdir -p <仓库>/results/pathvqa_push_20261026
cp $P/push.json $P/push.log <仓库>/results/pathvqa_push_20261026/
```

`push.json` 里没有 token 列表和数组，可以直接上传。提交并推送。

## 汇报

1. 是否全部 23 份通过检查；如果有报错，贴出错误信息和 `$P/failures/` 里的内容。
2. 最后打印的汇总表原样贴出。
3. 每份的目标 token 和 `p_target`（模型给这个词的概率），以及 `delta` 与 `delta_from_16_attn_direct` 两个数。
