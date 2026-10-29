# 服务器任务：ClockQA 读提示阶段，题目 token 与用户断言 token 看图像的比较

交给服务器上的 agent。只把 186 条提示本身（62 题 × 中性 / 用户说错 / 用户说对）喂进模型前向一次，不带任何输出、不生成。GPU 只用几分钟。

**规则**
- 不要修改任何代码、参数、阈值或清单，也不要额外运行检查命令。
- 报错就停下，汇报错误信息和 `$S/failures/` 里的内容。不要自己打补丁，不要自动重试，不要自己清理 `failures/`。
- 停进程用 `kill -TERM`。不要停止或抢占别人的进程。
- 全部在标注用的那份仓库 `$L` 里做。

## 步骤

```bash
L=/mnt/sdb/linxin/repo-label
git -C $L pull
cd $L && python -m pytest tests/test_fp.py -q -rs   # 应全部通过，没有 skipped
CQ=/mnt/sdb/linxin/conflict-cq-20261028
S=/mnt/sdb/linxin/sources-cq-20261029             # 和上次回放同一个目录
C=/mnt/sdb/linxin/pathvqa-fp-a800-20261027/repo/configs/qwen38_fp_local.yaml   # 上次回放用的配置
```

如果 pytest 有 skipped 或失败，停下汇报。

### 1. 提示前向（GPU，几分钟）

用一张空闲的 A100，没有空闲的就等，不要抢占。

```bash
cd $L
CUDA_VISIBLE_DEVICES=<空闲的卡号> python run.py fp-sources --phase prompt --config $C --cq $CQ --root $S 2>&1 | tee $S/prompt.log
```

### 2. 分析（CPU）

```bash
cd $L
python run.py fp-sources --phase prompt-analyze --cq $CQ --root $S 2>&1 | tee $S/prompt_analyze.log
```

输出 `$S/prompt_report.md` 和 `$S/prompt_summary.json`。

## 上传

```bash
D=$L/results/sources_cq_20261029
cp $S/prompt_report.md $S/prompt_summary.json $S/prompt.log $S/prompt_analyze.log $D/
```

不要上传 `$S/prompt/*.npz`。在 `$L` 里提交并推送。

## 汇报

1. pytest 最后一行。
2. `prompt.log` 最后 5 行，是否 186 条全部完成。
3. `prompt_analyze.log` 全文。
4. `prompt_report.md` 全文。
