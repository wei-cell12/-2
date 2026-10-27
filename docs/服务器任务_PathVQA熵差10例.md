# 服务器任务：PathVQA 熵差分析（10 例，离线，不生成）

交给服务器上的 agent。代码已写好（`run.py fp-entropy`），按步骤运行。**不要修改任何代码、参数、阈值**；报错或检查不通过就停下汇报错误信息和 `$P/failures/` 里的内容，不要自己打补丁、不要自动重试。

## 做什么

对 v2 那 10 例目录 `T` 里已保存的 40 条轨迹（每题：原始受压、原始中性、`dual_mix`、`dual_shuffle`），把“提示 + 已生成内容”整段喂给模型，分别用**受压提示**和**中性提示**各做一次前向（不采样、不生成），逐 token 记录两边的预测熵 H^P、H^N、两者之差 ΔH、JS 散度和实际 token 的对数概率差。用来检验：用户那句话造成的额外犹豫（ΔH）出现在哪里、能否把有冲突和没冲突的情况分开。

自动检查：重新构造的提示与记录一致；原始观察轨迹的实际 token 在原提示下排前 20 名的比例 ≥ 0.98；熵、JS 散度在合法范围内。

## 步骤

```bash
git pull                                       # 应包含 fp-entropy
python -m pytest tests/test_fp.py -q           # 应全部通过（包括 test_entropy_requires_hf）
T=/mnt/sdb/linxin/pathvqa-dual10-20261027      # v2 10 例目录，只读
P=/mnt/sdb/linxin/pathvqa-entropy10-20261027   # 新目录
C=<A800 主实验用的配置文件>                     # 不改任何设置
mkdir -p $P
CUDA_VISIBLE_DEVICES=<卡号> python run.py fp-entropy --config $C --root $T --out $P/entropy.json 2>&1 | tee $P/entropy.log
```

每条轨迹两次整段前向，预计每条十几秒，加模型加载。

## 上传

```bash
mkdir -p <仓库>/results/pathvqa_entropy10_20261027
cp $P/entropy.json $P/entropy.log <仓库>/results/pathvqa_entropy10_20261027/
```

`entropy.json` 里没有 token 编号和数组文件，可以直接上传（约几 MB）。提交并推送。

## 汇报

1. 40 条是否全部通过检查；日志最后一行。
2. 每条的 `in_top20_own_prompt`。
