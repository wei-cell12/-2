# 服务器任务：PathVQA 逐 token 注意力去向（10 例 + “用户说对”10 例，离线，不生成）

交给服务器上的 agent。代码已写好（`run.py fp-spans`），按步骤运行。**不要修改任何代码、参数、阈值**；报错或检查不通过就停下汇报错误信息和 `$W/failures/` 里的内容，不要自己打补丁、不要自动重试。

## 做什么

对已保存的 52 条轨迹：

- v2 10 例目录里的原始受压、原始中性、`dual_base`、`dual_mix`、`dual_shuffle`；
- “用户说对”10 例。

每条用生成时那份提示，把“提示 + 已生成内容”整段前向一次。模型不做任何修改，也不生成。

在 16 个标准注意力层上，逐个已生成 token 记录它的注意力分别落在五类内容上的份额：

1. 图像
2. 问题那一行
3. 用户主张那一行
4. 模型自己已经写的内容
5. 其余

用来看：从读图到准备谄媚的过程中，看图的量、看用户主张的量有没有转折。

## 自动检查（不通过会自动停下）

- 重新构造的提示与记录一致；问题行、主张行在提示里各恰好出现一次。
- 显式计算的注意力与原注意力函数的输出一致，标准与生成时的自检相同。
- 原始观察轨迹：重算出的逐 token 图像注意力要与生成时记录的 `pos_image_mass` 对得上，具体是：
  - 逐元素差的中位数 < 0.005；
  - 逐 token 相关系数 > 0.98；
  - 错开一个 token 时相关系数更低。

## 步骤

```bash
git pull                                       # 应包含 fp-spans
python -m pytest tests/test_fp.py -q           # 应全部通过（包括 test_spans_helpers、test_spans_requires_hf）
T=/mnt/sdb/linxin/pathvqa-dual10-20261027      # v2 10 例目录，只读
U=/mnt/sdb/linxin/pathvqa-agree10-20261027     # “用户说对”10 例目录，只读
W=/mnt/sdb/linxin/pathvqa-spans10-20261028     # 新目录
C=<A800 主实验用的配置文件>                     # 不改任何设置
mkdir -p $W
CUDA_VISIBLE_DEVICES=<卡号> python run.py fp-spans --config $C --roots $T $U --out $W/spans.json 2>&1 | tee $W/spans.log
```

两张卡可以各跑一半：

- 第一张卡：`--shard 0/2 --out $W/spans_0.json`
- 第二张卡：`--shard 1/2 --out $W/spans_1.json`

日志分别写成 `spans_0.log`、`spans_1.log`。

每条轨迹一次整段前向，预计每条几秒到十几秒，再加上模型加载。`$W/spans_heads/` 下会存逐（层, 头）的完整数组，共约 300 MB，**留在服务器，不上传**。

## 上传

```bash
mkdir -p <仓库>/results/pathvqa_spans10_20261028
cp $W/spans*.json $W/spans*.log <仓库>/results/pathvqa_spans10_20261028/
```

`spans*.json` 里没有 token 编号和数组文件，约 20 MB，可以直接上传。不要上传 `spans_heads/`。提交并推送。

## 汇报

1. 52 条是否全部通过检查；日志最后一行。
2. 原始观察轨迹（30 条）的“对照记录：中位差、r”的范围。
