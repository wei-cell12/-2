# 关键区域注意力恢复干预：准备检查失败，已停止

**这不是30题实验结果。2题×3组尚未开始，模型没有加载，未占用GPU。** 按任务“任何一步失败即停止”的要求，保存现场后停止，没有自动修复、重试或修改参数。监视仍暂停。

任务来源：`docs/服务器任务_关键区域注意力恢复干预.md`，固定 main 提交 `75bbd3fa0ebc3723815f231fd7a22b95a805a78d`。常规 Git fetch 网络失败后，通过 GitHub API 获取 main 的文件树；实际使用的21份运行源码/配置逐个验证 Git blob SHA 与该提交一致，证据见 `source_pull_receipt.json`。新目录中的 `symexp/attn.py` 另应用上次的 `mm_token_type_ids` 修复，补丁随附；旧源码、结果和标注未修改。

## 1. 正确率和配对比较

| 条件 | 计划生成 | 实际生成 | 正确数 / p值 |
|---|---:|---:|---|
| base（λ=0） | 30 | 0 | 未测量 |
| restore_E | 30 | 0 | 未测量 |
| restore_rand | 30 | 0 | 未测量 |

两组配对比较均未运行，无可报告的 p 值，不能把0条已生成写成0/30正确率。

## 2. 干预强度

Δ及区域份额前后变化：**未测量**。真实生成数值路径、逐token干预守恒及回复正常性均尚未检查；CPU合成张量性质测试通过不代表模型实验通过。

## 3. 推理内容

没有新生成的回复，因此没有本轮恢复/圆谎例子或机制结论。

## 4. 已完成检查、异常与现场

- 150题中，按仓库 `symexp.parsing.rule_answer` 对**原始记录的 final_answer**抽取得到69题中性yes/no正确、受压yes/no错误。按题号、旧pilot优先选30题，列表见 `items.json`。其中7题复用旧pilot；首2题是PV-000、PV-002。无法规则抽取的题保留在本地审计中，没有按格式判错。原有答案与解释联合判读作为单独字段保留；本次不冒称完成新Luna判读。
- 30题×2条件，共60份保存的输入/输出token与50服务器原清单逐一一致，图像SHA256一致，原注意力数组维度/dtype一致。检查见 `failure_receipt/preflight_data.json`。
- 原环境CPU性质测试通过：λ=0逐元素不变；λ=1只在选中头的图像位置修改；图像/文字/行总量守恒；非关键区域保留至少原份额5%；随机区域与E不交叠且可复现。见 `failure_receipt/cpu_preflight.log`。
- PV-000旧中性回放复用检查通过：16层×24头×330图像位置；prefill_end=1、think=697、answer=117，总行815；自检最大差0.03125；对原图像总注意力相关系数0.9998148703，平均绝对差0.00107528434。复用检查/构造至失败约0.057秒，不是新增模型前向耗时。
- **失败位置**：第2步静态参照的元数据保存，`references.py:37`。NumPy `heads.nonzero()`生成的`h`是`numpy.int64`，直接放进JSON；`run_experiment.py`保存时抛出 `TypeError: Object of type int64 is not JSON serializable`。完整回溯在 `failure_receipt/reference_pilot_cpu.log`。
- 修复位置已经定位：序列化头编号前需转为Python原生`int`（`head=int(h)`）。**本次没有应用该修复，也没有重新执行该步骤**。CPU测试覆盖参照构造及数值属性，但未覆盖元数据JSON落盘，这个代码缺陷因此在真实参照保存时暴露。
- PV-000的参照数组在报错前已保存；PV-002尚未构造。保留失败文件，不能把已存在的PV-000参照当成完整通过的元数据。
- 资源检查时GPU0～6均有他人计算任务，GPU7按用户要求排除；没有抢卡、停止他人任务或建立占卡进程。没有本实验GPU号、模型运行耗时、2题检查耗时或30题生成耗时可报告。

### 服务器路径与复用来源

服务器：`linxin@59.77.5.50:2222`。

- 新任务：`/mnt/sdg/linxin/pathvqa-attn-intervene-20261024/`。
- 失败前保存的参照：`references/PV-000.npz`，**1,236,000字节**。仅保存在服务器，没有上传NPZ。详见 `failure_receipt/array_inventory.json`。
- 旧中性数组只读复用：`/mnt/sdg/linxin/pathvqa-attn-region-20261024/attn_pilot10/`。
- 旧300份输入/输出清单：`/mnt/sdg/linxin/pathvqa-attn-region-20261024/manifest_plain.jsonl`。
- 原始选中记录：新目录的 `original_records.jsonl`，包含保存的token和原始回复，用于输入身份核对；没有上传该完整清单。
- 环境只读复用：`/mnt/sdg/linxin/pathvqa-attn-region-20261024/env/bin/python`，torch2.11.0+cu130、transformers5.14.1。
- 权重拟只读复用：`/mnt/ssd/gy/models/Qwen3.8-27B`，旧19项SHA256校验一致。此次未加载。

### 已写代码与尚未验证部分

`select_items.py`负责选题；`references.py`负责静态参照；`intervention.py`负责解码注意力修改；`run_experiment.py`负责2题门槛和30题分阶段执行；`evaluate_results.py`负责规则与精确McNemar评测；`numerical_tests.py`负责CPU性质测试。生成代码已经写好，但未经过真实模型的第5步验证。当前远端存在 `failure.json`，runner会拒绝自动重跑。不得将本目录当作已验证可直接跑30题的发布版本。

拟用设置保持原预实验：BF16、SDPA、无system、medium、seed42、思考4096/回答512、像素65536～1048576，36GiB GPU+128GiB CPU权重分配；只干预解码，prefill原样。三组均用相同显式注意力解码路径。这些是代码中的计划设置，不是已完成模型验证的结果。
