# 多模态谄媚与证据边界基准候选图片池

Tier 1.1，直接与证据矛盾。共 150 张，每域 50 张。日常逻辑 v2 已复核 50 张，完成其中 30 道中文题改写，配套 30 组正向与负向引导。

- [英文 Tier 1.1 基准，125 条、三条件双轨](benchmark_1_1.json)
- [严格 JSON Schema](benchmark_1_1_schema.json)
- [过滤清洗与分布报告](cleaning_report.md)
- [可复现构建与校验脚本](code/build_and_validate_1_1.py)
- [30 道日常逻辑改写题与逐题图片](daily_logic_v2/README.md)
- [改写题结构化数据](daily_logic_v2/questions.json)
- [90 条中性、正向、负向模型输入](daily_logic_v2/model_inputs.jsonl)
- [全部 50 张日常图片复核决定](daily_logic_v2/review_all_50.json)
- [统一来源索引](candidates_1_1.json)
- [筛选报告、来源与抽检提示](curation_report.md)
- [校验结果](validation.json)
- [图片目录](images/)
- [全体联系表](previews/)
- [25 张自绘图表底层数据与精确真值](synthetic_data/)
- [医学原始问答、框与掩码](medical_annotations/)
- [逐图 SHA-256 与来源修订](provenance/image_manifest.json)

已完成初筛。医疗标签沿用原数据集，临床专家复核待进行。

英文基准暂排除 25 张自绘图表。medical_003 的双侧框冲突已替换为单侧异常样本，历史图像和标注见 provenance/medical_replacements/。CT 异常器官子集的答案仍全部为 Liver。
