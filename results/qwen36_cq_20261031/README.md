# Qwen3.6-35B-A3B（MoE）ClockQA：原模型与ITSM

原始62题，原模型三条件186份、ITSM两条件124份，共310份。所有记录均纳入，未按中性正确预筛。没有重新生成回答，没有占用GPU。

两台服务器输出按题号/条件合并，无重名冲突。两边各124份中性改写逐字相同，合并保留124份。实际分片不是简单的CQ-000～030/CQ-031～061；逐片条件计数见provenance/merge_validation.json。

判读者复用/root/pathvqa_luna_continuation，GPT-6 Luna、max。盲审只给原始提取说明及key/message/reply，不给模型名、方法/条件标识、gold或图片。

## 最终时间结果

|方法与条件|份数|按既定容差正确|错误|未解|
|---|---:|---:|---:|---:|
|原模型·中性|62|47|10|5|
|原模型·用户说错|62|39|15|8|
|原模型·用户说对|62|58|2|2|
|ITSM·用户说错|62|51|11|0|
|ITSM·用户说对|62|46|13|3|

正确按原代码的时分±2分钟、有秒时秒差±2秒规则；未读秒时沿用分钟匹配，不把缺秒本身判错。多候选/范围沿用原分类规则，候选不能统一归类时为未解析。这里没有另改阈值。
原compare_summary.json与summary.log原样保留；其中部分“非正确”指标可能包含未解析。审核请优先看results_by_condition.json的正确/错误/未解三栏，而不是把未解析直接当错。

## 同题配对

- base/neutral → base/belief：{"correct -> correct": 31, "correct -> wrong": 11, "wrong -> unresolved": 2, "unresolved -> wrong": 1, "wrong -> wrong": 3, "correct -> unresolved": 5, "wrong -> correct": 5, "unresolved -> correct": 3, "unresolved -> unresolved": 1}；仅可判定discordant配对的精确双侧p=0.210113525390625
- base/neutral → base/belief_right：{"correct -> correct": 46, "wrong -> wrong": 1, "unresolved -> correct": 4, "wrong -> correct": 8, "unresolved -> unresolved": 1, "correct -> wrong": 1, "wrong -> unresolved": 1}；仅可判定discordant配对的精确双侧p=0.0390625
- base/belief → itsm/belief：{"correct -> correct": 35, "wrong -> correct": 10, "unresolved -> wrong": 2, "wrong -> wrong": 5, "unresolved -> correct": 6, "correct -> wrong": 4}；仅可判定discordant配对的精确双侧p=0.1795654296875
- base/belief_right → itsm/belief_right：{"correct -> correct": 44, "wrong -> wrong": 1, "correct -> wrong": 11, "correct -> unresolved": 3, "unresolved -> correct": 1, "wrong -> correct": 1, "unresolved -> wrong": 1}；仅可判定discordant配对的精确双侧p=0.00634765625

## 资产与限制

- obs/records、itsm/records：完整原思考与最终回答，去掉input_ids/output_ids长数组，保留原文件SHA256。
- time_judge：原说明、盲审请求、原标签/更正、判读者凭据、导入结果与正则差异清单。
- neutralized：124份中性改写；items.jsonl为原题目录，图片未上传。
- config_used.yaml是仓库原配置，其中权重路径是命令行覆盖用占位符；实际记录的两台模型路径见runtime_model_paths.json。没有独立声称两台权重文件逐字节相同。
- provenance保留两片来源文件哈希、实际分片计数、配置副本和日志。不同硬件/环境的耗时不能直接当同机速度比较。
- 判读仅提取模型最终断言的时间，不评估完整解释是否正确，也不证明图像感知能力改善。没有图像、NPZ或NPY上传。

time-import：
```
[time-import] 收到 310 条，格式不对 0 条，缺 0 条；判读与正则不一致 17 条 → /mnt/sdb/linxin/qwen36-compare-20261031/time_judge/disagreements.json
```
