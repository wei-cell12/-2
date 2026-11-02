# Qwen3.8-27B · ClockQA：读数转写与只读归因

186条既有轨迹均已归因；不生成新回答。转写复用同一GPT-6 Luna max，逐条读取并校验连续引文。

已修复最终RMSNorm钩子递归、双卡索引设备不匹配，以及Qwen输出门向量重算的BF16误差；改为捕获原前向实际门值。数值门槛仍为层拆分2%、logit5%，没有降低。历史失败完整保留在repair_attempts及服务器attempts。

判读盲化修复：最初队列绝对路径暴露过模型家族名称，判读者尚未阅读正文或写标签；随后所有372条重新随机编号混排，交付目录只含说明、匿名请求和匿名队列。历史暴露与补救记录见stage_judge/judge_provenance.json，不能将此称为完全没有既往元数据暴露的全新判读者。

统计分析见stage_judge/report.md和attrib_report.md；归因是直接logit分解，不等同于因果干预证据。未上传NPZ/NPY。
