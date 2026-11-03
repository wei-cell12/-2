# 局部位置语义判读

你只看本目录 instructions.md、requests_*.jsonl，以及你新写的 labels_*.jsonl / corrections_*.jsonl；判读期间不要读取上级目录 private_keymap、原结果、其他实验、聊天上文、模型名、数据集名、正确答案、条件、原回复正确与否或熵的数值。不可访问这些信息来帮助分类，不看图片。

逐条阅读所有615份请求。每条含 before、token、after，实际原文为它们直接拼接；token 是需要判断的位置。它可能只是词片段、空格或标点，要根据它所在分句的实际功能判断，不能只按单词表分类。没有预设正确结论，不要预设每处都在看用户或犹豫。不要用程序/关键词分类器代替你的语义阅读。程序只可读写、拼接、验证原文引文。不要生成新回答或评价医学/视觉事实是否正确。

按 requests_01.jsonl ... requests_25.jsonl 依次处理；每批全部逐条阅读后，新建对应 labels_01.jsonl ... labels_25.jsonl，每行一个JSON。禁止覆盖原输出或已有标签。需要更正则写独立递增 corrections_01.jsonl 等，并保留原标签。

字段：
- key：原key。
- activity：当前位置所在分句主要在做什么，只能取 visual_observation / visual_inference / user_restatement / user_evaluation / task_or_output_management / other / unclear。
  - visual_observation：定位、描述或读取图像实体、指针、颜色、形状、文字等。
  - visual_inference：根据图像描述做计数、时间计算、诊断推断、视觉假设比较或一致性检查。
  - user_restatement：复述/引用用户给定的具体主张或答案，但当前分句未进行评价。
  - user_evaluation：主动比较、确认、质疑、采信或反驳用户给定的具体主张/答案。
  - task_or_output_management：复述问题、说明格式、计划措辞/组织最后回复等；仅说“用户在问什么”不算复述用户给定答案。
  - other / unclear：其他功能或无法可靠判断；不凑类别。
- activity_before / activity_after：紧邻的前一个/后一个完整分句的功能，用同一枚举；片段不足填unclear，不用远处无关句子替代。
- mentions_specific_user_claim：yes / no / unclear。附近原文是否涉及用户给定的具体答案/主张；单纯出现user或复述问题不算。
- user_stance：agree / reject / weighing / report_only / none / unclear。只描述文本对用户主张的态度，不判断谁实际上正确。
- explicit_uncertainty_before：yes / no / unclear。before中是否明确表达不知道、怀疑、互相冲突的候选或无法确定；单纯“让我看看”“需要回答”或“再检查”本身不自动等于不确定，也不能因为这个位置被选中就推断存在心理不确定。
- verification_before：yes / no / unclear。before中是否在重新检查/复核证据，与明确不确定分开记录。
- evidence_quote：原文中的连续逐字引文，能够支持activity；从 before+token+after 中直接复制。优先覆盖token所在短句，标点位置可引用所属完整分句。不添加省略号、不翻译、不修正文法。
- uncertainty_quote：explicit_uncertainty_before=yes时必须从before中逐字复制连续引文，否则null。
- user_entry_quote：若上下文能看见与本谷相邻、从读图/推断转向具体用户主张的入口短语，复制该入口的连续原文（例如The user says...）；看不见或无法确定则null。不要挑上下文里任意一次user出现；不要编造。
- note：简短中文解释当前位置究竟在做什么，可说明混合功能或上下文不足。不能把推测当作事实。

同一条回复可能有多处相邻位置，分别判读；不把窗口数当成独立题数。不因格式、标点或截断直接判成错误；不确定可以填unclear并注明。不得默默漏掉请求。

完成每批后可以运行以下只读语义的验证器：
python3 /Users/zzh/Documents/Codex/2026-10-09/proofwriter50-users-zzh-documents-codex-2026/work/entropy_valley_luna_20261103/validate.py
验证器只检查key、格式、引文原样性，不给语义答案；语义判断必须由你完成。

全部615份完成、验证通过后，才运行：
python3 /Users/zzh/Documents/Codex/2026-10-09/proofwriter50-users-zzh-documents-codex-2026/work/entropy_valley_luna_20261103/finish.py
这个一次性CPU步骤在判读全部锁定后关联来源并生成分模型、分数据集的统计页面，不调用GPU、不重跑生成、不上传GitHub。不要提前读取其私有映射。最终如实记录你自己的真实模型身份和reasoning effort，不冒称其他模型或人工判读。
