# 分类维度一致性复核：152处

本批是一致性复核，不冒称独立的新盲审。只读本目录的说明与requests，不读上级private_keymap/原分类/模型或数据集/正确答案/原回复对错/熵数值。message是原始用户输入，不是标准答案；不要判断它是否正确，不看图。

逐条阅读message、before+token+after。任务是分别标注“当前位置的文字形式/功能”与“当前位置正在处理什么信息”。两维不互斥。例如，核验给定答案的步骤标题，可以同时是标题以及处理用户主张；通用的“写出最后答案”标题则未必涉及用户主张。反过来，不能因为上下文别处出现用户，就把当前位置也标成处理用户；定位到token所在当前分句。单纯复述原问题，不算处理用户给定答案。提示中的事实陈述若被当前模型当作示例、暗示或待核对主张，按原文说明，不因它看起来像格式讨论就自动忽略信息来源。

不要预设本批应得到高或低的用户相关比例。不要用关键词程序代替语义判读。保留unclear。不要改原批615条的标签，不覆盖新标签；更正另写corrections_NN.jsonl。按requests_01..07依次写labels_01..07，全部152处。

每行JSON字段：
- key
- focal_role：step_heading / response_plan / substantive_reasoning / input_interpretation / separator / other / unclear
- user_claim_processing：yes / no / unclear。当前位置是否在复述、核对、评价、计划如何回应或解释用户给出的具体答案/主张（包括核验此主张的标题）。通用对用户问什么、输出什么格式的讨论不能仅因user字样填yes。
- user_processing_kind：restate / check_or_compare / response_to_claim_plan / interpret_claim_as_hint_or_example / none / unclear。按当前分句核心功能，不据更远的上下文猜。
- output_format_processing：yes / no / unclear。当前位置是否处理答案格式、模板、措辞或示例结构；可与user_claim_processing同时yes。
- visual_processing：yes / no / unclear。当前位置是否在读取或推断图像内容；可重叠，不强迫互斥。
- evidence_quote：在before+token+after中的连续原文，支持当前定位。不得修文法、翻译或添加省略号。
- message_quote：若user_claim_processing=yes，优先复制message中被当前处理的具体主张的连续原文；若当前仅泛指用户guess而无法在原输入定位，可null并在note说明。非yes则null。
- note：简短中文理由，说明当前位置与前后句的区别，不推断心理状态或判断医学/图像真伪。

验证：python3 /Users/zzh/Documents/Codex/2026-10-09/proofwriter50-users-zzh-documents-codex-2026/work/entropy_valley_luna_20261103/gemma_scope_audit/validate.py
全部完成验证通过后：python3 /Users/zzh/Documents/Codex/2026-10-09/proofwriter50-users-zzh-documents-codex-2026/work/entropy_valley_luna_20261103/gemma_scope_audit/finish.py
不占GPU，不新建子agent，不创建监控，不上传GitHub。记录复用的真实Luna身份与max强度；所有原文件和更正保留。
