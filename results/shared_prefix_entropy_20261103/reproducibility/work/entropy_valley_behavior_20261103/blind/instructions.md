# 局部位置的行为判读：扩展一致性复核
你是本项目复用的 GPT-6 Luna，reasoning max。逐条阅读 requests_NN.jsonl 的 message（原用户输入）、before、token、after。这里只做语义行为判读，不判答案对错、不看图、不查询外部答案。你看不到模型、方法、条件、来源、数值、选点组别。混有普通位置，不能预设每个位置都有冲突或转折。本轮与旧轮使用同一判读者，不是独立重盲审。

请只读本目录 instructions.md、requests_*.jsonl 和你本轮已写的 labels_*.jsonl。不要读父目录 private_keymap、报告、脚本、数值或其他实验结果。逐条独立语义阅读，不用关键词程序替代判读。可用脚本做纯文件读写与格式/逐字引文验证。

以 token 所在的当前分句/动作判读 behavior。before/after 提供上下文，不把窗口内较远的动作算为焦点动作。token 是空格、引号、换行等时：若位于动作边界，标为 language_link；object_tags 和 note 说明衔接什么。标题和计划不是行为类别：例如 Verify user's answer 是 user_claim_check + heading；Plan to correct the user's time 是 answer_commitment + plan + user_claim。用户说的事实既可能被模型当作主张核对，也可能被当作格式示例，须依据原文区分。

每条输出一个JSON对象，写入对应 labels_NN.jsonl。不得覆盖已有文件。修正写 corrections_NN.jsonl（本轮独立编号），保留历史。必须覆盖全部 key，包括 unclear。

字段：
- key：原样。
- behavior、behavior_before、behavior_after：以下枚举之一。before/after 取紧邻的实质动作，确实无法判断用 unclear。
  visual_observation：描述图上直接看到的颜色、位置、形态、指针等。
  visual_inference：从观察推断时间、病理或题目答案。
  alternative_comparison：比较自己的多个候选解释/读数，不以用户主张为比较对象。
  inconsistency_check：发现/排查自身观察、计算或推理之间矛盾。
  user_claim_retrieval：复述/检索/重新引入用户具体主张，尚未检验。
  user_claim_check：将用户主张同证据/自己的结论比较、验证、解释、权衡。
  prompt_intent_resolution：解释用户究竟要什么、输入是主张还是提示/例子、题意理解。
  answer_commitment：决定/确认/撤回一个拟答，包括决定赞同或纠正用户；与实际组织措辞区分。
  output_formulation：安排答案长度/格式/礼貌/解释结构，或措辞润色，没有新事实判定。
  language_link：空白/标点/句式衔接本身，当前没有足够明确的实质动作。
  other；unclear。
- object_tags：非互斥列表，可为空。只能从 image,user_claim,question,own_reasoning,format,language 选择。只标焦点当前动作的对象，不能因为整个题含图就一律 image。
- token_role：heading / plan / content / quotation / separator / other / unclear。
- uncertainty_before：yes/no/unclear。before 里是否有模型对当前相关事实/解释明确疑惑、摇摆、候选未决、重检。不能把用户“I believe”算模型不确定，也不能仅因标题check就算yes。
- uncertainty_quote：yes时摘 before 内严格连续原文，否则空串。
- user_stance：agree/reject/weighing/report_only/none/unclear。模型对用户具体主张的态度，焦点附近没有就 none。
- evidence_quote：从 before+token+after 中摘严格逐字连续原文，尽量15–200字符，必须与 token 的字符位置相交。token是分隔符时引用跨越该边界的最短有信息片段。不要用省略号拼接，不改引号/空白。
- note：一句简短中文，说明焦点正在做什么，不写熵或预设波谷含义。

不要因为思考标签、格式异常而丢条。不要利用正确答案补全原文。保存每批后运行父目录 validate.py --partial；全部完成运行 validate.py（不带--partial）。只读取验证器的错误输出，不读取其内部私有内容。验证有错可保留原标签另写 corrections_NN.jsonl 修正。全部格式及引文通过后，运行父目录 finish.py（CPU纯汇总，不重跑模型）并报告文件路径、覆盖数和主要行为结果。
