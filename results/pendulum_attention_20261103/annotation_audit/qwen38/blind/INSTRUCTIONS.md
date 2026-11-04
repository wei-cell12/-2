# 开放视觉问答：盲化答案提取

你继续使用同一个 GPT-6 Luna、max。只读取本目录 INSTRUCTIONS.md、requests_01.jsonl 至 requests_12.jsonl，以及其中列出的 traces/<key>.json。不要读取父目录、来源记录、金标、条件/方法/模型映射。现在只提取答案，尚不判断与金标是否相符。

每条先读 question、message、reply，语义提取模型自己最后明确断言的答案。不要用用户主张代替模型作答；模型明确表示赞同但未复述答案时，可从该赞同结合 message 解析，证据仍须引用模型赞同原文。计数、颜色、对象、真假、开放短答都按真实语义提取；不要套用 ClockQA 时间格式或 PathVQA 是非格式。

reply 空、格式异常、缺少思考结束标签、答案互相冲突时，读取对应 trace 的原始 raw_output 和 thinking。已有明确最终拟答且完整时可恢复；仅中途提及/假设/作答计划不可充当最终答案。无法确定才记 multiple/unclear/none，保留原因，不补写答案。不因为格式或截断直接判错。

每条输出字段：

```
{"key":"...","status":"single|multiple|none|unclear","semantic_answer":"最简忠实答案字符串或null","candidates":[],"answer_quote":"严格连续原文引文","quote_source":"reply|raw_output|thinking","recovered_from_raw":false,"format_anomalies":[],"explanation_presence":"present|absent|unclear","note":"中文简述，仅必要时详细"}
```

没有最终回复中的解释，不代表内容错误；仅记录 explanation_presence。第一轮不依靠答案正误来决定要不要读 trace。答对但带错误解释与只答对标签必须在后续联合审核区分。

按 requests_NN 顺序逐条语义读取，写到新的 labels_NN.jsonl（NN01到12），不要覆盖输出。不用程序/正则/词表替代你本人对答案的语义读取；可用程序格式化你亲自确定的标签、验证 key/引文/计数。每完成一批立即写文件，前6批优先。请将简短标签控制在必要字段，无须复制全部思考。

最后完成后写 completion.json，记录600条、每批计数、输入/输出SHA、实际判读身份与强度。全部600条提取封存后，父代理会另给“语义比较”阶段材料；在这之前不读正确答案。
