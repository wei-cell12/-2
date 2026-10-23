# PathVQA：小试完整数据

SRT与本方法原小试均为5题×3条件=15份回答。本目录提供完整思考、回答、输入prompt、独立推理、图像核查及5张原图；同时保留截至2026-10-23 20:21:54的11份末步修正版。

## 逐题完整原文

- [PV-000：题目、图像、三条件、思考与回答](cases/PV-000.md)
- [PV-001：题目、图像、三条件、思考与回答](cases/PV-001.md)
- [PV-003：题目、图像、三条件、思考与回答](cases/PV-003.md)
- [PV-009：题目、图像、三条件、思考与回答](cases/PV-009.md)
- [PV-013：题目、图像、三条件、思考与回答](cases/PV-013.md)

## 原始JSONL

- [SRT：15份原回答](raw/runs/try_srt/final.jsonl)；[完整调用、输入、思考](raw/runs/try_srt/calls.jsonl)
- [本方法：15份原回答](raw/runs/try_ours/final.jsonl)；[全部调用](raw/runs/try_ours/calls.jsonl)
- [独立推理step2](raw/runs/try_ours/step2.jsonl)；[图像核查step3](raw/runs/try_ours/step3.jsonl)
- [末步修正版的已有回答](raw/runs/ours/final.jsonl)；[其调用原文](raw/runs/ours/calls.jsonl)
- [状态快照](summary.json)

新回答的解释尚未完成标注；内部结论不能代替正式回复正确率。原小试与修正版分别保存，未改写输出。修正版复用第2/3步，只在思考结尾补充“分析已完成，直接回答，不重复内部区块”的提醒，代码见raw/formal_after_pilot.py。复用调用不能重复计成本。
