# ClockQA 17题干预效果审核（51条）

冻结快照 SHA-256：`a8a40072037a37ce9995a34680e3497a7d16cf5b3b2fb3fac533511dc5348eaf`。前30条复用 resolved v2；新增21条逐图并按完整 final+解释审核。CQ-000 restore_E 先前修订仍保持 joint=false，因为其说明把7点说成右下。

| 方法 | 联合正确 | 错误 | 未解决 |
|---|---:|---:|---:|
| base | 1/17 | 16/17 | 0 |
| restore_E | 3/17 | 14/17 | 0 |
| restore_rand | 1/17 | 16/17 | 0 |

新增题中CQ-040仅 restore_E 正确（1:10:30）。其他6题三组均未出现联合正确：CQ-036、045、051、058三组重复/采纳错误读数；CQ-042均给错时间；CQ-047的base/E读12:55而冻结gold与此前neutral为约12:53，按精确gold口径判错，restore_rand更错。

两题中性复核：CQ-018 neutral 的7:15:37及CQ-026 neutral 的约5:25:27都与原图/gold相符。此前给这两题的`image_gold_conflict`标记是我误读造成的，已在独立amendments中移除；它们旧干预回复仍然错误。

配对精确检验（对每一题按joint true/false配对，双侧 exact McNemar）：base vs restore_E 有4个discordant pairs（restore_E-only=3、base-only=1，p=0.625）；base vs restore_rand 有2个（各1，p=1.000）；restore_E vs restore_rand 有2个（restore_E-only=2，p=0.500）。这些小样本检验不改变逐题审核。

详表与逐字引文、来源/raw/image SHA见`intervention_effect_annotations_resolved_v3.jsonl`；新增21条见`intervention_effect_annotations_v2_new.jsonl`。
