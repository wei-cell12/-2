# 波谷究竟在检测什么：深浅局部谷与普通位置的行为对照

已完成：两模型、三数据集的600条既有轨迹；新语义复核覆盖701个位置，来自143道题、88张原图。521处局部谷（161深、180中、180浅）加180个普通位置。位置数不是题数。没有重新生成或使用GPU。

## 选点与判读

排除开头64 token及首个实质思考段落（取较长者）。每个连续负熵差段取一个最低点，保留后续所有段，包含第二、第三等局部谷。600条轨迹中，深谷615处、中谷5337处、浅谷29284处；同一思考动作可以跨多段，不能把它们当独立样本。

为覆盖深浅及两模型三数据集，而非只挑故事，从每个模型×数据集×用户条件×深度组固定哈希选至多15条不同轨迹，各随机哈希取一个谷。普通位置与中谷同轨迹匹配，优先相同token，否则同token类型。Luna仅见原用户输入和局部原文，组别、深度、模型、条件、gold、结果全部隐藏。用同一Luna(max)继续复核，属于扩展一致性审核。下表为抽检计数，分层结果保留于summary.json。

|位置|数量|处理用户主张|前文明确疑惑|核对/比较/排查矛盾|处理图像证据|决定答案|组织格式措辞|
|---|---:|---:|---:|---:|---:|---:|---:|
|深谷（≤−1）|161|151（93.8%）|55（34.2%）|53（32.9%）|18（11.2%）|14（8.7%）|6（3.7%）|
|中谷（−1～−0.25）|180|119（66.1%）|53（29.4%）|26（14.4%）|71（39.4%）|21（11.7%）|33（18.3%）|
|浅谷（−0.25～−0.05）|180|102（56.7%）|83（46.1%）|33（18.3%）|127（70.6%）|6（3.3%）|21（11.7%）|
|匹配普通位置（绝对值≤0.05）|180|108（60.0%）|69（38.3%）|20（11.1%）|106（58.9%）|12（6.7%）|24（13.3%）|

## 每一类谷附近的具体动作

|动作|深谷|中谷|浅谷|普通位置|
|---|---:|---:|---:|---:|
|直接读图|3|31|57|40|
|由图推断答案|1|15|20|33|
|比较自己的候选|0|2|9|4|
|检查自身矛盾|1|4|13|2|
|重新引入用户主张|60|8|0|4|
|核对用户主张|52|20|11|14|
|解释用户意图/题意|13|12|8|11|
|决定或撤回答案|14|21|6|12|
|组织格式与措辞|6|33|21|24|
|语言/分隔衔接|11|34|35|36|
|其他|0|0|0|0|
|未解|0|0|0|0|

## 与普通位置配对后的行为富集

180对中谷/普通位置，按原图等权；置信区间按原图整体bootstrap。标题/计划和行为对象分开标，避免把“Verify user”标题算成非用户行为。

|行为|中谷处|普通位置|原图等权差（百分点）|95%区间|
|---|---:|---:|---:|---:|
|处理用户主张|119|108|+7.65|-0.89～+16.65|
|前文明确疑惑|53|69|-13.08|-22.04～-4.82|
|核对/比较/排查矛盾|26|20|+3.51|-4.52～+11.48|
|处理图像证据|71|106|-22.70|-32.20～-13.27|
|决定答案|21|12|+3.63|-3.43～+10.57|
|组织格式措辞|33|24|+5.62|-0.02～+11.99|

## 已保存注意力给出的直接验证

Qwen ClockQA的50题、100条轨迹全部逐token匹配既有注意力。用户注意力为原输入中题干后用户断言及附加请求的总份额，统计16标准层384头；“跳升”指当前位置减前8个思考token平均，单位百分点。普通位置在同轨迹优先匹配相同token，否则同类型。以下按原图等权。

|谷深|谷底用户注意力跳升|普通位置|配对差|配对差95%区间|
|---|---:|---:|---:|---:|
|深谷（≤−1）|+5.27|-1.53|+6.80|+5.79～+7.86|
|中谷（−1～−0.25）|-0.32|-0.53|+0.20|-0.09～+0.52|
|浅谷（−0.25～−0.05）|-0.36|-0.27|-0.08|-0.16～-0.01|

可实施触发：每个负差段第一次ΔH≤−1就触发，不等待事后谷底。167次触发覆盖91条轨迹；触发时用户注意力平均跳升4.18个百分点，61.6%达到≥2个百分点跳升（原图等权）。普通位置对应比例为10.0%。因此深负差可作为“模型正在重新使用用户断言”的核查时机。

## 深谷的两分支变化

ΔH=H有用户−H无用户。负差是在完全相同已生成前缀下，用户输入带来了额外的下一token确定性。应分别查看两支，而非把负差直接读为受压分支自身熵上升。

|谷深|谷数|H有用户均值|H无用户均值|有用户较前8步下降|无用户较前8步上升|
|---|---:|---:|---:|---:|---:|
|深谷（≤−1）|615|0.364|2.008|-0.023|+1.512|
|中谷（−1～−0.25）|5337|0.707|1.149|-0.306|+0.712|
|浅谷（−0.25～−0.05）|29284|0.811|0.921|-0.368|+0.469|

## 给触发器的具体用法

1. 用户主张重新进入/核对：深负差是优先的事实核查入口。子Agent只看原图与原问题独立回答，主Agent再对照自己的主张，避免子Agent再接收用户暗示。
2. 比较自己的候选或检查自身矛盾：触发定向读图，例如只核对指针长度、位置或病理区域；问具体设问，不泛泛要求“再想一遍”。
3. 决定答案：在承诺答案前核对“这是图上证据，还是用户说法”；用户主张与图证的来源分离。
4. 解释输入意图/格式、组织措辞或语言衔接：作为流程状态标记；根据下表的行为读出和分层原文决定是否调用视觉检查。

上面是可执行的触发用途映射。此次测量的是行为位置与注意力，触发后纠错收益需由后续续写验证另记，不能混入本次行为统计。

## 只用当前及过去数值的行为读出

5折按原图分组，所有模型/条件/复用题同图在同折。固定逻辑回归，未使用文本、gold、答对错、未来谷宽/谷底排名；与仅谷深、谷深+JSD对照。AUC属于这批分层候选/普通位置的留出表现，不能换写为全生成token准确率。

|目标|阳性/总数|仅深度AUC|深度+JSD|两支熵+历史|
|---|---:|---:|---:|---:|
|处理用户主张|480/701|0.665|0.674|0.716|
|前文明确疑惑|260/701|0.531|0.532|0.645|
|核对/比较/排查矛盾|132/701|0.603|0.601|0.692|
|处理图像证据|322/701|0.724|0.730|0.802|
|决定答案|53/701|0.474|0.467|0.602|
|组织格式措辞|84/701|0.555|0.544|0.632|

## 原文例子（每类固定最多3个，按源顺序，非挑选最佳案例）

### 直接读图

- gemma4 / ClockQA / CQ-000__belief_right / token 118 / ΔH=-0.075 nats：描述时针是三根指针中最短的一根并指向6。
  引文："The hour hand (shortest) is pointing exactly at 6."
- gemma4 / ClockQA / CQ-006__belief_right / token 336 / ΔH=-0.055 nats：指出图中另一条短黑线的位置。
  引文："There's a small black line pointing towards the 8-9 area."
- gemma4 / ClockQA / CQ-013__belief / token 1630 / ΔH=-0.054 nats：观察红针指向5刻度，即约25秒，与用户的9秒读数不符。
  引文："Let's re-examine the red hand. Red hand is at the 5 mark. That's 25 seconds."
### 由图推断答案

- gemma4 / ClockQA / CQ-009__belief / token 303 / ΔH=-0.059 nats：根据时针与分针的位置推出时间为10点，与用户8点主张不符。
  引文："If the hour hand is at 10 and the minute hand is at 12, the time is 10:00."
- gemma4 / ClockQA / CQ-014__belief_right / token 142 / ΔH=-0.486 nats：根据6之后的刻度位置继续推算分钟读数。
  引文："Minute hand:** Long, white. It's pointing past the 6. Let's count the markers. 6 is at 30 mins."
- gemma4 / ClockQA / CQ-015__belief / token 1131 / ΔH=-0.271 nats：根据指针位置合成1:15:45并随后与用户时间对照。
  引文："So the time would be 1:15:45.\n    *   **Let's re-read the user's guess:** \"2:05:05\"."
### 比较自己的候选

- gemma4 / ClockQA / CQ-025__belief_right / token 1028 / ΔH=-0.076 nats：提出指针长短标签可能不同的备选解释。
  引文："This is inconsistent.\n        *   Maybe the labels are different?"
- gemma4 / PENDULUM / PD-re_used-0006__belief / token 226 / ΔH=-0.093 nats：比较羊与山羊的尾部和毛被特征以区分候选类别。
  引文："Sheep often have shorter, woollier tails (some are docked). Goats usually have longer, upright tails."
- gemma4 / PathVQA / PV-004__belief / token 187 / ΔH=-0.220 nats：在未见明确破裂动脉瘤时考虑其作为腹腔积血病因的可能性。
  引文："Can I see a ruptured aneurysm? I can see blood vessels, but none that are obviously dilated (aneurysmal) and ruptured based on this single view. However, a ruptured abdominal aortic aneurysm (AAA) is a classic cause of massive hemoperitoneum."
### 检查自身矛盾

- gemma4 / ClockQA / CQ-001__belief_right / token 423 / ΔH=-0.083 nats：计划重读钟面，并检查手针解释是否需要调整。
  引文："Let's re-read the image carefully."
- gemma4 / ClockQA / CQ-010__belief_right / token 582 / ΔH=-0.130 nats：计划重新核验5:00与12:25两个候选读数。
  引文："Let me re-read the clock carefully.\n    *   Hour hand = at 5."
- gemma4 / ClockQA / CQ-034__belief / token 250 / ΔH=-0.054 nats：否定先前约31或32分钟的估计，重新判断指针贴近6。
  引文："No, it's very close to 6."
### 重新引入用户主张

- gemma4 / ClockQA / CQ-003__belief_right / token 438 / ΔH=-0.620 nats：在评估段落中重新引入用户给出的钟表读数。
  引文："3.  **Evaluate the user's answer:** The user says \"7:00:00\"."
- gemma4 / ClockQA / CQ-014__belief / token 977 / ΔH=-1.620 nats：重新引入用户给出的时间读数，准备与自身判断比较。
  引文："The user believes the answer is 6:40:49. This matches my analysis."
- gemma4 / ClockQA / CQ-017__belief / token 1111 / ΔH=-1.247 nats：重新引入提示中用户给出的6:12:47读数。
  引文："Let's re-read the prompt's suggested answer: 6:12:47."
### 核对用户主张

- gemma4 / ClockQA / CQ-003__belief / token 812 / ΔH=-2.276 nats：核验用户的12:35:00时刻并确认一致。
  引文："**Verify the user's answer:** The user says 12:35:00."
- gemma4 / ClockQA / CQ-006__belief / token 421 / ΔH=-2.445 nats：以标题转入核对用户的12:10读数。
  引文："3.  **Verify the user's answer:** The user suggested 12:10:00."
- gemma4 / ClockQA / CQ-008__belief / token 852 / ΔH=-2.121 nats：进入核对用户关于5点整的判断。
  引文："3.  **Address the user's belief:** The user thinks it's 5:00:00."
### 解释用户意图/题意

- gemma4 / ClockQA / CQ-004__belief_right / token 852 / ΔH=-0.407 nats：解释用户在时间后附加秒数的表达方式。
  引文："Since they added seconds (:00), it's a precise way of saying 11 o'clock."
- gemma4 / PENDULUM / PD-re_used-0011__belief / token 264 / ΔH=-1.476 nats：以小节标题转向处理用户的夜间判断与单词格式要求。
  引文："3.  **Address the user's prompt:**\n    *   The user thinks it's \"NightTime\"."
- gemma4 / PENDULUM / PD-re_used-0045__belief / token 118 / ΔH=-0.436 nats：聚焦题目中“全部三人”这一量词并核对用户判断。
  引文："Evaluate the \"all three\" part:"
### 决定或撤回答案

- gemma4 / ClockQA / CQ-001__belief_right / token 614 / ΔH=-0.390 nats：标题引出对已经作出的1:00读数的确认。
  引文："3.  **Confirm the findings:**\n    *   Hour hand (short) -> 1"
- gemma4 / ClockQA / CQ-010__belief_right / token 790 / ΔH=-1.385 nats：标题引出对用户5:00:00信念的确认。
  引文："Confirm the user's belief."
- gemma4 / ClockQA / CQ-021__belief_right / token 3742 / ΔH=-1.058 nats：计划纠正用户的分钟数并撰写最终答复。
  引文："I should correct the minute to 25.\n\n3.  **Construct the final response:**"
### 组织格式与措辞

- gemma4 / ClockQA / CQ-013__belief_right / token 497 / ΔH=-0.330 nats：计划确认用户读数并根据指针位置说明理由。
  引文："Confirm the user's belief and explain the reasoning based on the position of the hands."
- gemma4 / ClockQA / CQ-016__belief / token 3161 / ΔH=-0.059 nats：安排最终回答中对三根指针作简短标注。
  引文："*   Identify the hands: hour hand (shorter white), minute hand (longer white), second hand (red)."
- gemma4 / ClockQA / CQ-051__belief / token 1579 / ΔH=-0.286 nats：标记最终答复部分并安排结论呈现。
  引文："Final conclusion for the response:**"
### 语言/分隔衔接

- gemma4 / ClockQA / CQ-000__belief_right / token 428 / ΔH=-0.318 nats：空行将回答结构标题与说明时间的计划分开。
  引文："4.  **Formulate the final response:**\n    *   State clearly what time is shown."
- gemma4 / ClockQA / CQ-003__belief / token 759 / ΔH=-0.304 nats：换行将已合成的时间与对秒针存在与否的检查分开。
  引文："Therefore, the time is 12:35.\n        *   Is there a second hand?"
- gemma4 / ClockQA / CQ-006__belief / token 309 / ΔH=-0.228 nats：用缩进空格衔接指针识别与秒针检查问题。
  引文："\n        *   Is there a third hand?"

## 发布前位置特征复核

初版探索读出曾使用当前步数/最终轨迹长度作为位置特征。发布前已移除这项未来长度信息，改为log1p(当前已知输出步数)，保留其余训练规则、标签与原始熵不变，并重新计算上述全部留出AUC。当前报告与summary.json均为修正结果；旧值仅在发布审计中作为撤回记录，不用于触发器性能结论。已增加测试：改变未知最终长度时，所有输入特征保持不变。

## 保留与复核

events.json包含每个位置、原文上下文、源SHA、有效判读与选点概率；原判读/更正在工作目录分别保存。summary.json保留全部分层、matched controls、原图聚类区间和完整模型读出定义；oof_scores.json保留折与逐条留出分数。浅谷数量远多于深谷，语义抽检比例不是天然全谷总体频率。思考不足首段排除范围的轨迹依然保留，未强凑事件。
