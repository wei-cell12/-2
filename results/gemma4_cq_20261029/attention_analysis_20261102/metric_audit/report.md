# 两篇论文与ClockQA注意力分析的口径核查

## 核查结论

目前没有证据证明既有JSD加减、按题分组或求均值存在算术错误。已确认存在方法比较不等价，不能用JSD错误组/守住组的倍数来检验或否定CVPR论文的空间注意力散开结论。先前讨论应严格限于既有指标，不能外推为Gemma图像感知没有受损。

## CVPR 2026：Deeper Thought, Weaker Aim

官方页面：https://openaccess.thecvf.com/content/CVPR2026/html/Peng_Deeper_Thought_Weaker_Aim_Understanding_and_Mitigating_Perceptual_Impairment_during_CVPR_2026_paper.html
正文：https://arxiv.org/html/2603.14184v2

正文§3.1定义：从问题末尾token到图像的逐层逐头注意力，以真实相关区域bounding boxes计算RRAR，即区域内平均注意力/全图平均注意力；§3.2比较Direct、Reason和Region-guided提示。§3.3另定义图像内部空间熵H_img；§4采用同时考虑图像注意力与熵的选头策略，非只按图像总注意力高低选固定96头。

我们当前主要表是：中性/错误用户主张条件，思考前W个token的聚合图像内分布，按中性图像总注意力选96头，再比较JSD或扣内部波动的JSD。四处不同：实验操纵、query位置、测量指标、选头方法。相关区域也是由正确时分针方向定义的扇区代理，不是标注bounding boxes。短针外侧方向包含并非真实指针的背景块；不应冒称已经做了论文的ground-truth ROI验证。

JSD(P,Q)只测分布差异；熵H(P)测单个分布散开程度。相同JSD可对应更集中、更分散或平移后的集中分布。扣波动JSD及其组间比值都没有自动获得方向性。因此Gemma错误组JSD倍数较小，不是该论文发现的反例。

缓存粒度：reading/*.npz按8token存dist_sum，不保留每token分布。能计算8token分组或前缀聚合分布的熵，但不能把“聚合分布的熵”冒充“逐token熵的均值”。论文主要诊断query也不同。准确复现需要相应query缓存及独立相关区域定义，不能通过挑选有预期趋势的窗口代替。

## AAAI 2026：Causal Tracing of Object Representations

正式论文：https://ojs.aaai.org/index.php/AAAI/article/view/40431
正文：https://arxiv.org/html/2511.05923v2

其FCCT使用原图、加高斯噪声的图、恢复指定MHSA/FFN/隐藏状态的三次运行，用正确物体答案概率的恢复率衡量因果作用。其图像损坏操纵与我们“用户错误主张”不同；其activation-patching因果测量也不是注意力JSD。论文不能直接推出我们的压力实验每模型每分组都应有同方向JSD变化。

## 回放实现的静态检查

来源代码完整只读保存在implementation51.json；摘出attn51.py.txt，未修改远端代码。

- reading_one会重新构造输入，并对Gemma已保存input_ids逐项比较；输出长度匹配。
- ReadingRecorder按q_pos-(n_in-1)对应被预测的输出token位置；Gemma跳过3个思考开头标记。静态查看未发现显然的输出序号偏移错误，但这不是新的GPU数值验证。
- 图像索引由模型image_token_id匹配；529token与23×23网格匹配。
- 注意力重算使用调用传入scaling或module.scaling、重复KV头，并使用传入attention_mask；未发现简单把全局head_dim固定成滑动窗口head_dim的写死缩放。
- Gemma仅10个全局层被记录；50个sliding层不在本统计中，不能说已检查模型全部视觉处理。

### 尚未闭合的数值验证缺口

attn._recording_attention不是纯旁路读数：重算FP32 attention probabilities后，会把重算结果作为该层输出返回。它可能改变后续层状态。既有中性参照元数据的selfcheck_max_abs_diff=0.125；reading recorder设置selfcheck_left=0，没有新做每层/输出logit的等价核查。

0.125是attention-output最大绝对差，不是区域份额误差或JSD误差。不能直接断言缓存全部无效，也不能宣称已充分证明重算不影响结论。没有在本次读论文时静默修补或重跑，也没有把结构检查ok当作数值等价通过。

## 应修正的解释

保留可追溯的既有描述统计；撤除“JSD倍数较小即可排除空间注意力散开/读图受损”的解释。要回答论文式问题，需用方向明确的空间熵、独立相关区域RRAR以及对应query的逐层逐头分析，同时闭合记录器数值等价检查。不能先认定必须同向，再选择区域、头或窗口去追出预期结论。
