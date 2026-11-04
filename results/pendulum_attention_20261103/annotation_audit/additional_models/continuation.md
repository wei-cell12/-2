# 当前执行接续

1. 同一 /root/pathvqa_luna_continuation 正在 work/blind_pendulum_20261103 逐条提取1200份（24×50），按原说明本人语义提取，父代理不替代。未完成前不可给gold。
2. 全部labels及completion完成后：python3 work/pendulum_server51_20261102/prepare_additional_comparison.py 。先validate通过1200+来源引文，再封存，生成匿名comparison/requests_NN.jsonl。若引用错误只交回同一Luna修正，不改语义或原模型数据。
3. 给Luna同一comparison目录，要求读新INSTRUCTIONS.md，完成comparison_NN.jsonl。新输出须包括question_used/reference_answer_used/image_sha256_context，父端会核对防跨题污染。解释须看图，短答不虚构解释；标签争议单列，不能改gold。
4. CPU注意力已完成并取回：gemma4_attention_metrics_private.json、qwen36_moe_attention_metrics_private.json。无GPU占用，CPU进程已退出。主规则更新为各层类top25%（Qwen96、MoE40、Gemma全局80/滑动400）；固定96保留对照。旧CPU元数据写96primary已由selection_policy_20261103.json明确纠正解释，不改原指标。
5. 全比较完成后用 bundled Python（numpy/PIL）：finish_three_model_analysis.py，验证源SHA/每条key/上下文；产出three_model_results_20261103/REPORT.md、summary.json、sample_registry.json、两组judgments。旧Qwen600标注含之前3条更正，原样读取。
6. 运行prepare_primary_case_maps.py，按非争议可比flip中E份额最大降幅2题和最小降幅1题选例。把case_map_requests_private.json按server分组（Gemma51，Qwen和MoE59），将extract_primary_case_maps.py前加REQUESTS=<相应JSON字面量>，通过remote.py执行CPU聚合，只取JSON maps，不取NPZ；合并为three_model_results_20261103/primary_first64_maps_private.json。
7. bundled Python运行build_three_model_cases.py、build_three_model_overview.py。目视核验一张热图。读最终REPORT/summary，必要时增补真实分析结论与原文例子，不猜测数字或把JSD变成因果证据。分母明确200题×3=600/模型，注意力只100题×3=300/模型，每窗口单独样本数。
8. 最终向用户给三模型中性/用户说错/用户说对结果、按中性正确分组的变化、主要注意力结论与本地报告链接。用户这轮只要求标注分析，不自行重跑生成。

SSH：work/s2va_multimodel_20261101/remote.py 51|59 local_script --output path；网络require_escalated。不要读密码私钥，不碰他人进程。远端根/mnt/sdb/linxin/pendulum-20261102/expand4x50/。
BundledPython=/Users/zzh/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3。
