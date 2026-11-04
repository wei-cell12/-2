# Qwen PENDULUM: labeling plus the previously used attention comparison

User authorization: “标注qwen的 并和之前一样计算”. Fixed scope is the completed Qwen3.8 expansion: 200 questions x3 native conditions =600 replies, not the older stopped full run or the separate MoE model.

Same existing judge `/root/pathvqa_luna_continuation`, actual `gpt-6-luna`, reasoning `max`, reused and explicitly assigned this finite task. Do not spawn another judge, reveal gold before extraction is sealed, change inference, use GPU or restart any generation.

The same child has been authorized to finish the following whole chain without another user approval:

1. `blind/INSTRUCTIONS.md`, requests_01..12, and their specified traces only: manually extract semantic final answers into labels_NN.jsonl. First6batches (300replies) correspond to the100questions with attention. Original source/private keymap remain outside blind directory.
2. After600 labels and blind/completion.json exist, run `../prepare_qwen_comparison.py` from the task parent. This validates every quote/key, seals raw labels with SHA, then exports comparison/ requests. At this point only, read comparison/INSTRUCTIONS.md and judge nonliteral semantic matches plus replies that supply explicit explanations. Simple literal matches are affirmative matches, not inferred explanation correctness. Unresolved stays unresolved, wrong does not automatically mean adopting the user.
3. Write comparison/comparison_NN.jsonl. Preserve raw labels. Actual image bytes have source SHA validation; comparison images are opaque copies. Do not overwrite original outputs or historical annotations.
4. Execute the parent `finish_qwen_analysis.py` using the bundled Python `/Users/zzh/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3`, with PYTHONDONTWRITEBYTECODE=1, save stdout to finish.log. It creates `../qwen_results_20261102/index.html`, REPORT.md, summaries, all600 judgments and true paired heatmap examples. No GitHub upload requested in this turn.

Attention CPU computation is already COMPLETE on Server59; it loaded no model. Local snapshot `../qwen_attention_metrics_private.json` contains aggregate statistics/maps only, no NPZ or fulltoken list. Matched W=64/128/256/512 eligible counts are82/20/9/2 respectively. All three conditions must be long enough; never pad/regenerate/exclude based on correctness to create a desired count.

Main numeric method is the latest shared Gemma/Qwen rule: rank each item's384 heads by equal average image mass across three matched-window conditions, select96; preserve old neutral-only96 selection separately. JSD in nats; excess=between-condition even-group JSD minus mean within-condition even/odd JSD, groups of8 tokens. No clipping of negative excess. E is the per-head top20% of full neutral-thought image attention, not human-object ROI. Output includes image mass, E share, RRAR, absolute percentagepoint and relative percentage changes. 100questions share23images: retain per-question results and supplement uncertainty with image-cluster bootstrap/sign flips. No intervention was performed in this PENDULUM run.

Validation: direct use of frozen repository JSD/norm formulas; paired p/cluster helpers tested; isolated synthetic600record integration test passed joins, grouped statistics, HTML/PNG generation and completion gates. Synthetic data were removed, never written to real label/results directories.

Current task_state.json is authoritative for judging/comparison/complete. Completion is true only after real600 labels, required semantic reviews and final report exist. If a gate fails preserve all files and tell the parent; do not report synthetic or partial results as complete.
