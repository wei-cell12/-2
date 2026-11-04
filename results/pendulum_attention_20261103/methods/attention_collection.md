# PENDULUM spatial attention collection, 2026-11-02

User instruction: collect attention now for suitable remaining image types. Fixed scope: `re_used` and `self_captured`, first 50 source rows each, original three conditions (`neutral`, `belief_right`, `belief`), 300 saved-reply replays per model. No new sampling, outcome selection, or altered prompts/budgets.

These categories include question-specific object/attribute and counting evidence. Independent evidence masks must be annotated from the question and original image later; the model's top-attended 20% must not be called ground-truth evidence. Counting may require multiple disjoint regions. Attention alone does not establish whether the model recognized an object correctly.

## Deployment

Remote base on both machines: `/mnt/sdb/linxin/pendulum-20261102/expand4x50/attention`.

- Server51 Gemma: GPU1,2, collector PID1418313. Existing generation on GPU3,5 untouched. GPU4 excluded, GPU6 unused.
- Server59 Qwen3.8: finite CPU queue PID4021541 waits for own GPU0 generation worker0 to finish successfully, then rechecks GPU availability before loading.
- Server59 Qwen3.6 MoE: finite CPU queue PID4021542 waits for own GPU2 generation worker0, same resource checks.
- No other user's processes changed, no recurring automation created. Each collector has an exclusive lock, 48-hour finite deadline, 20GiB disk safety floor, failure preservation and natural GPU release after its collection ends.

Collector scripts are isolated in `attention/code/`; frozen inference source and live generator scripts are unchanged. Exact code hashes are in its `manifest.json`. Runtime/config hashes and source-record/image hashes accompany each result.

## Stored arrays

One compressed NPZ plus JSON metadata per source reply under `attention/<model>/records/`:

- All *standard softmax attention* layers/heads; Gemma all60 layers (50 sliding, 10 global). Qwen linear-attention blocks are not presented as token-position softmax maps.
- `L<layer>_image_mass` and `text_mass`: `[output_tokens, heads]`, FP32, all saved output tokens.
- `stage_sum`: `[3, heads, image_tokens]`, with `stage_count`, for last-prompt query, thought content, answer. Divide sums by counts to obtain stage means. The last-prompt query overlaps the first thought predictor for models whose thought content starts at output index0; this is explicit metadata, not independent observations.
- `early_sum`: `[64, heads, image_tokens]`, groups of8 consecutive thought tokens through token512, with counts. Supports matched64/128/256/512 windows and alternating-group split-half JSD. Incomplete windows must not be silently padded with observations.
- `visible_image_tokens`: `[output_tokens]` for each layer, so sliding-window loss of image visibility can be excluded/reported separately.
- Exact image-token positions, image grid, original image size/SHA, processor coordinates when available, and original input/output hashes. No complete token-by-token text attention matrix is stored.

## Verification already performed

Both environments passed two CPU cases covering causal/global and sliding masks, original attention return identity, query alignment, stage sums/counts, early groups, image mass, and image visibility.

First Gemma replay: `PD-re_used-0000__neutral`, 60 layers x32 heads, image grid28x19=532 tokens. Recorded/unrecorded original forward final logits were bitwise identical, max absolute difference0. Attention row-sum error7.152557e-7. Recomputed attention output relative diagnostic≈0.00394; this reconstruction is used for recording only and never returned to the model.

Current machine-readable local snapshots: `status_attention51.json`, `status_attention59.json`, `attention_first_item_audit51.json`. The latter reads compressed arrays on51 to verify the first item's available three conditions, tensor shapes, grids, counts and finite image mass. Subsequent collection remains in progress; no claim that all900 replays have completed.

Raw NPZ stays on the servers. Heatmaps/statistics can be derived after the three conditions are available; semantic labels and independent evidence masks remain separate work.
