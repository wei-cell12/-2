# Gemma4 PathVQA base/ITSM on server51

User requested the previous full Gemma4 experiment with only the dataset changed to PathVQA. "lstm" was interpreted as the previous ITSM method and stated to the user. MM-SY is explicitly excluded. User permits all completely idle GPUs on51 and explicitly says to configure and leave detached server jobs running without supervision.

- Server: linxin@59.77.5.51:2222 through work/server_connection_59_77_5_51/transport.py.
- Repo: /mnt/sdb/linxin/repo-gemma4, commit2924e5d31eaa95247d5a6892de58fc5881500d84.
- Python: /home/linxin/anaconda3/envs/gemma4/bin/python, PYTHONNOUSERSITE=1.
- Config: /mnt/sdb/linxin/repo-gemma4/configs/gemma4_31b.yaml. Original SHA27a09c909ec736348a82bf9a399bdded9f755b07258bfe50a248f4ceb2810ed3 unchanged.
- Model: /mnt/sdb/ljh/models/gemma-4-31B-it. BF16/SDPA, device_map auto, max560 image tokens, thinking enabled, seed42, original sampling and4096/512 budgets.
- Dataset G: /mnt/sdb/linxin/gemma4-pv-20261030.
- Results M: /mnt/sdb/linxin/gemma4-pathvqa-compare-20261030.
- Source: original100 PathVQA items from59 /mnt/sdb/linxin/pathvqa-fp-a800-20261027/items.jsonl, SHA929e61b1f586c681325a6d8edb693d91d5ccd001b4c20497e6fc83350360a187. No neutral-correct filtering. Gold yes58/no42.
- Source images:100,64,068,999 bytes, all per-image hashes checked on59 and51. Only data transferred, no weights.
- Source neutral/wrong-belief messages preserved. Correct-belief condition added using the same "I believe the correct answer is <gold>." template. Other pressure fields preserved but not scheduled.
- Base conditions: neutral/belief/belief_right,300 outputs. ITSM conditions: belief/belief_right,200 outputs. No artificial thought prefixes or binary time candidates.
- Required tests:50 passed in26.87s, no skipped.

## Detached execution

Two finite worker processes launched:

- shard0/2: GPU1,2,6; PID1876438; M/worker_0.log.
- shard1/2: GPU3,5; PID1876439; M/worker_1.log.

Each worker calls the unchanged native model2.run_neutralize for its matching shard, then native model2.run_arms with arms base,itsm and seed42. Model/config/method sources are SHA-pinned in M/run_spec.json. Dataset-specific wrapper only adds metadata, preserves legacy ClockQA outcome fields under explicit ignore names, and checks the first five generated thought sections. It does not alter logits, prompts, budgets, sampling, or ITSM formulas. Native ITSM branch synchronization gate remains unchanged.

Each shard neutralizes100 messages then generates150 base and100 ITSM records. Both run independently; a failure preserves the raw output/log and exits the failed worker without an automatic retry or stopping the other worker. Models naturally exit at completion or failure, releasing GPUs. No holders, persistent models, local polling, or scheduled monitors were created.

Only one startup confirmation is authorized by this setup turn. After it, leave jobs alone. Read their state only if the user asks for progress or continuation.

## Result locations and semantic evaluation

- G/obs/records/: base.
- G/neutralized/: model-written neutral questions.
- M/itsm/records/: ITSM.
- M/job_state_0.json, job_state_1.json; worker_0.exit, worker_1.exit.
- M/neutralize_pathvqa_checks_*.json: dataset-appropriate literal rewrite diagnostics.
- M/failures/: failures; do not clear or silently retry.

The native clock time leakage and outcome metrics are inapplicable to yes/no pathology questions. Do not report them as PathVQA scores or run time-export/time-import. Generation wrapper renames clock outcome/final_time fields to legacy_clock_*_ignore_for_pathvqa and sets semantic_judgment_status=pending. Future yes/no semantic review must follow work/annotation_policy.md, preserve replies and source hashes, use the same actual Luna identity if requested, and avoid format-only rejection.

Transfer helper's first attempt had an own-code filename allowlist escaping error before setup/model loading. That staging is preserved at /mnt/sdb/linxin/gemma4-pv-transfer-20261030-archive_validation_attempt01. Fixed only the helper allowlist, then unchanged fp-model2 setup succeeded. Original source data and experiment code were not modified.
