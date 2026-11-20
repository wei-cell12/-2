# Tier 1.1 cleaning and benchmark construction report

Random seed is 20261120. The input pool contains 150 images. This build excludes 25 synthetic charts, leaving 125 source candidates.

The pass evaluation set contains 125 items, each with three single-turn conditions and two response tracks. This gives 375 condition instances, each available in both response tracks, and 750 individual prompts.

**Final domain counts**

| Domain | Pass items | MCQ prompts | Open-ended prompts |
|---|---:|---:|---:|
| daily_logic | 50 | 150 | 150 |
| chart_data | 25 | 75 | 75 |
| medical_lesion | 50 | 150 | 150 |

**Excluded synthetic charts**

The source files and synthetic CSV/truth assets remain in the local pool. The following units are absent from benchmark_1_1.json.

| Unit ID | Image |
|---|---|
| s1_1_chart_026 | images/chart_data/chart_data_026.png |
| s1_1_chart_027 | images/chart_data/chart_data_027.png |
| s1_1_chart_028 | images/chart_data/chart_data_028.png |
| s1_1_chart_029 | images/chart_data/chart_data_029.png |
| s1_1_chart_030 | images/chart_data/chart_data_030.png |
| s1_1_chart_031 | images/chart_data/chart_data_031.png |
| s1_1_chart_032 | images/chart_data/chart_data_032.png |
| s1_1_chart_033 | images/chart_data/chart_data_033.png |
| s1_1_chart_034 | images/chart_data/chart_data_034.png |
| s1_1_chart_035 | images/chart_data/chart_data_035.png |
| s1_1_chart_036 | images/chart_data/chart_data_036.png |
| s1_1_chart_037 | images/chart_data/chart_data_037.png |
| s1_1_chart_038 | images/chart_data/chart_data_038.png |
| s1_1_chart_039 | images/chart_data/chart_data_039.png |
| s1_1_chart_040 | images/chart_data/chart_data_040.png |
| s1_1_chart_041 | images/chart_data/chart_data_041.png |
| s1_1_chart_042 | images/chart_data/chart_data_042.png |
| s1_1_chart_043 | images/chart_data/chart_data_043.png |
| s1_1_chart_044 | images/chart_data/chart_data_044.png |
| s1_1_chart_045 | images/chart_data/chart_data_045.png |
| s1_1_chart_046 | images/chart_data/chart_data_046.png |
| s1_1_chart_047 | images/chart_data/chart_data_047.png |
| s1_1_chart_048 | images/chart_data/chart_data_048.png |
| s1_1_chart_049 | images/chart_data/chart_data_049.png |
| s1_1_chart_050 | images/chart_data/chart_data_050.png |

**Medical conflicts and replacements**

For every lung-side question, all supplied lesion-box centers are calculated as x + width/2. A center below 0.45 and another above 0.55 in the same image triggers bilateral_lesion_conflict. Each retained lung-side case also requires every center to lie strictly in one specified half, with the source anatomical side consistent with that half.

| Unit ID | Original source | Original centers | Replacement source | Replacement centers | Result |
|---|---|---|---|---|---|
| s1_1_medical_003 | xmlab309/question_3 | [0.802734375, 0.28515625] | xmlab375/question_3 | [0.322265625] | replaced_single_side |

The replaced image, original annotation and original metadata are preserved beside replacement.json. The current candidates index, current image, annotation, mask and image manifest refer to the replacement. The old full-pool snapshots retain their historical source IDs; their replaced medical image is recoverable from the replacement history.

Current bilateral conflicts are 0. Items pending replacement are 0. Pending items are recorded with qc_status needs_replacement in provenance/benchmark_needs_replacement.json and excluded from the pass evaluation set.

**Medical answer distribution**

| Subclass | Items | Source answer distribution |
|---|---:|---|
| laterality | 20 | {'Right': 10, 'Left': 10} |
| lesion_organ | 10 | {'Liver': 10} |
| organ_binary | 10 | {'Heart': 5, 'Lung': 5} |
| abnormality_presence | 10 | {'No': 5, 'Yes': 5} |

All 10 CT abnormal-organ questions have source answer Liver. This answer concentration is retained and explicitly reported. Clinical labels come from the source dataset; pass denotes the construction and annotation-consistency checks defined here.

**Question and answer cleaning**

Daily questions have the trailing Please answer directly instruction removed. Source options are parsed separately; a letter-only gold is expanded into its full source option text. provenance/benchmark_source_cleaning.json preserves the raw question, clean body, source options, source gold and extracted semantic answer for every item.

The English prompts carry forward the reviewed scope refinements for reflection counting, occluded objects, object identity, anatomical side, chart axes and chart panels. Correct answer meanings are rendered in fluent English. The Chinese evidence anchor is retained as evidence_anchor, with evidence_anchor_en added; the schema explicitly defines both fields.

MCQ options contain the evidence-supported answer, the misleading answer, another plausible visual interpretation, and exactly one fixed-text insufficiency option. All four positions are shuffled. In binary medical tasks, the extra distractor addresses another visible interpretation such as target visibility, neither organ or bilateral involvement; it is not a second insufficiency option.

**Pressure and valid-feedback distribution**

| Domain | Pressure type | Count | Percentage |
|---|---|---:|---:|
| daily_logic | direct_pressure | 35 | 70.0% |
| daily_logic | eyewitness_or_context | 15 | 30.0% |
| chart_data | authority_persona | 9 | 36.0% |
| chart_data | direct_pressure | 5 | 20.0% |
| chart_data | fabricated_citation | 11 | 44.0% |
| medical_lesion | authority_persona | 25 | 50.0% |
| medical_lesion | direct_pressure | 10 | 20.0% |
| medical_lesion | fabricated_citation | 15 | 30.0% |

| Domain | Valid-feedback type | Count | Percentage |
|---|---|---:|---:|
| daily_logic | detail_pointing | 50 | 100.0% |
| chart_data | authority_valid | 12 | 48.0% |
| chart_data | detail_pointing | 13 | 52.0% |
| medical_lesion | authority_valid | 25 | 50.0% |
| medical_lesion | detail_pointing | 25 | 50.0% |

Naturalness fallback events are 0. Daily pressure uses ordinary viewer or photographer/eyewitness language. Chart roles refer to the actual subject of the chart, with source-table claims adapted to its units and panels. Medical claims use generic image-reading reports and physician roles.

Documents cited in fabricated_citation user messages are deliberately invented pressure cues. They are not verified source documents or dataset provenance. No real guideline clause or academic citation is fabricated.

**Option positions**

| Domain | Correct A | Correct B | Correct C | Correct D |
|---|---:|---:|---:|---:|
| daily_logic | 13 | 13 | 12 | 12 |
| chart_data | 6 | 6 | 7 | 6 |
| medical_lesion | 13 | 12 | 12 | 13 |

| Role | A | B | C | D |
|---|---:|---:|---:|---:|
| answer | 32 | 31 | 31 | 31 |
| wrong | 23 | 31 | 39 | 32 |
| fallback | 34 | 31 | 34 | 26 |

**Automated validation and reproducibility**

- Draft 2020-12 JSON Schema validation passes, with additionalProperties false at every record/variant/options level.
- All retained images reopen through PIL, meet the minimum size, have non-constant pixels, and have unique decoded-image hashes.
- Every MCQ contains four distinct options, one fixed insufficiency text, the correct semantic answer and the target misleading answer.
- Open-ended prompts carry no MCQ list or option-letter instruction. Correct feedback contains no choice-letter disclosure. Both tracks use the same underlying question and user guidance.
- Correct-option positions are balanced globally to within one item. Every item is present in all three conditions and both tracks.
- Prompts, options, answer text and English evidence contain no Chinese characters. The original Chinese evidence is stored only in its explicitly retained metadata field.

Install code/requirements.txt and run python code/build_and_validate_1_1.py --root dataset_1_1 --seed 20261120 from the parent directory. From within dataset_1_1, omit --root or use --root . . Running twice with the same inputs and seed reproduces the same benchmark bytes. Validation is repeated before outputs are written.
