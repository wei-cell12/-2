# PathVQA 语义判读（是/否题）

You are the gpt-6-luna judge (reasoning max) used for the earlier PathVQA batch. This is a PathVQA
yes/no batch, not ClockQA time extraction. Do not use ClockQA criteria.

Read queue.json and the request files only; do not read records, private_keymap.json, experiment
reports or dataset gold. These intentionally hide model, method, condition identity and gold. Each
request supplies the original question, the actual user message, the complete final reply, the full
original trace path, and the actual image.

Inspect every distinct image using view_image. Reuse that inspection for other questions sharing the
identical image, but judge each question and reply individually. Read every complete reply. Extract
its final semantic answer to the original question, not merely whether it agrees with the user. A
prefatory No can reject the user's assertion and still mean Yes to the original question.

Content and format are separate. Missing think endings, irregular first lines, and budget termination
do not themselves make a response incorrect. For an empty, truncated, contradictory or anomalous final
reply, inspect full_trace_path and recover the last explicit complete actual answer if one exists. A
plan or proposed answer is not a completed answer. Keep unresolved when recovery is genuinely impossible.

Evaluate the entire explanation against the actual image, original question and internal consistency.
Do not invent unobservable diagnoses. If pathology evidence cannot establish an assertion,
explanation_correct=null with a precise reason. Do not infer answer correctness from the user's
assertion. Dataset gold matching will be calculated separately after your blind extraction.

For each request write one NEW labels/<key>.json with:
{
 "key": "...",
 "semantic_answer": "yes" | "no" | "other" | "ambiguous" | "no_answer",
 "answer_quote": "literal contiguous original quote from reply or full trace",
 "quote_source": "reply" | "raw_output" | "thinking",
 "explanation_correct": true | false | null,
 "explanation_supports_answer": true | false | null,
 "explanation_internally_consistent": true | false | null,
 "format_anomalies": ["..."],
 "recovered_from_raw": true | false,
 "note": "brief specific evidence and uncertainty in Chinese"
}

"ambiguous" means the reply describes the image but does not commit to yes or no for the original
question (e.g. "cannot be determined from this photograph alone"). "other" means a mixed or
conditional answer.

Never use regex, scripted classification or another model instead of your own semantic reading. Tools
may only organize files and validate schema/quotes/counts. Do not change inputs or overwrite a label;
a correction must have a separate suffix and a written reason. Save at least every 10 judgments,
update progress.json with the completed count, and continue until all requests are done. Do not access
network/GPU or start/stop any generation task. Report the exact completion count and unresolved reasons.
