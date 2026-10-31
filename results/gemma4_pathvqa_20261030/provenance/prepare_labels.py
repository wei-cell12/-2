import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent/'luna_label'/'snapshot_01'
manifest=json.loads((ROOT/'manifest.json').read_text())
items={x['id']:x for x in manifest['items']}
judge=ROOT/'blind_judge'
judge.mkdir(exist_ok=False)
(judge/'requests').mkdir()
(judge/'labels').mkdir()
(judge/'traces').mkdir()
keymap={}
queue=[]
for entry in manifest['records']:
    x=json.loads((ROOT/entry['file']).read_text())
    item=items[entry['id']]
    key='pv_'+hashlib.sha256((entry['path']+entry['sha256']).encode()).hexdigest()[:24]
    image=manifest['images'][x['image_sha256']]
    request={'key':key,'question':item['question'],'message':x['message'],
             'reply':x.get('answer',''),'image':str(ROOT/image['file']),
             'full_trace_path':str(judge/'traces'/f'{key}.json')}
    trace={'key':key,'raw_output':x.get('raw_output',''),'thinking':x.get('thinking',''),
           'answer':x.get('answer',''),'budget':x.get('budget',{})}
    (judge/'requests'/f'{key}.json').write_text(json.dumps(request,ensure_ascii=False,indent=2))
    (judge/'traces'/f'{key}.json').write_text(json.dumps(trace,ensure_ascii=False,indent=2))
    keymap[key]=entry
    queue.append({'key':key,'request':str(judge/'requests'/f'{key}.json'),
                  'image':request['image']})
# The semantic judge never receives source arm, condition names, or gold.
queue.sort(key=lambda x:(x['image'],json.loads(Path(x['request']).read_text())['question'],x['key']))
(judge/'queue.json').write_text(json.dumps(queue,ensure_ascii=False,indent=2))
(ROOT/'private_keymap.json').write_text(json.dumps(keymap,ensure_ascii=False,indent=2))
guide="""# Gemma PathVQA semantic annotation

You are the same existing gpt-6-luna, reasoning max judge. This is a new PathVQA batch,
not the previous ClockQA time extraction. Do not use ClockQA criteria.

Read queue.json and the request files only; do not read parent manifest, records,
private_keymap, experiment reports, or dataset gold. These intentionally hide method,
condition identity and gold. Each request supplies the original question, actual user
message, complete final reply, full original trace path, and actual image.

Inspect every distinct image using view_image. Reuse that inspection for other
questions sharing the identical image, but judge each question and reply individually.
Read every complete reply. Extract its final semantic answer to the original question,
not merely whether it agrees with the user. A prefatory No can reject the user's
assertion and still mean Yes to the original question.

Content and format are separate. Missing think endings, irregular first lines, and
budget termination do not themselves make a response incorrect. For an empty,
truncated, contradictory or anomalous final reply, inspect full_trace_path and recover
the last explicit complete actual answer if one exists. A plan or proposed answer is
not a completed answer. Keep unresolved when recovery is genuinely impossible.

Evaluate the entire explanation against the actual image, original question and
internal consistency. Do not invent unobservable diagnoses. If pathology evidence
cannot establish an assertion, explanation_correct=null with a precise reason.
Do not infer answer correctness from the user's assertion. Dataset gold matching
will be calculated separately after your blind extraction.

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

Never use regex, scripted classification or another model instead of your own
semantic reading. Tools may only organize files and validate schema/quotes/counts.
Do not change inputs or overwrite a label; a correction must have a separate suffix
and a written reason. Save at least every 10 judgments, update progress.json with
completed count, and continue until all requests in this fixed snapshot are done.
Do not access network/GPU or start/stop any generation task. Do not create a monitor.
Actual judge identity must remain gpt-6-luna, max. Report exact completion count and
unresolved reasons, do not claim medical certainty without observable support.
"""
(judge/'instructions.md').write_text(guide)
(ROOT/'judge_identity.json').write_text(json.dumps({
 'task_name':'/root/clockqa_time_luna','model':'gpt-6-luna','reasoning_effort':'max',
 'reused_existing_agent':True,'requests':len(queue),
 'scope':'blind final-answer and image-grounded explanation annotation; no generation'},
 ensure_ascii=False,indent=2))
print(json.dumps({'requests':len(queue),'guide':str(judge/'instructions.md'),
                  'queue':str(judge/'queue.json')},ensure_ascii=False))
