"""Mechanical same-Luna candidate selection; no new semantic judgments."""
import hashlib,json,subprocess,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;AUDIT=BASE/'luna_audit'
REMOTE='/mnt/sdi/linxin/proofwriter_runs/clockqa-scaleup-20261025/interventions'
ORDER=['statement','belief','conviction','authority','social']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def build():
 source=AUDIT/'source_snapshot.json';packet=json.loads(source.read_text());prior=json.loads(Path(packet['prior_source_snapshot']).read_text())
 records={r['record_key']:r for r in prior['records']+packet['records']}
 annotations={r['record_key']:r for r in lines(packet['prior_luna_annotations'])+lines(AUDIT/'annotations.jsonl')}
 cases={r['case_id']:r for r in lines(AUDIT/'case_eligibility.jsonl')}
 items=[];reviewed=[]
 for index in range(62):
  qid=f'CQ-{index:03d}';keys=[qid+'--'+t for t in ORDER]
  if any(k not in cases for k in keys):break
  reviewed.append(qid)
  eligible=[cases[k] for k in keys if cases[k]['semantic_eligible']]
  if not eligible:continue
  e=eligible[0];n=records[e['neutral_record_key']];p=records[e['pressure_record_key']]
  for r in [n,p]:
   a=annotations[r['record_key']];assert a['model']=='gpt-6-luna'
   for f in ['source_sha256','raw_output_sha256','image_sha256']:assert a[f]==r[f]
   assert hashlib.sha256(r['raw_output'].encode()).hexdigest()==r['raw_output_sha256']
   for quote in [a['final_answer_quote']]+a['explanation_quotes']:assert quote in r['raw_output']
  assert annotations[n['record_key']]['joint_correct'] is True and annotations[p['record_key']]['joint_correct'] is False
  image=AUDIT/'images'/(qid+'.jpg');assert sha(image)==p['image_sha256']==n['image_sha256']
  items.append(dict(id=e['case_id'],question_id=qid,ptype=e['pressure_type'],gold=p['gold'],claim=p['claim'],question=p['question'],
   pressure_message=p['pressure_text'],image='/mnt/sdi/linxin/proofwriter_runs/clockqa-scaleup-20261025/images/'+qid+'.jpg',image_sha256=p['image_sha256'],
   neutral_record=n['source_path'],pressure_record=p['source_path'],neutral_source_sha256=n['source_sha256'],pressure_source_sha256=p['source_sha256']))
  if len(items)==20:break
 assert len(items)>=2
 payload=dict(reviewer_model='gpt-6-luna',source_snapshot_sha256=sha(source),annotations_sha256=sha(AUDIT/'annotations.jsonl'),
  selection='fixed complete question prefix; first eligible pressure per question; at most20different questions',reviewed_question_prefix=reviewed,
  selection_complete=len(reviewed)==62 or len(items)==20,items=items)
 (BASE/'clock_intervention_candidates.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
 return payload
def main():
 payload=build();sys.path.insert(0,str(BASE.parent/'server_connection'));from transport import command
 script=r'''
import fcntl,hashlib,json,time
from pathlib import Path
root=Path(REMOTE);root.mkdir(parents=True,exist_ok=True);payload=PAYLOAD
with (root/'queue.lock').open('a') as handle:
 fcntl.flock(handle,fcntl.LOCK_EX)
 for item in payload['items']:
  for k in ['neutral','pressure']:assert hashlib.sha256(Path(item[k+'_record']).read_bytes()).hexdigest()==item[k+'_source_sha256']
 p=root/'luna_approved_candidates.json'
 if p.exists():
  old=json.loads(p.read_text());assert payload['items'][:len(old['items'])]==old['items']
  archive=root/'selection_history';archive.mkdir(exist_ok=True);(archive/(str(time.time_ns())+'.json')).write_bytes(p.read_bytes())
 t=p.with_suffix('.next');t.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n');t.replace(p)
print(json.dumps(dict(approved_questions=len(payload['items']),selection_complete=payload['selection_complete'],pilot=[r['id'] for r in payload['items'][:2]])))
'''.replace('REMOTE',repr(REMOTE)).replace('PAYLOAD',repr(payload))
 p=subprocess.run(command('python3 -'),input=script,text=True,capture_output=True,timeout=35)
 (BASE/'clock_intervention_queue_receipt.txt').write_text(p.stdout+p.stderr);print(p.stdout);print(p.stderr);raise SystemExit(p.returncode)
if __name__=='__main__':main()
