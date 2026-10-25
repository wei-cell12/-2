"""Same-model seamless queue: two-question gate, then distinct approved cases."""
import contextlib,fcntl,hashlib,json,os,time
from pathlib import Path
PARENT=Path(os.environ['CLOCKQA_INTERVENTION_ROOT'])
def write(p,x):
 t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');t.replace(p)
@contextlib.contextmanager
def locked():
 with (PARENT/'queue.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);q=json.loads((PARENT/'queue.json').read_text());yield q;q['updated_unix']=time.time();write(PARENT/'queue.json',q)
def review_valid():
 p=PARENT/'pilot_review.json';g=PARENT/'pilot_numerical_gate.json'
 if not p.exists() or not g.exists():return False
 r=json.loads(p.read_text());gate=json.loads(g.read_text())
 assert r['ok'] and r['pilot_gate_sha256']==hashlib.sha256(g.read_bytes()).hexdigest() and len(r['read_replies'])==6
 for row in gate['rows']:
  h=hashlib.sha256(row['raw_output'].encode()).hexdigest()
  assert any(x['id']==row['id'] and x['condition']==row['method'] and x['raw_output_sha256']==h for x in r['read_replies'])
 return True
def claim(worker):
 with locked() as q:
  if q.get('failure'):raise RuntimeError('Preserved failure requires diagnosis: '+str(q['failure']))
  selection=json.loads((PARENT/'luna_approved_candidates.json').read_text());items=selection['items'];assert len({x['question_id'] for x in items})==len(items)<=20
  for i,item in enumerate(items):
   e=q['interventions'].setdefault(item['id'],dict(status='pending',item=item,pilot=i<2))
   if e['status']!='pending':continue
   for kind in ['neutral','pressure']:assert hashlib.sha256(Path(item[kind+'_record']).read_bytes()).hexdigest()==item[kind+'_source_sha256']
   if i>=2 and not review_valid():continue
   if e.get('reference_owner') is not None and not e.get('reference_ready'):continue
   e.update(status='running',owner=worker);return dict(task='intervene',item=item)
  # Extra cards can prepare neutral references while the two-question generation gate runs.
  for i,item in enumerate(items):
   e=q['interventions'].get(item['id'])
   if i>=2 and e and e['status']=='pending' and e.get('reference_owner') is None:
    e['reference_owner']=worker;return dict(task='reference',item=item)
  if selection['selection_complete'] and all(e['status']=='complete' for e in q['interventions'].values()):return dict(task='finish')
  return dict(task='wait')
def reference_complete(worker,key):
 with locked() as q:
  e=q['interventions'][key];assert e['reference_owner']==worker;e['reference_ready']=True
def complete(worker,key,root):
 with locked() as q:
  e=q['interventions'][key];assert e['status']=='running' and e['owner']==worker;e.update(status='complete',worker_root=str(root))
  pilots=[e for e in q['interventions'].values() if e['pilot']]
  if len(pilots)==2 and all(e['status']=='complete' for e in pilots) and not (PARENT/'pilot_numerical_gate.json').exists():
   rows=[]
   for e in pilots:
    for method in ['base','restore_E','restore_rand']:
     found=[json.loads(s) for s in (Path(e['worker_root'])/'runs'/method/'final.jsonl').read_text().splitlines() if json.loads(s)['id']==e['item']['id']];assert len(found)==1
     r=found[0];m=r['metrics'];assert all(m[k]<1e-3 for k in ['image_diff_max','text_diff_max','row_diff_max'])
     if method=='base':assert len(m['selfchecks'])==48 and max(x['max_abs_diff'] for x in m['selfchecks'])<.05
     else:assert m['active_head_tokens']>0 and m['region_after']>m['region_before']
     assert r['normality_checks']==dict(nonempty=True,replacement_character=False,repeated_sentence=False);rows.append(r)
   write(PARENT/'pilot_numerical_gate.json',dict(ok=True,rows=rows,manual_read_required=True))
   write(PARENT/'job_state.json',dict(status='pilot_ready_for_review_models_kept_for_handoff',time_unix=time.time()))
def fail(worker,error):
 with locked() as q:q['failure']=dict(worker=worker,error=repr(error),time_unix=time.time())
def finished(worker,reason):
 with locked() as q:
  q.setdefault('exited_workers',{})[str(worker)]=reason
  if len(q['exited_workers'])==q['workers']:write(PARENT/'job_state.json',dict(status='workers_exited',reasons=q['exited_workers'],time_unix=time.time()))
