"""Finite observation queue, with optional same-Luna-approved intervention bundles."""
import contextlib,fcntl,hashlib,json,os,time
from pathlib import Path
PARENT=Path(os.environ.get('PATHVQA_PARENT_ROOT',Path(__file__).resolve().parent))

def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp.'+str(os.getpid()))
    t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');t.replace(p)

@contextlib.contextmanager
def locked():
    with (PARENT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);q=json.loads((PARENT/'queue.json').read_text())
        yield q
        q['time_unix']=time.time();write(PARENT/'queue.json',q)

def review_valid():
    p=PARENT/'pilot_review.json';g=PARENT/'pilot_numerical_gate.json'
    if not p.exists() or not g.exists():return False
    review=json.loads(p.read_text());gate=json.loads(g.read_text())
    assert review['ok'] and review['pilot_gate_sha256']==hashlib.sha256(g.read_bytes()).hexdigest()
    assert len(review['read_replies'])==6
    for row in gate['rows']:
        digest=hashlib.sha256(row['raw_output'].encode()).hexdigest()
        assert any(x['id']==row['id'] and x['condition']==row['method'] and x['raw_output_sha256']==digest for x in review['read_replies'])
    return True

def claim(worker):
    with locked() as q:
        if q.get('failure'):raise RuntimeError('Saved failure requires diagnosis: '+str(q['failure']))
        # Approved candidates may be appended while native generation continues.
        approved=PARENT/'luna_approved_candidates.json'
        if approved.exists():
            candidates=json.loads(approved.read_text())
            assert candidates['reviewer_model']=='gpt-6-luna'
            rows=candidates['items'];assert len({x['question_id'] for x in rows})==len(rows)<=20
            for index,item in enumerate(rows):
                key=item['id'];entry=q.setdefault('interventions',{}).setdefault(key,dict(status='pending',item=item,pilot=index<2))
                if entry['status']!='pending' or (index>=2 and not review_valid()):continue
                for path,expected in [(item['neutral_record'],item['neutral_source_sha256']),(item['pressure_record'],item['pressure_source_sha256'])]:
                    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected
                entry.update(status='running',owner=worker)
                return dict(task='intervene',id=key,item=item,pilot=index<2)
        for key in q['order']:
            entry=q['entries'][key]
            if entry['status']!='pending':continue
            dep=entry.get('requires')
            if dep and q['entries'][dep]['status']!='complete':continue
            entry.update(status='running',owner=worker)
            return dict(task='observe',id=key)
        if any(e['status']=='pending' for e in q['entries'].values()):return dict(task='wait')
        return dict(task='finish')

def complete(worker,key,path):
    with locked() as q:
        e=q['entries'][key];assert e['owner']==worker and e['status']=='running'
        e.update(status='complete',source_record=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def intervention_complete(worker,key,root):
    with locked() as q:
        e=q['interventions'][key];assert e['owner']==worker and e['status']=='running'
        e.update(status='complete',worker_root=str(root))
        pilot=[row for row in q['interventions'].values() if row['pilot']]
        if len(pilot)==2 and all(row['status']=='complete' for row in pilot) and not (PARENT/'pilot_numerical_gate.json').exists():
            rows=[]
            for x in pilot:
                for condition in ['base','restore_E','restore_rand']:
                    path=Path(x['worker_root'])/'runs'/condition/'final.jsonl'
                    found=[json.loads(line) for line in path.read_text().splitlines() if json.loads(line)['id']==x['item']['id']]
                    assert len(found)==1;row=found[0];m=row['metrics']
                    assert all(m[k]<1e-3 for k in ['image_diff_max','text_diff_max','row_diff_max'])
                    if condition=='base':assert len(m['selfchecks'])==48 and max(z['max_abs_diff'] for z in m['selfchecks'])<.05
                    else:assert m['active_head_tokens']>0 and m['region_after']>m['region_before']
                    assert row['normality_checks']==dict(nonempty=True,replacement_character=False,repeated_sentence=False)
                    rows.append(row)
            write(PARENT/'pilot_numerical_gate.json',dict(ok=True,rows=rows,manual_read_required=True))

def fail(worker,error):
    with locked() as q:q['failure']=dict(worker=worker,error=repr(error),time_unix=time.time())

def finished(worker):
    with locked() as q:
        q.setdefault('exited_workers',[]).append(worker)
        if len(q['exited_workers'])==q['workers']:
            assert all(e['status']=='complete' for e in q['entries'].values())
            write(PARENT/'job_state.json',dict(status='observations_complete_check_luna_and_interventions',
                records=len(q['entries']),all_model_processes_finishing=True,interventions=q.get('interventions',{}),time_unix=time.time()))
