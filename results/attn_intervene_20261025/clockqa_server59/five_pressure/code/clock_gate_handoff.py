"""Fetch exact six pilot replies; write approval only after actual manual reading."""
import hashlib,json,subprocess,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'server_connection'));from transport import command
root='/mnt/sdi/linxin/proofwriter_runs/clockqa-scaleup-20261025/interventions'
if len(sys.argv)==1:
 script="from pathlib import Path\nimport hashlib,json\nr=Path("+repr(root)+")\np=r/'pilot_numerical_gate.json'\nprint(json.dumps(dict(ready=p.exists(),gate=json.loads(p.read_text()) if p.exists() else None,gate_sha256=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None)))\n"
 p=subprocess.run(command('python3 -'),input=script,text=True,capture_output=True,timeout=35)
 if p.returncode:raise RuntimeError(p.stderr)
 x=json.loads(p.stdout)
 if not x['ready']:print('pilot_not_ready');raise SystemExit(0)
 (BASE/'clock_pilot_review_source.json').write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
 for r in x['gate']['rows']:print('\nCASE',r['id'],r['method'],'\n'+r['raw_output'])
 print('GATE_SHA256',x['gate_sha256'])
else:
 assert sys.argv[1]=='approve-after-reading'
 x=json.loads((BASE/'clock_pilot_review_source.json').read_text());assert x['ready'] and x['gate']['ok'] and len(x['gate']['rows'])==6
 rows=x['gate']['rows'];assert len({(r['id'],r['method']) for r in rows})==6
 for r in rows:assert r['normality_checks']==dict(nonempty=True,replacement_character=False,repeated_sentence=False)
 reviewer=sys.argv[2] if len(sys.argv)>2 else 'root'
 review=dict(ok=True,pilot_gate_sha256=x['gate_sha256'],reviewer=reviewer,review_scope='actual full raw output read for normality; no answer-correctness requirement',
  read_replies=[dict(id=r['id'],condition=r['method'],raw_output_sha256=hashlib.sha256(r['raw_output'].encode()).hexdigest(),read=True,nonempty=True,no_replacement_character=True,no_pathological_repetition=True) for r in rows])
 script=r'''
import hashlib,json,time
from pathlib import Path
root=Path(ROOT);review=REVIEW;gate=root/'pilot_numerical_gate.json'
assert hashlib.sha256(gate.read_bytes()).hexdigest()==review['pilot_gate_sha256']
assert not json.loads((root/'queue.json').read_text()).get('failure')
for r in json.loads(gate.read_text())['rows']:
 assert any(x['id']==r['id'] and x['condition']==r['method'] and x['raw_output_sha256']==hashlib.sha256(r['raw_output'].encode()).hexdigest() for x in review['read_replies'])
p=root/'pilot_review.json';t=p.with_suffix('.next');t.write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n');t.replace(p)
(root/'job_state.json').write_text(json.dumps(dict(status='full_intervention_running_same_four_models',pilot_review_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),time_unix=time.time()),indent=2)+'\n')
print(json.dumps(dict(approved=True,six_raw_outputs_sha256_bound=True,models_continue_without_restart=True)))
'''.replace('ROOT',repr(root)).replace('REVIEW',repr(review))
 p=subprocess.run(command('python3 -'),input=script,text=True,capture_output=True,timeout=35)
 if p.returncode:raise RuntimeError(p.stderr)
 (BASE/'clock_pilot_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n');print(p.stdout)
