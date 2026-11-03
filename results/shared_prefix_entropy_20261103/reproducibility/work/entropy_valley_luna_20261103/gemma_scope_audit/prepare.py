import csv,hashlib,json,random
from pathlib import Path
B=Path(__file__).resolve().parent;P=B.parent;O=P.parent/'shared_prefix_entropy_150x2_20261103/results';oldreq={r['key']:r for p in sorted((P/'blind').glob('requests_*.jsonl')) for r in map(json.loads,p.read_text().splitlines())};ev=json.loads((O/'local_valleys/luna_generalization/events.json').read_text());text=json.loads((O/'gemma4/analysis/source_texts.json').read_text());qs=[];mapping={}
if (B/'private_keymap.json').exists():raise RuntimeError('Already prepared')
for r in ev:
 if r['model']!='gemma4' or r['dataset'] not in ['ClockQA','PENDULUM']:continue
 old=r['blind_key'];q=oldreq[old];key='audit_'+hashlib.sha256(('gemma_scope_20261103:'+old).encode()).hexdigest()[:22];qs.append(dict(key=key,message=text[r['key']]['condition']['message'],before=q['before'],token=q['token'],after=q['after']));mapping[key]=dict(original_blind_key=old,event=r)
assert len(qs)==152;random.Random(20261103).shuffle(qs)
for i,a in enumerate(range(0,len(qs),25),1):(B/'blind'/f'requests_{i:02d}.jsonl').write_text(''.join(json.dumps(q,ensure_ascii=False)+'\n' for q in qs[a:a+25]))
(B/'private_keymap.json').write_text(json.dumps(mapping,ensure_ascii=False,indent=2));(B/'state.json').write_text(json.dumps(dict(phase='prepared',count=152,purpose='Separate focal role from object: user information/format/image. Consistency audit; original labels retained.'),indent=2));print('Prepared152scope-audit windows')
