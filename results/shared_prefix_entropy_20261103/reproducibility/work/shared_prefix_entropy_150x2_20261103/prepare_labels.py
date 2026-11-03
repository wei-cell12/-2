import json,tarfile,hashlib,collections
from pathlib import Path
B=Path(__file__).resolve().parent;sel=json.loads((B/'selection.json').read_text());cat=json.loads((B/'catalog59.json').read_text());out=[]
with tarfile.open(B/'gemma_sources.tar.gz','r:gz') as t:gcq=json.load(t.extractfile('judgments/ClockQA.json'))
gpv=json.loads(Path('work/gemma4_pv_server51_20261030/publication/results/gemma4_pathvqa_20261030/judgments.json').read_text());pc=Path('work/pendulum_server51_20261102');pdq=json.loads((pc/'qwen_results_20261102/judgments.json').read_text());pdg=json.loads((pc/'three_model_results_20261103/gemma4_judgments.json').read_text());srcs=json.loads((B/'gemma_source_manifest.json').read_text())
models={'qwen38':{'ClockQA':cat['judgments_clockqa'],'PathVQA':cat['judgments_pathvqa'],'PENDULUM':pdq},'gemma4':{'ClockQA':gcq,'PathVQA':gpv,'PENDULUM':pdg}}
for model,datasets in models.items():
 for ds,jud in datasets.items():
  if ds=='ClockQA':
   lookup={(iid,c):jud[f'obs|{iid}|{c}'] for iid in sel['ids'][ds] for c in ['neutral','belief','belief_right']}
  elif ds=='PathVQA':lookup={(r['id'],r['condition']):r for r in jud if r['arm']=='base'}
  else:lookup={(r['id'],r['condition']):r for r in jud}
  for iid in sel['ids'][ds]:
   for cond in ['neutral','belief','belief_right']:
    r=lookup[iid,cond]
    if ds=='ClockQA':status='correct' if r['outcome']=='坚持（答对）' else 'unresolved' if r['outcome']=='未解析' else 'wrong'
    elif ds=='PathVQA':
     if 'answer_matches_gold' in r:status='correct' if r['answer_matches_gold'] is True else 'wrong' if r['answer_matches_gold'] is False else 'unresolved'
     else:status={'对':'correct','错':'wrong'}.get(r['status'],'unresolved')
    else:status=r['status']
    source=next(q for q in (cat[ds]['records'] if model=='qwen38' else srcs[ds]['records']) if q['id']==iid and q['condition']==cond)
    sha=r.get('record_sha256',r.get('source_sha256',r.get('original_record_sha256')))
    if sha:assert sha==source['source_sha256'],(model,ds,iid,cond,sha,source['source_sha256'])
    out.append(dict(model=model,dataset=ds,id=iid,condition=cond,status=status,source_sha256=source['source_sha256'],judgment_source_sha_present=bool(sha),judgment=r))
(B/'labels_snapshot.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
for model in models:
 for ds in models[model]:
  d={(r['id'],r['condition']):r['status'] for r in out if r['model']==model and r['dataset']==ds};counts=collections.Counter((d[i,'neutral'],d[i,'belief']) for i in sel['ids'][ds]);print(model,ds,dict(counts))
print('rows',len(out))
