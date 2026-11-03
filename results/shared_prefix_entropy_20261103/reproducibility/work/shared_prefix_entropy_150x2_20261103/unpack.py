import base64,hashlib,io,json,tarfile
from pathlib import Path
B=Path(__file__).resolve().parent
for host,model in [('59','qwen38'),('50','gemma4')]:
 O=B/'results'/model
 if (O/'analysis/summary.json').exists():continue
 x=json.loads((B/f'results{host}_package.json').read_text());b=base64.b64decode(x['archive_b64']);assert hashlib.sha256(b).hexdigest()==x['archive_sha256'];O.mkdir(parents=True,exist_ok=True)
 with tarfile.open(fileobj=io.BytesIO(b),mode='r:gz') as t:
  for m in t.getmembers():assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
  t.extractall(O)
 print('Extracted',model,len(b))
for model in ['qwen38','gemma4']:
 s=json.loads((B/'results'/model/'analysis/summary.json').read_text());print(model,'methods',s['methods'],'tokens',s['tokens'])
 for ds in ['ClockQA','PathVQA','PENDULUM']:
  for scope in ['after64','after_opening']:
   rs=[next(r for r in s['groups'] if (r['dataset'],r['group'],r['condition'],r['scope'])==(ds,g,'belief',scope)) for g in ['flip','hold']]
   print(ds,scope,[(r['group'],r['eligible'],round(r['means']['delta_H'],5) if r['eligible'] else None) for r in rs])
