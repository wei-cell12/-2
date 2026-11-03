import json,subprocess
from pathlib import Path
W=Path('/mnt/sdg/linxin/shared-prefix-entropy-150x2-20261103')
receipts=json.loads((W/'launch_receipt.json').read_text());out=[]
for q in receipts:
 rank=q['rank'];p=W/f'state_{rank}.json';s=json.loads(p.read_text()) if p.exists() else {};failed=W/f'failure_{rank}.json'
 out.append(dict(receipt=q,state=s,pid_exists=Path('/proc',str(q['pid'])).exists(),failure=json.loads(failed.read_text()) if failed.exists() else None,log_tail=(W/q.get('log',f'run_{rank}.log')).read_text()[-1600:]))
print(json.dumps(dict(workers=out,completed=len(list((W/'records').glob('*.json')))),ensure_ascii=False))
