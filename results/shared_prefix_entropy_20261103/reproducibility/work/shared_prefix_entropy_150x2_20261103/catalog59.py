import json,hashlib
from pathlib import Path
out={}
for ds,root in [('ClockQA','/mnt/sdb/linxin/conflict-cq-20261028'),('PathVQA','/mnt/sdb/linxin/qwen38-pv-20261031'),('PENDULUM','/mnt/sdb/linxin/pendulum-20261102/expand4x50')]:
 R=Path(root);items=list(map(json.loads,(R/'items.jsonl').read_text().splitlines()));records=[]
 for it in items:
  if ds=='PENDULUM' and it['domain'] not in ['re_used','self_captured']:continue
  for cond in ['neutral','belief','belief_right']:
   p=R/('qwen38/records' if ds=='PENDULUM' else 'obs/records')/f"{it['id']}__{cond}.json";r=json.loads(p.read_text());records.append(dict(id=it['id'],condition=cond,source=str(p),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),model=r.get('model'),n_out=r['n_out'],image_sha256=r['image_sha256'],has_integer_ids=bool(r.get('input_ids')) and bool(r.get('output_ids')),budget=r['budget']))
 out[ds]=dict(root=str(R),items=items,records=records)
for name,p in [('clockqa','/mnt/sdb/linxin/compare-cq-20261029-a100/time_judge/judgments.json'),('pathvqa','/mnt/sdb/linxin/qwen38-pathvqa-compare-20261031/yesno_judge/judgments.json')]:
 out['judgments_'+name]=json.loads(Path(p).read_text())
print(json.dumps(out,ensure_ascii=False))
