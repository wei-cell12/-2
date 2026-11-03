import csv,gzip,hashlib,json
from pathlib import Path
B=Path(__file__).resolve().parent;O=B/'results/local_valleys';R=B.parent/'clockqa_uncertainty_attention_20261101/results/records';events=[]
for iid,center in [('CQ-003',213),('CQ-009',416)]:
 p=R/f'{iid}--belief__wrong.jsonl.gz'
 with gzip.open(p,'rt') as f:a=[json.loads(l) for l in f]
 q=B/'results/qwen38/records'/f'{iid}__belief.csv';rs=list(csv.DictReader(q.open()));assert len(a)==len(rs) and all(x['token']==r['token_piece'] for x,r in zip(a,rs))
 window=[]
 for k in range(max(0,center-25),min(len(rs),center+25)):
  r=rs[k];h=a[k]['all_heads'];window.append(dict(step=k+1,token=r['token_piece'],H_with_U=float(r['H_pressure']),H_without_U=float(r['H_neutral']),delta_H=float(r['delta_H']),JSD=float(r['JSD']),pressure_attention_percent=h['pressure_mass']*100,image_attention_percent=h['image_mass']*100))
 events.append(dict(id=iid,condition='belief',center=center,rows=window,attention_source=str(p.resolve()),attention_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),entropy_source=str(q.resolve()),entropy_sha256=hashlib.sha256(q.read_bytes()).hexdigest(),all_output_tokens_and_length_equal=True,alignment='Both record query at prompt_length-1+k to predict output[k], before actual token emission; code verified in measure.py. Attention uses16standard layers x24heads=384heads. pressure mass = original prompt span after question, including user claim and appended request; not just time tokens. Native attention source replay is a separate saved replay, not measured in same entropy run.'))
(O/'two_cases_attention_alignment.json').write_text(json.dumps(events,ensure_ascii=False,indent=2));print('Verified2token-aligned saved attention trajectories; no GPU')
