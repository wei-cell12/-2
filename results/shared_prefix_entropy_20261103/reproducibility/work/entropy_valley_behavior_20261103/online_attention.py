import csv,gzip,json,statistics,collections,hashlib
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parent;O=B.parent/'shared_prefix_entropy_150x2_20261103/results';R=B.parent/'clockqa_uncertainty_attention_20261101/results/records'
points=[];alerts=[]
for c in json.loads((O/'qwen38/analysis/cases.json').read_text()):
 if c['dataset']!='ClockQA':continue
 rows=list(csv.DictReader((O/'qwen38/records'/(c['key']+'.csv')).open()))
 p=R/(c['id']+'--belief__'+('wrong' if c['condition']=='belief' else 'correct')+'.jsonl.gz')
 with gzip.open(p,'rt') as f:a=[json.loads(l) for l in f]
 assert len(a)==len(rows) and all(r['token_piece']==x['token'] for r,x in zip(rows,a))
 valid={i for i,r in enumerate(rows) if i>=c['opening_cutoff'] and r['phase']=='think' and r['forced_by_original_budget']=='False'}
 u=np.array([x['all_heads']['pressure_mass']*100 for x in a]);im=np.array([x['all_heads']['image_mass']*100 for x in a]);crossed=set()
 for i,r in enumerate(rows):
  if i not in valid:crossed=set();continue
  d=float(r['delta_H']);prev=[j for j in range(max(0,i-8),i) if j in valid]
  if d>=0:crossed=set()
  if len(prev)<4:continue
  z=dict(model='qwen38',dataset='ClockQA',id=c['id'],key=c['key'],condition=c['condition'],group=c['group'],image=c['image_sha256'],step=i+1,delta_H=d,token=r['token_piece'],user_jump=float(u[i]-u[prev].mean()),image_jump=float(im[i]-im[prev].mean()),user_mass=float(u[i]),image_mass=float(im[i]))
  points.append(z)
  for th in [1,.5,.25]:
   if d<=-th and th not in crossed:alerts.append(dict(**z,threshold=th));crossed.add(th)
def summarize(rr):
 imgs=sorted({r['image'] for r in rr})
 if not imgs:return None
 def avg(f):return statistics.mean(statistics.mean(f(r) for r in rr if r['image']==im) for im in imgs)
 return dict(n=len(rr),trajectories=len({r['key'] for r in rr}),images=len(imgs),mean_user_jump_pp=avg(lambda r:r['user_jump']),mean_image_jump_pp=avg(lambda r:r['image_jump']),user_jump_ge2pp=avg(lambda r:r['user_jump']>=2),mean_user_mass_pp=avg(lambda r:r['user_mass']))
s=dict(definition='Online:first crossing of deltaH<=-threshold per maximal negative excursion. No future minima, depth ranks, text or attention used in trigger. Attention outcome is simultaneous original input user-span mass minus prior8eligible-token average; unit percentage points. Image-weighted means. Results descriptive on QwenClockQA50images100trajectories.',thresholds={str(th):summarize([r for r in alerts if r['threshold']==th]) for th in [1,.5,.25]},subgroups={f'{cond}:{g}':summarize([r for r in alerts if r['threshold']==1 and r['condition']==cond and (g=='all' or r['group']==g)]) for cond in ['belief','belief_right'] for g in ['all','flip','hold']},ordinary=summarize([r for r in points if abs(r['delta_H'])<=.05]))
# Recompute strict same-token matching for already frozen offline valley/control pairs.
pairs=json.loads((B/'attention_pairs.json').read_text());s['same_token_only']={}
for band in ['deep','medium','shallow']:
 rr=[r for r in pairs if r['band']==band and r['match']=='same_token'];imgs=sorted({r['image_sha256'] for r in rr})
 def avg(f):return statistics.mean(statistics.mean(f(r) for r in rr if r['image_sha256']==im) for im in imgs)
 s['same_token_only'][band]=dict(n=len(rr),images=len(imgs),valley_user_jump_pp=avg(lambda r:r['attention']['user_jump']),control_user_jump_pp=avg(lambda r:r['control_attention']['user_jump']),paired_difference_pp=avg(lambda r:r['paired_user_jump_difference']),valley_jump_ge2pp=avg(lambda r:r['attention']['user_jump']>=2),control_jump_ge2pp=avg(lambda r:r['control_attention']['user_jump']>=2))
(B/'online_attention_summary.json').write_text(json.dumps(s,ensure_ascii=False,indent=2));(B/'online_alerts.json').write_text(json.dumps(alerts,ensure_ascii=False));print(json.dumps(s,ensure_ascii=False,indent=2))
