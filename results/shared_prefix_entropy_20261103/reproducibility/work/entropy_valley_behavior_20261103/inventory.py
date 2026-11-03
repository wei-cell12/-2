import csv,json,collections,hashlib,statistics
from pathlib import Path
B=Path(__file__).resolve().parent;O=B.parent/'shared_prefix_entropy_150x2_20261103/results';ev=[];tr=[]
for model in ['qwen38','gemma4']:
 for c in json.loads((O/model/'analysis/cases.json').read_text()):
  p=O/model/'records'/(c['key']+'.csv');rs=list(csv.DictReader(p.open()));valid={i for i,r in enumerate(rs) if i>=c['opening_cutoff'] and r['phase']=='think' and r['forced_by_original_budget']=='False'};x=[float(r['delta_H']) for r in rs];k=0;trajectory=[]
  while k<len(rs):
   if k not in valid or x[k]>=0:k+=1;continue
   a=k
   while k<len(rs) and k in valid and x[k]<0:k+=1
   b=k;i=min(range(a,b),key=lambda j:x[j]);r=rs[i];v=-x[i]
   band='deep' if v>=1 else 'medium' if v>=.25 else 'shallow' if v>=.05 else 'tiny'
   before=[j for j in range(max(0,i-8),i) if j in valid];mu=statistics.mean(float(rs[j]['H_pressure']) for j in before) if before else None;m0=statistics.mean(float(rs[j]['H_neutral']) for j in before) if before else None
   d=dict(model=model,dataset=c['dataset'],id=c['id'],key=c['key'],condition=c['condition'],group=c['group'],image_sha256=c['image_sha256'],source_sha256=c['source_sha256'],opening_cutoff=c['opening_cutoff'],step=i+1,run_start=a+1,run_end=b,run_width=b-a,band=band,depth=v,token=r['token_piece'],H_with_U=float(r['H_pressure']),H_without_U=float(r['H_neutral']),delta_H=x[i],JSD=float(r['JSD']),delta_logp=float(r['delta_logp']),prior8_H_with_U=mu,prior8_H_without_U=m0,drop_with_U=mu-float(r['H_pressure']) if mu is not None else None,rise_without_U=float(r['H_neutral'])-m0 if m0 is not None else None)
   trajectory.append(d)
  for rank,r in enumerate(sorted(trajectory,key=lambda r:r['depth'],reverse=True),1):r['depth_rank']=rank
  ev.extend(trajectory);tr.append(dict(model=model,dataset=c['dataset'],key=c['key'],id=c['id'],image_sha256=c['image_sha256'],eligible=len(valid),bands=dict(collections.Counter(r['band'] for r in trajectory))))
(B/'valleys_all.json').write_text(json.dumps(ev,ensure_ascii=False,indent=2));(B/'trajectories.json').write_text(json.dumps(tr,ensure_ascii=False,indent=2))
summary=[]
for model in ['qwen38','gemma4']:
 for ds in ['ClockQA','PathVQA','PENDULUM']:
  rr=[r for r in ev if r['model']==model and r['dataset']==ds];summary.append(dict(model=model,dataset=ds,counts=dict(collections.Counter(r['band'] for r in rr)),meaningful_trajectories=len({r['key'] for r in rr if r['band']!='tiny'})))
(B/'inventory.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False))
