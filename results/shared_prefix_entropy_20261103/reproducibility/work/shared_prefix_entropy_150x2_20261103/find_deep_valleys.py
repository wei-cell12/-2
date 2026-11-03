"""Local raw delta-H excursions. Counts are candidate points, not semantic events."""
import collections,csv,hashlib,json,statistics
from pathlib import Path
B=Path(__file__).resolve().parent;O=B/'results';V=O/'local_valleys';V.mkdir(exist_ok=True)
all_events=[];trajectories=[]
for model in ['qwen38','gemma4']:
 R=O/model;cases=json.loads((R/'analysis/cases.json').read_text())
 for c in cases:
  rows=list(csv.DictReader((R/'records'/(c['key']+'.csv')).open()));eligible=[i for i,r in enumerate(rows) if i>=c['opening_cutoff'] and r['phase']=='think' and r['forced_by_original_budget']=='False'];allowed=set(eligible);events=[];k=0
  text=''.join(r['token_piece'] for r in rows);pos=[0]
  for r in rows:pos.append(pos[-1]+len(r['token_piece']))
  while k<len(rows):
   if k not in allowed or float(rows[k]['delta_H'])>=0:k+=1;continue
   a=k
   while k<len(rows) and k in allowed and float(rows[k]['delta_H'])<0:k+=1
   b=k;mn=min(range(a,b),key=lambda q:float(rows[q]['delta_H']));r=rows[mn];value=float(r['delta_H'])
   if value>-1:continue
   events.append(dict(model=model,dataset=c['dataset'],id=c['id'],condition=c['condition'],group=c['group'],key=c['key'],negative_run_start=a+1,negative_run_end=b,step=mn+1,token=r['token_piece'],H_with_U=float(r['H_pressure']),H_without_U=float(r['H_neutral']),delta_H=value,JSD=float(r['JSD']),before=text[max(0,pos[mn]-200):pos[mn]],after=text[pos[mn+1]:pos[mn+1]+200],source_sha256=c['source_sha256']))
  all_events.extend(events);trajectories.append(dict(model=model,dataset=c['dataset'],id=c['id'],condition=c['condition'],group=c['group'],eligible_tokens=len(eligible),candidate_deep_valleys=len(events),minimum_delta_H=min((float(rows[i]['delta_H']) for i in eligible),default=None)))
summary=dict(definition='Raw per-token delta_H=H_with_U-H_without_U, nats. Exclude first substantive paragraph and at least64output tokens; ordinary thinking tokens only, no original budget-forced tokens. Each maximal contiguous negative run contributes its deepest token if delta_H<=-1. One user-reference event may include multiple runs; these are candidate valleys, not semantically labeled events. Offline extrema use future context and are not themselves an online trigger.',trajectory_counts=[])
for m in ['qwen38','gemma4']:
 for ds in ['ClockQA','PathVQA','PENDULUM']:
  rr=[r for r in trajectories if r['model']==m and r['dataset']==ds];ev=[r for r in all_events if r['model']==m and r['dataset']==ds]
  summary['trajectory_counts'].append(dict(model=m,dataset=ds,total=100,eligible=sum(r['eligible_tokens']>0 for r in rr),with_valley_le_minus1=sum(r['candidate_deep_valleys']>0 for r in rr),candidate_runs=len(ev),depth_1_to_2=sum(-2<r['delta_H']<=-1 for r in ev),depth_2_to_3=sum(-3<r['delta_H']<=-2 for r in ev),depth_ge3=sum(r['delta_H']<=-3 for r in ev),minimum=min((r['delta_H'] for r in ev),default=None)))
(V/'candidates.json').write_text(json.dumps(all_events,ensure_ascii=False,indent=2));(V/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));(V/'trajectories.json').write_text(json.dumps(trajectories,ensure_ascii=False,indent=2))
with (V/'candidates.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(all_events[0]));w.writeheader();w.writerows(all_events)
for r in summary['trajectory_counts']:print(r)
print('Total candidate runs',len(all_events))
