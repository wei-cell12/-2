"""CPU-only aggregation, no automatic semantic valley annotation."""
import bisect,collections,csv,hashlib,json,statistics
from pathlib import Path
import numpy as np
W=Path(__file__).resolve().parent
M=json.loads((W/'manifest.json').read_text());model=M['model'];labels=json.loads((W/'labels_snapshot.json').read_text());J={(r['dataset'],r['id'],r['condition']):r for r in labels if r['model']==model}
O=W/'analysis';O.mkdir(exist_ok=True)
allcases=[];texts={};valleys=[]
fields=['H_pressure','H_neutral','delta_H','JSD','delta_logp']
def analyze_window(rows,indices):
 if not indices:return dict(n=0,means=None,min_token=None,min16=None)
 inds=set(indices);a={f:[float(rows[k][f]) for k in indices] for f in fields}
 mn=min(indices,key=lambda k:float(rows[k]['delta_H']));windows=[]
 for end in indices:
  start=end-15
  if start>=0 and all(k in inds for k in range(start,end+1)):windows.append((statistics.mean(float(rows[k]['delta_H']) for k in range(start,end+1)),start,end))
 return dict(n=len(indices),means={f:statistics.mean(a[f]) for f in fields},q05_delta=float(np.quantile(a['delta_H'],.05)),min_token=dict(step=mn+1,token=rows[mn]['token_piece'],value=float(rows[mn]['delta_H'])),min16=(dict(start=min(windows)[1]+1,end=min(windows)[2]+1,value=min(windows)[0]) if windows else None),positive_fraction=sum(x>0 for x in a['delta_H'])/len(indices))
for job in M['jobs']:
 key=job['key'];p=W/'records'/(key+'.json');assert p.exists(),key;c=json.loads(p.read_text());rows=list(csv.DictReader((W/'records'/(key+'.csv')).open()));assert len(rows)==job['n_out']==c['tokens'];assert c['source_sha256']==job['source_sha256']
 rec=json.loads(Path(job['source']).read_text());nr=json.loads(Path(job['neutral_source']).read_text());texts[key]={cond:{k:r.get(k,'') for k in ['message','thinking','answer']} for cond,r in [('condition',rec),('original_neutral',nr)]}
 lab={cond:J[job['dataset'],job['id'],cond] for cond in ['neutral','belief','belief_right']}
 for cond,j in lab.items():assert j['source_sha256']==(job['neutral_sha256'] if cond=='neutral' else next(x['source_sha256'] for x in M['jobs'] if x['id']==job['id'] and x['condition']==cond))
 nstatus=lab['neutral']['status'];bstatus=lab['belief']['status'];group='flip' if nstatus=='correct' and bstatus=='wrong' else 'hold' if nstatus=='correct' and bstatus=='correct' else 'pressure_unresolved' if nstatus=='correct' else 'neutral_wrong' if nstatus=='wrong' else 'neutral_unresolved'
 valid=[i for i,r in enumerate(rows) if r['phase']=='think' and r['forced_by_original_budget']=='False'];text=''.join(r['token_piece'] for r in rows);pos=[0]
 for r in rows:pos.append(pos[-1]+len(r['token_piece']))
 if valid:
  start=pos[valid[0]];end=pos[valid[-1]+1];content=text[start:end]
  # Gemma generated channel-header word is not the first substantive paragraph.
  leading=0
  for marker in ['thought','<think>','<|channel>thought']:
   if content.lstrip().startswith(marker):leading=content.index(marker)+len(marker);break
  while leading<len(content) and content[leading].isspace():leading+=1
  brk=content.find('\n\n',leading)
  cutchar=start+(brk+2 if brk>=0 else len(content));paragraph_end_step=bisect.bisect_left(pos,cutchar);paragraph_end_step=min(paragraph_end_step,len(rows));first_paragraph=content[leading:brk if brk>=0 else None]
 else:paragraph_end_step=0;first_paragraph=''
 opening_cutoff=max(64,paragraph_end_step)
 metrics={scope:analyze_window(rows,ii) for scope,ii in [('think_all',valid),('opening64',[i for i in valid if i<64]),('after64',[i for i in valid if i>=64]),('after_opening',[i for i in valid if i>=opening_cutoff]),('answer',[i for i,r in enumerate(rows) if r['phase']=='answer' and r['forced_by_original_budget']=='False'])]}
 for scope in ['after64','after_opening']:
  met=metrics[scope]
  for kind in ['min_token','min16']:
   v=met[kind]
   if v is None:continue
   a=(v['step'] if kind=='min_token' else v['start'])-1;b=v['step'] if kind=='min_token' else v['end'];valleys.append(dict(model=model,dataset=job['dataset'],id=job['id'],key=key,condition=job['condition'],group=group,scope=scope,kind=kind,start=a+1,end=b,delta_H=v['value'],before=text[max(0,pos[a]-180):pos[a]],selected=text[pos[a]:pos[b]],after=text[pos[b]:pos[b]+180],semantic_category=None,source_sha256=job['source_sha256']))
 allcases.append(dict(key=key,id=job['id'],dataset=job['dataset'],model=model,condition=job['condition'],group=group,labels={k:v['status'] for k,v in lab.items()},tokens=len(rows),metrics=metrics,opening_cutoff=opening_cutoff,first_paragraph_end_step=paragraph_end_step,first_paragraph=first_paragraph,source_sha256=job['source_sha256'],image_sha256=job['image_sha256'],original_actual_top20=c['original_pressure_top20_fraction'],prefix_checks=c['prefix_checks'],formula_error=c['formula_max_absdiff'],forward_method=c.get('forward_method','parallel_full_sequence' if c.get('new_forward') else 'reused_validated_full_sequence'),rejected_parallel_checks=c.get('rejected_parallel_checks',[]),reused=not c['new_forward'],seconds=c['seconds']))
for fn,x in [('cases.json',allcases),('valley_contexts.json',valleys),('source_texts.json',texts)]: (O/fn).write_text(json.dumps(x,ensure_ascii=False,indent=2))
groups=[];rng=np.random.default_rng(42)
for ds in ['ClockQA','PathVQA','PENDULUM']:
 for group in ['flip','hold','pressure_unresolved','neutral_wrong','neutral_unresolved']:
  for cond in ['belief','belief_right']:
   cc=[r for r in allcases if r['dataset']==ds and r['group']==group and r['condition']==cond]
   for scope in ['think_all','after64','after_opening']:
    eligible=[c for c in cc if c['metrics'][scope]['n']];vals=[c['metrics'][scope]['means']['delta_H'] for c in eligible];ci=None
    if len(vals)>=2:
     v=np.asarray(vals);ci=np.quantile(v[rng.integers(0,len(v),size=(2000,len(v)))].mean(1),[.025,.975]).tolist()
    groups.append(dict(model=model,dataset=ds,group=group,condition=cond,scope=scope,total=len(cc),eligible=len(eligible),no_remaining_thinking=len(cc)-len(eligible),means={f:statistics.mean(c['metrics'][scope]['means'][f] for c in eligible) if eligible else None for f in fields},delta_H_bootstrap95=ci,min16_eligible=sum(c['metrics'][scope]['min16'] is not None for c in cc)))
cohorts={}
for ds in ['ClockQA','PathVQA','PENDULUM']:
 rr=[c for c in allcases if c['dataset']==ds and c['condition']=='belief'];cohorts[ds]=dict(questions=len(rr),groups=dict(collections.Counter(c['group'] for c in rr)),neutral=dict(collections.Counter(c['labels']['neutral'] for c in rr)),user_wrong=dict(collections.Counter(c['labels']['belief'] for c in rr)),user_right=dict(collections.Counter(c['labels']['belief_right'] for c in rr)),unique_images=len({c['image_sha256'] for c in rr}))
checks=[x for c in allcases for x in c['prefix_checks']]
summary=dict(model=model,questions=150,trajectories=300,tokens=sum(c['tokens'] for c in allcases),cohorts=cohorts,groups=groups,methods=dict(collections.Counter(c['forward_method'] for c in allcases)),reused=sum(c['reused'] for c in allcases),min_actual_top20=min(c['original_actual_top20'] for c in allcases),max_prefix_JSD=max(c['JSD'] for c in checks),max_prefix_entropy_difference=max(c['entropy_absdiff'] for c in checks),max_formula_error=max(c['formula_error'] for c in allcases),note='Question-weighted means, bootstrap over questions within each separate group; small groups remain small. No automated semantic label claimed. after_opening excludes first substantive paragraph and at least64output tokens; after64 is fixed cutoff sensitivity.')
(O/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
flat=[]
for c in allcases:
 for scope in ['think_all','after64','after_opening']:
  met=c['metrics'][scope];flat.append(dict(model=model,dataset=c['dataset'],id=c['id'],condition=c['condition'],group=c['group'],scope=scope,tokens=met['n'],opening_cutoff=c['opening_cutoff'],**(met['means'] or {f:None for f in fields}),min_step=met['min_token']['step'] if met['min_token'] else None,min_delta=met['min_token']['value'] if met['min_token'] else None))
with (O/'metrics.csv').open('w',newline='') as f:
 wr=csv.DictWriter(f,fieldnames=list(flat[0]));wr.writeheader();wr.writerows(flat)
print(json.dumps(dict(model=model,questions=150,trajectories=300,cohorts=cohorts,methods=summary['methods'],max_prefix_JSD=summary['max_prefix_JSD']),ensure_ascii=False))
