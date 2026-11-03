import csv,hashlib,json,random,collections
from pathlib import Path
B=Path(__file__).resolve().parent;D=B/'blind';D.mkdir(exist_ok=True);O=B.parent/'shared_prefix_entropy_150x2_20261103/results';events=json.loads((B/'valleys_all.json').read_text());rng=random.Random(20261103)
if (B/'private_keymap.json').exists():raise RuntimeError('Already frozen; do not overwrite')
selected=[];cache={};texts={m:json.loads((O/m/'analysis/source_texts.json').read_text()) for m in ['qwen38','gemma4']};caseindex={}
for m in ['qwen38','gemma4']:
 for c in json.loads((O/m/'analysis/cases.json').read_text()):caseindex[m,c['key']]=c
def rows(e):
 key=e['model'],e['key']
 if key not in cache:cache[key]=list(csv.DictReader((O/e['model']/'records'/(e['key']+'.csv')).open()))
 return cache[key]
def score(x):return hashlib.sha256(('behavior20261103|'+x).encode()).hexdigest()
def tokenkind(s):return 'space' if s.isspace() else 'number' if any(x.isdigit() for x in s) else 'word' if any(x.isalpha() for x in s) else 'punct'
for model in ['qwen38','gemma4']:
 for ds in ['ClockQA','PathVQA','PENDULUM']:
  for cond in ['belief','belief_right']:
   for band in ['deep','medium','shallow']:
    rr=[e for e in events if e['model']==model and e['dataset']==ds and e['condition']==cond and e['band']==band];by=collections.defaultdict(list)
    for e in rr:by[e['key']].append(e)
    ids=sorted(by,key=lambda k:score(model+'|'+ds+'|'+cond+'|'+band+'|'+k))[:15]
    for key in ids:
     r=min(by[key],key=lambda e:score(band+'|'+key+'|'+str(e['step'])));selected.append(dict(r,candidate_type=band,inclusion_trajectory_probability=min(1,15/len(by)),inclusion_event_probability=min(1,15/len(by))/len(by[key])))
# One quiet comparator paired to each selected medium valley, same trajectory.
controls=[]
for e in [x for x in selected if x['candidate_type']=='medium']:
 rs=rows(e);i=e['step']-1;c=caseindex[e['model'],e['key']];valid=[j for j,r in enumerate(rs) if j>=c['opening_cutoff'] and r['phase']=='think' and r['forced_by_original_budget']=='False'];quiet=[j for j in valid if abs(float(rs[j]['delta_H']))<=.05 and abs(j-i)>8]
 same=[j for j in quiet if rs[j]['token_piece']==e['token']];kind=[j for j in quiet if tokenkind(rs[j]['token_piece'])==tokenkind(e['token'])];pool=same or kind or quiet
 if not pool:continue
 j=min(pool,key=lambda j:abs(j-i));r=rs[j];controls.append(dict(model=e['model'],dataset=e['dataset'],id=e['id'],key=e['key'],condition=e['condition'],group=e['group'],image_sha256=e['image_sha256'],source_sha256=e['source_sha256'],opening_cutoff=c['opening_cutoff'],step=j+1,token=r['token_piece'],candidate_type='quiet_control',band='quiet_control',depth=-float(r['delta_H']),H_with_U=float(r['H_pressure']),H_without_U=float(r['H_neutral']),delta_H=float(r['delta_H']),JSD=float(r['JSD']),delta_logp=float(r['delta_logp']),matched_medium_step=e['step'],match_type='same_token' if same else 'same_token_kind' if kind else 'same_trajectory',position_distance=abs(j-i)))
selected+=controls;private={};requests=[];ids=set()
for n,e in enumerate(selected):
 rs=rows(e);i=e['step']-1;sig=(e['model'],e['key'],e['step'],e['candidate_type']);assert sig not in ids;ids.add(sig)
 key='beh_'+score(str(n)+'|'+str(sig))[:22];lo=max(0,i-96);hi=min(len(rs),i+65);before=''.join(r['token_piece'] for r in rs[lo:i]);after=''.join(r['token_piece'] for r in rs[i+1:hi]);prior=[k for k in range(max(0,i-16),i) if k>=e['opening_cutoff'] and rs[k]['phase']=='think' and rs[k]['forced_by_original_budget']=='False']
 def avg(f):return sum(float(rs[k][f]) for k in prior)/len(prior) if prior else None
 extra={f'prior16_{f}':avg(f) for f in ['H_pressure','H_neutral','delta_H','JSD']};extra.update(dict(context_first_output_step=lo+1,context_last_output_step=hi,token_kind=tokenkind(e['token']),source_output_tokens=len(rs),relative_position=i/max(1,len(rs)-1),forward_method=caseindex[e['model'],e['key']]['forward_method']))
 private[key]=dict(**e,**extra);requests.append(dict(key=key,message=texts[e['model']][e['key']]['condition']['message'],before=before,token=e['token'],after=after))
rng.shuffle(requests)
for j,a in enumerate(range(0,len(requests),25),1):(D/f'requests_{j:02d}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in requests[a:a+25]))
(B/'private_keymap.json').write_text(json.dumps(private,ensure_ascii=False,indent=2));counts=collections.Counter(e['candidate_type'] for e in selected);design=dict(total_requests=len(requests),counts=dict(counts),all_valleys_scanned=600,rule='Within each model/dataset/condition/depth band: sample up to15distinct trajectories by fixed hash,one uniformly hash-selected valley per trajectory. No selection on response outcome or semantic labels. Deep>=1, medium[.25,1), shallow[.05,.25). Quiet controls abs(deltaH)<=.05 matched to medium valley in same trajectory,nearest >8tokens, same token if available else same token kind. Cases/depth/control status hidden from Luna.',source_questions=len({(e['dataset'],e['id']) for e in selected}),source_images=len({e['image_sha256'] for e in selected}),source_model_trajectories=len({(e['model'],e['key']) for e in selected}),batches=(len(requests)+24)//25,request_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(D.glob('requests_*.jsonl'))})
(B/'design.json').write_text(json.dumps(design,ensure_ascii=False,indent=2));(B/'state.json').write_text(json.dumps(dict(phase='prepared',expected=len(requests),counts=dict(counts)),indent=2));print(json.dumps(design,ensure_ascii=False))
