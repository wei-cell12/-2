import csv,gzip,json,collections,hashlib,statistics
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parent;O=B.parent/'shared_prefix_entropy_150x2_20261103/results';R=B.parent/'clockqa_uncertainty_attention_20261101/results/records';V=json.loads((B/'valleys_all.json').read_text());cases=json.loads((O/'qwen38/analysis/cases.json').read_text());by=collections.defaultdict(list)
for r in V:
 if r['model']=='qwen38' and r['dataset']=='ClockQA' and r['band']!='tiny':by[r['key']].append(r)
def kind(s):return 'space' if s.isspace() else 'number' if any(ch.isdigit() for ch in s) else 'word' if any(ch.isalpha() for ch in s) else 'punct'
rows=[];coverage=[]
for c in cases:
 if c['dataset']!='ClockQA':continue
 rs=list(csv.DictReader((O/'qwen38/records'/(c['key']+'.csv')).open()));cond='wrong' if c['condition']=='belief' else 'correct';p=R/(c['id']+'--belief__'+cond+'.jsonl.gz')
 with gzip.open(p,'rt') as f:a=[json.loads(l) for l in f]
 assert len(a)==len(rs) and all(x['token']==r['token_piece'] for x,r in zip(a,rs))
 valid={i for i,r in enumerate(rs) if i>=c['opening_cutoff'] and r['phase']=='think' and r['forced_by_original_budget']=='False'};quiet=[i for i in valid if abs(float(rs[i]['delta_H']))<=.05];tok=collections.defaultdict(list);kt=collections.defaultdict(list)
 for i in quiet:tok[rs[i]['token_piece']].append(i);kt[kind(rs[i]['token_piece'])].append(i)
 pressure=np.array([r['all_heads']['pressure_mass']*100 for r in a]);im=np.array([r['all_heads']['image_mass']*100 for r in a]);share=pressure/np.maximum(pressure+im,1e-12)*100
 def sig(i):
  ix=[j for j in range(max(0,i-8),i) if j in valid]
  if len(ix)<4:return None
  return dict(user_attention=float(pressure[i]),image_attention=float(im[i]),user_jump=float(pressure[i]-pressure[ix].mean()),image_jump=float(im[i]-im[ix].mean()),user_image_share_jump=float(share[i]-share[ix].mean()))
 for e in by[c['key']]:
  i=e['step']-1;s=sig(i)
  if s is None:continue
  choices=[j for j in tok[e['token']] if abs(j-i)>8 and sig(j) is not None];match='same_token'
  if not choices:choices=[j for j in kt[kind(e['token'])] if abs(j-i)>8 and sig(j) is not None];match='same_token_kind'
  if not choices:continue
  control=min(choices,key=lambda j:abs(j-i));z=sig(control)
  rows.append(dict(**e,attention=s,control_step=control+1,control_token=rs[control]['token_piece'],control_delta_H=float(rs[control]['delta_H']),control_attention=z,match=match,paired_user_jump_difference=s['user_jump']-z['user_jump'],paired_share_jump_difference=s['user_image_share_jump']-z['user_image_share_jump']))
 coverage.append(dict(key=c['key'],tokens=len(rs),token_match=True,attention_source=str(p),attention_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
rng=np.random.default_rng(20261103);summary=[]
for band in ['deep','medium','shallow']:
 rr=[r for r in rows if r['band']==band];imgs=sorted({r['image_sha256'] for r in rr});diff=np.array([statistics.mean(r['paired_user_jump_difference'] for r in rr if r['image_sha256']==im) for im in imgs]);boots=diff[rng.integers(0,len(diff),size=(2000,len(diff)))].mean(1)
 def trajmean(f):return statistics.mean(statistics.mean(f(r) for r in rr if r['image_sha256']==im) for im in imgs)
 summary.append(dict(band=band,matched_valleys=len(rr),source_images=len(imgs),same_token_matches=sum(r['match']=='same_token' for r in rr),mean_valley_user_jump_pp=trajmean(lambda r:r['attention']['user_jump']),mean_control_user_jump_pp=trajmean(lambda r:r['control_attention']['user_jump']),paired_user_jump_difference_pp=float(diff.mean()),image_cluster_bootstrap95=np.quantile(boots,[.025,.975]).tolist(),valley_user_jump_ge2pp=trajmean(lambda r:r['attention']['user_jump']>=2),control_user_jump_ge2pp=trajmean(lambda r:r['control_attention']['user_jump']>=2),valley_user_jump_positive=trajmean(lambda r:r['attention']['user_jump']>0),control_user_jump_positive=trajmean(lambda r:r['control_attention']['user_jump']>0)))
(B/'attention_pairs.json').write_text(json.dumps(rows,ensure_ascii=False));(B/'attention_summary.json').write_text(json.dumps(dict(coverage=len(coverage),coverage_records=coverage,definition='QwenClockQA100trajectories; exact savedtoken/length alignment.16standardlayers384heads. User-span attention jump=current mass minus mean of prior8eligiblethought positions, at least4previous tokens. Quiet controls |deltaH|<=.05, same trajectory, same token when possible otherwise same token kind,nearest position outside8tokens. Per-image weighted paired means,2000imagecluster bootstraps. All bands frozen before attention review.',bands=summary),ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False))
