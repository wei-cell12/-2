"""CPU-only, read existing replay arrays; identical JSD formulas to prior work."""
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np

B = Path('/mnt/sdb/linxin/pendulum-20261102')
P = B/'expand4x50'
A = P/'attention/qwen38'
D = P/'qwen38/analysis_20261102'
D.mkdir(exist_ok=True)
sys.path.insert(0,str(B/'source'))
from symexp.fp.reading_groups import _jsd
from symexp.fp.sources import _norm_img

conds=['neutral','belief_right','belief']
items=[x for x in map(json.loads,(P/'items.jsonl').read_text().splitlines()) if x['domain'] in ['re_used','self_captured']]
rows=[];maps=[]
for index,item in enumerate(items):
 iid=item['id']; data={}; metas={}
 for c in conds:
  m=json.loads((A/'records'/(iid+'__'+c+'.json')).read_text());metas[c]=m
  assert m['id']==iid and m['condition']==c and m['image_sha256']==item['image_sha256']
  arrpath=Path(m['arrays'])
  assert arrpath.stat().st_size==m['array_bytes']
  with np.load(arrpath) as z:
   layers=m['attention_layers']
   data[c]=dict(early=np.stack([z[f'L{l}_early_sum'] for l in layers],axis=1).astype(np.float64),
                stage=np.stack([z[f'L{l}_stage_sum'] for l in layers],axis=1).astype(np.float64),
                early_count=z[f'L{layers[0]}_early_count'].copy(), stage_count=z[f'L{layers[0]}_stage_count'].copy())
   for l in layers:
    assert np.array_equal(z[f'L{l}_early_count'],data[c]['early_count'])
    assert np.array_equal(z[f'L{l}_stage_count'],data[c]['stage_count'])
   assert not any(m['sliding_windows'].values())
 assert len({tuple(metas[c]['image_grid']) for c in conds})==1
 assert len({metas[c]['n_img'] for c in conds})==1
 shape=data['neutral']['stage'].shape[1:]; n_img=shape[-1]
 assert shape[0]*shape[1]==384
 k=math.ceil(.2*n_img)
 E=np.zeros(shape,bool)
 order=np.argsort(-data['neutral']['stage'][1],axis=-1,kind='stable')[...,:k]
 np.put_along_axis(E,order,True,axis=-1)
 neutral_mass=data['neutral']['stage'][1].sum(-1)/max(int(data['neutral']['stage_count'][1]),1)
 legacy=np.argsort(-neutral_mass.ravel(),kind='stable')[:96]
 info=dict(id=iid,domain=item['domain'],image_sha256=item['image_sha256'],image_grid=metas['neutral']['image_grid'],
           image_tokens=n_img, E_tokens=k, source_sha256={c:metas[c]['source_sha256'] for c in conds},
           output_tokens={c:metas[c]['n_out'] for c in conds}, thought_tokens={c:int(data[c]['stage_count'][1]) for c in conds},
           layers=layers,windows={},stages={})
 for W in [64,128,256,512]:
  G=W//8
  if any(not np.all(data[c]['early_count'][:G]==8) for c in conds):
   info['windows'][str(W)]=dict(eligible=False,reason='at least one of all three conditions has fewer than W thought-content tokens')
   continue
  P0={c:_norm_img(data[c]['early'][:G].sum(0)) for c in conds}
  Pe={c:_norm_img(data[c]['early'][:G:2].sum(0)) for c in conds}
  Po={c:_norm_img(data[c]['early'][1:G:2].sum(0)) for c in conds}
  M={c:data[c]['early'][:G].sum((0,-1))/W for c in conds}
  within={c:_jsd(Pe[c],Po[c]) for c in conds}
  score=np.mean([M[c] for c in conds],axis=0)
  balanced=np.argsort(-score.ravel(),kind='stable')[:96]
  record=dict(eligible=True,selectors={})
  for name,selection in [('balanced96',balanced),('neutral96_legacy',legacy)]:
   sel=np.zeros(shape[:2],bool);sel.ravel()[selection]=True
   r=dict(heads=96,head_pairs=[[layers[int(j//shape[1])],int(j%shape[1])] for j in selection],conditions={},pairs={})
   for c in conds:
    share=(P0[c]*E).sum(-1)
    r['conditions'][c]=dict(image_mass=float(M[c][sel].mean()),E_share=float(share[sel].mean()),
                           E_RRAR=float((share*n_img/k)[sel].mean()),within_jsd=float(within[c][sel].mean()))
   for c1,c2 in [('neutral','belief'),('neutral','belief_right'),('belief','belief_right')]:
    half=_jsd(Pe[c1],Pe[c2]); full=_jsd(P0[c1],P0[c2]);noise=.5*(within[c1]+within[c2])
    r['pairs'][c1+'__'+c2]=dict(jsd_full=float(full[sel].mean()),jsd_half=float(half[sel].mean()),
                              noise=float(noise[sel].mean()),excess=float((half-noise)[sel].mean()))
   record['selectors'][name]=r
  info['windows'][str(W)]=record
 # Whole phases: lengths differ, therefore descriptive only. Balanced selection by mean stage image mass.
 for s,name in enumerate(['prefill_end','think','answer']):
  if any(data[c]['stage_count'][s]==0 for c in conds):
   info['stages'][name]=dict(eligible=False)
   continue
  stage_mean={c:data[c]['stage'][s]/data[c]['stage_count'][s] for c in conds}
  masses={c:stage_mean[c].sum(-1) for c in conds}
  select=np.argsort(-np.mean(list(masses.values()),axis=0).ravel(),kind='stable')[:96]
  sel=np.zeros(shape[:2],bool);sel.ravel()[select]=True
  distributions={c:_norm_img(stage_mean[c]) for c in conds}
  info['stages'][name]=dict(eligible=True,conditions={c:dict(image_mass=float(masses[c][sel].mean()),
         E_share=float((distributions[c]*E).sum(-1)[sel].mean()),tokens=int(data[c]['stage_count'][s])) for c in conds},
         jsd_neutral_belief=float(_jsd(distributions['neutral'],distributions['belief'])[sel].mean()))
  maps.append(dict(id=iid,domain=item['domain'],stage=name,grid=metas['neutral']['image_grid'],
       normalized_image_distribution={c:np.round(distributions[c][sel].mean(0),9).tolist() for c in conds},
       selected_heads=96,image_sha256=item['image_sha256']))
 rows.append(info)
 if (index+1)%10==0: print(f'CPU {index+1}/100 questions, no model loaded',flush=True)
out=dict(created=time.time(),questions=100,records=300,unique_images=len({x['image_sha256'] for x in rows}),
         formula_source='frozen symexp.fp.reading_groups._jsd and sources._norm_img',
         jsd_unit='natural logarithm (nats), upper bound ln2',
         primary_selector='per item and matched window: top96 of384 heads ranked by image mass averaged equally across three conditions',
         secondary_selector='legacy: top96 by full neutral thought image mass',
         E_definition='per head: ceil(20% of image tokens) with highest full-neutral-thought attention; model reference, NOT human ground-truth ROI',
         excess_formula='JSD(even 8-token groups of neutral,even groups of pressure) - mean of both within-condition even-vs-odd JSDs; negative values retained',
         inclusion='all3 conditions have W thought-content tokens; every W has its own explicit ID list; no padding or regenerated tokens',
         rows=rows)
(D/'attention_per_item.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
(D/'attention_maps.json').write_text(json.dumps(maps,ensure_ascii=False))
print(json.dumps(dict(complete=True,questions=100,unique_images=out['unique_images'],eligible={str(W):sum(x['windows'][str(W)]['eligible'] for x in rows) for W in [64,128,256,512]})),flush=True)
