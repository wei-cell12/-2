"""Neutral replay validation and fixed, per-stage/head reference construction."""
import hashlib,json,math
from pathlib import Path
import numpy as np
LAYERS=list(range(3,64,4));RANDOM_SEED=20261024

def validate_replay(d,row):
    assert d['layer_ids'].tolist()==LAYERS
    assert float(d['count_prefill_end'])==1
    assert int(d['count_think'])+int(d['count_answer'])+1==len(row['output_ids'])
    assert d['pos_image_mass'].shape==(len(row['output_ids']),16,24)
    diffs=d['selfcheck_max_abs_diff'];assert len(diffs) and float(np.max(diffs))<.05
    assert all(d[x].shape==(16,24,int(d['n_img'])) for x in ['think','answer'])
    assert float(d['count_think'])>0 and float(d['count_answer'])>0
    return dict(ok=True,id=row['id'],selfcheck_max=float(np.max(diffs)),counts={s:int(d['count_'+s]) for s in ['prefill_end','think','answer']},n_img=int(d['n_img']))

def construct(item,d,path):
    n=int(d['n_img']); k=math.ceil(.2*n)
    assert 0<k<=n-k
    weights=np.stack([d['think'],d['answer']]).astype(np.float64)
    mass=weights.sum(-1,keepdims=True);assert (mass>0).all() and np.isfinite(weights).all()
    r=(weights/mass).astype(np.float32)
    heads=np.zeros((16,24),bool)
    ranked=np.argsort(-mass[0,:,:,0].ravel(),kind='stable')[:96];heads.ravel()[ranked]=True
    mask=np.zeros_like(r,dtype=bool);randmask=np.zeros_like(mask);randref=np.zeros_like(r)
    for s in range(2):
        for l in range(16):
            for h in range(24):
                e=np.argsort(-r[s,l,h],kind='stable')[:k];mask[s,l,h,e]=True
                seed=int.from_bytes(hashlib.sha256(f'{RANDOM_SEED}|{item}|{s}|{LAYERS[l]}|{h}'.encode()).digest()[:8],'little')
                rng=np.random.default_rng(seed)
                fake=rng.choice(np.flatnonzero(~mask[s,l,h]),size=k,replace=False)
                randmask[s,l,h,fake]=True
                randref[s,l,h,fake]=rng.permutation(r[s,l,h,e])
    assert np.allclose(r.sum(-1),1) and not (mask&randmask).any()
    np.savez_compressed(path,layer_ids=np.array(LAYERS),E_r=r,E_mask=mask,random_r=randref,random_mask=randmask,heads=heads,grid_hw=d['grid_hw'],n_img=n)
    return dict(id=item,n_img=n,E_size=k,E_fraction=k/n,heads_count=int(heads.sum()),heads=[dict(layer=LAYERS[l],head=h,neutral_think_image_mass=float(mass[0,l,h,0])) for l,h in zip(*heads.nonzero())],head_selection='exact top96 of384 by neutral thinking image total; stable flattened layer/head tie-break',E_selection='ceil(0.2*n_img), stable descending r',random_seed=RANDOM_SEED,random_seed_derivation='first8 little-endian bytes SHA256(seed|item|stage|layer|head); NumPy default_rng; choose non-E without replacement and independently permute E reference values')
