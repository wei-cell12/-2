"""CPU only; REQUESTS supplied by launcher, never transfer raw arrays."""
import json
from pathlib import Path
import numpy as np

P=Path('/mnt/sdb/linxin/pendulum-20261102/expand4x50')
maps=[]
for req in REQUESTS:
    distributions={}
    for c in ['neutral','belief_right','belief']:
        path=P/'attention'/req['model']/'records'/(req['id']+'__'+c+'.json')
        m=json.loads(path.read_text())
        assert m['source_sha256']==req['source_sha256'][c]
        assert m['image_sha256']==req['image_sha256']
        assert Path(m['arrays']).stat().st_size==m['array_bytes']
        pairs=req['head_pairs']
        cache={}
        with np.load(m['arrays']) as z:
            for l,h in pairs:
                if l in cache:continue
                counts=z[f'L{l}_early_count']
                assert np.all(counts[:8]==8)
                assert np.all(z[f'L{l}_visible_image_tokens'][m['think_start']:m['think_start']+64]==m['n_img'])
                cache[l]=z[f'L{l}_early_sum'][:8].astype(np.float64).sum(0)
        v=np.stack([cache[l][h] for l,h in pairs])
        measured=float(v.sum(-1).mean()/64)
        assert abs(measured-req['expected_conditions'][c]['image_mass'])<1e-9,(req['model'],req['id'],c,measured)
        q=v/np.maximum(v.sum(-1,keepdims=True),1e-12)
        distributions[c]=q.mean(0).tolist()
    maps.append(dict(model=req['model'],id=req['id'],group=req['group'],selector=req['selector'],stage='first64',grid=req['grid'],image_sha256=req['image_sha256'],selected_heads=len(req['head_pairs']),normalized_image_distribution=distributions))
print(json.dumps(maps,ensure_ascii=False,allow_nan=False))
