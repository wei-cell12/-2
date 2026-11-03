from pathlib import Path
import json,statistics,collections
import numpy as np
B=Path(__file__).resolve().parent;rr=json.loads((B/'attention_pairs.json').read_text());rng=np.random.default_rng(20261103);out={}
for name,sel in [('deepest',lambda r:r['band']=='deep' and r['depth_rank']==1),('secondary_deep',lambda r:r['band']=='deep' and r['depth_rank']>1),('secondary_deep_same_token',lambda r:r['band']=='deep' and r['depth_rank']>1 and r['match']=='same_token')]:
 a=[r for r in rr if sel(r)];imgs=sorted({r['image_sha256'] for r in a})
 def avg(f):return statistics.mean(statistics.mean(f(r) for r in a if r['image_sha256']==im) for im in imgs)
 diff=np.array([statistics.mean(r['paired_user_jump_difference'] for r in a if r['image_sha256']==im) for im in imgs]);boots=diff[rng.integers(0,len(diff),(3000,len(diff)))].mean(1)
 out[name]=dict(valleys=len(a),images=len(imgs),trajectories=len({r['key'] for r in a}),valley_user_jump_pp=avg(lambda r:r['attention']['user_jump']),control_jump_pp=avg(lambda r:r['control_attention']['user_jump']),difference_pp=float(diff.mean()),image_cluster95_pp=np.quantile(boots,[.025,.975]).tolist(),valley_user_jump_ge2pp=avg(lambda r:r['attention']['user_jump']>=2),control_user_jump_ge2pp=avg(lambda r:r['control_attention']['user_jump']>=2))
(B/'secondary_attention_summary.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
