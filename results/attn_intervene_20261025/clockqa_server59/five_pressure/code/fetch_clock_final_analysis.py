"""CPU-only saved-array reduction, with exact provenance; no model or tokens transfer."""
import json,subprocess,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;AUDIT=BASE/'luna_audit';OUT=BASE/'final_results';OUT.mkdir(exist_ok=True)
packet=json.loads((AUDIT/'intervention_source_snapshot.json').read_text())
specs=[{k:r[k] for k in ['id','question_id','method','trace_path','trace_sha256','metrics','seconds']} for r in packet['records']]
native=json.loads((AUDIT/'source_snapshot.json').read_text());prior=json.loads(Path(native['prior_source_snapshot']).read_text())
native_specs=[dict(record_key=r['record_key'],source_path=r['source_path'],source_sha256=r['source_sha256']) for r in prior['records']+native['records'] if r.get('kind')=='observation']
assert len(native_specs)==372 and len(specs)==51
remote=r'''
import hashlib,json,time
from pathlib import Path
import numpy as np
root=Path('/mnt/sdi/linxin/proofwriter_runs/clockqa-scaleup-20261025/interventions');specs=SPECS;native_specs=NATIVE
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
out=dict(cases=[],native_totals=[],array_manifest=[],snapshot_unix=time.time(),source='actual saved intervention traces; neutral maps from validated exact-sequence replay; CPU only')
for ident in sorted({s['id'] for s in specs}):
 ref=root/'references'/(ident+'.npz')
 with np.load(ref) as z:heads=z['heads'].astype(bool);grid=z['grid_hw'].tolist();neutral=z['E_r'].astype(float);emask=z['E_mask'].astype(bool)
 assert heads.sum()==96
 def reduce(a):
  mass=a.sum(-1);p=np.divide(a,mass[...,None],out=np.zeros_like(a),where=mass[...,None]>0)
  return [dict(map=p[s][heads].mean(0).tolist(),E_share_stage_average=float((p[s]*emask[s]).sum(-1)[heads].mean()),selected_heads_image_mass=float(mass[s][heads].mean())) for s in range(2)]
 c=dict(id=ident,question_id=ident.split('--')[0],grid_hw=grid,neutral=reduce(neutral),groups={})
 out['array_manifest'].append(dict(path=str(ref),bytes=ref.stat().st_size,sha256=sha(ref)))
 for r in [s for s in specs if s['id']==ident]:
  p=Path(r['trace_path']);assert sha(p)==r['trace_sha256']
  with np.load(p) as z:
   assert np.array_equal(z['selected_heads'].astype(bool),heads)
   c['groups'][r['method']]=dict(before=reduce(z['spatial_before'].astype(float)),after=reduce(z['spatial_after'].astype(float)),metrics=r['metrics'],seconds=r['seconds'])
  out['array_manifest'].append(dict(path=str(p),bytes=p.stat().st_size,sha256=r['trace_sha256']))
 out['cases'].append(c)
for spec in native_specs:
 p=Path(spec['source_path']);assert sha(p)==spec['source_sha256'];r=json.loads(p.read_text());a=p.parent/'attention.npz'
 with np.load(a) as z:mass=z['generation_masses'][...,0].astype(float);labels=z['phase_labels']
 stages={}
 for name,tag in [('think','thinking'),('answer','answer')]:
  selected=mass[labels==tag];stages[name]=dict(tokens=len(selected),image_mass=float(selected.mean()) if len(selected) else None)
 out['native_totals'].append(dict(record_key=spec['record_key'],id=r['item_id'],question_id=r.get('question_id',r['item_id'].split('--')[0]),condition=r['condition'],pressure_type=r['pressure_type'],stages=stages,source_sha256=spec['source_sha256']))
 out['array_manifest'].append(dict(path=str(a),bytes=a.stat().st_size,sha256=r['attention_sha256']))
out['config']=json.loads((root/'config.json').read_text());out['launch']=json.loads((root/'launch_receipt.json').read_text());out['queue']=json.loads((root/'queue.json').read_text());out['pilot_gate_sha256']=sha(root/'pilot_numerical_gate.json');out['review']=json.loads((root/'pilot_review.json').read_text())
out['model_allocations']=[json.loads(p.read_text()) for p in sorted((root/'workers').glob('*/latest_model_allocation.json'))]
out['neutral_replay_checks']=[json.loads(s) for p in sorted((root/'workers').glob('*/neutral_replay_checks.jsonl')) for s in p.read_text().splitlines() if s.strip()]
print(json.dumps(out))
'''.replace('SPECS',repr(specs)).replace('NATIVE',repr(native_specs))
sys.path.insert(0,str(BASE.parent/'server_connection'));from transport import command
p=subprocess.run(command('/home/linxin/miniconda3/envs/vllm_new/bin/python -'),input=remote,text=True,capture_output=True,timeout=110)
if p.returncode:raise RuntimeError(p.stderr)
x=json.loads(p.stdout);(OUT/'derived_attention.json').write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(cases=len(x['cases']),native_totals=len(x['native_totals']),array_manifest=len(x['array_manifest']),derived_bytes=len(p.stdout),no_model_loaded=True,no_npz_or_token_lists_transferred=True)))
