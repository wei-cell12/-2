import base64,hashlib,json,os,subprocess,tarfile,time
from pathlib import Path
W=Path(ROOT);M=json.loads((W/'manifest.json').read_text());receipts=json.loads((W/'launch_receipt.json').read_text());states=[]
for q in receipts:
 s=json.loads((W/f"state_{q['rank']}.json").read_text());assert s['phase']=='complete',json.dumps(s)
 assert not Path('/proc',str(s['pid'])).exists(),'Worker has not naturally exited'
 states.append(s)
expected={j['key'] for j in M['jobs']};assert len(expected)==300
actual={p.stem for p in (W/'records').glob('*.json')};assert actual==expected,(len(actual),len(expected))
cs=[json.loads((W/'records'/(j['key']+'.json')).read_text()) for j in M['jobs']]
assert len(cs)==300 and all((W/'records'/(c['key']+'.csv')).exists() for c in cs)
(W/'summary.json').write_text(json.dumps(dict(model=M['model'],questions=150,trajectories=300,cases=cs),ensure_ascii=False,indent=2))
log=W/'analysis.log'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
with log.open('xb') as f:r=subprocess.run([M['python'],str(W/'analyze_server.py')],env=env,stdout=f,stderr=subprocess.STDOUT)
if r.returncode:print(log.read_text());raise SystemExit(r.returncode)
release=dict(owned_processes_exited=True,pids=[s['pid'] for s in states],finished_states=states,resource_snapshot=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout,verified_at=time.time());(W/'release_receipt.json').write_text(json.dumps(release,indent=2))
include=[]
for p in W.rglob('*'):
 if not p.is_file():continue
 rel=p.relative_to(W)
 if rel.parts[0]=='inputs' or p.name.endswith(('.tar.gz','.npz','.npy')):continue
 if '__pycache__' in rel.parts:continue
 if p.suffix in ['.json','.csv','.log','.py','.yaml','.md','.patch']:include.append(p)
filelist={str(p.relative_to(W)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in include};(W/'artifact_manifest.json').write_text(json.dumps(filelist,indent=2));include.append(W/'artifact_manifest.json')
archive=W/'results_light.tar.gz'
if archive.exists():raise RuntimeError('Archive already exists; do not repeat finalization')
with tarfile.open(archive,'w:gz') as t:
 for p in include:t.add(p,arcname=str(p.relative_to(W)),recursive=False)
b=archive.read_bytes();print(json.dumps(dict(model=M['model'],questions=150,trajectories=300,analysis_log=log.read_text(),archive_bytes=len(b),archive_sha256=hashlib.sha256(b).hexdigest(),archive_b64=base64.b64encode(b).decode()),ensure_ascii=False))
