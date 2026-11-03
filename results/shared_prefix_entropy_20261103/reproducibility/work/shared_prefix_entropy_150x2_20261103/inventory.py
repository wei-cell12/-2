import json,subprocess,os,socket
from pathlib import Path
print('HOST',socket.gethostname())
p=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True);print('GPU',p.stdout,p.stderr)
p=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True);print('PROCESSES',p.stdout)
roots=['/mnt/sdb/linxin/conflict-cq-20261028','/mnt/sdb/linxin/qwen38-pv-20261031','/mnt/sdb/linxin/gemma4-cq-20261029','/mnt/sdb/linxin/gemma4-pv-20261030','/mnt/sdb/linxin/pendulum-20261102/expand4x50','/mnt/sdg/linxin/s2va-20261101/qwen_cq','/mnt/sdg/linxin/s2va-20261101/qwen_pv','/mnt/sdg/linxin/s2va-20261101/gemma_cq','/mnt/sdg/linxin/s2va-20261101/gemma_pv']
for root in roots:
 R=Path(root)
 if not R.exists():continue
 print('ROOT',root)
 for f in ['items.jsonl','qwen38.yaml','configs/qwen38.yaml']:
  if (R/f).exists():print('FILE',f,(R/f).stat().st_size)
 for sub in ['obs/records','qwen38/records','gemma4/records']:
  files=sorted((R/sub).glob('*.json'))
  if files:
   r=json.loads(files[0].read_text());print('RECORDS',sub,len(files),list(r),{k:r.get(k) for k in ['id','condition','n_in','n_out','model','image','budget']})
for f in ['/mnt/sdb/linxin/repo-gemma4/configs/gemma4_31b.yaml','/mnt/sdg/linxin/s2va-20261101/configs/gemma4_31b.yaml','/mnt/sdb/linxin/pendulum-20261102/configs/gemma4.yaml']:
 p=Path(f)
 if p.exists():print('CONFIG',f,p.read_text())
for p in ['/mnt/sdb/linxin/ModelList/Qwen3.8-27B','/mnt/sdb/ljh/models/Qwen3.8-27B','/mnt/sdb/ljh/models/gemma-4-31B-it','/mnt/ssd/gy/models/Qwen3.8-27B']:
 print('MODEL',p,Path(p).exists())
