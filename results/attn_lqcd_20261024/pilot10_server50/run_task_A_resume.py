"""Reuse SHA256-verified existing weights: A1 -> first two pairs -> A3 -> ten pairs -> analysis; never retries."""
import datetime,fcntl,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PYTHON=ROOT/'env/bin/python'
STATE=ROOT/'job_state.json'
GPU=int(os.environ.get('PATHVQA_PHYSICAL_GPU','6'))
assert GPU != 7, 'GPU7 is excluded by the user'
def state(status,**extra):
 v=dict(status=status,pid=os.getpid(),time_beijing=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),gpu=GPU,**extra)
 tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,STATE)
def runstep(step,args,env):
 state('running',step=step)
 t=time.time()
 with (ROOT/(step+'.log')).open('w') as f:
  r=subprocess.run([str(PYTHON)]+list(map(str,args)),stdout=f,stderr=subprocess.STDOUT,cwd=ROOT/'source',env=env)
 assert r.returncode==0,(step,'exit',r.returncode)
 (ROOT/(step+'_receipt.json')).write_text(json.dumps(dict(ok=True,seconds=time.time()-t,args=list(map(str,args))),indent=2)+'\n')
def validate(step,expected,env):
 out=ROOT/(step+'.json')
 runstep(step,[ROOT/'source/run.py','attn-validate','--replay',ROOT/'attn_pilot10','--saved',str(ROOT/'original_attention/{id}__{condition}.npy'),'--out',out],env)
 s=json.loads(out.read_text())['summary']
 assert s['items']==expected and s['ok']==expected and s['rows_mismatch']==0,s
 assert any(s.get(k+'_corr_min',0)>=.99 and s.get(k+'_mean_abs_diff_max',1)<=.01 for k in ['raw','normalized']),s
 return s

def main():
 with (ROOT/'runner.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if STATE.exists():
   prior=json.loads(STATE.read_text())
   assert prior.get('status')=='waiting_for_gpu','Task already has a recorded launch; no automatic restart'
   STATE.rename(ROOT/('resource_wait_'+str(time.time_ns())+'.json'))
  step='A1_target'
  try:
   verified=json.loads((ROOT/'existing_27b_weight_sha256.json').read_text())
   assert verified['ok'] and len(verified['checks'])==19, 'Existing weight identity not verified'
   state('checking',step=step)
   import numpy as np
   rows=[json.loads(l) for l in (ROOT/'manifest_plain.jsonl').read_text().splitlines()]
   receipt=json.loads((ROOT/'A1_validation.json').read_text())
   assert receipt['ok'] and len(rows)==300
   assert hashlib.sha256((ROOT/'manifest_plain.jsonl').read_bytes()).hexdigest()==receipt['manifest_sha256']
   src={(r['id'],r['condition']):r for r in receipt['items']}
   for r in rows:
    meta=src[(r['id'],r['condition'])]
    assert hashlib.sha256(Path(r['image']).read_bytes()).hexdigest()==meta['image_sha256']
    saved=np.load(ROOT/'original_attention'/(r['id']+'__'+r['condition']+'.npy'),mmap_mode='r')
    assert saved.shape==(len(r['output_ids']),16,24) and saved.dtype==np.float32
   (ROOT/'A1_target_validation.json').write_text(json.dumps(dict(ok=True,records=300,images_verified=True,token_manifest_verified=True))+'\n')
   # Immediate check only for the selected GPU, then restrict visibility to it.
   import csv
   from io import StringIO
   q=subprocess.run(['nvidia-smi','-i',str(GPU),'--query-gpu=uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
   a=next(csv.reader(StringIO(q.stdout)));u=a[0].strip()
   p=subprocess.run(['nvidia-smi','-i',str(GPU),'--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
   if p.stdout.strip() or int(a[1])>1024 or int(a[2])<45000 or int(a[3])!=0:
    state('waiting_for_gpu',step='before_model',gpu_query=q.stdout,process_query=p.stdout);return
   env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=u,PATHVQA_PHYSICAL_GPU=str(GPU),PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
   base=[ROOT/'replay_failstop.py','attn-replay','--config',ROOT/'source/configs/qwen38_hf.yaml','--manifest',ROOT/'manifest_pilot10.jsonl','--out',ROOT/'attn_pilot10']
   settings=['--set','attn_implementation=sdpa','model_path=/mnt/ssd/gy/models/Qwen3.8-27B']
   # Resume A4 without revalidating an ever-growing directory as if it still had four files.
   if (ROOT/'A3.json').exists():
    pilot=json.loads((ROOT/'A3.json').read_text())['summary']
    assert pilot['items']==4 and pilot['ok']==4 and pilot['rows_mismatch']==0,pilot
    assert any(pilot.get(k+'_corr_min',0)>=.99 and pilot.get(k+'_mean_abs_diff_max',1)<=.01 for k in ['raw','normalized']),pilot
    for item in ['PV-000','PV-001']:
     for condition in ['neutral','user_wrong']:
      assert (ROOT/'attn_pilot10'/(item+'__'+condition+'.npz')).is_file()
    state('checking',step='A3_resume_existing_passed_gate',completed_records=len(list((ROOT/'attn_pilot10').glob('*.npz'))))
   else:
    step='A2';runstep(step,base+['--limit','2']+settings,env)
    logs=[json.loads(l) for l in (ROOT/'attn_pilot10/replay_log.jsonl').read_text().splitlines()]
    assert len(logs)==4 and all(r.get('ok') for r in logs),logs
    step='A3';pilot=validate(step,4,env)
   # The A2 process exits before validation; another user may take the GPU in that gap.
   # Recheck immediately before the next model load instead of assuming it stayed free.
   q4=subprocess.run(['nvidia-smi','-i',str(GPU),'--query-gpu=uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
   a4=next(csv.reader(StringIO(q4.stdout)))
   p4=subprocess.run(['nvidia-smi','-i',str(GPU),'--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
   if p4.stdout.strip() or int(a4[1])>1024 or int(a4[2])<45000 or int(a4[3])!=0:
    state('waiting_for_gpu',step='before_A4_model',gpu_query=q4.stdout,process_query=p4.stdout);return
   step='A4_pilot10';runstep(step,base+settings,env)
   logs=[json.loads(l) for l in (ROOT/'attn_pilot10/replay_log.jsonl').read_text().splitlines()]
   assert len(logs)==20 and all(r.get('ok') for r in logs)
   step='A4_pilot10_validation';full=validate(step,20,env)
   step='A5_pilot10'
   heatmaps=json.loads((ROOT/'pilot10_ids.json').read_text())
   runstep(step,[ROOT/'source/run.py','attn-analyze','--ref','plain_neutral='+str(ROOT/'attn_pilot10')+'@neutral','--cmp','plain_wrong='+str(ROOT/'attn_pilot10')+'@user_wrong','--runs',ROOT/'runs/plain',ROOT/'runs/plain_pressure_reference','--heatmaps']+heatmaps+['--out',ROOT/'attn_pilot10_report'],env)
   state('complete_pilot10',step='A5_pilot10',sequences=20,paired_questions=10,pilot_validation=pilot,full_validation=full,gpu_released=True,task_B_started=False)
  except BaseException as e:
   state('failed',step=step,error=repr(e),task_B_started=False)
   (ROOT/'failure_traceback.log').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
