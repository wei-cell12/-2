"""One handoff from explicit temporary reservation to four experiment models."""
import fcntl,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
RUNTIME='/home/linxin/miniconda3/envs/vllm_new/bin/python'
def write(p,x):
 t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def alive(pid):
 p=Path('/proc')/str(pid)/'stat';return p.exists() and p.read_text().split(') ',1)[1][0]!='Z'
def main():
 with (ROOT/'launch.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);receipt=ROOT/'launch_receipt.json'
  if receipt.exists():
   old=json.loads(receipt.read_text())
   if any(alive(w['pid']) for w in old['workers']):print(json.dumps(dict(status='already_running',**old)));return
   raise RuntimeError('Inspect previous exited workers before retry')
  assert json.loads((ROOT/'cpu_preflight.json').read_text())['passed']
  selection=json.loads((ROOT/'luna_approved_candidates.json').read_text());assert len(selection['items'])>=2
  holderroot=ROOT.parent/'intervention_transition';holder=json.loads((holderroot/'temporary_reservation_state.json').read_text())
  assert holder['status']=='reserved' and alive(holder['pid']) and len(holder['cards'])==4
  assert {c['index'] for c in holder['cards']}=={0,1,2,3}
  (holderroot/'handoff_reservation_memory').touch();deadline=time.time()+45
  while True:
   current=json.loads((holderroot/'temporary_reservation_state.json').read_text())
   if current['status']=='handoff_small_context_only':break
   assert time.time()<deadline,current
   time.sleep(.2)
  workers=[]
  for worker,card in enumerate(sorted(holder['cards'],key=lambda c:c['index'])):
   wr=ROOT/'workers'/str(worker);allocation=dict(physical_gpus=[card['index']],uuids=[card['uuid']],max_memory={'0':'70GiB','cpu':'128GiB'},all_gpu=True,temporary_holder_pids=[holder['pid']],time_unix=time.time())
   write(wr/'allocation_request.json',allocation)
   env=os.environ.copy();env.update(CLOCKQA_INTERVENTION_ROOT=str(ROOT),CLOCKQA_WORKER_INDEX=str(worker),CUDA_VISIBLE_DEVICES=card['uuid'],PATHVQA_PHYSICAL_GPU=str(card['index']),PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
   probe=subprocess.run([RUNTIME,'-c',"import ctypes; n=ctypes.CDLL('libcuda.so.1').cuInit(0); print(n); raise SystemExit(n!=0)"],env=env,capture_output=True,text=True,timeout=25)
   assert probe.returncode==0 and probe.stdout.strip()=='0',(card,probe.stdout,probe.stderr)
   with (wr/'runner.log').open('ab') as stream:
    proc=subprocess.Popen([RUNTIME,'-u',str(wr/'run_clock_intervention_worker.py')],cwd=wr,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
   workers.append(dict(worker=worker,gpu=card['index'],pid=proc.pid,root=str(wr),log=str(wr/'runner.log')))
   write(receipt,dict(workers=workers,phase='intervention_pilot_then_full',same_models_kept_across_gate=True,approved_questions=len(selection['items']),time_unix=time.time()))
  write(ROOT/'job_state.json',dict(status='loading_four_gpu_intervention',workers=workers,time_unix=time.time()))
  print(json.dumps(json.loads(receipt.read_text())),flush=True)
if __name__=='__main__':main()
