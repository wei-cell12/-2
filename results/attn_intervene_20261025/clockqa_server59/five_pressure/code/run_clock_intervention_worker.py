"""One load per card, preserve model across gate and approved intervention batches."""
import fcntl,gc,json,os,sys,time,traceback
from pathlib import Path
from workflow_common import *
import intervention_queue as queue
def verify_resources():
 import subprocess
 a=json.loads((ROOT/'allocation_request.json').read_text());card=a['physical_gpus'][0];assert card in [0,1,2,3]
 rows=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=20).stdout
 found=next([x.strip() for x in line.split(',')] for line in rows.splitlines() if int(line.split(',')[0])==card)
 assert found[1]==a['uuids'][0] and int(found[2])>=73728
 apps=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=20).stdout
 allowed=set(a['temporary_holder_pids'])|{os.getpid()}
 assert all(int(line.split(',')[1]) in allowed for line in apps.splitlines() if line.split(',')[0].strip()==found[1])
def main():
 worker=int(os.environ['CLOCKQA_WORKER_INDEX']);verify_resources();sys.path.insert(0,str(ROOT/'source'))
 assert json.loads((ROOT/'cpu_preflight.json').read_text())['passed']
 b=backend();write(ROOT/'model_ready.json',dict(pid=os.getpid(),worker=worker,time_unix=time.time()))
 if all((queue.PARENT/'workers'/str(i)/'model_ready.json').exists() for i in range(4)):
  (queue.PARENT.parent/'intervention_transition/release_temporary_reservation').touch()
 from run_intervention import ensure_reference,generate,CONDITIONS
 installed=False;last=0;reason='complete'
 while True:
  action=queue.claim(worker)
  if action['task']=='finish':break
  if action['task']=='wait':
   if time.time()>config()['transition_deadline_unix']:reason='bounded_wait_expired_no_current_work';break
   if time.time()-last>30:state('waiting_for_gate_or_same_luna_candidates',worker=worker);last=time.time()
   time.sleep(2);continue
  item=action['item'];ensure_reference(b,item)
  if action['task']=='reference':queue.reference_complete(worker,item['id']);continue
  if not installed:
   import intervention
   layers=intervention.install(b.model);write(ROOT/'module_check.json',dict(layers=layers,linear_layers_untouched=48));installed=True
  for method in CONDITIONS:generate(b,item,method)
  queue.complete(worker,item['id'],ROOT)
 del b;gc.collect()
 import torch
 torch.cuda.empty_cache();state('worker_complete',worker=worker,reason=reason,model_released=True);queue.finished(worker,reason)
if __name__=='__main__':
 with (ROOT/'runner.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:main()
  except BaseException as e:
   write(ROOT/'failure.json',dict(error=repr(e),time_unix=time.time()));(ROOT/'failure_traceback.log').write_text(traceback.format_exc());state('failed',error=repr(e));queue.fail(int(os.environ['CLOCKQA_WORKER_INDEX']),e)
   (queue.PARENT.parent/'intervention_transition/release_temporary_reservation').touch();raise
