"""One resource inspection per invocation; no polling loop or GPU reservation."""
import csv,fcntl,io,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PYTHON='/mnt/sdg/linxin/pathvqa-attn-region-20261024/env/bin/python'

def dump(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');os.replace(tmp,path)

def main():
    with (ROOT/'launcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (ROOT/'failure.json').exists(),'Unresolved failure; inspect, never blindly retry'
        repair=json.loads((ROOT/'repair_validation.json').read_text());assert repair['ok']
        current=json.loads((ROOT/'job_state.json').read_text())
        receipt=json.loads((ROOT/'launch_receipt.json').read_text()) if (ROOT/'launch_receipt.json').exists() else None
        if receipt:
            cmdline=Path('/proc')/str(receipt['pid'])/'cmdline'
            if cmdline.exists() and b'run_experiment.py' in cmdline.read_bytes():
                print(json.dumps(dict(status='already_running',launch=receipt,current=current)));return
        if current['status']=='complete':print(json.dumps(current));return
        if current['status']=='pilot_ready_for_review':
            if not (ROOT/'pilot_review.json').exists():
                print(json.dumps(dict(status='needs_pilot_reply_review',current=current)));return
            review=json.loads((ROOT/'pilot_review.json').read_text());assert review['ok']
            phase='full'
        else:
            assert current['status']=='waiting_for_gpu',current
            phase=current.get('phase','pilot')
        q=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        p=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        used={row[0].strip() for row in csv.reader(io.StringIO(p.stdout)) if row}
        candidates=[]
        for row in csv.reader(io.StringIO(q.stdout)):
            if not row:continue
            gpu=int(row[0]);uuid=row[1].strip()
            if gpu!=7 and uuid not in used and int(row[2])<=1024 and int(row[3])>=45000 and int(row[4])==0:candidates.append((gpu,uuid))
        resource=dict(time_unix=time.time(),gpus=q.stdout,processes=p.stdout,candidates=candidates,exclude_gpu7=True)
        dump(ROOT/'latest_resource_check.json',resource)
        if not candidates:
            dump(ROOT/'job_state.json',dict(status='waiting_for_gpu',phase=phase,repair_passed=True,time_unix=time.time()))
            print(json.dumps(dict(status='waiting_for_gpu',phase=phase)));return
        gpu,uuid=candidates[0]
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=uuid,PATHVQA_PHYSICAL_GPU=str(gpu),PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
        probe=subprocess.run([PYTHON,'-c','import ctypes,json;lib=ctypes.CDLL("libcuda.so.1");rc=lib.cuInit(0);print(json.dumps({"cuInit_returncode":rc}));raise SystemExit(0 if rc==0 else 1)'],env=env,capture_output=True,text=True,timeout=30)
        dump(ROOT/'cuda_health.json',dict(gpu=gpu,uuid=uuid,returncode=probe.returncode,stdout=probe.stdout,stderr=probe.stderr))
        if probe.returncode:
            dump(ROOT/'job_state.json',dict(status='waiting_for_gpu',phase=phase,health='CUDA unavailable',gpu=gpu));print(json.dumps(dict(status='waiting_for_cuda_health',gpu=gpu)));return
        # Immediate selected-card recheck before spawning the model process.
        q2=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        p2=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        row=next(csv.reader(io.StringIO(q2.stdout)))
        if p2.stdout.strip() or int(row[0])>1024 or int(row[1])<45000 or int(row[2])!=0:
            print(json.dumps(dict(status='resource_changed_before_launch',gpu=gpu)));return
        log=ROOT/(phase+'_attempt_'+str(time.time_ns())+'.log')
        with log.open('w') as f:
            proc=subprocess.Popen([PYTHON,str(ROOT/'run_experiment.py'),'--phase',phase],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        receipt=dict(pid=proc.pid,phase=phase,gpu=gpu,uuid=uuid,log=str(log),time_unix=time.time(),exclude_gpu7=True)
        dump(ROOT/'launch_receipt.json',receipt)
        print(json.dumps(dict(status='launched',launch=receipt)))
if __name__=='__main__':main()
