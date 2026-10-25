"""Finite two-GPU model worker: native observation plus approved interventions."""
import fcntl,gc,json,os,sys,time,traceback
from pathlib import Path
from workflow_common import *
import path_queue as queue

def reference_from_online(b,item):
    import numpy as np
    from references import construct
    dest=ROOT/'references'/(item['id']+'.npz');meta=ROOT/'reference_meta'/(item['id']+'.json')
    if dest.exists():assert meta.exists();return
    neutral=Path(item['neutral_record']);row=json.loads(neutral.read_text())
    attention=neutral.parent/'attention.npz';assert sha(attention)==row['attention_sha256']
    with np.load(attention) as z:maps=z['stage_image_attention'];counts=z['stage_token_counts'];grid=z['image_grid_thw']
    assert (counts>0).all(),'No stage reference; preserve reply for semantic/phase review'
    n=len(row['regions']['image']);assert maps.shape==(2,16,24,n)
    assert len(row['attention_checks']['selfchecks'])==48 and max(x['max_abs_diff'] for x in row['attention_checks']['selfchecks'])<.05
    merged=grid[1:]//b.processor.image_processor.merge_size;assert int(np.prod(merged))==n
    dest.parent.mkdir(exist_ok=True)
    metadata=construct(item['id'],dict(think=maps[0],answer=maps[1],n_img=n,grid_hw=merged),dest)
    metadata.update(source='same generation online maps; no replay required',neutral_record=str(neutral),
                    neutral_record_sha256=sha(neutral),attention_sha256=sha(attention),reference_sha256=sha(dest),stage_counts=counts.tolist())
    write(meta,metadata)

def main():
    sys.path.insert(0,str(ROOT/'source'));worker=int(os.environ['PATHVQA_WORKER_INDEX'])
    c=config();gate=json.loads((ROOT/'cpu_preflight.json').read_text())
    assert gate['passed'] and gate['config_sha256']==sha(ROOT/'config.json')
    allocation=json.loads((ROOT/'allocation_request.json').read_text());assert len(allocation['physical_gpus'])==2 and 7 not in allocation['physical_gpus']
    from launch_pathvqa import verify_cards
    verify_cards(allocation['physical_gpus'],allocation['uuids'])
    from fast_observe import Observer
    from fast_generate import generate_one
    observer=Observer(c['layers'],24);b=backend(observer)
    byid={r['item_id']:r for r in catalog()};frozen={r['record_id']:r for r in readlines(ROOT/'tokenized_inputs.jsonl')}
    installed=False;saved=0
    while True:
        action=queue.claim(worker)
        if action['task']=='finish':break
        if action['task']=='wait':state('waiting_for_neutral_dependency',worker=worker);time.sleep(1);continue
        if action['task']=='observe':
            ident=action['id'];item=byid[ident];condition=item['generation_condition']
            observer.active=False;b.model.set_attn_implementation({'text_config':'mathvision_spatial_readonly','vision_config':'sdpa'})
            generate_one(b,observer,item,condition,frozen[ident+'__'+condition])
            queue.complete(worker,ident,ROOT/'records'/(ident+'__'+condition)/'record.json');saved+=1
            state('observation_saved',worker=worker,record_id=ident+'__'+condition,saved=saved)
            continue
        assert action['task']=='intervene';item=action['item']
        observer.active=False;b.model.set_attn_implementation({'text_config':'sdpa','vision_config':'sdpa'})
        reference_from_online(b,item)
        import intervention
        if not installed:
            layers=intervention.install(b.model);installed=True
            write(ROOT/'intervention_module_check.json',dict(layers=layers,linear_layers_untouched=48))
        from run_intervention import generate,CONDITIONS
        for condition in CONDITIONS:generate(b,item,condition)
        queue.intervention_complete(worker,item['id'],ROOT)
    observer.active=False;del b;gc.collect()
    import torch
    torch.cuda.empty_cache();state('worker_complete',worker=worker,saved=saved,model_released=True)
    queue.finished(worker)

if __name__=='__main__':
    with (ROOT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:main()
        except BaseException as e:
            write(ROOT/'failure.json',dict(error=repr(e),time_unix=time.time()))
            (ROOT/'failure_traceback.log').write_text(traceback.format_exc());state('failed',error=repr(e))
            queue.fail(int(os.environ['PATHVQA_WORKER_INDEX']),e);raise

