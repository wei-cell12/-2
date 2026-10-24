"""Fail-stop staged runner: CPU/preflight -> 2x3 pilot -> reviewed gate -> 30x3.

Each invocation uses one empty non-7 GPU, never a reservation process. A pilot
invocation exits and releases the model before the agent reads its six replies.
Full invocation requires a separate SHA-bound review receipt; no automatic retry.
"""
import argparse,datetime,fcntl,hashlib,json,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OLD=Path('/mnt/sdg/linxin/pathvqa-attn-region-20261024')
sys.path.insert(0,str(ROOT/'source'))
MODEL='/mnt/ssd/gy/models/Qwen3.8-27B'
CONDITIONS=['base','restore_E','restore_rand']

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');os.replace(temp,path)
def append(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:f.write(json.dumps(data,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def state(status,**kw):write(ROOT/'job_state.json',dict(status=status,pid=os.getpid(),time_beijing=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),**kw))
def readjsonl(path):return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]

def preflight(items):
    import numpy as np
    manifest={(r['id'],r['condition']):r for r in readjsonl(OLD/'manifest_plain.jsonl')}
    originals={(r['id'],r['condition']):r for r in readjsonl(ROOT/'original_records.jsonl')}
    weight=json.loads((OLD/'existing_27b_weight_sha256.json').read_text());assert weight['ok'] and len(weight['checks'])==19
    a1=json.loads((OLD/'A1_validation.json').read_text());assert a1['ok'] and a1['manifest_sha256']==sha(OLD/'manifest_plain.jsonl')
    checks=[]
    for item in items:
        for cond in ['neutral','user_wrong']:
            r=manifest[item['id'],cond];o=originals[item['id'],cond]
            assert r['input_ids']==o['input_ids'] and r['output_ids']==o['output_ids']
            assert sha(r['image'])==o['image_sha256']
            prior=np.load(OLD/'original_attention'/f"{item['id']}__{cond}.npy",mmap_mode='r')
            assert prior.shape==(len(r['output_ids']),16,24) and prior.dtype==np.float32
            checks.append(dict(id=item['id'],condition=cond,prompt_tokens=len(r['input_ids']),saved_output_tokens=len(r['output_ids']),image_sha256=o['image_sha256']))
    write(ROOT/'preflight_data.json',dict(ok=True,records=checks,source_commit='75bbd3fa0ebc3723815f231fd7a22b95a805a78d',weight_identity_receipt_sha256=sha(OLD/'existing_27b_weight_sha256.json')))
    return manifest,originals

def ensure_reference(backend,item,row):
    import numpy as np
    from PIL import Image
    from symexp.attn import replay_item,save_npz
    from references import construct,validate_replay
    dest=ROOT/'references'/f"{item['id']}.npz"
    if dest.exists():
        assert (ROOT/'reference_meta'/f"{item['id']}.json").exists()
        return
    start=time.time();old=OLD/'attn_pilot10'/f"{item['id']}__neutral.npz"
    if old.exists():
        with np.load(old) as z:d={k:z[k] for k in z.files}
        origin=str(old);reused=True
    else:
        state('running',step='neutral_replay',item=item['id'])
        backend.model.model.rope_deltas=None
        image=Image.open(row['image']).convert('RGB')
        d=replay_item(backend,row['input_ids']+row['output_ids'],len(row['input_ids']),image)
        origin=str(ROOT/'neutral_replay'/f"{item['id']}__neutral.npz");save_npz(origin,d);reused=False
    check=validate_replay(d,row)
    # Same original-image-total check as the previous replay task, no tolerance changes.
    saved=np.load(OLD/'original_attention'/f"{item['id']}__neutral.npy")
    mass=d['pos_image_mass'].astype(np.float64);flat=saved.astype(np.float64).ravel();other=mass.ravel()
    corr=float(np.corrcoef(flat,other)[0,1]);mad=float(np.abs(flat-other).mean())
    assert corr>=.99 and mad<=.01,(item['id'],'replay agreement',corr,mad)
    check.update(corr=corr,mean_abs_difference=mad,reused=reused,source=origin,source_sha256=sha(origin),seconds=time.time()-start)
    append(ROOT/'neutral_replay_checks.jsonl',check)
    dest.parent.mkdir(exist_ok=True);meta=construct(item['id'],d,dest)
    meta.update(neutral_source=origin,neutral_source_sha256=sha(origin),reference_sha256=sha(dest))
    write(ROOT/'reference_meta'/f"{item['id']}.json",meta)

class Budget:
    """Original preexperiment ThinkingBudget, copied without parameter changes."""
    def __init__(self,n_in,close,eos,tb,ab):self.n_in=n_in;self.close=close;self.eos=eos;self.tb=tb;self.ab=ab;self.at=None;self.forced_close=False;self.forced_end=False
    def __call__(self,input_ids,scores):
        n=input_ids.shape[-1]-self.n_in
        if n and self.at is None and int(input_ids[0,-1])==self.close:self.at=n-1
        token=None
        if self.at is None and n>=self.tb:token=self.close;self.forced_close=True
        elif self.at is not None and n-self.at-1>=self.ab:token=self.eos;self.forced_end=True
        if token is None:return scores
        out=scores.new_full(scores.shape,float('-inf'));out[:,token]=0.;return out

def generate(backend,item,row,original,cond):
    import numpy as np
    import torch,transformers
    from PIL import Image
    from symexp.attn import image_token_id,merge_size
    from symexp.parsing import rule_answer
    import intervention
    finalpath=ROOT/'runs'/cond/'final.jsonl'
    if finalpath.exists() and any(r['id']==item['id'] for r in readjsonl(finalpath)):return
    start=time.time(); dev=backend.model.device;ids=row['input_ids'];tok=backend.tokenizer
    image=Image.open(row['image']).convert('RGB')
    # Processor is checked against the saved prompt, then the original IDs are used.
    batch=backend.processor(text=[original['rendered_chat_template']],images=[image],return_tensors='pt')
    assert batch['input_ids'][0].tolist()==ids,'Prompt changed relative to saved plain'
    assert 'mm_token_type_ids' in batch,'mm_token_type_ids missing'
    mm=backend.processor.create_mm_token_type_ids([ids]);assert batch['mm_token_type_ids'].tolist()==mm
    batch={k:v.to(dev) for k,v in batch.items()};batch['pixel_values']=batch['pixel_values'].to(backend.model.dtype)
    img=(batch['input_ids'][0]==image_token_id(backend)).nonzero().flatten().cpu().numpy()
    grid=batch['image_grid_thw'][0].cpu().tolist();expect=int(np.prod(grid))//merge_size(backend)**2
    refs=dict(np.load(ROOT/'references'/f"{item['id']}.npz"))
    assert len(img)==expect==int(refs['n_img'])
    ctrl=intervention.Controller(torch,refs,img,cond,backend.close_id)
    # before_forward establishes the stage from the actually generated close token.
    handle=backend.model.register_forward_pre_hook(ctrl.before_forward,with_kwargs=True)
    def progress(module,args,kwargs):
        if ctrl.decode and ctrl.step%32==0:state('running',step='generation',item=item['id'],condition=cond,decode_steps=ctrl.step,stage='answer' if ctrl.stage else 'think',gpu=int(os.environ['PATHVQA_PHYSICAL_GPU']))
    progress_handle=backend.model.register_forward_pre_hook(progress,with_kwargs=True)
    gc=transformers.GenerationConfig.from_pretrained(MODEL,local_files_only=True)
    gc.do_sample=True;gc.max_new_tokens=4096+512+2;gc.use_cache=True
    gc.temperature=1.;gc.top_p=.95;gc.top_k=20;gc.min_p=0.;gc.repetition_penalty=1.;gc.return_dict_in_generate=False
    eos=gc.eos_token_id[0] if isinstance(gc.eos_token_id,list) else gc.eos_token_id
    budget=Budget(len(ids),backend.close_id,eos,4096,512)
    backend.model.model.rope_deltas=None
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    state('running',step='generation',item=item['id'],condition=cond,decode_steps=0,gpu=int(os.environ['PATHVQA_PHYSICAL_GPU']))
    intervention.ACTIVE=ctrl
    try:
        with torch.inference_mode():out=backend.model.generate(**batch,generation_config=gc,logits_processor=transformers.LogitsProcessorList([budget]))
    finally:
        intervention.ACTIVE=None;handle.remove();progress_handle.remove()
    output=out[0,len(ids):].tolist();assert len(output)>3
    close=output.index(backend.close_id) if backend.close_id in output else None
    raw=tok.decode(output,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    thinking=tok.decode(output[:close],skip_special_tokens=True).strip() if close is not None else raw
    answer=tok.decode(output[close+1:],skip_special_tokens=True).strip() if close is not None else ''
    tracepath=ROOT/'traces'/f"{item['id']}__{cond}.npz";tracepath.parent.mkdir(exist_ok=True)
    metrics=ctrl.save(tracepath)
    record=dict(id=item['id'],method=cond,condition='user_wrong',ptype=item['ptype'],seed=42,gold=item['gold'],claim=item['claim'],question=item['question'],message=item['pressure_message'],reply=answer,thinking=thinking,raw_output=raw,rule_pred=rule_answer(answer),joint_correct=None,explanation_correct=None,judgment='rule_only_pending_semantic_review',input_tokens=len(ids),gen_tokens=len(output),seconds=time.time()-start,thinking_closed=close is not None,forced_close=budget.forced_close,forced_answer_end=budget.forced_end,metrics=metrics,trace_path=str(tracepath))
    # Output content/normality is inspected before advancing to the 30-item phase.
    normal=dict(nonempty=bool(answer.strip()),replacement_character='\ufffd' in raw,repeated_sentence=False)
    sentences=[s.strip() for s in __import__('re').split(r'[.!?\n]+',raw) if len(s.strip())>=25]
    if sentences:normal['repeated_sentence']=max(sentences.count(s) for s in set(sentences))>=8
    record['normality_checks']=normal
    append(ROOT/'runs'/cond/'calls.jsonl',record)
    write(ROOT/'output_tokens'/f"{item['id']}__{cond}.json",dict(input_ids=ids,output_ids=output))
    append(finalpath,record)
    if not normal['nonempty'] or normal['replacement_character'] or normal['repeated_sentence']:raise ValueError(f'Reply normality needs review: {item["id"]} {cond} {normal}')
    print(json.dumps(dict(event='completed_reply',id=item['id'],condition=cond,answer=record['rule_pred'],tokens=len(output),seconds=record['seconds'],metrics={k:v for k,v in metrics.items() if k not in ['selfchecks','by_stage']})),flush=True)

def pilot_gate(items):
    rows=[r for c in CONDITIONS for r in readjsonl(ROOT/'runs'/c/'final.jsonl') if r['id'] in {x['id'] for x in items[:2]}]
    assert len(rows)==6
    for r in rows:
        m=r['metrics'];assert m['image_diff_max']<1e-3 and m['text_diff_max']<1e-3 and m['row_diff_max']<1e-3
        if r['method']=='base':assert len(m['selfchecks'])==48 and max(x['max_abs_diff'] for x in m['selfchecks'])<.05
        else:assert m['active_head_tokens']>0 and m['region_after']>m['region_before']
        assert r['normality_checks']==dict(nonempty=True,replacement_character=False,repeated_sentence=False)
    write(ROOT/'pilot_numerical_gate.json',dict(ok=True,questions=[x['id'] for x in items[:2]],conditions=CONDITIONS,rows=rows,reply_manual_review_required=True,gate='image/text/row differences <1e-3; lambda0 all16layers x3steps <.05; region grows on nonzero deficits; per-head/token complete traces retained'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['pilot','full'],required=True);args=p.parse_args()
    with (ROOT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'failure.json').exists():raise RuntimeError('Failure scene exists; no automatic retry')
        stage='preflight';start=time.time()
        try:
            import torch,transformers
            assert torch.__version__=='2.11.0+cu130' and transformers.__version__=='5.14.1'
            items=json.loads((ROOT/'items.json').read_text());assert len(items)==30
            if args.phase=='full':
                review=json.loads((ROOT/'pilot_review.json').read_text())
                assert review['ok'] and review['pilot_gate_sha256']==sha(ROOT/'pilot_numerical_gate.json')
            state('checking',step=stage,phase=args.phase)
            manifest,originals=preflight(items)
            import numerical_tests
            numerical_tests.main()
            gpu=int(os.environ['PATHVQA_PHYSICAL_GPU']);assert gpu!=7
            import csv,io,subprocess
            q=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
            a=next(csv.reader(io.StringIO(q.stdout)))
            processes=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
            if processes.stdout.strip() or int(a[1])>1024 or int(a[2])<45000 or int(a[3])!=0:
                state('waiting_for_gpu',phase=args.phase,resource=q.stdout,processes=processes.stdout);return
            assert os.environ['CUDA_VISIBLE_DEVICES']==a[0].strip()
            stage='load_model';state('running',step=stage,phase=args.phase,gpu=gpu)
            from symexp.config import Config
            from symexp.backends import HFBackend
            cfg=Config.load(str(ROOT/'source/configs/qwen38_hf.yaml'),dict(model_path=MODEL,attn_implementation='sdpa'))
            write(ROOT/'effective_config.json',dict(config=cfg.to_dict(),seed=42,max_memory={'gpu':'36GiB','cpu':'128GiB'},stages=['think','answer'],prefill=False,lambda_main=1,random_seed=20261024,generation_prompt='exact saved user_wrong input_ids; processor exact rendered-template identity check',budget='original ThinkingBudget with forced close at4096, forced EOS after512 answer tokens',source_commit='75bbd3fa0ebc3723815f231fd7a22b95a805a78d'))
            cls=transformers.Qwen3_5ForConditionalGeneration;orig_load=cls.from_pretrained
            def loader(*a,**kw):
                assert kw['torch_dtype']==torch.bfloat16;kw['max_memory']={0:'36GiB','cpu':'128GiB'}
                model=orig_load(*a,**kw)
                assert all(str(d) in ['0','cuda:0','cpu'] for d in model.hf_device_map.values())
                write(ROOT/('model_allocation_'+args.phase+'.json'),dict(gpu=gpu,uuid=os.environ['CUDA_VISIBLE_DEVICES'],dtype=str(model.dtype),device_map={k:str(v) for k,v in model.hf_device_map.items()}))
                return model
            cls.from_pretrained=staticmethod(loader)
            backend=HFBackend(cfg);assert backend.close_id is not None
            import intervention
            # Neutral replay can install its recorder wrapper; restoration wraps it after references.
            selected=items[:2] if args.phase=='pilot' else items
            stage='neutral_references'
            for item in selected:ensure_reference(backend,item,manifest[item['id'],'neutral'])
            write(ROOT/'reference_metadata.json',dict(items=[json.loads((ROOT/'reference_meta'/f"{x['id']}.json").read_text()) for x in selected]))
            layers=intervention.install(backend.model);write(ROOT/'module_check.json',dict(ok=True,layers=layers,heads_per_layer=24,linear_layers_untouched=48))
            stage='generation'
            for item in selected:
                for cond in CONDITIONS:generate(backend,item,manifest[item['id'],'user_wrong'],originals[item['id'],'user_wrong'],cond)
            if args.phase=='pilot':
                stage='pilot_gate';pilot_gate(items)
                state('pilot_ready_for_review',questions=2,replies=6,seconds=time.time()-start,gpu=gpu,model_process_exits_after_save=True)
            else:
                from evaluate_results import evaluate
                stage='evaluate';evaluate(ROOT)
                state('complete',questions=30,replies=90,seconds=time.time()-start,gpu=gpu,model_process_exits_after_save=True)
        except BaseException as e:
            write(ROOT/'failure.json',dict(step=stage,phase=args.phase,error=repr(e),time_unix=time.time()))
            (ROOT/'failure_traceback.log').write_text(traceback.format_exc())
            state('failed',step=stage,phase=args.phase,error=repr(e));raise
if __name__=='__main__':main()
