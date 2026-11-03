"""Three datasets,50questions per dataset,existing trajectories only; no generation, original records read-only."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from entropy_core import aligned_slice,signals_numpy,signals_torch

W=Path(__file__).resolve().parent
MANIFEST=json.loads((W/'manifest.json').read_text())
REPO=Path(MANIFEST['repo']);C=Path(MANIFEST['config']);sys.path.insert(0,str(REPO))
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--rank',type=int,required=True);args=parser.parse_args();RANK=args.rank
RECORDS=W/'records';RECORDS.mkdir(exist_ok=True)
JOBS=[j for j in MANIFEST['jobs'] if j['rank']==RANK]

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p,x):
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(p)

state=dict(phase='starting',pid=os.getpid(),started=time.time(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),generated_tokens=0,model=MANIFEST['model'],rank=RANK,total=len(JOBS))

def reset(model):
    if hasattr(getattr(model,'model',None),'rope_deltas'):model.model.rope_deltas=None

def predict(backend,inputs,tokens):
    """One causal forward: prompt+y[:-1]; row P-1+k predicts y[k]."""
    t,m=backend.torch,backend.model;dev=m.device
    if not tokens:raise ValueError('Empty trajectory')
    batch={k:v.to(dev) for k,v in inputs.items() if hasattr(v,'to')}
    P=batch['input_ids'].shape[1]
    extra=t.tensor([tokens[:-1]],dtype=batch['input_ids'].dtype,device=dev)
    batch['input_ids']=t.cat([batch['input_ids'],extra],1)
    mask=inputs.get('attention_mask')
    batch['attention_mask']=t.cat([mask.to(dev) if mask is not None else t.ones((1,P),device=dev,dtype=t.long),t.ones_like(extra)],1)
    for name in ['mm_token_type_ids','token_type_ids']:
        if name in batch:batch[name]=t.cat([batch[name],t.zeros_like(extra)],1)
    if 'position_ids' in batch:raise ValueError('Explicit processor position_ids require a model-specific adapter')
    if 'pixel_values' in batch:batch['pixel_values']=batch['pixel_values'].to(m.dtype)
    reset(m)
    with t.inference_mode():out=m(**batch,use_cache=False,return_dict=True)
    logits=out.logits[0,aligned_slice(int(P),len(tokens))].contiguous()
    assert logits.shape[0]==len(tokens)
    del out,batch
    return logits

def load_backend(cfg):
    if MANIFEST['model']=='qwen38':
        from symexp.backends import HFBackend
        return HFBackend(cfg)
    # Original HF backend initialization, with GPU placement cap as the only
    # extra loading argument. Native model forward/attention/logits unmodified.
    import re,threading,torch,transformers
    from transformers import AutoProcessor
    from symexp.backends import HFBackend,Backend
    b=HFBackend.__new__(HFBackend);Backend.__init__(b,cfg)
    b.torch=torch;b.transformers=transformers
    b.processor=AutoProcessor.from_pretrained(cfg.model_path,trust_remote_code=True,**(cfg.processor_kwargs or {}))
    b.tokenizer=getattr(b.processor,'tokenizer',b.processor)
    if cfg.image_max_soft_tokens:b.processor.image_processor.max_soft_tokens=int(cfg.image_max_soft_tokens)
    cls=getattr(transformers,cfg.model_class)
    ver=tuple(int(x) for x in re.findall(r"\d+",transformers.__version__)[:2]);dkey='dtype' if ver>=(4,56) else 'torch_dtype'
    kw={dkey:getattr(torch,cfg.torch_dtype),'device_map':cfg.device_map,'trust_remote_code':True,'attn_implementation':cfg.attn_implementation,'max_memory':{int(k):v for k,v in MANIFEST['max_memory'].items()}}
    b.model=cls.from_pretrained(cfg.model_path,**kw).eval();b.lock=threading.Lock()
    ids=b.tokenizer.encode(cfg.think_close,add_special_tokens=False);b.close_id=ids[0] if cfg.hf_single_pass and len(ids)==1 else None
    assert b.close_id is not None
    return b

def main():
    from symexp.config import Config
    from symexp.backends import HFBackend
    from symexp.fp.generate import build_inputs
    import torch
    import transformers
    cfg=Config.load(str(C))
    for job in JOBS:
        assert digest(job['source'])==job['source_sha256'] and digest(job['neutral_source'])==job['neutral_sha256']
        assert digest(job['item']['image'])==job['image_sha256']
        assert digest(job['config'])==job['config_sha256']
    state.update(phase='loading',torch=torch.__version__,transformers=transformers.__version__,config_sha256=digest(C))
    write(W/f'state_{RANK}.json',state)
    backend=load_backend(cfg)
    summaries=[]
    for job in JOBS:
        started=time.time();iid=job['id'];group=job.get('group','join_existing_judgments_after_replay');cond=job['condition'];key=job['key'];it=job['item']
        src=Path(job['source']);neutral_src=Path(job['neutral_source'])
        src_hash=digest(src);neutral_hash=digest(neutral_src)
        rec=json.loads(src.read_text());nr=json.loads(neutral_src.read_text());toks=rec['output_ids']
        assert rec['id']==iid and rec['condition']==cond
        if (RECORDS/(key+'.json')).exists():
            done=json.loads((RECORDS/(key+'.json')).read_text());assert done['source_sha256']==src_hash and (RECORDS/(key+'.csv')).exists();summaries.append(done);continue
        if job.get('reuse_root'):
            import shutil
            old=Path(job['reuse_root']);summary=json.loads((old/(iid+'.json')).read_text())
            summary.update(key=key,model=MANIFEST['model'],dataset=job['dataset'],group=group,condition=cond,reused_from=str(old),new_forward=False)
            shutil.copyfile(old/(iid+'.csv'),RECORDS/(key+'.csv'))
            write(RECORDS/(key+'.json'),summary);summaries.append(summary)
            print(json.dumps(dict(key=key,reused=True,tokens=len(toks))),flush=True)
            continue
        assert digest(it['image'])==rec['image_sha256']==nr['image_sha256']
        _,p=build_inputs(backend,it['image'],it['messages'][cond])
        _,n=build_inputs(backend,it['image'],it['messages']['neutral'])
        assert p['input_ids'][0].tolist()==rec['input_ids'],'Pressure prompt mismatch'
        assert n['input_ids'][0].tolist()==nr['input_ids'],'Neutral prompt mismatch'
        for name in ['pixel_values','image_grid_thw']:
            if name in p:assert torch.equal(p[name],n[name]),'Image input differs: '+name
        state.update(phase='computing',id=iid,condition=cond,completed=len(summaries),n_out=len(toks));write(W/f'state_{RANK}.json',state)
        lp=predict(backend,p,toks)
        ln=predict(backend,n,toks)
        sg=signals_torch(torch,lp,ln,toks)
        # Independent float64 formula checks on actual full-vocabulary model logits.
        math_errors=[]
        for k in [0,1,len(toks)-1]:
            ref=signals_numpy(lp[k:k+1].float().cpu().numpy(),ln[k:k+1].float().cpu().numpy(),toks[k:k+1])
            math_errors.append(max(abs(sg[name][k]-float(ref[name][0])) for name in ['H_pressure','H_neutral','JSD','delta_H','delta_logp']))
        assert max(math_errors)<1e-4,'Entropy formula precision check failed'
        # Actual extra prefix-only forwards verify indexing, not only a synthetic test.
        checks=[]
        for k in [0,1,15]:
            for branch,inp,full in [('pressure',p,lp),('neutral',n,ln)]:
                short=predict(backend,inp,toks[:k+1])[-1].float()
                ref=full[k].float();x=signals_torch(torch,ref[None],short[None],[toks[k]])
                check=dict(step=k+1,branch=branch,prefix_tokens=k,maxabs_logits=float((ref-short).abs().max()),
                    entropy_absdiff=abs(x['delta_H'][0]),JSD=x['JSD'][0],top1_agrees=x['top1_pressure'][0]==x['top1_neutral'][0])
                checks.append(check)
                # BF16 sequence-length differences need not be bit exact; compare distributions.
                assert check['JSD']<1e-3 and check['entropy_absdiff']<0.1,'Prefix replay alignment check failed'
                del short,ref
        zero=signals_torch(torch,lp[:3],lp[:3],toks[:3])
        assert max(abs(x) for x in zero['delta_H'])==0 and max(abs(x) for x in zero['JSD'])<1e-6
        close=next((k for k,x in enumerate(toks) if x==backend.close_id),None)
        forced=set()
        if rec.get('budget',{}).get('thinking_budget_hit') and close is not None:forced.add(close)
        if rec.get('budget',{}).get('answer_budget_hit'):forced.add(len(toks)-1)
        ordinary=[k for k in range(len(toks)) if k not in forced]
        top20=sum(sg['rank_pressure'][k]<=20 for k in ordinary)/len(ordinary)
        assert top20>=.98,f'Original token replay top20 too low: {top20}'
        rows=[]
        for k,token in enumerate(toks):
            special=token in backend.tokenizer.all_special_ids
            phase='special' if special else 'think' if close is None or k<close else 'answer'
            row=dict(step=k+1,prefix_output_tokens=k,phase=phase,forced_by_original_budget=k in forced,
                token_piece=backend.tokenizer.decode([token],skip_special_tokens=False,clean_up_tokenization_spaces=False))
            row.update({name:values[k] for name,values in sg.items() if not name.startswith('top1_')})
            for branch in ['pressure','neutral']:
                row['top1_'+branch]=backend.tokenizer.decode([sg['top1_'+branch][k]],skip_special_tokens=False,clean_up_tokenization_spaces=False)
            rows.append(row)
        with (RECORDS/(key+'.csv')).open('x',newline='',encoding='utf-8') as f:
            wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
        def avg(indices,name):return float(np.mean([sg[name][k] for k in indices])) if indices else None
        fields=['H_pressure','H_neutral','delta_H','JSD','delta_logp']
        summary=dict(id=iid,key=key,model=MANIFEST['model'],dataset=job['dataset'],group=group,condition=cond,new_forward=True,tokens=len(toks),gold=it['gold'],user_claim=it['claim'],
            pressure_answer=rec['answer'],original_neutral_answer=nr['answer'],
            source_sha256=digest(src),neutral_prompt_record_sha256=digest(neutral_src),image_sha256=rec['image_sha256'],
            output_ids_sha256=hashlib.sha256(json.dumps(toks,separators=(',',':')).encode()).hexdigest(),
            entropy_unit='nats; raw full-vocabulary logits, before sampling filters',
            prediction_alignment='row P-1+k predicts actual pressured output[k] from prompt+output[:k]; neutral never feeds its own top1',
            original_pressure_top20_fraction=top20,formula_max_absdiff=max(math_errors),prefix_checks=checks,
            same_prompt_control=dict(delta_H=zero['delta_H'],JSD=zero['JSD']),
            means={phase:{f:avg([k for k,r in enumerate(rows) if phase=='all' or r['phase']==phase],f) for f in fields} for phase in ['all','think','answer']},
            max_delta=dict(step=int(np.argmax(sg['delta_H']))+1,value=max(sg['delta_H'])),
            min_delta=dict(step=int(np.argmin(sg['delta_H']))+1,value=min(sg['delta_H'])),
            seconds=time.time()-started)
        assert digest(src)==src_hash and digest(neutral_src)==neutral_hash,'Source records changed during analysis'
        write(RECORDS/(key+'.json'),summary);summaries.append(summary)
        del lp,ln
        torch.cuda.empty_cache()
        print(json.dumps(dict(id=iid,condition=cond,tokens=len(toks),top20=top20,means=summary['means'],seconds=summary['seconds'])),flush=True)
    assert len(summaries)==len(JOBS)
    write(W/f'summary_{RANK}.json',dict(cases=summaries,model=MANIFEST['model'],rank=RANK,config_sha256=digest(C),manifest_sha256=digest(W/'manifest.json'),trajectories=len(summaries),new_replays=sum(c['new_forward'] for c in summaries)))
    state.update(phase='complete',finished=time.time(),records=len(summaries));write(W/f'state_{RANK}.json',state)

if __name__=='__main__':
    try:main()
    except Exception as e:
        state.update(phase='failed',error=repr(e),traceback=traceback.format_exc());write(W/f'failure_{RANK}.json',state);write(W/f'state_{RANK}.json',state)
        raise
