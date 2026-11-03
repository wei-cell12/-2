from pathlib import Path
B=Path(__file__).resolve().parent;P=B.parent/'shared_prefix_entropy_multi12_20261103'
s=(P/'multi_worker.py').read_text().replace('Twelve questions, twenty-four existing trajectories','Three datasets,50questions per dataset,existing trajectories only')
s=s.replace("R=Path('/mnt/sdb/linxin/conflict-cq-20261028')\nREPO=Path('/mnt/sdb/linxin/pathvqa-fp-a800-20261027/repo')\nC=REPO/'configs/qwen38_fp_local.yaml'\nsys.path.insert(0,str(REPO))\nMANIFEST=json.loads((W/'manifest.json').read_text())", "MANIFEST=json.loads((W/'manifest.json').read_text())\nREPO=Path(MANIFEST['repo']);C=Path(MANIFEST['config']);sys.path.insert(0,str(REPO))\nimport argparse\nparser=argparse.ArgumentParser();parser.add_argument('--rank',type=int,required=True);args=parser.parse_args();RANK=args.rank\nRECORDS=W/'records';RECORDS.mkdir(exist_ok=True)\nJOBS=[j for j in MANIFEST['jobs'] if j['rank']==RANK]")
s=s.replace("gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),generated_tokens=0)","gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),generated_tokens=0,model=MANIFEST['model'],rank=RANK,total=len(JOBS))")
s=s.replace("def main():",'''def load_backend(cfg):
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
    ver=tuple(int(x) for x in re.findall(r"\\d+",transformers.__version__)[:2]);dkey='dtype' if ver>=(4,56) else 'torch_dtype'
    kw={dkey:getattr(torch,cfg.torch_dtype),'device_map':cfg.device_map,'trust_remote_code':True,'attn_implementation':cfg.attn_implementation,'max_memory':{int(k):v for k,v in MANIFEST['max_memory'].items()}}
    b.model=cls.from_pretrained(cfg.model_path,**kw).eval();b.lock=threading.Lock()
    ids=b.tokenizer.encode(cfg.think_close,add_special_tokens=False);b.close_id=ids[0] if cfg.hf_single_pass and len(ids)==1 else None
    assert b.close_id is not None
    return b

def main():''')
s=s.replace("for job in MANIFEST['jobs']:","for job in JOBS:")
s=s.replace("backend=HFBackend(cfg)","backend=load_backend(cfg)")
s=s.replace("group=job['group']","group=job.get('group','join_existing_judgments_after_replay')")
s=s.replace("write(W/'state.json',state)","write(W/f'state_{RANK}.json',state)")
s=s.replace("write(W/'failure.json',state)","write(W/f'failure_{RANK}.json',state)")
s=s.replace("W/(key+'.csv')","RECORDS/(key+'.csv')").replace("W/(key+'.json')","RECORDS/(key+'.json')")
s=s.replace("old/(iid+'.json')","old/(key+'.json')").replace("old/(iid+'.csv')","old/(key+'.csv')")
s=s.replace("summary.update(key=key,dataset=job['dataset'],group=group,condition=cond,reused_from=str(old),new_forward=False)","summary.update(key=key,model=MANIFEST['model'],dataset=job['dataset'],group=group,condition=cond,reused_from=str(old),new_forward=False)")
s=s.replace("summary=dict(id=iid,key=key,dataset=job['dataset']","summary=dict(id=iid,key=key,model=MANIFEST['model'],dataset=job['dataset']")
s=s.replace("        assert rec['id']==iid and rec['condition']==cond", "        assert rec['id']==iid and rec['condition']==cond\n        if (RECORDS/(key+'.json')).exists():\n            done=json.loads((RECORDS/(key+'.json')).read_text());assert done['source_sha256']==src_hash and (RECORDS/(key+'.csv')).exists();summaries.append(done);continue")
s=s.replace("assert len(summaries)==24","assert len(summaries)==len(JOBS)")
a=s.index("    write(W/'summary.json'");b=s.index("    state.update(phase='complete'",a)
s=s[:a]+"    write(W/f'summary_{RANK}.json',dict(cases=summaries,model=MANIFEST['model'],rank=RANK,config_sha256=digest(C),manifest_sha256=digest(W/'manifest.json'),trajectories=len(summaries),new_replays=sum(c['new_forward'] for c in summaries)))\n"+s[b:]
compile(s,'worker.py','exec');(B/'worker.py').write_text(s)
for n in ['entropy_core.py','test_entropy.py']:(B/n).write_bytes((P/n).read_bytes())
print('worker',len(s))
