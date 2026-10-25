import hashlib,json,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()))
    tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def append(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:f.write(json.dumps(d,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def readlines(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()] if Path(p).exists() else []
def config():return json.loads((ROOT/'config.json').read_text())
def state(status,**kw):write(ROOT/'job_state.json',dict(status=status,pid=os.getpid(),time_unix=time.time(),**kw))
def alive(pid):
    p=Path('/proc')/str(pid)
    if not p.exists():return False
    return p.joinpath('stat').read_text().split(') ',1)[1][0]!='Z'
def old_finished():
    # This model has a separate, regenerated observation corpus.
    return (ROOT/'cpu_preflight.json').exists() and json.loads((ROOT/'cpu_preflight.json').read_text())['passed']
def catalog():return readlines(ROOT/'catalog.jsonl')
def original(id,condition):
    parent=Path(os.environ['CLOCKQA_INTERVENTION_ROOT'])
    item=next(r for r in json.loads((parent/'luna_approved_candidates.json').read_text())['items'] if r['id']==id)
    path=Path(item[condition+'_record']);assert sha(path)==item[condition+'_source_sha256']
    return json.loads(path.read_text())
def import_original():
    raise RuntimeError('Cross-model original observation import is forbidden')
def prepare_cpu():
    import generation_common as g
    c=config();assert sha(ROOT/'catalog.jsonl')==c['catalog_sha256']
    ident=Path(c['model_identity_root'])/'model_identity.json';assert sha(ident)==c['model_identity_sha256']
    d=json.loads(ident.read_text())
    for n,h in d['metadata_sha256'].items():assert sha(Path(c['model_dir'])/n)==h
    for weight in d['weight_files']:
        st=(Path(c['model_dir'])/weight['name']).stat()
        assert st.st_size==weight['bytes'] and st.st_mtime_ns==weight['mtime_ns'],('Weight metadata changed',weight['name'])
    p=g.processor_for(c);frozen=[]
    for item in catalog():
        for cond in ['neutral','pressure']:
            batch,e=g.prepare(item,cond,p,c);frozen.append(e);del batch
    body=''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in frozen)
    path=ROOT/'tokenized_inputs.jsonl'
    if path.exists():assert path.read_text()==body
    else:path.write_text(body)
    from clock_rules import selftest
    write(ROOT/'cpu_preflight.json',dict(passed=True,questions=62,sequences=124,all_inputs_regenerated_for_new_model=True,old_model_references_reused=False,model_identity_sha256=sha(ident),config_sha256=sha(ROOT/'config.json'),tokenized_sha256=sha(path),rule_tests=selftest()))
    return p,{x['record_id']:x for x in frozen}
def backend(observer=None):
    import torch,transformers
    from types import SimpleNamespace
    import generation_common as g
    c=config();a=json.loads((ROOT/'allocation_request.json').read_text())
    assert torch.__version__==c['expected_torch'] and transformers.__version__==c['expected_transformers']
    assert torch.cuda.device_count()==len(a['physical_gpus'])
    p=g.processor_for(c)
    assert c['sdpa_decode_kernel']=='math' and c['sdpa_decode_kernel_change_user_authorized']
    from sdpa_math_decode import install as install_math_decode
    install_math_decode()
    name=observer.register() if observer else 'sdpa'
    memory={int(k) if k.isdigit() else k:v for k,v in a['max_memory'].items()}
    state('loading_model',allocation=a,attention_backend=name)
    model=transformers.Qwen3_5ForConditionalGeneration.from_pretrained(c['model_dir'],local_files_only=True,dtype=torch.bfloat16,
        attn_implementation={'text_config':name,'vision_config':'sdpa'},device_map='sequential',max_memory=memory).eval()
    from collections import Counter
    parameter_devices=Counter(str(p.device) for p in model.parameters())
    assert parameter_devices and all(d.startswith('cuda:') for d in parameter_devices),'All model parameters must actually reside on GPU'
    device_map=getattr(model,'hf_device_map',None)
    close=p.tokenizer.encode('</think>',add_special_tokens=False);assert len(close)==1
    write(ROOT/'latest_model_allocation.json',dict(pid=os.getpid(),physical_gpus=a['physical_gpus'],dtype=str(model.dtype),device_map={k:str(v) for k,v in device_map.items()} if device_map is not None else None,actual_parameter_devices=dict(parameter_devices),allocation_verification='actual parameter devices; hf_device_map optional in transformers5.14',torch=torch.__version__,transformers=transformers.__version__,sdpa_decode_kernel='math',sdpa_math_scope=c['sdpa_decode_kernel_scope']))
    return SimpleNamespace(torch=torch,model=model,processor=p,tokenizer=p.tokenizer,close_id=close[0])
def scoring(record):
    from clock_rules import judge
    return judge(record.get('final_answer',record.get('reply')),record.get('correct_answer',record.get('gold')))
def screen_report():
    items=catalog();byid={x['item_id']:x for x in items};qual=[];pending=[];rows=[]
    pilot=set(config()['old_pilot_ids'])
    for id in sorted(byid,key=lambda i:(i not in pilot,i)):
        paths={k:ROOT/'records'/(id+'__'+k)/'record.json' for k in ['neutral','pressure']}
        if not all(p.exists() for p in paths.values()):continue
        records={k:json.loads(p.read_text()) for k,p in paths.items()};scores={k:scoring(v) for k,v in records.items()}
        row=dict(id=id,neutral=scores['neutral'],pressure=scores['pressure'],source_sha256={k:sha(v) for k,v in paths.items()})
        rows.append(row)
        if any(v['correct'] is None for v in scores.values()):pending.append(row)
        if scores['neutral']['correct'] is True and scores['pressure']['correct'] is False:
            src=byid[id];qual.append(dict(id=id,ptype=src['pressure_type'],gold=src['correct_answer'],claim=src['wrong_answer'],question=src['question'],pressure_message=src['pressure_text'],image=str(ROOT/src['image_path']),image_sha256=src['image_sha256'],neutral_record=str(paths['neutral']),pressure_record=str(paths['pressure']),screening=row,joint_category=None,annotation_status='time-content rule only; joint answer/explanation not assigned'))
    report=dict(paired_screened=len(rows),eligible=len(qual),target=20,unresolved=len(pending),records=rows,pending=pending,time_unix=time.time(),selection='original10 qualifying IDs first, then remaining qualifying IDs ascending; no resampling')
    write(ROOT/'screening_report.json',report)
    if len(qual)>=20:write(ROOT/'items.json',qual[:20])
    return report
