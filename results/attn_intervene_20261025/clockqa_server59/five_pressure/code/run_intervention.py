"""ClockQA two-question gate followed by SHA-reviewed20-question intervention."""
import argparse,fcntl,json,sys,time,traceback
from pathlib import Path
from workflow_common import *
sys.path.insert(0,str(ROOT/'source'))
CONDITIONS=['base','restore_E','restore_rand']
def ensure_reference(b,item):
    import numpy as np
    from PIL import Image
    from symexp.attn import replay_item,save_npz
    from references import construct,validate_replay
    dest=ROOT/'references'/f"{item['id']}.npz";meta=ROOT/'reference_meta'/f"{item['id']}.json"
    if dest.exists():assert meta.exists() and json.loads(meta.read_text())['reference_sha256']==sha(dest);return
    rec=original(item['id'],'neutral');row=dict(rec,id=item['id'])
    assert sha(item['image'])==item['image_sha256']==rec['image_sha256']
    old=Path(config()['prior_experiment_root'])/'neutral_replay'/f"{item['question_id']}__neutral.npz"
    if old.exists():
        with np.load(old) as z:d={k:z[k] for k in z.files}
        source=old;reused=True
    else:
        b.model.model.rope_deltas=None
        d=replay_item(b,rec['input_ids']+rec['output_ids'],len(rec['input_ids']),Image.open(item['image']).convert('RGB'),selfcheck=2)
        source=ROOT/'neutral_replay'/f"{item['id']}__neutral.npz";save_npz(str(source),d);reused=False
    check=validate_replay(d,row)
    with np.load(ROOT/'records'/rec['record_id']/'attention.npz') as z:native=z['generation_masses'][...,0].astype(np.float64)
    replay=d['pos_image_mass'].astype(np.float64);assert native.shape==replay.shape
    corr=float(np.corrcoef(native.ravel(),replay.ravel())[0,1]);mad=float(np.abs(native-replay).mean())
    check.update(corr=corr,mad=mad,rows_mismatch=0,source=str(source),source_sha256=sha(source),reused=reused)
    append(ROOT/'neutral_replay_checks.jsonl',check)
    assert corr>=.99 and mad<=.01,(item['id'],corr,mad)
    dest.parent.mkdir(exist_ok=True);m=construct(item['id'],d,dest);m.update(source=str(source),source_sha256=sha(source),reference_sha256=sha(dest));write(meta,m)
def generate(b,item,cond):
    import numpy as np,torch,transformers
    from PIL import Image
    from symexp.attn import image_token_id,merge_size
    import generation_common as g,intervention
    final=ROOT/'runs'/cond/'final.jsonl'
    if any(x['id']==item['id'] for x in readlines(final)):return
    rec=original(item['id'],'pressure');ids=rec['input_ids'];start=time.time()
    batch=b.processor(text=[rec['rendered_chat_template']],images=[Image.open(item['image']).convert('RGB')],return_tensors='pt')
    assert batch['input_ids'][0].tolist()==ids and batch['mm_token_type_ids'].tolist()==b.processor.create_mm_token_type_ids([ids])
    batch={k:v.to(b.model.device) for k,v in batch.items()};batch['pixel_values']=batch['pixel_values'].to(b.model.dtype)
    img=(batch['input_ids'][0]==image_token_id(b)).nonzero().flatten().cpu().numpy()
    refs=dict(np.load(ROOT/'references'/f"{item['id']}.npz"));assert len(img)==int(refs['n_img'])
    ctrl=intervention.Controller(torch,refs,img,cond,b.close_id)
    h=b.model.register_forward_pre_hook(ctrl.before_forward,with_kwargs=True)
    def progress(module,args,kwargs):
        if ctrl.decode and ctrl.step%64==0:state('intervening',id=item['id'],condition=cond,decode_steps=ctrl.step,stage='answer' if ctrl.stage else 'think')
    h2=b.model.register_forward_pre_hook(progress,with_kwargs=True)
    c=config();gc=transformers.GenerationConfig.from_pretrained(c['model_dir'],local_files_only=True)
    gc.do_sample=True;gc.max_new_tokens=c['max_new_tokens'];gc.use_cache=True;gc.return_dict_in_generate=False
    for k in ['temperature','top_p','top_k','min_p','repetition_penalty']:setattr(gc,k,c['sampling'][k])
    eos=gc.eos_token_id if isinstance(gc.eos_token_id,list) else [gc.eos_token_id]
    budget=g.ThinkingBudget(len(ids),b.close_id,eos[0],4096,512)
    b.model.model.rope_deltas=None;torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    intervention.ACTIVE=ctrl
    state('intervening',id=item['id'],condition=cond,decode_steps=0)
    # Stream partial intervention outputs too; they are never silently resampled.
    live=g.LiveOutput(b.tokenizer,dict(rec,record_id=item['id']+'__'+cond))
    try:
        with torch.inference_mode():out=b.model.generate(**batch,generation_config=gc,streamer=live,logits_processor=transformers.LogitsProcessorList([budget]))
    finally:intervention.ACTIVE=None;h.remove();h2.remove()
    output=out[0,len(ids):].tolist();assert output==live.ids
    close=output.index(b.close_id) if b.close_id in output else None
    raw=b.tokenizer.decode(output,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    thinking=b.tokenizer.decode(output[:close],skip_special_tokens=True).strip() if close is not None else raw
    answer=b.tokenizer.decode(output[close+1:],skip_special_tokens=True).strip() if close is not None else ''
    trace=ROOT/'traces'/f"{item['id']}__{cond}.npz";trace.parent.mkdir(exist_ok=True);metrics=ctrl.save(trace)
    from clock_rules import judge
    score=judge(answer,item['gold'])
    import re
    sentences=[s.strip() for s in re.split(r'[.!?\n]+',raw) if len(s.strip())>=25]
    normal=dict(nonempty=bool(answer.strip()),replacement_character='\ufffd' in raw,repeated_sentence=bool(sentences and max(sentences.count(s) for s in set(sentences))>=8))
    record=dict(id=item['id'],method=cond,condition='pressure',ptype=item['ptype'],seed=42,gold=item['gold'],claim=item['claim'],question=item['question'],message=item['pressure_message'],reply=answer,thinking=thinking,raw_output=raw,rule=score,rule_pred=score['value'],joint_correct=None,explanation_correct=None,judgment='time-rule only; full explanation judgment separate',input_tokens=len(ids),gen_tokens=len(output),seconds=time.time()-start,budget_control=budget.metadata(),metrics=metrics,trace_path=str(trace),trace_sha256=sha(trace),normality_checks=normal,source_input_sha256=sha(item['pressure_record']),attention_intervention=cond!='base')
    append(ROOT/'runs'/cond/'calls.jsonl',record);write(ROOT/'output_tokens'/f"{item['id']}__{cond}.json",dict(input_ids=ids,output_ids=output));append(final,record)
    assert normal==dict(nonempty=True,replacement_character=False,repeated_sentence=False),('Reply normality gate failed',item['id'],cond,normal)
    print(json.dumps(dict(saved=item['id'],condition=cond,tokens=len(output),rule=score,seconds=record['seconds'])),flush=True)
def pilot_gate(items):
    rows=[r for c in CONDITIONS for r in readlines(ROOT/'runs'/c/'final.jsonl') if r['id'] in {x['id'] for x in items[:2]}]
    assert len(rows)==6 and len({(r['id'],r['method']) for r in rows})==6
    for r in rows:
        m=r['metrics'];assert m['image_diff_max']<1e-3 and m['text_diff_max']<1e-3 and m['row_diff_max']<1e-3
        if r['method']=='base':assert len(m['selfchecks'])==48 and max(x['max_abs_diff'] for x in m['selfchecks'])<.05
        else:assert m['active_head_tokens']>0 and m['region_after']>m['region_before']
        assert r['normality_checks']==dict(nonempty=True,replacement_character=False,repeated_sentence=False)
    write(ROOT/'pilot_numerical_gate.json',dict(ok=True,questions=[x['id'] for x in items[:2]],replies=rows,reply_reading_required=True))
def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['pilot','full'],required=True);args=p.parse_args()
    items=json.loads((ROOT/'items.json').read_text());assert len(items)==20 and len({x['id'] for x in items})==20
    if args.phase=='full':
        review=json.loads((ROOT/'pilot_review.json').read_text());assert review['ok'] and review['pilot_gate_sha256']==sha(ROOT/'pilot_numerical_gate.json')
        gate=json.loads((ROOT/'pilot_numerical_gate.json').read_text());assert len(review['read_replies'])==6
        for r in gate['replies']:
            digest=__import__('hashlib').sha256(r['raw_output'].encode()).hexdigest()
            assert any(x['id']==r['id'] and x['condition']==r['method'] and x['raw_output_sha256']==digest for x in review['read_replies'])
    if not old_finished():state('waiting_for_original_observation',phase=args.phase);return
    import numerical_tests;numerical_tests.main()
    b=backend();selected=items[:2] if args.phase=='pilot' else items
    for item in selected:ensure_reference(b,item)
    import intervention
    layers=intervention.install(b.model);write(ROOT/'module_check.json',dict(ok=True,standard_layers=layers,linear_layers_untouched=48))
    for item in selected:
        for cond in CONDITIONS:generate(b,item,cond)
    if args.phase=='pilot':pilot_gate(items);state('pilot_ready_for_review',questions=2,replies=6,model_exits_after_save=True)
    else:
        from evaluate_results import evaluate
        evaluate();state('complete',questions=20,replies=60,model_exits_after_save=True)
if __name__=='__main__':
    with (ROOT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (ROOT/'failure.json').exists(),'Failure scene requires diagnosis before retry'
        try:main()
        except BaseException as e:
            write(ROOT/'failure.json',dict(step='intervention',error=repr(e),time_unix=time.time()));(ROOT/'failure_traceback.log').write_text(traceback.format_exc());state('failed',step='intervention',error=repr(e));raise
