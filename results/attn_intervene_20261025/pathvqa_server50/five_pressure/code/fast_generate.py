"""Same generation settings; save image totals and stage maps in one pass."""
import json,time
from workflow_common import *

def generate_one(b,observer,item,condition,entry):
    import numpy as np,torch,transformers
    import generation_common as g
    c=config();rid=entry['record_id'];dest=ROOT/'records'/rid
    if (dest/'record.json').exists():return
    batch,actual=g.prepare(item,condition,b.processor,c);assert actual==entry
    batch={k:v.to(b.model.device) for k,v in batch.items()};batch['pixel_values']=batch['pixel_values'].to(b.model.dtype)
    gc=transformers.GenerationConfig.from_pretrained(c['model_dir'],local_files_only=True)
    gc.do_sample=True;gc.max_new_tokens=c['max_new_tokens'];gc.use_cache=True;gc.return_dict_in_generate=False
    for key in ['temperature','top_p','top_k','min_p','repetition_penalty']:setattr(gc,key,c['sampling'][key])
    eos=gc.eos_token_id if isinstance(gc.eos_token_id,list) else [gc.eos_token_id]
    budget=g.ThinkingBudget(entry['input_tokens'],b.close_id,eos[0],c['thinking_budget'],c['answer_budget'])
    if not (ROOT/'native_probe.json').exists():
        b.model.model.rope_deltas=None;observer.active=False
        with torch.inference_mode():
            plain=b.model(**batch,use_cache=True,logits_to_keep=1);baseline=plain.logits.detach().cpu();del plain
            b.model.model.rope_deltas=None;observer.reset(entry)
            observed=b.model(**batch,use_cache=True,logits_to_keep=1)
            diff=float((baseline-observed.logits.detach().cpu()).abs().max());observer.finish();del observed,baseline
        assert diff==0,'Observation changed native model output'
        write(ROOT/'native_probe.json',dict(passed=True,max_abs_diff=diff,record_id=rid))
    observer.reset(entry,lambda n:state('generating',record_id=rid,generated_predictions=n))
    class Stream(g.LiveOutput):
        def put(self,value):
            prompt=self.prompt
            super().put(value)
            if not prompt:
                tokens=value.reshape(-1).tolist();assert len(tokens)==1
                observer.accept(tokens[0],b.close_id,set(eos))
    processors=[budget]
    prior=ROOT/'live'/(rid+'.json')
    if prior.exists():
        saved=json.loads(prior.read_text());assert saved['record_id']==rid and saved['request_sha256']==sha(ROOT/'request.json')
        if saved['output_ids']:
            processors.append(g.PreservedPrefix(entry['input_tokens'],saved['output_ids']))
            append(ROOT/'prefix_reuse.jsonl',dict(record_id=rid,preserved_tokens=len(saved['output_ids']),source_sha256=sha(prior)))
    live=Stream(b.tokenizer,entry)
    b.model.model.rope_deltas=None;torch.manual_seed(42);torch.cuda.manual_seed_all(42);started=time.time()
    with torch.inference_mode():
        out=b.model.generate(**batch,generation_config=gc,streamer=live,logits_processor=transformers.LogitsProcessorList(processors))
    ids=out[0,entry['input_tokens']:].tolist();masses,checks=observer.finish()
    assert ids==live.ids and len(ids)==len(masses)==observer.accepted
    labels,boundary,close=g.phases(ids,b.tokenizer,set(eos));maps,counts=observer.spatial()
    assert counts.tolist()==[labels.count('thinking'),labels.count('answer')]
    assert labels.count('thinking')<=4096 and labels.count('answer')<=512
    thinking=b.tokenizer.decode(ids[:boundary],skip_special_tokens=True).strip() if boundary is not None else None
    answer=b.tokenizer.decode(ids[boundary+len(close):],skip_special_tokens=True).strip() if boundary is not None else None
    dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/'attention.npz',generation_image_mass=masses[...,0],
        stage_image_attention=maps,stage_token_counts=counts,stage_names=np.asarray(['think','answer']),
        phase_labels=np.asarray(labels),input_ids=np.asarray(entry['input_ids']),output_ids=np.asarray(ids),
        full_attention_layer_ids=np.asarray(c['layers']),image_indices=np.asarray(entry['regions']['image']),
        image_grid_thw=np.asarray(entry['image_grid_thw']))
    totals={}
    for s,name in enumerate(['think','answer']):
        subset=masses[np.asarray(labels)==('thinking' if s==0 else 'answer'),...,0]
        totals[name]=dict(tokens=int(counts[s]),mean_image_attention=float(subset.mean()) if len(subset) else None,
                          map_mean_image_attention=float(maps[s].sum(-1).mean()) if counts[s] else None)
        if counts[s]:assert abs(totals[name]['mean_image_attention']-totals[name]['map_mean_image_attention'])<1e-6
    row=dict(**entry,raw_output=b.tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False),
        public_reasoning=thinking,final_answer=answer,output_ids=ids,generated_tokens=len(ids),
        stopped_on_eos=ids[-1] in eos,budget_control=budget.metadata(),phase_lengths={x:labels.count(x) for x in set(labels)},
        attention_checks=checks,attention_sha256=sha(dest/'attention.npz'),image_attention_by_stage=totals,
        elapsed_seconds=time.time()-started,request_sha256=sha(ROOT/'config.json'),joint_correct=None,
        explanation_correct=None,judgment_status='same Luna pending; format does not determine correctness',
        spatial_recording='online during native generation; no second replay; no full attention matrices saved',time_unix=time.time())
    write(dest/'record.json',row)
    append(ROOT/'calls.jsonl',dict(record_id=rid,tokens=len(ids),seconds=row['elapsed_seconds'],source_sha256=sha(dest/'record.json')))
    print(json.dumps(dict(saved=rid,tokens=len(ids),seconds=row['elapsed_seconds'],phase_counts=counts.tolist())),flush=True)
    del batch,out,masses;observer.rows=[];torch.cuda.empty_cache()
