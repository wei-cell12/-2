"""20 ClockQA image-conditioned generations, native SDPA unchanged; then region replay."""
import fcntl,hashlib,json,os,re,shutil,time
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic(path,value):
    temp=path.with_name(path.name+'.tmp.'+str(os.getpid()));temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');os.replace(temp,path)
def state(status,**kw):atomic(ROOT/'job_state.json',dict(status=status,time_unix=time.time(),planned=20,**kw))
def rows():return [json.loads(x) for x in (ROOT/'items.jsonl').read_text().splitlines() if x.strip()]
def cfg():return json.loads((ROOT/'request.json').read_text())
def processor_for(c):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(c['model_dir'],local_files_only=True,
        min_pixels=c['image_min_pixels'],max_pixels=c['image_max_pixels'])
def prepare(item,condition,processor,config):
    import torch
    from PIL import Image
    imagefile=ROOT/item['image_path'];assert sha(imagefile)==item['image_sha256']
    question=re.sub(r'<image\d+>','',item['question']).strip()
    task=question+('\n'+item['options'].strip() if item['options'].strip() else '')
    pressure=item['pressure_text'] if condition=='pressure' else ''
    text=task+('\n'+pressure if pressure else '')
    image=Image.open(imagefile).convert('RGB')
    messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':text}]}]
    rendered=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=True,
                                            reasoning_effort=config['reasoning_effort'])
    assert config['reasoning_effort']=='medium' and 'Reasoning effort is set to xhigh.' not in rendered
    batch=processor(text=[rendered],images=[image],return_tensors='pt')
    ids=batch['input_ids'][0].tolist();tok=processor.tokenizer
    image_token_id=processor.image_token_id if hasattr(processor,'image_token_id') else tok.convert_tokens_to_ids(processor.image_token)
    vi=[i for i,v in enumerate(ids) if v==image_token_id];assert vi
    grid=batch['image_grid_thw'][0].tolist();assert len(vi)==grid[0]*grid[1]*grid[2]//processor.image_processor.merge_size**2
    expanded=tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    encoded=tok(expanded,add_special_tokens=False,return_offsets_mapping=True)
    assert encoded['input_ids']==ids,'Expanded multimodal token stream must round-trip'
    offsets=encoded['offset_mapping'];task_start=expanded.index(task);task_end=task_start+len(task)
    assert expanded.count(task)==1
    pstart=expanded.index(pressure,task_end) if pressure else 0;pend=pstart+len(pressure)
    taskidx=[i for i,(a,b) in enumerate(offsets) if b>task_start and a<task_end]
    pidx=[i for i,(a,b) in enumerate(offsets) if pressure and b>pstart and a<pend]
    assert not set(vi)&set(taskidx) and not set(pidx)&set(taskidx)
    occupied=set(vi+taskidx+pidx);other=[i for i in range(len(ids)) if i not in occupied]
    qidx=list(range(max(vi)+1,len(ids)))
    entry={**item,'record_id':item['item_id']+'__'+condition,'condition':condition,
       'input_ids':ids,'input_tokens':len(ids),'image_tokens':len(vi),'non_image_tokens':len(ids)-len(vi),
       'image_grid_thw':grid,'processed_height':grid[1]*processor.image_processor.patch_size,
       'processed_width':grid[2]*processor.image_processor.patch_size,
       'text':text,'rendered_chat_template':rendered,'expanded_input_sha256':hashlib.sha256(expanded.encode()).hexdigest(),
       'regions':dict(image=vi,task_text=taskidx,user_pressure=pidx,input_other=other),
       'prefill_query_indices':qidx,'prefill_visible_body_indices':sorted(taskidx+pidx),
       'pressure_insert':pressure,'attention_intervention':False}
    assert len(ids)+config['max_new_tokens']<=processor.tokenizer.model_max_length
    return batch,entry
def validate():
    c=cfg();assert sha(ROOT/'items.jsonl')==c['items_sha256']
    identfile=Path(c['model_identity_root'])/'model_identity.json';assert sha(identfile)==c['model_identity_sha256']
    ident=json.loads(identfile.read_text());assert Path(ident['model_dir']).resolve()==Path(c['model_dir']).resolve()
    for name,digest in ident['metadata_sha256'].items():assert sha(Path(c['model_dir'])/name)==digest
    items=rows();assert len(items)==10 and Counter(x['pressure_type'] for x in items)==c['pressure_counts']
    p=processor_for(c);prepared=[];lengths=[]
    for item in items:
        pair=[]
        for cond in ['neutral','pressure']:
            batch,e=prepare(item,cond,p,c);pair.append(e);prepared.append(e);del batch
        n,s=pair
        assert n['image_tokens']==s['image_tokens'] and n['image_grid_thw']==s['image_grid_thw']
        assert n['regions']['image']==s['regions']['image'] and s['input_tokens']>n['input_tokens']
        lengths.append(dict(item_id=item['item_id'],pressure_type=item['pressure_type'],neutral_tokens=n['input_tokens'],
             pressure_tokens=s['input_tokens'],added_tokens=s['input_tokens']-n['input_tokens'],image_tokens=n['image_tokens']))
    body=''.join(json.dumps(e,ensure_ascii=False)+'\n' for e in prepared)
    dest=ROOT/'tokenized_inputs.jsonl'
    if dest.exists():assert dest.read_text()==body
    else:dest.write_text(body)
    atomic(ROOT/'length_audit.json',dict(pairs=lengths,min_added_tokens=min(x['added_tokens'] for x in lengths),
       max_added_tokens=max(x['added_tokens'] for x in lengths),mean_added_tokens=sum(x['added_tokens'] for x in lengths)/len(lengths),
       all_pair_image_positions_and_counts_equal=True,text_length_matched=False))
    value=dict(passed=True,planned=20,questions=10,pressure_counts=c['pressure_counts'],request_sha256=sha(ROOT/'request.json'),
      items_sha256=sha(ROOT/'items.jsonl'),tokenized_sha256=sha(dest),model_identity_sha256=sha(identfile),
      max_input_tokens=max(e['input_tokens'] for e in prepared),max_visual_tokens=max(e['image_tokens'] for e in prepared),
      image_shape_and_masks_verified=True,all_images_processed=True,time_unix=time.time())
    test=budget_selftest()
    close=p.tokenizer.encode('</think>',add_special_tokens=False);assert len(close)==1
    assert c['max_new_tokens']==c['thinking_budget']+c['answer_budget']+2
    value.update(reasoning_effort=c['reasoning_effort'],thinking_budget=c['thinking_budget'],answer_budget=c['answer_budget'],
                 close_think_token_id=close[0],budget_selftest_passed=test['passed'])
    atomic(ROOT/'cpu_validation.json',value);print(json.dumps(value),flush=True)
    return p,prepared
def phases(ids,tok,eos):
    close=tok.encode('</think>',add_special_tokens=False)
    boundary=next((i for i in range(len(ids)-len(close)+1) if ids[i:i+len(close)]==close),None)
    labels=['thinking']*len(ids)
    if boundary is not None:
        labels[boundary:boundary+len(close)]=['boundary']*len(close)
        labels[boundary+len(close):]=['answer']*(len(ids)-boundary-len(close))
    for i,t in enumerate(ids):
        if t in eos:labels[i]='special'
    return labels,boundary,close
def counts():
    rec=[json.loads(f.read_text()) for f in (ROOT/'results').glob('*/record.json')]
    return dict(saved=len(rec),neutral=sum(r['condition']=='neutral' for r in rec),pressure=sum(r['condition']=='pressure' for r in rec),
                paired_questions=sum(len(v)==2 for v in ({i:[r for r in rec if r['item_id']==i] for i in {r['item_id'] for r in rec}}).values()),
                ended_on_eos=sum(r['stopped_on_eos'] for r in rec),
                thinking_budget_hit=sum(r.get('budget_control',{}).get('thinking_budget_hit',False) for r in rec),
                answer_budget_hit=sum(r.get('budget_control',{}).get('answer_budget_hit',False) for r in rec),
                natural_eos=sum(r.get('natural_eos',False) for r in rec),unreviewed=len(rec))
class LiveOutput:
    """Preserve sampled tokens periodically without changing generation."""
    def __init__(self,tokenizer,entry):
        self.tokenizer=tokenizer;self.entry=entry;self.ids=[];self.prompt=True
        self.root=ROOT/'live';self.root.mkdir(exist_ok=True)
    def put(self,value):
        ids=value.reshape(-1).tolist()
        if self.prompt:
            assert ids==self.entry['input_ids'];self.prompt=False;return
        self.ids.extend(ids)
        if len(self.ids)%64==0:self.save(False)
    def save(self,ended):
        atomic(self.root/(self.entry['record_id']+'.json'),dict(record_id=self.entry['record_id'],
          output_ids=self.ids,generated_tokens=len(self.ids),generation_ended=ended,
          raw_output=self.tokenizer.decode(self.ids,skip_special_tokens=False,clean_up_tokenization_spaces=False),
          partial_output_only=not ended,request_sha256=sha(ROOT/'request.json'),time_unix=time.time()))
    def end(self):self.save(True)
class ThinkingBudget:
    """Bound thinking separately; reserve answer space and label forced boundaries."""
    def __init__(self,prompt_length,close_token,eos_token,thinking_limit,answer_limit):
        self.prompt_length=prompt_length;self.close_token=close_token;self.eos_token=eos_token
        self.thinking_limit=thinking_limit;self.answer_limit=answer_limit;self.close_position=None
        self.forced_close=False;self.forced_answer_end=False
    def __call__(self,input_ids,scores):
        n=input_ids.shape[-1]-self.prompt_length
        if n and self.close_position is None and int(input_ids[0,-1])==self.close_token:self.close_position=n-1
        forced=None
        if self.close_position is None and n>=self.thinking_limit:
            forced=self.close_token;self.forced_close=True
        elif self.close_position is not None and n-self.close_position-1>=self.answer_limit:
            forced=self.eos_token;self.forced_answer_end=True
        if forced is None:return scores
        result=scores.new_full(scores.shape,float('-inf'));result[:,forced]=0.;return result
    def metadata(self):
        return dict(thinking_limit=self.thinking_limit,answer_limit=self.answer_limit,
          thinking_budget_hit=self.forced_close,answer_budget_hit=self.forced_answer_end,
          forced_think_closure=self.forced_close,forced_answer_eos=self.forced_answer_end,
          closing_token_generated_position=self.close_position)
class PreservedPrefix:
    """Replay an archived sampled prefix; sample only after its final token."""
    def __init__(self,prompt_length,ids):self.prompt_length=prompt_length;self.ids=ids
    def __call__(self,input_ids,scores):
        n=input_ids.shape[-1]-self.prompt_length
        if n<=len(self.ids):assert input_ids[0,self.prompt_length:].tolist()==self.ids[:n]
        if n>=len(self.ids):return scores
        result=scores.new_full(scores.shape,float('-inf'));result[:,self.ids[n]]=0.;return result
def budget_selftest():
    import torch
    prefix=[1,7,1];out=[];budget=ThinkingBudget(3,7,8,2,3)
    for _ in range(10):
        ids=torch.tensor([prefix+out]);original=torch.zeros((1,10));scores=budget(ids,original)
        if scores is original:token=5 if budget.close_position is None else 9
        else:
            assert int(torch.isfinite(scores).sum())==1;token=int(scores.argmax(-1))
        out.append(token)
        if token==8:break
    assert out==[5,5,7,9,9,9,8],out
    natural=ThinkingBudget(3,7,8,2,3);original=torch.zeros((1,10))
    assert natural(torch.tensor([prefix+[5,7]]),original) is original
    assert natural.close_position==1 and not natural.forced_close
    assert int(natural(torch.tensor([prefix+[5,7,9,9,9]]),original).argmax(-1))==8
    value=dict(passed=True,overlong_thinking_closed_at_exact_budget=True,separate_answer_budget_reserved=True,
      natural_think_closure_preserved=True,prompt_close_tokens_ignored=True,under_budget_logits_unchanged=True)
    atomic(ROOT/'budget_selftest.json',value);return value
# Only shared input/stream/budget helpers are imported; no standalone runner.
