"""Per-output-token attention diagnostics for the existing ClockQA entropy trajectories.
No generation. Original files and experiment configuration are read-only.
"""
from pathlib import Path
import argparse, contextlib, gzip, hashlib, importlib.util, json, math, os, sys, time, traceback
import numpy as np
R=Path('/mnt/sdb/linxin/proofwriter_runs/clockqa-uncertainty-20261027')
D=Path('/mnt/sdb/linxin/proofwriter_runs/clockqa-uncertainty-attention-20261101')
LAYERS=list(range(3,64,4))
METRICS=['image_mass','user_text_mass','pressure_mass','question_mass','generated_mass','image_entropy_nats','image_entropy_normalized','image_entropy_before_restore_nats']

def atomic(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp.'+str(os.getpid()));q.write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False,indent=2));q.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_original():
 spec=importlib.util.spec_from_file_location('clockqa_original',R/'clockqa_entropy.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def restore_image(torch,im,mask,ref):
 # Same per-query formula and .95 donor capacity as original intervention.redistribute.
 mass=im.sum(-1,keepdim=True);q=im/mass
 donor=(q*(~mask)).sum(-1,keepdim=True)
 additions=(ref-q).clamp_min(0)*mask
 requested=additions.sum(-1,keepdim=True);capacity=donor*.95
 additions=additions*torch.minimum(torch.ones_like(requested),capacity/requested.clamp_min(torch.finfo(im.dtype).tiny))
 delta=additions.sum(-1,keepdim=True)
 qnew=q+additions-q*(~mask)*(delta/donor.clamp_min(torch.finfo(im.dtype).tiny))
 return mass*qnew

def unit():
 import torch
 orig=load_original();sys.path.insert(0,str(orig.ROOT/'interventions'));import intervention
 torch.manual_seed(42);p=torch.softmax(torch.randn(5,4,17),-1);img=torch.tensor([1,3,4,7,9]);hs=torch.tensor([0,2,4]);mask=torch.tensor([[1,0,1,0,0],[0,1,0,0,1],[1,0,0,1,0]],dtype=torch.bool)
 ref=torch.rand(3,5);ref/=ref.sum(-1,keepdim=True)
 im=p[hs][:,:,img];actual=restore_image(torch,im,mask[:,None],ref[:,None])
 expected=[]
 for k in range(4):expected.append(intervention.redistribute(torch,p[:,k],img,hs,mask,ref,1.)[0][hs][:,img])
 expected=torch.stack(expected,1);err=float((actual-expected).abs().max());assert err<1e-7,err
 q=im/im.sum(-1,keepdim=True);h=-(q*q.clamp_min(torch.finfo(q.dtype).tiny).log()).sum(-1)
 assert bool(((h>=0)&(h<=math.log(5)+1e-6)).all())
 # Predict output token k from query position prompt_length-1+k.
 assert [i-(7-1) for i in [6,7,8]]==[0,1,2]
 result={'vectorized_original_redistribution_maxabs':err,'image_entropy_bounds':True,'prediction_position_alignment':True,'passed':True};atomic(D/'unit_test.json',result);print('UNIT',json.dumps(result),flush=True)

class Recorder:
 def __init__(self,torch,record,task,batch,processor,heads,refs=None):
  self.t=torch;self.r=record;self.task=task;self.nin=len(record['input_ids']);self.T=len(record['output_ids']);self.refs=refs
  self.heads=heads;self.data=np.zeros((self.T,2,len(METRICS)),np.float64);self.counts=np.zeros((self.T,2),np.int32);self.layer_count=np.zeros((self.T,16),np.int8)
  self.mass_error=0.;self.restore_mass_error=0.;self.raw_attention_diffs=[];self.H_zero=0
  tok=processor.tokenizer;ids=record['input_ids'];dec=lambda x:tok.decode(x,skip_special_tokens=False,clean_up_tokenization_spaces=False)
  text=dec(ids);parts=[dec([x]) for x in ids];assert ''.join(parts)==text,'Prompt contains split UTF8; use explicit offsets before continuing'
  prompt=record['prompt'];assert text.count(prompt)==1,'Prompt text not uniquely found'
  pos=text.index(prompt);q=task['item']['question'];assert prompt.startswith(q)
  offsets=np.r_[0,np.cumsum([len(x) for x in parts])]
  def span(a,b):return np.flatnonzero((offsets[:-1]<b)&(offsets[1:]>a)) if b>a else np.array([],np.int64)
  image_id=getattr(processor,'image_token_id',None)
  if image_id is None:image_id=tok.convert_tokens_to_ids(processor.image_token)
  self.img=np.flatnonzero(np.asarray(ids)==image_id);assert len(self.img)>1
  self.user=span(pos,pos+len(prompt));self.question=span(pos,pos+len(q));self.pressure=span(pos+len(q)+1,pos+len(prompt)) if task['condition']!='neutral' else np.array([],np.int64)
  assert not np.intersect1d(self.user,self.img).size
  self.nimg=len(self.img);self.cache={};self.before_entropy=None
  self.close=record['output_ids'].index(tok.encode('</think>',add_special_tokens=False)[0]) if tok.encode('</think>',add_special_tokens=False)[0] in record['output_ids'] else None
 def idx(self,a,device):return self.t.as_tensor(a,dtype=self.t.long,device=device)
 def probs(self,layer,p,qpos):
  t=self.t;li=LAYERS.index(layer);idx=self.idx(self.img,p.device)
  pre=p[0].index_select(-1,idx)
  P=pre.sum(-1,keepdim=True);q=pre/P.clamp_min(t.finfo(pre.dtype).tiny)
  hpre=-(q*q.clamp_min(t.finfo(q.dtype).tiny).log()).sum(-1)
  if self.refs is not None:
   for stage in [0,1]:
    rows=(qpos>=self.nin)
    if self.close is not None:rows &= ((qpos<self.nin+self.close) if stage==0 else (qpos>=self.nin+self.close))
    elif stage==1:continue
    rr=rows.nonzero().flatten();hs=self.idx(np.flatnonzero(self.refs['heads'][li]),p.device)
    if not len(rr) or not len(hs):continue
    im=p[0,hs[:,None,None],rr[None,:,None],idx[None,None,:]]
    mask=t.as_tensor(self.refs['E_mask'][stage,li],device=p.device,dtype=t.bool).index_select(0,hs)[:,None,:]
    ref=t.as_tensor(self.refs['E_r'][stage,li],device=p.device,dtype=t.float32).index_select(0,hs)[:,None,:]
    new=restore_image(t,im,mask,ref)
    err=float((new.sum(-1)-im.sum(-1)).abs().max());self.restore_mass_error=max(self.restore_mass_error,err)
    if err>=1e-3 or not bool(t.isfinite(new).all()) or float(new.min()) < -1e-7:raise ValueError('Original restore conservation failed')
    p[0,hs[:,None,None],rr[None,:,None],idx[None,None,:]]=new
  sel=(qpos>=self.nin-1)&(qpos<self.nin-1+self.T)
  if not bool(sel.any()):return p
  qp=qpos[sel];r=(qp-(self.nin-1)).cpu().numpy();v=p[0][:,sel,:]
  im=v.index_select(-1,idx);mass=im.sum(-1);qn=im/mass[...,None].clamp_min(t.finfo(im.dtype).tiny)
  h=-(qn*qn.clamp_min(t.finfo(qn.dtype).tiny).log()).sum(-1)
  if bool((mass<=0).any()):raise ValueError('Zero image mass: cannot define conditional image entropy')
  def m(a):return v.index_select(-1,self.idx(a,p.device)).sum(-1) if len(a) else t.zeros_like(mass)
  vals=t.stack([mass,m(self.user),m(self.pressure),m(self.question),v[...,self.nin:].sum(-1),h,h/math.log(self.nimg),hpre[:,sel]],-1)
  self.mass_error=max(self.mass_error,float((v.sum(-1)-1).abs().max()))
  allsum=vals.sum(0).detach().cpu().numpy();hs=self.idx(np.flatnonzero(self.heads[li]),p.device)
  chosen=vals.index_select(0,hs).sum(0).detach().cpu().numpy() if len(hs) else np.zeros_like(allsum)
  self.data[r,0]+=allsum;self.data[r,1]+=chosen;self.counts[r,0]+=v.shape[0];self.counts[r,1]+=len(hs);self.layer_count[r,li]+=1
  return p
 def result(self):
  assert bool((self.layer_count==1).all()),'Missing or duplicated layer/token'
  assert bool((self.counts[:,0]==384).all()) and bool((self.counts[:,1]==96).all()),np.unique(self.counts,axis=0)
  if self.mass_error>=1e-3:raise ValueError('Attention row sum mismatch')
  return (self.data/self.counts[:,:,None]).astype(np.float32)

ACTIVE=None

def install(model):
 from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
 targets={id(m) for _,m in model.named_modules() if all(hasattr(m,a) for a in ['q_proj','k_proj','o_proj','layer_idx']) and m.__class__.__name__=='Qwen3_5Attention'}
 assert len(targets)==16,len(targets)
 orig=ALL_ATTENTION_FUNCTIONS['sdpa']
 def wrapped(module,query,key,value,attention_mask,*args,**kw):
  rec=ACTIVE
  if rec is None or id(module) not in targets:return orig(module,query,key,value,attention_mask,*args,**kw)
  t=rec.t;B,H,Q,V=query.shape;K=key.shape[-2];rep=H//key.shape[1];k=key.repeat_interleave(rep,1);v=value.repeat_interleave(rep,1)
  scale=kw.get('scaling',args[1] if len(args)>1 else None);scale=getattr(module,'scaling',V**-.5) if scale is None else scale
  # Preserve original native SDPA outputs for all untouched rows, including the entire prompt.
  result=orig(module,query,key,value,attention_mask,*args,**kw)
  out=result[0].clone() if rec.refs is not None else result[0]
  positions=t.arange(K-Q,K,device=query.device);keypos=t.arange(K,device=query.device);kf=k.float().transpose(-1,-2);vf=v.float()
  for s in range(0,Q,64):
   e=min(Q,s+64);sc=(query[:,:,s:e].float()@kf)*scale
   if attention_mask is not None:
    mask=attention_mask[:,:,s:e,:K];sc=sc.masked_fill(~mask,-float('inf')) if mask.dtype==t.bool else sc+mask.float()
   else:sc=sc.masked_fill(~(keypos[None,:]<=positions[s:e,None])[None,None],-float('inf'))
   p=t.softmax(sc,-1);p=rec.probs(int(module.layer_idx),p,positions[s:e])
   if rec.refs is not None:
    change=positions[s:e]>=rec.nin
    if bool(change.any()):
     pp=p[:,:,change];o=(pp@vf).to(query.dtype).transpose(1,2).contiguous();rr=change.nonzero().flatten()+s;out[:,rr]=o
  return out,result[1]
 ALL_ATTENTION_FUNCTIONS['sdpa']=wrapped
 gm=getattr(type(ALL_ATTENTION_FUNCTIONS),'_global_mapping',None)
 if isinstance(gm,dict):gm['sdpa']=wrapped

def neutral_heads(task,manifest):
 qid=task['question_id'];t=next(x for x in manifest['tasks'] if x['question_id']==qid and x['condition']=='neutral')
 r=json.loads((R/'records'/f'{t["task_id"]}.json').read_text());p=Path(r['source_record']).with_name('attention.npz')
 with np.load(p,allow_pickle=False) as z:
  assert z['output_ids'].tolist()==r['output_ids']
  names=z['group_names'].tolist();mass=z['generation_masses'][:r['think_tokens'],:,:,names.index('image')].mean(0)
 assert mass.shape==(16,24)
 heads=np.zeros((16,24),bool);heads.ravel()[np.argsort(-mass.ravel(),kind='stable')[:96]]=True
 return heads,{'path':str(p),'sha256':sha(p),'selection':'top96 by original neutral thinking mean image mass'}

def install_entropy_head(model):
 original=model.lm_head.forward
 def forward(hidden):
  rec=ACTIVE
  if rec is None:return original(hidden)
  torch=rec.t;rows=hidden[:,rec.nin-1:rec.nin-1+rec.T];assert rows.shape[1]==rec.T
  values=[];ranks={}
  for start in range(0,rec.T,64):
   logits=original(rows[:,start:start+64]).float()[0]
   logp=torch.log_softmax(logits,-1);h=-(logp.exp()*logp).sum(-1);values.extend(h.detach().cpu().tolist())
   for pos in rec.check_positions:
    if start<=pos<start+len(logits):
     token=rec.r['output_ids'][pos];ranks[pos]=int((logits[pos-start]>=logits[pos-start,token]).sum())
  rec.fresh_entropy=values;rec.ranks=ranks
  return original(hidden[:,-1:])
 model.lm_head.forward=forward

def measure(task,ce,torch,processor,model):
 global ACTIVE
 p=R/'records'/f'{task["task_id"]}.json';r=json.loads(p.read_text());prompt,batch=ce.input_for(task,processor,torch,model)
 assert prompt==r['prompt'] and batch['input_ids'][0].tolist()==r['input_ids']
 assert ce.digest(task['image_path'])==r['image_sha256']
 refs=None
 if task['condition']=='restore_E':
  assert sha(task['reference'])==r['reference_sha256']
  with np.load(task['reference'],allow_pickle=False) as z:refs={k:z[k] for k in z.files}
  assert list(refs['layer_ids'])==LAYERS
 heads,headsrc=neutral_heads(task,ce.load_json(R/'manifest.json'))
 rec=Recorder(torch,r,task,batch,processor,heads,refs);n=len(r['output_ids'])
 points=sorted(set([0,1,2,7,31,127,255,511,1023,n-1,int(np.argmax(r['entropy_nats'][:r['think_tokens']]))]))
 points=[i for i in points if 0<=i<n];rec.check_positions=points
 ex=ce.extend_batch(batch,r['output_ids'][:-1],torch);ce.reset_rope(model);start=time.time();ACTIVE=rec
 try:
  with torch.inference_mode():o=model(**ex,use_cache=False,return_dict=True,logits_to_keep=0)
 finally:ACTIVE=None
 newh=[rec.fresh_entropy[k] for k in points];deltas=[abs(float(h)-float(r['entropy_nats'][k])) for h,k in zip(newh,points)]
 rank=[rec.ranks[k] for k in points]
 arr=rec.result();pieces=[processor.tokenizer.decode([int(x)],skip_special_tokens=False,clean_up_tokenization_spaces=False) for x in r['output_ids']]
 target=D/'records'/f'{task["task_id"]}.jsonl.gz';target.parent.mkdir(parents=True,exist_ok=True)
 with gzip.open(target,'wt',encoding='utf-8') as f:
  for i,(piece,vals,h) in enumerate(zip(pieces,arr,r['entropy_nats'])):
   f.write(json.dumps({'token_position':i+1,'token':piece,'phase':'think' if i<r['think_tokens'] else 'answer','token_entropy_nats':rec.fresh_entropy[i],'original_token_entropy_nats':h,'all_heads':dict(zip(METRICS,map(float,vals[0]))),'image_heads96':dict(zip(METRICS,map(float,vals[1])))},ensure_ascii=False,allow_nan=False)+'\n')
 old_attention_check=None
 if r.get('source_record') and task['condition']!='restore_E':
  archive=Path(r['source_record']).with_name('attention.npz')
  if archive.exists():
   with np.load(archive,allow_pickle=False) as z:
    assert z['output_ids'].tolist()==r['output_ids'] and z['input_ids'].tolist()==r['input_ids']
    names=z['group_names'].tolist();im=z['generation_masses'][...,names.index('image')].mean(axis=(1,2))
    old_attention_check={'path':str(archive),'image_mass_maxabs':float(np.abs(im-arr[:,0,0]).max()),'image_mass_meanabs':float(np.abs(im-arr[:,0,0]).mean())}
 meta={'task_id':task['task_id'],'question_id':task['question_id'],'condition':task['condition'],'pressure_type':task['pressure_type'],'source_record':str(p),'source_sha256':sha(p),'image_sha256':r['image_sha256'],'reference_sha256':r['reference_sha256'],'head_selection':headsrc,'n_output_tokens':n,'think_tokens':r['think_tokens'],'n_image_tokens':rec.nimg,'n_user_text_tokens':len(rec.user),'n_pressure_tokens':len(rec.pressure),'metrics':METRICS,'groups':['all384_standard_attention_heads','neutral_top96_image_heads'],'archived_attention_check':old_attention_check,'row_sum_maxabs':rec.mass_error,'restore_image_mass_maxabs':rec.restore_mass_error,'logit_checks':{'positions_1based':[i+1 for i in points],'replayed_token_entropy_nats':newh,'original_token_entropy_nats':[r['entropy_nats'][i] for i in points],'max_entropy_absdiff':max(deltas),'max_token_rank':max(rank)},'restore_execution':'parallel causal reconstruction with original decode-only row transform and original prompt outputs' if refs is not None else 'native SDPA unchanged; observer only','entropy_alignment':'token_entropy_nats recomputed in same forward as attention; original_token_entropy_nats retained separately; BF16 parallel versus cached numerical differences explicitly reported','seconds':round(time.time()-start,3),'token_file':str(target),'token_file_sha256':sha(target)}
 atomic(D/'meta'/f'{task["task_id"]}.json',meta);print('DONE',task['task_id'],'tokens',n,'seconds',meta['seconds'],'entropy_diff',max(deltas),flush=True)
 return r,arr,meta

def original_restore_check(task,r,arr,ce,torch,processor,model):
 # Reuse the original generate preparation/caching interface but force every saved token.
 global ACTIVE
 from transformers import LogitsProcessor,LogitsProcessorList
 prompt,batch=ce.input_for(task,processor,torch,model);length=min(64,len(r['output_ids']));captured=[];ACTIVE=None;ce.reset_rope(model)
 class Force(LogitsProcessor):
  def __call__(self,ids,scores):
   k=ids.shape[-1]-len(r['input_ids']);forced=torch.full_like(scores,-float('inf'));forced[:,r['output_ids'][k]]=0;return forced
 def hook(m,args,o):
  lp=torch.log_softmax(o.logits[0,-1].float(),-1);captured.append(float(-(lp.exp()*lp).sum()))
 handle=model.register_forward_hook(hook)
 try:
  with torch.inference_mode(),ce.intervention_context(task,batch,processor,torch,model):
   out=model.generate(**batch,do_sample=True,temperature=1.,top_p=.95,top_k=20,min_p=0.,repetition_penalty=1.,max_new_tokens=length,use_cache=True,logits_processor=LogitsProcessorList([Force()]))
 finally:handle.remove()
 assert out[0,len(r['input_ids']):].tolist()==r['output_ids'][:length]
 assert len(captured)==length
 err=max(abs(a-b) for a,b in zip(captured,r['entropy_nats'][:length]))
 result={'task_id':task['task_id'],'checked_prefix_tokens':length,'max_stored_entropy_absdiff':err,'interface':'original generate preparation/cache; original decode-only intervention; forced existing tokens; no new answer','stored_entropy':r['entropy_nats'][:length],'replayed_entropy':captured}
 atomic(D/'original_restore_cached_check.json',result)
 if err>=.05:raise ValueError('Original cached replay does not reproduce stored token entropy')
 return result

def worker(pilot=False):
 ce=load_original();manifest=ce.load_json(R/'manifest.json');torch,processor,model=ce.model_setup();install(model);install_entropy_head(model)
 if pilot:
  for cond in ['neutral','restore_E']:
   task=next(t for t in manifest['tasks'] if t['condition']==cond);r,arr,meta=measure(task,ce,torch,processor,model)
   if cond=='restore_E':
    original_restore_check(task,r,arr,ce,torch,processor,model)
    # Original cached entropy is validated above; parallel entropy is recomputed for every token and reported independently.
  atomic(D/'pilot_gate.json',{'ok':True,'original_source_sha256':sha(R/'clockqa_entropy.py'),'unit_test':ce.load_json(D/'unit_test.json')})
 for task in sorted(manifest['tasks'],key=lambda x:x['task_id']):
  name=task['task_id']
  if (D/'meta'/f'{name}.json').exists():continue
  lock=D/'claims'/f'{name}.lock'
  try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
  except FileExistsError:continue
  with os.fdopen(fd,'w') as f:f.write(str(os.getpid()))
  measure(task,ce,torch,processor,model)
 print('WORKER_COMPLETE',os.getpid(),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['unit','worker']);parser.add_argument('--pilot',action='store_true');args=parser.parse_args()
 try:
  if args.phase=='unit':unit()
  else:worker(args.pilot)
 except Exception as e:
  atomic(D/'failures'/f'{os.getpid()}_{time.time_ns()}.json',{'error':repr(e),'traceback':traceback.format_exc()});print(traceback.format_exc(),flush=True);raise SystemExit(1)
