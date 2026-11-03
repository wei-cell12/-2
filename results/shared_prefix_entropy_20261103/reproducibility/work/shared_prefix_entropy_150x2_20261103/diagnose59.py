import json,os,sys,time,subprocess
from pathlib import Path
W=Path('/mnt/sdb/linxin/shared-prefix-entropy-150x2-20261103');sys.path.insert(0,str(W));sys.argv=['diagnostic','--rank','1']
import worker as w
from symexp.config import Config
from symexp.fp.generate import build_inputs
import torch
from entropy_core import signals_torch
m=json.loads((W/'manifest.json').read_text());j=next(j for j in m['jobs'] if j['key']=='CQ-008__belief_right');r=json.loads(Path(j['source']).read_text());tokens=r['output_ids'];t0=time.time()
print('DIAGNOSTIC START',os.getpid(),flush=True);b=w.load_backend(Config.load(m['config']));_,p=build_inputs(b,j['item']['image'],j['item']['messages'][j['condition']]);_,n=build_inputs(b,j['item']['image'],j['item']['messages']['neutral'])
def cached(inp,toks,chunk):
 model=b.model;w.reset(model);dev=model.device;batch={k:v.to(dev) for k,v in inp.items() if hasattr(v,'to')}
 if 'pixel_values' in batch:batch['pixel_values']=batch['pixel_values'].to(model.dtype)
 with torch.inference_mode():o=model(**batch,use_cache=True,return_dict=True,logits_to_keep=1)
 past=o.past_key_values;parts=[o.logits[0,-1:].detach().cpu()];del o,batch
 with torch.inference_mode():
  for a in range(0,len(toks)-1,chunk):
   ids=torch.tensor([toks[a:min(a+chunk,len(toks)-1)]],device=dev)
   o=model(input_ids=ids,past_key_values=past,use_cache=True,return_dict=True,logits_to_keep=0)
   past=o.past_key_values;parts.append(o.logits[0].detach().cpu());del o
 del past
 return torch.cat(parts,0)
result=dict(id=j['key'],checks=[],cached_checks=[])
for branch,inp in [('pressure',p),('neutral',n)]:
 full=w.predict(b,inp,tokens).cpu();ref1=cached(inp,tokens,1)
 for k in [0,1,15]:
  short=w.predict(b,inp,tokens[:k+1])[-1:].cpu();s=signals_torch(torch,full[k:k+1],short,[tokens[k]])
  x=dict(branch=branch,step=k+1,entropy_absdiff=abs(s['delta_H'][0]),JSD=s['JSD'][0]);result['checks'].append(x);print('FULL_SHORT',json.dumps(x),flush=True)
 for chunk in [16,32,64]:
  q=cached(inp,tokens,chunk);sg=signals_torch(torch,q,ref1,tokens)
  x=dict(branch=branch,chunk=chunk,max_abs_delta_H=max(abs(v) for v in sg['delta_H']),max_JSD=max(sg['JSD']),mean_JSD=sum(sg['JSD'])/len(tokens),top1_agree=sum(a==z for a,z in zip(sg['top1_pressure'],sg['top1_neutral']))/len(tokens))
  result['cached_checks'].append(x);print('CACHED_COMPARE',json.dumps(x),flush=True);del q
 del full,ref1;torch.cuda.empty_cache()
result['seconds']=time.time()-t0;(W/'diagnostic_alignment.json').write_text(json.dumps(result,indent=2));print('DONE',result['seconds'],flush=True)
