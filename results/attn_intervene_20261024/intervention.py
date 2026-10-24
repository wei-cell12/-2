"""Decode-only within-image restoration; all three arms share the explicit attention path.

No text-key renormalization, no prefill intervention, no linear-attention changes.
Reference r and selected heads are static neutral stage averages. RNG for fake
regions is independent of the model's sampling RNG (see references.py).
"""
import sys
import numpy as np

ACTIVE=None
TARGETS=set()

def redistribute(torch, p, image_idx, selected_heads, mask, reference, strength):
    """p: [H,K]; return copy plus per-selected-head diagnostics, FP32 throughout."""
    hs=selected_heads; im=p.index_select(0,hs).index_select(1,image_idx)
    P=im.sum(-1,keepdim=True)
    if not bool((P>0).all()): raise ValueError('Zero image mass in a selected head')
    q=im/P
    region_before=(q*mask).sum(-1)
    donor=(q*(~mask)).sum(-1,keepdim=True)
    additions=(reference-q).clamp_min(0)*mask*strength
    requested=additions.sum(-1,keepdim=True)
    capacity=donor*.95
    ratio=torch.minimum(torch.ones_like(requested),capacity/requested.clamp_min(torch.finfo(p.dtype).tiny))
    additions=additions*ratio
    delta=additions.sum(-1,keepdim=True)
    qnew=q+additions-q*(~mask)*(delta/donor.clamp_min(torch.finfo(p.dtype).tiny))
    # lambda=0 is exactly unchanged: avoid divide/multiply round-trip noise.
    new_im=im if strength==0 else P*qnew
    out=p.clone(); out[hs[:,None],image_idx[None,:]]=new_im
    Pnew=new_im.sum(-1,keepdim=True)
    text_before=p.index_select(0,hs).sum(-1)-P[:,0]
    text_after=out.index_select(0,hs).sum(-1)-Pnew[:,0]
    region_after=((new_im/Pnew)*mask).sum(-1)
    diagnostics=torch.stack([region_before,region_after,P[:,0],Pnew[:,0],text_before,text_after,delta[:,0],(requested[:,0]>capacity[:,0]).float(),(out.index_select(0,hs).sum(-1)-p.index_select(0,hs).sum(-1)).abs()],dim=-1)
    if not bool(torch.isfinite(out).all()) or float(out.min())< -1e-7: raise ValueError('Invalid attention probabilities')
    if float((Pnew-P).abs().max())>=1e-3 or float((text_after-text_before).abs().max())>=1e-3 or float(diagnostics[:,-1].max())>=1e-3: raise ValueError('Image/text/row mass conservation failed')
    if strength>0 and (bool((region_after+1e-6<region_before).any()) or bool(((delta[:,0]>1e-6)&(region_after<=region_before)).any())): raise ValueError('Region share failed to rise with nonzero addition')
    return out,diagnostics

class Controller:
    def __init__(self,torch,refs,image_idx,condition,close_id):
        self.torch=torch;self.refs=refs;self.image_idx=np.asarray(image_idx);self.condition=condition;self.close_id=close_id
        self.stage=0; self.step=-1;self.decode=False;self.trace=[];self.checks=[];self.cache={};self.prefill_calls=0;self.decode_calls=0

    def before_forward(self,module,args,kwargs):
        ids=kwargs.get('input_ids', args[0] if args else None)
        if ids is None: raise ValueError('Generation forward missing input_ids')
        self.decode=ids.shape[-1]==1
        if self.decode:
            self.step+=1
            if int(ids[0,-1])==self.close_id:self.stage=1
        elif self.step>=0:raise ValueError('Unexpected second prefill')

    def tensors(self,layer,device):
        key=(layer,str(device),self.stage,self.condition)
        if key not in self.cache:
            t=self.torch; li=list(self.refs['layer_ids']).index(layer)
            heads=np.flatnonzero(self.refs['heads'][li]); kind='random' if self.condition=='restore_rand' else 'E'
            mask=self.refs[f'{kind}_mask'][self.stage,li,heads]
            ref=self.refs[f'{kind}_r'][self.stage,li,heads]
            self.cache[key]=(t.as_tensor(self.image_idx,dtype=t.long,device=device),t.as_tensor(heads,dtype=t.long,device=device),t.as_tensor(mask,dtype=t.bool,device=device),t.as_tensor(ref,dtype=t.float32,device=device))
        return self.cache[key]

    def attention(self,orig,module,query,key,value,attention_mask,*args,**kwargs):
        if not self.decode:
            self.prefill_calls+=1
            return orig(module,query,key,value,attention_mask,*args,**kwargs)
        t=self.torch; B,H,Q,D=query.shape; assert B==1 and Q==1
        self.decode_calls+=1
        rep=H//key.shape[1]; k=key.repeat_interleave(rep,dim=1);v=value.repeat_interleave(rep,dim=1)
        scale=kwargs.get('scaling',args[1] if len(args)>1 else None)
        scale=getattr(module,'scaling',D**-.5) if scale is None else scale
        sc=t.matmul(query.float(),k.float().transpose(-1,-2))*scale
        if attention_mask is not None:
            m=attention_mask[:,:,-1:,:k.shape[2]]
            sc=sc.masked_fill(~m,float('-inf')) if m.dtype==t.bool else sc+m.float()
        p=t.softmax(sc,dim=-1)
        layer=int(module.layer_idx);img,heads,mask,ref=self.tensors(layer,query.device)
        if heads.numel():
            p2,stats=redistribute(t,p[0,:,0],img,heads,mask,ref,0. if self.condition=='base' else 1.)
            p=p2[None,:,None,:]
            header=t.stack([t.full_like(heads,self.step),t.full_like(heads,layer),heads,t.full_like(heads,self.stage)],dim=-1).float()
            self.trace.append(t.cat([header,stats],dim=-1).detach().cpu().numpy())
        result=t.matmul(p,v.float()).to(query.dtype).transpose(1,2).contiguous()
        # All 16 layers at each of the first three decode steps of lambda=0.
        if self.condition=='base' and self.step<3:
            original=orig(module,query,key,value,attention_mask,*args,**kwargs)[0]
            diff=float((result.float()-original.float()).abs().max())
            self.checks.append(dict(step=self.step,layer=layer,max_abs_diff=diff))
            if not diff<.05:raise ValueError(f'lambda0 SDPA selfcheck failed: step={self.step} layer={layer} maxabs={diff}')
        return result,None

    def save(self,path):
        trace=np.concatenate(self.trace,axis=0) if self.trace else np.empty((0,13),np.float32)
        np.savez_compressed(path,trace=trace,columns=np.array(['step','layer','head','stage','region_before','region_after','image_before','image_after','text_before','text_after','delta','clamped','row_difference']))
        assert len(trace)>0 and self.prefill_calls==16 and self.decode_calls==16*(self.step+1)
        if self.condition=='base':assert len(self.checks)==48 and max(x['max_abs_diff'] for x in self.checks)<.05
        summary=dict(head_token_rows=len(trace),decode_steps=self.step+1,prefill_calls=self.prefill_calls,decode_calls=self.decode_calls,prefill_intervened=False,
          mean_delta=float(trace[:,10].mean()),region_before=float(trace[:,4].mean()),region_after=float(trace[:,5].mean()),clamped_head_tokens=int(trace[:,11].sum()),
          image_diff_max=float(np.abs(trace[:,7]-trace[:,6]).max()),text_diff_max=float(np.abs(trace[:,9]-trace[:,8]).max()),row_diff_max=float(trace[:,12].max()),
          active_head_tokens=int((trace[:,10]>1e-6).sum()),no_deficit_head_tokens=int((trace[:,10]<=1e-6).sum()),selfchecks=self.checks)
        summary['by_stage']={}
        for s,name in enumerate(['think','answer']):
            z=trace[trace[:,3]==s]
            summary['by_stage'][name]=dict(head_token_rows=len(z),mean_delta=float(z[:,10].mean()),region_before=float(z[:,4].mean()),region_after=float(z[:,5].mean()),clamped_head_tokens=int(z[:,11].sum())) if len(z) else dict(head_token_rows=0)
        return summary

def install(model):
    from symexp.attn import find_targets
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    found=find_targets(model);TARGETS.update(id(m) for _,m in found)
    layers=sorted(int(m.layer_idx) for _,m in found);assert layers==list(range(3,64,4))
    def wrap(orig):
        def call(module,query,key,value,attention_mask,*args,**kwargs):
            if ACTIVE is None or id(module) not in TARGETS:return orig(module,query,key,value,attention_mask,*args,**kwargs)
            return ACTIVE.attention(orig,module,query,key,value,attention_mask,*args,**kwargs)
        call._region_restore=True
        return call
    for name in ['sdpa']:
        orig=ALL_ATTENTION_FUNCTIONS[name]
        assert not getattr(orig,'_region_restore',False),'Double installation'
        wrapped=wrap(orig);ALL_ATTENTION_FUNCTIONS[name]=wrapped
        gm=getattr(type(ALL_ATTENTION_FUNCTIONS),'_global_mapping',None)
        if isinstance(gm,dict):gm[name]=wrapped
    return layers
