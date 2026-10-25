"""Read-only native attention, with online stage maps instead of a second replay."""
import numpy as np
import torch

GROUPS = ['image', 'task_text', 'user_pressure', 'input_other', 'generated_history']

class Observer:
    def __init__(self, layers, heads, chunk=16):
        self.layers=list(layers);self.index={v:i for i,v in enumerate(layers)}
        self.heads=heads;self.active=False

    def reset(self, entry, on_step=None):
        self.n=entry['input_tokens'];self.qindices=[self.n-1]
        self.regions=[entry['regions'][g] for g in GROUPS[:-1]]
        assert sorted(i for group in self.regions for i in group)==list(range(self.n))
        self.image=entry['regions']['image'];self.on_step=on_step
        self.prefill=np.full((1,len(self.layers),self.heads,5),np.nan,np.float32)
        self.rows=[];self.norm_error=0.;self.mass_error=0.;self.active=True
        self.phase=0;self.counts=[0,0];self.accepted=0
        self.spatial_sum=None;self.current_spatial=None;self.selfchecks=[]

    def accept(self, token, close_id, eos):
        # Called once after sampling, so boundary and EOS rows are not assigned
        # to the preceding stage merely because they were predicted there.
        assert len(self.rows)==self.accepted+1
        self.accepted+=1
        if token==close_id:self.phase=1;return
        if token in eos:return
        self.spatial_sum[self.phase].add_(self.current_spatial.double())
        self.counts[self.phase]+=1

    def weights(self, query, key, mask, scale, selected):
        qi=torch.as_tensor(selected,device=query.device)
        keys=key.repeat_interleave(query.shape[1]//key.shape[1],dim=1)
        score=torch.matmul(query.index_select(-2,qi).float(),keys.float().transpose(-1,-2))*scale
        absolute=qi+key.shape[-2]-query.shape[-2]
        causal=torch.arange(key.shape[-2],device=query.device)[None,:]<=absolute[:,None]
        score.masked_fill_(~causal[None,None,:,:],float('-inf'))
        if mask is not None:
            m=mask if mask.shape[-2]==1 else mask.index_select(-2,qi)
            m=m[...,:key.shape[-2]]
            if m.dtype==torch.bool:score.masked_fill_(~m,float('-inf'))
            else:score+=m.float()
        p=torch.softmax(score,-1,dtype=torch.float32)
        self.norm_error=max(self.norm_error,float((p.sum(-1)-1).abs().max()))
        return p

    def register(self):
        from transformers import AttentionInterface,AttentionMaskInterface
        from transformers.masking_utils import sdpa_mask
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        original=ALL_ATTENTION_FUNCTIONS['sdpa']
        def wrapped(module,query,key,value,attention_mask,scaling=None,dropout=0.,**kwargs):
            native=original(module,query,key,value,attention_mask,scaling=scaling,dropout=dropout,**kwargs)
            if not self.active or module.__class__.__name__!='Qwen3_5Attention':return native
            li=self.index[module.layer_idx];assert query.shape[:2]==(1,self.heads) and dropout==0
            step=key.shape[-2]-self.n;assert step>=0
            if li==0:
                self.current=torch.zeros((len(self.layers),self.heads,5),device=query.device,dtype=torch.float32)
                self.current_spatial=torch.zeros((len(self.layers),self.heads,len(self.image)),device=query.device,dtype=torch.float32)
                if self.spatial_sum is None:
                    self.spatial_sum=torch.zeros((2,len(self.layers),self.heads,len(self.image)),device=query.device,dtype=torch.float64)
                self.seen=set();self.step=step
            assert self.step==step and li not in self.seen
            assert query.shape[-2]==(self.n if step==0 else 1)
            p=self.weights(query,key,attention_mask,scaling,[self.n-1] if step==0 else [0])
            partition=torch.stack([p[...,r].sum(-1) for r in self.regions]+[p[...,self.n:].sum(-1)],dim=-1)
            self.mass_error=max(self.mass_error,float((partition.sum(-1)-1).abs().max()))
            self.current[li]=partition[0,:,0,:]
            image_index=torch.as_tensor(self.image,dtype=torch.long,device=p.device)
            self.current_spatial[li].copy_(p[0,:,0,:].index_select(-1,image_index))
            if step==0:self.prefill[0,li]=partition[0,:,0,:].detach().cpu().numpy()
            if 0<step<=3:
                v=value.repeat_interleave(query.shape[1]//value.shape[1],dim=1)
                explicit=(p@v.float()).to(query.dtype).transpose(1,2).contiguous()
                diff=float((explicit.float()-native[0].float()).abs().max())
                self.selfchecks.append(dict(layer=int(module.layer_idx),decode_step=step,max_abs_diff=diff))
                assert diff<.05,('Read-only attention selfcheck',module.layer_idx,step,diff)
            self.seen.add(li)
            if li==len(self.layers)-1:
                assert len(self.seen)==len(self.layers) and step==len(self.rows)
                self.rows.append(self.current.detach().cpu().numpy())
                if self.on_step and step%64==0:self.on_step(step)
            return native
        AttentionInterface.register('mathvision_spatial_readonly',wrapped)
        AttentionMaskInterface.register('mathvision_spatial_readonly',sdpa_mask)
        return 'mathvision_spatial_readonly'

    def finish(self):
        self.active=False;rows=np.stack(self.rows)
        assert np.isfinite(rows).all() and np.isfinite(self.prefill).all()
        assert self.norm_error<4e-6 and self.mass_error<4e-6
        return rows,dict(max_row_normalization_error=self.norm_error,max_partition_error=self.mass_error,
                         native_attention_output_modified=False,online_spatial=True,
                         selfchecks=self.selfchecks,recorded_prefill_queries='last input query only')

    def spatial(self):
        sums=self.spatial_sum.detach().cpu().numpy()
        means=np.stack([sums[s]/self.counts[s] if self.counts[s] else np.zeros_like(sums[s]) for s in range(2)]).astype(np.float32)
        return means,np.asarray(self.counts,np.int32)

