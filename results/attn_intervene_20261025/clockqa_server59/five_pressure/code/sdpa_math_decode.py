"""Proposed protocol change: math SDPA for standard-layer decode only."""
def install():
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    original=ALL_ATTENTION_FUNCTIONS['sdpa']
    assert not getattr(original,'_clockqa_math_decode',False),'Already installed'
    def call(module,query,key,value,attention_mask,*args,**kwargs):
        if module.__class__.__name__=='Qwen3_5Attention' and query.shape[-2]==1:
            with sdpa_kernel(SDPBackend.MATH):
                return original(module,query,key,value,attention_mask,*args,**kwargs)
        return original(module,query,key,value,attention_mask,*args,**kwargs)
    call._clockqa_math_decode=True
    ALL_ATTENTION_FUNCTIONS['sdpa']=call
    mapping=getattr(type(ALL_ATTENTION_FUNCTIONS),'_global_mapping',None)
    if isinstance(mapping,dict):mapping['sdpa']=call
