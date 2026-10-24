"""Upstream replay, with documented CPU weight offload and fail-fast validation gates."""
import json, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;SOURCE=ROOT/'source'
sys.path.insert(0,str(SOURCE))
import torch,transformers
EXPECTED={'torch':'2.11.0+cu130','transformers':'5.14.1'}

def fail(step,error):
 (ROOT/'failure.json').write_text(json.dumps(dict(step=step,error=str(error),time_unix=time.time()),ensure_ascii=False,indent=2)+'\n')
 raise SystemExit(str(error))
if torch.__version__!=EXPECTED['torch'] or transformers.__version__!=EXPECTED['transformers']:
 fail('environment',dict(torch=torch.__version__,transformers=transformers.__version__))
cls=transformers.Qwen3_5ForConditionalGeneration
original_load=cls.from_pretrained

def load(*args,**kw):
 assert kw.get('torch_dtype')==torch.bfloat16
 assert not kw.get('load_in_4bit') and not kw.get('load_in_8bit')
 kw['max_memory']={0:'36GiB','cpu':'128GiB'}
 model=original_load(*args,**kw)
 allocation={k:str(v) for k,v in model.hf_device_map.items()}
 assert all(v in ['0','cuda:0','cpu'] for v in allocation.values()),allocation
 assert any(v in ['0','cuda:0'] for v in allocation.values()),allocation
 (ROOT/'model_allocation.json').write_text(json.dumps(dict(device_map=allocation,max_memory={'GPU'+os.environ.get('PATHVQA_PHYSICAL_GPU','7'):'36GiB','cpu':'128GiB'},torch=torch.__version__,transformers=transformers.__version__,dtype=str(model.dtype),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES')),indent=2)+'\n')
 return model
cls.from_pretrained=staticmethod(load)
import symexp.attn as attn
import symexp.attn_replay as replay
original_install=attn.install; original_replay=replay.replay_item

def checked_install(*args,**kwargs):
 result=original_install(*args,**kwargs)
 if result!=list(range(3,64,4)):fail('A2_module_check',repr(result))
 (ROOT/'module_check.json').write_text(json.dumps(dict(ok=True,layers=result,count=len(result)))+'\n')
 return result

def checked_replay(backend,ids,prompt_len,*args,**kwargs):
 try:
  result=original_replay(backend,ids,prompt_len,*args,**kwargs)
  diffs=result['selfcheck_max_abs_diff']
  assert len(diffs)>0 and all(float(x)<=0.05 for x in diffs),('selfcheck',diffs.tolist())
  assert float(result['count_prefill_end'])==1
  assert int(result['count_think'])+int(result['count_answer'])+1==len(ids)-prompt_len
  assert result['layer_ids'].tolist()==list(range(3,64,4))
  assert result['pos_image_mass'].shape==(len(ids)-prompt_len,16,24)
  return result
 except Exception as e:
  import traceback
  (ROOT/'forward_failure_traceback.log').write_text(traceback.format_exc())
  fail('A2_replay_check',repr(e))
attn.install=checked_install; replay.replay_item=checked_replay
import runpy
sys.argv=[str(SOURCE/'run.py')]+sys.argv[1:]
runpy.run_path(str(SOURCE/'run.py'),run_name='__main__')
