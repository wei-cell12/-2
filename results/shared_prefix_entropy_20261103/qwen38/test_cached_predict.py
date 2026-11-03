import ast,types,unittest
from pathlib import Path
import torch
src=Path(__file__).with_name('worker_native_fallback.py').read_text();tree=ast.parse(src);ns={}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['reset','predict_cached']],type_ignores=[]),'cached_functions','exec'),ns)
predict_cached=ns['predict_cached']
class Toy:
 device=torch.device('cpu');dtype=torch.float32
 def __init__(self):self.histories=[]
 def __call__(self,input_ids,past_key_values=None,**kw):
  hist=[] if past_key_values is None else list(past_key_values);rows=[]
  for token in input_ids[0].tolist():
   hist.append(token);center=sum((i+1)*t for i,t in enumerate(hist))%7;rows.append(-torch.square(torch.arange(7,dtype=torch.float32)-center));self.histories.append(list(hist))
  logits=torch.stack(rows)[None]
  if kw.get('logits_to_keep')==1:logits=logits[:,-1:]
  return types.SimpleNamespace(logits=logits,past_key_values=hist)
class CachedTests(unittest.TestCase):
 def backend(self):return types.SimpleNamespace(torch=torch,model=Toy())
 def test_cached_shared_actual_prefix(self):
  b=self.backend();prompt=[6,2,1];actual=[4,3,0,5];r=predict_cached(b,{'input_ids':torch.tensor([prompt])},actual)
  expected=[]
  for k in range(len(actual)):
   hist=prompt+actual[:k];center=sum((i+1)*t for i,t in enumerate(hist))%7;expected.append(-torch.square(torch.arange(7,dtype=torch.float32)-center))
  torch.testing.assert_close(r,torch.stack(expected),rtol=0,atol=0)
  self.assertEqual(b.model.histories[-1],prompt+actual[:-1])
 def test_cached_future_not_used(self):
  p={'input_ids':torch.tensor([[6,2,1]])};a=predict_cached(self.backend(),p,[4,3,0,5]);b=predict_cached(self.backend(),p,[4,3,6,2]);torch.testing.assert_close(a[:3],b[:3],rtol=0,atol=0)
if __name__=='__main__':unittest.main()
