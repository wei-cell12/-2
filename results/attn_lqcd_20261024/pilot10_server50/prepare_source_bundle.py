"""A1: reuse original saved token IDs and original image bytes, without tokenization."""
import argparse, hashlib, json, time, zipfile, ast, struct, shutil
from pathlib import Path
import numpy as np

ROOTS = [
 '/mnt/sdi/linxin/proofwriter_runs/pathvqa-qwen38-27b-pilot10-20261018',
 '/mnt/sdi/linxin/proofwriter_runs/pathvqa-qwen38-27b-pressure150-20261018',
]

def main(out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 started=time.time(); checks=[]; rows=[]; seen=set()
 try:
  for root in map(Path,ROOTS):
   files=sorted((root/'results').glob('*/record.json'))
   assert len(files)==(20 if 'pilot10' in root.name else 280),(root,len(files))
   for f in files:
    r=json.loads(f.read_text()); ident=r['item_id']; c={'neutral':'neutral','pressure':'user_wrong'}[r['condition']]
    assert (ident,c) not in seen,('duplicate',ident,c)
    seen.add((ident,c)); inp=r['input_ids']; gen=r['output_ids']
    assert inp and gen and all(type(x)==int for x in inp+gen)
    img=Path(r['image_path']); img=img if img.is_absolute() else root/img
    h=hashlib.sha256(img.read_bytes()).hexdigest()
    assert h==r['image_sha256'],('image hash mismatch',ident,c)
    with np.load(f.with_name('attention.npz'),allow_pickle=False) as z:
     with zipfile.ZipFile(f.with_name('attention.npz')) as archive:
      with archive.open('generation_masses.npy') as array_file:
       assert array_file.read(6)==b'\x93NUMPY'
       major,minor=array_file.read(2)
       length=struct.unpack('<H' if major==1 else '<I',array_file.read(2 if major==1 else 4))[0]
       header=ast.literal_eval(array_file.read(length).decode('latin1'))
       n=header['shape'][0]
     assert len(gen)==n,('output length',ident,c,len(gen),n)
     assert np.array_equal(z['input_ids'].reshape(-1),inp),('input IDs differ',ident,c)
     assert np.array_equal(z['output_ids'].reshape(-1),gen),('output IDs differ',ident,c)
     layers=z['full_attention_layer_ids'].tolist()
     assert layers==list(range(3,64,4)),('layers',layers)
    assert len(r['regions']['image'])==r['image_tokens']
    image_target=out/'images'/(ident+'.jpg')
    image_target.parent.mkdir(exist_ok=True)
    if image_target.exists():assert hashlib.sha256(image_target.read_bytes()).hexdigest()==h
    else:shutil.copyfile(img,image_target)
    mass_target=out/'original_attention'/(ident+'__'+c+'.npy')
    mass_target.parent.mkdir(exist_ok=True)
    with np.load(f.with_name('attention.npz'),allow_pickle=False) as z:
     assert str(z['group_names'][0])=='image'
     np.save(mass_target,z['generation_masses'][...,0].copy())
    rows.append(dict(id=ident,condition=c,image='/mnt/sdg/linxin/pathvqa-attn-region-20261024/images/'+ident+'.jpg',input_ids=inp,output_ids=gen))
    checks.append(dict(id=ident,condition=c,source_record=str(f),source_npz=str(f.with_name('attention.npz')),image_sha256=h,input_tokens=len(inp),output_tokens=n,n_img=r['image_tokens'],image_grid_thw=r['image_grid_thw']))
  assert len(rows)==300 and len({x['id'] for x in rows})==150
  assert all({c for i,c in seen if i==ident}=={'neutral','user_wrong'} for ident in {x['id'] for x in rows})
  rows.sort(key=lambda r:(r['id'],r['condition']))
  checks.sort(key=lambda r:(r['id'],r['condition']))
  assert next(r for r in rows if r['id']=='PV-000' and r['condition']=='neutral')['output_ids'].__len__()==815
  assert next(r for r in rows if r['id']=='PV-000' and r['condition']=='user_wrong')['output_ids'].__len__()==1355
  manifest=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
  (out/'manifest_plain.jsonl').write_text(manifest)
  (out/'manifest_first3.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows[:3]))
  receipt=dict(ok=True,rows=300,paired_questions=150,source_counts=[20,280],retokenized=False,original_attention_extraction='FP32 generation_masses[...,0] saved as NPY; no rescaling',image_hashes_verified=True,token_arrays_verified=True,manifest_sha256=hashlib.sha256(manifest.encode()).hexdigest(),seconds=time.time()-started,items=checks)
  (out/'A1_validation.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
  print(json.dumps({k:v for k,v in receipt.items() if k!='items'}))
 except BaseException as e:
  (out/'A1_failure.json').write_text(json.dumps(dict(ok=False,error=repr(e),checked=len(checks),seconds=time.time()-started),ensure_ascii=False,indent=2)+'\n')
  raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();main(a.out)
