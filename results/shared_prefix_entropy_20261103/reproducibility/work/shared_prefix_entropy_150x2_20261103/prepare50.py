import json,hashlib,shutil,os,subprocess
from pathlib import Path
import yaml
W=Path('/mnt/sdg/linxin/shared-prefix-entropy-150x2-20261103');D=W/'inputs';src=json.loads((D/'source_manifest.json').read_text())
if (W/'manifest.json').exists():raise RuntimeError('Already prepared; do not overwrite')
C=W/'config_used.yaml';cfg=yaml.safe_load((D/'config_original.yaml').read_text());cfg['model_path']='/mnt/sdg/ljh/models/gemma-4-31B-it';C.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False))
assert Path(cfg['model_path']).exists()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
jobs=[]
for ds,meta in src.items():
 items={r['original_item']['id']:r for r in meta['items']};records={(r['id'],r['condition']):r for r in meta['records']}
 for iid,im in items.items():
  it=dict(im['original_item']);it['image']=str(D/im['package_image']);assert sha(it['image'])==im['image_sha256']
  ni=records[iid,'neutral'];ns=D/ni['package_record'];assert sha(ns)==ni['source_sha256']
  for cond in ['belief','belief_right']:
   ri=records[iid,cond];p=D/ri['package_record'];r=json.loads(p.read_text());assert sha(p)==ri['source_sha256'];assert r['message']==it['messages'][cond]
   jobs.append(dict(key=iid+'__'+cond,id=iid,model='gemma4',dataset=ds,condition=cond,item=it,source=str(p),source_original_path=ri['original_path'],neutral_source=str(ns),source_sha256=ri['source_sha256'],neutral_sha256=ni['source_sha256'],image_sha256=im['image_sha256'],output_ids_sha256=ri['output_ids_sha256'],n_out=ri['n_out'],n_in=r['n_in'],config=str(C),config_sha256=sha(C)))
loads=[0]*3
for j in sorted(jobs,key=lambda r:r['n_out']+r['n_in'],reverse=True):
 k=min(range(3),key=lambda k:loads[k]);j['rank']=k;loads[k]+=j['n_out']+j['n_in']
M=dict(model='gemma4',questions=150,trajectories=300,shards=3,repo='/mnt/sdg/linxin/repo-s2va-20261101',python='/mnt/sdg/linxin/pathvqa-attn-region-20261024/env/bin/python',config=str(C),config_sha256=sha(C),model_path=cfg['model_path'],max_memory={'0':'34GiB','1':'34GiB'},jobs=jobs,reused=0,shard_loads=loads,source_config_sha256=sha(D/'config_original.yaml'),runtime_changes=['model_path localized to existing50 weights','GPU loading max_memory34GiB per visible card; device_map auto unchanged'])
(W/'manifest.json').write_text(json.dumps(M,ensure_ascii=False,indent=2));print(json.dumps(dict(questions=150,trajectories=300,max_tokens=max(j['n_out'] for j in jobs),shards=[sum(j['rank']==k for j in jobs) for k in range(3)],manifest_sha256=sha(W/'manifest.json'))))
py=M['python'];r=subprocess.run([py,'-c','import torch,transformers;from transformers import Gemma4ForConditionalGeneration;print(torch.__version__,transformers.__version__)'],capture_output=True,text=True,env=dict(os.environ,PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES=''));print(r.stdout,r.stderr);assert r.returncode==0
