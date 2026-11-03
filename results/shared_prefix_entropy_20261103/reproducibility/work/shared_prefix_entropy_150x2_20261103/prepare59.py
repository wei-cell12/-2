import json,hashlib,shutil
from pathlib import Path
IDS={'ClockQA': ['CQ-000', 'CQ-001', 'CQ-002', 'CQ-003', 'CQ-004', 'CQ-006', 'CQ-008', 'CQ-009', 'CQ-010', 'CQ-011', 'CQ-013', 'CQ-014', 'CQ-015', 'CQ-016', 'CQ-017', 'CQ-018', 'CQ-021', 'CQ-022', 'CQ-023', 'CQ-024', 'CQ-025', 'CQ-026', 'CQ-027', 'CQ-028', 'CQ-029', 'CQ-030', 'CQ-032', 'CQ-033', 'CQ-034', 'CQ-035', 'CQ-036', 'CQ-037', 'CQ-038', 'CQ-039', 'CQ-040', 'CQ-041', 'CQ-042', 'CQ-043', 'CQ-044', 'CQ-045', 'CQ-046', 'CQ-048', 'CQ-049', 'CQ-050', 'CQ-051', 'CQ-052', 'CQ-053', 'CQ-055', 'CQ-057', 'CQ-061'], 'PathVQA': ['PV-000', 'PV-001', 'PV-003', 'PV-004', 'PV-005', 'PV-009', 'PV-010', 'PV-011', 'PV-013', 'PV-014', 'PV-015', 'PV-020', 'PV-021', 'PV-022', 'PV-026', 'PV-028', 'PV-029', 'PV-030', 'PV-034', 'PV-036', 'PV-037', 'PV-038', 'PV-040', 'PV-041', 'PV-043', 'PV-045', 'PV-047', 'PV-052', 'PV-054', 'PV-057', 'PV-059', 'PV-061', 'PV-063', 'PV-065', 'PV-066', 'PV-069', 'PV-071', 'PV-072', 'PV-076', 'PV-078', 'PV-079', 'PV-080', 'PV-081', 'PV-084', 'PV-085', 'PV-086', 'PV-089', 'PV-091', 'PV-096', 'PV-097'], 'PENDULUM': ['PD-re_used-0006', 'PD-re_used-0008', 'PD-re_used-0009', 'PD-re_used-0011', 'PD-re_used-0014', 'PD-re_used-0015', 'PD-re_used-0016', 'PD-re_used-0017', 'PD-re_used-0018', 'PD-re_used-0019', 'PD-re_used-0025', 'PD-re_used-0027', 'PD-re_used-0029', 'PD-re_used-0030', 'PD-re_used-0033', 'PD-re_used-0034', 'PD-re_used-0036', 'PD-re_used-0037', 'PD-re_used-0038', 'PD-re_used-0040', 'PD-re_used-0042', 'PD-re_used-0044', 'PD-re_used-0045', 'PD-re_used-0046', 'PD-re_used-0049', 'PD-self_captured-0007', 'PD-self_captured-0009', 'PD-self_captured-0010', 'PD-self_captured-0011', 'PD-self_captured-0012', 'PD-self_captured-0013', 'PD-self_captured-0014', 'PD-self_captured-0017', 'PD-self_captured-0021', 'PD-self_captured-0022', 'PD-self_captured-0023', 'PD-self_captured-0024', 'PD-self_captured-0025', 'PD-self_captured-0026', 'PD-self_captured-0029', 'PD-self_captured-0030', 'PD-self_captured-0031', 'PD-self_captured-0032', 'PD-self_captured-0033', 'PD-self_captured-0034', 'PD-self_captured-0037', 'PD-self_captured-0043', 'PD-self_captured-0046', 'PD-self_captured-0047', 'PD-self_captured-0048']}
W=Path('/mnt/sdb/linxin/shared-prefix-entropy-150x2-20261103')
if (W/'manifest.json').exists():raise RuntimeError('Already prepared, inspect before repeating')
W.mkdir(exist_ok=True)
roots={'ClockQA':Path('/mnt/sdb/linxin/conflict-cq-20261028'),'PathVQA':Path('/mnt/sdb/linxin/qwen38-pv-20261031'),'PENDULUM':Path('/mnt/sdb/linxin/pendulum-20261102/expand4x50')}
C=Path('/mnt/sdb/linxin/pathvqa-fp-a800-20261027/repo/configs/qwen38_fp_local.yaml')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
jobs=[];old=Path('/mnt/sdb/linxin/shared-prefix-entropy-multi12-20261103')
for ds,root in roots.items():
 items={r['id']:r for r in map(json.loads,(root/'items.jsonl').read_text().splitlines())}
 for iid in IDS[ds]:
  it=items[iid];rp=root/('qwen38/records' if ds=='PENDULUM' else 'obs/records');ns=rp/(iid+'__neutral.json');nr=json.loads(ns.read_text());im=sha(it['image'])
  assert im==nr['image_sha256']
  for cond in ['belief','belief_right']:
   src=rp/f'{iid}__{cond}.json';r=json.loads(src.read_text());key=iid+'__'+cond
   assert r['seed']==42 and r.get('input_ids') and r.get('output_ids') and im==r['image_sha256']
   if ds!='ClockQA':assert r['model']=='/mnt/sdb/linxin/ModelList/Qwen3.8-27B'
   assert r['message']==it['messages'][cond]
   job=dict(key=key,id=iid,model='qwen38',dataset=ds,condition=cond,item=it,source=str(src),neutral_source=str(ns),source_sha256=sha(src),neutral_sha256=sha(ns),image_sha256=im,output_ids_sha256=hashlib.sha256(json.dumps(r['output_ids'],separators=(',',':')).encode()).hexdigest(),n_out=len(r['output_ids']),n_in=r['n_in'],config=str(C),config_sha256=sha(C))
   op=old/(key+'.json')
   if op.exists():
    oc=json.loads(op.read_text())
    assert oc['source_sha256']==job['source_sha256'] and oc['output_ids_sha256']==job['output_ids_sha256'] and oc['neutral_prompt_record_sha256']==job['neutral_sha256']
    job['reuse_root']=str(old)
   jobs.append(job)
loads=[0]*4
for j in sorted(jobs,key=lambda r:r['n_out']+r['n_in'],reverse=True):
 rank=min(range(4),key=lambda k:loads[k]);j['rank']=rank;loads[rank]+=j['n_out']+j['n_in']
manifest=dict(model='qwen38',questions=150,trajectories=300,shards=4,repo='/mnt/sdb/linxin/pathvqa-fp-a800-20261027/repo',python='/home/linxin/miniconda3/envs/vllm_new/bin/python',config=str(C),config_sha256=sha(C),model_path='/mnt/sdb/linxin/ModelList/Qwen3.8-27B',max_memory=None,jobs=jobs,reused=sum('reuse_root' in j for j in jobs),shard_loads=loads)
(W/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print(json.dumps(dict(questions=150,trajectories=300,reused=manifest['reused'],max_tokens=max(j['n_out'] for j in jobs),shards=[sum(j['rank']==k for j in jobs) for k in range(4)],manifest_sha256=sha(W/'manifest.json'))))
