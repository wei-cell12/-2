import json,hashlib,io,tarfile,base64,os
from pathlib import Path
SEL={'ClockQA': ['CQ-000', 'CQ-001', 'CQ-002', 'CQ-003', 'CQ-004', 'CQ-006', 'CQ-008', 'CQ-009', 'CQ-010', 'CQ-011', 'CQ-013', 'CQ-014', 'CQ-015', 'CQ-016', 'CQ-017', 'CQ-018', 'CQ-021', 'CQ-022', 'CQ-023', 'CQ-024', 'CQ-025', 'CQ-026', 'CQ-027', 'CQ-028', 'CQ-029', 'CQ-030', 'CQ-032', 'CQ-033', 'CQ-034', 'CQ-035', 'CQ-036', 'CQ-037', 'CQ-038', 'CQ-039', 'CQ-040', 'CQ-041', 'CQ-042', 'CQ-043', 'CQ-044', 'CQ-045', 'CQ-046', 'CQ-048', 'CQ-049', 'CQ-050', 'CQ-051', 'CQ-052', 'CQ-053', 'CQ-055', 'CQ-057', 'CQ-061'], 'PathVQA': ['PV-000', 'PV-001', 'PV-003', 'PV-004', 'PV-005', 'PV-009', 'PV-010', 'PV-011', 'PV-013', 'PV-014', 'PV-015', 'PV-020', 'PV-021', 'PV-022', 'PV-026', 'PV-028', 'PV-029', 'PV-030', 'PV-034', 'PV-036', 'PV-037', 'PV-038', 'PV-040', 'PV-041', 'PV-043', 'PV-045', 'PV-047', 'PV-052', 'PV-054', 'PV-057', 'PV-059', 'PV-061', 'PV-063', 'PV-065', 'PV-066', 'PV-069', 'PV-071', 'PV-072', 'PV-076', 'PV-078', 'PV-079', 'PV-080', 'PV-081', 'PV-084', 'PV-085', 'PV-086', 'PV-089', 'PV-091', 'PV-096', 'PV-097'], 'PENDULUM': ['PD-re_used-0006', 'PD-re_used-0008', 'PD-re_used-0009', 'PD-re_used-0011', 'PD-re_used-0014', 'PD-re_used-0015', 'PD-re_used-0016', 'PD-re_used-0017', 'PD-re_used-0018', 'PD-re_used-0019', 'PD-re_used-0025', 'PD-re_used-0027', 'PD-re_used-0029', 'PD-re_used-0030', 'PD-re_used-0033', 'PD-re_used-0034', 'PD-re_used-0036', 'PD-re_used-0037', 'PD-re_used-0038', 'PD-re_used-0040', 'PD-re_used-0042', 'PD-re_used-0044', 'PD-re_used-0045', 'PD-re_used-0046', 'PD-re_used-0049', 'PD-self_captured-0007', 'PD-self_captured-0009', 'PD-self_captured-0010', 'PD-self_captured-0011', 'PD-self_captured-0012', 'PD-self_captured-0013', 'PD-self_captured-0014', 'PD-self_captured-0017', 'PD-self_captured-0021', 'PD-self_captured-0022', 'PD-self_captured-0023', 'PD-self_captured-0024', 'PD-self_captured-0025', 'PD-self_captured-0026', 'PD-self_captured-0029', 'PD-self_captured-0030', 'PD-self_captured-0031', 'PD-self_captured-0032', 'PD-self_captured-0033', 'PD-self_captured-0034', 'PD-self_captured-0037', 'PD-self_captured-0043', 'PD-self_captured-0046', 'PD-self_captured-0047', 'PD-self_captured-0048']}
W=Path('/mnt/sdb/linxin/shared-prefix-entropy-150x2-20261103');W.mkdir(exist_ok=True)
if (W/'gemma_sources_v2.tar.gz').exists():raise RuntimeError('Source package already exists; do not replace')
roots={'ClockQA':Path('/mnt/sdb/linxin/gemma4-cq-20261029'),'PathVQA':Path('/mnt/sdb/linxin/gemma4-pv-20261030'),'PENDULUM':Path('/mnt/sdb/linxin/pendulum-20261102/expand4x50')}
manifest={};images=set()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
with tarfile.open(W/'gemma_sources_v2.tar.gz','w:gz') as tar:
 def addtext(n,obj):
  data=json.dumps(obj,ensure_ascii=False).encode();info=tarfile.TarInfo(n);info.size=len(data);tar.addfile(info,io.BytesIO(data))
 for ds,root in roots.items():
  items={r['id']:r for r in map(json.loads,(root/'items.jsonl').read_text().splitlines())};manifest[ds]=dict(items=[],records=[])
  for iid in SEL[ds]:
   it=items[iid];rr=root/('gemma4/records' if ds=='PENDULUM' else 'obs/records');im=Path(it['image']);assert im.exists()
   ext=im.suffix;iname=f'images/{ds}/{iid}{ext}'
   if iname not in images:tar.add(im,arcname=iname,recursive=False);images.add(iname)
   manifest[ds]['items'].append(dict(original_item=it,package_image=iname,image_sha256=sha(im)))
   for cond in ['neutral','belief','belief_right']:
    p=rr/f'{iid}__{cond}.json';r=json.loads(p.read_text());assert r.get('input_ids') and r.get('output_ids'),str(p)
    assert r['image_sha256']==sha(im);assert r['seed']==42
    dest=f'records/{ds}/{iid}__{cond}.json';tar.add(p,arcname=dest,recursive=False)
    manifest[ds]['records'].append(dict(id=iid,condition=cond,original_path=str(p),package_record=dest,source_sha256=sha(p),n_out=r['n_out'],output_ids_sha256=hashlib.sha256(json.dumps(r['output_ids'],separators=(',',':')).encode()).hexdigest()))
 for ds,p in [('ClockQA','/mnt/sdb/linxin/gemma4-compare-20261029/time_judge/judgments.json')]:
  fp=Path(p);assert fp.exists(),p
  manifest[ds]['judgments_source']=dict(path=str(fp),sha256=sha(fp));tar.add(fp,arcname=f'judgments/{ds}.json',recursive=False)
 addtext('source_manifest.json',manifest)
 tar.add('/mnt/sdb/linxin/repo-gemma4/configs/gemma4_31b.yaml',arcname='config_original.yaml',recursive=False)
print(json.dumps(dict(manifest=manifest,package_bytes=(W/'gemma_sources_v2.tar.gz').stat().st_size,package_sha256=sha(W/'gemma_sources_v2.tar.gz'),package_b64=base64.b64encode((W/'gemma_sources_v2.tar.gz').read_bytes()).decode()),ensure_ascii=False))
