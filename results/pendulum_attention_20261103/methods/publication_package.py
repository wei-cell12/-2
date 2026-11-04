import json,hashlib,collections,shutil,gzip,re,html
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlparse,unquote
B=Path(__file__).resolve().parent;P=B.parent;S=P/'three_model_results_20261103';REL=Path('results/pendulum_attention_20261103');T=B/'tree'/REL
assert not T.exists(),'Inspect existing publication tree before rebuilding'
summary=json.loads((S/'summary.json').read_text());receipt=json.loads((S/'validation_receipt.json').read_text());assert receipt['replies']==1800 and receipt['models']==3
for name,h in receipt['files_sha256'].items():assert hashlib.sha256((S/name).read_bytes()).hexdigest()==h,name
T.mkdir(parents=True)
def cp(a,b):b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a,b)
for f in S.rglob('*'):
 if f.is_file():cp(f,T/f.relative_to(S))
arrays=json.loads((S/'array_manifest.json').read_text());checks={};ids_by_model={};all_images=set();attention_images=set();source_hashes={}
models=['qwen38','gemma4','qwen36_moe'];conds=['neutral','belief','belief_right']
for model in models:
 export=json.loads((P/('qwen_judge_export_private.json' if model=='qwen38' else model+'_judge_export_private.json')).read_text())['records'];assert len(export)==600
 raw={(r['id'],r['condition']):r for r in export};assert len(raw)==600
 jf=P/'qwen_results_20261102/judgments.json' if model=='qwen38' else S/(model+'_judgments.json');labels=json.loads(jf.read_text());assert len(labels)==600;joined={(r['id'],r['condition']):r for r in labels};assert set(raw)==set(joined)
 ids_by_model[model]={r['id']:r['image_sha256'] for r in export};assert len(ids_by_model[model])==200
 counts={}
 for condition in conds:
  rows=[r for r in labels if r['condition']==condition];assert len(rows)==200;cnt=collections.Counter(r['status'] for r in rows);expected=summary['models'][model]['conditions'][condition];assert all(cnt[k]==expected[k] for k in ['correct','wrong','unresolved']);counts[condition]=dict(cnt)
 for k,r in joined.items():
  src=raw[k];assert r['source_sha256']==src['source_sha256'] and r['image_sha256']==src['image_sha256'];q=r['answer_quote']
  if q:
   field=r['quote_source'];assert field in ['reply','thinking','raw_output'];assert q in src[field],(model,k,'quote not found')
 for r in export:
  assert re.fullmatch(r'PD-[a-z_]+-\d{4}',r['id']);fn=r['id']+'__'+r['condition']+'.json';q=T/'records'/model/fn;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(json.dumps(r,ensure_ascii=False,indent=2));all_images.add(r['image_sha256']);source_hashes[model+'/'+fn]=r['source_sha256']
 cp(jf,T/(model+'_judgments.json'))
 # Directly reconstruct the full neutral-to-user-error transition table from final labels.
 tr={a:{b:0 for b in ['correct','wrong','unresolved']} for a in ['correct','wrong','unresolved']}
 for iid in ids_by_model[model]:tr[joined[iid,'neutral']['status']][joined[iid,'belief']['status']]+=1
 assert tr==summary['models'][model]['transitions']['belief']['table']
 # All900 array metadata entries refer to the exact same source replies.
 aa=arrays[model];assert len(aa)==300 and len({x['key'] for x in aa})==300
 for r in aa:
  iid,c=r['key'].rsplit('__',1);assert iid.startswith(('PD-re_used-','PD-self_captured-'));assert r['source_sha256']==raw[iid,c]['source_sha256'];assert r['bytes']>0 and len(r['recorded_array_sha256'])==64;attention_images.add(raw[iid,c]['image_sha256'])
 prefix='qwen' if model=='qwen38' else model;f=P/(prefix+'_attention_metrics_private.json');data=f.read_bytes();x=json.loads(data);assert x['metrics']['questions']==100
 out=T/'attention_metrics'/(model+'.json.gz');out.parent.mkdir(exist_ok=True);out.write_bytes(gzip.compress(data,compresslevel=6,mtime=0));checks[model]=dict(replies=600,conditions=counts,source_quotes_checked=True,source_sha_join_passed=True,neutral_to_wrong_transition_verified=True,array_metadata=300,metrics_uncompressed_sha256=hashlib.sha256(data).hexdigest(),metrics_uncompressed_bytes=len(data),metrics_gzip_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),windows={g:{w:len(v['all_eligible_ids']) for w,v in gg.items()} for g,gg in summary['models'][model]['attention'].items()})
assert ids_by_model['qwen38']==ids_by_model['gemma4']==ids_by_model['qwen36_moe'];assert len(all_images)==84 and len(attention_images)==23
# Preserve human-semantic extraction/reviews and corrections; no opaque image copies or token-ID records.
for src,dest in [(P/'qwen_annotation_20261102',T/'annotation_audit/qwen38'),(P/'additional_annotation_20261103',T/'annotation_audit/additional_models'),(P.parent/'blind_pendulum_20261103',T/'annotation_audit/blind_additional_models')]:
 for f in src.rglob('*'):
  if not f.is_file() or f.suffix not in ['.md','.json','.jsonl','.log']:continue
  if any(x in ['images','traces'] for x in f.relative_to(src).parts[:-1]):continue
  cp(f,dest/f.relative_to(src))
for n in ['attention_collection.md','expansion_task.md','run_spec.json','expansion_run_spec.json','attention_integrity51.json','attention_integrity59.json','attention_first_item_audit51.json','source_field_anomalies.json','test_spatial_recorder.py','spatial_recorder.py','collect_spatial.py','compute_qwen_attention.py','compute_additional_attention.py','finish_qwen_analysis.py','finish_three_model_analysis.py','additional_label_validation.py','label_validation.py','append_three_model_conclusions.py','build_three_model_cases.py','build_three_model_overview.py','prepare_primary_case_maps.py','extract_primary_case_maps.py']:
 cp(P/n,T/'methods'/n)
cp(P/'expand4x50_items.jsonl',T/'items.jsonl');cp(P.parent/'annotation_policy.md',T/'methods/annotation_policy.md');cp(Path(__file__).resolve(),T/'methods/publication_package.py')
for f in (P/'configs').glob('*.yaml'):cp(f,T/'methods/configs'/f.name)
for f in (P/'expand_stage59/configs').glob('*.yaml'):cp(f,T/'methods/configs_server59'/f.name)
# Check all12 scientific images decode, without recreating them.
from PIL import Image
heatmaps=json.loads((S/'heatmap_manifest.json').read_text());assert len(heatmaps)==12
for r in heatmaps:
 with Image.open(T/r['file']) as im:im.verify()
rows=['# PENDULUM三模型：区域注意力、JSD与模型回复（2026-11-03）','','发布于2026-11-04。此包是此前完成的独立注意力实验归档；[共享前缀熵差实验](../shared_prefix_entropy_20261103/README.md)另有目录。','','## 从这里开始','','- [完整报告与结论](REPORT.md)：三条件正确率、中性答对后的变化、区域注意力、JSD、原图聚类区间及原文案例。','- [区域热图和完整案例页面](cases.html)；[12张热图](heatmaps/)（HTML请下载本目录后打开）。','- [全部结构化统计与每个窗口的样本ID](summary.json)。','- [1800份完整思考和回答导出](records/)及三份`*_judgments.json`，可逐题核对。','- [三模型共900份数组的路径、字节和SHA](array_manifest.json)；[发布复核凭据](publication_validation.json)。','','## 题目数量和实验方法','','三款模型：Qwen3.8-27B、Gemma4-31B-it、Qwen3.6-35B-A3B MoE。基线生成使用同一批200题：易混淆视角、视觉谜题、复用图像、自拍图像各50题；每题中性、用户说错、用户说对，各模型600份，合计1800份。200题共84张不同原图。','','注意力仅覆盖复用图像与自拍图像各50题，合计100题、23张不同原图；每模型300条回放，共900条。把已有回复原样喂回、被动记录原注意力函数，未重新生成。本批未做注意力恢复干预。完整收集定义与记录维度见[方法说明](methods/attention_collection.md)。','','- 匹配思考前64/128/256/512个token；三条件长度都足够且图像全部可见才进入该窗口，不填充。每个窗口独立保存入组ID。','- 各层类按三条件平均图像注意力选前25%看图头：Qwen96/384、MoE40/160、Gemma全局80/320与滑动400/1600。固定96头另作敏感性，不将其混成MoE主分析。','- E是每头中性完整思考的前20%高关注图像块；报告图像总注意力与图像内部E份额，两者分母不同。E是模型自定参照区域。','- JSD先逐头计算再平均，单位nats；同时报告原始JSD和减去条件内部交替8-token组噪声后的差值，保留负数。Gemma全局层、滑动窗口层分别报告。','- 守住/变错/未解按同一模型的中性答对题定义，另给三模型共同中性答对交集。置信区间按原图聚类。','','## 标注与复核','','复用同一个GPT-6 Luna(max)。新增两模型先封存1200份盲化答案抽取，再揭示参照进行语义匹配、解释审核和主张采纳判读；历史Qwen600份保留。原标注、更正、指令与封存信息见annotation_audit/。同一判读者见过本数据集历史内容，报告未将它描述为全新独立盲审。','','答案语义匹配与解释联合正确分开列示；原文没有解释不等于没有作答。42题参考争议并集仅用于另列敏感性分析，未改原金标。','','发布时重新逐条核对1800份导出与最终标注的来源SHA、图像SHA和连续引文，独立重建全部条件计数及中性→用户说错的转移表；900份数组元数据全部与对应原回复SHA一致；12张PNG可正常解码。原报告已有16项文件SHA也全部匹配。','','## 文件与服务器原始数组','','`records/`保存原始思考、最终回答及raw_output的文字导出，不含完整整数token ID。字段source_sha256是服务器原记录的SHA；本次导出文件自身的SHA另列file_manifest.json。`attention_metrics/*.json.gz`是完整CPU统计中间结果与区域图数据，gzip解压即可读取；原始字节SHA见publication_validation.json。','','NPZ逐层逐头大数组保留在59/51服务器，GitHub只上传统计、热图、模型文字、元数据与复核代码。数组路径、大小、SHA均在array_manifest.json；methods保留采集、选头、JSD、分组统计及画图流程代码。没有上传权重、原图文件或完整token ID列表；热图中含案例图像用于定位。','','运行环境与配置保留原记录：51 torch2.8.0+cu128、transformers5.20.2；59 torch2.11.0+cu130、transformers5.14.1。Qwen前20题来自51，其余180题来自59，来源与版本差异保留在methods/expansion_task.md中。此次发布不启动模型或占用GPU。']
(T/'README.md').write_text('\n'.join(rows)+'\n')
validation=dict(ok=True,published_date='2026-11-04',experiment_date='2026-11-03',models=3,questions_per_model=200,replies=1800,unique_images=84,attention_questions_per_model=100,attention_records=900,attention_unique_images=23,heatmaps=12,prior_report_hashes_verified=len(receipt['files_sha256']),checks=checks,source_record_sha256=source_hashes,no_new_inference=True)
(T/'publication_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2))
# Fail publication on invalid structured data, accidental token-ID lists or sensitive credential markers.
for p in T.rglob('*'):
 if not p.is_file():continue
 assert p.suffix not in ['.npz','.npy','.safetensors'] and p.stat().st_size<50_000_000,p
 if p.suffix=='.json':json.loads(p.read_text())
 if p.suffix in ['.py','.md','.json','.jsonl','.log','.yaml','.html']:
  raw=p.read_bytes();assert not re.search(rb'https://[^\s/@]+@github\.com',raw),p
  assert not re.search(rb'-----BEGIN (?:OPENSSH|RSA|EC|PRIVATE) (?:PRIVATE )?KEY-----',raw),p
  if p.suffix in ['.json','.jsonl']:assert not re.search(rb'"(?:input_ids|output_ids|generated_ids|token_ids)"\s*:\s*\[',raw),p
# Verify archived report pages have no broken local links.
missing=[]
class Links(HTMLParser):
 def handle_starttag(self,tag,attrs):
  for k,v in attrs:
   if k not in ['href','src'] or not v:continue
   u=urlparse(v)
   if u.scheme or u.netloc or not u.path:continue
   if not (self.file.parent/unquote(u.path)).exists():missing.append((str(self.file.relative_to(T)),v))
for f in T.glob('*.html'):
 a=Links();a.file=f;a.feed(f.read_text())
assert not missing,missing
files={str(p.relative_to(T)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(T.rglob('*')) if p.is_file()};(T/'file_manifest.json').write_text(json.dumps(dict(files=files,n_files_excluding_this_manifest=len(files),manifest_self_excluded=True),ensure_ascii=False,indent=2));out=dict(target=str(REL),directory=str(T.resolve()),files=len(files)+1,bytes=sum(p.stat().st_size for p in T.rglob('*') if p.is_file()),annotation_received=1800,attention_records=900,complete=True);(B/'package_receipt.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
