import json,shutil,hashlib,gzip,html,re,collections,csv
from pathlib import Path
B=Path(__file__).resolve().parent;W=B.parent;WORK=W.parent;SRC=WORK/'shared_prefix_entropy_150x2_20261103';REL=Path('results/shared_prefix_entropy_20261103');T=B/'tree'/REL
lock=json.loads((W/'completion.json').read_text());assert lock['complete'] and lock['received']==701
confirmation=json.loads((B/'numerical_confirmation.json').read_text());assert confirmation['ok']
subprocess_result=None
import subprocess
r=subprocess.run(['python3',str(W/'validate.py')],capture_output=True,text=True);assert r.returncode==0,r.stdout+r.stderr
if T.exists():raise RuntimeError('Publication tree already exists; inspect its receipt before creating another package')
T.mkdir(parents=True)
def cp(a,b):
 b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a,b)
def cptree(a,b):
 for p in a.rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.pyc','.npz','.npy']:cp(p,b/p.relative_to(a))
cptree(SRC/'results',T)
REPRO=T/'reproducibility';WR=REPRO/'work';compressed=[]
for source,name in [(SRC,'shared_prefix_entropy_150x2_20261103'),(W,'entropy_valley_behavior_20261103'),(WORK/'entropy_valley_luna_20261103','entropy_valley_luna_20261103')]:
 dest=WR/name;dest.mkdir(parents=True,exist_ok=True)
 for p in source.iterdir():
  if not p.is_file() or p.suffix not in ['.py','.md','.json','.log','.yaml','.patch']:continue
  if p.name in ['plot_payload.json','valley_plots_base64.json','results50_package.json','results59_package.json','export51.json','backend50.log','diagnostic59_receipt.json']:continue
  if p.suffix=='.json' and p.stat().st_size>5_000_000:
   q=dest/(p.name+'.gz');raw=p.read_bytes();q.write_bytes(gzip.compress(raw,compresslevel=6,mtime=0));compressed.append(dict(path=str(q.relative_to(REPRO)),original_sha256=hashlib.sha256(raw).hexdigest(),original_bytes=len(raw),gzip_sha256=hashlib.sha256(q.read_bytes()).hexdigest()))
  else:
   name=p.name
   if name=='repair_reuse59_receipt.json':name='repair_reuse59_receipt.log'  # Historical CLI text, not JSON; retain exact bytes.
   cp(p,dest/name)
 if (source/'blind').is_dir():cptree(source/'blind',dest/'blind')
 if (source/'gemma_scope_audit').is_dir():cptree(source/'gemma_scope_audit',dest/'gemma_scope_audit')
# Exact saved source attention for all100 aligned QwenClockQA trajectories.
a=json.loads((W/'attention_summary.json').read_text());AS=WORK/'clockqa_uncertainty_attention_20261101';AD=WR/AS.name;cp(AS/'measure.py',AD/'measure.py');am=[]
for z in a['coverage_records']:
 p=Path(z['attention_source']);assert hashlib.sha256(p.read_bytes()).hexdigest()==z['attention_sha256'];cp(p,AD/'results/records'/p.name);m=AS/'results/meta'/p.name.replace('.jsonl.gz','.json');exists=m.exists()
 if exists:cp(m,AD/'results/meta'/m.name)
 am.append(dict(key=z['key'],tokens=z['tokens'],token_match=z['token_match'],relative_path=str((AD/'results/records'/p.name).relative_to(T)),sha256=z['attention_sha256'],source_meta_included=exists))
(T/'attention_source_manifest.json').write_text(json.dumps(am,ensure_ascii=False,indent=2));(REPRO/'compressed_files.json').write_text(json.dumps(compressed,indent=2));cp(B/'numerical_confirmation.json',T/'numerical_confirmation.json')
for n in ['verify_numerics.py','package_results.py','verification_tests.log']:cp(B/n,REPRO/'publication'/n)
cp(WORK/'annotation_policy.md',REPRO/'annotation_policy.md')
for n in ['reason.json','feature_correction.patch','readout_comparison.json']:cp(B/'prepublication_position_feature_audit'/n,REPRO/'prepublication_feature_audit'/n)
# CPU bootstrap restores only compressed JSON and the analysis data link. No network/GPU.
bootstrap='''import gzip,json,hashlib\nfrom pathlib import Path\nB=Path(__file__).resolve().parent\nfor x in json.loads((B/'compressed_files.json').read_text()):\n p=B/x['path'];q=p.with_suffix('');b=gzip.decompress(p.read_bytes());assert hashlib.sha256(b).hexdigest()==x['original_sha256']\n if q.exists():assert hashlib.sha256(q.read_bytes()).hexdigest()==x['original_sha256']\n else:q.write_bytes(b)\np=B/'work/shared_prefix_entropy_150x2_20261103/results';target=Path('../../..')\nif p.is_symlink():assert p.resolve()==B.parent.resolve()\nelif p.exists():raise RuntimeError('Existing results path is not the expected symlink')\nelse:p.symlink_to(target,target_is_directory=True)\nprint('CPU reproduction inputs ready; no inference launched.')\n'''
(REPRO/'bootstrap.py').write_text(bootstrap)
(REPRO/'README.md').write_text('''# 复核与复现\n\n此目录保留实际使用的前向/统计/语义校验代码、原判读与更正、盲化请求、解盲映射、配置来源SHA及源数据清单。全量逐token数值在顶层qwen38/records与gemma4/records；完整模型思考/回答在各analysis/source_texts.json。原图与权重不随本次上传。\n\nCPU复核（Python3与numpy）：\n\n```bash\npython3 bootstrap.py\npython3 work/entropy_valley_behavior_20261103/validate.py\npython3 work/entropy_valley_behavior_20261103/check_analysis_helpers.py\npython3 work/entropy_valley_behavior_20261103/inventory.py\npython3 work/entropy_valley_behavior_20261103/attention_test.py\npython3 work/entropy_valley_behavior_20261103/online_attention.py\npython3 work/entropy_valley_behavior_20261103/secondary_attention.py\npython3 work/entropy_valley_behavior_20261103/finish.py\n```\n\nbootstrap只解压带SHA的JSON并建立到本包结果的本地软链接，不下载、不启动模型；后续统计脚本会在本地重新写分析输出。请在副本中运行以保留下载文件的原始SHA。旧运行器含当时服务器路径，仅为留档，复核本包不需要执行launch/resume/prepare_blind或重新判读。项目Luna标签是语义阅读结果，不能用程序关键词替代。\n\nGemma配置内容见顶层gemma4/config_used.yaml。Qwen本机未保留完全相同的配置文本；manifest.json已保存当时配置绝对路径与SHA256=3c417bba9fb6160f64101909530700e9458c85110916259eb30603a26f298eaf。本次上传时原59共享连接未连接，未为了补配置而改变模型设置或要求重登录；这不影响已有CSV和CPU统计复核。历史源Gemma生成环境transformers5.20.2，熵回放在50使用5.14.1，版本差异在主说明中保留。\n\nprivate_keymap是审核结束后的题目/条件解盲映射，不含凭据。盲审期间Luna未读取这些映射。审核者为同一个gpt-6-luna(max)，历轮不能当作独立多裁判。原标注和修正分开保存。\n''')
s=json.loads((T/'local_valleys/behavior_extension/summary.json').read_text());assert s['complete'] and s['received']==701
sec=json.loads((W/'secondary_attention_summary.json').read_text())['secondary_deep'];on=s['online_attention']['thresholds']['1']
readme=['# 共享前缀熵差与局部波谷行为分析（2026-11-03）','','已完成并复核：Qwen3.8-27B、Gemma4-31B-it；ClockQA、PathVQA、PENDULUM各同一批50题。150道题，两模型各150题，每题用户说错/说对两条既有轨迹，共600条，536646个输出token位置。不是600道独立题；原图89张，复用题按原图分组。没有新生成回答。','','## 先看哪里','','- [完整波谷行为分析](local_valleys/behavior_extension/REPORT.md)：深浅谷、普通位置、行为富集、留出读出、具体设问触发用途。','- [600条轨迹总体报告](REPORT.md)；[交互轨迹页面](index.html)（HTML需下载本目录后本地打开）。','- [逐条行为原文与标签](local_valleys/behavior_extension/events.json)；[结构化统计](local_valleys/behavior_extension/summary.json)。','- [逐token数值复核凭据](numerical_confirmation.json)；[CPU复核代码与说明](reproducibility/README.md)。','','## 算的是什么','','固定已有受压输出前缀y[:k]，比较pU(.|图像,问题,用户话语,y[:k])与p0(.|图像,问题,y[:k])的完整词表原始logits熵。两分支都喂实际受压前缀，中性分支不喂自己的预测。ΔH=HU−H0，单位nats；CSV的第k+1行是生成该token之前的预测。','','负差说明该位置用户信息带来了额外的下一步确定性，适合定位用户信息重新介入判断的核查窗口。局部负差段是数值事件，不与独立语义事件/独立题目划等号。','','## 这次确认的具体结果','',f'排除每条最大谷后，仍有255处次级深谷；旧轮逐段判读中133处前文有明确疑惑/候选未决，91处接着重新引用或评价用户，59处呈现读图→核对用户→再读图。类别可重叠。', '',f'Qwen ClockQA50题100条轨迹与既有注意力逐token完全匹配。76处次级深谷对原输入用户断言及附加请求的注意力平均跳升+{sec["valley_user_jump_pp"]:.2f}个百分点，匹配普通位置{sec["control_jump_pp"]:.2f}个百分点；配对差+{sec["difference_pp"]:.2f}个百分点，原图聚类95%区间+{sec["image_cluster95_pp"][0]:.2f}～+{sec["image_cluster95_pp"][1]:.2f}。使用16标准层全部384头，不限于96个预选头。', '',f'在线可实现规则：每个连续负差段第一次ΔH≤−1提示一次，不等待事后最低点。167次提示覆盖91条轨迹，用户输入注意力平均跳升+{on["mean_user_jump_pp"]:.2f}个百分点；按原图等权，{on["user_jump_ge2pp"]:.1%}同时出现≥2个百分点跳升，普通位置为{s["online_attention"]["ordinary"]["user_jump_ge2pp"]:.1%}。该比例的目标是明确的注意力行为，不是最终答错率。', '', '## 新增深浅谷对照：判读已完整结束','','固定哈希分层抽取701个位置，来自143道题、88张图、379条模型轨迹：161深谷、180中谷、180浅谷、180普通位置。两模型、三数据集、两用户条件均覆盖；同轨迹普通位置优先匹配相同token，否则同类型。Luna只见原用户输入及局部原文，模型/条件/深度/选点组别/gold/结果均隐藏。行为和标题/计划角色分开，用户对象/图像对象可同时标注。','','|位置组|样本位置数|处理用户主张|前文明确疑惑|核对/比较/排查矛盾|决定答案|','|---|---:|---:|---:|---:|---:|']
cn={'deep':'深谷≤−1','medium':'中谷−1～−0.25','shallow':'浅谷−0.25～−0.05','quiet_control':'普通位置绝对熵差≤0.05'}
for k in ['deep','medium','shallow','quiet_control']:
 z=s['bands'][k];readme.append('|'+cn[k]+'|'+str(z['n'])+'|'+'|'.join(f'{z["targets"][t]["n"]}（{z["targets"][t]["rate"]:.1%}）' for t in ['user_processing','explicit_uncertainty_before','fact_checking','answer_commitment'])+'|')
readme+=['','上表是分层抽检计数，不是自然全谷总体频率。中谷与180个普通位置的配对差、原图聚类区间、模型/数据集分层及按原图分组的5折留出AUC，见完整行为报告。新旧轮的行为定义不同，旧互斥“答复组织”中包含核对用户的标题/计划，不能拿旧主标签比例当作全部用户信息处理率。','','## 发布前修正与确认','','探索分类器的初版位置特征用过最终轨迹长度。发布前已改成log1p(当前输出步数)，重新计算全部AUC，并加入未来长度不变性测试；完整报告和summary.json均为修正结果，旧值只保存在[撤回/更正审计](reproducibility/prepublication_feature_audit/readout_comparison.json)，不作为当前性能结论。','','## 数值确认和原始记录','','- 600份CSV的步号、输出长度、源SHA、同题同图对齐、熵差恒等式和JSD范围全部通过；最大熵差恒等误差2.22e−7。','- 独立从CSV重建87341个连续负差段，与保存数据逐段一致：深谷615、中谷5337、浅谷29284、极浅段52105。','- 6项本地核心数值/同前缀因果测试通过；既有两个cached历史测试日志同时保留。701条语义标签覆盖完整，格式、逐字引文及焦点交叠校验全部通过。','- 前向门槛保持原值：实际token前20比例≥.98；前缀JSD<.001、熵差绝对偏差<.1nats；公式误差<1e−4。全部接受记录通过。','- BF16整段与单前缀检查不通过的5条保留失败现场并改用原生逐token缓存完整重算两支，原门槛不变。Qwen282整段+14旧同源复用+4缓存；Gemma299整段+1缓存。未把被拒绝数值混入有效CSV。','- 历史int/模型生成不属于本次重新执行：此次只是复用已有轨迹回放并做CPU分析。','','## 模型、环境和保存范围','','Qwen在59跑四个worker，GPU0～3；Gemma在50跑三个双卡worker，GPU0/1、2/3、4/5，每卡权重上限34GiB、device_map auto。计算及保存后进程全部退出，未建立常驻模型或占卡进程。回放环境torch2.11.0+cu130、transformers5.14.1。历史Gemma在51生成时transformers5.20.2，此版本差异保留，不能将本次回放描述为与原生成环境逐版本完全相同。','','每条records/*.json保留耗时、检查值、源SHA、图像SHA和方法；manifest/日志与release_receipt保留设备、版本、耗时来源。Gemma配置文本随包，Qwen原配置路径与SHA保留，详见复现说明。','','上传含全部CSV、模型原文、判读原始文件及更正、报告、曲线、代码、100份既有注意力压缩记录及源元数据、SHA/字节清单。没有模型权重、原图、NPZ/NPY、完整整数token ID列表或登录凭据。较大的可重建JSON以gzip保存，bootstrap校验SHA后恢复。']
(T/'README.md').write_text('\n'.join(readme)+'\n')
# Current findings HTML must reflect its preserved markdown.
p=T/'local_valleys/behavior_extension/CURRENT_FINDINGS.md';(p.parent/'current_findings.html').write_text('<!doctype html><meta charset="utf-8"><title>已确认的波谷行为</title><pre style="white-space:pre-wrap">'+html.escape(p.read_text())+'</pre>')
# Scan only publication content for forbidden artifacts or accidental credential strings.
bad=[]
for p in T.rglob('*'):
 if not p.is_file():continue
 if p.suffix in ['.npz','.npy','.safetensors'] or p.stat().st_size>=50_000_000:bad.append(str(p.relative_to(T)))
 if p.suffix in ['.py','.json','.md','.yaml','.log','.html','.csv','.jsonl']:
  b=p.read_bytes()
  if re.search(rb'https://[^\s/@]+@github\.com',b) or re.search(rb'-----BEGIN (?:OPENSSH|RSA|EC|PRIVATE) (?:PRIVATE )?KEY-----',b):bad.append('credential-pattern:'+str(p.relative_to(T)))
assert not bad,bad
files={str(p.relative_to(T)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(T.rglob('*')) if p.is_file()}
(T/'file_manifest.json').write_text(json.dumps(dict(files=files,n_files_excluding_this_manifest=len(files),total_bytes_excluding_this_manifest=sum(x['bytes'] for x in files.values()),forbidden_artifact_scan='passed',manifest_self_excluded=True),ensure_ascii=False,indent=2))
receipt=dict(target=str(REL),directory=str(T),files=len(files)+1,bytes=sum(p.stat().st_size for p in T.rglob('*') if p.is_file()),annotation_received=701,source_trajectories=600,maximum_file_bytes=max(p.stat().st_size for p in T.rglob('*') if p.is_file()),complete=True)
(B/'package_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
