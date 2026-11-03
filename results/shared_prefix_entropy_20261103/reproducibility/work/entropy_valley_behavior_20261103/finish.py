"""Mechanical aggregation of locked semantic annotations and existing numerical records; no inference."""
import sys,subprocess,os,json,hashlib,statistics,collections,html,csv,math
from pathlib import Path
B=Path(__file__).resolve().parent
try:import numpy as np
except ImportError:
 py='/Users/zzh/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3'
 raise SystemExit(subprocess.call([py,str(Path(__file__).resolve())]))
from validate import load
req,lab,errors,files=load()
assert not errors,errors
assert len(lab)==len(req)==701,(len(lab),len(req))
D=json.loads((B/'design.json').read_text()); meta=json.loads((B/'private_keymap.json').read_text())
for name,sha in D['request_sha256'].items():assert hashlib.sha256((B/'blind'/name).read_bytes()).hexdigest()==sha
O=B.parent/'shared_prefix_entropy_150x2_20261103/results/local_valleys/behavior_extension';O.mkdir(parents=True,exist_ok=True)
rows=[]
for k,x in lab.items():
 m=meta[k];r=dict(**m,blind_key=k,request=req[k],judgment=x)
 r['targets']={'user_processing':'user_claim' in x['object_tags'],'explicit_uncertainty_before':x['uncertainty_before']=='yes','fact_checking':x['behavior'] in ['user_claim_check','alternative_comparison','inconsistency_check'],'visual_processing':'image' in x['object_tags'],'answer_commitment':x['behavior']=='answer_commitment','output_formulation':x['behavior']=='output_formulation'}
 rows.append(r)
rows.sort(key=lambda r:(r['model'],r['dataset'],r['key'],r['step'],r['candidate_type']))
TARGETS=list(rows[0]['targets']);KINDS=['deep','medium','shallow','quiet_control'];rng=np.random.default_rng(20261103)
def stats(rr):
 z=dict(n=len(rr),trajectories=len({(r['model'],r['key']) for r in rr}),images=len({r['image_sha256'] for r in rr}),behaviors=dict(collections.Counter(r['judgment']['behavior'] for r in rr)),roles=dict(collections.Counter(r['judgment']['token_role'] for r in rr)),targets={t:dict(n=sum(r['targets'][t] for r in rr),rate=statistics.mean(r['targets'][t] for r in rr)) for t in TARGETS})
 return z
bands={k:stats([r for r in rows if r['candidate_type']==k]) for k in KINDS};strata={f'{m}:{ds}:{k}':stats([r for r in rows if r['model']==m and r['dataset']==ds and r['candidate_type']==k]) for m in ['qwen38','gemma4'] for ds in ['ClockQA','PathVQA','PENDULUM'] for k in KINDS if any(r['model']==m and r['dataset']==ds and r['candidate_type']==k for r in rows)}
# Matched medium and ordinary positions are a paired test, not independent rows.
look={(r['model'],r['key'],r['step']):r for r in rows if r['candidate_type']=='medium'};pairs=[]
for r in rows:
 if r['candidate_type']=='quiet_control':
  v=look[(r['model'],r['key'],r['matched_medium_step'])];pairs.append((v,r))
paired={}
for t in TARGETS:
 imgs=sorted({v['image_sha256'] for v,c in pairs});d=np.array([statistics.mean(int(v['targets'][t])-int(c['targets'][t]) for v,c in pairs if v['image_sha256']==im) for im in imgs]);boot=d[rng.integers(0,len(d),(3000,len(d)))].mean(1)
 paired[t]=dict(n_pairs=len(pairs),images=len(imgs),valley_yes=sum(v['targets'][t] for v,c in pairs),control_yes=sum(c['targets'][t] for v,c in pairs),image_equal_difference_pp=float(d.mean()*100),image_cluster95_pp=(np.quantile(boot,[.025,.975])*100).tolist(),valley_only=sum(v['targets'][t] and not c['targets'][t] for v,c in pairs),control_only=sum(c['targets'][t] and not v['targets'][t] for v,c in pairs))
# Full numerical inventory, not just semantic sample. Decompose same-prefix entropy gap against past8.
allv=json.loads((B/'valleys_all.json').read_text());morph={}
for k in ['deep','medium','shallow']:
 rr=[r for r in allv if r['band']==k];valid=[r for r in rr if r['drop_with_U'] is not None and r['rise_without_U'] is not None]
 morph[k]=dict(n=len(rr),valid_baseline=len(valid),mean_H_with_U=statistics.mean(r['H_with_U'] for r in rr),mean_H_without_U=statistics.mean(r['H_without_U'] for r in rr),mean_drop_U=statistics.mean(r['drop_with_U'] for r in valid),mean_rise_without_U=statistics.mean(r['rise_without_U'] for r in valid),with_U_below1_n=sum(r['H_with_U']<1 for r in rr),without_U_above1_n=sum(r['H_without_U']>=1 for r in rr),both_rise_from_prior8_n=sum(r['drop_with_U']<0 and r['rise_without_U']>0 for r in valid),with_U_drop_and_without_U_rise_n=sum(r['drop_with_U']>0 and r['rise_without_U']>0 for r in valid))
# CPU-only image-group out-of-fold behavioral readout, no token text, identity, outcome or future data.
feat=[];groups=[]
for r in rows:
 missing=r['prior16_H_pressure'] is None or r['prior16_H_neutral'] is None;a=r['H_with_U'] if missing else r['prior16_H_pressure'];b=r['H_without_U'] if missing else r['prior16_H_neutral'];feat.append([-r['delta_H'],r['JSD'],r['H_with_U'],r['H_without_U'],a,b,r['H_with_U']-a,r['H_without_U']-b,math.log1p(r['step']),float(missing)]);groups.append(r['dataset']+':'+r['image_sha256'])
X=np.array(feat,float);unique=sorted(set(groups),key=lambda g:hashlib.sha256(('fold:'+g).encode()).hexdigest());foldmap={g:i%5 for i,g in enumerate(unique)};fold=np.array([foldmap[g] for g in groups]);
def auc(y,s):
 pos=s[y==1];neg=s[y==0]
 return None if len(pos)==0 or len(neg)==0 else float(((pos[:,None]>neg).sum()+.5*(pos[:,None]==neg).sum())/(len(pos)*len(neg)))
def cv(y,cols):
 p=np.zeros(len(y))
 for f in range(5):
  tr=fold!=f;te=~tr;a=X[tr][:,cols];b=X[te][:,cols];mu=a.mean(0);sd=a.std(0);sd=np.where(sd<1e-8,1,sd);a=np.c_[np.ones(len(a)),(a-mu)/sd];b=np.c_[np.ones(len(b)),(b-mu)/sd];w=np.zeros(a.shape[1]);yt=y[tr];sw=np.where(yt==1,.5/max(yt.mean(),1e-6),.5/max(1-yt.mean(),1e-6))
  for _ in range(400):
   pr=1/(1+np.exp(-np.clip(a@w,-30,30)));grad=a.T@((pr-yt)*sw)/len(a);grad[1:]+=.03*w[1:];w-=.15*grad
  p[te]=1/(1+np.exp(-np.clip(b@w,-30,30)))
 return p
readout={};preds={}
for t in TARGETS:
 y=np.array([r['targets'][t] for r in rows],int)
 if min(y.sum(),len(y)-y.sum())<20:continue
 scores={};ps={}
 for name,cols in [('depth_only',[0]),('depth_jsd',[0,1]),('branch_entropy_and_history',list(range(10)))]:
  p=cv(y,cols);scores[name]=dict(oof_auc=auc(y,p));ps[name]=p
 readout[t]=dict(n=len(y),positive=int(y.sum()),models=scores);preds[t]={name:p.tolist() for name,p in ps.items()}
summary=dict(complete=True,expected=len(req),received=len(rows),questions=D['source_questions'],images=D['source_images'],trajectories=D['source_model_trajectories'],sample_counts=D['counts'],sampling=D['rule'],bands=bands,strata=strata,medium_paired_control=paired,morphology=morph,readout=readout,readout_design='Exploratory 5-fold held-out original-image groups, shared across models/questions/conditions. Training-only standardization, fixed balanced logistic regression L2=.03,400iterations,lr=.15. Scores concern the balanced candidate/control sample, not deployment precision. Features current entropy/JSD, prior16 entropies and log1p(current output step); missing prior-history baseline uses current entropy with an explicit missing flag (no samples discarded); no future valley endpoint/rank/width, token text, source/condition or outcome. Labels from one reused Luna, not independent annotators.',label_sha256=files,request_sha256=D['request_sha256'],attention=json.loads((B/'attention_summary.json').read_text())['bands'],online_attention=json.loads((B/'online_attention_summary.json').read_text()))
(O/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));(O/'events.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));(O/'oof_scores.json').write_text(json.dumps(dict(keys=[r['blind_key'] for r in rows],image_fold=fold.tolist(),scores=preds),ensure_ascii=False));(O/'effective_labels.json').write_text(json.dumps(lab,ensure_ascii=False,indent=2))
CN={'deep':'深谷（≤−1）','medium':'中谷（−1～−0.25）','shallow':'浅谷（−0.25～−0.05）','quiet_control':'匹配普通位置（绝对值≤0.05）'}
BCN={'visual_observation':'直接读图','visual_inference':'由图推断答案','alternative_comparison':'比较自己的候选','inconsistency_check':'检查自身矛盾','user_claim_retrieval':'重新引入用户主张','user_claim_check':'核对用户主张','prompt_intent_resolution':'解释用户意图/题意','answer_commitment':'决定或撤回答案','output_formulation':'组织格式与措辞','language_link':'语言/分隔衔接','other':'其他','unclear':'未解'}
TCN={'user_processing':'处理用户主张','explicit_uncertainty_before':'前文明确疑惑','fact_checking':'核对/比较/排查矛盾','visual_processing':'处理图像证据','answer_commitment':'决定答案','output_formulation':'组织格式措辞'}
lines=['# 波谷究竟在检测什么：深浅局部谷与普通位置的行为对照','',f'已完成：两模型、三数据集的600条既有轨迹；新语义复核覆盖{len(rows)}个位置，来自{D["source_questions"]}道题、{D["source_images"]}张原图。521处局部谷（161深、180中、180浅）加180个普通位置。位置数不是题数。没有重新生成或使用GPU。','', '## 选点与判读','', '排除开头64 token及首个实质思考段落（取较长者）。每个连续负熵差段取一个最低点，保留后续所有段，包含第二、第三等局部谷。600条轨迹中，深谷615处、中谷5337处、浅谷29284处；同一思考动作可以跨多段，不能把它们当独立样本。','', '为覆盖深浅及两模型三数据集，而非只挑故事，从每个模型×数据集×用户条件×深度组固定哈希选至多15条不同轨迹，各随机哈希取一个谷。普通位置与中谷同轨迹匹配，优先相同token，否则同token类型。Luna仅见原用户输入和局部原文，组别、深度、模型、条件、gold、结果全部隐藏。用同一Luna(max)继续复核，属于扩展一致性审核。下表为抽检计数，分层结果保留于summary.json。','', '|位置|数量|处理用户主张|前文明确疑惑|核对/比较/排查矛盾|处理图像证据|决定答案|组织格式措辞|','|---|---:|---:|---:|---:|---:|---:|---:|']
for k in KINDS:
 z=bands[k];lines.append('|'+CN[k]+'|'+str(z['n'])+'|'+'|'.join(f'{z["targets"][t]["n"]}（{z["targets"][t]["rate"]:.1%}）' for t in TARGETS)+'|')
lines+=['','## 每一类谷附近的具体动作','','|动作|深谷|中谷|浅谷|普通位置|','|---|---:|---:|---:|---:|']
for b,cn in BCN.items():lines.append('|'+cn+'|'+'|'.join(str(bands[k]['behaviors'].get(b,0)) for k in KINDS)+'|')
lines+=['','## 与普通位置配对后的行为富集','','180对中谷/普通位置，按原图等权；置信区间按原图整体bootstrap。标题/计划和行为对象分开标，避免把“Verify user”标题算成非用户行为。','','|行为|中谷处|普通位置|原图等权差（百分点）|95%区间|','|---|---:|---:|---:|---:|']
for t,z in paired.items():lines.append(f'|{TCN[t]}|{z["valley_yes"]}|{z["control_yes"]}|{z["image_equal_difference_pp"]:+.2f}|{z["image_cluster95_pp"][0]:+.2f}～{z["image_cluster95_pp"][1]:+.2f}|')
lines+=['','## 已保存注意力给出的直接验证','','Qwen ClockQA的50题、100条轨迹全部逐token匹配既有注意力。用户注意力为原输入中题干后用户断言及附加请求的总份额，统计16标准层384头；“跳升”指当前位置减前8个思考token平均，单位百分点。普通位置在同轨迹优先匹配相同token，否则同类型。以下按原图等权。','','|谷深|谷底用户注意力跳升|普通位置|配对差|配对差95%区间|','|---|---:|---:|---:|---:|']
for z in summary['attention']:lines.append(f'|{CN[z["band"]]}|{z["mean_valley_user_jump_pp"]:+.2f}|{z["mean_control_user_jump_pp"]:+.2f}|{z["paired_user_jump_difference_pp"]:+.2f}|{z["image_cluster_bootstrap95"][0]:+.2f}～{z["image_cluster_bootstrap95"][1]:+.2f}|')
z=summary['online_attention']['thresholds']['1'];lines += ['',f'可实施触发：每个负差段第一次ΔH≤−1就触发，不等待事后谷底。167次触发覆盖91条轨迹；触发时用户注意力平均跳升{z["mean_user_jump_pp"]:.2f}个百分点，{z["user_jump_ge2pp"]:.1%}达到≥2个百分点跳升（原图等权）。普通位置对应比例为{summary["online_attention"]["ordinary"]["user_jump_ge2pp"]:.1%}。因此深负差可作为“模型正在重新使用用户断言”的核查时机。','', '## 深谷的两分支变化','','ΔH=H有用户−H无用户。负差是在完全相同已生成前缀下，用户输入带来了额外的下一token确定性。应分别查看两支，而非把负差直接读为受压分支自身熵上升。','','|谷深|谷数|H有用户均值|H无用户均值|有用户较前8步下降|无用户较前8步上升|','|---|---:|---:|---:|---:|---:|']
for k,z in morph.items():lines.append(f'|{CN[k]}|{z["n"]}|{z["mean_H_with_U"]:.3f}|{z["mean_H_without_U"]:.3f}|{z["mean_drop_U"]:+.3f}|{z["mean_rise_without_U"]:+.3f}|')
lines+=['','## 给触发器的具体用法','','1. 用户主张重新进入/核对：深负差是优先的事实核查入口。子Agent只看原图与原问题独立回答，主Agent再对照自己的主张，避免子Agent再接收用户暗示。','2. 比较自己的候选或检查自身矛盾：触发定向读图，例如只核对指针长度、位置或病理区域；问具体设问，不泛泛要求“再想一遍”。','3. 决定答案：在承诺答案前核对“这是图上证据，还是用户说法”；用户主张与图证的来源分离。','4. 解释输入意图/格式、组织措辞或语言衔接：作为流程状态标记；根据下表的行为读出和分层原文决定是否调用视觉检查。','','上面是可执行的触发用途映射。此次测量的是行为位置与注意力，触发后纠错收益需由后续续写验证另记，不能混入本次行为统计。','', '## 只用当前及过去数值的行为读出','','5折按原图分组，所有模型/条件/复用题同图在同折。固定逻辑回归，未使用文本、gold、答对错、未来谷宽/谷底排名；与仅谷深、谷深+JSD对照。AUC属于这批分层候选/普通位置的留出表现，不能换写为全生成token准确率。','','|目标|阳性/总数|仅深度AUC|深度+JSD|两支熵+历史|','|---|---:|---:|---:|---:|']
for t,z in readout.items():lines.append('|'+TCN[t]+f'|{z["positive"]}/{z["n"]}|'+ '|'.join(f'{z["models"][n]["oof_auc"]:.3f}' for n in ['depth_only','depth_jsd','branch_entropy_and_history'])+'|')
lines+=['','## 原文例子（每类固定最多3个，按源顺序，非挑选最佳案例）','']
for behavior,cn in BCN.items():
 ex=[r for r in rows if r['candidate_type']!='quiet_control' and r['judgment']['behavior']==behavior][:3]
 if not ex:continue
 lines+=['### '+cn,'']
 for r in ex:
  j=r['judgment'];lines += [f'- {r["model"]} / {r["dataset"]} / {r["key"]} / token {r["step"]} / ΔH={r["delta_H"]:.3f} nats：{j["note"]}',f'  引文：{json.dumps(j["evidence_quote"],ensure_ascii=False)}']
lines+=['','## 发布前位置特征复核','','初版探索读出曾使用当前步数/最终轨迹长度作为位置特征。发布前已移除这项未来长度信息，改为log1p(当前已知输出步数)，保留其余训练规则、标签与原始熵不变，并重新计算上述全部留出AUC。当前报告与summary.json均为修正结果；旧值仅在发布审计中作为撤回记录，不用于触发器性能结论。已增加测试：改变未知最终长度时，所有输入特征保持不变。','','## 保留与复核','','events.json包含每个位置、原文上下文、源SHA、有效判读与选点概率；原判读/更正在工作目录分别保存。summary.json保留全部分层、matched controls、原图聚类区间和完整模型读出定义；oof_scores.json保留折与逐条留出分数。浅谷数量远多于深谷，语义抽检比例不是天然全谷总体频率。思考不足首段排除范围的轨迹依然保留，未强凑事件。']
report='\n'.join(lines)+'\n';(O/'REPORT.md').write_text(report)
# Render a standalone searchable audit view; plain prose report is escaped intact.
content='<!doctype html><meta charset="utf-8"><title>波谷行为核验</title><style>body{font:16px/1.7 system-ui;max-width:1160px;margin:36px auto;padding:0 24px;color:#182334}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{border:1px solid #ddd;padding:10px;margin:10px 0}blockquote{background:#f5f7fa;padding:10px}</style><h1>深浅波谷的行为含义与触发用途</h1><p>Ctrl/Cmd+F搜索题号、动作、模型。<a href="REPORT.md">报告</a> · <a href="summary.json">统计</a></p><pre>'+html.escape(report)+'</pre><h2>全部701处原文</h2>'
for r in rows:
 j=r['judgment'];q=r['request'];title=f'{r["candidate_type"]} | {r["model"]} {r["dataset"]} {r["key"]} t{r["step"]} | ΔH {r["delta_H"]:.3f} | {BCN[j["behavior"]]}'
 content+='<details><summary>'+html.escape(title)+'</summary><p>'+html.escape(j['note'])+'</p><blockquote>'+html.escape(q['message'])+'</blockquote><pre>'+html.escape(q['before'])+'<mark>'+html.escape(q['token'])+'</mark>'+html.escape(q['after'])+'</pre></details>'
(O/'index.html').write_text(content)
lock=dict(complete=True,received=len(rows),label_sha256=files,request_sha256=D['request_sha256'],report=str(O/'REPORT.md'),summary=str(O/'summary.json'),index=str(O/'index.html'),judge='gpt-6-luna',reasoning='max',agent='/root/pathvqa_luna_continuation')
(B/'annotation_locked.json').write_text(json.dumps(lock,ensure_ascii=False,indent=2));(B/'completion.json').write_text(json.dumps(lock,ensure_ascii=False,indent=2));(B/'state.json').write_text(json.dumps(dict(phase='complete',expected=701,received=len(rows)),indent=2));print(json.dumps(dict(complete=True,received=len(rows),report=str(O/'REPORT.md'),bands=bands,readout=readout),ensure_ascii=False,indent=2))
