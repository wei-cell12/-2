"""One-shot CPU postprocessing AFTER all615blind semantic labels are locked."""
import bisect,collections,csv,hashlib,html,json,random,re,statistics
from pathlib import Path
from validate import validate
B=Path(__file__).resolve().parent
if (B/'completion.json').exists():raise RuntimeError('Already completed; do not rerun or overwrite')
requests,labels,v=validate(require_complete=True);assert len(labels)==615
private=json.loads((B/'private_keymap.json').read_text());assert set(private)==set(labels)
O=B.parent/'shared_prefix_entropy_150x2_20261103/results';D=O/'local_valleys/luna_generalization';D.mkdir(exist_ok=True)
useracts={'user_restatement','user_evaluation'};visual={'visual_observation','visual_inference'}
locked=dict(count=615,request_sha256=v['request_sha256'],label_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((B/'blind').glob('labels_*.jsonl'))},correction_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((B/'blind').glob('corrections_*.jsonl'))})
(B/'annotation_locked.json').write_text(json.dumps(locked,ensure_ascii=False,indent=2));events=[];cache={}
for key,lab in labels.items():
 r=private[key];q=requests[key];ctx=q['before']+q['token']+q['after'];row=dict(blind_key=key,**r,judgment=lab,user_related_at_valley=lab['activity'] in useracts,visual_to_user_transition=lab['activity_before'] in visual and lab['activity'] in useracts,explicit_uncertainty_then_user=lab['explicit_uncertainty_before']=='yes' and lab['activity'] in useracts,entry_alignment=None)
 en=lab['user_entry_quote']
 if en:
  matches=[m.start() for m in re.finditer(re.escape(en),ctx)]
  if len(matches)==1:
   mk=(r['model'],r['key'])
   if mk not in cache:
    rs=list(csv.DictReader((O/r['model']/'records'/(r['key']+'.csv')).open()));offsets=[0]
    for x in rs:offsets.append(offsets[-1]+len(x['token_piece']))
    cache[mk]=(rs,offsets)
   _,offsets=cache[mk];base=offsets[r['context_first_output_step']-1];startchar=base+matches[0];endchar=startchar+len(en);start=bisect.bisect_right(offsets,startchar)-1+1;end=bisect.bisect_left(offsets,endchar)
   row['entry_alignment']=dict(quote=en,start_output_step=start,end_output_step=end,valley_minus_quote_start=r['step']-start,relative='before' if r['step']<start else 'inside_quote' if r['step']<=end else 'after')
  else:row['entry_alignment']=dict(ambiguous_occurrences=len(matches),quote=en)
 events.append(row)
allcases=[]
for model in ['qwen38','gemma4']:allcases.extend(json.loads((O/model/'analysis/cases.json').read_text()))
rng=random.Random(20261103)
def ci(rows,clusters):
 by=collections.defaultdict(lambda:[0,0])
 for e in rows:
  a=by[e['dataset']+'|'+e['image_sha256']];a[0]+=int(e['user_related_at_valley']);a[1]+=1
 ids=sorted(clusters);res=[]
 for _ in range(2000):
  sample=rng.choices(ids,k=len(ids));num=sum(by[k][0] for k in sample);den=sum(by[k][1] for k in sample)
  if den:res.append(num/den)
 if not res:return None
 res.sort();return [res[int(.025*(len(res)-1))],res[int(.975*(len(res)-1))]]
def summarize(rows,cases):
 counts=collections.Counter(e['judgment']['activity'] for e in rows);n=len(rows);clusters={c['dataset']+'|'+c['image_sha256'] for c in cases};deeptrajectories={(e['model'],e['key']) for e in rows};usertraj={(e['model'],e['key']) for e in rows if e['user_related_at_valley']}
 return dict(valleys=n,activity_counts=dict(counts),user_related=sum(e['user_related_at_valley'] for e in rows),user_related_fraction=sum(e['user_related_at_valley'] for e in rows)/n if n else None,user_related_image_cluster_bootstrap95=ci(rows,clusters) if n else None,visual_to_user=sum(e['visual_to_user_transition'] for e in rows),explicit_uncertainty_before=sum(e['judgment']['explicit_uncertainty_before']=='yes' for e in rows),verification_before=sum(e['judgment']['verification_before']=='yes' for e in rows),explicit_uncertainty_then_user=sum(e['explicit_uncertainty_then_user'] for e in rows),user_stance_counts=dict(collections.Counter(e['judgment']['user_stance'] for e in rows)),quote_position_counts=dict(collections.Counter(e['entry_alignment']['relative'] for e in rows if e['entry_alignment'] and 'relative' in e['entry_alignment'])),quote_position_unavailable=sum(not e['entry_alignment'] or 'relative' not in e['entry_alignment'] for e in rows),source_images=len(clusters),source_trajectories=len(cases),source_trajectories_with_postopening_thinking=sum(c['metrics']['after_opening']['n']>0 for c in cases),trajectories_with_deep_valley=len(deeptrajectories),trajectories_with_user_related_valley=len(usertraj))
summary=dict(design='Detect raw deep valleys first, then615blind Luna semantic annotations. All600trajectories scanned. No event-first selection.',overall=summarize(events,allcases),by_model_dataset=[],by_condition=[],by_cohort=[])
for model in ['qwen38','gemma4']:
 for ds in ['ClockQA','PathVQA','PENDULUM']:
  subset=[e for e in events if e['model']==model and e['dataset']==ds];cc=[c for c in allcases if c['model']==model and c['dataset']==ds];summary['by_model_dataset'].append(dict(model=model,dataset=ds,**summarize(subset,cc)))
  for cond in ['belief','belief_right']:summary['by_condition'].append(dict(model=model,dataset=ds,condition=cond,**summarize([e for e in subset if e['condition']==cond],[c for c in cc if c['condition']==cond])))
  for g in ['flip','hold','neutral_wrong','neutral_unresolved','pressure_unresolved']:summary['by_cohort'].append(dict(model=model,dataset=ds,group=g,**summarize([e for e in subset if e['group']==g],[c for c in cc if c['group']==g])))
(D/'events.json').write_text(json.dumps(events,ensure_ascii=False,indent=2));(D/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));(D/'effective_labels.json').write_text(json.dumps(list(labels.values()),ensure_ascii=False,indent=2));(D/'validation.json').write_text(json.dumps(v,ensure_ascii=False,indent=2));(D/'annotation_locked.json').write_text(json.dumps(locked,ensure_ascii=False,indent=2))
with (D/'events.csv').open('w',newline='') as f:
 cols=['model','dataset','id','condition','group','step','token','delta_H','activity','activity_before','activity_after','mentions_specific_user_claim','user_stance','explicit_uncertainty_before','verification_before','user_related_at_valley','visual_to_user_transition','evidence_quote','uncertainty_quote','user_entry_quote','note'];w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
 for e in events:
  r={k:e.get(k) for k in cols};r.update({k:e['judgment'][k] for k in ['activity','activity_before','activity_after','mentions_specific_user_claim','user_stance','explicit_uncertainty_before','verification_before','evidence_quote','uncertainty_quote','user_entry_quote','note']});w.writerow(r)
actcn={'visual_observation':'读图、识别图像内容','visual_inference':'根据图像推断或检验','user_restatement':'仅复述用户主张','user_evaluation':'比较、采信或反驳用户主张','task_or_output_management':'问题、格式、措辞或回答组织','other':'其他活动','unclear':'上下文不足/无法可靠判断'}
a=summary['overall'];lines=['# Luna：局部深谷在做什么，全量普遍性核查','','## 顺序和覆盖','','按用户最新明确要求，先从全部600条既有轨迹检测原始逐token深谷，再让同一个gpt-6-luna(max)盲读每个谷底token及前后原文。没有先标用户事件来反配低谷，也没有只挑典型例子。615个候选负值段全部判读。模型、数据集、正确答案、条件、原回复对错、熵数值在判读时隐藏。','','深谷定义沿用本轮冻结口径：去掉完整首段且至少64个输出token，普通思考token，取每个最大连续负值段最深token，仅保留ΔH≤−1nats。600条中548条还有可分析思考，360条至少出现一个这样的深谷。615是候选负值段数，不是独立题数，也不是已确定的615次犹豫。','','## 全部615个深谷的语义功能','','|当前位置功能|谷段数|占全部深谷|','|---|---:|---:|']
for k,cn in actcn.items():lines.append(f'|{cn}|{a["activity_counts"].get(k,0)}|{a["activity_counts"].get(k,0)/615:.1%}|')
lines+=['',f'“仅复述用户”与“评价用户”合计{a["user_related"]}/615（{a["user_related_fraction"]:.1%}）；复述不等于权衡，更不等于不确定。谷前窗口有明确犹豫/冲突候选文字的{a["explicit_uncertainty_before"]}处，复核证据的{a["verification_before"]}处；两者分开记录。按相邻分句标签推导的“视觉观察/推断→用户主张”转向有{a["visual_to_user"]}处。谷前明确犹豫且谷底涉及用户的有{a["explicit_uncertainty_then_user"]}处，不能把其余谷底一律叫不确定。','', '## 是否跨模型、跨数据集成立','','|模型|数据集|深谷数|仅复述用户|评价用户|两者合计比例|图像聚类bootstrap95%区间|谷前明确犹豫|','|---|---|---:|---:|---:|---:|---|---:|']
for r in summary['by_model_dataset']:
 cc=r['activity_counts'];ci95=r['user_related_image_cluster_bootstrap95'];interval='—' if not ci95 else f'{ci95[0]:.1%}–{ci95[1]:.1%}'
 lines.append(f'|{r["model"]}|{r["dataset"]}|{r["valleys"]}|{cc.get("user_restatement",0)}|{cc.get("user_evaluation",0)}|{r["user_related_fraction"]:.1%}|{interval}|{r["explicit_uncertainty_before"]}|')
lines+=['','区间按原图聚类重采样2000次，不把同一图上的问题、两种条件或相邻谷段当独立样本。组间比例差异直接报告，不因为某组不支持预期就换题或换阈值。更细的用户说错/说对、中性正确后变错/守住等拆分在summary.json。','','## 谷底与用户入口的前后关系','','这里的入口是Luna复制的原文短语，不是注意力变化起点。仅在引文在窗口内唯一时映射到token；缺失/多次出现均保留、单列。谷底相对入口引文的前/句中/后计数：'+json.dumps(a['quote_position_counts'],ensure_ascii=False)+f'；不可定位{a["quote_position_unavailable"]}处。没有把这个事后谷底当作已验证的在线提前触发。','','## 能回答什么、不能冒称什么','','本批直接检验的是“深谷token到底在做什么、其中多大比例涉及重新参考/权衡用户、是否跨模型数据集重复”。没有先标完所有回看用户的时刻，因此不能据此算全部用户事件的召回率/漏检率，也未标注普通无谷位置，不能冒称已测得相对普通位置的富集倍数。600条所有轨迹扫描覆盖已保留，没有谷或没有剩余思考的仍保留。','','文字明确犹豫只是一种可观察证据，不能自动等于内部心理不确定；评价用户也包括正确反驳。两例旧注意力对齐不代表全部615谷段都测了用户段注意力。全部引文和标签可逐条审查；unclear保留，不按格式判错或丢掉。','','## 审核与文件','','events.csv/json含全部来源、深度、token位置、逐字证据及语义标签；effective_labels.json为盲审有效标签；annotation_locked.json记录请求、原标签和独立更正文件的哈希；validation.json给出615条覆盖与引文校验。原始labels和更正文件保留在盲审工作目录。']
(D/'REPORT.md').write_text('\n'.join(lines)+'\n')
e=html.escape
body=['''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Luna全量局部深谷判读</title><style>body{max-width:1250px;margin:26px auto;padding:0 18px;font:17px/1.65 system-ui;color:#243348}table{border-collapse:collapse;width:100%;font-size:15px}td,th{padding:8px;border-bottom:1px solid #dce4ed;text-align:left}pre{white-space:pre-wrap;background:#f4f6fa;padding:12px}a{color:#155f9e}details{border:1px solid #dce4ed;padding:12px;margin:12px 0}.note{background:#fff2d0;padding:14px}</style><h1>Luna：615个局部深谷在做什么</h1><p>先检测深谷，再盲审token功能；覆盖全部600条轨迹扫描产生的615个候选深谷。没有先找用户语句来反配。</p><p><a href="REPORT.md">完整普遍性报告</a> · <a href="events.csv">全部判读CSV</a> · <a href="summary.json">分模型/数据集/条件/结果统计</a> · <a href="../index.html">原始局部曲线</a></p>''']
body.append(f'<p class=note>615个深谷中，仅复述或评价用户的共{a["user_related"]}个（{a["user_related_fraction"]:.1%}）；视觉→用户转向{a["visual_to_user"]}个；谷前有明确犹豫且谷底涉及用户{a["explicit_uncertainty_then_user"]}个。复述、权衡、明确犹豫分开统计，不以出现user一词自动归类。</p>')
body.append('<table><tr><th>模型</th><th>数据集</th><th>深谷数</th><th>仅复述</th><th>评价用户</th><th>合计比例</th><th>谷前明确犹豫</th></tr>')
for r in summary['by_model_dataset']:body.append(f'<tr><td>{r["model"]}</td><td>{r["dataset"]}</td><td>{r["valleys"]}</td><td>{r["activity_counts"].get("user_restatement",0)}</td><td>{r["activity_counts"].get("user_evaluation",0)}</td><td>{r["user_related_fraction"]:.1%}</td><td>{r["explicit_uncertainty_before"]}</td></tr>')
body.append('</table><h2>全部原文证据</h2><p>不是只展示典型案例；下方包含615个谷段。</p>')
for r in sorted(events,key=lambda r:(r['model'],r['dataset'],r['id'],r['condition'],r['step'])):
 q=requests[r['blind_key']];lab=r['judgment'];body.append('<details><summary>'+e(f'{r["model"]} {r["id"]} {r["condition"]} 第{r["step"]}token ΔH={r["delta_H"]:.4f} | {actcn[lab["activity"]]}')+'</summary><pre>'+e(q['before'])+'【'+e(q['token'])+'】'+e(q['after'])+'</pre><p>证据：'+e(lab['evidence_quote'])+'</p><p>'+e(lab['note'])+'</p><p>谷前明确犹豫：'+lab['explicit_uncertainty_before']+'；复核证据：'+lab['verification_before']+'；对用户态度：'+lab['user_stance']+'</p></details>')
body.append('</html>');(D/'index.html').write_text('\n'.join(body))
for p in [O/'index.html',O/'local_valleys/index.html']:
 t=p.read_text();notice='<p id="luna-generalization-ready"><strong>Luna已完成全部615处判读：</strong><a href="'+('local_valleys/luna_generalization/index.html' if p==O/'index.html' else 'luna_generalization/index.html')+'">查看跨模型、跨数据集的全量普遍性结果</a></p>'
 if 'id="luna-generalization-ready"' not in t:p.write_text(t.replace('</h1>','</h1>'+notice,1))
completion=dict(complete=True,trajectories_scanned=600,candidate_valleys=615,semantic_labels=615,validated=True,agent_name='/root/pathvqa_luna_continuation',model_requested='gpt-6-luna',reasoning_effort_requested='max',instructions_sha256=hashlib.sha256((B/'blind/instructions.md').read_bytes()).hexdigest(),corrections=v['corrections'],report=str((D/'index.html').resolve()),summary=summary)
(B/'completion.json').write_text(json.dumps(completion,ensure_ascii=False,indent=2));(B/'batch_state.json').write_text(json.dumps(dict(phase='complete',received=615,validated=True,report=str((D/'index.html').resolve())),indent=2));print(json.dumps(dict(complete=True,received=615,format_or_quote_errors=0,report=str((D/'index.html').resolve()),overall=a),ensure_ascii=False))
