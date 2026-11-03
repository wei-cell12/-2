import collections,csv,hashlib,html,json,statistics,time
from pathlib import Path
B=Path(__file__).resolve().parent;O=B/'results';models=['qwen38','gemma4'];dslist=['ClockQA','PathVQA','PENDULUM'];names={'qwen38':'Qwen3.8-27B','gemma4':'Gemma4-31B-it'};cn={'belief':'用户说错','belief_right':'用户说对'};gn={'flip':'中性答对→用户说错后答错','hold':'中性答对→用户说错后仍答对','pressure_unresolved':'中性答对→用户说错后未解','neutral_wrong':'中性原本答错','neutral_unresolved':'中性原本未解'}
S={m:json.loads((O/m/'analysis/summary.json').read_text()) for m in models};C={m:json.loads((O/m/'analysis/cases.json').read_text()) for m in models};allcases=sum(C.values(),[]);assert len(allcases)==600
for ds in dslist:
 a={c['id']:c['image_sha256'] for c in C['qwen38'] if c['dataset']==ds};b={c['id']:c['image_sha256'] for c in C['gemma4'] if c['dataset']==ds};assert a==b and len(a)==50,(ds,len(a),len(b))
def group(m,ds,g,cond,scope):return next(r for r in S[m]['groups'] if (r['dataset'],r['group'],r['condition'],r['scope'])==(ds,g,cond,scope))
def fmt(x):return '—' if x is None else f'{x:.5f}'
rt={m:json.loads((O/m/'release_receipt.json').read_text()) for m in models};total=sum(s['tokens'] for s in S.values());report=['# 150道同题、两模型：共享前缀熵差','',f'ClockQA、PathVQA、PENDULUM各50题，共150道不同题。Qwen3.8-27B和Gemma4-31B-it使用同一批题，每题保留“用户说错／用户说对”两条已有回复，共600条轨迹、{total:,}个输出token。没有生成新回答。Qwen复用14条来源完全一致的旧结果；新增回放586条。','',
'## 算法与开头排除','',
'令k从0开始（显示位置为k+1），比较原模型的 `p_U(·|I,Q,U,y[:k])` 与 `p_0(·|I,Q,y[:k])`。两分支都接收同一条已有受压回复的真实前缀；去U分支不接自己的预测。完整词表原生logits、采样筛选前计算，ΔH=H有U−H去U，单位nats；另存JSD、实际token的log概率差与排名。','',
'开头不作为主要波谷证据。固定口径是去掉前64个输出token；另提供更保守的口径：**去掉完整首段，而且至少去掉前64token**。首段边界是原输出的第一个实质段落，不是自动语义判读。若短回复只有一段，保守口径可能没有剩余思考；原回复仍完整保留，不补题、不算错。两个口径同时报告，不能把它们当作已精确剔除了全部复述。','',
'这是在给定受压历史下，显式用户主张对下一token分布的局部影响。共享历史本身可能已包含用户的说法；后期差值变小不等于压力总影响消失。熵差不直接表示token或视觉判断是否正确。','',
'## 这50题按既有答案判读如何分组','',
'以下每行恰好50道题，分组按“用户说错”那条回复；“用户说对”轨迹不按结果二次筛选。“答对”沿用已有答案语义判读，解释正确性和格式异常保留在原标注，不混为一个数字。','',
'|模型|数据集|中性答对→答错|中性答对→守住|中性答对→未解|中性原本答错|中性原本未解|','|---|---|---:|---:|---:|---:|---:|']
for m in models:
 for ds in dslist:
  g=S[m]['cohorts'][ds]['groups'];report.append('|'+names[m]+'|'+ds+'|'+'|'.join(str(g.get(k,0)) for k in ['flip','hold','pressure_unresolved','neutral_wrong','neutral_unresolved'])+'|')
report+=['','未解沿用已有判读，不因格式异常直接算错。精确的逐题分组、参考答案、原文证据和源SHA在selection.json、labels_snapshot.json及各模型analysis/cases.json。','','## 当前结果的含义','', '去掉开头后，平均熵差没有跨模型、跨数据集一致的分离方向：ClockQA里Qwen变错组更负，Gemma却是守住组更负；Gemma PathVQA的两组均值在前64排除口径下均约−0.02344nats。不能把一个统一的平均熵差阈值直接当成谄媚触发条件。这个结果不等于其他逐token特征没有信息。','', 'PENDULUM的短回复需要特别注意分母：Qwen守住组在固定64口径下有40条有效，保守去首段后有24条；组均值从−0.16603到−0.10077不能当作同样40条轨迹的变化。限制在共同有效24条时，固定64均值为−0.11649，保守首段口径为−0.10077。','', '## 去掉前64token后的思考平均熵差','','每条轨迹先按自己的剩余思考token求平均，再按题等权平均。下表只展示中性答对的两组；其他组也已计算，保存在完整CSV/JSON。两条件使用各自原轨迹，不把同编号token误当作语义对齐。','','|模型|数据集|用户说错：答错组（有效条数）|用户说错：守住组（有效条数）|用户说对：原答错组同题|用户说对：原守住组同题|','|---|---|---:|---:|---:|---:|']
for m in models:
 for ds in dslist:
  rr=[group(m,ds,g,cond,'after64') for g,cond in [('flip','belief'),('hold','belief'),('flip','belief_right'),('hold','belief_right')]];report.append('|'+names[m]+'|'+ds+'|'+'|'.join(fmt(r['means']['delta_H'])+f'（{r["eligible"]}）' for r in rr)+'|')
report+=['','单位nats。平均ΔH为负仅表示该历史下有用户主张时预测更集中，不代表接受了错误信息。不能把“更负”直接转换为谄媚严重程度。','','![固定64token排除后的组间分布](fixed64_sensitivity.png)','','## 保守排除完整首段后的敏感性检查','','|模型|数据集|用户说错：答错组|用户说错：守住组|用户说对：原答错组同题|用户说对：原守住组同题|','|---|---|---:|---:|---:|---:|']
for m in models:
 for ds in dslist:
  rr=[group(m,ds,g,cond,'after_opening') for g,cond in [('flip','belief'),('hold','belief'),('flip','belief_right'),('hold','belief_right')]];report.append('|'+names[m]+'|'+ds+'|'+'|'.join(fmt(r['means']['delta_H'])+f'（{r["eligible"]}/{r["total"]}）' for r in rr)+'|')
report+=['','括号是有剩余普通思考token的轨迹数／该组原轨迹数。无剩余的原回复仍在600条中，没有删除或补齐。两模型的词表、token划分和原图像处理不同，应优先在各模型内部比较，不把跨模型nats大小直接当作同一不确定性刻度。','','![保守开头排除后的逐题分布](cohort_overview.png)','','## 波谷位置怎么查','',
'网页数值为显示舍入，完整精度以CSV为准。每条已保存两个口径下的最小单tokenΔH位置，以及完整连续16token平均ΔH最小的窗口，均限普通思考token，排除特殊token及原预算强制token。目录的逐token页面可点曲线或拖动滑块查看准确位置和周围原文。所有原值及完整精度CSV也保留。','',
'本次没有冒称已经对600条波谷做了语义标注。valley_contexts.json列出了原文上下文，semantic_category保持空。不能把此前12题中“多数最深谷涉及用户”的人工阅读计数直接套到这150题上。','',
'## 选题与可比性','',
'- 两模型各数据集的50个ID及原图SHA逐一相等。SHA256固定排序、seed42，不根据这次熵差或答案对错挑选。',
'- ClockQA有62题，排除错误时间格式无效的CQ-005、CQ-012、CQ-047后，从其余59题固定选50题。',
'- PathVQA从已有100题固定选50题，对应19张去重病理图；PENDULUM的50题对应20张去重照片。两者不能把50道题当作50个独立图像样本。',
'- PENDULUM只取re_used和self_captured两类真实照片，各25题；排除已有参考答案争议标记，并使用Qwen在59同一模型路径下生成的源记录。不是其余两类图片的混合样本。',
'- 既有判读全部复用，未启动新判读者、未新生成回答。答案未解、解释未能验证、输出格式问题分别保留。','',
'|数据集|题数|去重原图数|','|---|---:|---:|']
for ds in dslist:report.append(f'|{ds}|50|{S["qwen38"]["cohorts"][ds]["unique_images"]}|')
report+=['','## 执行和数值验证','','|模型|完成轨迹|复用|整段回放通过|原生逐token缓存回放|最小原token前20比例|最大前缀JSD|最大前缀熵绝对差|','|---|---:|---:|---:|---:|---:|---:|---:|']
for m in models:
 s=S[m];methods=s['methods'];report.append('|'+names[m]+'|300|'+str(s['reused'])+'|'+str(methods.get('parallel_full_sequence',0))+'|'+str(methods.get('native_single_token_cached',0))+'|'+f'{s["min_actual_top20"]:.6f}|{s["max_prefix_JSD"]:.6g}|{s["max_prefix_entropy_difference"]:.6g}|')
report+=['','- 重建两种提示的input_ids逐条与原记录一致；同题两分支图像张量相同；源记录、图像、原输出ID哈希在清单中保存，未改原记录。',
'- 独立NumPy float64核对FP32熵/JSD公式；同提示控制ΔH=0。前缀步号1、2、16各分支核对，门槛始终JSD<0.001、熵绝对差<0.1nats。BF16不声称bit一致或top1必然一致。',
'- 6项原CPU熵/因果测试，加2项真实调用缓存辅助函数的CPU测试，共8项通过，验证缓存喂入保存的实际历史、未来token不影响早先预测。',
'- 部分整段回放因BF16批处理与短前缀数值差异超门槛，被拒绝。受影响轨迹的两个完整分支改为原生逐token缓存回放，再以独立短前缀缓存运行核对。未采用缓存16/32/64块来绕过问题，未放宽原门槛。原失败、诊断、被拒绝检查及最终计算方法逐条保存。',
'- 准备阶段旧结果文件名接口和本地脚本select命名冲突已修复并保留现场；已完成的结果不重复计算，不停止他人进程。',
'- Qwen使用59的GPU0–3，Gemma使用50的0/1、2/3、4/5三组；完成后各模型自然退出，没有保留占卡进程。51仅导出选定原回复与图片，没有启动新GPU任务；不传输模型权重。',
'- 实际回放环境两机均torch2.11.0+cu130、transformers5.14.1。Gemma历史51生成环境记录为transformers5.20.2；本次使用50既有环境，不能声称生成与回放的软件版本完全相同。输入ID重建和原token排名校验结果如上。两款模型各自的原精度、图像处理、模板、预算和采样配置均保留；回放不采样。',
'- Gemma仅把模型路径定位到50已有副本，并在device_map=auto加载时给每张可见卡34GiB权重分配上限，留前向余量；没有改变模型权重、BF16或实验公式。','',
'## 统计边界与文件','',
'analysis/summary.json给出按题bootstrap的探索性区间；PathVQA和PENDULUM都复用图像，使题目不独立，这些区间不能当作严格图像独立显著性结论。Gemma PathVQA、两模型PENDULUM的“变错”组较小，不能强行拟合普适触发阈值。','',
'每模型目录包含records/*.csv/json、analysis/cases.json、metrics.csv、source_texts.json、valley_contexts.json、summary.json、配置、源码和完整日志。顶层trajectory_registry.json索引600条轨迹；trajectories/提供逐token页面。原始大词表logits未保存，原始整数token列表和图像留在服务器源目录，不上传NPZ。']
(O/'REPORT.md').write_text('\n'.join(report)+'\n')
# Compact readable landing page plus selectable full trajectory viewer.
e=html.escape
head='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>150题两模型共享前缀熵差</title><style>body{max-width:1350px;margin:26px auto;padding:0 18px;font:17px/1.6 system-ui;background:#f5f7fa;color:#233248}section{background:white;padding:22px;border:1px solid #dbe3ed;border-radius:10px;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:15px}td,th{padding:9px;border-bottom:1px solid #dce4ed;text-align:left}img{width:100%;height:auto}select{font:inherit;max-width:100%;margin:8px;padding:7px}iframe{width:100%;height:1350px;border:1px solid #d7e0eb;background:white}a{color:#155f9d}.note{background:#fff1d0;padding:14px}</style><h1>150道同题，两模型，共600条熵差轨迹</h1><p>ClockQA / PathVQA / PENDULUM各50题 · Qwen3.8-27B与Gemma4-31B-it · 使用已有回复，没有新生成</p><p><a href="REPORT.md">完整方法、结果与执行报告</a> · <a href="trajectory_registry.json">600条索引</a> · <a href="selection.json">固定同题清单</a> · <a href="qwen38/analysis/metrics.csv">Qwen统计CSV</a> · <a href="gemma4/analysis/metrics.csv">Gemma统计CSV</a></p><p class="note">开头单独处理：固定去前64token，并额外给出去完整首段且至少64token的保守口径。正确率分组复用原判读；本次未自动把600条波谷标成“用户复述”。</p>'''
body=[head,'<section><h2>每行50道题，按用户说错时的结果分组</h2><table><tr><th>模型</th><th>数据集</th><th>中性对→错</th><th>中性对→仍对</th><th>中性对→未解</th><th>中性原本错</th><th>中性原本未解</th></tr>']
for m in models:
 for ds in dslist:
  g=S[m]['cohorts'][ds]['groups'];body.append('<tr><td>'+names[m]+'</td><td>'+ds+'</td>'+''.join('<td>'+str(g.get(k,0))+'</td>' for k in ['flip','hold','pressure_unresolved','neutral_wrong','neutral_unresolved'])+'</tr>')
body+=['</table></section>','<section><h2>保守排除开头后的逐题熵差</h2><img src="cohort_overview.png"><p>红为用户说错，蓝为用户说对。每个点是一题的一条轨迹，横条为等权平均；n是该条件有剩余思考的轨迹数。F与H不是均匀抽取的两组，人数由原判读决定。</p><details><summary>固定去前64token的敏感性图</summary><img src="fixed64_sensitivity.png"></details></section>','<section><h2>逐token查看：位置、曲线、思考与回答</h2><select id="model"><option value="qwen38">Qwen3.8-27B</option><option value="gemma4">Gemma4-31B-it</option></select><select id="dataset">'+''.join('<option>'+ds+'</option>' for ds in dslist)+'</select><select id="condition"><option value="belief">用户说错</option><option value="belief_right">用户说对</option></select><select id="group"><option value="all">全部分组</option>'+''.join('<option value="'+g+'">'+gn[g]+'</option>' for g in gn)+'</select><br><select id="case"></select><p id="description"></p><p><a id="direct" href="">单独打开这条完整轨迹</a></p><iframe id="viewer" title="逐token熵差与原文"></iframe></section>']
registry=json.loads((O/'trajectory_registry.json').read_text());payload=json.dumps(registry,ensure_ascii=False).replace('<','\\u003c');body.append('<script>const registry='+payload+''';let current=[];function choose(){let idx=Number(document.getElementById('case').value),r=current[idx];if(!r){document.getElementById('description').textContent='这一组合没有样本';return}document.getElementById('viewer').src=r.page;document.getElementById('direct').href=r.page;document.getElementById('description').textContent=r.id+' | '+r.group+' | 共'+r.tokens+'个输出token；前64token后剩余思考'+r.after64.n+'；排除完整首段且至少64token后剩余'+r.after_opening.n;}function filter(){let m=document.getElementById('model').value,d=document.getElementById('dataset').value,c=document.getElementById('condition').value,g=document.getElementById('group').value;current=registry.filter(r=>r.model===m&&r.dataset===d&&r.condition===c&&(g==='all'||r.group===g));const s=document.getElementById('case');s.replaceChildren();current.forEach((r,i)=>{let o=document.createElement('option');o.value=i;o.textContent=r.id+' · '+r.group;s.appendChild(o)});choose()}for(const id of ['model','dataset','condition','group'])document.getElementById(id).onchange=filter;document.getElementById('case').onchange=choose;filter();</script></html>''');(O/'index.html').write_text('\n'.join(body))
for n in ['selection.json','analysis_plan.json','labels_snapshot.json']:(O/n).write_bytes((B/n).read_bytes())
checks=dict(questions=150,datasets=3,questions_per_dataset=50,models=2,trajectories=600,output_tokens=total,same_question_ids_and_image_hashes=True,all_source_pids_exited=all(r['owned_processes_exited'] for r in rt.values()),model_checks={m:{k:S[m][k] for k in ['methods','min_actual_top20','max_prefix_JSD','max_prefix_entropy_difference','max_formula_error']} for m in models})
(O/'validation_receipt.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2));print('Built report, validated same150image hashes,600trajectories, token rows',total)
