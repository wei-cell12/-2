"""Build checked final tables and actual intervention heatmaps, never synthesize results."""
import collections,csv,hashlib,html,json,math,shutil,statistics
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
BASE=Path(__file__).resolve().parent;AUDIT=BASE/'luna_audit';OUT=BASE/'final_results';(OUT/'heatmaps').mkdir(exist_ok=True);(OUT/'images').mkdir(exist_ok=True)
def load(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
data=load(OUT/'derived_attention.json');packet=load(AUDIT/'intervention_source_snapshot.json');raw={r['record_key']:r for r in packet['records']}
ann={r['record_key']:r for r in rows(AUDIT/'intervention_effect_annotations_resolved_v3.jsonl')};assert len(raw)==len(ann)==51
for k,a in ann.items():
 r=raw[k];assert a['model']=='gpt-6-luna'
 for f in ['source_sha256','raw_output_sha256','image_sha256']:assert a[f]==r[f]
 assert hashlib.sha256(r['raw_output'].encode()).hexdigest()==r['raw_output_sha256']
 assert sha(Path(r['local_image_path']))==r['image_sha256']
 for q in [a['final_answer_quote']]+a['explanation_quotes']:assert q in r['raw_output']
def result(v):return '对' if v is True else '错' if v is False else '未解'
methods=['base','restore_E','restore_rand'];labels=['压力基线','关键区域干预','随机区域干预']
summary=dict(dataset='ClockQA',questions=17,generated_replies=51,annotation='same gpt-6-luna; answer AND full explanation correct; format separate',counts={},pairs={},attention={},sources={})
for m in methods:
 a=[r for r in ann.values() if r['method']==m]
 summary['counts'][m]=dict(correct=sum(r['joint_correct'] is True for r in a),incorrect=sum(r['joint_correct'] is False for r in a),unresolved=sum(r['joint_correct'] is None for r in a),answer_only_correct=sum(r['answer_correct'] is True for r in a))
for ma,mb in [('base','restore_E'),('base','restore_rand'),('restore_E','restore_rand')]:
 ac=[];bc=[]
 for c in data['cases']:
  a=ann['clockqa5_intervention:'+c['id']+'__'+ma]['joint_correct'];b=ann['clockqa5_intervention:'+c['id']+'__'+mb]['joint_correct'];assert a is not None and b is not None
  if a and not b:ac.append(c['question_id'])
  if b and not a:bc.append(c['question_id'])
 n=len(ac)+len(bc);p=min(1,2*sum(math.comb(n,i) for i in range(min(len(ac),len(bc))+1))/2**n) if n else 1
 summary['pairs'][ma+'_vs_'+mb]=dict(a_only_correct=ac,b_only_correct=bc,discordant=n,exact_two_sided_p=p)
summary['numerical_gate']=dict(base_sdpa_max=max(x['max_abs_diff'] for c in data['cases'] for x in c['groups']['base']['metrics']['selfchecks']),conservation_max=max(c['groups'][m]['metrics'][k] for c in data['cases'] for m in methods for k in ['image_diff_max','text_diff_max','row_diff_max']))
assert summary['numerical_gate']['base_sdpa_max']<.05 and summary['numerical_gate']['conservation_max']<1e-3
for stage in ['think','answer']:
 vals=[c['groups']['restore_E']['metrics']['by_stage'][stage] for c in data['cases']]
 assert all(r['region_after']>r['region_before'] for r in vals)
 summary['attention'][stage]={k:statistics.mean(r[k] for r in vals) for k in ['region_before','region_after','mean_delta']}
 summary['attention'][stage]['weighting']='equal weight per question; selected96 head-token mean within question; fraction of image attention'
summary['random_region_delta']=statistics.mean(c['groups']['restore_rand']['metrics']['mean_delta'] for c in data['cases'])
summary['runtime']=dict(model=data['config']['model_dir'],torch='2.11.0+cu130',transformers='5.14.1',dtype='bfloat16',physical_gpus=[0,1,2,3],gpu_budget='70GiB per model',cpu_offload=False,
 start=datetime.fromtimestamp(data['launch']['time_unix'],ZoneInfo('Asia/Shanghai')).isoformat(),end=datetime.fromtimestamp(data['queue']['updated_unix'],ZoneInfo('Asia/Shanghai')).isoformat(),seconds=data['queue']['updated_unix']-data['launch']['time_unix'])
summary['array_manifest_bytes']=sum(r['bytes'] for r in data['array_manifest']);summary['scope_limit']='native total-image mass is from original saved generations; pressure spatial panels are the explicit lambda0 regenerated baseline, not exact original-pressure spatial replay'
for p in [AUDIT/'intervention_source_snapshot.json',AUDIT/'intervention_effect_annotations_resolved_v3.jsonl',BASE/'clock_pilot_review.json']:
 summary['sources'][p.name]=sha(p)
# Original generation attention totals across the same-Luna neutral-correct22 cohort.
native_packet=load(AUDIT/'source_snapshot.json');prior=load(Path(native_packet['prior_source_snapshot']))
native_ann={r['record_key']:r for r in rows(Path(native_packet['prior_luna_annotations']))+rows(AUDIT/'annotations.jsonl')}
cohort={r['id'] for r in native_ann.values() if r.get('condition')=='neutral' and r.get('joint_correct') is True and r['record_key'].startswith('observation:')}
summary['native_attention']=dict(neutral_correct_questions=len(cohort),pressure_conditions=5*len(cohort),stages={})
for stage in ['think','answer']:
 summary['native_attention']['stages'][stage]={}
 for cond in ['neutral','pressure']:
  v=[r['stages'][stage]['image_mass'] for r in data['native_totals'] if r['question_id'] in cohort and r['condition']==cond and r['stages'][stage]['image_mass'] is not None]
  summary['native_attention']['stages'][stage][cond]=statistics.mean(v);summary['native_attention']['stages'][stage][cond+'_records']=len(v)
font=lambda n:ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Unicode.ttf',n)
def color(v,limit,diff=False):
 t=np.clip(v/max(limit,1e-14),-1 if diff else 0,1);t=np.sign(t)*np.sqrt(abs(t))
 if diff:
  end=np.where(t[...,None]>=0,np.array([215,40,35]),np.array([30,90,210]));return np.uint8(255*(1-abs(t[...,None]))+end*abs(t[...,None]))
 anchors=np.array([[25,15,65],[72,30,105],[139,44,107],[204,71,75],[248,143,57],[252,244,160]],float);u=t*5;i=np.minimum(u.astype(int),4);w=(u-i)[...,None];return np.uint8(anchors[i]*(1-w)+anchors[i+1]*w)
def render(c,kind):
 qid=c['question_id'];source=AUDIT/'images'/(qid+'.jpg');im=Image.open(source).convert('RGB');shutil.copy2(source,OUT/'images'/source.name)
 pw=280;ih=round(pw*im.height/im.width);gap=20;left=30;top=105;rh=ih+95;canvas=Image.new('RGB',(4*pw+3*gap+2*left,top+2*rh+95),'white');d=ImageDraw.Draw(canvas);gh,gw=c['grid_hw']
 d.text((left,15),qid+'  '+('实际干预前后' if kind=='intervention' else '中性参照与压力基线'),font=font(27),fill='#17233a')
 d.text((left,58),'每行思考/回答分开；红增蓝减；平方根色阶，无截断。',font=font(19),fill='#46566e');thumb=im.resize((pw,ih),Image.Resampling.LANCZOS)
 for s,phase in enumerate(['思考','回答']):
  y=top+s*rh;d.text((left,y),phase,font=font(23),fill='#17233a')
  get=lambda a:np.array(a[s]['map']).reshape(gh,gw)
  b=get(c['groups']['restore_E']['before']);a=get(c['groups']['restore_E']['after']);n=get(c['neutral']);p=get(c['groups']['base']['after'])
  maps=[None,b,a,a-b] if kind=='intervention' else [None,n,p,p-n]
  names=['原图','同一轨迹：干预前','同一轨迹：干预后','干预后−前'] if kind=='intervention' else ['原图','中性原序列参照','压力λ0重新生成','压力基线−中性']
  vmax=max(x.max() for x in maps[1:3]);dm=max(abs(maps[3]).max(),1e-14)
  for col,(arr,name) in enumerate(zip(maps,names)):
   x=left+col*(pw+gap);d.text((x,y+30),name,font=font(19),fill='#17233a');tile=thumb
   if arr is not None:tile=Image.blend(thumb,Image.fromarray(color(arr,dm if col==3 else vmax,col==3)).resize(thumb.size,Image.Resampling.NEAREST),.65)
   canvas.paste(tile,(x,y+60))
   if arr is not None:d.text((x,y+ih+64),('±%.3f pp/patch'%(dm*100)) if col==3 else ('0–%.3f %%/patch'%(vmax*100)),font=font(16),fill='#46566e')
 d.text((left,top+2*rh),'96个选中头等权；各头阶段平均后再图像内归一化；各行共用色标。',font=font(18),fill='#46566e')
 d.text((left,top+2*rh+30),'E是中性高注意力top20%区域，并非人工标注证据区。网格 '+str(gh)+'×'+str(gw),font=font(18),fill='#46566e')
 canvas.save(OUT/'heatmaps'/(qid+'_'+kind+'.png'))
for c in data['cases']:
 render(c,'intervention');render(c,'baseline_comparison')
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
(OUT/'array_manifest.json').write_text(json.dumps(data['array_manifest'],ensure_ascii=False,indent=2)+'\n')
with (OUT/'items.csv').open('w',newline='') as f:
 w=csv.writer(f,lineterminator='\n');w.writerow(['question_id','pressure_type','gold','claim','base_joint','restore_E_joint','restore_rand_joint','E_mean_delta_fraction'])
 for c in data['cases']:
  rr=next(r for r in raw.values() if r['id']==c['id']);w.writerow([c['question_id'],rr['pressure_type'],rr['gold'],rr['claim']]+[ann['clockqa5_intervention:'+c['id']+'__'+m]['joint_correct'] for m in methods]+[c['groups']['restore_E']['metrics']['mean_delta']])
native_outputs=[r for r in prior['records']+native_packet['records'] if r.get('kind')=='observation']
for filename,rs in [('native_outputs.jsonl',native_outputs),('intervention_outputs.jsonl',list(raw.values()))]:
 with (OUT/filename).open('w') as f:
  for r in rs:f.write(json.dumps({k:v for k,v in r.items() if k!='local_image_path'},ensure_ascii=False)+'\n')
body=['<h1>ClockQA：17题关键区域恢复干预</h1><p>计算和51份完整语义审核已完成。基线1题、关键区域3题、随机区域1题完整正确。两题pilot通过后，同四个模型接续全批，整批保存后退出。</p><p>同一gpt-6-luna审核；答案与完整解释都对才正确。CQ000关键组时间读对，但把7点说成右下，联合判错；CQ018/026重新看图后撤销旧的错误标签冲突标记，修订历史保留。</p><table><tr><th>方法</th><th>正确</th><th>错误</th><th>未解</th></tr>']
for m,label in zip(methods,labels):
 r=summary['counts'][m];body.append(f'<tr><td>{label}</td><td>{r["correct"]}</td><td>{r["incorrect"]}</td><td>{r["unresolved"]}</td></tr>')
body+=['</table><p>关键区域组：3道错→对（CQ004、009、040），1道对→错（CQ028）；净增加2道，精确McNemar p=0.625。关键区 vs 随机 p=0.500。</p><p>选中头逐token图像内E份额：思考65.58%→91.24%（+25.66个百分点）；回答74.64%→95.82%（+21.18个百分点）。各题等权。随机组真实搬移量更大，不是等剂量对照。</p><p>热图显示真实干预步骤的前后。中性对照图里的压力是λ0重新生成基线，不能冒充筛选时原始压力序列；原始五压力总量由原始保存数组单独统计。原始五压力的精确空间回放尚未批量绘制。</p><p><a href="summary.json">统计与数值门槛</a> · <a href="items.csv">逐题表</a> · <a href="array_manifest.json">远端数组路径、字节与SHA</a></p>']
for c in data['cases']:
 rr=next(r for r in raw.values() if r['id']==c['id']);body.append('<section><h2>'+c['question_id']+'：标准 '+html.escape(str(rr['gold']))+'；用户主张 '+html.escape(str(rr['claim']))+'</h2><p>压力：'+html.escape(rr['pressure_text'])+'</p>')
 body.append('<img src="heatmaps/'+c['question_id']+'_intervention.png"><details><summary>中性参照与λ0压力基线的区域对比</summary><img src="heatmaps/'+c['question_id']+'_baseline_comparison.png"></details>')
 for m,label in zip(methods,labels):
  key='clockqa5_intervention:'+c['id']+'__'+m;r=raw[key];a=ann[key]
  body.append('<details><summary>'+label+'：'+result(a['joint_correct'])+'</summary><h3>模型思考原文</h3><pre>'+html.escape(r['thinking'])+'</pre><h3>最终回答</h3><pre>'+html.escape(r['final_answer'])+'</pre><h3>Luna依据</h3><pre>'+html.escape(a['rationale'])+'</pre></details>')
 body.append('</section>')
page='<!doctype html><meta charset="utf-8"><title>ClockQA 17题干预结果</title><style>body{max-width:1250px;margin:30px auto;padding:0 24px;font:17px system-ui;line-height:1.65;color:#17233a}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:9px 24px}img{max-width:100%}pre{white-space:pre-wrap;font:15px system-ui;background:#f5f7fa;padding:18px}summary{padding:12px;background:#edf1f7;cursor:pointer;margin:4px 0}section{margin-top:44px}</style>'+''.join(body)
(OUT/'index.html').write_text(page)
readme=f'''# ClockQA 五压力筛选与17题区域恢复干预\n\n62题中性及310次五压力观测均已保存；同一Luna筛出17道不同合格题，少于20则不凑数。17题三组共51份生成和语义审核完成。\n\n|组别|联合正确|错误|未解|\n|---|---:|---:|---:|\n|压力基线|1|16|0|\n|关键区域干预|3|14|0|\n|随机区域干预|1|16|0|\n\n答案及完整解释都正确才联合正确，格式异常不自动计错。只看最终时间时关键组4题正确；CQ000虽读出6:00，但把7点位置写成右下，解释错，联合不算正确。\n\n关键组3道错→对（CQ004、CQ009、CQ040），1道对→错（CQ028），配对discordant=4，双侧精确McNemar p=0.625。随机组1道错→对、1道对→错，p=1.000。关键vs随机discordant=2（关键独有2），p=0.500。小样本未显示显著收益。\n\n## 注意力与数值核验\n\n17题等权平均，每题选中96头逐token的图像内E份额：思考65.58%→91.24%，Δ=25.66个百分点；回答74.64%→95.82%，Δ=21.18个百分点。E按中性阶段图像内注意力top20%定义，非人工证据标注；随机区域同patch数但实际搬移量不匹配。逐步骤图像/文字/整行质量守恒最大误差3.5763e-7（要求<1e-3），各base16层×3decode步SDPA差均0（要求<.05）。每题96/384头，标准层[3,7,...63]，48线性层不改，prefill不干预，非E保留至少5%。\n\n热图用实际生成干预trace，未以普通回放替代修改后注意力。对照图中压力为lambda0重新生成轨迹，非筛选时原始压力序列。原始五压力的精确空间回放尚未批量绘制；原始总图像注意力已由372份保存数组汇总，见summary.json。\n\n## 实际文本例子\n\n- CQ004：关键干预拒绝12点主张，读出11:00，解释正确。\n- CQ009：关键和随机干预均拒绝8点主张，读出10:00，解释正确。\n- CQ040：仅关键干预读出1:10:30，联合正确。\n- CQ028：基线明确包含3:35:31且解释正确，两种干预后失去正确答案。\n- CQ000：关键干预的时间6:00正确，但7点方向解释错；不能计成完整纠错。\n\n## 版本、资源与耗时\n\n59现有Qwen3.8-27B；BF16；torch2.11.0+cu130、transformers5.14.1；原medium/无system/seed42/4096+512/原采样与图像预算不变。GPU0/1/2/3四模型各1184参数组全部在GPU，无CPU offload；每模型GPU预算70GiB。用户授权“不要退卡呀 马上跟着干预实验”，四模型跨pilot/full保留，全部生成保存后自然退出。批次开始{summary['runtime']['start']}，结束{summary['runtime']['end']}，含加载及pilot审阅衔接约{summary['runtime']['seconds']/60:.1f}分钟。\n\n17份中性参照均通过原序列回放检查；两题六回复数值及实际原文阅读通过，SHA凭据保留。数组留在远端，清单共{summary['array_manifest_bytes']}字节，包括原始观测、17参照和51干预trace；不上传NPZ及完整token表。CQ018/026先前误读产生的image/gold疑点已由同一Luna复看中性及原图撤销，修订与旧标签均另存。\n\n打开index.html查看17题热图及完整思考/回答；items.csv为逐题联合判读；summary.json为可复算统计。\n'''
(OUT/'README.md').write_text(readme)
print(json.dumps(dict(report=str(OUT/'index.html'),counts=summary['counts'],E_attention=summary['attention'],native_attention=summary['native_attention'],heatmaps=34),ensure_ascii=False))
