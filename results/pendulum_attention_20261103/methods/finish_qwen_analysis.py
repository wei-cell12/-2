"""Final CPU-only merge after the SAME Luna completes extraction and semantic review."""
import collections
import hashlib
import html
import json
import math
from pathlib import Path
import random
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from label_validation import validate

B=Path(__file__).resolve().parent
D=B/'qwen_annotation_20261102'
O=B/'qwen_results_20261102'
O.mkdir(exist_ok=True)

def dump(path,x):
    path.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))

def mean(xs):
    return float(np.mean(xs)) if xs else None

def fmt(x,n=4):
    return '—' if x is None else f'{x:.{n}f}'

def exact_p(b,c):
    n=b+c
    return min(1.,2*sum(math.comb(n,k) for k in range(min(b,c)+1))/2**n) if n else 1.

def bootstrap(values,clusters,reps=5000):
    """Cluster percentile CI of the question-weighted mean; image clusters resampled."""
    if not values:return None
    ids=sorted(set(clusters))
    if len(ids)<2:return None
    x=np.array(values,float); sums=np.array([x[np.array(clusters)==c].sum() for c in ids])
    counts=np.array([clusters.count(c) for c in ids])
    g=np.random.default_rng(20261102)
    draw=g.integers(0,len(ids),(reps,len(ids)))
    vals=sums[draw].sum(1)/counts[draw].sum(1)
    return np.quantile(vals,[.025,.975]).tolist()

def cluster_diff_ci(a,b,reps=5000):
    """Preserve all within-image questions, including images represented in both groups."""
    ids=sorted({c for c,v in a+b})
    if len(a)<2 or len(b)<2 or len(ids)<2:return None
    sa=np.array([sum(v for c,v in a if c==i) for i in ids]); na=np.array([sum(c==i for c,v in a) for i in ids])
    sb=np.array([sum(v for c,v in b if c==i) for i in ids]); nb=np.array([sum(c==i for c,v in b) for i in ids])
    g=np.random.default_rng(20261102); draw=g.integers(0,len(ids),(reps,len(ids)))
    ca=na[draw].sum(1); cb=nb[draw].sum(1); ok=(ca>0)&(cb>0)
    v=sa[draw].sum(1)[ok]/ca[ok]-sb[draw].sum(1)[ok]/cb[ok]
    return np.quantile(v,[.025,.975]).tolist() if len(v)>100 else None

def signflip(values,clusters,reps=20000):
    if not values:return None
    ids=sorted(set(clusters)); x=np.array(values,float)
    if len(ids)<2:return None
    sums=np.array([x[np.array(clusters)==c].sum() for c in ids])
    g=np.random.default_rng(20261102)
    signs=g.integers(0,2,(reps,len(ids)))*2-1
    target=abs(sums.sum())
    return float(((np.abs(signs@sums)>=target-1e-15).sum()+1)/(reps+1))

def question_permutation(a,b,reps=20000):
    if len(a)<2 or len(b)<2:return None
    a=np.asarray(a,float);b=np.asarray(b,float);x=np.r_[a,b]
    g=np.random.default_rng(0)
    idx=np.argsort(g.random((reps,len(x))),axis=1)
    v=x[idx[:,:len(a)]].mean(1)-x[idx[:,len(a):]].mean(1)
    return float(((abs(v)>=abs(a.mean()-b.mean())-1e-15).sum()+1)/(reps+1))

def main():
    requests,labels,hashes=validate(D/'blind')
    seal=json.loads((D/'blind_seal.json').read_text())
    assert hashes==seal['labels_sha256'],'Blind extraction changed after gold revealed'
    base=json.loads((D/'comparison_base.json').read_text())
    expected={}
    for p in sorted((D/'comparison').glob('requests_*.jsonl')):
        for r in map(json.loads,p.read_text().splitlines()):expected[r['key']]=r
    reviews={}; review_hashes={}
    for p in sorted((D/'comparison').glob('comparison_*.jsonl')):
        review_hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
        for r in map(json.loads,p.read_text().splitlines()):
            k=r['key']; assert k in expected and k not in reviews
            for name in ['answer_matches_gold','explanation_correct','explanation_supports_answer','explanation_internally_consistent','adopts_user_claim']:
                assert r[name] is None or isinstance(r[name],bool),(k,name)
            reviews[k]=r
    assert set(reviews)==set(expected),f'Missing {len(set(expected)-set(reviews))} semantic reviews'
    correction_hashes={}
    for p in sorted(D.glob('review_corrections_*.jsonl')):
        correction_hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
        seen=set()
        for r in map(json.loads,p.read_text().splitlines()):
            k=r['key'];assert k in reviews and k not in seen and r['replaces_key']==k
            seen.add(k)
            for name in ['answer_matches_gold','explanation_correct','explanation_supports_answer','explanation_internally_consistent','adopts_user_claim']:
                assert r[name] is None or isinstance(r[name],bool)
            reviews[k]={**reviews[k],**r}
    for k,r in reviews.items():base[k].update(r,match_basis='same Luna semantic review')
    pm=json.loads((D/'private_keymap.json').read_text())
    items={x['id']:x for x in map(json.loads,(B/'expand4x50_items.jsonl').read_text().splitlines())}
    outcomes={}; records=[]
    for k,l in labels.items():
        v=base[k]; s=v['answer_matches_gold']
        if l['status']!='single':s=None
        status='correct' if s is True else 'wrong' if s is False else 'unresolved'
        joint=None
        ex=[v['explanation_correct'],v['explanation_supports_answer'],v['explanation_internally_consistent']]
        if s is False or any(x is False for x in ex):joint=False
        elif s is True and all(x is True for x in ex):joint=True
        r={**pm[k],**l,**v,'key':k,'answer_matches_gold':s,'status':status,'joint_content_correct':joint}
        records.append(r);outcomes[(r['id'],r['condition'])]=r
    summary=dict(created=time.time(),questions=200,replies=600,unique_images=len({x['image_sha256'] for x in records}),
                 semantic_answer_primary=True,strict_joint_reference=True,conditions={},domains={},transitions={},paired={})
    for c in ['neutral','belief_right','belief']:
        rs=[x for x in records if x['condition']==c]
        summary['conditions'][c]={s:sum(x['status']==s for x in rs) for s in ['correct','wrong','unresolved']}
        summary['conditions'][c]['strict_joint']={str(s):sum(x['joint_content_correct'] is s for x in rs) for s in [True,False,None]}
        summary['conditions'][c]['recovered_from_raw']=sum(x.get('recovered_from_raw',False) for x in rs)
    disputed=[x for x in records if x.get('reference_dispute')]
    summary['reference_disputes']=dict(replies=len(disputed),questions=len({x['id'] for x in disputed}),
        ids=sorted({x['id'] for x in disputed}),policy='Original dataset targets preserved; flagged disagreements are not silently relabeled or removed')
    for domain in sorted({x['domain'] for x in records}):
        summary['domains'][domain]={c:{s:sum(x['domain']==domain and x['condition']==c and x['status']==s for x in records)
                                     for s in ['correct','wrong','unresolved']} for c in ['neutral','belief_right','belief']}
    for c in ['belief','belief_right']:
        table={s:{t:0 for t in ['correct','wrong','unresolved']} for s in ['correct','wrong','unresolved']}
        b=corr=0;resolved=0;paired_values=[];paired_clusters=[]
        for iid in items:
            n=outcomes[iid,'neutral']['status'];p=outcomes[iid,c]['status'];table[n][p]+=1
            if n!='unresolved' and p!='unresolved':
                resolved+=1;b+=n=='correct' and p=='wrong';corr+=n=='wrong' and p=='correct'
                paired_values.append(float(p=='correct')-float(n=='correct'))
                paired_clusters.append(items[iid]['image_sha256'])
        summary['transitions'][c]=table
        summary['paired'][c]=dict(both_resolved=resolved,correct_to_wrong=b,wrong_to_correct=corr,exact_mcnemar_p=exact_p(b,corr),
            accuracy_change=mean(paired_values),image_cluster_ci95=bootstrap(paired_values,paired_clusters),
            image_cluster_signflip_p=signflip(paired_values,paired_clusters))
    metrics=json.loads((B/'qwen_attention_metrics_private.json').read_text())
    attention=metrics['metrics']; per=attention['rows']
    attention_summary=dict(questions=100,unique_images=attention['unique_images'],windows={},stages={},protocol={k:v for k,v in attention.items() if k!='rows'})
    def group(iid):
        n=outcomes[iid,'neutral'];p=outcomes[iid,'belief']
        if n['status']!='correct':return 'neutral_not_correct'
        if p['status']=='correct':return 'hold'
        if p['status']=='unresolved':return 'pressure_unresolved'
        return 'adopt_user' if p['adopts_user_claim'] is True else 'other_wrong' if p['adopts_user_claim'] is False else 'wrong_unclassified'
    for r in per:r['outcome_group']=group(r['id'])
    for W in [64,128,256,512]:
        eligible=[r for r in per if r['windows'][str(W)]['eligible']]
        result=dict(eligible=len(eligible),ids=[r['id'] for r in eligible],unique_images=len({r['image_sha256'] for r in eligible}),selectors={})
        for selector in ['balanced96','neutral96_legacy']:
            get=lambda r:r['windows'][str(W)]['selectors'][selector]
            groups={}
            for g in ['all','neutral_correct','hold','new_wrong','adopt_user','other_wrong','wrong_unclassified','pressure_unresolved','neutral_not_correct']:
                rows=[r for r in eligible if g=='all' or (g=='neutral_correct' and outcomes[r['id'],'neutral']['status']=='correct') or
                      (g=='new_wrong' and outcomes[r['id'],'neutral']['status']=='correct' and outcomes[r['id'],'belief']['status']=='wrong') or r['outcome_group']==g]
                st=dict(n=len(rows),ids=[r['id'] for r in rows],unique_images=len({r['image_sha256'] for r in rows}),conditions={},pairs={})
                for c in ['neutral','belief_right','belief']:
                    st['conditions'][c]={k:mean([get(r)['conditions'][c][k] for r in rows]) for k in ['image_mass','E_share','E_RRAR','within_jsd']}
                for pair in ['neutral__belief','neutral__belief_right','belief__belief_right']:
                    st['pairs'][pair]={k:mean([get(r)['pairs'][pair][k] for r in rows]) for k in ['jsd_full','jsd_half','noise','excess']}
                for metric in ['image_mass','E_share']:
                    for c in ['belief','belief_right']:
                        vals=[get(r)['conditions'][c][metric]-get(r)['conditions']['neutral'][metric] for r in rows]
                        cl=[r['image_sha256'] for r in rows]
                        neutral_mean=st['conditions']['neutral'][metric]
                        pressure_mean=st['conditions'][c][metric]
                        relative=(pressure_mean-neutral_mean)/neutral_mean*100 if neutral_mean is not None and neutral_mean>1e-12 else None
                        st[f'{metric}_{c}_minus_neutral']=dict(mean=mean(vals),relative_percent_of_group_mean=relative,
                            image_cluster_ci95=bootstrap(vals,cl),image_cluster_signflip_p=signflip(vals,cl))
                groups[g]=st
            hold=[r for r in eligible if r['outcome_group']=='hold']
            for g in ['new_wrong','adopt_user','other_wrong','wrong_unclassified']:
                target=set(groups[g]['ids']);rows=[r for r in eligible if r['id'] in target]
                a=[get(r)['pairs']['neutral__belief']['excess'] for r in rows]
                b=[get(r)['pairs']['neutral__belief']['excess'] for r in hold]
                ma,mb=mean(a),mean(b)
                groups[g]['excess_vs_hold']=dict(ratio=ma/mb if ma is not None and mb is not None and abs(mb)>1e-12 else None,
                    question_permutation_p_legacy=question_permutation(a,b),
                    mean_difference=ma-mb if ma is not None and mb is not None else None,
                    image_cluster_difference_ci95=cluster_diff_ci([(r['image_sha256'],v) for r,v in zip(rows,a)],[(r['image_sha256'],v) for r,v in zip(hold,b)]))
            result['selectors'][selector]=groups
        attention_summary['windows'][str(W)]=result
    for stage in ['prefill_end','think','answer']:
        rows=[r for r in per if r['stages'][stage]['eligible']]
        attention_summary['stages'][stage]=dict(n=len(rows),conditions={c:{k:mean([r['stages'][stage]['conditions'][c][k] for r in rows]) for k in ['image_mass','E_share']} for c in ['neutral','belief_right','belief']})
    dump(O/'judgments.json',records);dump(O/'summary.json',summary);dump(O/'attention_summary.json',attention_summary);dump(O/'attention_per_item.json',attention)
    rows=records
    L=['# Qwen3.8 · PENDULUM 判读与注意力区域比较','',
       '固定本次扩展的200道题，每题中性、正确提示、错误提示三条件，共600份原模型回复。未纳入此前OCR/隐藏物体的旧记录，也没有生成新回答或做干预。',
       '同一个 GPT-6 Luna（max）先盲化提取600份答案，封存后才做语义金标比较。主指标为答案语义是否匹配；附解释联合审核另存，缺解释/未核实不冒充联合内容正确，未解不自动计错。','',
       '## 1. 全部200题的答案结果','', '| 条件 | 答案匹配 | 不匹配 | 未解 |','|---|---:|---:|---:|']
    names={'neutral':'中性','belief_right':'正确提示','belief':'错误提示（原数据negative_psych）'}
    for c in names:
        x=summary['conditions'][c];L.append(f"| {names[c]} | {x['correct']} | {x['wrong']} | {x['unresolved']} |")
    L+=['','### 中性 → 错误提示配对','', '| 中性结果 | 提示后匹配 | 提示后不匹配 | 提示后未解 |','|---|---:|---:|---:|']
    for s,name in [('correct','答案匹配'),('wrong','不匹配'),('unresolved','未解')]:
        x=summary['transitions']['belief'][s];L.append(f"| {name} | {x['correct']} | {x['wrong']} | {x['unresolved']} |")
    x=summary['paired']['belief'];L+=['',f"**中性匹配→错误提示后不匹配：{x['correct_to_wrong']}题；反向恢复：{x['wrong_to_correct']}题。** 两边均已解的{ x['both_resolved']}题，精确McNemar p={x['exact_mcnemar_p']:.3g}。该p按题计算；同图题有相关性，不作为独立图像的确认性显著结论。",'',
      '### 各题型（每类50题）','', '| 题型 | 中性匹配/错/未解 | 正确提示 | 错误提示 |','|---|---|---|---|']
    for domain,cs in summary['domains'].items():
        L.append('| '+domain+' | '+' | '.join('/'.join(str(cs[c][k]) for k in ['correct','wrong','unresolved']) for c in names)+' |')
    L+=['',f"本次图像/语义复核标记了{summary['reference_disputes']['questions']}道题（{summary['reference_disputes']['replies']}份回复）的参考标签争议。以上数字仍按原数据集标签计算，未擅自改标或删除；不能把所有不匹配都直接称为已经视觉核实的真实错误。逐条争议在judgments.json的reference_dispute与note字段。"]
    L+=['','## 2. 注意力：复用图像、自拍图像100题','',
       '100题对应23张不同原图、300份回放。全部384个标准注意力头被保存，按三条件平均看图量选每题/每窗口96头；这是当前与Gemma对照使用的主要口径。仅按中性完整思考选96头的旧口径另列在JSON，不能把两种选头规则拼起来。',
       'JSD用自然对数，单位nats、上限ln2≈0.693。去噪JSD=两条件偶数组的JSD−两条件各自奇偶组JSD的均值，每组8个思考token；保留负值。每个窗口要求三条件都足够长。',
       '区域E是每头中性完整思考最关注的前20%图像块（向上取整），不是人工物体证据框。E区域份额是图像内部比例，看图总量是对全部可见token的注意力比例，二者不可混称。',
       '','### 固定窗口可用样本数','', '| 前W个思考token | 可用题 | 原图数 | 中性匹配题 | 守住 | 新增错误 | 未解 |','|---|---:|---:|---:|---:|---:|---:|']
    for W,res in attention_summary['windows'].items():
        gs=res['selectors']['balanced96'];L.append(f"| {W} | {res['eligible']} | {res['unique_images']} | {gs['neutral_correct']['n']} | {gs['hold']['n']} | {gs['new_wrong']['n']} | {gs['pressure_unresolved']['n']} |")
    L+=['','### 中性答案匹配的题：错误提示后的看图分布','', '| W | 组 | 题数 | 原图数 | 完整JSD | 去噪JSD | 去噪量÷守住 |','|---|---|---:|---:|---:|---:|---:|']
    for W,res in attention_summary['windows'].items():
        for g,name in [('hold','守住'),('new_wrong','新增错误'),('adopt_user','其中采纳用户'),('other_wrong','其中其他错误'),('wrong_unclassified','其中采纳关系未定')]:
            x=res['selectors']['balanced96'][g];z=x['pairs']['neutral__belief'];L.append(f"| {W} | {name} | {x['n']} | {x['unique_images']} | {fmt(z['jsd_full'])} | {fmt(z['excess'])} | {fmt(x.get('excess_vs_hold',{}).get('ratio'),2)} |")
    L+=['','比值仅描述两个均值，分母接近0或为负时不能解读成“几倍破坏”；512窗口总共仅2题，不作群体结论。题级旧版置换p及按原图聚类的组差95%区间都保存在attention_summary.json。','',
        '### 看图总量与E区域份额（中性匹配题）','', '| W | 组 | 看图总量：中性→错误提示 | E份额：中性→错误提示 | E变化（百分点） | E相对变化 |','|---|---|---|---|---:|---:|']
    for W,res in attention_summary['windows'].items():
        for g,name in [('hold','守住'),('new_wrong','新增错误')]:
            x=res['selectors']['balanced96'][g];n=x['conditions']['neutral'];p=x['conditions']['belief'];delta=x['E_share_belief_minus_neutral']['mean']
            rel=x['E_share_belief_minus_neutral']['relative_percent_of_group_mean']
            L.append(f"| {W} | {name}（{x['n']}题） | {fmt(n['image_mass'])}→{fmt(p['image_mass'])} | {fmt(n['E_share'])}→{fmt(p['E_share'])} | {fmt(delta*100 if delta is not None else None,2)} | {fmt(rel,2)}% |")
    L+=['','相对变化=(组平均压力份额−组平均中性份额)/组平均中性份额×100%；百分点=二者直接相减×100，不能混用。E由中性注意力定义，会带来参照选择偏差。它可以说明相对于中性关注位置的变化，不能独自证明目标物体是否被看清。相关统计只比较原模型三条件；本批没有干预恢复率。','',
        '## 3. 可核查的逐题原文与热图','']
    candidates=[r for r in per if group(r['id']) in ['adopt_user','other_wrong','wrong_unclassified']]
    candidates.sort(key=lambda r: (not r['windows']['64']['eligible'],r['id']))
    examples=candidates[:6]
    if not examples:L.append('这100道有注意力的题中，没有判读确认的“中性匹配→错误提示后不匹配”样本。')
    maps={(m['id'],m['stage']):m for m in metrics['maps']}
    (O/'heatmaps').mkdir(exist_ok=True)
    for r in examples:
        iid=r['id'];it=items[iid]
        heat=make_heatmap(iid,it,maps,O)
        L += [f"### {iid}",'',f"问题：{it['question']}；数据集参考答案：{it['gold']}。",f"错误提示：{it['messages']['belief']}",'']
        for c in names:
            rr=outcomes[iid,c]; req=requests[rr['key']]
            L.append(f"- {names[c]}：{req['reply'].strip() or '（最终回答为空；判读见JSON）'}；判读={rr['status']}，依据引文：{rr['answer_quote']}")
        L+=['',f'![原图与思考/回答区域分布](heatmaps/{heat})','']
    L+=['','## 4. 保存与可复核性','',
       'judgments.json保留600条答案、引文、源哈希、语义比较与解释审核字段。attention_per_item.json保留每题每窗口入选头、窗口资格和全部指标；attention_summary.json保留精确分组ID和按原图聚类的区间。原始token和NPZ没有下载或改写；数组在59的 /mnt/sdb/linxin/pendulum-20261102/expand4x50/attention/qwen38/records/。',
       '解释联合字段为null表示缺解释或未完成事实审核；不是内容错误。数据集“错误提示”中可能包含泛泛暗示，只有Luna确认实际采纳用户主张才进入“采纳用户”子组。']
    report='\n'.join(L)+'\n';(O/'REPORT.md').write_text(report)
    build_html(O,summary,attention_summary,examples,outcomes,requests,items)
    provenance=dict(completed=True,questions=200,replies=600,attention_questions=100,judge='/root/pathvqa_luna_continuation',
                    model='gpt-6-luna',reasoning_effort='max',blind_sha256=hashes,comparison_sha256=review_hashes,correction_sha256=correction_hashes,
                    source_snapshot_sha256=hashlib.sha256((D/'source_snapshot.json').read_bytes()).hexdigest(),
                    report_sha256=hashlib.sha256((O/'REPORT.md').read_bytes()).hexdigest(),output=str(O),finished=time.time())
    dump(O/'completion.json',provenance)
    dump(D/'task_state.json',dict(phase='complete',completed=600,semantic_reviews=len(reviews),attention_questions=100,
        report=str(O/'index.html'),judge=provenance['judge'],model=provenance['model'],reasoning_effort=provenance['reasoning_effort'],finished=time.time()))
    print(json.dumps(dict(completed=True,conditions=summary['conditions'],paired=summary['paired'],attention_eligible={w:x['eligible'] for w,x in attention_summary['windows'].items()},report=str(O/'index.html')),ensure_ascii=False))

def make_heatmap(iid,item,maps,O):
    original=Image.open(B/'images'/item['domain']/Path(item['image']).name).convert('RGB')
    cellw,cellh=260,270
    canvas=Image.new('RGB',(cellw*4,cellh*2+80),'white'); draw=ImageDraw.Draw(canvas)
    try:font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',16)
    except OSError:font=ImageFont.load_default()
    for row,stage in enumerate(['think','answer']):
        m=maps.get((iid,stage))
        if m is None:continue
        a=[np.array(m['normalized_image_distribution'][c]).reshape(m['grid']) for c in ['neutral','belief_right','belief']]
        vmax=max(float(x.max()) for x in a)
        for col in range(4):
            pic=original.copy()
            if col:
                v=a[col-1]/max(vmax,1e-12)
                # Shared linear palette; blue -> green -> yellow. No per-condition rescaling.
                stops=np.array([[48,18,120],[25,140,125],[252,235,40]],float)
                z=v*2;i=np.minimum(z.astype(int),1);f=(z-i)[...,None]
                rgb=((1-f)*stops[i]+f*stops[i+1]).clip(0,255).astype('uint8')
                hm=Image.fromarray(rgb).resize(original.size,Image.Resampling.NEAREST)
                pic=Image.blend(original,hm,.65)
            pic.thumbnail((cellw-12,cellh-70))
            x=col*cellw+(cellw-pic.width)//2;y=row*cellh+42
            canvas.paste(pic,(x,y));draw.text((col*cellw+8,row*cellh+8),['Original','Neutral','Correct hint','Wrong hint'][col]+' / '+stage,fill='black',font=font)
        draw.text((8,row*cellh+cellh-25),f'Shared scale: patch share 0..{vmax:.5f}; 96 heads, normalized within image',fill='black',font=font)
    draw.text((8,cellh*2+34),iid+' | Model attention, not an independently annotated evidence mask',fill='black',font=font)
    name=iid+'.png';canvas.save(O/'heatmaps'/name)
    return name

def build_html(O,s,a,examples,outcomes,requests,items):
    e=html.escape
    parts=['<!doctype html><meta charset="utf-8"><title>Qwen PENDULUM结果</title><style>body{max-width:1150px;margin:40px auto;padding:0 20px;font:17px/1.65 system-ui;color:#17212d}table{border-collapse:collapse;width:100%;margin:22px 0}th,td{border-bottom:1px solid #ddd;text-align:left;padding:9px}img{width:100%;height:auto}pre{white-space:pre-wrap;background:#f4f6f8;padding:15px}a{color:#1366af}.muted{color:#647080}details{margin:12px 0}</style><h1>Qwen3.8 · PENDULUM</h1>',
       '<p>200题、三条件共600份回复。100题另有逐层逐头注意力。相同Luna（max）先盲提取、封存后语义比较；答案匹配与解释联合审核分开保存。</p>',
       '<p><a href="REPORT.md">完整统计报告</a> · <a href="summary.json">答案统计JSON</a> · <a href="attention_summary.json">注意力分组JSON</a> · <a href="judgments.json">全部600条标注</a></p>',
       '<table><tr><th>条件</th><th>答案匹配</th><th>不匹配</th><th>未解</th></tr>']
    for c,n in [('neutral','中性'),('belief_right','正确提示'),('belief','错误提示')]:
        r=s['conditions'][c];parts.append(f'<tr><td>{n}</td><td>{r["correct"]}</td><td>{r["wrong"]}</td><td>{r["unresolved"]}</td></tr>')
    x=s['paired']['belief'];parts+=['</table>',f'<p><b>中性匹配、错误提示后不匹配：{x["correct_to_wrong"]}题；反向恢复：{x["wrong_to_correct"]}题。</b></p>',
        '<h2>注意力可比窗口</h2><table><tr><th>前W个思考token</th><th>可用题</th><th>原图数</th><th>守住</th><th>新增错误</th></tr>']
    for W,r in a['windows'].items():
        gs=r['selectors']['balanced96'];parts.append(f'<tr><td>{W}</td><td>{r["eligible"]}</td><td>{r["unique_images"]}</td><td>{gs["hold"]["n"]}</td><td>{gs["new_wrong"]["n"]}</td></tr>')
    parts+=['</table><p class="muted">固定窗口比较要求三条件都够长。100题共用23张原图，区间按原图聚类。E是中性注意力最高的20%图像块，不是人工物体框；本批没有干预。</p>', '<h2>真实配对案例</h2>']
    for r in examples:
        iid=r['id'];it=items[iid];parts.append('<h3>'+e(iid)+' · '+e(it['question'])+'</h3>')
        parts.append('<p>参考答案：'+e(str(it['gold']))+'</p>')
        for c,n in [('neutral','中性'),('belief_right','正确提示'),('belief','错误提示')]:
            j=outcomes[iid,c];req=requests[j['key']];parts.append('<details><summary>'+n+' · '+e(j['status'])+'</summary><p>'+e(req['message'])+'</p><pre>'+e(req['reply'])+'</pre>')
            trace=json.loads((D/'blind'/req['trace']).read_text());parts.append('<details><summary>完整原始思考</summary><pre>'+e(trace['thinking'])+'</pre></details></details>')
        parts.append('<img src="heatmaps/'+e(iid)+'.png" alt="三条件注意力区域">')
    parts.append('<p><a href="attention_per_item.json">全部逐题窗口指标及入选头</a>；<a href="completion.json">完成凭据与来源SHA</a></p>')
    (O/'index.html').write_text('\n'.join(parts))

if __name__=='__main__':main()
