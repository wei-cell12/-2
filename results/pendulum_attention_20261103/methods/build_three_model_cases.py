"""Render selected existing trajectories and shared-scale scientific heatmaps."""
import html
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

B=Path(__file__).resolve().parent
O=B/'three_model_results_20261103'
S=json.loads((O/'summary.json').read_text())
items={r['id']:r for r in map(json.loads,(B/'expand4x50_items.jsonl').read_text().splitlines())}
names={'qwen38':'Qwen3.8-27B','gemma4':'Gemma 4 31B','qwen36_moe':'Qwen3.6-35B-A3B'}
e=html.escape
(O/'heatmaps').mkdir(exist_ok=True)
parts=['<!doctype html><meta charset="utf-8"><title>逐题轨迹与区域热图</title><style>body{max-width:1300px;margin:30px auto;font:17px/1.6 system-ui}pre{white-space:pre-wrap;background:#f4f6f9;padding:16px}img{max-width:100%}article{border-top:2px solid #ccc;margin:30px 0}</style><h1>逐题轨迹与区域热图</h1><p>案例选择规则：在中性答案匹配、用户说错后不匹配、三条件均有64个思考token的无参考争议样本中，取E份额降幅最大的两题和降幅最小的一题（去重）。这不是随机样本，不用于代替总体统计。没有合格案例则如实留空。</p>']
manifest=[]
all_maps=json.loads((O/'primary_first64_maps_private.json').read_text())
try:font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',16)
except OSError:font=ImageFont.load_default()
for model,name in names.items():
    source=json.loads((B/('qwen_judge_export_private.json' if model=='qwen38' else model+'_judge_export_private.json')).read_text())
    src={(r['id'],r['condition']):r for r in source['records']}
    metric=json.loads((B/('qwen_attention_metrics_private.json' if model=='qwen38' else model+'_attention_metrics_private.json')).read_text())
    rows={r['id']:r for r in metric['metrics']['rows']}
    maps=[r for r in all_maps if r['model']==model]
    selector='balanced96' if model=='qwen38' else 'balanced25pct'
    group='global'
    s=S['models'][model]
    eligible=s['attention'][group]['64']['cohorts']['flip']['ids']
    disputed=set(S['reference_dispute_sensitivity']['excluded_ids'])
    def window(iid,g='global'):
        r=rows[iid]
        return (r if model=='qwen38' else r['groups'][g])['windows']['64']
    def delta(iid):
        c=window(iid)['selectors'][selector]['conditions']
        return c['belief']['E_share']-c['neutral']['E_share']
    available={r['id'] for r in maps if r.get('stage','first64')=='first64'}
    pool=sorted([i for i in eligible if i not in disputed and i in available],key=delta)
    chosen=list(dict.fromkeys(pool[:2]+pool[-1:]))
    parts.append('<h2>'+e(name)+'</h2>')
    if not chosen:parts.append('<p>当前本地没有合格的同窗口案例图，不用完整思考热图冒充。</p>')
    for iid in chosen:
        it=items[iid];impath=B/'images'/it['domain']/Path(it['image']).name
        im=Image.open(impath).convert('RGB')
        groups=['global'] if model!='gemma4' else ['global','sliding']
        parts.append('<article><h3>'+e(iid)+' · '+e(it['question'])+'</h3><p>数据集参照：'+e(str(it['gold']))+'</p>')
        for g in groups:
            mm=next((r for r in maps if r['id']==iid and r.get('group','global')==g and r.get('stage','first64')=='first64'),None)
            if mm is None:continue
            w=window(iid,g)
            if not w['eligible']:continue
            vals=w['selectors'][selector]['conditions']
            arrays=[np.array(mm['normalized_image_distribution'][c]).reshape(mm['grid']) for c in ['neutral','belief_right','belief']]
            vmax=max(float(a.max()) for a in arrays)
            canvas=Image.new('RGB',(1280,450),'white');draw=ImageDraw.Draw(canvas)
            stops=np.array([[48,18,120],[25,140,125],[252,235,40]],float)
            for col in range(4):
                pic=im.copy()
                if col:
                    z=arrays[col-1]/max(vmax,1e-12)*2;j=np.minimum(z.astype(int),1);f=(z-j)[...,None]
                    rgb=np.clip((1-f)*stops[j]+f*stops[j+1],0,255).astype('uint8')
                    heat=Image.fromarray(rgb).resize(im.size,Image.Resampling.NEAREST)
                    pic=Image.blend(im,heat,.65)
                pic.thumbnail((310,295));canvas.paste(pic,(320*col+(320-pic.width)//2,50))
                draw.text((320*col+8,8),['Original','Neutral','User correct','User wrong'][col],fill='black',font=font)
                if col:
                    c=['neutral','belief_right','belief'][col-1];v=vals[c]
                    draw.text((320*col+8,355),f"Image mass={v['image_mass']*100:.2f}%",fill='black',font=font)
                    draw.text((320*col+8,381),f"E share={v['E_share']*100:.2f}%",fill='black',font=font)
            draw.text((8,415),f'{name} | {iid} | {g} | first64 thought tokens; same {mm["selected_heads"]} heads (top25%); common scale 0..{vmax:.5f}',fill='black',font=font)
            dest=f'{model}_{iid}_{g}_first64.png';canvas.save(O/'heatmaps'/dest)
            parts.append('<p>层类：'+g+'。图像内部归一化、三条件同色标；紫色低、黄色高。</p><img src="heatmaps/'+dest+'">')
            manifest.append(dict(model=model,id=iid,group=g,file='heatmaps/'+dest,selection='two largest E decreases + smallest decrease among eligible non-disputed flips',E_delta=delta(iid)))
        for c,cn in [('neutral','中性'),('belief_right','用户说对'),('belief','用户说错')]:
            r=src[iid,c]
            parts.append('<details><summary>'+cn+'：'+e(str(r['reply']))+'</summary><p>输入：'+e(r['message'])+'</p><h4>原始思考</h4><pre>'+e(r['thinking'])+'</pre><h4>最终回答</h4><pre>'+e(r['reply'])+'</pre><p>源SHA256：'+e(r['source_sha256'])+'</p></details>')
        parts.append('</article>')
(O/'cases.html').write_text('\n'.join(parts))
(O/'heatmap_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
p=O/'index.html';s=p.read_text();s=s.replace('<h1>PENDULUM 三模型结果</h1>','<h1>PENDULUM 三模型结果</h1><p><a href="cases.html">查看逐题原始思考、回答与同窗口热图</a></p>');p.write_text(s)
print(json.dumps(dict(cases=len({(r['model'],r['id']) for r in manifest}),heatmaps=len(manifest))))
