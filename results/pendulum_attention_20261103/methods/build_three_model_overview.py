import html
import json
from pathlib import Path

B=Path(__file__).resolve().parent
O=B/'three_model_results_20261103'
x=json.loads((O/'summary.json').read_text())
e=html.escape
models={'qwen38':'Qwen3.8-27B','gemma4':'Gemma 4 31B','qwen36_moe':'Qwen3.6-35B-A3B'}
conds={'neutral':'中性','belief':'用户说错','belief_right':'用户说对'}
parts=['''<!doctype html><html lang="zh"><meta charset="utf-8"><title>PENDULUM 三模型结果</title>
<style>body{margin:32px auto;max-width:1250px;padding:0 22px;background:#f8fafc;color:#1d2939;font:17px/1.65 system-ui}h1,h2{line-height:1.3}section{background:white;border:1px solid #dce3ec;border-radius:12px;padding:24px;margin:22px 0}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:12px;border-bottom:1px solid #e1e7ef}th{background:#f1f5fa}.small{font-size:14px;color:#586778}.note{border-left:4px solid #c58917;padding:14px;background:#fff9e9}a{color:#1363a8}select,input{font:inherit;padding:8px;margin:8px}details{margin:12px 0}.scroll{overflow:auto}.bad{color:#a6332d}.good{color:#146a4e}</style>
<h1>PENDULUM：三模型、同一批200题</h1><p>四类图像各50题；每题中性、用户说错、用户说对各一次。三款模型各600份回复，生成和注意力收集均已结束。</p>
<p><a href="REPORT.md">完整分析报告</a> · <a href="cases.html">原始思考、回答与区域热图</a> · <a href="summary.json">完整统计与分组ID</a></p>
<section><h2>1. 答案结果</h2><p>每格为“正确 / 错误 / 未解”，分母均为200。这里按答案语义匹配原数据集标签；解释联合审核另列。</p><table><tr><th>模型</th><th>中性</th><th>用户说错</th><th>用户说对</th></tr>''']
for m,n in models.items():
    s=x['models'][m]
    cells=[' / '.join(str(s['conditions'][c][t]) for t in ['correct','wrong','unresolved']) for c in conds]
    parts.append('<tr><th>'+n+'</th>'+''.join('<td>'+c+'</td>' for c in cells)+'</tr>')
parts.append('</table></section><section><h2>2. 只看中性答对的题，压力后怎样</h2><table><tr><th>模型</th><th>中性答对</th><th>用户说错后守住</th><th>被带偏</th><th>未解</th></tr>')
for m,n in models.items():
    s=x['models'][m];t=s['transitions']['belief']['table']['correct']
    parts.append(f"<tr><th>{n}</th><td>{len(s['cohort_ids']['neutral_correct'])}</td><td>{t['correct']}</td><td class=bad>{t['wrong']}</td><td>{t['unresolved']}</td></tr>")
parts.append(f"</table><p>各模型中性答对的题目可能不同。三模型共同中性答对的固定交集为{x['common_neutral_correct']['n']}题，交集结果见完整报告数据。</p></section>")
parts.append('<section><h2>3. 注意力区域变化：固定前64个思考token</h2><p>只看中性答对且三条件都能比较的题。变化=用户说错−中性，负值表示下降。看图总量和E份额的变化单位为<strong>百分点</strong>，JSD单位为nats。</p><div class=scroll><table><tr><th>模型/层</th><th>分组</th><th>题数</th><th>看图总量变化</th><th>E份额变化</th><th>原始JSD</th><th>扣噪声JSD</th></tr>')
for m,n in models.items():
    for g,ws in x['models'][m]['attention'].items():
        if g=='all':continue
        for cohort,cn in [('hold','守住'),('flip','带偏')]:
            r=ws['64']['cohorts'][cohort];p=r['selectors']['balanced96' if m=='qwen38' else 'balanced25pct']['pairs']['neutral__belief']
            def f(v,mul=1):return '—' if v is None else f'{v*mul:+.4f}'
            parts.append('<tr><th>'+n+' / '+{'global':'全局层','sliding':'滑动层'}[g]+'</th><td>'+cn+'</td><td>'+str(r['n'])+'</td><td>'+f(p['image_mass_delta']['mean'],100)+'</td><td>'+f(p['E_share_delta']['mean'],100)+'</td><td>'+f(p['jsd_full'])+'</td><td>'+f(p['excess'])+'</td></tr>')
parts.append('</table></div><p class=note>E是模型中性完整思考最关注的前20%图像块，<strong>不是人工标定的真实目标区域</strong>。扣噪声后可以为负，不把负分母算成“几倍”。Gemma滑动层只纳入图像完全可见的窗口。</p><p class=small>注意力覆盖re_used、self_captured各50题，共100题、23张原图。其他两类没有采集区域注意力。统计按图像聚类给区间；主比较统一top25%看图头，固定96头对照、64/128/256/512窗口分组名单都在summary.json。</p></section>')
parts.append('<section><h2>4. 逐题核对</h2><p>每行一款模型的一道题。可筛选查看三条件答案、是否属于“中性对→压力错”、以及是否有注意力数据。</p><label>模型<select id=model><option value="">全部</option>'+''.join('<option value="'+m+'">'+n+'</option>' for m,n in models.items())+'</select></label><label>筛选<select id=cohort><option value="">全部</option><option value="flip">仅中性对→用户说错后错</option><option value="neutral">仅中性答对</option><option value="attention">仅有注意力数据</option></select></label><input id=search placeholder="题目ID或答案关键词"><div class=scroll><table id=registry><thead><tr><th>模型/题目</th><th>中性</th><th>用户说错</th><th>用户说对</th><th>备注</th></tr></thead><tbody>')
registry=json.loads((O/'sample_registry.json').read_text())
for r in registry:
    notes=[]
    if r['flip']:notes.append('中性对→压力错')
    if r['attention_collected']:notes.append('有注意力')
    if r['reference_dispute_any_model']:notes.append('参照有争议')
    row='<tr data-model="'+r['model']+'" data-flip="'+str(int(r['flip']))+'" data-neutral="'+str(int(r['neutral_correct']))+'" data-attention="'+str(int(r['attention_collected']))+'"><th>'+models[r['model']]+'<br>'+e(r['id'])+'</th>'
    for c in conds:
        row+='<td>'+e(str(r['answers'][c]))+'<br><span class=small>'+{'correct':'答案匹配','wrong':'答案不匹配','unresolved':'未解'}[r['statuses'][c]]+'</span></td>'
    parts.append(row+'<td>'+e('；'.join(notes))+'</td></tr>')
parts.append('''</tbody></table></div></section><script>
const m=document.querySelector('#model'),c=document.querySelector('#cohort'),q=document.querySelector('#search');
function filter(){for(const r of document.querySelectorAll('#registry tbody tr')){const t=c.value; r.hidden=(m.value&&r.dataset.model!==m.value)||(t&&r.dataset[t]!=='1')||!r.innerText.toLowerCase().includes(q.value.toLowerCase());}}
for(const el of [m,c,q])el.addEventListener('input',filter);
</script></html>''')
(O/'index.html').write_text('\n'.join(parts))
print(O/'index.html')
