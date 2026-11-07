"""Compare frozen pilot windows across Gemma attention layer groups."""
import json
from pathlib import Path
import numpy as np

ROOT = Path('/home/linxin/trigger-regions-gemma4-all-layers-20261108')

def normalize(x):
    s = x.sum(-1, keepdims=True)
    return np.divide(x, s, out=np.zeros_like(x, dtype=np.float64), where=s > 0)

def jsd(p, q):
    m = (p+q)/2
    def kl(a):
        ok = a > 0
        return float((a[ok]*np.log(a[ok]/m[ok])).sum())
    return (kl(p)+kl(q))/2

def spatial(dist, parts, allowed=None):
    d = dist.astype(np.float64)
    if allowed is not None:
        if not allowed.any():
            return None
        d = d[:, allowed]
    needed = np.concatenate(list(parts.values()))
    if np.any(d[needed].sum(-1) <= 0):
        return None
    d = normalize(d)
    avgs = {k:normalize(d[v].mean(0)) for k,v in parts.items()}
    e = np.argsort(-avgs['before'], kind='stable')[:max(1,round(d.shape[-1]*.2))]
    return {'focus':{k:float(v[e].sum()) for k,v in avgs.items()},
            'jsd_nats':{k:jsd(avgs['before'],v) for k,v in avgs.items()},
            'support_blocks':int(d.shape[-1]), 'focus_blocks':int(len(e))}

def select_heads(masses):
    scores = np.mean([m.mean(0) for m in masses],axis=0)
    sel = np.zeros(scores.size,bool)
    sel[np.argsort(-scores.ravel(),kind='stable')[:max(1,round(scores.size*.25))]]=True
    return sel.reshape(scores.shape)

def analyze():
    assert (ROOT/'replay_complete.json').exists(), 'Replay incomplete'
    specs = json.loads((ROOT/'layer_specs.json').read_text())
    groups = {k:[int(l) for l,s in specs.items() if s['kind']==k] for k in ['global','sliding']}
    groups['all'] = sorted(map(int,specs))
    metas = [json.loads(p.read_text()) for p in sorted((ROOT/'records').glob('*.json'))]
    pairs = {}
    for meta in metas:
        pairs.setdefault(meta['id'],{})[meta['condition']] = meta
    result=[]
    for iid, cs in pairs.items():
        zs = {c:dict(np.load(ROOT/'records'/f'{iid}__{c}.npz')) for c in cs}
        selections={g:select_heads([np.stack([z[f'mass_{l}'] for l in layers],1) for z in zs.values()]) for g,layers in groups.items()}
        for c,meta in cs.items():
            z=zs[c];index={int(r):i for i,r in enumerate(z['rows'])}
            for event in meta['events']:
                t,pre=event['t'],event['pre']
                parts={k:np.array([index[r] for r in rr]) for k,rr in {'before':range(t-pre,t),'at':[t],'after':range(t+1,t+33)}.items()}
                ev={**event,'step_1based':t+1,'id':iid,'condition':c,'group':meta['group'],'layers':{}}
                if iid=='PD-re_used-0009' and c=='belief' and t==64:
                    ev['semantic_position']='transition_after_user_restatement'
                else:
                    ev['semantic_position']='not_reannotated; legacy window category retained'
                for g,layers in groups.items():
                    mass=np.stack([z[f'mass_{l}'] for l in layers],1)
                    dist=np.mean([z[f'dist_{l}'].astype(np.float64) for l in layers],0)
                    visible=np.stack([z[f'visible_{l}'] for l in layers],1)
                    union_rows=np.concatenate(list(parts.values()))
                    common=visible[union_rows].all(axis=(0,1))
                    vis={}
                    for k,rr in parts.items():
                        counts=visible[rr].sum(-1)
                        vis[k]={'min_blocks':int(counts.min()),'max_blocks':int(counts.max()),'full_layer_rows':int((counts==visible.shape[-1]).sum()),'partial_layer_rows':int(((counts>0)&(counts<visible.shape[-1])).sum()),'zero_layer_rows':int((counts==0).sum()),'total_layer_rows':int(counts.size)}
                    ev['layers'][g]={'n_layers':len(layers),'selected_heads':int(selections[g].sum()),
                        'mass':{k:float(mass[rr][:,selections[g]].mean()) for k,rr in parts.items()},
                        'mass_all_heads':{k:float(mass[rr].mean()) for k,rr in parts.items()},
                        'raw_spatial':spatial(dist,parts),'common_visible_spatial':spatial(dist,parts,common),
                        'common_visible_blocks':int(common.sum()),'image_blocks':int(visible.shape[-1]),'visibility':vis}
                result.append(ev)
    out={'scope':'Same six questions, twelve original replies and 48 saved windows; passive replay; no new generation or intervention.',
         'head_selection':'Top 25% separately within each layer group; mean over recorded rows within each condition, then equal weight across the two conditions.',
         'focus_definition':'Top 20% blocks of the pre-window attention distribution, not a ground-truth region. Each layer uses all heads weighted by image attention; layers then averaged equally.',
         'window_note':'Original frozen windows retained to isolate adding layers. Original opening cutoff is not a semantic restatement label; PD-re_used-0009 step65 is a transition after the restatement.',
         'common_visible_note':'Intersect blocks actually visible for every layer and every row in pre/at/post; renormalize within that support. Zero support or zero image mass makes spatial statistics null.',
         'events':result}
    (ROOT/'all_layers_events.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False))
    chosen=[('CQ-003','belief',812),('PV-013','belief',295),('PD-re_used-0009','belief',65)]
    lines=['# Gemma：加入滑动窗口层后的六题复查','',
      '同一批6道题、12条原回答、48个原有窗口。记录10个全局层和50个滑动窗口层；不生成新回答，不改变模型前向输出。',
      '','表中数值是报警前32步→后32步。看图总量、原焦点份额以百分比表示；JSD单位为nats。每个层组分别按两条件平均看图量选前25%的头来算总量；焦点与JSD用各层所有头。',
      '','“原焦点”是报警前注意力最高的20%图块，不等于人工确认的正确区域。滑动层看不见的图块不能解释为主动忽视。','']
    labels={'global':'10个全局层','sliding':'50个滑动层','all':'全部60层'}
    for iid,c,step in chosen:
        ev=next(x for x in result if (x['id'],x['condition'],x['step_1based'])==(iid,c,step))
        lines += [f'## {iid}，第{step}步','',f'预测token：`{ev["token"]!r}`；熵差{ev["dH"]:.4f} nats。'+('这是复述用户结束后的换段位置，旧文件的opening分类不代表仍在复述。' if step==65 else ''),'',
        '|层组|看图总量|原焦点份额|前后JSD|全窗口共同可见图块|','|---|---:|---:|---:|---:|']
        for g in groups:
            a=ev['layers'][g];m=a['mass'];s=a['raw_spatial'];fo=f'{s["focus"]["before"]*100:.1f}% → {s["focus"]["after"]*100:.1f}%' if s else '不可计算（图像不可见）';j=f'{s["jsd_nats"]["after"]:.5f}' if s else '—'
            lines += [f'|{labels[g]}|{m["before"]*100:.2f}% → {m["after"]*100:.2f}%|{fo}|{j}|{a["common_visible_blocks"]}/{a["image_blocks"]}|']
        s=ev['layers']['sliding'];v=s['visibility']
        lines += ['',f'滑动层可见图块范围：前窗口{v["before"]["min_blocks"]}–{v["before"]["max_blocks"]}块，报警行{v["at"]["min_blocks"]}–{v["at"]["max_blocks"]}块，后窗口{v["after"]["min_blocks"]}–{v["after"]["max_blocks"]}块。']
        if s['common_visible_spatial'] is not None:
            a=s['common_visible_spatial'];lines += [f'只比较全程共同可见图块：滑动层焦点份额{a["focus"]["before"]*100:.1f}% → {a["focus"]["after"]*100:.1f}%，JSD={a["jsd_nats"]["after"]:.5f}。']
        else:
            lines += ['没有足够的共同可见图块来计算同范围的空间变化；原始空间变化不能直接解释为主动转移关注。']
        lines += ['']
    checks=[max(m['global_baseline_check'].values()) for m in metas]
    lines += ['## 核对','',f'12条均完成。与旧10全局层数组逐项比较，最大绝对差={max(checks):.8g}；第一条的原生前向logits前后差={metas[0]["native_logits_maxabs"]}。',
        '','原回答、熵差、窗口、配置、模型均未更改。所有层仍返回模型原有注意力函数的输出。数组只保留在服务器，不上传。']
    (ROOT/'all_layers_report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__=='__main__':
    analyze()
