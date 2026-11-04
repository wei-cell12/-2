"""Choose cases from completed results; request only aggregate fixed-window maps."""
import json
from pathlib import Path

B=Path(__file__).resolve().parent
O=B/'three_model_results_20261103'
summary=json.loads((O/'summary.json').read_text())
requests=[]
selection={}
for model in ['qwen38','gemma4','qwen36_moe']:
    met=json.loads((B/('qwen_attention_metrics_private.json' if model=='qwen38' else model+'_attention_metrics_private.json')).read_text())['metrics']
    rows={r['id']:r for r in met['rows']}
    s=summary['models'][model]
    selector='balanced96' if model=='qwen38' else 'balanced25pct'
    def w(iid,g='global'):
        r=rows[iid]
        return (r if model=='qwen38' else r['groups'][g])['windows']['64']
    def delta(iid):
        a=w(iid)['selectors'][selector]['conditions']
        return a['belief']['E_share']-a['neutral']['E_share']
    pool=sorted([i for i in s['attention']['global']['64']['cohorts']['flip']['ids'] if i not in summary['reference_dispute_sensitivity']['excluded_ids']],key=delta)
    chosen=list(dict.fromkeys(pool[:2]+pool[-1:]))
    selection[model]=dict(pool_ids=pool,chosen_ids=chosen,rule='largest two E-share decreases and smallest E-share decrease among non-disputed global fully-visible first64 flip cases',selector=selector)
    for iid in chosen:
        for group in (['global','sliding'] if model=='gemma4' else ['global']):
            win=w(iid,group)
            if not win['eligible']:continue
            r=rows[iid];sel=win['selectors'][selector]
            requests.append(dict(model=model,id=iid,group=group,selector=selector,head_pairs=sel['head_pairs'],
                source_sha256=r['source_sha256'],image_sha256=r['image_sha256'],grid=r['image_grid'],
                expected_conditions=sel['conditions']))
(O/'case_selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2))
(O/'case_map_requests_private.json').write_text(json.dumps(requests,ensure_ascii=False))
print(json.dumps({m:v['chosen_ids'] for m,v in selection.items()}))
