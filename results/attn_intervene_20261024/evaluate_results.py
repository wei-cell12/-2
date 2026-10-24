"""Rule-only evaluation; unresolved predictions are displayed, never silently wrong."""
import csv,json,math
from pathlib import Path
CONDITIONS=['base','restore_E','restore_rand']
def exact_p(b,c):
    n=b+c
    return min(1.,2*sum(math.comb(n,k) for k in range(min(b,c)+1))/2**n) if n else 1.
def evaluate(root):
    root=Path(root);items=json.loads((root/'items.json').read_text())
    runs={c:{r['id']:r for r in map(json.loads,(root/'runs'/c/'final.jsonl').read_text().splitlines())} for c in CONDITIONS}
    assert all(set(r)=={x['id'] for x in items} for r in runs.values())
    summary=dict(n=30,seed=42,metric='rule yes/no only; semantic answer+explanation pending',conditions={},pairs={})
    table=[]
    for c in CONDITIONS:
        rr=list(runs[c].values());valid=[r for r in rr if r['rule_pred'] in ('yes','no')]
        w=sum(r['metrics']['head_token_rows'] for r in rr)
        summary['conditions'][c]=dict(correct=sum(r['rule_pred']==r['gold'] for r in valid),wrong=sum(r['rule_pred']!=r['gold'] for r in valid),unresolved=[r['id'] for r in rr if r['rule_pred'] not in ('yes','no')],head_token_rows=w,mean_delta=sum(r['metrics']['mean_delta']*r['metrics']['head_token_rows'] for r in rr)/w,region_before=sum(r['metrics']['region_before']*r['metrics']['head_token_rows'] for r in rr)/w,region_after=sum(r['metrics']['region_after']*r['metrics']['head_token_rows'] for r in rr)/w,clamped_head_tokens=sum(r['metrics']['clamped_head_tokens'] for r in rr))
    for other in ['base','restore_rand']:
        b=c=0;excluded=[]
        for x in items:
            a,z=runs['restore_E'][x['id']],runs[other][x['id']]
            if a['rule_pred'] not in ('yes','no') or z['rule_pred'] not in ('yes','no'):excluded.append(x['id']);continue
            ac=a['rule_pred']==a['gold'];zc=z['rule_pred']==z['gold'];b+=ac and not zc;c+=zc and not ac
        summary['pairs']['restore_E_vs_'+other]=dict(restore_E_only=b,other_only=c,p_exact=exact_p(b,c),excluded_unresolved=excluded)
    for x in items:
        row=dict(id=x['id'],ptype=x['ptype'],gold=x['gold'],old_joint_qualifies=x['qualifies_joint'])
        for cond in CONDITIONS:
            r=runs[cond][x['id']];row.update({cond+'_answer':r['rule_pred'],cond+'_correct':r['rule_pred']==r['gold'] if r['rule_pred'] in ('yes','no') else None,cond+'_delta':r['metrics']['mean_delta'],cond+'_tokens':r['gen_tokens']})
        table.append(row)
    with (root/'items.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    (root/'evaluation_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    return summary
if __name__=='__main__':evaluate(Path(__file__).resolve().parent)
