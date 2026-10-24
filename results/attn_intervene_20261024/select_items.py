"""Select by the repository's rule_answer on original final_answer, not Luna extraction."""
import hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
from symexp.parsing import rule_answer
PILOT=['PV-000','PV-001','PV-002','PV-003','PV-004','PV-005','PV-009','PV-013','PV-017','PV-031']

def main():
    runs=ROOT.parent/'pathvqa_symexp_v1_20261023/runs'
    data={}; labels={}; audit=[]; originals=[]
    for name,condition in [('plain','neutral'),('plain_pressure_reference','user_wrong')]:
        data[condition]={r['id']:r for r in map(json.loads,(runs/name/'final.jsonl').read_text().splitlines())}
        labels[condition]={r['key']:r for r in map(json.loads,(runs/name/'judgments.jsonl').read_text().splitlines())}
    for item in sorted(data['neutral']):
        n,p=data['neutral'][item],data['user_wrong'][item]
        records={c:json.loads(Path(row['source_record']).read_text()) for c,row in [('neutral',n),('user_wrong',p)]}
        pred={c:rule_answer(r.get('final_answer','')) for c,r in records.items()}
        gold=n['gold'].lower(); assert gold==p['gold'].lower() and gold in ('yes','no')
        qual=pred['neutral']==gold and pred['user_wrong'] in ('yes','no') and pred['user_wrong']!=gold
        jn,jp=labels['neutral'][n['key']],labels['user_wrong'][p['key']]
        strict=True if jn.get('joint_correct') is True and jp.get('joint_correct') is False else (False if jn.get('joint_correct') is False or jp.get('joint_correct') is True else None)
        a=dict(id=item,ptype=p['ptype'],gold=gold,neutral_answer=pred['neutral'],pressure_answer=pred['user_wrong'],qualifies_rule=qual,qualifies_joint=strict,neutral_joint=jn.get('joint_correct'),pressure_joint=jp.get('joint_correct'),pilot10=item in PILOT,question=n['question'],claim=p['claim'],pressure_message=p['message'])
        audit.append(a)
        if qual: originals.append((a,records))
    chosen=sorted(originals,key=lambda x:(not x[0]['pilot10'],x[0]['id']))[:30]
    assert len(chosen)==30, f'Only {len(chosen)} eligible questions'
    (ROOT/'items.json').write_text(json.dumps([x[0] for x in chosen],ensure_ascii=False,indent=2)+'\n')
    with (ROOT/'original_records.jsonl').open('w') as f:
        for a,rs in chosen:
            for cond,r in rs.items():
                f.write(json.dumps(dict(id=a['id'],condition=cond,input_ids=r['input_ids'],output_ids=r['output_ids'],image_sha256=r['image_sha256'],rendered_chat_template=r['rendered_chat_template'],text=r['text'],sampling=r['sampling'],decoding=r.get('decoding'),final_answer=r.get('final_answer'),raw_output=r.get('raw_output')),ensure_ascii=False)+'\n')
    (ROOT/'selection_audit.json').write_text(json.dumps(dict(rule='symexp.parsing.rule_answer(original final_answer); unresolved retained in audit, not silently counted wrong',total=150,eligible=len(originals),selected=30,priority='eligible pilot10 first, ascending ID, then other eligible ascending ID',rows=audit),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(eligible=len(originals),selected=[a['id'] for a,_ in chosen],pilot=[a['id'] for a,_ in chosen if a['pilot10']],first_two=[a['id'] for a,_ in chosen[:2]])))
if __name__=='__main__':main()
