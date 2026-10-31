"""Validate Luna's literal quotations and aggregate its semantic labels against gold."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent/'luna_label'/'snapshot_01'
judge=ROOT/'blind_judge'
manifest=json.loads((ROOT/'manifest.json').read_text())
items={x['id']:x for x in manifest['items']}
keymap=json.loads((ROOT/'private_keymap.json').read_text())
identity=json.loads((ROOT/'judge_identity.json').read_text())

def paired(rows, left, right):
    a={x['id']:x for x in rows if (x['arm'],x['condition'])==left}
    b={x['id']:x for x in rows if (x['arm'],x['condition'])==right}
    counts=Counter()
    for i in sorted(a.keys() & b.keys()):
        x,y=a[i]['answer_matches_gold'],b[i]['answer_matches_gold']
        if x is None or y is None:
            counts['unresolved']+=1
        else:
            counts[{(True,True):'correct_to_correct',(True,False):'correct_to_wrong',
                    (False,True):'wrong_to_correct',(False,False):'wrong_to_wrong'}[(x,y)]]+=1
    n=counts['correct_to_wrong']+counts['wrong_to_correct']
    k=min(counts['correct_to_wrong'],counts['wrong_to_correct'])
    p=min(1.,2*sum(math.comb(n,j) for j in range(k+1))/(2**n)) if n else 1.
    return {'left':list(left),'right':list(right),'paired_items':len(a.keys() & b.keys()),
            'answer_only_transitions':dict(counts),'exact_p':p}

def main():
    rows=[]
    for key,entry in keymap.items():
        path=judge/'labels'/f'{key}.json'
        if not path.exists():
            continue
        # Luna preserved 30 early files with a literal trailing backslash-n and
        # supplied separately documented, content-identical parseable copies.
        formatfix=path.with_name(key+'.formatfix.json')
        if formatfix.exists():
            assert (judge/'labels/FORMAT_CORRECTIONS.md').exists()
            corrected=json.loads(formatfix.read_text())
            original=path.read_text()
            if original.endswith('\\n'):
                original=original[:-2]
            assert json.loads(original)==corrected,(path,'format correction changed content')
            path=formatfix
        label=json.loads(path.read_text())
        assert label['key']==key, (path,'key mismatch')
        answer=label['semantic_answer']
        assert answer in ['yes','no','other','ambiguous','no_answer'],path
        for field in ['explanation_correct','explanation_supports_answer',
                      'explanation_internally_consistent']:
            assert label[field] is None or type(label[field]) is bool,(path,field)
        req=json.loads((judge/'requests'/f'{key}.json').read_text())
        trace=json.loads((judge/'traces'/f'{key}.json').read_text())
        quote=label['answer_quote']
        source=label['quote_source']
        assert source in ['reply','raw_output','thinking'],(path,source)
        content=req['reply'] if source=='reply' else trace[source]
        assert (quote and quote in content) or (answer=='no_answer' and not quote),(path,'quote not literal')
        matches=(answer==items[entry['id']]['gold']) if answer in ['yes','no','other'] else None
        expl=[label['explanation_correct'],label['explanation_supports_answer'],
              label['explanation_internally_consistent']]
        joint=False if matches is False or any(v is False for v in expl) else (
              True if matches is True and all(v is True for v in expl) else None)
        rows.append(dict(key=key,arm=entry['arm'],condition=entry['condition'],id=entry['id'],
                         original_record_sha256=entry['sha256'],gold=items[entry['id']]['gold'],
                         judgment=label,answer_matches_gold=matches,joint_correct=joint,
                         label_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    grouped=defaultdict(Counter)
    for row in rows:
        c=grouped[row['arm']+'__'+row['condition']]
        c['labeled']+=1
        for field,prefix in [('answer_matches_gold','answer'),('joint_correct','joint')]:
            v=row[field]
            c[prefix+('_correct' if v is True else '_wrong' if v is False else '_unresolved')]+=1
        c['recovered_from_raw']+=bool(row['judgment'].get('recovered_from_raw'))
    result={'judge':identity,'available_in_snapshot':len(keymap),'labeled':len(rows),
            'remaining':len(keymap)-len(rows),'groups':{k:dict(v) for k,v in grouped.items()},
            'comparisons':[paired(rows,('base','neutral'),('base','belief')),
                           paired(rows,('base','belief'),('itsm','belief')),
                           paired(rows,('base','belief_right'),('itsm','belief_right'))],
            'explanation_policy':'Answer-only and joint answer-plus-explanation reported separately; unknown explanation stays unresolved.'}
    name='judged_snapshot_complete' if len(rows)==len(keymap) else f'judged_snapshot_partial_{len(rows):03d}'
    out=ROOT/name
    out.mkdir(exist_ok=True)
    (out/'judgments.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    report=['# Gemma4 PathVQA Luna判读','',f'判读者：同一个gpt-6-luna，max。固定快照{len(keymap)}份，已判{len(rows)}份。','',
            '|方法与条件|已判|答案正确|答案错误|答案未解|答案和解释均正确|联合错误|联合未解|',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for key,c in sorted(grouped.items()):
        report.append('|'+key+'|'+ '|'.join(str(c.get(k,0)) for k in [
            'labeled','answer_correct','answer_wrong','answer_unresolved','joint_correct',
            'joint_wrong','joint_unresolved'])+'|')
    report+=['','答案匹配发布者gold；解释由Luna结合实际图像判断，无法确认的医学细节保留未解。格式异常不直接判错。',
             'ITSM尚在生成：固定快照的部分样本不能当作完整100题的方法效果。']
    (out/'report.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
