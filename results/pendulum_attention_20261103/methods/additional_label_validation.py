"""Validate effective extraction amendments without altering sealed originals."""
import hashlib
import json
from pathlib import Path
from label_validation import validate

def load(blind):
    blind=Path(blind)
    requests,labels,hashes=validate(blind,require_all=False)
    assert len(labels)==len(requests)==1200
    for p in sorted(blind.glob('extraction_corrections_*.jsonl')):
        hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
        seen=set()
        for r in map(json.loads,p.read_text().splitlines()):
            k=r['key']
            assert k in labels and k not in seen and r['replaces_key']==k
            seen.add(k)
            assert r['status'] in ['single','multiple','none','unclear']
            assert r['quote_source'] in ['reply','thinking','raw_output']
            assert r['explanation_presence'] in ['present','absent','unclear']
            req=requests[k]
            source=req['reply'] if r['quote_source']=='reply' else json.loads((blind/req['trace']).read_text())[r['quote_source']]
            assert isinstance(r['answer_quote'],str) and r['answer_quote'] in source
            if r['status']=='single':
                assert r['semantic_answer'] and isinstance(r['semantic_answer'],str) and r['answer_quote']
            labels[k]={**labels[k],**r,'raw_extraction_status':labels[k]['status'],'extraction_amendment_file':p.name}
    return requests,labels,hashes
