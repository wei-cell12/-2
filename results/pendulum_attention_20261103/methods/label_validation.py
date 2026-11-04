"""No semantic extraction here: validate Luna-produced files and preserve provenance."""
import hashlib
import json
from pathlib import Path

def validate(blind, require_all=True):
    blind=Path(blind)
    requests={}
    for f in sorted(blind.glob('requests_*.jsonl')):
        for r in map(json.loads,f.read_text().splitlines()):
            assert r['key'] not in requests
            requests[r['key']]=r
    labels={}
    hashes={}
    for f in sorted(blind.glob('labels_*.jsonl')):
        hashes[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
        for r in map(json.loads,f.read_text().splitlines()):
            k=r['key']; assert k in requests and k not in labels
            assert r['status'] in ['single','multiple','none','unclear']
            assert r['quote_source'] in ['reply','thinking','raw_output']
            src=requests[k]
            if r['quote_source']=='reply': text=src['reply']
            else: text=json.loads((blind/src['trace']).read_text())[r['quote_source']]
            assert isinstance(r['answer_quote'],str) and r['answer_quote'] in text, (k,'quote not a contiguous source substring')
            if r['status']=='single':
                assert isinstance(r['semantic_answer'],str) and r['semantic_answer'].strip()
                assert r['answer_quote']
            assert r['explanation_presence'] in ['present','absent','unclear']
            labels[k]=r
    if require_all: assert len(labels)==len(requests)==600,(len(labels),len(requests))
    return requests,labels,hashes

if __name__=='__main__':
    D=Path(__file__).resolve().parent/'qwen_annotation_20261102'
    requests,labels,hashes=validate(D/'blind')
    report=dict(received=len(labels),expected=len(requests),invalid=0,missing=0,duplicate=0,contiguous_quotes=True,
                labels_sha256=hashes,judge='/root/pathvqa_luna_continuation',model='gpt-6-luna',reasoning_effort='max')
    (D/'validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))
