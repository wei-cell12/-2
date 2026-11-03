"""Only schema/key/exact quote checks. No semantic classifications supplied."""
import hashlib,json
from pathlib import Path
B=Path(__file__).resolve().parent;D=B/'blind'
ACT={'visual_observation','visual_inference','user_restatement','user_evaluation','task_or_output_management','other','unclear'};TRI={'yes','no','unclear'};STANCE={'agree','reject','weighing','report_only','none','unclear'}
def validate(require_complete=False):
 requests={};rsha={}
 for p in sorted(D.glob('requests_*.jsonl')):
  rsha[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
  for l in p.read_text().splitlines():
   r=json.loads(l);assert r['key'] not in requests;requests[r['key']]=r
 labels={};errors=[];corrections=[]
 manifest=B/'request_manifest.json'
 if manifest.exists() and json.loads(manifest.read_text())!=rsha:errors.append(dict(error='Request hashes differ from frozen manifest'))
 for p in sorted(D.glob('labels_*.jsonl')):
  for n,l in enumerate(p.read_text().splitlines(),1):
   try:
    r=json.loads(l);assert r['key'] not in labels,'duplicate key';labels[r['key']]=r
   except Exception as e:errors.append(dict(file=p.name,line=n,error=str(e)))
 for p in sorted(D.glob('corrections_*.jsonl')):
  for n,l in enumerate(p.read_text().splitlines(),1):
   try:
    r=json.loads(l);assert r['key'] in labels,'correction key has no original';labels[r['key']]=r;corrections.append(dict(file=p.name,line=n,key=r['key']))
   except Exception as e:errors.append(dict(file=p.name,line=n,error=str(e)))
 for key,r in labels.items():
  try:
   q=requests[key];ctx=q['before']+q['token']+q['after']
   for k in ['activity','activity_before','activity_after']:assert r[k] in ACT,k
   for k in ['mentions_specific_user_claim','explicit_uncertainty_before','verification_before']:assert r[k] in TRI,k
   assert r['user_stance'] in STANCE
   assert isinstance(r['evidence_quote'],str) and r['evidence_quote'] and r['evidence_quote'] in ctx,'evidence_quote not exact substring'
   uq=r.get('uncertainty_quote')
   if r['explicit_uncertainty_before']=='yes':assert isinstance(uq,str) and uq and uq in q['before'],'uncertainty_quote not in before'
   else:assert uq is None,'uncertainty_quote must be null when not yes'
   en=r.get('user_entry_quote');assert en is None or isinstance(en,str) and en and en in ctx,'entry quote not exact substring'
   assert isinstance(r['note'],str) and r['note'].strip(),'missing note'
  except Exception as e:errors.append(dict(key=key,error=str(e)))
 missing=sorted(set(requests)-set(labels));res=dict(requests=len(requests),received=len(labels),missing=len(missing),format_or_quote_errors=errors,missing_keys=missing,corrections=corrections,request_sha256=rsha,complete=not errors and not missing)
 if require_complete:assert res['complete'],json.dumps(res,ensure_ascii=False)
 return requests,labels,res
if __name__=='__main__':
 _,_,r=validate();(B/'validation_current.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='missing_keys'},ensure_ascii=False))
 if r['format_or_quote_errors']:raise SystemExit(1)
