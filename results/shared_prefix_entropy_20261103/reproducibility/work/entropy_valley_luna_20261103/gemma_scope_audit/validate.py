import json
from pathlib import Path
B=Path(__file__).resolve().parent;D=B/'blind'
def validate(strict=False):
 req={r['key']:r for p in sorted(D.glob('requests_*.jsonl')) for r in map(json.loads,p.read_text().splitlines())};labels={};errors=[]
 for p in sorted(D.glob('labels_*.jsonl')):
  for l in p.read_text().splitlines():
   try:
    r=json.loads(l);assert r['key'] not in labels,'duplicate key';labels[r['key']]=r
   except Exception as e:errors.append(str(e))
 for p in sorted(D.glob('corrections_*.jsonl')):
  for l in p.read_text().splitlines():
   try:r=json.loads(l);assert r['key'] in labels;labels[r['key']]=r
   except Exception as e:errors.append(str(e))
 for k,r in labels.items():
  try:
   q=req[k];context=q['before']+q['token']+q['after'];assert r['focal_role'] in ['step_heading','response_plan','substantive_reasoning','input_interpretation','separator','other','unclear']
   assert r['user_processing_kind'] in ['restate','check_or_compare','response_to_claim_plan','interpret_claim_as_hint_or_example','none','unclear']
   for f in ['user_claim_processing','output_format_processing','visual_processing']:assert r[f] in ['yes','no','unclear']
   assert r['evidence_quote'] and r['evidence_quote'] in context
   z=r.get('message_quote');assert z is None or z and z in q['message']
   if r['user_claim_processing']!='yes':assert z is None
   assert r['note']
  except Exception as e:errors.append(dict(key=k,error=str(e)))
 res=dict(received=len(labels),expected=152,missing=len(set(req)-set(labels)),errors=errors,complete=len(labels)==152 and not errors)
 if strict:assert res['complete'],res
 return req,labels,res
if __name__=='__main__':
 _,_,r=validate();(B/'validation_current.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps(r,ensure_ascii=False))
 if r['errors']:raise SystemExit(1)
