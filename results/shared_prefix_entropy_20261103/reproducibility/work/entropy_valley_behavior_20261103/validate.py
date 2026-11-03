import json,hashlib,argparse
from pathlib import Path
B=Path(__file__).resolve().parent; D=B/'blind'
BEH=set('visual_observation visual_inference alternative_comparison inconsistency_check user_claim_retrieval user_claim_check prompt_intent_resolution answer_commitment output_formulation language_link other unclear'.split())
TAGS=set('image user_claim question own_reasoning format language'.split())
def load():
 req={};lab={};errors=[];files={}
 for f in sorted(D.glob('requests_*.jsonl')):
  for line in f.read_text().splitlines():
   x=json.loads(line);assert x['key'] not in req;req[x['key']]=x
 for pat in ['labels_*.jsonl','corrections_*.jsonl']:
  seen=set()
  for f in sorted(D.glob(pat)):
   files[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
   for n,line in enumerate(f.read_text().splitlines(),1):
    try:x=json.loads(line)
    except Exception as e:errors.append([f.name,n,'json',str(e)]);continue
    k=x.get('key')
    if pat.startswith('labels') and k in seen:errors.append([f.name,n,'duplicate',k])
    seen.add(k)
    if k not in req:errors.append([f.name,n,'unknown',k]);continue
    lab[k]=x
 for k,x in lab.items():
  r=req[k];s=r['before']+r['token']+r['after'];focus=len(r['before']);end=focus+len(r['token'])
  for field in ['behavior','behavior_before','behavior_after']:
   if x.get(field) not in BEH:errors.append([k,field,x.get(field)])
  tags=x.get('object_tags')
  if not isinstance(tags,list) or any(t not in TAGS for t in tags) or len(tags)!=len(set(tags)):errors.append([k,'object_tags',tags])
  for field,opts in [('token_role','heading plan content quotation separator other unclear'),('uncertainty_before','yes no unclear'),('user_stance','agree reject weighing report_only none unclear')]:
   if x.get(field) not in opts.split():errors.append([k,field,x.get(field)])
  q=x.get('evidence_quote','');starts=[i for i in range(len(s)) if q and s.startswith(q,i)];overlap=any(i<end and i+len(q)>focus for i in starts)
  if not q or not overlap:errors.append([k,'evidence_quote_not_overlapping',q])
  uq=x.get('uncertainty_quote','')
  if x.get('uncertainty_before')=='yes' and (not uq or uq not in r['before']):errors.append([k,'uncertainty_quote',uq])
  if x.get('uncertainty_before')!='yes' and uq:errors.append([k,'unexpected_uncertainty_quote',uq])
  if not isinstance(x.get('note'),str) or not x['note']:errors.append([k,'note'])
 return req,lab,errors,files
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--partial',action='store_true');args=a.parse_args();req,lab,err,files=load();missing=sorted(set(req)-set(lab));z=dict(expected=len(req),received=len(lab),missing=len(missing),errors=err,files=files,ok=not err and (args.partial or not missing));(B/'validation.json').write_text(json.dumps(z,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in z.items() if k!='files'},ensure_ascii=False));raise SystemExit(0 if z['ok'] else 1)
