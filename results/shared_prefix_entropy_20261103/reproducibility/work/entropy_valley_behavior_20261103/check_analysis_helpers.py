import ast, json, tempfile, importlib.util
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('valcheck',B/'validate.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
with tempfile.TemporaryDirectory() as p:
 v.D=Path(p);request=dict(key='test',message='Question',before='Not sure. ',token='user',after=' says yes.');label=dict(key='test',behavior='user_claim_retrieval',behavior_before='alternative_comparison',behavior_after='user_claim_check',object_tags=['user_claim'],token_role='content',uncertainty_before='yes',uncertainty_quote='Not sure.',user_stance='report_only',evidence_quote='user says yes.',note='Synthetic fixture')
 (v.D/'requests_01.jsonl').write_text(json.dumps(request)+'\n');(v.D/'labels_01.jsonl').write_text(json.dumps(label)+'\n');assert not v.load()[2]
 label['evidence_quote']='Not sure.';(v.D/'labels_01.jsonl').write_text(json.dumps(label)+'\n');assert any('evidence_quote_not_overlapping' in e for e in v.load()[2])
 label['evidence_quote']='user says yes.';label['uncertainty_quote']='says yes.';(v.D/'labels_01.jsonl').write_text(json.dumps(label)+'\n');assert any('uncertainty_quote' in e for e in v.load()[2])
mod=ast.parse((B/'finish.py').read_text());functions=[x for x in mod.body if isinstance(x,ast.FunctionDef) and x.name in ['auc','cv']];ctx={'np':np};exec(compile(ast.Module(body=functions,type_ignores=[]),'helpers','exec'),ctx)
y=np.array([0,0,1,1]);assert ctx['auc'](y,np.array([0,0,1,1]))==1.;assert ctx['auc'](y,np.zeros(4))==.5
r=np.random.default_rng(8);x=r.normal(size=150);y=(x>0).astype(int);ctx['X']=x[:,None];ctx['fold']=np.arange(len(x))%5;pred=ctx['cv'](y,[0]);assert np.isfinite(pred).all() and ctx['auc'](y,pred)>.98
meta=json.loads((B/'private_keymap.json').read_text());rows=list(meta.values());groups={}
for x in rows:
 key=x['dataset']+':'+x['image_sha256'];groups.setdefault(key,[]).append(x)
assert len(groups)==88
print('PASS: focus-overlap and uncertainty-quote validators; tied/perfect AUC; grouped-readout numeric synthetic sanity; 88 original-image groups.')
# Final trajectory length and normalized final position must not affect online features.
module=ast.parse((B/'finish.py').read_text());loops=[n for n in module.body if isinstance(n,ast.For) and any(isinstance(z,ast.Call) and isinstance(z.func,ast.Attribute) and isinstance(z.func.value,ast.Name) and z.func.value.id=='feat' and z.func.attr=='append' for z in ast.walk(n))];assert len(loops)==1
import math
sample=dict(delta_H=-1.2,JSD=.2,H_with_U=.3,H_without_U=1.5,prior16_H_pressure=.8,prior16_H_neutral=.9,step=200,dataset='synthetic',image_sha256='same',relative_position=.2,source_output_tokens=1000)
altered=dict(sample,relative_position=.02,source_output_tokens=10000);ns=dict(rows=[sample,altered],feat=[],groups=[],math=math);exec(compile(ast.Module(body=loops,type_ignores=[]),'feature_loop','exec'),ns);assert ns['feat'][0]==ns['feat'][1]
print('PASS: changing unknown final trajectory length/normalized position leaves trigger features unchanged.')
