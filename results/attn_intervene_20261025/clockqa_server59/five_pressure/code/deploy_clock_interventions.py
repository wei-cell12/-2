"""Small-code deployment; same local weights, same inputs, same intervention math."""
import hashlib,json,subprocess,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;OLD=BASE.parent/'clockqa_attn_intervene_server59_20261025/qwen38_math_decode'
REMOTE='/mnt/sdi/linxin/proofwriter_runs/clockqa-scaleup-20261025/interventions'
def main():
 names=['workflow_common.py','generation_common.py','run_intervention.py','intervention.py','references.py','numerical_tests.py','clock_rules.py','sdpa_math_decode.py']
 files={n:(OLD/n).read_text() for n in names};expected={n:hashlib.sha256(s.encode()).hexdigest() for n,s in files.items()}
 old="def original(id,condition):return json.loads((ROOT/'records'/(id+'__'+condition)/'record.json').read_text())"
 new="""def original(id,condition):
    parent=Path(os.environ['CLOCKQA_INTERVENTION_ROOT'])
    item=next(r for r in json.loads((parent/'luna_approved_candidates.json').read_text())['items'] if r['id']==id)
    path=Path(item[condition+'_record']);assert sha(path)==item[condition+'_source_sha256']
    return json.loads(path.read_text())"""
 assert old in files['workflow_common.py'];files['workflow_common.py']=files['workflow_common.py'].replace(old,new)
 old="old=Path(config()['old_root'])/'attn'/f\"{item['id']}__neutral.npz\""
 new="old=Path(config()['prior_experiment_root'])/'neutral_replay'/f\"{item['question_id']}__neutral.npz\""
 assert old in files['run_intervention.py'];files['run_intervention.py']=files['run_intervention.py'].replace(old,new)
 files['run_intervention.py']=files['run_intervention.py'].replace("source_input_sha256=sha(ROOT/'records'/rec['record_id']/'record.json')","source_input_sha256=sha(item['pressure_record'])")
 for n in ['intervention_queue.py','run_clock_intervention_worker.py']:files[n]=(BASE/n).read_text()
 payload=dict(files=files,expected_original_code_sha256=expected)
 (BASE/'clock_intervention_deployment_payload.json').write_text(json.dumps(payload,ensure_ascii=False))
 remote=r'''
import hashlib,json,os,py_compile,shutil,subprocess,time
from pathlib import Path
payload=PAYLOAD;root=Path(REMOTE);scale=root.parent;old=Path(json.loads((scale/'config.json').read_text())['prior_experiment_root'])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (root/'launch_receipt.json').exists(),'Inspect previous launch before modifying deployed code'
for n,h in payload['expected_original_code_sha256'].items():assert sha(old/n)==h,(n,'Original code changed; inspect first')
selection=json.loads((root/'luna_approved_candidates.json').read_text());assert len(selection['items'])>=2
holder=json.loads((scale/'intervention_transition/temporary_reservation_state.json').read_text());assert holder['status']=='reserved' and len(holder['cards'])==4
data=Path('/mnt/sdb/linxin/proofwriter_runs/clockqa-five-pressure-20261025/interventions');data.mkdir(exist_ok=True)
# Uncompressed traces: at most96 selected heads×4610decode×13columns×4bytes, plus stage maps/references/replay and text.
required=60*96*4610*13*4+20*160*1024**2+2*1024**3
assert shutil.disk_usage(data).free>required,('Intervention disk budget',shutil.disk_usage(data).free,required)
for n,s in payload['files'].items():
 p=root/n
 if p.exists():assert p.read_text()==s,(n,'Preserve prior deployment')
 else:p.write_text(s)
 py_compile.compile(str(p),doraise=True)
c=json.loads((scale/'config.json').read_text());c.update(experiment_id='clockqa-server59-five-pressure-intervention',attention_intervention=True,
 pressure_only_generation_stage=False,transition_deadline_unix=holder['deadline'],same_model_between_batches=True,
 user_seamless_authorization='不要退卡呀 马上跟着干预实验',release_condition='after all approved intervention work and saving; bounded wait when no available reviewed work')
(root/'config.json').write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n');(root/'request.json').write_bytes((root/'config.json').read_bytes())
for n,target in [('images',scale/'images'),('source',old/'source'),('records',old/'records'),('catalog.jsonl',scale/'catalog.jsonl')]:
 p=root/n
 if not p.exists():p.symlink_to(target,target_is_directory=target.is_dir())
for n in ['references','reference_meta','neutral_replay','traces','output_tokens']:
 target=data/n;target.mkdir(exist_ok=True);p=root/n
 if not p.exists():p.symlink_to(target,target_is_directory=True)
ident=Path(c['model_identity_root'])/'model_identity.json';assert sha(ident)==c['model_identity_sha256']
identity=json.loads(ident.read_text())
for n,h in identity['metadata_sha256'].items():assert sha(Path(c['model_dir'])/n)==h
for w in identity['weight_files']:
 st=(Path(c['model_dir'])/w['name']).stat();assert (st.st_size,st.st_mtime_ns)==(w['bytes'],w['mtime_ns'])
for item in selection['items']:
 for kind in ['neutral','pressure']:
  p=Path(item[kind+'_record']);assert sha(p)==item[kind+'_source_sha256'];r=json.loads(p.read_text());assert r['image_sha256']==item['image_sha256'] and r['input_ids']
 assert sha(Path(item['image']))==item['image_sha256']
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',CLOCKQA_INTERVENTION_ROOT=str(root))
runtime='/home/linxin/miniconda3/envs/vllm_new/bin/python'
p=subprocess.run([runtime,str(root/'numerical_tests.py')],cwd=root,env=env,capture_output=True,text=True,timeout=60)
(root/'numerical_tests.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr
gate=dict(passed=True,config_sha256=sha(root/'config.json'),source_mapping_verified=len(selection['items']),model_identity_sha256=sha(ident),disk_free=shutil.disk_usage(data).free,disk_required=required,numerical_tests=json.loads(p.stdout.strip().splitlines()[-1]),no_weights_transferred=True,unchanged_algorithm_and_thresholds=True,time_unix=time.time())
(root/'cpu_preflight.json').write_text(json.dumps(gate,indent=2)+'\n')
queue=root/'queue.json'
if not queue.exists():queue.write_text(json.dumps(dict(interventions={},workers=4,failure=None,exited_workers={},created_unix=time.time()),indent=2)+'\n')
for i in range(4):
 wr=root/'workers'/str(i);wr.mkdir(parents=True,exist_ok=True)
 for n in payload['files']:shutil.copy2(root/n,wr/n)
 for n in ['config.json','request.json','cpu_preflight.json']:shutil.copy2(root/n,wr/n)
 for n in ['images','source','records','catalog.jsonl','references','reference_meta','neutral_replay','traces','output_tokens']:
  p=wr/n
  if not p.exists():p.symlink_to(root/n,target_is_directory=(root/n).is_dir())
 (wr/'live').mkdir(exist_ok=True)
print(json.dumps(dict(deployed=True,root=str(root),workers=4,approved=len(selection['items']),gate=gate)))
'''.replace('REMOTE',repr(REMOTE)).replace('PAYLOAD',repr(payload))
 sys.path.insert(0,str(BASE.parent/'server_connection'));from transport import command
 p=subprocess.run(command('python3 -'),input=remote,text=True,capture_output=True,timeout=100)
 (BASE/'clock_intervention_deployment_receipt.txt').write_text(p.stdout+p.stderr);print(p.stdout);print(p.stderr);raise SystemExit(p.returncode)
if __name__=='__main__':main()
