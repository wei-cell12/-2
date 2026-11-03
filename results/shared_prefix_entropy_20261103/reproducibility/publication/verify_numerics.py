import json,csv,math,hashlib,collections
from pathlib import Path
B=Path(__file__).resolve().parent;W=(B.parent/'work/entropy_valley_behavior_20261103') if (B.parent/'work').is_dir() else B.parent;R=(B.parent.parent) if (B.parent/'work').is_dir() else W.parent/'shared_prefix_entropy_150x2_20261103/results'
all_saved=json.loads((W/'valleys_all.json').read_text());ix={(r['model'],r['key'],r['step']):r for r in all_saved};assert len(ix)==len(all_saved)
seen=set();counts=collections.Counter();modelsets={};row_total=0;max_identity_error=0.;min_jsd=1.;max_jsd=0.;max_gate_jsd=0.;max_gate_entropy=0.;min_top20=1.;max_formula=0.;methods=collections.Counter();records=[]
for model in ['qwen38','gemma4']:
 cases=json.loads((R/model/'analysis/cases.json').read_text());assert len(cases)==300;modelsets[model]={(c['dataset'],c['id'],c['condition'],c['image_sha256']) for c in cases};assert len(modelsets[model])==300
 for c in cases:
  p=R/model/'records'/(c['key']+'.csv');j=json.loads(p.with_suffix('.json').read_text());rs=list(csv.DictReader(p.open()));assert len(rs)==c['tokens']==j['tokens'];assert c['image_sha256']==j['image_sha256'];assert c['source_sha256']==j['source_sha256'];assert j['original_pressure_top20_fraction']>=.98
  min_top20=min(min_top20,j['original_pressure_top20_fraction']);max_formula=max(max_formula,j['formula_max_absdiff']);assert j['formula_max_absdiff']<1e-4
  for x in j['prefix_checks']:
   assert x['JSD']<.001 and x['entropy_absdiff']<.1,(model,c['key'],x)
   max_gate_jsd=max(max_gate_jsd,x['JSD']);max_gate_entropy=max(max_gate_entropy,x['entropy_absdiff'])
  methods[model+':'+j.get('forward_method','reused_validated_full_sequence' if j.get('new_forward') is False else 'parallel_full_sequence')]+=1
  valid=[]
  for i,r in enumerate(rs):
   assert int(r['step'])==i+1 and int(r['prefix_output_tokens'])==i
   hu,h0,d,js=map(float,[r['H_pressure'],r['H_neutral'],r['delta_H'],r['JSD']]);assert all(math.isfinite(x) for x in [hu,h0,d,js]);assert hu>=-1e-5 and h0>=-1e-5 and -1e-5<=js<=math.log(2)+1e-5
   err=abs((hu-h0)-d);max_identity_error=max(max_identity_error,err);assert err<1e-5;min_jsd=min(min_jsd,js);max_jsd=max(max_jsd,js)
   valid.append(i>=c['opening_cutoff'] and r['phase']=='think' and r['forced_by_original_budget']=='False')
  k=0;local=[]
  while k<len(rs):
   if not valid[k] or float(rs[k]['delta_H'])>=0:k+=1;continue
   lo=k
   while k<len(rs) and valid[k] and float(rs[k]['delta_H'])<0:k+=1
   hi=k;pos=min(range(lo,hi),key=lambda z:float(rs[z]['delta_H']));depth=-float(rs[pos]['delta_H']);band='deep' if depth>=1 else 'medium' if depth>=.25 else 'shallow' if depth>=.05 else 'tiny';key=(model,c['key'],pos+1);v=ix[key];assert v['run_start']==lo+1 and v['run_end']==hi and v['band']==band and v['delta_H']==float(rs[pos]['delta_H']);seen.add(key);counts[band]+=1;local.append(v)
  for rank,v in enumerate(sorted(local,key=lambda z:z['depth'],reverse=True),1):assert v['depth_rank']==rank
  row_total+=len(rs);records.append(dict(model=model,key=c['key'],dataset=c['dataset'],condition=c['condition'],tokens=len(rs),csv_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),json_sha256=hashlib.sha256(p.with_suffix('.json').read_bytes()).hexdigest()))
assert seen==set(ix);assert modelsets['qwen38']==modelsets['gemma4'];assert row_total==536646;assert counts==dict(deep=615,medium=5337,shallow=29284,tiny=52105)
old=json.loads((R/'local_valleys/luna_generalization/events.json').read_text());assert len(old)==615;secondary=[e for e in old if ix[(e['model'],e['key'],e['step'])]['depth_rank']>1];assert len(secondary)==255
user=lambda e:e['judgment']['activity'] in ['user_restatement','user_evaluation'];visual=lambda s:s in ['visual_observation','visual_inference'];summary=dict(ok=True,questions=150,trajectories=len(records),same_questions_conditions_and_image_hashes=True,output_token_rows=row_total,max_entropy_identity_error=max_identity_error,min_jsd_raw=min_jsd,max_jsd_raw=max_jsd,negative_jsd_roundoff_tolerance=1e-5,actual_top20_min=min_top20,accepted_prefix_max_jsd=max_gate_jsd,accepted_prefix_max_entropy_difference=max_gate_entropy,max_formula_error=max_formula,forward_methods=dict(methods),valley_counts=dict(counts),reconstructed_valleys_equal_saved=True,secondary_deep=dict(n=len(secondary),explicit_uncertainty_before=sum(e['judgment']['explicit_uncertainty_before']=='yes' for e in secondary),explicit_uncertainty_then_user=sum(user(e) and e['judgment']['explicit_uncertainty_before']=='yes' for e in secondary),visual_user_visual=sum(user(e) and visual(e['judgment']['activity_before']) and visual(e['judgment']['activity_after']) for e in secondary)),records=records)
(B/'numerical_confirmation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='records'},ensure_ascii=False,indent=2))
