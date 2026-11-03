import hashlib,json,re
from pathlib import Path
B=Path(__file__).resolve().parent;x=json.loads((B/'catalog59.json').read_text());pd=Path('work/pendulum_server51_20261102');flags=json.loads((pd/'three_model_results_20261103/summary.json').read_text())['reference_dispute_sensitivity']['excluded_ids']
def score(ds,iid):return hashlib.sha256(f'entropy150_seed42|{ds}|{iid}'.encode()).hexdigest()
def pick(ds,items,n):return sorted(sorted(items,key=lambda r:score(ds,r['id']))[:n],key=lambda r:r['id'])
cq=[];exc=[]
for r in x['ClockQA']['items']:
 nums=[int(n) for n in re.findall(r'\d+',str(r['claim']))]
 if len(nums)==3 and 0<=nums[0]<=23 and all(0<=t<60 for t in nums[1:]):cq.append(r)
 else:exc.append(r['id'])
pdm={(r['id'],r['condition']):r for r in x['PENDULUM']['records']}
pditems=[r for r in x['PENDULUM']['items'] if r['domain'] in ['re_used','self_captured'] and r['id'] not in flags and all(pdm[r['id'],c]['model']=='/mnt/sdb/linxin/ModelList/Qwen3.8-27B' and pdm[r['id'],c]['has_integer_ids'] for c in ['neutral','belief','belief_right'])]
selected={'ClockQA':pick('ClockQA',cq,50),'PathVQA':pick('PathVQA',x['PathVQA']['items'],50),'PENDULUM':sum((pick('PENDULUM_'+dom,[r for r in pditems if r['domain']==dom],25) for dom in ['re_used','self_captured']),[])}
assert all(len(v)==50 for v in selected.values())
out=dict(seed=42,questions_per_dataset=50,unique_questions=150,models=['qwen38','gemma4'],conditions=['belief','belief_right'],expected_trajectories=600,matching='same50 questions per dataset for both models; no filtering on outcome',selection_rule='SHA256 deterministic rank entropy150_seed42|dataset|id; select50 ClockQA valid wrong-claim times,50 PathVQA,25 each real-photo PENDULUM domain; exclude known PENDULUM reference disputes and Qwen records from different weight paths; no entropy/outcome based selection',excluded_clock_invalid_times=exc,excluded_pendulum_reference_disputes=flags,ids={ds:[r['id'] for r in v] for ds,v in selected.items()},items=selected)
(B/'selection.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps(out['ids'],ensure_ascii=False));
plan=dict(algorithm='For each condition trajectory y, compare p_U(.|I,Q,U,y[:k]) with p_0(.|I,Q,y[:k]), k=0..T-1. No independent neutral generation.',zero_based_predictor_index=True,output_step='k+1',entropy='full vocabulary native logits, before sampling filters, nats',primary_analysis='ordinary thinking tokens after output step64; keep opening separate',secondary_analysis='exclude semantic opening task-restatement sentences separately; save raw curves and report distinction from fixed64 cutoff',signals=['H_pressure','H_neutral','delta_H','JSD','delta_logp','rank_pressure','rank_neutral','top1_pressure','top1_neutral'],cohorts='each model existing judgments: neutral correct->wrong,neutral correct->correct,neutral wrong,unresolved; no filtering on right-claim outcome',group_weighting='question-level means; datasets/models reported separately; PathVQA repeated images not independent image samples',valley_units=['minimum raw thinking token after64','minimum16-token continuous thinking window after64'],checks=dict(cpu_tests=6,source_and_image_hashes=True,prompt_ids_exact=True,original_actual_top20_min=.98,formula_error_max=1e-4,prefix_jsd_max=1e-3,prefix_entropy_absdiff_max=.1),runtime='multi-GPU sharding only; original per-model config; device_map/max_memory placement may be set explicitly and recorded; no numerical/sampling/prompt/image/model changes; natural process exit')
(B/'analysis_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
