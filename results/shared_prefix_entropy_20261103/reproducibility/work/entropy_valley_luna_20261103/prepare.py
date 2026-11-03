import csv,hashlib,json,random
from pathlib import Path
B=Path(__file__).resolve().parent;O=B.parent/'shared_prefix_entropy_150x2_20261103/results';V=O/'local_valleys';events=json.loads((V/'candidates.json').read_text());assert len(events)==615
if (B/'private_keymap.json').exists():raise RuntimeError('Already prepared; no overwrite')
cache={};requests=[];private={}
for i,e in enumerate(events):
 mk=(e['model'],e['key'])
 if mk not in cache:cache[mk]=list(csv.DictReader((O/e['model']/'records'/(e['key']+'.csv')).open()))
 rows=cache[mk];idx=e['step']-1;lo=max(0,idx-96);hi=min(len(rows),idx+65)
 before=''.join(r['token_piece'] for r in rows[lo:idx]);focus=rows[idx]['token_piece'];after=''.join(r['token_piece'] for r in rows[idx+1:hi]);assert focus==e['token']
 key='val_'+hashlib.sha256(f'entropy_valleys_blind_20261103:{i}:{e["source_sha256"]}:{e["step"]}'.encode()).hexdigest()[:22]
 req=dict(key=key,before=before,token=focus,after=after);requests.append(req)
 private[key]=dict(**e,context_first_output_step=lo+1,context_last_output_step=hi,context_char_before=len(before),context_sha256=hashlib.sha256((before+focus+after).encode()).hexdigest())
random.Random(20261103).shuffle(requests)
for b,a in enumerate(range(0,len(requests),25),1):
 (B/'blind'/f'requests_{b:02d}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in requests[a:a+25]))
(B/'private_keymap.json').write_text(json.dumps(private,ensure_ascii=False,indent=2));(B/'coverage_600.json').write_bytes((V/'trajectories.json').read_bytes());(B/'valley_detection_summary.json').write_bytes((V/'summary.json').read_bytes());(B/'batch_state.json').write_text(json.dumps(dict(phase='prepared',total_trajectories_scanned=600,candidate_valleys=615,batches=25,annotation_order='valley first, then semantic annotation; no event-first selection',agent='/root/pathvqa_luna_continuation',model='gpt-6-luna',reasoning_effort='max'),indent=2))
print('Prepared615blind valley windows in25batches; model/dataset/outcome/depth hidden')
