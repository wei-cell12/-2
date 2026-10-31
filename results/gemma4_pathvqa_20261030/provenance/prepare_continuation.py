"""Retrieve only unjudged immutable records; retain the original blind instructions."""
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
OLD = ROOT / 'luna_label/snapshot_01'
NEW = ROOT / 'luna_label/snapshot_02'


def main():
    if NEW.exists():
        raise RuntimeError('Refusing to overwrite snapshot_02')
    old = json.loads((OLD / 'manifest.json').read_text())
    previous = {x['path']: x['sha256'] for x in old['records']}
    remote = '''
import base64,hashlib,io,json,tarfile,time
from pathlib import Path
previous=PREVIOUS
g=Path('/mnt/sdb/linxin/gemma4-pv-20261030')
m=Path('/mnt/sdb/linxin/gemma4-pathvqa-compare-20261030')
items=[json.loads(s) for s in (g/'items.jsonl').read_text().splitlines() if s.strip()]
manifest={'created_at':time.time(),'records':[],'items':items,'images':{},'existing_verified':0,'total_records':0}
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as tar:
 def add(name,data):
  info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,io.BytesIO(data))
 for arm,folder in [('base',g/'obs/records'),('itsm',m/'itsm/records')]:
  for p in sorted(folder.glob('*.json')):
   raw=p.read_bytes();sha=hashlib.sha256(raw).hexdigest();manifest['total_records']+=1
   if str(p) in previous:
    assert previous[str(p)]==sha,('Previously judged record changed',str(p))
    manifest['existing_verified']+=1
    continue
   x=json.loads(raw);slim={k:v for k,v in x.items() if k not in ['input_ids','output_ids']}
   slim['original_record_sha256']=sha;slim['original_record_path']=str(p)
   dest='records/'+arm+'/'+p.name
   add(dest,json.dumps(slim,ensure_ascii=False).encode())
   manifest['records'].append({'arm':arm,'file':dest,'id':x['id'],'condition':x['condition'],'sha256':sha,'path':str(p)})
 add('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2).encode())
print(base64.b64encode(buf.getvalue()).decode())
'''.replace('PREVIOUS', repr(previous))
    sys.path.insert(0, str(ROOT.parent / 'server_connection_59_77_5_51'))
    from transport import command
    p = subprocess.run(command(), input=remote, text=True, capture_output=True)
    if p.returncode:
        raise RuntimeError(p.stderr)
    raw = base64.b64decode(p.stdout.strip(), validate=True)
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as tar:
        members = tar.getmembers()
        assert all(x.isfile() and not x.name.startswith('/') and '..' not in Path(x.name).parts for x in members)
        NEW.mkdir()
        for entry in members:
            dest = NEW / entry.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(tar.extractfile(entry).read())
    manifest = json.loads((NEW / 'manifest.json').read_text())
    assert (manifest['total_records'], manifest['existing_verified'], len(manifest['records'])) == (500, 319, 181)
    judge = NEW / 'blind_judge'
    for sub in ['requests', 'traces', 'labels']:
        (judge / sub).mkdir(parents=True, exist_ok=False)
    (judge / 'instructions.md').write_bytes((OLD / 'blind_judge/instructions.md').read_bytes())
    queue, keymap = [], {}
    for entry in manifest['records']:
        x = json.loads((NEW / entry['file']).read_text())
        item = next(y for y in old['items'] if y['id'] == entry['id'])
        key = 'pv_' + hashlib.sha256((entry['path'] + entry['sha256']).encode()).hexdigest()[:24]
        image = OLD / old['images'][x['image_sha256']]['file']
        assert hashlib.sha256(image.read_bytes()).hexdigest() == x['image_sha256']
        req = {'key': key, 'question': item['question'], 'message': x['message'],
               'reply': x.get('answer', ''), 'image': str(image),
               'full_trace_path': str(judge / 'traces' / f'{key}.json')}
        trace = {'key': key, 'raw_output': x.get('raw_output', ''), 'thinking': x.get('thinking', ''),
                 'answer': x.get('answer', ''), 'budget': x.get('budget', {})}
        (judge / 'requests' / f'{key}.json').write_text(json.dumps(req, ensure_ascii=False, indent=2))
        (judge / 'traces' / f'{key}.json').write_text(json.dumps(trace, ensure_ascii=False, indent=2))
        queue.append({'key': key, 'request': str(judge / 'requests' / f'{key}.json'), 'image': str(image)})
        keymap[key] = entry
    queue.sort(key=lambda x: (x['image'], x['key']))
    (judge / 'queue.json').write_text(json.dumps(queue, ensure_ascii=False, indent=2))
    (NEW / 'private_keymap.json').write_text(json.dumps(keymap, ensure_ascii=False, indent=2))
    receipt = {'status': 'prepared_awaiting_judge_identity_authorization', 'requests': len(queue),
               'existing_records_sha_verified': 319, 'existing_labels_preserved': True,
               'instructions_sha256': hashlib.sha256((judge / 'instructions.md').read_bytes()).hexdigest(),
               'previous_agent_unavailable': '/root/clockqa_time_luna',
               'judge_started': False, 'gpu_used': False}
    (NEW / 'preparation_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
