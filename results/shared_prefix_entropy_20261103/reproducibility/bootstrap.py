import gzip,json,hashlib
from pathlib import Path
B=Path(__file__).resolve().parent
for x in json.loads((B/'compressed_files.json').read_text()):
 p=B/x['path'];q=p.with_suffix('');b=gzip.decompress(p.read_bytes());assert hashlib.sha256(b).hexdigest()==x['original_sha256']
 if q.exists():assert hashlib.sha256(q.read_bytes()).hexdigest()==x['original_sha256']
 else:q.write_bytes(b)
p=B/'work/shared_prefix_entropy_150x2_20261103/results';target=Path('../../..')
if p.is_symlink():assert p.resolve()==B.parent.resolve()
elif p.exists():raise RuntimeError('Existing results path is not the expected symlink')
else:p.symlink_to(target,target_is_directory=True)
print('CPU reproduction inputs ready; no inference launched.')
