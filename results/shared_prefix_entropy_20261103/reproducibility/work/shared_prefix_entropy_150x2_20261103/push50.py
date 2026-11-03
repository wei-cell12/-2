import json,base64,subprocess,shlex,sys,hashlib
from pathlib import Path
B=Path(__file__).resolve().parent;x=json.loads((B/'export51.json').read_text());data=base64.b64decode(x['package_b64']);assert hashlib.sha256(data).hexdigest()==x['package_sha256']
(B/'gemma_source_manifest.json').write_text(json.dumps(x['manifest'],ensure_ascii=False,indent=2));(B/'gemma_sources.tar.gz').write_bytes(data)
sys.path.insert(0,str(B.parent/'server_connection_59_77_5_50'));from transport import command
script='''import sys,hashlib,json,tarfile,io
from pathlib import Path
W=Path('/mnt/sdg/linxin/shared-prefix-entropy-150x2-20261103');W.mkdir(exist_ok=True)
p=W/'gemma_sources.tar.gz'
if p.exists():raise RuntimeError('Source bundle already installed; inspect instead of overwriting')
b=sys.stdin.buffer.read();assert hashlib.sha256(b).hexdigest()==EXPECTED
p.write_bytes(b)
D=W/'inputs';D.mkdir()
with tarfile.open(fileobj=io.BytesIO(b),mode='r:gz') as t:
 for m in t.getmembers():
  q=Path(m.name);assert not q.is_absolute() and '..' not in q.parts and m.isfile(),m.name
 t.extractall(D)
print(json.dumps(dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),record_files=len(list((D/'records').rglob('*.json'))),image_files=len(list((D/'images').rglob('*.*'))))))
'''.replace('EXPECTED',repr(x['package_sha256']))
r=subprocess.run(command('python3 -c '+shlex.quote(script)),input=data,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);print(r.stdout.decode());(B/'push50.log').write_bytes(r.stdout);raise SystemExit(r.returncode)
