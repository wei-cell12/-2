import base64,json,shlex,subprocess,sys
from pathlib import Path
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B.parent/'server_connection'));from transport import command
files={n:base64.b64encode((B/n).read_bytes()).decode() for n in ['plot_results.py','plot_payload.json']}
s='''import base64,json,subprocess,sys
from pathlib import Path
W=Path('/mnt/sdb/linxin/shared-prefix-entropy-150x2-20261103/viewer_plots');W.mkdir(exist_ok=True)
for n,b in json.load(sys.stdin).items():(W/n).write_bytes(base64.b64decode(b))
r=subprocess.run([sys.executable,str(W/'plot_results.py')],capture_output=True,text=True)
if r.returncode:print(r.stdout+r.stderr);raise SystemExit(r.returncode)
print(json.dumps(dict(log=r.stdout,images={p.name:base64.b64encode(p.read_bytes()).decode() for p in W.glob('*.png')})))
'''
r=subprocess.run(command('python3 -c '+shlex.quote(s)),input=json.dumps(files).encode(),stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
if r.returncode:print(r.stdout.decode());raise SystemExit(r.returncode)
x=json.loads(r.stdout);O=B/'results'
for n,b in x['images'].items():(O/n).write_bytes(base64.b64decode(b))
(B/'render_receipt.json').write_text(json.dumps(dict(log=x['log'],images=list(x['images'])),indent=2));print(x['log'])
