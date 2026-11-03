import argparse,base64,hashlib,json,shlex,subprocess,sys
from pathlib import Path
B=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('server',choices=['59','50']);a=p.parse_args();folder='server_connection' if a.server=='59' else 'server_connection_59_77_5_50';sys.path.insert(0,str(B.parent/folder));from transport import command
files={n:base64.b64encode((B/n).read_bytes()).decode() for n in ['analyze_server.py','selection.json','analysis_plan.json','task.md']};labels=json.loads((B/'labels_snapshot.json').read_text());model='qwen38' if a.server=='59' else 'gemma4';files['labels_snapshot.json']=base64.b64encode(json.dumps([r for r in labels if r['model']==model],ensure_ascii=False,indent=2).encode()).decode()
root='/mnt/sdb/linxin/shared-prefix-entropy-150x2-20261103' if a.server=='59' else '/mnt/sdg/linxin/shared-prefix-entropy-150x2-20261103'
s='''import sys,json,base64,hashlib
from pathlib import Path
W=Path(ROOT);files=json.load(sys.stdin);out={}
for n,b in files.items():
 assert Path(n).name==n
 data=base64.b64decode(b);p=W/n
 if p.exists():assert p.read_bytes()==data,'Different existing artifact '+n
 else:p.write_bytes(data)
 out[n]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
print(json.dumps(out))
'''.replace('ROOT',repr(root))
r=subprocess.run(command('python3 -c '+shlex.quote(s)),input=json.dumps(files).encode(),stdout=subprocess.PIPE,stderr=subprocess.STDOUT);print(r.stdout.decode());(B/f'deploy_analysis{a.server}.log').write_bytes(r.stdout);raise SystemExit(r.returncode)
