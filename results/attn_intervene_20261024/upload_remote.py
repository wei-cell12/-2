"""Upload this new, independent task; no model weights or old results modified."""
import base64,io,json,subprocess,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'server_connection_59_77_5_50'))
from transport import command
REMOTE='/mnt/sdg/linxin/pathvqa-attn-intervene-20261024'

def main():
    files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['original_records.jsonl']]
    files.append(ROOT/'original_records.jsonl')
    bio=io.BytesIO()
    with tarfile.open(fileobj=bio,mode='w:gz') as tar:
        for p in files:tar.add(p,arcname=str(p.relative_to(ROOT)))
    payload=base64.b64encode(bio.getvalue()).decode()
    script=f'''import base64,io,json,pathlib,tarfile
r=pathlib.Path({REMOTE!r})
assert not r.exists(),"New experiment directory already exists; no overwrite"
r.mkdir()
b=base64.b64decode({payload!r})
with tarfile.open(fileobj=io.BytesIO(b),mode='r:gz') as t:
 for m in t.getmembers():
  p=r/m.name
  assert p.resolve().is_relative_to(r.resolve())
 t.extractall(r)
print(json.dumps(dict(ok=True,root=str(r),uploaded_bytes=len(b))))
'''
    p=subprocess.run(command(),input=script,text=True,capture_output=True,timeout=120)
    print(p.stdout);print(p.stderr);raise SystemExit(p.returncode)
if __name__=='__main__':main()
