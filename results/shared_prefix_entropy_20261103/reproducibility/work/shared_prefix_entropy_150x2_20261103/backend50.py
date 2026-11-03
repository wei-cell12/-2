from pathlib import Path
import json
R=Path('/mnt/sdg/linxin/repo-s2va-20261101')
for f in ['symexp/config.py','symexp/backends.py','symexp/fp/generate.py']:
 p=R/f;s=p.read_text();print('SOURCE',f)
 if f.endswith('generate.py'):
  a=s.index('def build_inputs');b=s.index('\ndef ',a+5);print(s[a:b])
 elif f.endswith('backends.py'):
  a=s.index('class HFBackend');print(s[a:a+11500])
 else:
  print('\n'.join(l for l in s.splitlines() if any(x in l for x in ['max_memory','device_map','class Config','extra','model_path'])))
for f in ['/mnt/sdg/ljh/models/gemma-4-31B-it/config.json','/mnt/sdg/linxin/repo-s2va-20261101/configs/gemma4_31b.yaml']:
 p=Path(f);print('FILE',f,p.exists());print(p.read_text() if p.exists() else '')
