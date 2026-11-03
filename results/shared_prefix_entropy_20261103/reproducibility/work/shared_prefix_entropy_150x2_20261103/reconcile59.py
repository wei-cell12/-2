import json
from pathlib import Path
W=Path('/mnt/sdb/linxin/shared-prefix-entropy-150x2-20261103');q=json.loads((W/'launch_receipt.json').read_text())
for i,r in enumerate(q):
 p=W/f"state_{r['rank']}.json"
 if not p.exists():continue
 s=json.loads(p.read_text());pid=s['pid']
 if pid!=r['pid'] and Path('/proc',str(pid)).exists():
  cmd=Path('/proc',str(pid),'cmdline').read_bytes().replace(b'\0',b' ').decode();assert str(W/'worker.py') in cmd and '--rank '+str(r['rank']) in cmd
  q[i].update(pid=pid,previous_pid=r['pid'],log=f"run_{r['rank']}_v2.log")
(W/'launch_receipt.json').write_text(json.dumps(q,indent=2))
print(json.dumps([dict(rank=r['rank'],receipt_pid=r['pid'],state=json.loads((W/f"state_{r['rank']}.json").read_text()) if (W/f"state_{r['rank']}.json").exists() else {}) for r in q]))
