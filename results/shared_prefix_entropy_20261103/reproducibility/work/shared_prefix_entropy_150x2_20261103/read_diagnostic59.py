import json
from pathlib import Path
W=Path('/mnt/sdb/linxin/shared-prefix-entropy-150x2-20261103');p=W/'diagnostic_alignment.json';r=json.loads((W/'diagnostic_receipt.json').read_text());print(json.dumps(dict(pid=r['pid'],alive=Path('/proc',str(r['pid'])).exists(),result=json.loads(p.read_text()) if p.exists() else None,log_tail=(W/'diagnostic_alignment.log').read_text()[-3500:])))
