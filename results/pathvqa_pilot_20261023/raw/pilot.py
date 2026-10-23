"""Five items per method, three conditions each; one model; exit releases GPU."""
import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))
from preflight import configuration, sha, read


def state(status, **fields):
    value = dict(status=status, pid=os.getpid(), time_unix=time.time(), **fields)
    temp = ROOT / ('pilot_state.json.tmp.' + str(os.getpid()))
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temp, ROOT / 'pilot_state.json')


def main():
    from symexp.backends import make_backend
    from symexp.methods import Context, METHODS, final_key
    import torch
    req, cfg = configuration()
    receipt = json.loads((ROOT / 'preflight_receipt.json').read_text())
    assert receipt['passed'] and receipt['request_sha256'] == sha(ROOT/'request.json')
    assert receipt['conditions_sha256'] == sha(ROOT/'data/conditions.jsonl')
    for name, digest in receipt['source_hashes'].items():
        assert sha(ROOT/name) == digest
    assert os.environ.get('CUDA_VISIBLE_DEVICES') and torch.cuda.device_count() == 1
    gpu = os.environ['CUDA_VISIBLE_DEVICES']
    import subprocess
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader,nounits'], text=True, timeout=20)
    assert not any(line.split(',')[0].strip()==gpu for line in apps.splitlines()), 'GPU became busy before model load'
    # Initialize the model's own CUDA context before loading weights; this
    # context exits with the experiment and is not an independent GPU holder.
    torch.cuda.init()
    state('loading_model', gpu_uuid=gpu, planned_each=15)
    backend = make_backend(cfg)
    assert backend.close_id is not None, 'Single-pass closure is required; do not silently fall back'
    items = [r for r in read(ROOT/'data/conditions.jsonl') if r['id'] in req['pilot_ids']]
    assert len(items) == 15
    completed = {}
    for method in ['srt', 'ours']:
        run_dir = ROOT / 'runs' / ('try_' + method)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir/'config.json').write_text(json.dumps(dict(method=method, data=str(ROOT/'data/conditions.jsonl'),
            seeds=[42], ids=req['pilot_ids'], config=cfg.to_dict(), started=time.time(), phase='pilot'), indent=2)+'\n')
        ctx = Context(cfg, backend, str(run_dir), str(ROOT/'data'), method)
        for it in items:
            key = final_key(it, 42)
            saved = ctx.final.get(key)
            if saved:
                assert not saved.get('error'), 'Prior error must be reviewed before resuming'
                continue
            state('generating', method=method, key=key, completed=completed, gpu_uuid=gpu)
            row = METHODS[method](ctx, it, 42)
            ctx.final.put(row)
            completed[method] = len(ctx.final.values())
            print(json.dumps(dict(method=method, id=it['id'], condition=it['condition'], completed=completed[method]), ensure_ascii=False), flush=True)
        assert len(ctx.final.values()) == 15
    state('pilot_complete_review_required', completed=completed, planned_each=15,
          gpu_uuid=gpu, process_will_exit=True, formal_experiment_started=False)


if __name__ == '__main__':
    with (ROOT/'pilot.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (ROOT/'pilot_state.json').exists():
            assert json.loads((ROOT/'pilot_state.json').read_text())['status'] != 'pilot_complete_review_required', 'Pilot already completed'
        try:
            main()
        except BaseException as exc:
            state('failed', error=repr(exc), checkpoints_retained=True, process_will_exit=True)
            raise
