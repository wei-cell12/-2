"""Finite PathVQA workflow using unchanged Gemma4 model2 generation functions."""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback

R = Path('/mnt/sdb/linxin/repo-gemma4')
G = Path('/mnt/sdb/linxin/gemma4-pv-20261030')
M = Path('/mnt/sdb/linxin/gemma4-pathvqa-compare-20261030')
sys.path.insert(0, str(R))
parser = argparse.ArgumentParser()
parser.add_argument('--shard', required=True)
args = parser.parse_args()
K, N = map(int, args.shard.split('/'))
STATE = {'pid': os.getpid(), 'shard': args.shard, 'phase': 'starting',
         'gpus': os.environ['CUDA_VISIBLE_DEVICES'].split(','),
         'started_at': time.time(), 'first_five': []}


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def save():
    write_json(M / f'job_state_{K}.json', STATE)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_free():
    for gpu in STATE['gpus']:
        r = subprocess.run(['nvidia-smi', '-i', gpu, '--query-gpu=memory.free,utilization.gpu',
                            '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
        free, util = map(int, r.stdout.strip().split(','))
        if free < 43000 or util > 10:
            raise RuntimeError('GPU unavailable ' + gpu + ': ' + r.stdout)
        r = subprocess.run(['nvidia-smi', '-i', gpu, '--query-compute-apps=pid',
                            '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
        if r.stdout.strip():
            raise RuntimeError('GPU occupied ' + gpu + ': ' + r.stdout)


def main():
    import fcntl
    lock = (M / f'worker_{K}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (M / f'worker_{K}.exit').exists():
        raise RuntimeError('Worker has already exited; no automatic retry')
    save()
    if (M / 'pytest.exit').read_text().strip() != '0':
        raise RuntimeError('Required tests did not pass')
    ready = json.loads((M / 'dataset_ready.json').read_text())
    if ready['items'] != 100:
        raise RuntimeError('Expected original complete 100-item dataset')
    spec = json.loads((M / 'run_spec.json').read_text())
    for name, expected in spec['source_sha256'].items():
        if sha(R / name) != expected:
            raise RuntimeError('Original code/config changed: ' + name)
    check_free()
    from symexp.config import Config
    from symexp.fp import model2
    from symexp.fp.conflict import _items
    from symexp.fp.pipeline import _shard
    cfg = Config.load(str(R / 'configs/gemma4_31b.yaml'))
    items = _items(str(G))
    pairs = [(i, c) for i in sorted(items) for c in model2.CONDS_ARM]
    tasks = [(a, i, c) for a in ['base', 'itsm'] for i in sorted(items)
             for c in (model2.CONDS_BASE if a == 'base' else model2.CONDS_ARM)]
    own = _shard(tasks, args.shard)
    if {(i, c) for a, i, c in own if a == 'itsm'} != set(_shard(pairs, args.shard)):
        raise RuntimeError('Neutralize and generation shard assignments do not align')
    STATE['expected_records'] = {a: sum(t[0] == a for t in own) for a in ['base', 'itsm']}
    STATE['phase'] = 'neutralize'
    save()
    model2.run_neutralize(cfg, str(G), args.shard)
    rewrite_checks = []
    for iid, cond in _shard(pairs, args.shard):
        rec = json.loads((G / 'neutralized' / f'{iid}__{cond}.json').read_text())
        rewrite_checks.append({'id': iid, 'condition': cond, 'rewritten': rec['neutral'],
                               'question_preserved_verbatim': items[iid]['question'] in rec['neutral'],
                               'exactly_original_question': rec['neutral'].strip() == items[iid]['messages']['neutral'].strip(),
                               'has_opinion_words': rec['has_opinion_words']})
    write_json(M / f'neutralize_pathvqa_checks_{K}.json',
               {'checks': rewrite_checks, 'clock_claim_time_check_not_applicable': True,
                'exact_original_question_count': sum(x['exactly_original_question'] for x in rewrite_checks)})
    # End the first model instance before starting native generation, as in the original two CLI stages.
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    native_generate = model2.generate2
    generated = 0

    def with_metadata_and_safety(*aa, **kw):
        nonlocal generated
        rec = native_generate(*aa, **kw)
        generated += 1
        rec['dataset'] = 'pathvqa'
        rec['semantic_judgment_status'] = 'pending'
        if 'outcome' in rec:
            rec['legacy_clock_outcome_ignore_for_pathvqa'] = rec.pop('outcome')
        if 'final_time' in rec:
            rec['legacy_clock_final_time_ignore_for_pathvqa'] = rec.pop('final_time')
        if generated <= 5:
            STATE['first_five'].append({'id': rec['id'], 'condition': rec['condition'],
                                        'thinking_empty': rec['thinking_empty'], 'n_out': rec['n_out'],
                                        'seconds': rec['seconds']})
            save()
            if rec['thinking_empty']:
                d = M / 'failures'
                d.mkdir(exist_ok=True)
                write_json(d / f'first_five_{K}_{generated}.json', rec)
                raise model2.Model2CheckFailed('Empty thinking among first five; original output preserved')
        if rec.get('neutral_sync') is not None and 'first_itsm_sync' not in STATE:
            STATE['first_itsm_sync'] = rec['neutral_sync']
            save()
        return rec

    model2.generate2 = with_metadata_and_safety
    STATE['phase'] = 'base_then_itsm'
    STATE['neutralize_finished_at'] = time.time()
    save()
    model2.run_arms(cfg, str(G), str(M), ['base', 'itsm'], [42], args.shard)
    STATE.update(phase='complete', generated=generated, finished_at=time.time())
    save()
    print('WORKER_COMPLETE', json.dumps(STATE, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    rc = 0
    try:
        main()
    except Exception as exc:
        rc = 1
        STATE.update(phase='failed', error=repr(exc), traceback=traceback.format_exc(), failed_at=time.time())
        save()
        failures = M / 'failures'
        failures.mkdir(exist_ok=True)
        write_json(failures / f'worker_{K}_{time.time_ns()}.json', STATE)
        traceback.print_exc()
    finally:
        (M / f'worker_{K}.exit').write_text(str(rc))
    raise SystemExit(rc)
