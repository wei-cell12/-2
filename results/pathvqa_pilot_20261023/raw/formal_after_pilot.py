"""Continue the authorized pilot into 150 x 3 conditions for both methods.

The same process holds a tiny CUDA context while our pilot finishes, then
replaces that temporary allocation with the real model. It never stops a
process, retries a failed experiment, or holds the card after completion.
"""
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))
from preflight import configuration, read, sha

# Complete the existing assistant thought, rather than introducing a new
# system instruction or changing the evidence/verification steps.
FINAL_RESPONSE_REMINDER = (
    'The independent analysis and all structured checks are complete. '
    'I will now answer the original image question in the user message directly, '
    'give the relevant visual evidence, and address the user\'s view if present. '
    'I will not repeat the internal identification, criteria, observation, or rejection-reason blocks.'
)


def atomic(path, value):
    temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temp.open('w') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def state(status, **fields):
    atomic(ROOT / 'formal_state.json', dict(status=status, pid=os.getpid(),
        time_unix=time.time(), gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),
        planned_each=450, release_gpu_on_exit=True, automatic_retry=False, **fields))


def alive_pilot(pid):
    p = Path('/proc') / str(pid) / 'cmdline'
    try:
        return str(ROOT / 'pilot.py').encode() in p.read_bytes()
    except FileNotFoundError:
        return False


def apps(gpu):
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], text=True, timeout=20)
    return {int(pid.strip()) for line in rows.splitlines() if ',' in line
            for uuid, pid in [line.split(',')[:2]] if uuid.strip() == gpu}


def provenance_gate(req, cfg):
    receipt = json.loads((ROOT / 'preflight_receipt.json').read_text())
    assert receipt['passed'] and receipt['request_sha256'] == sha(ROOT / 'request.json')
    assert receipt['conditions_sha256'] == sha(ROOT / 'data/conditions.jsonl')
    for name, digest in receipt['source_hashes'].items():
        assert sha(ROOT / name) == digest, 'Source changed: ' + name
    assert sha(req['model_identity_file']) == req['model_identity_sha256']
    assert receipt['config'] == cfg.to_dict(), 'Preflight configuration changed'
    data = read(ROOT / 'data/conditions.jsonl')
    assert len(data) == 450 and len({v['id'] for v in data}) == 150
    assert Counter(v['condition'] for v in data) == {
        'neutral': 150, 'user_wrong': 150, 'user_right': 150}
    return data


def pilot_gate(req, cfg, data):
    expected = {f"{v['id']}|{v['condition']}|{v['ptype']}|42": v
                for v in data if v['id'] in req['pilot_ids']}
    assert len(expected) == 15
    report = {}
    for method in ['srt', 'ours']:
        folder = ROOT / 'runs' / ('try_' + method)
        config = json.loads((folder / 'config.json').read_text())
        assert config['config'] == cfg.to_dict() and config['method'] == method
        rows = read(folder / 'final.jsonl')
        assert len(rows) == 15 and {r['key'] for r in rows} == set(expected)
        for r in rows:
            assert not r.get('error') and r['method'] == method
            v = expected[r['key']]
            for field in ['id', 'condition', 'ptype', 'question', 'message', 'image']:
                assert r[field] == v[field], (method, r['key'], field)
            assert r['gold'] == v['answer'] and r['seed'] == 42
            # A formatting flag never becomes a correctness judgment.
            assert str(r.get('reply', '')).strip() or str(r.get('reply_thinking', '')).strip(), 'No saved text'
            if method == 'ours':
                assert r['step2_parse_ok'] and r['prefix_chars'] > 0
                assert 'parse_failed' not in str(r.get('step3_decision'))
        if method == 'ours':
            step2 = read(folder / 'step2.jsonl')
            assert len(step2) == 5 and all(r['parse_ok'] for r in step2)
        report[method] = dict(completed=15, errors=0,
            final_sha256=sha(folder / 'final.jsonl'),
            format_flags=sum(r.get('srt_format_ok') is False for r in rows))
    atomic(ROOT / 'pilot_structure_gate.json', dict(passed=True, time_unix=time.time(),
        methods=report, checks='inputs, configuration, source hashes, saved outputs and method pipeline',
        content_correctness_not_assumed=True, format_anomalies_retained=True))


def seed_formal(method, cfg):
    source = ROOT / 'runs' / ('try_' + method)
    target = ROOT / 'runs' / method
    target.mkdir(exist_ok=True)
    records = {}
    names = ['final.jsonl', 'calls.jsonl'] if method == 'srt' else ['calls.jsonl', 'step2.jsonl', 'step3.jsonl']
    for name in names:
        src, dst = source / name, target / name
        if not src.exists():
            continue
        if dst.exists():
            assert dst.read_bytes() == src.read_bytes(), 'Formal run already contains different data; inspect first'
        else:
            with dst.open('xb') as f:
                f.write(src.read_bytes())
        records[name] = dict(source=str(src), sha256=sha(src))
    config = dict(method=method, data=str(ROOT / 'data/conditions.jsonl'), seeds=[42],
        config=cfg.to_dict(), started=time.time(), phase='formal', planned=450,
        reused_pilot_records=15 if method == 'srt' else 0, pilot_sources=records,
        original_pilot_outputs_preserved=True, plain_reused=True,
        final_response_reminder=FINAL_RESPONSE_REMINDER if method == 'ours' else None,
        runner_sha256=sha(ROOT / 'formal_after_pilot.py'))
    assert not (target / 'config.json').exists(), 'Formal run already configured; inspect instead of restarting'
    atomic(target / 'config.json', config)
    return target


def main():
    req, cfg = configuration()
    authorization = json.loads((ROOT / 'formal_authorization.json').read_text())
    assert authorization['formal_auto_start'] is True
    assert authorization['request_sha256'] == sha(ROOT / 'request.json')
    assert authorization['planned_each_method'] == 450
    gpu, pilot_pid = authorization['gpu_uuid'], authorization['pilot_pid']
    assert os.environ['CUDA_VISIBLE_DEVICES'] == gpu
    assert shutil.disk_usage(ROOT).free >= 5 * 1024**3, 'waiting_for_storage; no files deleted'
    data = provenance_gate(req, cfg)
    q = json.loads((ROOT / 'queue_state.json').read_text())
    assert q['gpu_uuid'] == gpu and q['pilot_pid'] == pilot_pid
    assert not (ROOT / 'formal_state.json').exists(), 'Already handed off: inspect instead of restarting'
    assert apps(gpu) <= {pilot_pid, os.getpid()}, 'Unexpected GPU process; no process modified'
    import torch
    torch.cuda.init()
    temporary = torch.empty(1024, dtype=torch.float32, device='cuda')
    torch.cuda.synchronize()
    state('waiting_for_pilot_same_gpu', pilot_pid=pilot_pid,
          temporary_hold_release='model loads after successful pilot; any failure exits and releases')
    deadline = time.monotonic() + 6 * 3600
    while alive_pilot(pilot_pid):
        saved = json.loads((ROOT / 'pilot_state.json').read_text())
        assert saved['status'] != 'failed', 'Pilot failed; no automatic retry'
        assert time.monotonic() < deadline, 'Pilot handoff wait exceeded six hours'
        time.sleep(3)
    saved = json.loads((ROOT / 'pilot_state.json').read_text())
    assert saved['status'] == 'pilot_complete_review_required', 'Pilot did not finish successfully'
    with (ROOT / 'pilot.lock').open('a') as pilot_lock:
        fcntl.flock(pilot_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pilot_gate(req, cfg, data)
        assert apps(gpu) <= {os.getpid()}, 'Unexpected GPU process; no process modified'
        assert shutil.disk_usage(ROOT).free >= 5 * 1024**3, 'waiting_for_storage; no files deleted'
        directories = {m: seed_formal(m, cfg) for m in ['srt', 'ours']}
        del temporary
        torch.cuda.empty_cache()
        state('loading_model_for_step4_check', completed={'srt': 15, 'ours': 0})
        from symexp.backends import make_backend
        from symexp.methods import Context, METHODS, final_key
        import symexp.methods as method_module
        original_build_prefix = method_module.build_prefix
        def completed_thought(s2, s3):
            prefix, conclusion, confidence = original_build_prefix(s2, s3)
            return prefix + '\n\n' + FINAL_RESPONSE_REMINDER, conclusion, confidence
        method_module.build_prefix = completed_thought
        backend = make_backend(cfg)
        assert backend.close_id is not None
        contexts = {m: Context(cfg, backend, str(d), str(ROOT / 'data'), m) for m, d in directories.items()}
        counts = {m: len(c.final.values()) for m, c in contexts.items()}
        # Reuse saved step2/step3. Regenerate only the final reply for the
        # original pilot's 15 conditions before admitting the full run.
        checked = []
        for item in data:
            if item['id'] not in req['pilot_ids']:
                continue
            state('checking_pilot_step4', key=final_key(item, 42), completed=counts)
            row = METHODS['ours'](contexts['ours'], item, 42)
            contexts['ours'].final.put(row)
            counts['ours'] = len(contexts['ours'].final.values())
            reply = row.get('reply', '').strip()
            assert reply and row['step4_gen_tokens'] > 0, 'No saved final reply; retain output for review'
            incomplete_repeat = (row['step4_gen_tokens'] >= cfg.answer_budget
                and reply.startswith('[IDENTIFICATION]') and '[CONCLUSION]' not in reply)
            assert not incomplete_repeat, 'Step4 still repeats internal blocks and truncates; inspect preserved output'
            checked.append(dict(key=row['key'], step4_gen_tokens=row['step4_gen_tokens'],
                                formal_config_sha256=sha(ROOT / 'runs/ours/config.json')))
        assert len(checked) == counts['ours'] == 15
        atomic(ROOT / 'pilot_runtime_gate.json', dict(passed=True, time_unix=time.time(),
            original_pilot_each=15, srt_pilot_reused=True, ours_step2_step3_reused=True,
            ours_final_rechecked=checked, final_response_reminder=FINAL_RESPONSE_REMINDER,
            content_correctness_not_assumed=True, format_anomalies_retained=True))
        for item in data:
            for method in ['srt', 'ours']:
                ctx = contexts[method]
                key = final_key(item, 42)
                saved = ctx.final.get(key)
                if saved:
                    assert not saved.get('error'), 'Prior error requires review'
                    continue
                state('generating', method=method, key=key, completed=counts,
                      pilot_runtime_gate_passed=True)
                try:
                    row = METHODS[method](ctx, item, 42)
                except Exception as exc:
                    ctx.final.put(dict(key=key, id=item['id'], method=method,
                        condition=item['condition'], ptype=item['ptype'], seed=42,
                        error=repr(exc), traceback=traceback.format_exc()))
                    raise
                ctx.final.put(row)
                counts[method] = len(ctx.final.values())
                print(json.dumps(dict(method=method, key=key, completed=counts,
                                      time_unix=time.time())), flush=True)
        assert counts == {'srt': 450, 'ours': 450}
        state('complete', completed=counts, plain_reused=True, process_will_exit=True)


if __name__ == '__main__':
    with (ROOT / 'formal.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            main()
        except BaseException as exc:
            state('failed', error=repr(exc), checkpoints_retained=True, process_will_exit=True)
            raise
