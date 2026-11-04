"""Finite, isolated replay of saved replies. Never changes the generation workers."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

B = Path('/mnt/sdb/linxin/pendulum-20261102')
P = B / 'expand4x50'
sys.path.insert(0, str(B / 'source'))
ap = argparse.ArgumentParser()
ap.add_argument('--model', choices=['gemma4', 'qwen38', 'qwen36_moe'], required=True)
ap.add_argument('--server', choices=['51', '59'], required=True)
ap.add_argument('--wait-worker', type=int)
args = ap.parse_args()
A = P / 'attention' / args.model
A.mkdir(parents=True, exist_ok=True)
(A / 'records').mkdir(exist_ok=True)
state = dict(phase='starting', pid=os.getpid(), model=args.model, server=args.server,
             gpus=os.environ['CUDA_VISIBLE_DEVICES'], expected=300, completed=0, started=time.time())


def write(path, obj):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    tmp.replace(path)


def save(**kw):
    state.update(**kw, updated=time.time())
    write(A / 'state.json', state)


def disk_check():
    u = shutil.disk_usage(A)
    if u.free < 20 * 2**30:
        raise RuntimeError('Less than 20 GiB free; preserve existing results and stop')
    return u.free


def worker_states():
    return [json.loads(p.read_text()) for p in sorted((P / args.model).glob('job_state_*.json'))]


def wait_gpu():
    save(phase='waiting_for_own_generator' if args.wait_worker is not None else 'checking_resources')
    while args.wait_worker is not None:
        own = P / args.model
        ex = own / f'worker_{args.wait_worker}.exit'
        if ex.exists():
            if ex.read_text().strip() != '0':
                raise RuntimeError('Own preceding generation worker failed')
            break
        for s in worker_states():
            if s.get('phase') == 'failed':
                raise RuntimeError('Generation failure; attention queue stopped')
        if time.time() - state['started'] > 48 * 3600:
            raise TimeoutError('48-hour queue deadline exceeded')
        time.sleep(30)
    while True:
        result = []
        for g in state['gpus'].split(','):
            r = subprocess.run(['nvidia-smi', '-i', g, '--query-gpu=memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
            used, free, util = map(int, r.stdout.strip().split(','))
            result.append(dict(gpu=g, used=used, free=free, util=util))
        if all(r['used'] < 100 and r['util'] <= 10 and r['free'] >= (45000 if args.server == '51' else 79000) for r in result):
            save(resource_check=result)
            return
        save(phase='waiting_for_free_gpu', resource_check=result)
        if time.time() - state['started'] > 48 * 3600:
            raise TimeoutError('48-hour queue deadline exceeded')
        time.sleep(30)


def reset_rope(model):
    inner = getattr(model, 'model', None)
    if inner is not None and hasattr(inner, 'rope_deltas'):
        inner.rope_deltas = None


def main():
    lock = (A / 'collector.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (A / 'exit').exists():
        raise RuntimeError('Collector already exited; no blind automatic restart')
    expected_gpu = {'gemma4': '1,2', 'qwen38': '0', 'qwen36_moe': '2'}[args.model]
    if state['gpus'] != expected_gpu:
        raise ValueError('Unexpected GPU placement')
    items = [json.loads(x) for x in (P / 'items.jsonl').read_text().splitlines()]
    items = [x for x in items if x['domain'] in ['re_used', 'self_captured']]
    if len(items) != 100 or any(sum(x['domain'] == d for x in items) != 50 for d in ['re_used', 'self_captured']):
        raise ValueError('Fixed 2 x 50 scope mismatch')
    save(free_bytes=disk_check(), domains=['re_used', 'self_captured'], items_sha256=hashlib.sha256((P / 'items.jsonl').read_bytes()).hexdigest())
    wait_gpu()
    import torch
    import transformers
    import numpy as np
    from PIL import Image
    from symexp.config import Config
    from symexp.backends import HFBackend
    from symexp import attn
    from symexp.fp.generate import build_inputs
    from symexp.fp.entropy import _batch
    from symexp.fp.reading_groups import image_grid, think_skip
    from spatial_recorder import SpatialRecorder
    cfg_path = B / 'configs' / (args.model + '.yaml')
    cfg = Config.load(str(cfg_path))
    save(phase='loading', torch=torch.__version__, transformers=transformers.__version__, config_sha256=hashlib.sha256(cfg_path.read_bytes()).hexdigest(), config=cfg.to_dict())
    cls = getattr(transformers, cfg.model_class)
    orig = cls.from_pretrained
    def limited(*a, **kw):
        kw['max_memory'] = {0: '40GiB', 1: '40GiB', 'cpu': '128GiB'} if args.server == '51' else {0: '74GiB', 'cpu': '128GiB'}
        return orig(*a, **kw)
    cls.from_pretrained = limited
    try:
        backend = HFBackend(cfg)
    finally:
        cls.from_pretrained = orig
    model = backend.model
    layers = attn.install(model, verbose=True)
    mods = {int(m.layer_idx): m for _, m in attn.find_targets(model)}
    windows = {i: int(getattr(m, 'sliding_window', 0) or 0) if getattr(m, 'is_sliding', False) else 0 for i, m in mods.items()}
    save(phase='collecting', loaded=time.time(), layers=layers, windows=windows,
         device_map=getattr(model, 'hf_device_map', None))
    tasks = [(item, cond) for item in items for cond in ['neutral', 'belief_right', 'belief']]
    first_check = False
    while tasks:
        pending = []
        progressed = False
        for item, cond in tasks:
            key = item['id'] + '__' + cond
            dst = A / 'records' / (key + '.json')
            src = P / args.model / 'records' / (key + '.json')
            if dst.exists():
                raise RuntimeError('Unexpected preexisting destination; inspect before resuming')
            if not src.exists():
                pending.append((item, cond))
                continue
            started = time.time()
            disk_check()
            raw = src.read_bytes()
            rec = json.loads(raw)
            image_path = Path(item['image'])
            imsha = hashlib.sha256(image_path.read_bytes()).hexdigest()
            if imsha != item['image_sha256'] or imsha != rec['image_sha256'] or rec['message'] != item['messages'][cond] or rec['seed'] != 42:
                raise ValueError('Source image/message/seed mismatch: ' + key)
            prompt, inputs = build_inputs(backend, str(image_path), rec['message'])
            ids = inputs['input_ids'][0].tolist()
            if ids != rec['input_ids'] or prompt != rec['prompt_text']:
                raise ValueError('Prompt replay not token-identical: ' + key)
            out = [int(x) for x in rec['output_ids']]
            if len(out) != rec['n_out'] or not out:
                raise ValueError('Output IDs mismatch or empty: ' + key)
            imgpos = np.flatnonzero(np.array(ids) == attn.image_token_id(backend))
            if len(imgpos) != rec['n_img'] or len(imgpos) == 0:
                raise ValueError('Image token count mismatch')
            close = rec.get('close_position')
            anomaly = None
            try:
                skip = think_skip(backend.tokenizer, cfg, out)
            except Exception as e:
                anomaly = str(e)
                skip = 0
            rr = SpatialRecorder(torch, len(ids), len(out), imgpos, skip, close, windows)
            batch = _batch(backend, inputs, out)
            with torch.inference_mode():
                reference = None
                if not first_check:
                    reset_rope(model)
                    reference = model(**batch, use_cache=False, return_dict=True, logits_to_keep=1).logits.detach().cpu()
                reset_rope(model)
                with attn.recording(rr):
                    logits = model(**batch, use_cache=False, return_dict=True, logits_to_keep=1).logits.detach().cpu()
                if reference is not None:
                    equal = bool(torch.equal(reference, logits))
                    maxabs = float((reference.float() - logits.float()).abs().max())
                    write(A / 'passive_forward_check.json', dict(key=key, logits_bitwise_equal=equal, max_abs=maxabs))
                    if not equal:
                        raise ValueError('Recorder changed original forward logits')
                    first_check = True
            arrays = rr.arrays(layers)
            grid = image_grid(backend, inputs)
            arrays['image_grid'] = grid
            for name in ['image_position_ids', 'image_grid_thw']:
                if name in inputs:
                    arrays['processor_' + name] = inputs[name].detach().cpu().numpy()
            arrpath = A / 'records' / (key + '.npz')
            tmppath = arrpath.with_suffix('.npz.tmp')
            with tmppath.open('wb') as f:
                np.savez_compressed(f, **arrays)
            tmppath.replace(arrpath)
            if hashlib.sha256(src.read_bytes()).hexdigest() != hashlib.sha256(raw).hexdigest():
                raise ValueError('Source changed during replay')
            with Image.open(image_path) as pil:
                image_size = list(pil.size)
            meta = dict(key=key, id=item['id'], condition=cond, domain=item['domain'], model=args.model,
                        source=str(src), source_sha256=hashlib.sha256(raw).hexdigest(),
                        image=str(image_path), image_sha256=imsha, original_image_wh=image_size,
                        n_in=len(ids), n_out=len(out), n_img=len(imgpos), image_grid=grid.tolist(),
                        output_ids_sha256=hashlib.sha256(json.dumps(out, separators=(',', ':')).encode()).hexdigest(),
                        arrays=str(arrpath), array_bytes=arrpath.stat().st_size,
                        arrays_sha256=hashlib.sha256(arrpath.read_bytes()).hexdigest(),
                        attention_layers=layers, sliding_windows=windows,
                        recorded_heads={i: rr.data[i]['image_mass'].shape[1] for i in layers},
                        stage_counts=next(iter(rr.data.values()))['stage_count'].tolist(),
                        stages=['prefill_end: output_index=0 predictor', 'think: output_index>=think_start and <close', 'answer: output_index>close'],
                        think_start=skip, close_position=close, marker_anomaly=anomaly,
                        early_groups='64 consecutive groups of 8 thought tokens, at most first 512; sums plus counts',
                        query_alignment='query_position=n_in-1+output_index; full teacher-forced original output',
                        visibility='number of image keys geometrically visible after causal/sliding mask, per output token and layer',
                        passive_original_output=True, recomputed_output_diagnostic=rr.rel_checked,
                        max_row_sum_error=rr.max_row_error, seconds=round(time.time()-started, 3), saved=time.time())
            write(dst, meta)
            state['completed'] += 1
            state['array_bytes'] = state.get('array_bytes', 0) + meta['array_bytes']
            save(last_key=key, last_seconds=meta['seconds'])
            print(f"SAVED {state['completed']}/300 {key} layers={len(layers)} image_tokens={len(imgpos)} bytes={meta['array_bytes']} seconds={meta['seconds']} row_error={rr.max_row_error:.3g} original_forward=True", flush=True)
            progressed = True
            del batch, logits, arrays, rr
        tasks = pending
        if tasks:
            ss = worker_states()
            if any(s.get('phase') == 'failed' for s in ss):
                raise RuntimeError('Generation worker failed; preserve replay and stop')
            if ss and all(s.get('phase') == 'complete' for s in ss):
                raise RuntimeError('Generation finished but requested source records missing')
            if time.time() - state['started'] > 48 * 3600:
                raise TimeoutError('48-hour deadline exceeded')
            save(phase='waiting_for_source_records', remaining=len(tasks))
            time.sleep(15 if progressed else 30)
            save(phase='collecting')
    save(phase='complete', finished=time.time())
    print('COMPLETE 300/300; model exits and releases GPUs', flush=True)


if __name__ == '__main__':
    rc = 0
    try:
        main()
    except Exception as e:
        rc = 1
        save(phase='failed', error=repr(e), traceback=traceback.format_exc(), failed=time.time())
        (A / 'failures').mkdir(exist_ok=True)
        write(A / 'failures' / f'{time.time_ns()}.json', state)
        traceback.print_exc()
    finally:
        (A / 'exit').write_text(str(rc))
    raise SystemExit(rc)
