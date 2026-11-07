"""Passive Gemma replay of the same pilot windows, including sliding layers.

Original model outputs, configuration, source replies, token windows, and gates
are unchanged. Visibility follows the actual mask supplied to native attention.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

CODE = Path('/home/linxin/trigger-regions-pilot-20261108/code')
PKG = Path('/home/linxin/trigger-regions-gemma4-pilot-20261108')
BASE = Path('/home/linxin/trigger-regions-gemma4-20261108')
ROOT = Path('/home/linxin/trigger-regions-gemma4-all-layers-20261108')
IDS = {'CQ-003', 'CQ-009', 'PV-013', 'PV-001', 'PD-re_used-0009', 'PD-re_used-0006'}

def atomic(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + '.tmp')
    q.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))
    q.replace(p)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class AllRecorder:
    def __init__(self, torch, rows, n_in, image_idx, masks, specs, chunk):
        self.torch, self.rows, self.n_in = torch, list(rows), n_in
        self.image_idx = image_idx
        self.masks = masks
        self.specs = specs
        self.layers = sorted(specs)
        self.chunk = chunk
        self.selfcheck_left = 0
        self.selfcheck_diffs = []
        self.passthrough = True
        self.rel_checked = {}
        self.index = {int(r): i for i, r in enumerate(rows)}
        self.rows_t = torch.tensor(rows, dtype=torch.long)
        self.data = {}
        self.visibility = {}

    def stages_for(self, q_pos):
        return self.torch.full_like(q_pos, -1)

    def observe_mask(self, module, query, key, mask):
        t = self.torch
        layer = int(module.layer_idx)
        ql, kl = query.shape[-2], key.shape[-2]
        abspos = np.array(self.rows) + self.n_in - 1
        local = abspos - (kl - ql)
        assert np.all((local >= 0) & (local < ql))
        qi = t.as_tensor(local, device=query.device)
        ii = self.image_idx.to(query.device)
        if mask is None:
            allowed = ii[None, :] <= t.as_tensor(abspos, device=query.device)[:, None]
            win = self.specs[layer]['window']
            if self.specs[layer]['kind'] == 'sliding' and win:
                allowed &= ii[None, :] > t.as_tensor(abspos, device=query.device)[:, None] - win
        else:
            m = mask[0, 0]
            if m.shape[-2] == 1:
                m = m.expand(ql, -1)
            assert m.shape[-2] == ql
            m = m.index_select(0, qi).index_select(-1, ii)
            if m.dtype == t.bool:
                allowed = m
            else:
                allowed = t.isfinite(m) & (m != t.finfo(m.dtype).min)
                # Additive masks must be native zero/sentinel masks, not an
                # undocumented score bias that could invalidate visibility.
                assert bool((m[allowed] == 0).all()), 'Unexpected additive mask values'
        self.visibility[layer] = allowed.cpu().numpy()

    def accumulate(self, layer, probs, stages, q_pos):
        t = self.torch
        k = (q_pos - (self.n_in - 1)).cpu()
        hit = t.isin(k, self.rows_t)
        if not bool(hit.any()):
            return
        local = hit.nonzero().flatten()
        p = probs[0].index_select(1, local.to(probs.device)).index_select(-1, self.image_idx.to(probs.device)).float()
        mass = p.sum(-1)
        conditional = p / mass.clamp_min(1e-12).unsqueeze(-1)
        masks = t.as_tensor(self.masks, device=p.device)
        shares = t.einsum('hrn,gn->hrg', conditional, masks)
        weighted = p.sum(0)
        dist = weighted / weighted.sum(-1, keepdim=True).clamp_min(1e-12)
        n, h, im = len(self.rows), p.shape[0], p.shape[-1]
        if layer not in self.data:
            self.data[layer] = {'mass': np.zeros((n, h), np.float32), 'shares': np.zeros((n, h, len(self.masks)), np.float32), 'dist': np.zeros((n, im), np.float32), 'filled': np.zeros(n, np.uint8)}
        d = self.data[layer]
        ri = np.array([self.index[int(x)] for x in k[local].tolist()])
        d['mass'][ri] = mass.transpose(0, 1).cpu().numpy()
        d['shares'][ri] = shares.permute(1, 0, 2).cpu().numpy()
        d['dist'][ri] = dist.cpu().numpy()
        d['filled'][ri] += 1

    def result(self):
        assert sorted(self.data) == self.layers == sorted(self.visibility)
        out = {'rows': np.array(self.rows), 'layer_ids': np.array(self.layers), 'masks': self.masks}
        for layer in self.layers:
            d, v = self.data[layer], self.visibility[layer]
            assert bool((d['filled'] == 1).all())
            assert np.isfinite(d['mass']).all() and np.isfinite(d['dist']).all()
            assert not np.any((v.sum(-1) == 0) & (d['mass'].max(-1) > 1e-10)), 'Invisible image has nonzero mass'
            assert np.max(np.abs(d['dist'][~v]), initial=0) < 1e-7
            for field in ['mass', 'shares', 'dist']:
                out[f'{field}_{layer}'] = d[field]
            out[f'visible_{layer}'] = v
        return out

def test_recorder():
    import torch
    torch.manual_seed(42)
    rows = [0, 1, 2]
    ii = torch.tensor([0, 1, 2])
    specs = {0: {'kind': 'global', 'window': None}, 1: {'kind': 'sliding', 'window': 4}}
    rec = AllRecorder(torch, rows, 5, ii, np.array([[1, 0, 0], [0, 1, 0]], np.float32), specs, 256)
    class Mod:
        layer_idx = 0
    q = torch.zeros(1, 2, 7, 4)
    for layer in [0, 1]:
        mod = Mod(); mod.layer_idx = layer
        allowed = torch.arange(7)[None, :] <= torch.arange(7)[:, None]
        if layer == 1:
            allowed &= torch.arange(7)[None, :] > torch.arange(7)[:, None] - 4
        mask = torch.where(allowed, 0., -float('inf'))[None, None]
        rec.observe_mask(mod, q, q, mask)
        probs = torch.softmax(mask.expand(1, 2, 7, 7), -1)
        rec.accumulate(layer, probs, None, torch.arange(7))
    z = rec.result()
    assert z['visible_0'].sum(-1).tolist() == [3, 3, 3]
    assert z['visible_1'].sum(-1).tolist() == [2, 1, 0]
    assert z['mass_1'][-1].tolist() == [0., 0.]
    assert z['dist_1'][-1].sum() == 0
    print('UNIT_PASS: mask visibility full/partial/none; zero-image rows; causal prediction alignment', flush=True)

def replay():
    sys.path.insert(0, str(CODE))
    from symexp.config import Config
    from symexp.fp.conflict import _engine
    from symexp.fp import trigger_regions as tr
    from symexp.fp.entropy import _batch
    from symexp.fp.generate import build_inputs
    from symexp.fp.reading_groups import image_grid
    from symexp.fp.sources import dial_geometry
    from symexp import attn
    import torch
    cfg = Config.load(str(PKG / 'config_runtime.yaml'))
    assert cfg.torch_dtype == 'bfloat16' and cfg.attn_implementation == 'sdpa'
    cfg_bytes = (PKG / 'config_runtime.yaml').read_bytes()
    (ROOT / 'config_used.yaml').write_bytes(cfg_bytes)
    # Small finite CUDA health check before allocating the model.
    a = torch.ones((16, 16), device='cuda')
    assert float((a @ a)[0, 0]) == 16
    torch.cuda.synchronize()
    del a
    torch.cuda.empty_cache()
    eng = _engine(cfg)
    backend = eng.backend
    specs = {}
    for _, m in attn.find_targets(backend.model):
        specs[int(m.layer_idx)] = dict(kind='sliding' if getattr(m, 'is_sliding', False) else 'global', window=getattr(m, 'sliding_window', None))
    assert len(specs) == 60 and sum(x['kind'] == 'sliding' for x in specs.values()) == 50
    atomic(ROOT / 'layer_specs.json', specs)
    print('LAYERS', json.dumps(specs), flush=True)
    attn.install(backend.model, verbose=False, only=None)
    original_recording = attn._recording_attention
    def recording(orig, module, query, key, value, mask, *args, **kwargs):
        active = attn._ACTIVE
        if active is not None and hasattr(active, 'observe_mask'):
            active.observe_mask(module, query, key, mask)
        return original_recording(orig, module, query, key, value, mask, *args, **kwargs)
    attn._recording_attention = recording
    ent = tr.load_entropy(str(PKG / 'entropy'), 'gemma4')
    tasks = sorted(k for k in ent if k[0] in IDS)
    assert len(tasks) == 12
    for i, key in enumerate(tasks):
        iid, cond = key
        name = iid + '__' + cond
        start = time.time()
        oldmeta = json.loads((BASE / 'trigger_regions' / (name + '.json')).read_text())
        e = ent[key]; job = e['job']; item = job['item']
        source = Path(tr.resolve_source(job)); rec = json.loads(source.read_text())
        image = tr.resolve_image(job, rec)
        dH, ok, tokens, n_think = tr.read_steps(e['csv'])
        events = copy.deepcopy(oldmeta['events'])
        rows = tr.window_rows([(x['kind'], x['t'], x['pre']) for x in events])
        with np.load(BASE / 'trigger_regions' / (name + '.npz')) as baseline:
            assert baseline['rows'].tolist() == rows
            masks = baseline['masks'].copy()
            baseline_mass = baseline['mass'].copy()
            baseline_shares = baseline['shares'].copy()
            baseline_dist = baseline['dist'].copy()
            baseline_layers = baseline['layer_ids'].tolist()
        _, inp = build_inputs(backend, image, item['messages'][cond])
        in_ids = inp['input_ids'][0].tolist()
        assert len(in_ids) == oldmeta['n_in'] == int(job['n_in'])
        if rec.get('input_ids') is not None:
            assert in_ids == list(map(int, rec['input_ids']))
        out_ids = list(map(int, rec['output_ids']))[:rows[-1] + 1]
        assert len(dH) == len(rec['output_ids'])
        for x in events:
            assert tokens[x['t']] == x['token']
        ii = torch.tensor([j for j, token in enumerate(in_ids) if token == attn.image_token_id(backend)])
        assert masks.shape[-1] == len(ii) == int(image_grid(backend, inp).prod())
        recorder = AllRecorder(torch, rows, len(in_ids), ii, masks, specs, cfg.attn_chunk)
        batch = _batch(backend, inp, out_ids)
        with torch.inference_mode():
            reference = backend.model(**batch, use_cache=False, return_dict=True, logits_to_keep=1).logits.detach().cpu() if i == 0 else None
            with attn.recording(recorder):
                logits = backend.model(**batch, use_cache=False, return_dict=True, logits_to_keep=1).logits.detach().cpu()
        z = recorder.result()
        no_op = float((logits.float()-reference.float()).abs().max()) if reference is not None else None
        if no_op is not None:
            assert no_op == 0, ('Observation changed model output', no_op)
        global_mass = np.stack([z[f'mass_{l}'] for l in baseline_layers], axis=1)
        global_shares = np.stack([z[f'shares_{l}'] for l in baseline_layers], axis=1)
        global_dist = np.mean([z[f'dist_{l}'] for l in baseline_layers], axis=0)
        checks = dict(mass_maxabs=float(np.max(np.abs(global_mass-baseline_mass))), shares_maxabs=float(np.max(np.abs(global_shares-baseline_shares))), dist_maxabs=float(np.max(np.abs(global_dist.astype(np.float16).astype(np.float32)-baseline_dist.astype(np.float32)))))
        # Existing dist is float16; compare with identical storage rounding.
        assert max(checks.values()) <= 1e-4, ('Global-only baseline changed', checks)
        target = ROOT / 'records' / (name + '.npz')
        np.savez_compressed(target, **z)
        meta = copy.deepcopy(oldmeta)
        meta.update(layer_ids=sorted(specs), source_full_record_sha256=sha(source), source_entropy_csv_sha256=sha(e['csv']), image_sha256=sha(image), output_check=recorder.rel_checked, native_logits_maxabs=no_op, global_baseline_check=checks, seconds=round(time.time()-start, 2), arrays_bytes=target.stat().st_size, arrays_sha256=sha(target), layer_visibility={str(l): dict(min_blocks=int(z[f'visible_{l}'].sum(-1).min()), max_blocks=int(z[f'visible_{l}'].sum(-1).max()), total_blocks=len(ii)) for l in specs})
        atomic(ROOT / 'records' / (name + '.json'), meta)
        print('DONE', name, 'rows', len(rows), 'seconds', meta['seconds'], 'global_baseline', checks, 'output_checks', recorder.rel_checked, flush=True)
    atomic(ROOT / 'replay_complete.json', {'records': 12, 'source_code_modified': False, 'global_layers': 10, 'sliding_layers': 50, 'native_outputs_preserved': True, 'config_sha256': sha(ROOT / 'config_used.yaml')})

if __name__ == '__main__':
    ap = argparse.ArgumentParser();ap.add_argument('phase', choices=['unit', 'replay']);a=ap.parse_args()
    if a.phase == 'unit':
        test_recorder()
    else:
        ROOT.mkdir(exist_ok=True);(ROOT/'records').mkdir(exist_ok=True)
        if (ROOT/'failure.json').exists() or list((ROOT/'records').glob('*.json')):
            raise SystemExit('Existing failure or outputs; no automatic retry or overwrite')
        try:
            replay()
        except BaseException as ex:
            atomic(ROOT/'failure.json', {'error':repr(ex),'traceback':traceback.format_exc()})
            raise
