"""CPU-only analysis of preserved arrays; no inference or source mutations."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np

B = Path('/mnt/sdb/linxin/pendulum-20261102')
P = B / 'expand4x50'
model = os.environ['ANALYSIS_MODEL']
A = P / 'attention' / model / 'records'
D = P / model / 'analysis_20261103'
D.mkdir(exist_ok=True)
sys.path.insert(0, str(B / 'source'))
from symexp.fp.reading_groups import _jsd
from symexp.fp.sources import _norm_img

conds = ['neutral', 'belief_right', 'belief']
items = [json.loads(s) for s in (P / 'items.jsonl').read_text().splitlines()]
items = [x for x in items if x['domain'] in ['re_used', 'self_captured']]
rows = []
maps = []

for ix, item in enumerate(items):
    iid = item['id']
    data, metas = {}, {}
    for c in conds:
        m = json.loads((A / (iid + '__' + c + '.json')).read_text())
        metas[c] = m
        assert m['id'] == iid and m['condition'] == c
        assert m['image_sha256'] == item['image_sha256']
        p = Path(m['arrays'])
        assert p.stat().st_size == m['array_bytes']
        with np.load(p) as z:
            layers = m['attention_layers']
            data[c] = dict(
                early=np.stack([z[f'L{l}_early_sum'] for l in layers], axis=1).astype(np.float64),
                stage=np.stack([z[f'L{l}_stage_sum'] for l in layers], axis=1).astype(np.float64),
                early_count=z[f'L{layers[0]}_early_count'].copy(),
                stage_count=z[f'L{layers[0]}_stage_count'].copy(),
                visible=np.stack([z[f'L{l}_visible_image_tokens'] for l in layers]),
                think_start=int(z['think_start']) if 'think_start' in z else None)
            for l in layers:
                assert np.array_equal(z[f'L{l}_early_count'], data[c]['early_count'])
                assert np.array_equal(z[f'L{l}_stage_count'], data[c]['stage_count'])
    assert len({tuple(metas[c]['image_grid']) for c in conds}) == 1
    assert len({metas[c]['n_img'] for c in conds}) == 1
    layers = metas['neutral']['attention_layers']
    sw = {int(k): v for k, v in metas['neutral']['sliding_windows'].items()}
    sets = {'all': list(range(len(layers))),
            'global': [j for j, l in enumerate(layers) if not sw[l]],
            'sliding': [j for j, l in enumerate(layers) if sw[l]]}
    info = dict(id=iid, domain=item['domain'], image_sha256=item['image_sha256'],
                image_grid=metas['neutral']['image_grid'],
                source_sha256={c: metas[c]['source_sha256'] for c in conds}, groups={})
    for group, js in sets.items():
        if not js:
            continue
        ls = [layers[j] for j in js]
        shape = data['neutral']['stage'][:, js].shape[1:]
        nh = shape[0] * shape[1]
        n_img = shape[-1]
        k = math.ceil(.2 * n_img)
        neutral_stage = data['neutral']['stage'][1, js]
        E = np.zeros(shape, bool)
        np.put_along_axis(E, np.argsort(-neutral_stage, axis=-1, kind='stable')[..., :k], True, axis=-1)
        legacy = np.argsort(-neutral_stage.sum(-1).ravel(), kind='stable')[:96]
        g = dict(layers=ls, total_heads=nh, image_tokens=n_img, E_tokens=k, windows={}, stages={})
        for W in [64, 128, 256, 512]:
            G = W // 8
            if any(not np.all(data[c]['early_count'][:G] == 8) for c in conds):
                g['windows'][str(W)] = dict(eligible=False, reason='one of three conditions has fewer than W thought-content tokens')
                continue
            vis = {}
            for c in conds:
                # Metadata stores exact thought bounds used by the recorder.
                start = metas[c]['think_start']
                v = data[c]['visible'][js, start:start + W]
                vis[c] = dict(min=int(v.min()), max=int(v.max()), complete=bool(np.all(v == n_img)))
            if not all(v['complete'] for v in vis.values()):
                g['windows'][str(W)] = dict(eligible=False, reason='image partially or fully outside sliding window', visibility=vis)
                continue
            raw = {c: data[c]['early'][:G, js] for c in conds}
            p0 = {c: _norm_img(raw[c].sum(0)) for c in conds}
            pe = {c: _norm_img(raw[c][::2].sum(0)) for c in conds}
            po = {c: _norm_img(raw[c][1::2].sum(0)) for c in conds}
            mass = {c: raw[c].sum((0, -1)) / W for c in conds}
            within = {c: _jsd(pe[c], po[c]) for c in conds}
            order = np.argsort(-np.mean(list(mass.values()), axis=0).ravel(), kind='stable')
            selectors = {'balanced96': order[:96], 'balanced25pct': order[:math.ceil(.25 * nh)], 'neutral96_legacy': legacy}
            rec = dict(eligible=True, visibility=vis, selectors={})
            for name, sel in selectors.items():
                r = dict(heads=len(sel), total_heads=nh,
                         head_pairs=[[ls[int(j // shape[1])], int(j % shape[1])] for j in sel], conditions={}, pairs={})
                for c in conds:
                    share = (p0[c] * E).sum(-1)
                    r['conditions'][c] = dict(image_mass=float(mass[c].ravel()[sel].mean()),
                        E_share=float(share.ravel()[sel].mean()), within_jsd=float(within[c].ravel()[sel].mean()))
                for a, b in [('neutral', 'belief'), ('neutral', 'belief_right'), ('belief', 'belief_right')]:
                    half, full = _jsd(pe[a], pe[b]), _jsd(p0[a], p0[b])
                    noise = .5 * (within[a] + within[b])
                    r['pairs'][a + '__' + b] = dict(jsd_full=float(full.ravel()[sel].mean()), jsd_half=float(half.ravel()[sel].mean()),
                        noise=float(noise.ravel()[sel].mean()), excess=float((half - noise).ravel()[sel].mean()))
                rec['selectors'][name] = r
                if W == 64 and name == 'balanced96':
                    maps.append(dict(id=iid, group=group, stage='first64', grid=metas['neutral']['image_grid'],
                        image_sha256=item['image_sha256'], selected_heads=len(sel),
                        normalized_image_distribution={c: np.round(p0[c].reshape(-1, n_img)[sel].mean(0), 9).tolist() for c in conds}))
            g['windows'][str(W)] = rec
        for si, stage in enumerate(['prefill_end', 'think', 'answer']):
            # Full thought/answer for sliding layers is confounded by window visibility.
            if any(sw[l] for l in ls) and si > 0:
                g['stages'][stage] = dict(eligible=False, reason='whole-phase sliding visibility may change; use matched fully-visible windows')
                continue
            if any(data[c]['stage_count'][si] == 0 for c in conds):
                g['stages'][stage] = dict(eligible=False, reason='empty stage')
                continue
            avg = {c: data[c]['stage'][si, js] / data[c]['stage_count'][si] for c in conds}
            mass = {c: avg[c].sum(-1) for c in conds}
            sel = np.argsort(-np.mean(list(mass.values()), axis=0).ravel(), kind='stable')[:96]
            probs = {c: _norm_img(avg[c]) for c in conds}
            g['stages'][stage] = dict(eligible=True, conditions={c: dict(image_mass=float(mass[c].ravel()[sel].mean()),
                E_share=float((probs[c]*E).sum(-1).ravel()[sel].mean()), tokens=int(data[c]['stage_count'][si])) for c in conds})
        info['groups'][group] = g
    rows.append(info)
    print(f'CPU {ix+1}/100 {iid}; no model loaded', flush=True)

out = dict(model=model, created=time.time(), questions=100, records=300,
    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    jsd_unit='nats, maximum ln(2)', E_definition='per head top ceil(20% image blocks) by full-neutral thought; model reference, not annotated true ROI',
    selector='balanced96 primary; equal 3-condition image-mass ranking per item/window; balanced25pct sensitivity',
    inclusion='all 3 conditions >= W thought tokens and every image key visible at every compared token/layer',
    excess_formula='even-eight-token-group between-condition JSD minus mean even-vs-odd within-condition JSD; negatives retained', rows=rows)
(D/'attention_per_item.json').write_text(json.dumps(out, ensure_ascii=False, allow_nan=False))
(D/'attention_maps.json').write_text(json.dumps(maps, ensure_ascii=False, allow_nan=False))
print(json.dumps(dict(complete=True, questions=len(rows), groups={g:{str(w):sum(r['groups'][g]['windows'][str(w)]['eligible'] for r in rows) for w in [64,128,256,512]} for g in rows[0]['groups']})), flush=True)
