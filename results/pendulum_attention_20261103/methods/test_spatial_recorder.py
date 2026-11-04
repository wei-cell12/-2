"""CPU checks of passive output, query alignment, grouping, and sliding masking."""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

sys.path.insert(0, '/mnt/sdb/linxin/pendulum-20261102/source')
from symexp import attn
from spatial_recorder import SpatialRecorder

torch.manual_seed(42)
for window in [0, 5]:
    n_in, n_out, heads, width = 5, 20, 2, 4
    total = n_in + n_out
    q = torch.randn(1, heads, total, width)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    pos = torch.arange(total)
    mask = pos[None, :] <= pos[:, None]
    if window:
        mask &= pos[None, :] > pos[:, None] - window
    p = torch.softmax((q @ k.transpose(-1, -2) * width**-.5).masked_fill(~mask, -torch.inf), -1)
    reference = (p @ v).transpose(1, 2).contiguous()
    returned = (reference, 'original marker')
    def original(*args, **kw):
        return returned
    module = SimpleNamespace(layer_idx=0, scaling=width**-.5, is_sliding=bool(window), sliding_window=window)
    rr = SpatialRecorder(torch, n_in, n_out, [1, 2], 2, 18, {0: window})
    rr.chunk = 3
    with attn.recording(rr):
        result = attn._recording_attention(original, module, q, k, v, None)
    assert result is returned
    arr = rr.arrays([0])
    expected = p[0, :, n_in-1:n_in-1+n_out, :][:, :, [1, 2]].numpy()
    np.testing.assert_allclose(arr['L0_image_mass'], expected.sum(-1).T, atol=1e-7)
    np.testing.assert_allclose(arr['L0_stage_sum'][0], expected[:, 0], atol=1e-7)
    np.testing.assert_allclose(arr['L0_stage_sum'][1], expected[:, 2:18].sum(1), atol=1e-7)
    np.testing.assert_allclose(arr['L0_stage_sum'][2], expected[:, 19:20].sum(1), atol=1e-7)
    assert arr['L0_stage_count'].tolist() == [1, 16, 1]
    assert arr['L0_early_count'][:3].tolist() == [8, 8, 0]
    np.testing.assert_allclose(arr['L0_early_sum'][0], expected[:, 2:10].sum(1), atol=1e-7)
    assert arr['L0_visible_image_tokens'][-1] == (0 if window else 2)
    assert rr.rel_checked['sliding' if window else 'global']['max_abs'] < 1e-6
    print(f'PASS window={window}: original return identity, mass, stages, groups, query alignment, visibility', flush=True)
print('2 passive recorder CPU cases passed', flush=True)
