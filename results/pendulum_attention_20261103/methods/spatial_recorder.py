"""Passive PENDULUM spatial attention aggregation; no sampling or intervention."""
import numpy as np


class SpatialRecorder:
    passthrough = True
    chunk = 64
    selfcheck_left = 0

    def __init__(self, torch, n_in, n_out, image_positions, think_start, close, windows):
        self.torch = torch
        self.n_in, self.n_out = n_in, n_out
        self.image_positions = np.asarray(image_positions, dtype=np.int64)
        self.think_start = think_start
        self.close = n_out if close is None else close
        self.windows = windows
        self.rel_checked, self.data = {}, {}
        self.selfcheck_diffs = []
        self.max_row_error = 0.0

    def stages_for(self, qpos):
        return self.torch.full_like(qpos, -1)

    def accumulate(self, layer, probs, stages, qpos):
        torch = self.torch
        assert probs.shape[0] == 1
        rows = qpos.detach().cpu().numpy() - (self.n_in - 1)
        valid = (rows >= 0) & (rows < self.n_out)
        if not valid.any():
            return
        rows = rows[valid]
        ix = torch.as_tensor(np.flatnonzero(valid), device=probs.device)
        p = probs[0].index_select(1, ix)
        err = float((p.sum(-1) - 1).abs().max())
        if not torch.isfinite(p).all() or err >= 1e-4:
            raise ValueError(f'Invalid attention probabilities at layer {layer}: {err}')
        self.max_row_error = max(self.max_row_error, err)
        im = p.index_select(-1, torch.as_tensor(self.image_positions, device=p.device))
        im = im.float().cpu().numpy()
        mass = im.sum(-1)
        H, _, I = im.shape
        if layer not in self.data:
            self.data[layer] = dict(
                stage_sum=np.zeros((3, H, I), np.float32),
                stage_count=np.zeros(3, np.int32),
                early_sum=np.zeros((64, H, I), np.float32),
                early_count=np.zeros(64, np.int32),
                image_mass=np.full((self.n_out, H), np.nan, np.float32),
                visible_image_tokens=np.zeros(self.n_out, np.int32),
                seen=np.zeros(self.n_out, bool))
        d = self.data[layer]
        if d['seen'][rows].any():
            raise ValueError('Duplicate query rows')
        d['seen'][rows] = True
        d['image_mass'][rows] = mass.T
        query = rows + self.n_in - 1
        win = self.windows[layer]
        visible = self.image_positions[None, :] <= query[:, None]
        if win:
            visible &= self.image_positions[None, :] > query[:, None] - win
        d['visible_image_tokens'][rows] = visible.sum(-1)
        masks = [rows == 0,
                 (rows >= self.think_start) & (rows < self.close),
                 rows > self.close]
        for stage, mask in enumerate(masks):
            if mask.any():
                d['stage_sum'][stage] += im[:, mask].sum(1)
                d['stage_count'][stage] += mask.sum()
        early = rows - self.think_start
        for group in np.unique(early[(early >= 0) & (early < 512)] // 8):
            mask = (early >= group * 8) & (early < (group + 1) * 8) & (rows < self.close)
            if mask.any():
                d['early_sum'][group] += im[:, mask].sum(1)
                d['early_count'][group] += mask.sum()

    def arrays(self, expected_layers):
        if set(self.data) != set(expected_layers):
            raise ValueError('Missing attention layers')
        arrays = dict(image_positions=self.image_positions)
        for layer, d in sorted(self.data.items()):
            if not d['seen'].all() or not np.isfinite(d['image_mass']).all():
                raise ValueError(f'Incomplete query coverage at layer {layer}')
            for key, value in d.items():
                if key != 'seen':
                    arrays[f'L{layer}_{key}'] = value
            arrays[f'L{layer}_text_mass'] = 1 - d['image_mass']
        return arrays
