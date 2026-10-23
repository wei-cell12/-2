# -*- coding: utf-8 -*-
"""记录标准注意力层对图像 token 的注意力（逐图像块），按阶段平均。

用途：
  1. 只读前向回放：把已保存的输入 + 输出 token 原样喂回模型（不生成），取三个阶段的图像注意力。
  2. 生成时在线记录（LQCD 的两条分支都用它）。

做法：把 transformers 的注意力函数（ALL_ATTENTION_FUNCTIONS[sdpa] 等，或模型文件里的
eager_attention_forward）换成一个包装：对“记录目标”模块（语言模型里的标准自注意力层），
自己按查询分块算 softmax(QK^T)·V，先对全部 key 归一化，再截取图像列累加；其他模块原样调用。
线性注意力层没有位置级的注意力分布，不记录。

阶段（按“查询位置” q 定义；位置 p 的 token 由查询 p-1 生成）：
  prefill_end  q = prompt_len - 1                 还没写任何字时，模型看哪里
  think        prompt_len <= q < close_pos        生成思考 token 的查询
  answer       close_pos <= q <= total_len - 2    </think> 之后生成回答 token 的查询
close_pos 是 prompt_len 之后第一个 </think> 的位置；没有 </think> 时全部算 think。
逐位置的图像注意力总量（用于和预实验保存的 generation_masses 校验）按行存：
第 k 行 = 查询 prompt_len - 1 + k，即生成第 k 个 token 时的注意力，共 total_len - prompt_len 行。
"""

import sys
from typing import Dict, List, Optional, Tuple

STAGES = ["prefill_end", "think", "answer"]
STAGE_ID = {s: i for i, s in enumerate(STAGES)}

_ACTIVE = None          # 当前生效的 Recorder；None 时包装函数直接调用原函数
_TARGETS = set()        # 记录目标模块的 id()
_INSTALLED = False


def stage_bounds(prompt_len: int, close_pos: Optional[int], total_len: int) -> Dict[str, Tuple[int, int]]:
    """各阶段查询位置的半开区间 [start, end)。"""
    last = total_len - 1  # 最后一个 token 作为查询不生成任何东西
    out = {"prefill_end": (prompt_len - 1, prompt_len)}
    if close_pos is None or close_pos >= total_len:
        out["think"] = (prompt_len, last)
        out["answer"] = (last, last)
    else:
        out["think"] = (prompt_len, close_pos)
        out["answer"] = (close_pos, last)
    return out


def find_close(ids: List[int], prompt_len: int, close_id: int) -> Optional[int]:
    for i in range(prompt_len, len(ids)):
        if ids[i] == close_id:
            return i
    return None


class Recorder:
    """一条序列（或一条分支）的累加器。"""

    def __init__(self, torch, image_idx, stage_map=None, rows: Optional[Tuple[int, int]] = None,
                 chunk: int = 256, selfcheck: int = 2):
        self.torch = torch
        self.image_idx = image_idx            # LongTensor[n_img]，图像 token 的位置（key 端）
        self.n_img = int(image_idx.numel())
        self.stage_map = stage_map            # LongTensor[total_len]：每个查询位置的阶段编号，-1 不记录
        self.step_stage = None                # 逐 token 解码时由调用方设置当前查询的阶段
        self.rows = rows                      # (row_start, row_end)：要保存逐位置图像总量的查询位置区间
        self.chunk = chunk
        self.sums: Dict[int, object] = {}     # layer_idx -> Tensor[S, H, n_img]
        self.counts: Dict[int, object] = {}   # layer_idx -> Tensor[S]
        self.pos_mass: Dict[int, Dict[int, object]] = {}  # layer_idx -> {query_pos: Tensor[H]}
        self.selfcheck_left = selfcheck
        self.selfcheck_diffs: List[float] = []

    def stages_for(self, q_pos):
        torch = self.torch
        if self.step_stage is not None and q_pos.numel() == 1:
            return torch.full_like(q_pos, self.step_stage)
        if self.stage_map is None:
            return torch.full_like(q_pos, -1)
        sm = self.stage_map.to(q_pos.device)
        out = torch.full_like(q_pos, -1)
        ok = q_pos < sm.numel()
        out[ok] = sm[q_pos[ok]]
        return out

    def accumulate(self, layer_idx: int, probs, stages, q_pos):
        """probs: [B=1, H, c, kL]（已对全部 key 归一化），stages / q_pos: [c]。"""
        torch = self.torch
        H = probs.shape[1]
        img = probs[0][:, :, self.image_idx.to(probs.device)]          # [H, c, n_img]
        if layer_idx not in self.sums:
            self.sums[layer_idx] = torch.zeros(len(STAGES), H, self.n_img, dtype=torch.float64)
            self.counts[layer_idx] = torch.zeros(len(STAGES), dtype=torch.float64)
        for s in range(len(STAGES)):
            sel = (stages == s)
            if bool(sel.any()):
                self.sums[layer_idx][s] += img[:, sel.to(img.device), :].sum(1).double().cpu()
                self.counts[layer_idx][s] += float(sel.sum())
        if self.rows is not None:
            r0, r1 = self.rows
            sel = (q_pos >= r0) & (q_pos < r1)
            if bool(sel.any()):
                mass = img.sum(-1)                                      # [H, c]
                d = self.pos_mass.setdefault(layer_idx, {})
                idx = sel.nonzero().flatten().tolist()
                for j in idx:
                    d[int(q_pos[j])] = mass[:, j].float().cpu()

    def result(self) -> Dict[str, object]:
        """返回 numpy 数组：每阶段 [L, H, n_img] 的平均值，层按 layer_idx 升序。"""
        import numpy as np
        layers = sorted(self.sums)
        out = {"layer_ids": np.array(layers, dtype=np.int32), "n_img": self.n_img,
               "selfcheck_max_abs_diff": np.array(self.selfcheck_diffs, dtype=np.float32)}
        for s, name in enumerate(STAGES):
            cnt = np.array([float(self.counts[l][s]) for l in layers]) if layers else np.zeros(0)
            out[f"count_{name}"] = cnt[0] if len(cnt) else 0.0
            if len(cnt) and cnt[0] > 0:
                out[name] = np.stack([(self.sums[l][s] / self.counts[l][s]).numpy() for l in layers]).astype(np.float32)
            else:
                out[name] = np.zeros((len(layers), 0, self.n_img), dtype=np.float32)
        if self.rows is not None and layers:
            r0, r1 = self.rows
            H = self.sums[layers[0]].shape[1]
            pm = np.full((r1 - r0, len(layers), H), np.nan, dtype=np.float32)
            for li, l in enumerate(layers):
                for p, v in self.pos_mass.get(l, {}).items():
                    pm[p - r0, li] = v.numpy()
            out["pos_image_mass"] = pm.astype(np.float16)
        return out


# ---------------------------------------------------------------------------------------------
def _recording_attention(orig, module, query, key, value, attention_mask, *args, **kwargs):
    rec = _ACTIVE
    torch = rec.torch
    dropout = kwargs.get("dropout", args[0] if len(args) > 0 else 0.0)
    scaling = kwargs.get("scaling", args[1] if len(args) > 1 else None)
    B, H, qL, D = query.shape
    kL = key.shape[2]
    rep = H // key.shape[1]
    k = key.repeat_interleave(rep, dim=1) if rep > 1 else key          # 与 repeat_kv 的头顺序一致
    v = value.repeat_interleave(rep, dim=1) if rep > 1 else value
    scale = scaling if scaling is not None else getattr(module, "scaling", D ** -0.5)
    q_pos = torch.arange(kL - qL, kL, device=query.device)
    stages = rec.stages_for(q_pos)
    out = torch.empty(B, H, qL, v.shape[-1], dtype=query.dtype, device=query.device)
    kf = k.float().transpose(-1, -2)
    vf = v.float()
    key_pos = torch.arange(kL, device=query.device)
    for s in range(0, qL, rec.chunk):
        e = min(qL, s + rec.chunk)
        sc = torch.matmul(query[:, :, s:e].float(), kf) * scale          # [B, H, c, kL]
        if attention_mask is not None:
            m = attention_mask[:, :, s:e, :kL]
            if m.dtype == torch.bool:
                sc = sc.masked_fill(~m, float("-inf"))
            else:
                sc = sc + m.float()
        else:
            allowed = key_pos[None, :] <= q_pos[s:e][:, None]
            sc = sc.masked_fill(~allowed[None, None], float("-inf"))
        p = torch.softmax(sc, dim=-1)
        out[:, :, s:e] = torch.matmul(p, vf).to(query.dtype)
        rec.accumulate(int(module.layer_idx), p, stages[s:e], q_pos[s:e])
        del sc, p
    result = out.transpose(1, 2).contiguous()
    if rec.selfcheck_left > 0:  # 与原注意力函数的输出比对，确认重算一致
        rec.selfcheck_left -= 1
        ref = orig(module, query, key, value, attention_mask, *args, **kwargs)[0]
        rec.selfcheck_diffs.append(float((result.float() - ref.float()).abs().max()))
    return result, None


def _wrap(orig):
    def wrapped(module, query, key, value, attention_mask, *args, **kwargs):
        if _ACTIVE is None or id(module) not in _TARGETS:
            return orig(module, query, key, value, attention_mask, *args, **kwargs)
        return _recording_attention(orig, module, query, key, value, attention_mask, *args, **kwargs)
    wrapped._symexp_wrapped = True
    return wrapped


def find_targets(model) -> List[Tuple[str, object]]:
    """语言模型里的标准自注意力模块：有 q_proj/k_proj/o_proj 和 layer_idx，名字里没有 visual/vision。"""
    found = []
    for name, m in model.named_modules():
        low = name.lower()
        if "visual" in low or "vision" in low:
            continue
        if all(hasattr(m, a) for a in ("q_proj", "k_proj", "o_proj")) and hasattr(m, "layer_idx"):
            found.append((name, m))
    return found


def install(model, verbose: bool = True) -> List[int]:
    """安装包装（幂等）。返回记录目标的 layer_idx 列表。"""
    global _INSTALLED
    targets = find_targets(model)
    _TARGETS.clear()
    _TARGETS.update(id(m) for _, m in targets)
    layer_ids = sorted(int(m.layer_idx) for _, m in targets)
    if not _INSTALLED:
        try:
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        except ImportError:
            ALL_ATTENTION_FUNCTIONS = None
        if ALL_ATTENTION_FUNCTIONS is not None:
            for impl in ("sdpa", "flash_attention_2", "flex_attention"):
                try:
                    orig = ALL_ATTENTION_FUNCTIONS[impl]
                except Exception:
                    continue
                if getattr(orig, "_symexp_wrapped", False):
                    continue
                w = _wrap(orig)
                try:
                    ALL_ATTENTION_FUNCTIONS[impl] = w
                except Exception:
                    pass
                gm = getattr(type(ALL_ATTENTION_FUNCTIONS), "_global_mapping", None)
                if isinstance(gm, dict) and impl in gm:
                    gm[impl] = w
        for mod_name in {type(m).__module__ for _, m in targets}:  # eager 路径：模型文件里的全局函数
            mod = sys.modules.get(mod_name)
            if mod is not None and hasattr(mod, "eager_attention_forward"):
                f = mod.eager_attention_forward
                if not getattr(f, "_symexp_wrapped", False):
                    mod.eager_attention_forward = _wrap(f)
        _INSTALLED = True
    if verbose:
        print(f"[attn] recording targets: {len(targets)} modules, layer_idx={layer_ids}", file=sys.stderr)
    return layer_ids


class recording:
    """with recording(rec): ...  —— 在这段代码里，目标模块的注意力被记录到 rec。"""

    def __init__(self, rec: Optional[Recorder]):
        self.rec = rec
        self.prev = None

    def __enter__(self):
        global _ACTIVE
        self.prev, _ACTIVE = _ACTIVE, self.rec
        return self.rec

    def __exit__(self, *exc):
        global _ACTIVE
        _ACTIVE = self.prev
        return False


# ---------------------------------------------------------------------------------------------
def image_token_id(backend) -> int:
    for obj in (backend.model.config, getattr(backend.model.config, "text_config", None), backend.processor):
        for name in ("image_token_id", "image_token_index"):
            v = getattr(obj, name, None) if obj is not None else None
            if isinstance(v, int):
                return v
    return backend.tokenizer.convert_tokens_to_ids("<|image_pad|>")


def merge_size(backend) -> int:
    ip = getattr(backend.processor, "image_processor", None)
    return int(getattr(ip, "merge_size", 2) or 2)


def replay_item(backend, ids: List[int], prompt_len: int, image, keep_positions: bool = True,
                chunk: int = 256, selfcheck: int = 2) -> Dict[str, object]:
    """把一条完整序列（输入 + 输出 token）做一次前向，返回各阶段的图像注意力。"""
    import numpy as np
    torch = backend.torch
    model = backend.model
    install(model, verbose=False)
    dev = model.device
    ip = backend.processor.image_processor
    vis = ip(images=[image], return_tensors="pt")
    pixel_values = vis["pixel_values"].to(dev, dtype=model.dtype)
    grid = vis["image_grid_thw"].to(dev)
    ids_t = torch.tensor([ids], device=dev)
    img_tok = image_token_id(backend)
    image_idx = (ids_t[0] == img_tok).nonzero().flatten()
    ms = merge_size(backend)
    t, h, w = [int(x) for x in grid[0].tolist()]
    expect = t * h * w // (ms * ms)
    if int(image_idx.numel()) != expect:
        raise ValueError(f"图像 token 数 {int(image_idx.numel())} 与图像处理结果 {expect} 不一致："
                         f"像素范围或 processor 设置和原实验不同")
    close_pos = find_close(ids, prompt_len, backend.close_id) if backend.close_id is not None else None
    total = len(ids)
    bounds = stage_bounds(prompt_len, close_pos, total)
    stage_map = torch.full((total,), -1, dtype=torch.long)
    for name, (a, b) in bounds.items():
        if b > a:
            stage_map[a:b] = STAGE_ID[name]
    rows = (prompt_len - 1, total - 1) if keep_positions else None
    rec = Recorder(torch, image_idx.cpu(), stage_map, rows, chunk=chunk, selfcheck=selfcheck)
    kw = dict(input_ids=ids_t, attention_mask=torch.ones_like(ids_t), pixel_values=pixel_values,
              image_grid_thw=grid, use_cache=False)
    with torch.inference_mode(), recording(rec):
        try:
            model(**kw, logits_to_keep=1)
        except TypeError:
            model(**kw)
    out = rec.result()
    out.update(prompt_len=prompt_len, close_pos=-1 if close_pos is None else close_pos, total_len=total,
               grid_thw=np.array([t, h, w], dtype=np.int32), merge_size=ms,
               grid_hw=np.array([h // ms, w // ms], dtype=np.int32),
               image_pos=np.array([int(image_idx[0]), int(image_idx[-1]) + 1], dtype=np.int64))
    return out


def save_npz(path: str, data: Dict[str, object], **extra) -> None:
    import os
    import numpy as np
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arrays = {}
    for k, v in {**data, **extra}.items():
        arrays[k] = np.asarray(v)
    np.savez_compressed(path, **arrays)
