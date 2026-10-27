# -*- coding: utf-8 -*-
"""锚点一 v2：逐步参照（双分支同步）。说明见 docs/服务器任务_PathVQA_v2逐步参照10例.md。

两条分支，同一张图、同一个问题：
  受压分支：带用户那句话，由 HF generate 正常采样，只有它的输出被保存；
  中性分支：不带用户那句话。每一步把受压分支刚采样的 token 同样喂给中性分支（两条分支的“已生成内容”
            完全相同），只用它的注意力，不用它的输出。
受压分支每次 forward 之前（模型的 forward pre-hook 里）先让中性分支前进一步，记下它在 16 个标准层、
选中的 96 个看图头上的图像内部分布 r_t；随后受压分支在同一层、同一头上把图像内部分布换成
  q' = (1 − λ)·q + λ·r_t                                    （rules.mix）
图像总量、文字部分、整行总和不变；prefill 不干预；48 个线性层和视觉编码器不动。

组别：
  dual_base     λ = 0：中性分支照常运行，受压分支不改。必须与观察阶段逐 token 一致（检验中性分支不影响受压分支）
  dual_mix      λ（默认 1）：换成中性分支此刻的分布
  dual_shuffle  对照：目标换成 r_t 的图像位置随机打乱（每题、每层、每头固定一种打乱），
                每个头的改动量与 dual_mix 用 r_t 时相同（rules.mix 的 ref_alt）

自动检查（不通过就抛异常）：
  - 受压分支：前 3 个解码步显式计算与原注意力函数一致（同 v1）；守恒误差 < 1e-3、没有负概率
  - 中性分支 prefill 最后一个查询的图像注意力 = 观察阶段中性记录里的 prefill_end（最大差 < 1e-3）
  - 中性分支的位置编码：生成结束后，把“中性提示 + 前 K 个生成 token”整段 forward 一次，
    与中性分支逐步算出的下一个 token 分布比较（top-1 一致、实际 token 的对数概率差）
  - 每一步用到的 r_t 必须是中性分支在同一步算出的
"""

import hashlib
import inspect
import time
from typing import Dict, Optional, Tuple

import numpy as np

from . import LAYERS, RANDOM_SEED
from .controller import LAYER_INDEX, STAGE_NAMES, CheckFailed, FPController, active, install
from .rules import XP, mix

DUAL_ARMS = ["dual_base", "dual_mix", "dual_shuffle"]
DUAL_COLS = ["step", "layer", "head", "stage", "image_mass", "js_before", "js_after", "dose", "requested",
             "lam_eff", "cons_image", "cons_text", "cons_row"]
NEUTRAL_CHECK_K = 16
TOL_PREFILL = 1e-3
TOL_TOP1_MISS = 2          # 前 K+1 个位置里 top-1 最多允许不一致的个数
TOL_LOGP_MAX = 1.0         # 实际 token 的对数概率最大差（自然对数）
TOL_LOGP_MEDIAN = 0.1


def shuffle_perm(item_id: str, layer: int, head: int, n: int) -> np.ndarray:
    key = f"{RANDOM_SEED}|dual_shuffle|{item_id}|{layer}|{head}".encode()
    g = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "little"))
    return g.permutation(n)


class DualController(FPController):
    def __init__(self, torch, mode: str, image_idx, close_id: int, ref: Dict[str, np.ndarray], item_id: str,
                 lam: float = 1.0, **kw):
        if mode not in DUAL_ARMS:
            raise ValueError(mode)
        super().__init__(torch, "base", image_idx, close_id, ref=ref, lam=0.0, **kw)
        self.mode = mode
        self.lam_v2 = 0.0 if mode == "dual_base" else float(lam)
        self.item_id = item_id
        self.branch = "pressure"
        self.neutral = None          # NeutralBranch，由 generate_dual 设置
        self.neu_r = {}              # layer -> (step, Tensor[n_sel, n_img])
        self.neu_prefill = {}        # layer -> Tensor[H, n_img]（中性 prefill 最后一个查询的图像注意力）
        self.rope_pressure = None

    # ------------------------------------------------------------ 受压分支每次 forward 之前
    def pre_hook(self, module, args, kwargs):
        if self.branch == "neutral":
            return                                   # 中性分支自己的 forward，不处理
        ids = kwargs.get("input_ids", args[0] if args else None)
        first_prefill = not (ids is not None and ids.shape[-1] == 1 and self.prefill_calls > 0)
        super().pre_hook(module, args, kwargs)
        if first_prefill:
            self.neutral.prefill(self, kwargs)
        else:
            if self.step == 0:
                self.neutral.check_rope(self)
            self.neutral.step(self, kwargs, int(ids[0, -1]))

    # ------------------------------------------------------------ 工具
    def _heads(self, layer: int, device):
        key = ("heads", layer, str(device))
        if key not in self.cache:
            hs = np.flatnonzero(self.ref["heads"][LAYER_INDEX[layer]])
            self.cache[key] = self.torch.as_tensor(hs, dtype=self.torch.long, device=device)
        return self.cache[key]

    def _perm(self, layer: int, device):
        key = ("perm", layer, str(device))
        if key not in self.cache:
            hs = np.flatnonzero(self.ref["heads"][LAYER_INDEX[layer]])
            pm = np.stack([shuffle_perm(self.item_id, layer, int(h), self.n_img) for h in hs]) if len(hs) else \
                np.zeros((0, self.n_img), np.int64)
            self.cache[key] = self.torch.as_tensor(pm, dtype=self.torch.long, device=device)
        return self.cache[key]

    # ------------------------------------------------------------ 注意力
    def attention(self, orig, module, query, key, value, attention_mask, *args, **kwargs):
        if self.branch == "neutral":
            return self._attention_neutral(orig, module, query, key, value, attention_mask, *args, **kwargs)
        return self._attention_pressure(orig, module, query, key, value, attention_mask, *args, **kwargs)

    def _attention_neutral(self, orig, module, query, key, value, attention_mask, *args, **kwargs):
        layer = int(module.layer_idx)
        img = self._dev(query.device)
        if query.shape[2] > 1:
            m = attention_mask[:, :, -1:, :] if attention_mask is not None and attention_mask.dim() == 4 else None
            p = self._probs(module, query[:, :, -1:], key, m, args, kwargs)
            self.neu_prefill[layer] = p.index_select(1, img)
        else:
            p = self._probs(module, query, key, attention_mask, args, kwargs)
            hs = self._heads(layer, query.device)
            sub = p.index_select(1, img).index_select(0, hs)
            self.neu_r[layer] = (self.step, sub / sub.sum(-1, keepdim=True).clamp_min(1e-30))
        return orig(module, query, key, value, attention_mask, *args, **kwargs)

    def _attention_pressure(self, orig, module, query, key, value, attention_mask, *args, **kwargs):
        t = self.torch
        layer = int(module.layer_idx)
        img = self._dev(query.device)
        if query.shape[2] > 1:                                      # prefill：与 v1 相同，不干预
            if self.decode:
                raise CheckFailed("解码阶段出现多查询调用")
            self.prefill_calls += 1 if layer == LAYERS[0] else 0
            out = orig(module, query, key, value, attention_mask, *args, **kwargs)
            m = attention_mask[:, :, -1:, :] if attention_mask is not None and attention_mask.dim() == 4 else None
            p = self._probs(module, query[:, :, -1:], key, m, args, kwargs)
            pi = p.index_select(1, img)
            self._accumulate(layer, 0, pi, pi)
            return out
        if query.shape[0] != 1:
            raise CheckFailed("只支持 batch=1")
        if layer == LAYERS[0]:
            self.decode_calls += 1
        p = self._probs(module, query, key, attention_mask, args, kwargs)
        p_img = p.index_select(1, img)
        slot = 1 + self.stage
        p_new = p
        hs = self._heads(layer, query.device)
        if hs.numel() > 0:
            got = self.neu_r.get(layer)
            if got is None or got[0] != self.step:
                raise CheckFailed(f"第 {self.step} 步第 {layer} 层没有中性分支同一步的参照")
            r = got[1]
            sub = p_img.index_select(0, hs)
            alt = t.gather(r, 1, self._perm(layer, query.device)) if self.mode == "dual_shuffle" else None
            new_sub, info = mix(self.xp, sub, r, self.lam_v2, alt)
            if self.lam_v2 > 0:
                p_new = p.clone()
                p_new[hs[:, None], img[None, :]] = new_sub
            a, b = p.index_select(0, hs), p_new.index_select(0, hs)
            ia, ib = a.index_select(1, img).sum(-1), b.index_select(1, img).sum(-1)
            ra, rb = a.sum(-1), b.sum(-1)
            c_img, c_txt, c_row = (ib - ia).abs(), ((rb - ib) - (ra - ia)).abs(), (rb - ra).abs()
            v = self._violation(query.device)
            v[0] = t.maximum(v[0], t.stack([c_img.max(), c_txt.max(), c_row.max()]).max())
            v[1] = t.maximum(v[1], -b.min())
            n = hs.numel()
            if layer not in self.trace_buf:
                self.trace_buf[layer] = t.zeros(self.max_rows, n, len(DUAL_COLS), dtype=t.float32, device=query.device)
            f = lambda x: t.full((n,), float(x), dtype=t.float32, device=query.device)
            self.trace_buf[layer][self.step] = t.stack(
                [f(self.step), f(layer), hs.float(), f(self.stage), info["P"].float(), info["js_before"].float(),
                 info["js_after"].float(), info["dose"].float(), info["requested"].float(), info["lam_eff"].float(),
                 c_img.float(), c_txt.float(), c_row.float()], -1)
        self._accumulate(layer, slot, p_img, p_new.index_select(1, img))
        rep = p.shape[0] // value.shape[1]
        v_ = value.repeat_interleave(rep, dim=1) if rep > 1 else value
        result = t.matmul(p_new[None, :, None, :], v_.float()).to(query.dtype).transpose(1, 2).contiguous()
        if self.step < self.selfcheck_steps:
            plain = result if p_new is p else t.matmul(p[None, :, None, :], v_.float()).to(query.dtype).transpose(1, 2)
            refo = orig(module, query, key, value, attention_mask, *args, **kwargs)[0].float()
            d = (plain.float() - refo).abs()
            allowed = t.clamp(refo.abs() * 2.0 ** -6, min=self.tol_selfcheck)
            ratio = float((d / allowed).max())
            self.selfchecks.append({"step": self.step, "layer": layer, "max_abs_diff": float(d.max()),
                                    "max_ratio": ratio, "ref_absmax": float(refo.abs().max())})
            if not ratio <= 1.0:
                raise CheckFailed(f"显式注意力与原函数不一致：step={self.step} layer={layer} 允许差之比={ratio:.3f}")
        return result, None

    def _check_violations(self):
        for dev, v in self.viol.items():
            cons, neg, _ = [float(x) for x in v.cpu()]
            if cons >= self.tol_mass or neg > 1e-7:
                raise CheckFailed(f"守恒检查失败（{dev}）：最大误差 {cons}，最小概率 {-neg}，截至解码步 {self.step}")

    # ------------------------------------------------------------ 结果
    def result(self) -> Dict[str, np.ndarray]:
        out = super().result()
        out["trace_columns"] = np.array(DUAL_COLS)
        return out

    def summary(self) -> Dict[str, object]:
        tr = self._trace_np if self._trace_np is not None else np.zeros((0, len(DUAL_COLS)), np.float32)
        c = {n: i for i, n in enumerate(DUAL_COLS)}
        s = {"mode": self.mode, "lam": self.lam_v2, "decode_steps": self.step + 1, "prefill_calls": self.prefill_calls,
             "decode_calls": self.decode_calls, "stage_counts": {n: int(k) for n, k in zip(STAGE_NAMES, self.counts)},
             "selfcheck_max": float(max([x["max_abs_diff"] for x in self.selfchecks] or [0.0])),
             "selfcheck_ratio_max": float(max([x["max_ratio"] for x in self.selfchecks] or [0.0])),
             "selfchecks": len(self.selfchecks)}
        if len(tr):
            s.update(rows=int(len(tr)), js_before=float(tr[:, c["js_before"]].mean()),
                     js_after=float(tr[:, c["js_after"]].mean()), mean_dose=float(tr[:, c["dose"]].mean()),
                     mean_requested=float(tr[:, c["requested"]].mean()), lam_eff_mean=float(tr[:, c["lam_eff"]].mean()),
                     dose_gap_max=float(np.abs(tr[:, c["requested"]] - tr[:, c["dose"]]).max()) if self.lam_v2 > 0 else 0.0,
                     capped_rows=int((tr[:, c["lam_eff"]] >= 1.0 - 1e-6).sum()) if self.mode == "dual_shuffle" else 0,
                     cons_max=float(tr[:, c["cons_image"]:].max()))
            for st, nm in ((0, "think"), (1, "answer")):
                z = tr[tr[:, c["stage"]] == st]
                if len(z):
                    s[nm] = {"rows": int(len(z)), "js_before": float(z[:, c["js_before"]].mean()),
                             "js_after": float(z[:, c["js_after"]].mean()), "mean_dose": float(z[:, c["dose"]].mean())}
        if self.neutral is not None:
            s["neutral_branch"] = self.neutral.report
        return s


class NeutralBranch:
    """中性分支：自己的缓存，跟着受压分支逐 token 前进。"""

    def __init__(self, backend, batch: Dict, n_in: int, obs_prefill: Optional[np.ndarray]):
        self.backend, self.model, self.torch = backend, backend.model, backend.torch
        self.batch, self.n_in = batch, n_in
        self.obs_prefill = obs_prefill           # [16, 24, n]：观察阶段中性记录的 prefill_end
        self.cache = None
        self.rope = None
        self.offset = None                       # 受压提示长度 − 中性提示长度
        self.logits = []                         # 前 K+1 个“下一个 token 分布”（log_softmax）
        self.report = {}
        params = inspect.signature(self.model.forward).parameters
        self._accepts_any = any(p.kind == p.VAR_KEYWORD for p in params.values())
        self._params = set(params)

    def _inner(self):
        return getattr(self.model, "model", None)

    def _call(self, ctrl, kw):
        inner = self._inner()
        saved = getattr(inner, "rope_deltas", None) if inner is not None else None
        ctrl.branch = "neutral"
        try:
            out = self.model(**kw)
        finally:
            ctrl.branch = "pressure"
        rope_after = getattr(inner, "rope_deltas", None) if inner is not None else None
        if inner is not None and hasattr(inner, "rope_deltas"):
            inner.rope_deltas = saved
        return out, rope_after

    def _ok(self, name: str) -> bool:
        return self._accepts_any or name in self._params

    def _keep(self, logits):
        if len(self.logits) <= NEUTRAL_CHECK_K:
            self.logits.append(self.torch.log_softmax(logits[0, -1].float(), -1))

    def prefill(self, ctrl, pkw):
        t = self.torch
        kw = dict(self.batch, use_cache=True, return_dict=True)
        if "cache_position" in pkw and self._ok("cache_position"):
            kw["cache_position"] = t.arange(self.n_in, device=self.model.device)
        p_len = int(pkw["input_ids"].shape[-1])
        self.offset = p_len - self.n_in
        if self.offset <= 0:
            raise CheckFailed(f"受压提示（{p_len}）应比中性提示（{self.n_in}）长")
        out, self.rope = self._call(ctrl, kw)
        self.cache = getattr(out, "past_key_values", None)
        if self.cache is None:
            raise CheckFailed("中性分支 prefill 没有返回缓存（past_key_values）")
        self._keep(out.logits)
        if self.obs_prefill is not None:
            diffs = []
            for layer, a in ctrl.neu_prefill.items():
                b = t.as_tensor(self.obs_prefill[LAYER_INDEX[layer]], device=a.device, dtype=t.float32)
                diffs.append(float((a.float() - b).abs().max()))
            d = max(diffs) if diffs else float("nan")
            self.report["prefill_vs_observation_maxabs"] = d
            if not d < TOL_PREFILL:
                raise CheckFailed(f"中性分支 prefill 的图像注意力与观察阶段中性记录不一致：最大差 {d}")

    def check_rope(self, ctrl):
        inner = self._inner()
        rp = getattr(inner, "rope_deltas", None) if inner is not None else None
        t = self.torch
        if isinstance(rp, t.Tensor) and isinstance(self.rope, t.Tensor):
            same = rp.shape == self.rope.shape and bool(t.equal(rp.to(self.rope.device), self.rope))
            self.report["rope_deltas_equal"] = same
            if not same:
                raise CheckFailed(f"两条分支的 rope_deltas 不同：受压 {rp.tolist()}，中性 {self.rope.tolist()}")
        else:
            self.report["rope_deltas_equal"] = None   # 这一版没有这个属性；位置是否正确由结束后的整段比较检查

    def step(self, ctrl, pkw, token: int):
        t = self.torch
        dev = self.model.device
        n_past = self.n_in + ctrl.step
        kw = {}
        for k, v in pkw.items():
            if k in ("input_ids", "past_key_values", "attention_mask", "cache_position", "position_ids",
                     "pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
                continue
            if k == "mm_token_type_ids" and isinstance(v, t.Tensor) and v.shape[-1] != 1:
                # 整段传入的版本：换成中性提示的类型 + 已生成的文字（0）
                if v.shape[-1] != n_past + self.offset + 1:
                    raise CheckFailed(f"mm_token_type_ids 长度 {v.shape[-1]} 与受压序列长度不符")
                base = self.batch["mm_token_type_ids"]
                v = t.cat([base, t.zeros((1, ctrl.step + 1), dtype=base.dtype, device=base.device)], 1)
            kw[k] = v
        kw["input_ids"] = t.tensor([[token]], device=dev)
        kw["past_key_values"] = self.cache
        kw["use_cache"] = True
        kw["return_dict"] = True
        if pkw.get("attention_mask") is not None:
            kw["attention_mask"] = t.ones((1, n_past + 1), dtype=pkw["attention_mask"].dtype, device=dev)
        if pkw.get("cache_position") is not None:
            cp = pkw["cache_position"] - self.offset
            if int(cp[-1]) != n_past:
                raise CheckFailed(f"中性分支缓存位置 {int(cp[-1])} 与预期 {n_past} 不符")
            kw["cache_position"] = cp
        if pkw.get("position_ids") is not None:
            kw["position_ids"] = pkw["position_ids"] - self.offset     # 图像之后的文字位置在各维都逐一递增
        out, _ = self._call(ctrl, kw)
        new_cache = getattr(out, "past_key_values", None)
        if new_cache is not None:
            self.cache = new_cache
        self._keep(out.logits)

    def verify_positions(self, output_ids):
        """把“中性提示 + 前 K 个生成 token”整段 forward 一次，与逐步算出的分布比较。"""
        t = self.torch
        k = min(NEUTRAL_CHECK_K, len(output_ids) - 1, len(self.logits) - 1)
        if k < 4:
            raise CheckFailed("生成太短，无法检查中性分支的位置编码")
        b = dict(self.batch)
        ext = t.tensor([output_ids[:k]], device=b["input_ids"].device, dtype=b["input_ids"].dtype)
        b["input_ids"] = t.cat([b["input_ids"], ext], 1)
        if "attention_mask" in b:
            b["attention_mask"] = t.ones_like(b["input_ids"])
        if "mm_token_type_ids" in b:
            b["mm_token_type_ids"] = t.cat([b["mm_token_type_ids"], t.zeros_like(ext)], 1)
        inner = self._inner()
        if inner is not None and hasattr(inner, "rope_deltas"):
            inner.rope_deltas = None
        full = t.log_softmax(self.model(**b, use_cache=False, return_dict=True).logits[0].float(), -1)
        rows = full[self.n_in - 1: self.n_in + k]                      # 预测 output[0..k]
        step = t.stack(self.logits[: k + 1])
        truth = t.tensor(output_ids[: k + 1], device=rows.device)
        top1_miss = int((rows.argmax(-1) != step.to(rows.device).argmax(-1)).sum())
        dl = (rows.gather(1, truth[:, None]) - step.to(rows.device).gather(1, truth[:, None])).abs()[:, 0]
        self.report.update(position_check_k=k + 1, top1_miss=top1_miss, logp_diff_max=float(dl.max()),
                           logp_diff_median=float(dl.median()))
        if top1_miss > TOL_TOP1_MISS or float(dl.max()) > TOL_LOGP_MAX or float(dl.median()) > TOL_LOGP_MEDIAN:
            raise CheckFailed(f"中性分支逐步计算与整段计算不一致（位置编码或缓存可能有误）：{self.report}")


def run(cfg, root: str, phase: str, shard: str = "", seed: int = 42, lam: float = 1.0) -> None:
    """fp-dual：预检（前 2 个候选 × 三组）→ 阅读确认 → 全部候选（dual_mix、dual_shuffle）。
    记录写在 root/int/{dual_base,dual_mix,dual_shuffle}/，之后照常 fp-export / fp-import --phase int、fp-public。"""
    import json
    import os
    import sys
    from .pipeline import Engine, _check_no_failure, _done, _fail, _save_result, _sha, _shard, ensure_reference, load_items
    _check_no_failure(root)
    cands = json.load(open(os.path.join(root, "candidates.json"), encoding="utf-8"))["candidates"]
    items = {i["id"]: i for i in load_items(root)}
    gate_p, review_p = os.path.join(root, "dual_gate.json"), os.path.join(root, "dual_review.json")
    if phase == "pilot":
        cases, arms = cands[:2], list(DUAL_ARMS)
    else:
        if not (os.path.exists(gate_p) and json.load(open(gate_p))["ok"]):
            raise SystemExit("预检没有通过（dual_gate.json），不能跑全部")
        rv = json.load(open(review_p, encoding="utf-8")) if os.path.exists(review_p) else {}
        if not rv.get("ok") or rv.get("dual_gate_sha256") != _sha(gate_p):
            raise SystemExit("还没有阅读预检回复并写 dual_review.json（ok=true，并带 dual_gate_sha256）")
        cases, arms = cands, ["dual_mix", "dual_shuffle"]
    tasks = _shard([(c, a) for c in cases for a in arms], shard)
    todo = [(c, a) for c, a in tasks if not _done(root, f"int/{a}", c["id"], c["condition"])]
    print(f"[fp-dual] {phase}：λ={lam}，本 worker {len(tasks)} 项，待做 {len(todo)} 项", file=sys.stderr)
    eng = Engine(cfg) if todo else None
    for k, (c, a) in enumerate(todo):
        it = items[c["id"]]
        task = f"int/{a}/{c['id']}__{c['condition']}"
        try:
            ref = ensure_reference(root, c["id"])
            if eng.backend is None:
                from .mock import generate_one as mock_one
                rec, arr = mock_one(it, c["condition"], a, ref, seed)
            else:
                with np.load(os.path.join(root, "obs", "attn", f"{c['id']}__neutral.npz")) as z:
                    obs_prefill = z["prefill_end"]
                rec, arr = generate_dual(eng.backend, cfg, it, c["condition"], a, ref, lam, seed, obs_prefill)
            _save_result(root, f"int/{a}", it, c["condition"], a, rec, arr)
        except BaseException as e:
            _fail(root, task, e)
            raise
        cs = rec["controller"]
        nb = cs.get("neutral_branch", {})
        print(f"  {k + 1}/{len(todo)} {task} tokens={rec['n_out']} {rec['seconds']}s "
              f"JS {cs.get('js_before', float('nan')):.4f}->{cs.get('js_after', float('nan')):.4f} "
              f"dose={cs.get('mean_dose', 0):.3f} prefill_diff={nb.get('prefill_vs_observation_maxabs')} "
              f"top1_miss={nb.get('top1_miss')} logp_max={nb.get('logp_diff_max')}", file=sys.stderr)
    if phase == "pilot":
        gate(root, cases)


def gate(root: str, cases) -> Optional[dict]:
    import json
    import os
    import sys
    from .pipeline import _atomic_json
    rows, ok = [], True
    for c in cases:
        obs = json.load(open(os.path.join(root, "obs", "records", f"{c['id']}__{c['condition']}.json"), encoding="utf-8"))
        for a in DUAL_ARMS:
            p = os.path.join(root, "int", a, "records", f"{c['id']}__{c['condition']}.json")
            if not os.path.exists(p):
                return None                                  # 其他 worker 还没做完
            r = json.load(open(p, encoding="utf-8"))
            cs = r["controller"]
            nb = cs.get("neutral_branch", {})
            checks = {"selfcheck_within_tolerance": cs.get("selfcheck_ratio_max", 0.0) <= 1.0,
                      "conservation_lt_1e-3": cs.get("cons_max", 0.0) < 1e-3,
                      "neutral_prefill_matches_observation": (nb.get("prefill_vs_observation_maxabs") or 0.0) < TOL_PREFILL,
                      "neutral_positions_ok": nb.get("top1_miss", 99) <= TOL_TOP1_MISS,
                      "reply_nonempty_no_garbage": bool(r["answer"].strip()) and "�" not in r["raw_output"]}
            if a == "dual_base":
                checks["same_as_observation"] = r["output_ids"] == obs["output_ids"]
            if a == "dual_mix":
                checks["js_drops"] = cs.get("js_after", 1.0) < cs.get("js_before", 0.0)
            if a == "dual_shuffle":
                checks["dose_matched_lt_1e-3_or_capped"] = cs.get("dose_gap_max", 0.0) < 1e-3 or cs.get("capped_rows", 0) > 0
            ok = ok and all(checks.values())
            rows.append({"id": c["id"], "condition": c["condition"], "arm": a, "checks": checks, "controller": cs,
                         "answer": r["answer"], "raw_output_sha256": hashlib.sha256(r["raw_output"].encode()).hexdigest()})
    g = {"ok": ok, "cases": [f"{c['id']}__{c['condition']}" for c in cases], "replies": rows,
         "next": "阅读这 6 份回复的完整原文（int/dual_*/records/*.json 的 raw_output），确认通顺后写 dual_review.json："
                 "{\"ok\": true, \"dual_gate_sha256\": <本文件 SHA256>, \"read\": [...]}"}
    _atomic_json(os.path.join(root, "dual_gate.json"), g)
    print(f"[fp-dual] 预检 ok={ok} → {root}/dual_gate.json", file=sys.stderr)
    return g


def generate_dual(backend, cfg, item: dict, cond: str, mode: str, ref: Dict[str, np.ndarray], lam: float,
                  seed: int = 42, obs_prefill: Optional[np.ndarray] = None) -> Tuple[Dict, Dict[str, np.ndarray]]:
    from ..attn import image_token_id, merge_size
    from .generate import build_inputs, make_budget
    torch, tf, model = backend.torch, backend.transformers, backend.model
    if backend.close_id is None:
        raise RuntimeError("</think> 不是单个 token，无法一次生成")
    install(model)
    prompt, inputs = build_inputs(backend, item["image"], item["messages"][cond])
    n_prompt, n_inputs = build_inputs(backend, item["image"], item["messages"]["neutral"])
    ids, nids = inputs["input_ids"][0], n_inputs["input_ids"][0]
    tid = image_token_id(backend)
    img_idx = (ids == tid).nonzero().flatten()
    if not torch.equal(img_idx, (nids == tid).nonzero().flatten()):
        raise RuntimeError("两条分支的图像 token 位置不同")
    grid = inputs["image_grid_thw"][0].tolist()
    ms = merge_size(backend)
    if int(img_idx.numel()) != grid[0] * grid[1] * grid[2] // (ms * ms) or int(ref["n_img"]) != int(img_idx.numel()):
        raise RuntimeError("图像 token 数与 grid 或参照不一致")
    dev = model.device

    def to_dev(x):
        b = {k: v.to(dev) for k, v in x.items()}
        if "pixel_values" in b:
            b["pixel_values"] = b["pixel_values"].to(model.dtype)
        return b
    batch, nbatch = to_dev(inputs), to_dev(n_inputs)
    n_in = int(ids.shape[0])
    gc = model.generation_config
    eos = gc.eos_token_id if isinstance(gc.eos_token_id, list) else [gc.eos_token_id]
    tb, ab = cfg.think_budget, cfg.answer_budget
    budget = make_budget(tf, torch, n_in, backend.close_id, int(eos[0]), tb, ab)
    ctrl = DualController(torch, mode, img_idx, backend.close_id, ref, item["id"], lam=lam, max_rows=tb + ab + 2)
    ctrl.neutral = NeutralBranch(backend, nbatch, int(nids.shape[0]), obs_prefill)
    hook = model.register_forward_pre_hook(ctrl.pre_hook, with_kwargs=True)
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "rope_deltas"):
        inner.rope_deltas = None
    tf.set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    t0 = time.time()
    try:
        with torch.inference_mode(), active(ctrl):
            out = model.generate(**batch, do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p,
                                 top_k=cfg.top_k, min_p=cfg.min_p, repetition_penalty=cfg.repetition_penalty,
                                 max_new_tokens=tb + ab + 2, use_cache=True,
                                 logits_processor=tf.LogitsProcessorList([budget]))
    finally:
        hook.remove()
    seconds = round(time.time() - t0, 2)
    output = out[0, n_in:].tolist()
    with torch.inference_mode():
        ctrl.neutral.verify_positions(output)
    close = output.index(backend.close_id) if backend.close_id in output else None
    tok = backend.tokenizer
    raw = tok.decode(output, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    thinking = tok.decode(output[:close], skip_special_tokens=True).strip() if close is not None else \
        tok.decode(output, skip_special_tokens=True).strip()
    answer = tok.decode(output[close + 1:], skip_special_tokens=True).strip() if close is not None else ""
    arrays = ctrl.result()
    arrays.update(grid_hw=np.array([grid[1] // ms, grid[2] // ms], np.int32), grid_thw=np.array(grid, np.int32),
                  merge_size=np.array(ms), image=np.array(str(item["image"])), prompt_len=np.array(n_in),
                  close_pos=np.array(-1 if close is None else n_in + close), total_len=np.array(n_in + len(output)))
    if arrays["pos_image_mass"].shape[0] != len(output):
        raise RuntimeError("逐 token 记录行数与生成 token 数不一致")
    record = {"prompt_text": prompt, "neutral_prompt_text": n_prompt, "input_ids": ids.tolist(), "output_ids": output,
              "raw_output": raw, "thinking": thinking, "answer": answer, "n_in": n_in, "n_out": len(output),
              "n_img": int(img_idx.numel()), "seconds": seconds, "seed": seed, "budget": budget.metadata(),
              "controller": ctrl.summary()}
    return record, arrays
