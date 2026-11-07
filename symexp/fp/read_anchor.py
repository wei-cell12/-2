# -*- coding: utf-8 -*-
"""锚点一试跑：读图阶段，把受压回答的看图注意力往"干净参照分支"拉（ClockQA，几题）。

两条分支，同一张图：
  主分支：完整提示（问题 + 用户那句话），正常采样生成，只保存它的输出；
  参照分支：提示去掉用户那句话；思考里删掉开头的复述段（思考开头到第一个空行）；之后跟着主分支，
            把主分支写出的读图 token 逐个读入（不生成），只用它的注意力。
            = 复述验证（fp-restate）里的 D。
窗口：复述段结束后，主分支生成接下来 WINDOW 个 token 的那些步。窗口外不动，参照分支也不跑。
每一步、每一层、每个头（主分支 m、q；参照分支 m*、q*）：
  看图总量 m' = max(m, m*)：图像部分整体乘 m'/m，文字部分整体乘 (1 − m')/(1 − m)；
  图像内部分配（POS 层）：q'_j = (1 − λ) q_j + λ q*_j，固定位置（这一步在两条分支里份额都 ≥ 5 倍平均值的图块）保持原值，
  参照的非固定部分先按比例缩放到与主分支的非固定部分同样多，保证 q' 加起来是 1。
层：Qwen 16 个标准注意力层都调总量和分配（48 个线性注意力层改不了）；Gemma 全部 60 层调总量，只有滑动窗口层调分配。
组：base（参照分支照跑，不改主分支；同一随机种子）、pull（按上式改）。两组在窗口开始前逐 token 相同。
"""

import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

from .rules import XP

WINDOW = 128
LAM = 0.5
SINK_MULT = 5.0
MAX_RESTATE = 200
ARMS = ["base", "pull"]
VERIFY_K = 16
TOL_TOP1_MISS = 2
TOL_LOGP_MAX = 1.0
TOL_LOGP_MEDIAN = 0.1


class AnchorCheckFailed(RuntimeError):
    pass


# ---------------------------------------------------------------- 纯函数
def pull(xp: XP, P, R, lam: float, pos: bool, mult: float = SINK_MULT):
    """P、R：[H, n] 主分支 / 参照分支分给各图块的注意力（每行加起来 = 各自的看图总量）。
    返回 (新的图像部分 [H, n]，文字部分的缩放 [H, 1]，信息)。"""
    tiny = 1e-30
    n = P.shape[-1]
    m, ms = xp.sum(P), xp.sum(R)
    q = P / xp.maximum(m, xp.full_like(m, tiny))
    qs = R / xp.maximum(ms, xp.full_like(ms, tiny))
    if pos and lam > 0:
        f = xp.fl((q >= mult / n) & (qs >= mult / n))
        qS, qsS = q * (1 - f), qs * (1 - f)
        a, b = xp.sum(qS), xp.sum(qsS)
        scale = a / xp.maximum(b, xp.full_like(b, tiny))
        mixS = xp.where(b > tiny, (1 - lam) * qS + lam * qsS * scale, qS)
        qn = q * f + mixS
    else:
        qn = q
    m2 = xp.maximum(m, ms)
    text = (1 - m2) / xp.maximum(1 - m, xp.full_like(m, tiny))
    return m2 * qn, text, {"m": m, "m_ref": ms, "m_new": m2, "q": q, "q_ref": qs, "q_new": qn}


def jsd_rows(xp: XP, p, q):
    """逐行 JSD（自然对数）。"""
    tiny = 1e-30
    mm = 0.5 * (p + q)
    kl = lambda a, b: xp.sum(xp.where(a > 0, a * (xp.log(xp.maximum(a, xp.full_like(a, tiny)))
                                                  - xp.log(xp.maximum(b, xp.full_like(b, tiny)))), xp.full_like(a, 0.0)))
    return 0.5 * kl(p, mm) + 0.5 * kl(q, mm)


def restate_end(text_from_skip: str, n_tok: int, min_tok: int = 4) -> bool:
    """复述段是否在当前 token 结束：从思考开头起的文字里第一次出现空行，并且至少已有 min_tok 个 token。"""
    return n_tok >= min_tok and "\n\n" in text_from_skip.lstrip()


def first_divergence(a: List[int], b: List[int]) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


# ---------------------------------------------------------------- 控制器
class AnchorController:
    """装在 attn 的包装上（attn._ACTIVE）：参照分支前向时记下每层的看图注意力；主分支前向时按 pull 改。"""

    SKIP_KEYS = ("input_ids", "past_key_values", "attention_mask", "cache_position", "position_ids", "pixel_values",
                 "image_grid_thw", "pixel_values_videos", "video_grid_thw", "image_sizes")

    def __init__(self, backend, cfg, arm: str, nbatch: Dict, n_in: int, img_abs, layers: List[int], sliding: Dict[int, bool],
                 lam: float = LAM, window: int = WINDOW):
        from . import reading_groups
        self.b, self.cfg, self.arm, self.lam, self.window = backend, cfg, arm, lam, window
        self.torch, self.model, self.tok = backend.torch, backend.model, backend.tokenizer
        self.xp = XP("torch")
        self.nbatch, self.n_in, self.img_abs = nbatch, n_in, img_abs
        self.layers, self.sliding = set(layers), sliding
        self.think_skip = reading_groups.think_skip
        self.chunk = cfg.attn_chunk
        self.selfcheck_left, self.selfcheck_diffs = 0, []
        self.branch = "main"
        self.out: List[int] = []
        self.prefilled = False
        self.state = "wait"              # wait → window → done；或 none（没找到复述段）
        self.skip = 0 if cfg.think_prefill else None
        self.end = None
        self.wstep, self._cur = 0, 0
        self.ref_cache, self.ref_rope, self.ref_ids, self.ref_logits = None, None, None, []
        self.ref_img: Dict[int, object] = {}
        self.abs_q = None
        self.stats = []                  # 每步每层：[wstep, layer, sliding, m, m_ref, m_new, jsd_before, jsd_after, n_fixed]
        self.skipped_layers = 0
        self._rel = {}

    # attn 包装会读这些属性
    @property
    def passthrough(self):
        return self.branch == "ref" or self.arm == "base"

    @property
    def rel_checked(self):
        return self._rel if self.branch == "main" else None

    def stages_for(self, q_pos):
        return self.torch.full_like(q_pos, -1)

    def accumulate(self, layer_idx, probs, stages, q_pos):
        pass

    def _img_keys(self, kL: int, device):
        off = self.abs_q + 1 - kL                      # 滑动窗口层的缓存可能只留最近一段
        k = self.img_abs.to(device) - off
        return k if bool((k >= 0).all()) else None

    def modify(self, layer_idx, p, q_pos):
        if layer_idx not in self.layers or p.shape[2] != 1:
            return p
        img = self._img_keys(p.shape[-1], p.device)
        if self.branch == "ref":
            if img is not None:
                self.ref_img[layer_idx] = (self.wstep, p[0, :, 0].index_select(-1, img).float())
            return p
        got = self.ref_img.get(layer_idx)
        if img is None or got is None or got[0] != self._cur:
            self.skipped_layers += 1
            return p
        t = self.torch
        row = p[0, :, 0].float()                                   # [H, kL]
        P = row.index_select(-1, img)
        pos = self.sliding.get(layer_idx, False) or not any(self.sliding.values())
        new_img, text, info = pull(self.xp, P, got[1], self.lam, pos)
        jb = jsd_rows(self.xp, info["q"], info["q_ref"])
        ja = jsd_rows(self.xp, info["q_new"], info["q_ref"])
        n = P.shape[-1]
        nf = ((info["q"] >= SINK_MULT / n) & (info["q_ref"] >= SINK_MULT / n)).float().sum(-1, keepdim=True)
        self.stats.append(t.stack([info["m"].mean(), info["m_ref"].mean(), info["m_new"].mean(), jb.mean(), ja.mean(),
                                   nf.mean()]).cpu().numpy().tolist() + [self._cur, layer_idx, int(self.sliding.get(layer_idx, False))])
        if self.arm == "base":
            return p
        new = row * text
        new[:, img] = new_img
        return new.to(p.dtype)[None, :, None, :]

    # ------------------------------------------------------------ 模型 forward 之前
    def pre_hook(self, module, args, kwargs):
        from .. import attn
        if self.branch == "ref":
            return
        attn._ACTIVE = None
        ids = kwargs.get("input_ids", args[0] if args else None)
        if not self.prefilled:
            self.prefilled = True
            return
        if ids is None or ids.shape[-1] != 1:
            raise AnchorCheckFailed("解码阶段出现多 token 输入")
        tok_id = int(ids[0, -1])
        self.out.append(tok_id)
        k = len(self.out)
        if self.state == "wait":
            if self.skip is None and k >= 13:
                self.skip = self.think_skip(self.tok, self.cfg, self.out)
            if self.skip is not None and k > self.skip:
                if k - self.skip > MAX_RESTATE:
                    self.state = "none"
                elif restate_end(self.tok.decode(self.out[self.skip:k], skip_special_tokens=False), k - self.skip):
                    self.end = k
                    self._ref_prefill()
                    self.state = "window"
        if self.state == "window":
            if self.wstep >= self.window or tok_id == self.b.close_id:
                self.state = "done"
                return
            ref_tok = self.ref_ids[-1] if self.wstep == 0 else tok_id
            self._ref_step(kwargs, ref_tok)              # 参照分支先走这一步，记下每层的看图注意力（标记为第 wstep 步）
            self._cur = self.wstep
            self.abs_q = self.n_in + k - 1
            attn._ACTIVE = self                          # 主分支这一步按参照改
            self.wstep += 1

    def _inner(self):
        return getattr(self.model, "model", None)

    def _call(self, kw, rope, record: bool = True):
        inner = self._inner()
        has = inner is not None and hasattr(inner, "rope_deltas")
        saved = inner.rope_deltas if has else None
        if has:
            inner.rope_deltas = rope
        self.branch = "ref"
        from .. import attn
        attn._ACTIVE = self if record else None
        try:
            out = self.model(**kw)
        finally:
            self.branch = "main"
            attn._ACTIVE = None
        after = inner.rope_deltas if has else None
        if has:
            inner.rope_deltas = saved
        return out, after

    def _is_type(self, key, v, n) -> bool:
        """逐 token 的类型字段（如 mm_token_type_ids）：二维、长度等于序列长度、不是 input_ids / attention_mask。"""
        t = self.torch
        return isinstance(v, t.Tensor) and v.dim() == 2 and v.shape[-1] == n and key not in ("input_ids", "attention_mask")

    def _ref_prefill(self):
        """参照分支：中性提示 + 主分支写出的思考开头标记（Gemma），先读入除最后一个之外的全部 token；
        最后一个留到窗口第 0 步作为查询（与 fp-restate 的 D 完全相同的序列）。"""
        t = self.torch
        dev = self.model.device
        n0 = int(self.nbatch["input_ids"].shape[-1])
        self.ref_ids = self.nbatch["input_ids"][0].tolist() + self.out[:self.skip]
        L = len(self.ref_ids) - 1
        kw = {}
        extra = t.tensor([self.out[:self.skip]], dtype=self.nbatch["input_ids"].dtype, device=dev).reshape(1, -1)
        for key, v in self.nbatch.items():
            if key == "input_ids":
                v = t.cat([v, extra], 1)[:, :L]
            elif key == "attention_mask":
                v = t.ones((1, L), dtype=v.dtype, device=dev)
            elif self._is_type(key, v, n0):
                v = t.cat([v, t.zeros((1, extra.shape[-1]), dtype=v.dtype, device=v.device)], 1)[:, :L]
            kw[key] = v
        kw.update(use_cache=True, return_dict=True)
        self.wstep = 0
        out, self.ref_rope = self._call(kw, None, record=False)
        self.ref_cache = getattr(out, "past_key_values", None)
        if self.ref_cache is None:
            raise AnchorCheckFailed("参照分支 prefill 没有返回缓存")

    def _ref_step(self, pkw, token: int):
        t = self.torch
        dev = self.model.device
        k = len(self.out)
        main_total = self.n_in + k                    # 主分支这一步之后的长度
        ref_total = len(self.ref_ids) + self.wstep    # 参照分支这一步之后的长度
        kw = {}
        for key, v in pkw.items():
            if key in self.SKIP_KEYS:
                continue
            if self._is_type(key, v, main_total) and v.shape[-1] != 1:
                if key not in self.nbatch:
                    raise AnchorCheckFailed(f"主分支每步传入整段的 {key}，参照分支没有对应字段")
                base = self.nbatch[key]
                v = t.cat([base, t.zeros((1, ref_total - base.shape[-1]), dtype=base.dtype, device=base.device)], 1)
            kw[key] = v
        kw.update(input_ids=t.tensor([[token]], device=dev), past_key_values=self.ref_cache, use_cache=True, return_dict=True)
        if pkw.get("attention_mask") is not None:
            kw["attention_mask"] = t.ones((1, ref_total), dtype=pkw["attention_mask"].dtype, device=dev)
        if pkw.get("cache_position") is not None:
            kw["cache_position"] = t.tensor([ref_total - 1], device=dev)
        if pkw.get("position_ids") is not None:
            kw["position_ids"] = pkw["position_ids"] - (main_total - ref_total)    # 图像之后的文字位置逐一递增
        self.abs_q = ref_total - 1
        out, _ = self._call(kw, self.ref_rope)
        nc = getattr(out, "past_key_values", None)
        if nc is not None:
            self.ref_cache = nc
        if len(self.ref_logits) <= VERIFY_K:
            self.ref_logits.append(t.log_softmax(out.logits[0, -1].float(), -1))

    def verify_ref(self, output: List[int]) -> dict:
        """把参照分支的整段（参照提示 + 读图 token）一次 forward，与逐步算出的下一个 token 分布比较（位置编码与缓存是否正确）。"""
        from .entropy import _batch
        t = self.torch
        kk = min(VERIFY_K, len(self.ref_logits) - 1, len(output) - self.end - 1)
        if kk < 4:
            return {"checked": False}
        ext = self.out[:self.skip] + output[self.end:self.end + kk]
        inner = self._inner()
        if inner is not None and hasattr(inner, "rope_deltas"):
            inner.rope_deltas = None
        n0 = int(self.nbatch["input_ids"].shape[-1])
        types = [k for k, v in self.nbatch.items() if self._is_type(k, v, n0)]
        b = _batch(self.b, dict(self.nbatch), ext)
        for key in types:
            if b[key].shape[-1] != b["input_ids"].shape[-1]:
                b[key] = t.cat([b[key], t.zeros((1, b["input_ids"].shape[-1] - b[key].shape[-1]), dtype=b[key].dtype,
                                                device=b[key].device)], 1)
        full = t.log_softmax(self.model(**b, use_cache=False, return_dict=True).logits[0].float(), -1)
        R = len(self.ref_ids)
        rows = full[R - 1: R + kk]
        step = t.stack(self.ref_logits[: kk + 1]).to(rows.device)
        truth = t.tensor(output[self.end:self.end + kk + 1], device=rows.device)
        miss = int((rows.argmax(-1) != step.argmax(-1)).sum())
        dl = (rows.gather(1, truth[:, None]) - step.gather(1, truth[:, None])).abs()[:, 0]
        rep = {"checked": True, "k": kk + 1, "top1_miss": miss, "logp_diff_max": float(dl.max()),
               "logp_diff_median": float(dl.median())}
        if miss > TOL_TOP1_MISS or rep["logp_diff_max"] > TOL_LOGP_MAX or rep["logp_diff_median"] > TOL_LOGP_MEDIAN:
            raise AnchorCheckFailed(f"参照分支逐步计算与整段计算不一致（位置编码或缓存可能有误）：{rep}")
        return rep

    def summary(self) -> dict:
        s = {"state": self.state, "skip": self.skip, "restate_end": self.end, "window_steps": self.wstep,
             "skipped_layer_calls": self.skipped_layers, "explicit_vs_original": self._rel or None}
        if self.stats:
            a = np.array(self.stats, float)
            kinds = {"全局层": a[:, 8] == 0, "滑动窗口层": a[:, 8] == 1} if (a[:, 8] == 1).any() else {"全部层": a[:, 8] >= 0}
            s["by_kind"] = {k: {"rows": int(m.sum()), "mass": float(a[m, 0].mean()), "mass_ref": float(a[m, 1].mean()),
                                "mass_new": float(a[m, 2].mean()), "jsd_before": float(a[m, 3].mean()),
                                "jsd_after": float(a[m, 4].mean()), "fixed_patches": float(a[m, 5].mean())}
                            for k, m in kinds.items() if m.any()}
        return s


# ---------------------------------------------------------------- 生成一条
def generate(backend, cfg, item: dict, cond: str, arm: str, seed: int = 42, lam: float = LAM, window: int = WINDOW) -> dict:
    from .. import attn
    from .conflict import claim_of
    from .generate import build_inputs, make_budget
    from .verify import final_time, outcome
    torch, tf, model, tok = backend.torch, backend.transformers, backend.model, backend.tokenizer
    if backend.close_id is None:
        raise AnchorCheckFailed("思考结束标记不是单个 token")
    targets = attn.find_targets(model)
    layers = sorted(int(m.layer_idx) for _, m in targets)
    sliding = {int(m.layer_idx): bool(getattr(m, "is_sliding", False)) for _, m in targets}
    attn.install(model, verbose=False)
    prompt, inputs = build_inputs(backend, item["image"], item["messages"][cond])
    n_prompt, n_inputs = build_inputs(backend, item["image"], item["messages"]["neutral"])
    tid = attn.image_token_id(backend)
    ids, nids = inputs["input_ids"][0], n_inputs["input_ids"][0]
    img_idx = (ids == tid).nonzero().flatten()
    if img_idx.numel() == 0 or not torch.equal(img_idx, (nids == tid).nonzero().flatten()):
        raise AnchorCheckFailed("两条分支的图像 token 位置不同（或没有图像 token）")
    dev = model.device

    def to_dev(x):
        b = {k: v.to(dev) for k, v in x.items() if hasattr(v, "to")}
        if "pixel_values" in b:
            b["pixel_values"] = b["pixel_values"].to(model.dtype)
        return b
    batch, nbatch = to_dev(inputs), to_dev(n_inputs)
    n_in = int(ids.shape[0])
    gc = model.generation_config
    eos = gc.eos_token_id if isinstance(gc.eos_token_id, list) else [gc.eos_token_id]
    tb, ab = cfg.think_budget, cfg.answer_budget
    budget = make_budget(tf, torch, n_in, backend.close_id, int(eos[0]), tb, ab)
    ctrl = AnchorController(backend, cfg, arm, nbatch, n_in, img_idx.to(dev), layers, sliding, lam=lam, window=window)
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
        with torch.inference_mode():
            out = model.generate(**batch, do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p, top_k=cfg.top_k,
                                 min_p=cfg.min_p, repetition_penalty=cfg.repetition_penalty, max_new_tokens=tb + ab + 2,
                                 use_cache=True, logits_processor=tf.LogitsProcessorList([budget]))
    finally:
        hook.remove()
        attn._ACTIVE = None
    seconds = round(time.time() - t0, 2)
    output = out[0, n_in:].tolist()
    verify = {"checked": False}
    if ctrl.end is not None:
        with torch.inference_mode():
            verify = ctrl.verify_ref(output)
    close = output.index(backend.close_id) if backend.close_id in output else None
    thinking = tok.decode(output[:close] if close is not None else output, skip_special_tokens=True).strip()
    answer = tok.decode(output[close + 1:], skip_special_tokens=True).strip() if close is not None else ""
    claim = claim_of(item, cond)
    rec = {"id": item["id"], "condition": cond, "arm": arm, "lam": lam, "window": window, "seed": seed,
           "gold": item["gold"], "claim": claim, "output_ids": output,
           "raw_output": tok.decode(output, skip_special_tokens=False), "thinking": thinking, "answer": answer,
           "outcome": outcome(answer, item["gold"], claim), "final_time": final_time(answer, claim),
           "n_in": n_in, "n_out": len(output), "n_img": int(img_idx.numel()), "seconds": seconds, "budget": budget.metadata(),
           "restated": tok.decode(output[ctrl.skip:ctrl.end], skip_special_tokens=False) if ctrl.end else None,
           "controller": ctrl.summary(), "ref_check": verify}
    return rec


# ---------------------------------------------------------------- 入口
def run(cfg, root: str, ent_dir: str, model_key: str, ids: List[str], image_dirs: Optional[List[str]] = None,
        label: str = "", seed: int = 42, lam: float = LAM, window: int = WINDOW) -> None:
    from .conflict import _engine
    from .trigger_regions import load_entropy, resolve_image, resolve_source
    from .verify import outcome
    os.makedirs(os.path.join(root, "records"), exist_ok=True)
    ent = load_entropy(ent_dir, model_key, ["ClockQA"])
    keys = [k for k in sorted(ent) if k[0] in ids]
    missing = sorted(set(ids) - {k[0] for k in keys})
    if missing:
        raise SystemExit(f"熵差实验里没有这些题：{missing}")
    eng = None
    for iid, c in keys:
        e = ent[(iid, c)]
        job = e["job"]
        item = dict(job["item"])
        orig = json.load(open(resolve_source(job), encoding="utf-8"))
        item["image"] = resolve_image(job, orig, image_dirs)
        for arm in ARMS:
            p = os.path.join(root, "records", f"{iid}__{c}__{arm}.json")
            if os.path.exists(p):
                continue
            if eng is None:
                eng = _engine(cfg)
            rec = generate(eng.backend, cfg, item, c, arm, seed=seed, lam=lam, window=window)
            rec["group"] = e["group"]
            rec["original"] = {"outcome": outcome(orig.get("answer", ""), item["gold"], rec["claim"]),
                               "answer": orig.get("answer", "")}
            with open(p, "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=1)
            cs = rec["controller"]
            print(f"  {iid}__{c} {arm}：{rec['outcome']}，{rec['n_out']} token，{rec['seconds']}s，窗口 {cs['window_steps']} 步，"
                  f"复述段结束于第 {cs['restate_end']} 个 token，跳过层次数 {cs['skipped_layer_calls']}，参照核对 {rec['ref_check']}",
                  file=sys.stderr)
    report(root, label)


def report(root: str, label: str = "") -> None:
    d = os.path.join(root, "records")
    recs = {}
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".json"):
            r = json.load(open(os.path.join(d, fn), encoding="utf-8"))
            recs[(r["id"], r["condition"], r["arm"])] = r
    cname = {"belief": "用户说错", "belief_right": "用户说对"}
    gname = {"flip": "被带偏", "hold": "守住"}
    L = [f"# 锚点一试跑：读图阶段把看图注意力往参照分支拉 · {label}", "",
         "- 参照分支：提示去掉用户那句话，思考里删掉开头的复述段，之后读入主分支写出的读图 token；",
         f"- 窗口：复述段结束后的 {WINDOW} 个 token；看图总量 m' = max(m, m*)；图像内部分配向参照靠 λ = {LAM}（固定位置不动）；",
         "- base：参照分支照跑，不改主分支；pull：按上式改。两组用同一个随机种子，窗口开始前逐 token 相同；",
         "- 原回答：熵差实验用的那条回答（另一次采样）。", "",
         "## 答案", "", "| 题 · 条件 | 原回答 | base | pull | base 与 pull 第一次不同的位置 | 复述段结束位置 |", "|---|---|---|---|---:|---:|"]
    pairs = sorted({(i, c) for i, c, _ in recs})
    for i, c in pairs:
        b, pl = recs.get((i, c, "base")), recs.get((i, c, "pull"))
        if not b or not pl:
            continue
        div = first_divergence(b["output_ids"], pl["output_ids"])
        L.append(f"| {i} · {cname.get(c, c)}（{gname.get(b['group'], b['group'])}） | {b['original']['outcome']} | {b['outcome']} | "
                 f"{pl['outcome']} | {div} | {pl['controller']['restate_end']} |")
    L += ["", "## 窗口内的看图注意力（逐步、逐层、逐头平均）", "",
          "看图总量：全部注意力里落在图像上的比例；与参照的差别：图像内部分配与参照分支的 JSD。"
          "pull 组的\"调整后\"是实际用上的注意力；base 组只算不改。", ""]
    for i, c in pairs:
        for arm in ARMS:
            r = recs.get((i, c, arm))
            if not r or "by_kind" not in r["controller"]:
                continue
            L += [f"**{i} · {cname.get(c, c)} · {arm}**", "", "| 层 | 看图总量（主分支） | 看图总量（参照） | 看图总量（调整后） | 与参照的差别（调整前） | 与参照的差别（调整后） | 固定位置图块数 |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
            for k, v in r["controller"]["by_kind"].items():
                L.append(f"| {k} | {v['mass'] * 100:.1f}% | {v['mass_ref'] * 100:.1f}% | {v['mass_new'] * 100:.1f}% | "
                         f"{v['jsd_before']:.4f} | {v['jsd_after']:.4f} | {v['fixed_patches']:.1f} |")
            L.append("")
    L += ["## 复述段与最终回答", ""]
    for i, c in pairs:
        for arm in ARMS:
            r = recs.get((i, c, arm))
            if r:
                L += [f"**{i} · {cname.get(c, c)} · {arm}**：复述段 {r['restated']!r}", "", f"最终回答：{r['answer'][:600]!r}", ""]
    with open(os.path.join(root, "read_anchor_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"[fp-anchor1] {len(recs)} 条 → {root}/read_anchor_report.md", file=sys.stderr)
