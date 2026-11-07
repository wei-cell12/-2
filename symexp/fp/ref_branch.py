# -*- coding: utf-8 -*-
"""无压力分支的看图注意力能不能当参照（ClockQA，试跑几题）。

同一条受压回答，读图阶段（思考前 W 个 token）比较四种看图注意力：
  ① 主分支：完整提示（问题 + 用户那句话）+ 受压回答已写的思考；
  ② 无压力分支：只有问题 + 同一段已写的思考（和熵差实验的无用户分支相同）；
  ③ 无压力分支 + 屏蔽：同 ②，但注意力里屏蔽"抄来的用户答案"——熵差实验里含 ΔH ≤ −1 的那些连续负值段；
  ④ 中性回答：只有问题 + 中性回答自己写的思考（标准答案）。
都只回放、不生成。③ 只能屏蔽标准注意力层（Qwen 的 48 层线性注意力看得到这些 token）。
"""

import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

WINDOWS = [64, 128]
BRANCHES = ["main", "nouser", "nouser_masked", "neutral"]
BRANCH_NAME = {"main": "① 主分支（有用户那句话）", "nouser": "② 无压力分支", "nouser_masked": "③ 无压力分支 + 屏蔽抄来的用户答案",
               "neutral": "④ 中性回答（标准答案）"}


def copied_runs(dH: List[float], upto: int, thresh: float = -1.0) -> List[int]:
    """前 upto 个输出位置里，含 ΔH ≤ thresh 的连续负值段的全部位置（抄来的用户答案）。"""
    out, run = [], []
    for t in range(min(upto, len(dH))):
        if dH[t] < 0:
            run.append(t)
        else:
            if any(dH[x] <= thresh for x in run):
                out += run
            run = []
    if any(dH[x] <= thresh for x in run):
        out += run
    return out


class BranchRecorder:
    """记录输出行 0..W-1 上，每层所有头加起来分给各图块的注意力 [W, L, n] 和看图总量 [W, L]（对头平均）。
    mask_keys：可选，这些位置的注意力置零后重新归一化（只影响被记录的标准注意力层）。"""

    def __init__(self, torch, W: int, n_in: int, image_idx, layers: List[int], chunk: int = 256, mask_keys=None):
        self.torch, self.W, self.n_in, self.image_idx, self.layers, self.chunk = torch, W, n_in, image_idx, list(layers), chunk
        self.selfcheck_left, self.selfcheck_diffs = 0, []
        self.dist = np.zeros((W, len(layers), int(image_idx.numel())))
        self.mass = np.zeros((W, len(layers)))
        self.seen = {l: 0 for l in layers}
        self.mask_keys = None if mask_keys is None or len(mask_keys) == 0 else torch.as_tensor(sorted(mask_keys), dtype=torch.long)
        if self.mask_keys is not None:
            self.modify = self._modify

    def stages_for(self, q_pos):
        return self.torch.full_like(q_pos, -1)

    def _modify(self, layer_idx, p, q_pos):
        if layer_idx not in self.layers:
            return p
        k = self.mask_keys.to(p.device)
        k = k[k < p.shape[-1]]
        p = p.clone()
        p[..., k] = 0
        return p / p.sum(-1, keepdim=True).clamp_min(1e-12)

    def accumulate(self, layer_idx, probs, stages, q_pos):
        if layer_idx not in self.layers:
            return
        li = self.layers.index(layer_idx)
        rows = (q_pos - (self.n_in - 1)).cpu().numpy()
        hit = np.flatnonzero((rows >= 0) & (rows < self.W))
        if not len(hit):
            return
        p = probs[0][:, hit, :][:, :, self.image_idx.to(probs.device)].float()     # [H, r, n]
        self.dist[rows[hit], li] = p.sum(0).cpu().numpy()
        self.mass[rows[hit], li] = p.sum(-1).mean(0).cpu().numpy()
        self.seen[layer_idx] += len(hit)


def branch_stats(dist: np.ndarray, mass: np.ndarray, W: int, layer_mask: np.ndarray, hands: Optional[np.ndarray]) -> dict:
    """前 W 行、选定的层：图像内部分布（按看图量加权）、看图总量、指针区域份额。"""
    d = dist[:W][:, layer_mask].sum((0, 1))
    d = d / max(d.sum(), 1e-12)
    out = {"dist": d, "mass": float(mass[:W][:, layer_mask].mean())}
    if hands is not None:
        out["hands"] = float(d[hands].sum())
    return out


def _jsd(p, q):
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(np.where(a > 0, a * np.log(np.maximum(a, 1e-300) / np.maximum(b, 1e-300)), 0.0)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def compare(stats: Dict[str, dict], frac: float = 0.2) -> dict:
    """以 ④ 中性回答为标准：E = ④ 里注意力最高的 frac 图块；各分支的 E 份额、与 ④ 的 JSD。"""
    ref = stats["neutral"]["dist"]
    k = max(1, int(round(len(ref) * frac)))
    E = np.argsort(-ref, kind="stable")[:k]
    out = {}
    for b, s in stats.items():
        out[b] = {"mass": s["mass"], "E": float(s["dist"][E].sum()), "jsd_to_neutral": _jsd(s["dist"], ref)}
        if "hands" in s:
            out[b]["hands"] = s["hands"]
    return out


def _sha(p):
    from .trigger_regions import _sha as s
    return s(p)


def run(cfg, root: str, ent_dir: str, model_key: str, ids: List[str], image_dirs: Optional[List[str]] = None,
        label: str = "") -> dict:
    from .. import attn
    from .conflict import _engine
    from .entropy import _batch
    from .generate import build_inputs
    from .reading_groups import image_grid, record_layers
    from .sources import dial_geometry
    from .trigger_regions import hand_shape, load_entropy, read_steps, resolve_image, resolve_source
    os.makedirs(root, exist_ok=True)
    ent = load_entropy(ent_dir, model_key, ["ClockQA"])
    keys = [k for k in sorted(ent) if k[0] in ids]
    if not keys:
        raise SystemExit(f"熵差实验里没有这些题：{ids}")
    eng = _engine(cfg)
    b = eng.backend
    torch = b.torch
    all_l = sorted(int(m.layer_idx) for _, m in attn.find_targets(b.model))
    sliding = {int(m.layer_idx): bool(getattr(m, "is_sliding", False)) for _, m in attn.find_targets(b.model)}
    attn.install(b.model, verbose=False)
    W = max(WINDOWS)
    res, L = {}, []
    for iid, c in keys:
        e = ent[(iid, c)]
        job, item = e["job"], e["job"]["item"]
        rec = json.load(open(resolve_source(job), encoding="utf-8"))
        npath = job.get("neutral_source")
        if not npath or not os.path.exists(npath) or _sha(npath) != job.get("neutral_sha256"):
            raise SystemExit(f"{iid}__{c}：中性回答 {npath} 不存在或 SHA 不一致")
        nrec = json.load(open(npath, encoding="utf-8"))
        image = resolve_image(job, rec, image_dirs)
        dH, ok, toks, n_think = read_steps(e["csv"])
        masked = copied_runs(dH, W)
        runs = {"main": (item["messages"][c], rec["output_ids"], None),
                "nouser": (item["messages"]["neutral"], rec["output_ids"], None),
                "nouser_masked": (item["messages"]["neutral"], rec["output_ids"], masked),
                "neutral": (item["messages"]["neutral"], nrec["output_ids"], None)}
        stats_by_kind = {}
        meta = {"id": iid, "condition": c, "group": e["group"], "masked_tokens": [toks[t] for t in masked],
                "masked_positions": masked, "seconds": {}}
        hands = None
        for br, (msg, out_ids, mask_rows) in runs.items():
            _, inp = build_inputs(b, image, msg)
            ids_in = inp["input_ids"][0].tolist()
            n_in = len(ids_in)
            out = [int(x) for x in out_ids][:W]
            img_tok = attn.image_token_id(b)
            image_idx = torch.tensor([i for i, x in enumerate(ids_in) if x == img_tok])
            if hands is None:
                grid = image_grid(b, inp)
                hands = hand_shape(dial_geometry(image), grid, item["gold"])
                if int(grid.prod()) != int(image_idx.numel()):
                    raise SystemExit(f"{iid}：图像 token 排列与个数不一致")
            mask_keys = None if mask_rows is None else [n_in + t for t in mask_rows]
            r = BranchRecorder(torch, W, n_in, image_idx, all_l, chunk=cfg.attn_chunk, mask_keys=mask_keys)
            r.passthrough, r.rel_checked = mask_rows is None, {}
            inner = getattr(b.model, "model", None)
            if inner is not None and hasattr(inner, "rope_deltas"):
                inner.rope_deltas = None
            t0 = time.time()
            with torch.inference_mode(), attn.recording(r):
                try:
                    b.model(**_batch(b, inp, out), use_cache=False, return_dict=True, logits_to_keep=1)
                except TypeError:
                    b.model(**_batch(b, inp, out), use_cache=False, return_dict=True)
            meta["seconds"][br] = round(time.time() - t0, 2)
            sl = np.array([sliding[l] for l in all_l])
            kinds = {"全局层": ~sl, "滑动窗口层": sl} if sl.any() else {"全部层": np.ones(len(all_l), bool)}
            for kind, lm in kinds.items():
                for Wd in WINDOWS:
                    stats_by_kind.setdefault((kind, Wd), {})[br] = branch_stats(r.dist, r.mass, Wd, lm, hands)
        cmp = {f"{k}|{w}": compare(s) for (k, w), s in stats_by_kind.items()}
        res[f"{iid}__{c}"] = {**meta, "compare": cmp}
        print(f"  {iid}__{c}（{e['group']}）屏蔽 {len(masked)} 个 token；用时 {meta['seconds']}", file=sys.stderr)
    with open(os.path.join(root, "ref_branch.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    gname = {"flip": "被带偏", "hold": "守住"}
    cname = {"belief": "用户说错", "belief_right": "用户说对"}
    L = [f"# 无压力分支的看图注意力能不能当参照：{label}（试跑）", "",
         "同一条受压回答，读图阶段（思考前 W 个 token），四种看图注意力：", ""]
    L += [f"- {BRANCH_NAME[x]}" for x in BRANCHES]
    L += ["", "**指标**：",
          "- 指针区域：看图注意力里落在真实指针上的比例（由正确时间算出）；",
          "- 关键区：④ 中性回答里注意力最高的 20% 图块，各分支有多少落在上面（④ 自己天然偏高）；",
          "- 与 ④ 的差别：看图分布与 ④ 的 JSD，越小越像中性回答；",
          "- 看图总量：全部注意力里落在图像上的比例。",
          "**怎么看**：② 或 ③ 的数接近 ④，说明可以当参照；接近 ①，说明被上文里抄来的用户答案带偏了。", ""]
    for key, v in res.items():
        L += [f"## {v['id']} · {cname[v['condition']]}（{gname.get(v['group'], v['group'])}）", "",
              f"屏蔽的 token（{len(v['masked_tokens'])} 个）：{''.join(v['masked_tokens'])!r}", ""]
        for ck, cm in v["compare"].items():
            kind, Wd = ck.split("|")
            L += [f"**{kind}，前 {Wd} 个 token**", "", "| 分支 | 指针区域 | 关键区 | 与 ④ 的差别（JSD） | 看图总量 |", "|---|---:|---:|---:|---:|"]
            for br in BRANCHES:
                x = cm[br]
                L.append(f"| {BRANCH_NAME[br]} | {x.get('hands', float('nan')) * 100:.1f}% | {x['E'] * 100:.1f}% | {x['jsd_to_neutral']:.4f} | {x['mass'] * 100:.1f}% |")
            L.append("")
    with open(os.path.join(root, "ref_branch_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"[fp-refbranch] {len(res)} 条 → {root}/ref_branch_report.md", file=sys.stderr)
    return res
