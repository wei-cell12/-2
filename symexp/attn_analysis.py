# -*- coding: utf-8 -*-
"""图像注意力区域比较：中性（参照） vs 用户说错（或 LQCD 的某条分支）。

输入是 attn.replay_item / LQCD 记录下的 npz：每阶段一个 [层, 头, 图像token] 的平均注意力。
每个 (层, 头) 先在图像内部归一化，只比较“看图像的哪里”，不受图像注意力总量影响。

区域定义不需要标注，但要避免“同一份数据既挑区域又测量”造成的回归均值偏差：
标准层按顺序分成两半（偶数位、奇数位），用一半的参照图挑区域，在另一半上测量，两个方向取平均。

  E 区：参照（中性）注意力最高的 20% 图像块，即“本来看的位置”
  U 区：比较组相对参照增加最多的 20% 图像块，即“压力后多看的位置”

逐题指标（正值都表示“离开中性时看的位置”）：
  dlogL     log(区域外平均 / E 区平均) 的变化（你给的 L_attn = A_out / A_in）
  dE        E 区在图像注意力中的份额变化（负值 = 证据区被冷落）
  dU        U 区份额的相对变化（在另一半层上测，正值 = 稳定地多看了这些位置）
  retention 比较组注意力最高的 20% 里，有多少落在 E 区（越低 = 视线移得越远）
  js        两张图像注意力分布的 JS 散度
  dmass     图像注意力总量的对数比（对照 B 组“恢复总量”的思路）
"""

import csv
import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from .attn import STAGES

TOP = 0.2


def parse_source(spec: str) -> Tuple[str, str, str]:
    """NAME=DIR@TAG，文件为 DIR/{id}__{TAG}.npz；TAG 里可以用 * 通配。
    回放结果：TAG = neutral 或 user_wrong。LQCD：TAG = user_wrong__*__lead 或 user_wrong__*__neutral。"""
    name, rest = spec.split("=", 1)
    d, tag = rest.rsplit("@", 1)
    return name, d, tag


def load_source(d: str, tag: str) -> Dict[str, str]:
    import fnmatch
    out = {}
    if not os.path.isdir(d):
        return out
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".npz") or "__" not in fn:
            continue
        i, rest = fn[:-4].split("__", 1)
        if fnmatch.fnmatchcase(rest, tag):
            if i in out:
                raise ValueError(f"{d} 里题 {i} 有多个文件匹配 {tag}，请把 TAG 写具体（例如带上压力类型）")
            out[i] = os.path.join(d, fn)
    return out


def _load(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------------------------
def norm_heads(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """x: [L, H, n] -> (图像内部归一化的分布 [L, H, n], 图像总量 [L, H])。"""
    mass = x.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        q = x / mass[..., None]
    q[~np.isfinite(q)] = np.nan
    return q, mass


def layer_maps(x: np.ndarray, heads: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """返回每层的图像分布 [L, n]（对所选头平均）和每层的图像总量 [L]。heads: [L, H] 布尔。"""
    q, mass = norm_heads(x)
    if heads is None:
        heads = np.ones(q.shape[:2], dtype=bool)
    L = q.shape[0]
    maps = np.full((L, q.shape[2]), np.nan)
    ms = np.full(L, np.nan)
    for l in range(L):
        sel = heads[l] & np.isfinite(q[l]).all(-1)
        if sel.any():
            maps[l] = q[l, sel].mean(0)
            ms[l] = mass[l, sel].mean()
    return maps, ms


def topk_idx(v: np.ndarray, frac: float = TOP) -> np.ndarray:
    k = max(1, int(math.ceil(frac * v.size)))
    return np.argsort(-v)[:k]


def js(p: np.ndarray, q: np.ndarray) -> float:
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(p > 0, p * np.log(p / m), 0.0).sum()
        b = np.where(q > 0, q * np.log(q / m), 0.0).sum()
    return float(0.5 * (a + b))


def _half_mean(maps: np.ndarray, idx: np.ndarray) -> Optional[np.ndarray]:
    sub = maps[idx]
    sub = sub[np.isfinite(sub).all(-1)]
    if not len(sub):
        return None
    m = sub.mean(0)
    s = m.sum()
    return m / s if s > 0 else None


def item_metrics(ref: np.ndarray, cmp: np.ndarray, heads: Optional[np.ndarray]) -> Optional[Dict[str, float]]:
    """ref / cmp: 同一阶段 [L, H, n]。"""
    if ref.size == 0 or cmp.size == 0 or ref.shape != cmp.shape:
        return None
    rm, rmass = layer_maps(ref, heads)
    cm, cmass = layer_maps(cmp, heads)
    L = rm.shape[0]
    halves = (np.arange(0, L, 2), np.arange(1, L, 2))
    vals = defaultdict(list)
    for a, b in ((0, 1), (1, 0)):
        r_a, r_b = _half_mean(rm, halves[a]), _half_mean(rm, halves[b])
        c_a, c_b = _half_mean(cm, halves[a]), _half_mean(cm, halves[b])
        if r_a is None or r_b is None or c_a is None or c_b is None:
            continue
        E = topk_idx(r_a)
        out_mask = np.ones(r_b.size, dtype=bool)
        out_mask[E] = False
        eps = 1e-12
        L_r = (r_b[out_mask].mean() + eps) / (r_b[E].mean() + eps)
        L_c = (c_b[out_mask].mean() + eps) / (c_b[E].mean() + eps)
        vals["dlogL"].append(math.log(L_c / L_r))
        vals["dE"].append(float(c_b[E].sum() - r_b[E].sum()))
        vals["retention"].append(float(np.isin(topk_idx(c_b), E).mean()))
        U = topk_idx(c_a - r_a)
        vals["dU"].append(float((c_b[U].sum() - r_b[U].sum()) / (r_b[U].sum() + eps)))
    if not vals:
        return None
    out = {k: float(np.mean(v)) for k, v in vals.items()}
    r_all, c_all = _half_mean(rm, np.arange(L)), _half_mean(cm, np.arange(L))
    out["js"] = js(r_all, c_all) if r_all is not None and c_all is not None else float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        out["dmass"] = float(np.nanmean(np.log(cmass / rmass)))
    return out


def layer_profile(ref: np.ndarray, cmp: np.ndarray, heads: Optional[np.ndarray]) -> List[float]:
    """每层的 E 区份额变化；E 用另一半层的参照图来挑。"""
    rm, _ = layer_maps(ref, heads)
    cm, _ = layer_maps(cmp, heads)
    L = rm.shape[0]
    halves = (np.arange(0, L, 2), np.arange(1, L, 2))
    out = []
    for l in range(L):
        other = halves[1] if l % 2 == 0 else halves[0]
        r_o = _half_mean(rm, other)
        if r_o is None or not np.isfinite(rm[l]).all() or not np.isfinite(cm[l]).all():
            out.append(float("nan"))
            continue
        E = topk_idx(r_o)
        r_l, c_l = rm[l] / rm[l].sum(), cm[l] / cm[l].sum()
        out.append(float(c_l[E].sum() - r_l[E].sum()))
    return out


# ---------------------------------------------------------------------------------------------
def bootstrap_ci(x: np.ndarray, reps: int = 2000, seed: int = 0) -> Tuple[float, float]:
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(reps, len(x)), replace=True).mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def signflip_p(x: np.ndarray, reps: int = 5000, seed: int = 0) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    obs = abs(x.mean())
    s = rng.choice([-1.0, 1.0], size=(reps, len(x)))
    null = np.abs((s * x).mean(1))
    return float((1 + (null >= obs).sum()) / (reps + 1))


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1
    # 平均秩处理并列
    for v in np.unique(allv):
        idx = allv == v
        if idx.sum() > 1:
            ranks[idx] = ranks[idx].mean()
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ---------------------------------------------------------------------------------------------
def select_image_heads(ref_files: Dict[str, str], stage: str = "think", frac: float = 0.25) -> Optional[np.ndarray]:
    """在参照组上挑“看图最多”的头：各题图像总量平均后取前 frac。"""
    acc, n = None, 0
    for p in ref_files.values():
        d = _load(p)
        x = d.get(stage)
        if x is None or x.size == 0:
            continue
        m = x.sum(-1)
        acc = m if acc is None else acc + m
        n += 1
    if acc is None:
        return None
    acc = acc / n
    thr = np.quantile(acc, 1 - frac)
    return acc >= thr


def flip_labels_from_runs(run_dirs: List[str], crit: str = "answer", method: str = "plain") -> Dict[str, bool]:
    """中性答对的题（按 plain 中性）里，指定方法在用户说错时语义答案是否跟着用户（True = 改口）。"""
    from .evaluate import load_run, reference_sets
    runs = [load_run(d) for d in run_dirs]
    C, _, _ = reference_sets(runs, crit)
    out = {}
    for run in runs:
        if run["method"] != method:
            continue
        for r in run["rows"]:
            if r["condition"] == "user_wrong" and r["id"] in (C or set()) and r["pred"] is not None:
                out[r["id"]] = r["pred"] == r["claim"]
    return out


SHIFT = {"dlogL": 1, "dE": -1, "dU": 1, "retention": -1, "js": 1}   # 乘上符号后，越大 = 越偏离中性


def analyze(ref_spec: str, cmp_specs: List[str], out_dir: str, labels: Optional[Dict[str, bool]] = None,
            heatmap_ids: Optional[List[str]] = None, reps: int = 2000) -> str:
    os.makedirs(out_dir, exist_ok=True)
    rname, rdir, rtag = parse_source(ref_spec)
    ref_files = load_source(rdir, rtag)
    heads_img = select_image_heads(ref_files)
    lines = ["# 图像注意力区域比较", "",
             f"参照组：`{rname}`（{rdir}，{rtag}），{len(ref_files)} 题。",
             "每个 (层, 头) 先在图像内部归一化；区域用一半标准层挑选、在另一半上测量。"
             "“看图头”= 参照组思考阶段图像注意力总量最高的 25% 的头。", ""]
    summary = {"ref": ref_spec, "comparisons": {}}
    rows_csv = []
    for spec in cmp_specs:
        cname, cdir, ctag = parse_source(spec)
        cmp_files = load_source(cdir, ctag)
        ids = sorted(set(ref_files) & set(cmp_files))
        summary["comparisons"][cname] = {}
        lines += [f"## {cname} vs {rname}（{len(ids)} 题配对）", "",
                  "| 阶段 | 头 | n | dE（E 区份额变化） | dlogL | dU（U 区相对变化） | 留存率 | JS | 图像总量对数比 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        per_item = defaultdict(dict)
        profiles = defaultdict(list)
        for i in ids:
            dr, dc = _load(ref_files[i]), _load(cmp_files[i])
            for st in STAGES:
                for hname, hmask in (("全部", None), ("看图头", heads_img)):
                    m = item_metrics(dr.get(st, np.zeros(0)), dc.get(st, np.zeros(0)), hmask)
                    if m is None:
                        continue
                    per_item[(st, hname)][i] = m
                    rows_csv.append({"comparison": cname, "id": i, "stage": st, "heads": hname, **m,
                                     "flip": "" if labels is None or i not in labels else int(labels[i])})
                if dr.get(st) is not None and dr[st].size and dc.get(st) is not None and dc[st].shape == dr[st].shape:
                    profiles[st].append(layer_profile(dr[st], dc[st], heads_img))
        for st in STAGES:
            for hname in ("全部", "看图头"):
                items = per_item.get((st, hname), {})
                if not items:
                    continue
                cell = {}
                for k in ("dE", "dlogL", "dU", "retention", "js", "dmass"):
                    x = np.array([v[k] for v in items.values()], dtype=float)
                    lo, hi = bootstrap_ci(x, reps)
                    cell[k] = {"mean": float(np.nanmean(x)), "ci": [lo, hi],
                               "p": signflip_p(x - (0.2 if k == "retention" else 0.0))}
                summary["comparisons"][cname][f"{st}|{hname}"] = {"n": len(items), "metrics": cell}

                def f(k):
                    c = cell[k]
                    p = c["p"]
                    return f"{c['mean']:+.4f} [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]" + (f" p={p:.3g}" if k != "js" else "")
                lines.append(f"| {st} | {hname} | {len(items)} | {f('dE')} | {f('dlogL')} | {f('dU')} | "
                             f"{cell['retention']['mean']:.3f} | {cell['js']['mean']:.4f} | {f('dmass')} |")
        lines += ["", "留存率的 p 值检验的是“是否不同于 0.2”（随机位置的期望值）；dU 的 p 值检验是否不同于 0。", ""]
        # 每层曲线
        if profiles:
            lines += ["每层 E 区份额变化（看图头，均值）：", ""]
            for st, prof in profiles.items():
                arr = np.array(prof, dtype=float)
                means = np.nanmean(arr, 0)
                summary["comparisons"][cname][f"layer_profile|{st}"] = [float(x) for x in means]
                lines.append(f"- {st}：" + " ".join(f"{x:+.3f}" for x in means))
            lines.append("")
        # 与改口的关系
        if labels:
            lines += ["与改口的关系（中性答对的题；AUC > 0.5 = 改口的题偏离更大）：", "",
                      "| 阶段 | 头 | 改口 / 守住 | " + " | ".join(f"AUC({k})" for k in SHIFT) + " |",
                      "|---|---|---|" + "---|" * len(SHIFT)]
            for st in STAGES:
                for hname in ("全部", "看图头"):
                    items = per_item.get((st, hname), {})
                    lab = {i: labels[i] for i in items if i in labels}
                    if not lab:
                        continue
                    res = {}
                    for k, sgn in SHIFT.items():
                        pos = np.array([sgn * items[i][k] for i in lab if lab[i]], dtype=float)
                        neg = np.array([sgn * items[i][k] for i in lab if not lab[i]], dtype=float)
                        res[k] = auc(pos, neg)
                    n_pos = sum(lab.values())
                    summary["comparisons"][cname][f"auc|{st}|{hname}"] = {"n_flip": n_pos,
                                                                          "n_held": len(lab) - n_pos, **res}
                    lines.append(f"| {st} | {hname} | {n_pos} / {len(lab) - n_pos} | " +
                                 " | ".join(f"{res[k]:.3f}" for k in SHIFT) + " |")
            lines.append("")
        if heatmap_ids:
            for i in heatmap_ids:
                if i in ref_files and i in cmp_files:
                    try:
                        heatmap(ref_files[i], cmp_files[i], os.path.join(out_dir, "heatmaps", f"{i}__{cname}.png"),
                                heads_img)
                    except Exception as e:  # 热图只是辅助
                        lines.append(f"（{i} 热图失败：{type(e).__name__}: {e}）")
    if rows_csv:
        keys = list(rows_csv[0].keys())
        with open(os.path.join(out_dir, "items.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows_csv)
    report = "\n".join(lines)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return report


# ---------------------------------------------------------------------------------------------
def _colormap(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0, 1)
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def heatmap(ref_path: str, cmp_path: str, out_png: str, heads: Optional[np.ndarray] = None, stage: str = "think"):
    """四联图：原图 | 中性 | 比较组 | 差值（红 = 增加，蓝 = 减少）。"""
    from PIL import Image
    dr, dc = _load(ref_path), _load(cmp_path)
    gh, gw = [int(x) for x in dr["grid_hw"]]
    img_path = str(dr.get("image", dc.get("image", "")))
    base = Image.open(img_path).convert("RGB") if img_path and os.path.exists(img_path) \
        else Image.new("RGB", (gw * 28, gh * 28), (128, 128, 128))
    W, Hh = base.size
    rm, _ = layer_maps(dr[stage], heads)
    cm, _ = layer_maps(dc[stage], heads)
    r = np.nanmean(rm, 0).reshape(gh, gw)
    c = np.nanmean(cm, 0).reshape(gh, gw)
    vmax = max(r.max(), c.max())
    panels = [base]
    for m in (r, c):
        rgb = Image.fromarray(_colormap(m / vmax)).resize((W, Hh), Image.NEAREST)
        panels.append(Image.blend(base, rgb, 0.5))
    d = c - r
    a = np.abs(d).max() or 1.0
    rgb = np.zeros(d.shape + (3,), dtype=np.uint8)
    rgb[..., 0] = (np.clip(d / a, 0, 1) * 255).astype(np.uint8)
    rgb[..., 2] = (np.clip(-d / a, 0, 1) * 255).astype(np.uint8)
    panels.append(Image.blend(base, Image.fromarray(rgb).resize((W, Hh), Image.NEAREST), 0.6))
    out = Image.new("RGB", (W * 4, Hh), (255, 255, 255))
    for k, p in enumerate(panels):
        out.paste(p, (k * W, 0))
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    out.save(out_png)


# ---------------------------------------------------------------------------------------------
def validate(replay_dir: str, saved_pattern: str, array: Optional[str] = None, class_index: int = 0,
             tag_conditions: Tuple[str, ...] = ("neutral", "user_wrong")) -> Dict[str, object]:
    """把回放得到的逐位置图像总量和预实验保存的 generation_masses 对比。
    saved_pattern 里用 {id} 和 {condition} 占位，文件可以是 .npy，或 .npz（用 array 指定数组名）。
    保存的数组为 [行, 层, 头, 类别] 时取 class_index 那一类，并另外报告按类别和归一化后的结果。"""
    res = []
    for cond in tag_conditions:
        for i, p in sorted(load_source(replay_dir, cond).items()):
            sp = saved_pattern.format(id=i, condition=cond)
            if not os.path.exists(sp):
                res.append({"id": i, "condition": cond, "status": "saved_missing"})
                continue
            if sp.endswith(".npz"):
                with np.load(sp) as z:
                    s = z[array] if array else z[z.files[0]]
            else:
                s = np.load(sp)
            ours = _load(p).get("pos_image_mass")
            if ours is None:
                res.append({"id": i, "condition": cond, "status": "no_positions_in_replay"})
                continue
            ours = ours.astype(np.float64)
            variants = {}
            if s.ndim == 4:
                variants["raw"] = s[..., class_index].astype(np.float64)
                tot = s.sum(-1).astype(np.float64)
                with np.errstate(invalid="ignore", divide="ignore"):
                    variants["normalized"] = s[..., class_index] / tot
            else:
                variants["raw"] = s.astype(np.float64)
            n = min(len(ours), len(variants["raw"]))
            row = {"id": i, "condition": cond, "status": "ok", "rows_ours": int(len(ours)),
                   "rows_saved": int(len(variants["raw"]))}
            for k, v in variants.items():
                a, b = ours[:n].ravel(), v[:n].ravel()
                ok = np.isfinite(a) & np.isfinite(b)
                if ok.sum() < 2:
                    continue
                row[f"{k}_max_abs_diff"] = float(np.abs(a[ok] - b[ok]).max())
                row[f"{k}_mean_abs_diff"] = float(np.abs(a[ok] - b[ok]).mean())
                row[f"{k}_corr"] = float(np.corrcoef(a[ok], b[ok])[0, 1])
            res.append(row)
    ok = [r for r in res if r.get("status") == "ok"]
    summ = {"items": len(res), "ok": len(ok)}
    for k in ("raw", "normalized"):
        cs = [r[f"{k}_corr"] for r in ok if f"{k}_corr" in r]
        if cs:
            summ[f"{k}_corr_min"] = float(min(cs))
            summ[f"{k}_corr_median"] = float(np.median(cs))
            summ[f"{k}_mean_abs_diff_max"] = float(max(r[f"{k}_mean_abs_diff"] for r in ok if f"{k}_mean_abs_diff" in r))
    summ["rows_mismatch"] = sum(1 for r in ok if r["rows_ours"] != r["rows_saved"])
    return {"summary": summ, "items": res}
