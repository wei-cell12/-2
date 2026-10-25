# -*- coding: utf-8 -*-
"""五压力实验的汇总。两部分分开报告：

A. 注意力变化（观察阶段）：只用中性答对的题；每种压力分别和中性比较，三个阶段，全部头与看图头；
   受压结果分“采纳 / 守住”时的 AUC；三阶段热图（统一色阶）。
B. 干预效果：全部候选（中性答对、受压答错），按“次”统计 对 / 采纳 / 其他错 / 未解，
   分压力类型、陈述型单列；restore_E 与 rand_matched 逐次配对，按图像分组做精确符号翻转检验；
   干预强度（关键区域份额前后、抬高量、剂量差、被截断的行数）。
"""

import csv
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import numpy as np

from . import PRESSURES
from .pipeline import load_items, load_judgments, load_records, outcome
from .stats import cluster_bootstrap_rate, cluster_signflip

OUTS = ["对", "采纳", "其他错", "未解"]


def _fmt_counts(c: Counter, n: int) -> str:
    return " | ".join(f"{c.get(k, 0)}" for k in OUTS) + f" | {n}"


def part_a(root: str, out_dir: str, heatmaps: int = 12) -> List[str]:
    from ..attn_analysis import analyze
    items = {i["id"]: i for i in load_items(root)}
    js = load_judgments(root, "obs")
    nc = [i for i in sorted(items) if outcome(js.get(f"obs|{i}|neutral"), items[i]["gold"], None) == "对"]
    attn_dir = os.path.join(root, "obs", "attn")
    lines = ["## A. 注意力变化（中性答对的题，每种压力 vs 中性）", "",
             f"中性答对 {len(nc)} 题。每种压力的完整报告见 `A_attention/<压力>/report.md`。", "",
             "| 压力 | 阶段 | 头 | n | dE（关键区域份额变化） | dU（有利区域相对变化） | 图像总量对数比 | 采纳 / 守住 | AUC(dE) |",
             "|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    for p in PRESSURES:
        labels = {}
        for i in nc:
            o = outcome(js.get(f"obs|{i}|{p}"), items[i]["gold"], items[i]["claim"])
            if o in ("采纳", "对"):
                labels[i] = o == "采纳"
        sub = os.path.join(out_dir, "A_attention", p)
        analyze(f"neutral={attn_dir}@neutral", [f"{p}={attn_dir}@{p}"], sub, labels=labels,
                heatmap_ids=nc[:heatmaps], shared_scale=True, ids_filter=nc)
        s = json.load(open(os.path.join(sub, "summary.json"), encoding="utf-8"))["comparisons"].get(p, {})
        summary[p] = s
        for st in ("prefill_end", "think", "answer"):
            for h in ("全部", "看图头"):
                cell = s.get(f"{st}|{h}")
                if not cell:
                    continue
                m = cell["metrics"]
                a = s.get(f"auc|{st}|{h}", {})
                f = lambda k: f"{m[k]['mean']:+.4f} (p={m[k]['p']:.3g})"
                lines.append(f"| {p} | {st} | {h} | {cell['n']} | {f('dE')} | {f('dU')} | {f('dmass')} | "
                             f"{a.get('n_flip', '')} / {a.get('n_held', '')} | {a.get('dE', float('nan')):.3f} |")
    lines += ["", "dE < 0：中性时重点看的区域份额下降；图像总量对数比 < 0：对整张图的注意力减少。"
              "AUC(dE) > 0.5：采纳主张的观测，关键区域下降得更多。", ""]
    return lines, summary


def part_b(root: str, out_dir: str) -> List[str]:
    items = {i["id"]: i for i in load_items(root)}
    cpath = os.path.join(root, "candidates.json")
    if not os.path.exists(cpath):
        return ["## B. 干预效果", "", "还没有 candidates.json。", ""], {}
    cands = json.load(open(cpath, encoding="utf-8"))["candidates"]
    js = load_judgments(root, "int")
    arms = [a for a in ("restore_E", "rand_matched", "base") if os.path.isdir(os.path.join(root, "int", a, "records"))]
    recs = {a: {(r["id"], r["condition"]): r for r in load_records(root, f"int/{a}")} for a in arms}
    rows = []
    for c in cands:
        it = items[c["id"]]
        row = {"id": c["id"], "condition": c["condition"], "image": it["image_sha256"][:12],
               "native": c["pressure_outcome"]}
        for a in arms:
            r = recs[a].get((c["id"], c["condition"]))
            row[a] = None if r is None else outcome(js.get(f"int/{a}|{c['id']}|{c['condition']}"), it["gold"], it["claim"])
            if r is not None:
                cs = r["controller"]
                row[a + "_region"] = (cs.get("region_before"), cs.get("region_after"))
                row[a + "_delta"] = cs.get("mean_delta")
        rows.append(row)
    lines = ["## B. 干预效果（全部“中性答对、受压答错”的观测）", "",
             f"候选 {len(cands)} 次，来自 {len({c['id'] for c in cands})} 题、"
             f"{len({items[c['id']]['image_sha256'] for c in cands})} 张不同图像。", "",
             "| 组别 | " + " | ".join(OUTS) + " | 已完成 |", "|---|" + "---|" * (len(OUTS) + 1)]
    summary = {"n": len(cands), "arms": {}}
    for a in arms:
        done = [r for r in rows if r[a] is not None]
        cnt = Counter(r[a] for r in done)
        summary["arms"][a] = dict(cnt, done=len(done))
        lines.append(f"| {a} | " + _fmt_counts(cnt, len(done)) + " |")
    lines += ["", "原始受压输出全部是错的（选题条件），所以应以 rand_matched 为比较基准。", ""]
    # 按压力类型
    lines += ["### 按压力类型（答对次数 / 已完成）", "", "| 压力 | 次数 | " + " | ".join(arms) + " |",
              "|---|---|" + "---|" * len(arms)]
    for group, sel in [(p, lambda r, p=p: r["condition"] == p) for p in PRESSURES] + \
                      [("非陈述合计", lambda r: r["condition"] != "statement")]:
        g = [r for r in rows if sel(r)]
        cells = []
        for a in arms:
            d = [r for r in g if r[a] is not None]
            cells.append(f"{sum(r[a] == '对' for r in d)} / {len(d)}")
        lines.append(f"| {group} | {len(g)} | " + " | ".join(cells) + " |")
    lines.append("")
    # 按原始结果类型
    lines += ["### 按原始受压结果", "", "| 原始结果 | 次数 | " + " | ".join(f"{a}：对 / 采纳" for a in arms) + " |",
              "|---|---|" + "---|" * len(arms)]
    for nat in ("采纳", "其他错"):
        g = [r for r in rows if r["native"] == nat]
        cells = [f"{sum(r[a] == '对' for r in g if r[a])} / {sum(r[a] == '采纳' for r in g if r[a])}" for a in arms]
        lines.append(f"| {nat} | {len(g)} | " + " | ".join(cells) + " |")
    lines.append("")
    # 配对检验
    if "restore_E" in arms and "rand_matched" in arms:
        lines += ["### restore_E 与 rand_matched 逐次配对（按图像分组的符号翻转检验，双侧）", "",
                  "| 范围 | 指标 | 只有 E | 只有 rand | 两者 | 差（E − rand） | 检验 |", "|---|---|---|---|---|---|---|"]
        summary["paired"] = {}
        for scope, sel in (("全部", lambda r: True), ("非陈述", lambda r: r["condition"] != "statement"),
                           ("陈述", lambda r: r["condition"] == "statement")):
            g = [r for r in rows if sel(r) and r["restore_E"] is not None and r["rand_matched"] is not None
                 and "未解" not in (r["restore_E"], r["rand_matched"])]
            for name, target, sign in (("答对", "对", 1), ("采纳", "采纳", -1)):
                e = [r["restore_E"] == target for r in g]
                z = [r["rand_matched"] == target for r in g]
                d = [(r["image"], float(a) - float(b)) for r, a, b in zip(g, e, z)]
                t = cluster_signflip(d)
                only_e = sum(a and not b for a, b in zip(e, z))
                only_r = sum(b and not a for a, b in zip(e, z))
                both = sum(a and b for a, b in zip(e, z))
                summary["paired"][f"{scope}|{name}"] = dict(t, only_E=only_e, only_rand=only_r, both=both, n=len(g))
                lines.append(f"| {scope}（{len(g)}） | {name} | {only_e} | {only_r} | {both} | {only_e - only_r:+d} | "
                             f"p={t['p']:.4g}（{t['method']}，{t['nonzero_clusters']} 组） |")
        lines += ["", "未解的观测不进入配对。", ""]
    # 干预强度
    lines += ["### 干预强度（每次生成的平均值）", "", "| 组别 | 区域份额（前 → 后） | 平均抬高量 |", "|---|---|---|"]
    for a in arms:
        if a == "base":
            continue
        reg = [r[a + "_region"] for r in rows if r.get(a + "_region") and None not in r[a + "_region"]]
        dl = [r[a + "_delta"] for r in rows if r.get(a + "_delta") is not None]
        if reg:
            b, f = np.mean([x[0] for x in reg]), np.mean([x[1] for x in reg])
            lines.append(f"| {a} | {b:.3f} → {f:.3f} | {np.mean(dl):.3f} |")
    lines += ["", "rand_matched 的区域份额指随机区域本身；两组抬高量应相同（剂量匹配）。", ""]
    with open(os.path.join(out_dir, "B_items.csv"), "w", encoding="utf-8", newline="") as f:
        keys = ["id", "condition", "image", "native"] + arms
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return lines, summary


def analyze(root: str, out_dir: Optional[str] = None, heatmaps: int = 12) -> str:
    out_dir = out_dir or os.path.join(root, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    items = load_items(root)
    obs = load_records(root, "obs")
    lines = ["# PathVQA 五压力实验", "",
             f"题目 {len(items)} 道；观察记录 {len(obs)} 份。判读：Gemini 3.8 Flash（medium），答案和解释都对才算“对”。", ""]
    a_lines, a_sum = part_a(root, out_dir, heatmaps) if obs else (["## A", "", "还没有观察记录。", ""], {})
    b_lines, b_sum = part_b(root, out_dir)
    lines += a_lines + b_lines
    text = "\n".join(lines)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"A": a_sum, "B": b_sum}, f, ensure_ascii=False, indent=1)
    return text
