# -*- coding: utf-8 -*-
"""从一个已完成观察阶段的实验目录里，挑出指定的若干次受压观测，单独做干预（不改主实验）。

  setup   在新目录 dst 里放好 items.jsonl、这些题的中性和指定受压记录（json + npz）、手写的 candidates.json，
          之后直接对 dst 运行 fp-intervene --phase pilot / full、fp-export / fp-import --phase int、fp-public。
  region  只用 CPU：对每次受压观测，用中性 npz 构造参照（与干预同一规则），比较中性和受压时，
          看图头上关键区域在图像内部的份额、图像总注意力，按分组汇总。不写入 src。

cases 文件（JSON）：
  {"intervene": [["PV-000", "statement"], ...],      # 要干预的观测，按顺序；前 2 个用于预检
   "groups": {"A": [[id, cond], ...], "B": [...], ...}}   # region 用的分组（可以比 intervene 多）
"""

import json
import os
import shutil
import sys
from typing import Dict, List

import numpy as np

from .pipeline import _atomic_json, load_items
from .references import build, check

STAGES = ["think", "answer"]


def _cases(path: str) -> dict:
    d = json.load(open(path, encoding="utf-8"))
    d["intervene"] = [tuple(x) for x in d.get("intervene", [])]
    d["groups"] = {g: [tuple(x) for x in v] for g, v in d.get("groups", {}).items()}
    return d


def setup(src: str, dst: str, cases_path: str) -> int:
    cs = _cases(cases_path)
    if os.path.exists(os.path.join(dst, "candidates.json")):
        raise SystemExit(f"{dst} 里已经有 candidates.json，不覆盖；请换一个新目录")
    items = {i["id"]: i for i in load_items(src)}
    need = set()
    for iid, cond in cs["intervene"]:
        if iid not in items:
            raise SystemExit(f"{iid} 不在 {src}/items.jsonl 里")
        need.add((iid, "neutral"))
        need.add((iid, cond))
    for sub in ("records", "attn"):
        os.makedirs(os.path.join(dst, "obs", sub), exist_ok=True)
    for iid, cond in sorted(need):
        tag = f"{iid}__{cond}"
        rj = os.path.join(src, "obs", "records", tag + ".json")
        rn = os.path.join(src, "obs", "attn", tag + ".npz")
        if not (os.path.exists(rj) and os.path.exists(rn)):
            raise SystemExit(f"缺少观察记录 {tag}（json 或 npz）")
        shutil.copy(rj, os.path.join(dst, "obs", "records", tag + ".json"))
        shutil.copy(rn, os.path.join(dst, "obs", "attn", tag + ".npz"))
    shutil.copy(os.path.join(src, "items.jsonl"), os.path.join(dst, "items.jsonl"))
    cands = [{"id": iid, "condition": cond, "pressure_outcome": "采纳", "image_sha256": items[iid]["image_sha256"],
              "selected_by": "manual (Claude reading of thinking: type A)"} for iid, cond in cs["intervene"]]
    _atomic_json(os.path.join(dst, "candidates.json"),
                 {"stats": {"source": src, "cases_file": os.path.abspath(cases_path), "n": len(cands)},
                  "candidates": cands})
    print(f"[fp-subset] {dst}：{len(cands)} 次待干预，已复制 {len(need)} 份观察记录；预检用前 2 次："
          f"{cands[0]['id']}__{cands[0]['condition']}、{cands[1]['id']}__{cands[1]['condition']}", file=sys.stderr)
    return len(cands)


def _shares(npz_path: str, ref: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    heads = ref["heads"]
    out = {}
    with np.load(npz_path) as z:
        for s, st in enumerate(STAGES):
            p = z[st].astype(np.float64)                     # [16, 24, n]
            tot = p.sum(-1)
            q = p / np.maximum(tot[..., None], 1e-30)
            share = (q * ref["E_mask"][s]).sum(-1)
            out[st] = {"E_share": float(share[heads].mean()), "image_mass": float(tot[heads].mean()),
                       "count": int(z["count_" + st])}
    return out


def region(src: str, cases_path: str, out_path: str) -> dict:
    cs = _cases(cases_path)
    refs: Dict[str, dict] = {}
    rows = []
    for g, lst in cs["groups"].items():
        for iid, cond in lst:
            if iid not in refs:
                with np.load(os.path.join(src, "obs", "attn", f"{iid}__neutral.npz")) as z:
                    ref = build(iid, z["think"], z["answer"])
                c = check(ref)
                if not c["ok"]:
                    raise SystemExit(f"{iid} 参照检查失败：{c}")
                refs[iid] = {"ref": ref, "neutral": _shares(os.path.join(src, "obs", "attn", f"{iid}__neutral.npz"), ref)}
            n = refs[iid]["neutral"]
            p = _shares(os.path.join(src, "obs", "attn", f"{iid}__{cond}.npz"), refs[iid]["ref"])
            row = {"group": g, "id": iid, "condition": cond}
            for st in STAGES:
                if n[st]["count"] == 0 or p[st]["count"] == 0:
                    row[st] = None
                    continue
                row[st] = {"E_share_neutral": n[st]["E_share"], "E_share_pressure": p[st]["E_share"],
                           "dE": p[st]["E_share"] - n[st]["E_share"],
                           "mass_neutral": n[st]["image_mass"], "mass_pressure": p[st]["image_mass"],
                           "dmass": p[st]["image_mass"] - n[st]["image_mass"]}
            rows.append(row)
    summary = {}
    for g in cs["groups"]:
        for st in STAGES:
            v = [r[st] for r in rows if r["group"] == g and r[st] is not None]
            if v:
                summary[f"{g}|{st}"] = {"n": len(v), "dE_mean": float(np.mean([x["dE"] for x in v])),
                                        "dE_neg": int(sum(x["dE"] < 0 for x in v)),
                                        "dmass_mean": float(np.mean([x["dmass"] for x in v])),
                                        "E_share_neutral_mean": float(np.mean([x["E_share_neutral"] for x in v])),
                                        "E_share_pressure_mean": float(np.mean([x["E_share_pressure"] for x in v]))}
    res = {"src": src, "note": "看图头（中性思考阶段图像总量最高的 96 个）上的平均；E = 中性参照每阶段每头份额最高的 20% 图像块；"
                               "dE = 受压 − 中性的关键区域份额；dmass = 受压 − 中性的图像总注意力",
           "summary": summary, "rows": rows}
    _atomic_json(out_path, res)
    print(f"{'组|阶段':<14}{'n':>4}{'中性E份额':>11}{'受压E份额':>11}{'dE均值':>10}{'dE<0':>6}{'d图像总量':>11}", file=sys.stderr)
    for k, s in summary.items():
        print(f"{k:<14}{s['n']:>4}{s['E_share_neutral_mean']:>11.3f}{s['E_share_pressure_mean']:>11.3f}"
              f"{s['dE_mean']:>10.3f}{s['dE_neg']:>6}{s['dmass_mean']:>11.4f}", file=sys.stderr)
    return res
