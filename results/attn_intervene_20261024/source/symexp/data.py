# -*- coding: utf-8 -*-
"""数据准备与条件构造。

base.jsonl：每题一行 {"id", "image"（相对 base.jsonl 的路径）, "question", "answer"（yes/no）, "source"}
conditions.jsonl：默认每题分配一种压力方式，3 行（neutral、user_wrong、user_right）；
                  --ptype-mode all 时每题 11 行（neutral；user_wrong × 5；user_right × 5）
"""

import os
import random
from collections import defaultdict
from typing import List, Optional

from PIL import Image

from .prompts import PRESSURE_TEMPLATES
from .store import read_jsonl, write_jsonl

SYCO_REPO = "mahzzz/Sycophancy-Dataset"
SYCO_REVISION = "e8c3b29be7dc264b682f2182f412d1646fcf457d"   # 预实验用的 revision
PATHVQA_REPO = "flaviagiammarino/path-vqa"                    # 原始 PathVQA（备选来源）


def _norm_yn(s) -> Optional[str]:
    t = str(s or "").strip().lower().strip(".")
    return t if t in ("yes", "no") else None


def _save(img: Image.Image, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.convert("RGB").save(path)


def _stratify(rows: List[dict], n: int, seed: int) -> List[dict]:
    if not n or n >= len(rows):
        return rows
    by = defaultdict(list)
    for r in rows:
        by[r["answer"]].append(r)
    rng = random.Random(seed)
    out = []
    keys = sorted(by)
    per = n // len(keys)
    for i, k in enumerate(keys):
        want = per + (1 if i < n % len(keys) else 0)
        pool = sorted(by[k], key=lambda r: r["id"])
        rng.shuffle(pool)
        out += pool[:want]
    return sorted(out, key=lambda r: r["id"])


def prepare_sycophancy(out_dir: str, n: int = 0, seed: int = 0, subset_filter: str = "path",
                       revision: str = SYCO_REVISION) -> List[dict]:
    """从预实验用的数据集取 PathVQA 部分的全部是非题；n>0 时按答案分层抽 n 题。"""
    from datasets import load_dataset
    ds = load_dataset(SYCO_REPO, split="train", revision=revision)
    rows = []
    for ex in ds:
        if subset_filter.lower() not in str(ex["dataset"]).lower():
            continue
        ans = _norm_yn(ex["correct_answer"])
        if ans is None:
            continue
        pid = f"PV-{int(ex['sample_idx']):03d}"
        rel = f"images/{pid}.png"
        _save(ex["image"], os.path.join(out_dir, rel))
        rows.append({"id": pid, "image": rel, "question": str(ex["question"]).strip(), "answer": ans,
                     "source": f"{SYCO_REPO}@{revision[:8]}"})
    print(f"[prepare] {len(rows)} yes/no items from {SYCO_REPO} (filter '{subset_filter}')")
    rows = _stratify(rows, n, seed)
    write_jsonl(os.path.join(out_dir, "base.jsonl"), rows)
    return rows


def prepare_pathvqa(out_dir: str, n: int = 300, seed: int = 0, split: str = "test") -> List[dict]:
    """从原始 PathVQA 取是非题，按答案分层抽 n 题。"""
    from datasets import load_dataset
    ds = load_dataset(PATHVQA_REPO, split=split)
    cand = [i for i, ex in enumerate(ds) if _norm_yn(ex["answer"]) is not None]
    tmp = [{"id": f"PQ-{split}-{i:05d}", "idx": i, "answer": _norm_yn(ds[i]["answer"])} for i in cand]
    picked = _stratify(tmp, n, seed)
    rows = []
    for r in picked:
        ex = ds[r["idx"]]
        rel = f"images/{r['id']}.png"
        _save(ex["image"], os.path.join(out_dir, rel))
        rows.append({"id": r["id"], "image": rel, "question": str(ex["question"]).strip(), "answer": r["answer"],
                     "source": f"{PATHVQA_REPO}:{split}"})
    print(f"[prepare] {len(rows)} items sampled from {len(cand)} yes/no items of {PATHVQA_REPO}:{split}")
    write_jsonl(os.path.join(out_dir, "base.jsonl"), rows)
    return rows


def prepare_synthetic(out_dir: str, n: int = 8, seed: int = 0) -> List[dict]:
    """给 mock 测试用的假数据。"""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        pid = f"SY-{i:03d}"
        rel = f"images/{pid}.png"
        _save(Image.new("RGB", (rng.randint(200, 640), rng.randint(200, 480)),
                        (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))), os.path.join(out_dir, rel))
        rows.append({"id": pid, "image": rel, "question": f"Is necrosis present in sample {i}?",
                     "answer": "yes" if i % 2 == 0 else "no", "source": "synthetic"})
    write_jsonl(os.path.join(out_dir, "base.jsonl"), rows)
    return rows


def assign_ptypes(base: List[dict], seed: int = 0) -> dict:
    """每题分配一种压力方式：按标准答案分层后轮流分配，使 5 种方式的题数、yes/no 比例都尽量均衡。"""
    types = list(PRESSURE_TEMPLATES)
    rng = random.Random(seed)
    out, k = {}, 0
    by = defaultdict(list)
    for it in base:
        by[it["answer"]].append(it["id"])
    for ans in sorted(by):
        ids = sorted(by[ans])
        rng.shuffle(ids)
        for i in ids:
            out[i] = types[k % len(types)]
            k += 1
    return out


def build_conditions(base_path: str, out_path: str, sep: str = "\n", ptype_mode: str = "assign",
                     ptype_seed: int = 0) -> List[dict]:
    """ptype_mode="assign"：每题一种压力方式，3 个条件（neutral、user_wrong、user_right）。
    ptype_mode="all"：每题 5 种压力方式全跑，11 个条件。"""
    base = read_jsonl(base_path)
    base_dir = os.path.dirname(os.path.abspath(base_path))
    out_dir = os.path.dirname(os.path.abspath(out_path))
    assigned = assign_ptypes(base, ptype_seed) if ptype_mode == "assign" else None
    rows = []
    for it in base:
        img = os.path.relpath(os.path.join(base_dir, it["image"]), out_dir)
        gold = it["answer"]
        wrong = "no" if gold == "yes" else "yes"
        common = {"id": it["id"], "image": img, "question": it["question"], "answer": gold}
        rows.append({**common, "condition": "neutral", "ptype": "none", "claim": None, "message": it["question"]})
        ptypes = [assigned[it["id"]]] if assigned else list(PRESSURE_TEMPLATES)
        for cond, a in (("user_wrong", wrong), ("user_right", gold)):
            for ptype in ptypes:
                rows.append({**common, "condition": cond, "ptype": ptype, "claim": a,
                             "message": it["question"] + sep + PRESSURE_TEMPLATES[ptype].format(a=a)})
    write_jsonl(out_path, rows)
    per = len(rows) // max(1, len(base))
    print(f"[conditions] mode={ptype_mode}: {len(base)} items x {per} = {len(rows)} rows -> {out_path}")
    if assigned:
        cnt = defaultdict(lambda: [0, 0])
        gold_of = {it["id"]: it["answer"] for it in base}
        for i, p in assigned.items():
            cnt[p][0 if gold_of[i] == "yes" else 1] += 1
        print("[conditions] items per pressure type (yes/no): " +
              ", ".join(f"{p} {v[0] + v[1]} ({v[0]}/{v[1]})" for p, v in cnt.items()))
    return rows
