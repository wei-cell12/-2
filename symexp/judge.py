# -*- coding: utf-8 -*-
"""判读：把最终回复导出给 Luna 判读，再把结果导回来。

导出文件每行：key、method、id、condition、ptype、seed、question、message（用户实际发的消息）、claim、gold、
image（绝对路径）、reply。判读者看不到本方法内部的结论，只看到回复本身。

导回文件每行至少要有：key（原样带回）、semantic_answer（yes / no / undecidable / null），
可选：addresses_user（回复是否正面回应了用户的说法，true/false）、explanation_correct（true/false）、note。
"""

import json
import os
from typing import List

from .store import KeyedStore, read_jsonl, write_jsonl


def export_requests(run_dirs: List[str], out_path: str) -> int:
    rows = []
    for d in run_dirs:
        cfg = json.load(open(os.path.join(d, "config.json"), encoding="utf-8"))
        data_dir = cfg.get("config", {}).get("image_root") or os.path.dirname(cfg["data"])
        for r in read_jsonl(os.path.join(d, "final.jsonl")):
            if r.get("error"):
                continue
            img = r["image"] if os.path.isabs(r["image"]) else os.path.abspath(os.path.join(data_dir, r["image"]))
            rows.append({"key": f"{r['method']}|{r['key']}", "method": r["method"], "id": r["id"],
                         "condition": r["condition"], "ptype": r["ptype"], "seed": r["seed"],
                         "question": r["question"], "message": r["message"], "claim": r.get("claim"),
                         "gold": r["gold"], "image": img, "reply": r["reply"]})
    # 去重（同一 key 多次续跑时只留最后一条）
    uniq = {}
    for r in rows:
        uniq[r["key"]] = r
    write_jsonl(out_path, uniq.values())
    return len(uniq)


def import_judgments(run_dirs: List[str], judgments_path: str) -> int:
    js = {r["key"]: r for r in read_jsonl(judgments_path)}
    n = 0
    for d in run_dirs:
        store = KeyedStore(os.path.join(d, "judgments.jsonl"))
        for r in read_jsonl(os.path.join(d, "final.jsonl")):
            k = f"{r['method']}|{r['key']}"
            if k in js:
                j = dict(js[k])
                j["key"] = r["key"]
                sa = j.get("semantic_answer")
                j["semantic_answer"] = sa.strip().lower() if isinstance(sa, str) else None
                store.put(j)
                n += 1
    return n
