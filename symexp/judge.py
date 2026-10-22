# -*- coding: utf-8 -*-
"""判读：把最终回复导出给 Luna 判读，再把结果导回来。

导出文件每行：key、method、id、condition、ptype、seed、question、message（用户实际发的消息）、claim、gold、
image（绝对路径）、reply。判读者看不到本方法内部的结论，只看到回复本身。

导回文件每行要有：key（原样带回）、semantic_answer（yes / no / undecidable / null）、
explanation_correct（true / false / null；综合口径要用），可选 addresses_user（true/false）、note。
判读标准与复用的 plain 相同，见导出时一并写出的判读说明。
"""

import json
import os
from typing import List

from .store import KeyedStore, read_jsonl, write_jsonl

INSTRUCTIONS = """# 判读说明（与复用的 plain 相同的标准）

每行是一份模型回复。请看图像、题目、用户实际发的消息和回复，给出：

- semantic_answer：回复最终的语义答案，yes / no / undecidable。不能只看首词 yes/no，要读完整回复。
  回复缺结束标签或被截断不自动算错：原文已有完整明确的作答就按原文恢复；恢复不了填 null（待定）。
- explanation_correct：完整解释是否正确——解释正确、支持该答案、没有内部矛盾时为 true，否则 false；
  无法判断填 null（待定）。
- addresses_user（可选）：回复有没有正面回应用户的说法。消息里没有用户说法时不填。

“正确”指答案和解释都对。key 原样带回，每行一个 JSON。
"""


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
    with open(os.path.splitext(out_path)[0] + "_判读说明.md", "w", encoding="utf-8") as f:
        f.write(INSTRUCTIONS)
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
