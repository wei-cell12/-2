# -*- coding: utf-8 -*-
"""“用户说对”条件（没有冲突的对照）：同样的压力话术，但主张换成正确答案。

  setup  从一个实验目录（只读）复制 items.jsonl，给清单里每一题加一条消息
         messages["{压力}_right"] = 问题 + "\\n" + 同一话术模板.format(正确答案)，
         写 candidates.json（id、condition = "{压力}_right"），不复制别的记录。
  run    对 candidates 里的每一条生成一次（与观察阶段完全相同的设置，mode=observe），
         记录写在 root/obs/records/，claim 字段为正确答案。之后照常 fp-export / fp-import --phase obs、
         fp-entropy --root root、fp-public。
"""

import json
import os
import shutil
import sys

from ..prompts import PRESSURE_TEMPLATES


def cond_right(p: str) -> str:
    return f"{p}_right"


def setup(src: str, dst: str, cases_path: str) -> int:
    from .pipeline import _atomic_json, load_items
    from ..store import write_jsonl
    if os.path.exists(os.path.join(dst, "candidates.json")):
        raise SystemExit(f"{dst} 里已经有 candidates.json，不覆盖；请换一个新目录")
    cs = json.load(open(cases_path, encoding="utf-8"))
    items = load_items(src)
    byid = {i["id"]: i for i in items}
    cands = []
    for iid, p in cs["intervene"]:
        if iid not in byid:
            raise SystemExit(f"{iid} 不在 {src}/items.jsonl 里")
        it = byid[iid]
        it["messages"][cond_right(p)] = it["question"] + "\n" + PRESSURE_TEMPLATES[p].format(a=it["gold"])
        cands.append({"id": iid, "condition": cond_right(p), "pressure_type": p, "user_claim": it["gold"],
                      "user_claim_correct": True, "image_sha256": it["image_sha256"]})
    os.makedirs(dst, exist_ok=True)
    write_jsonl(os.path.join(dst, "items.jsonl"), items)
    _atomic_json(os.path.join(dst, "candidates.json"),
                 {"stats": {"source": src, "cases_file": os.path.abspath(cases_path), "n": len(cands),
                            "note": "用户说对（主张 = 正确答案），没有冲突的对照"}, "candidates": cands})
    print(f"[fp-agree-setup] {dst}：{len(cands)} 条“用户说对”条件，例如 {cands[0]['id']}："
          f"{byid[cands[0]['id']]['messages'][cands[0]['condition']]!r}", file=sys.stderr)
    return len(cands)


def run(cfg, root: str, shard: str = "", seed: int = 42) -> None:
    from .pipeline import Engine, _atomic_json, _check_no_failure, _done, _fail, _save_result, _shard, load_items
    _check_no_failure(root)
    items = {i["id"]: i for i in load_items(root)}
    cands = json.load(open(os.path.join(root, "candidates.json"), encoding="utf-8"))["candidates"]
    tasks = _shard(cands, shard)
    todo = [c for c in tasks if not _done(root, "obs", c["id"], c["condition"])]
    print(f"[fp-agree] 本 worker {len(tasks)} 条，待做 {len(todo)} 条", file=sys.stderr)
    if not todo:
        return
    eng = Engine(cfg)
    for k, c in enumerate(todo):
        it = items[c["id"]]
        task = f"obs/{c['id']}__{c['condition']}"
        try:
            rec, arr = eng.run(it, c["condition"], "observe", None, seed)
            _save_result(root, "obs", it, c["condition"], "observe", rec, arr)
            p = os.path.join(root, "obs", "records", f"{c['id']}__{c['condition']}.json")
            r = json.load(open(p, encoding="utf-8"))
            r.update(claim=it["gold"], user_claim_correct=True)
            _atomic_json(p, r)
        except BaseException as e:
            _fail(root, task, e)
            raise
        print(f"  {k + 1}/{len(todo)} {task} tokens={rec['n_out']} {rec['seconds']}s "
              f"selfcheck_ratio={rec['controller'].get('selfcheck_ratio_max')}", file=sys.stderr)
