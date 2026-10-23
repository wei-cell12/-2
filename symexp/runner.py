# -*- coding: utf-8 -*-
"""批量运行与续跑。"""

import json
import os
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from .backends import make_backend
from .config import Config
from .methods import METHODS, Context, final_key
from .store import read_jsonl


def select_items(items: List[dict], conditions: Optional[List[str]], ptypes: Optional[List[str]],
                 limit: int, ids: Optional[List[str]]) -> List[dict]:
    if conditions:
        items = [it for it in items if it["condition"] in conditions]
    if ptypes:
        items = [it for it in items if it["ptype"] in ptypes or it["condition"] == "neutral"]
    if ids:
        s = set(ids)
        items = [it for it in items if it["id"] in s]
    if limit:
        keep = list(dict.fromkeys(it["id"] for it in items))[:limit]
        ks = set(keep)
        items = [it for it in items if it["id"] in ks]
    return items


def run(cfg: Config, method: str, data_path: str, run_dir: str, seeds: List[int],
        conditions: Optional[List[str]] = None, ptypes: Optional[List[str]] = None, limit: int = 0,
        ids: Optional[List[str]] = None) -> None:
    os.makedirs(run_dir, exist_ok=True)
    from .prompts import PROMPT_VERSION
    old_cfg = os.path.join(run_dir, "config.json")
    # 只有本方法用到 v1.1 改动的提示词；plain、SRT 的旧结果可以照常续跑
    if method.startswith("ours") and os.path.exists(old_cfg) and read_jsonl(os.path.join(run_dir, "final.jsonl")):
        prev = json.load(open(old_cfg, encoding="utf-8")).get("prompt_version")
        if prev != PROMPT_VERSION:
            raise SystemExit(f"{run_dir} 里已有提示词版本 {prev} 的结果，当前是 {PROMPT_VERSION}。"
                             f"请换一个输出目录，不要把两个版本混在一起。")
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"method": method, "prompt_version": PROMPT_VERSION, "data": os.path.abspath(data_path), "seeds": seeds,
                   "conditions": conditions, "ptypes": ptypes, "limit": limit, "config": cfg.to_dict(),
                   "started": time.strftime("%Y-%m-%d %H:%M:%S")}, f, ensure_ascii=False, indent=2)
    items = select_items(read_jsonl(data_path), conditions, ptypes, limit, ids)
    backend = make_backend(cfg)
    ctx = Context(cfg, backend, run_dir, os.path.dirname(os.path.abspath(data_path)), method)
    fn = METHODS[method]

    todo = [(it, s) for it in items for s in seeds
            if not (ctx.final.get(final_key(it, s)) and not ctx.final.get(final_key(it, s)).get("error"))]
    print(f"[run] method={method} items={len(items)} seeds={seeds} todo={len(todo)} -> {run_dir}", file=sys.stderr)

    # 本方法按 (题, seed) 分组，组内依次跑，保证第 2、3 步只跑一次
    groups = defaultdict(list)
    for it, s in todo:
        groups[(it["id"], s) if method.startswith("ours") else (final_key(it, s),)].append((it, s))

    def work(units):
        out = []
        for it, s in units:
            try:
                out.append(fn(ctx, it, s))
            except Exception as e:  # 单题失败不影响其他题
                row = {"key": final_key(it, s), "method": method, "id": it["id"], "condition": it["condition"],
                       "ptype": it["ptype"], "seed": s, "error": f"{type(e).__name__}: {e}",
                       "traceback": traceback.format_exc()[-2000:]}
                out.append(row)
            ctx.final.put(out[-1])
        return len(out)

    done, last = 0, 0
    total = len(todo)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, cfg.workers)) as ex:
        futs = [ex.submit(work, g) for g in groups.values()]
        for fu in as_completed(futs):
            done += fu.result()
            if done == total or done - last >= 50:
                last = done
                print(f"  {done}/{total}  {(time.time() - t0) / 60:.1f} min", file=sys.stderr)
    errs = [r for r in ctx.final.values() if r.get("error")]
    if errs:
        print(f"[run] {len(errs)} units failed; first error: {errs[0]['error']}", file=sys.stderr)
