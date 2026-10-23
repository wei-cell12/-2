# -*- coding: utf-8 -*-
"""只读前向回放：把已保存的完整 token 序列喂回模型，按阶段记录图像注意力。不生成、不改动已有结果。

清单（JSONL）每行一条序列，三种写法任选其一：
  {"id": "PV-000", "condition": "neutral", "image": "/abs/or/relative.jpg", "input_ids": [...], "output_ids": [...]}
  {"id": ..., "condition": ..., "image": ..., "ids": [...], "prompt_len": 1234}
  {"id": ..., "condition": ..., "image": ..., "ids_file": "seq.npy 或 seq.json", "prompt_len": 1234}
input_ids 是送进模型的完整提示（含模板、图像占位、以 <think> 结尾的生成起点），output_ids 是模型生成的全部 token
（思考、</think>、回答，含结束标记与否都可以）。image 的相对路径相对清单文件所在目录。
"""

import json
import os
import sys
import time
from typing import Optional

from PIL import Image

from .attn import replay_item, save_npz
from .store import read_jsonl


def _ids(row: dict, base_dir: str):
    if "input_ids" in row:
        ids = list(row["input_ids"]) + list(row.get("output_ids", []))
        return [int(x) for x in ids], len(row["input_ids"])
    if "ids" in row:
        return [int(x) for x in row["ids"]], int(row["prompt_len"])
    path = row["ids_file"]
    path = path if os.path.isabs(path) else os.path.join(base_dir, path)
    if path.endswith(".npy"):
        import numpy as np
        ids = np.load(path).reshape(-1).tolist()
    else:
        ids = json.load(open(path, encoding="utf-8"))
    return [int(x) for x in ids], int(row["prompt_len"])


def run_replay(cfg, manifest: str, out_dir: str, limit: int = 0, keep_positions: bool = True,
               ids_filter: Optional[list] = None) -> None:
    from .backends import make_backend
    rows = read_jsonl(manifest)
    if ids_filter:
        keep = set(ids_filter)
        rows = [r for r in rows if r["id"] in keep]
    if limit:
        seen, sel = [], []
        for r in rows:
            if r["id"] not in seen:
                seen.append(r["id"])
            if len(seen) <= limit:
                sel.append(r)
        rows = sel
    base_dir = os.path.dirname(os.path.abspath(manifest))
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "replay_log.jsonl")
    todo = [r for r in rows if not os.path.exists(os.path.join(out_dir, f"{r['id']}__{r['condition']}.npz"))]
    print(f"[attn-replay] {len(rows)} sequences, {len(todo)} to do -> {out_dir}", file=sys.stderr)
    if not todo:
        return
    backend = make_backend(cfg)
    if cfg.backend != "hf":
        raise SystemExit("attn-replay 需要 hf 后端")
    from .attn import install
    layer_ids = install(backend.model)
    with open(os.path.join(out_dir, "replay_config.json"), "w", encoding="utf-8") as f:
        json.dump({"manifest": os.path.abspath(manifest), "layer_ids": layer_ids, "config": cfg.to_dict(),
                   "attn_implementation": getattr(backend.model.config, "_attn_implementation", None),
                   "started": time.strftime("%Y-%m-%d %H:%M:%S")}, f, ensure_ascii=False, indent=2)
    for k, r in enumerate(todo):
        t0 = time.time()
        ids, prompt_len = _ids(r, base_dir)
        img_path = r["image"] if os.path.isabs(r["image"]) else os.path.join(base_dir, r["image"])
        image = Image.open(img_path).convert("RGB")
        status = {"id": r["id"], "condition": r["condition"], "total_len": len(ids), "prompt_len": prompt_len}
        try:
            data = replay_item(backend, ids, prompt_len, image, keep_positions=keep_positions,
                               chunk=cfg.attn_chunk, selfcheck=cfg.attn_selfcheck)
            path = os.path.join(out_dir, f"{r['id']}__{r['condition']}.npz")
            save_npz(path, data, image=os.path.abspath(img_path), id=r["id"], condition=r["condition"])
            sc = [float(x) for x in data["selfcheck_max_abs_diff"]]
            status.update(ok=True, seconds=round(time.time() - t0, 2), selfcheck_max_abs_diff=sc,
                          counts={s: float(data[f"count_{s}"]) for s in ("prefill_end", "think", "answer")},
                          n_img=int(data["n_img"]), close_pos=int(data["close_pos"]))
        except Exception as e:
            status.update(ok=False, error=f"{type(e).__name__}: {e}")
            print(f"[attn-replay] {r['id']} {r['condition']} failed: {status['error']}", file=sys.stderr)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(status, ensure_ascii=False) + "\n")
        print(f"  {k + 1}/{len(todo)} {r['id']} {r['condition']} ok={status.get('ok')} "
              f"{status.get('seconds', '')}s selfcheck={status.get('selfcheck_max_abs_diff', '')}", file=sys.stderr)
