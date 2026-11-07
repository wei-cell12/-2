# -*- coding: utf-8 -*-
"""读图区域的偏移，主要来自提示里用户那句话，还是思考开头对用户答案的复述？（ClockQA，试跑几题）

同一条受压回答，取开头复述段（思考开头到第一个空行，并且这一段里出现了用户给的时间）之后的 W 个读图 token。
在这同一批读图 token 上，比较四种上下文下的看图注意力：
  A：提示里有用户那句话 + 保留复述段（原回答）
  B：提示里没有用户那句话 + 保留复述段
  C：提示里有用户那句话 + 删掉复述段
  D：提示里没有用户那句话 + 删掉复述段（最干净，作参照）
用户那句话的作用 = B→A、D→C；复述段的作用 = D→B、C→A。只回放、不生成。
"""

import json
import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np

W = 64
MAX_RESTATE = 200
VARIANTS = ["A", "B", "C", "D"]
VNAME = {"A": "A 有用户那句话 + 有复述（原回答）", "B": "B 没有用户那句话 + 有复述",
         "C": "C 有用户那句话 + 没有复述", "D": "D 没有用户那句话 + 没有复述（参照）"}


def find_restatement(pieces: List[str], skip: int, claim: str, max_tok: int = MAX_RESTATE) -> Optional[int]:
    """pieces：输出 token 的文字。从 skip 开始累加，找到"已经出现用户的时间"之后的第一个空行（\\n\\n）结束处，
    返回复述段之后第一个 token 的下标；找不到返回 None。用户的时间按"时:分"匹配（12:35:00 → 12:35）。"""
    key = ":".join(claim.split(":")[:2]) if claim else ""
    text, seen = "", False
    for t in range(skip, min(len(pieces), skip + max_tok)):
        text += pieces[t]
        if key and key in text:
            seen = True
        if seen and "\n\n" in pieces[t]:
            return t + 1
    return None


class RowRecorder:
    """记录输出行 [r0, r0+W) 上，每层所有头加起来分给各图块的注意力，以及看图总量。"""

    def __init__(self, torch, r0: int, w: int, n_in: int, image_idx, layers: List[int], chunk: int = 256):
        self.torch, self.r0, self.w, self.n_in, self.image_idx, self.layers, self.chunk = torch, r0, w, n_in, image_idx, list(layers), chunk
        self.selfcheck_left, self.selfcheck_diffs = 0, []
        self.dist = np.zeros((w, len(layers), int(image_idx.numel())))
        self.mass = np.zeros((w, len(layers)))

    def stages_for(self, q_pos):
        return self.torch.full_like(q_pos, -1)

    def accumulate(self, layer_idx, probs, stages, q_pos):
        if layer_idx not in self.layers:
            return
        li = self.layers.index(layer_idx)
        rows = (q_pos - (self.n_in - 1)).cpu().numpy() - self.r0
        hit = np.flatnonzero((rows >= 0) & (rows < self.w))
        if not len(hit):
            return
        p = probs[0][:, hit, :][:, :, self.image_idx.to(probs.device)].float()
        self.dist[rows[hit], li] = p.sum(0).cpu().numpy()
        self.mass[rows[hit], li] = p.sum(-1).mean(0).cpu().numpy()


def summarize(dist, mass, lm, hands) -> dict:
    d = dist[:, lm].sum((0, 1))
    d = d / max(d.sum(), 1e-12)
    return {"dist": d, "mass": float(mass[:, lm].mean()), "hands": float(d[hands].sum())}


def _jsd(p, q):
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(np.where(a > 0, a * np.log(np.maximum(a, 1e-300) / np.maximum(b, 1e-300)), 0.0)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def effects(s: dict) -> dict:
    """s：四种上下文的 summarize 结果。用户那句话 / 复述段各自造成的变化（两种情况取平均）。"""
    j = lambda a, b: _jsd(s[a]["dist"], s[b]["dist"])
    return {"jsd_to_D": {v: _jsd(s[v]["dist"], s["D"]["dist"]) for v in VARIANTS},
            "prompt_effect_jsd": 0.5 * (j("B", "A") + j("D", "C")),
            "restate_effect_jsd": 0.5 * (j("D", "B") + j("C", "A")),
            "prompt_effect_mass": 0.5 * ((s["A"]["mass"] - s["B"]["mass"]) + (s["C"]["mass"] - s["D"]["mass"])),
            "restate_effect_mass": 0.5 * ((s["B"]["mass"] - s["D"]["mass"]) + (s["A"]["mass"] - s["C"]["mass"])),
            "prompt_effect_hands": 0.5 * ((s["A"]["hands"] - s["B"]["hands"]) + (s["C"]["hands"] - s["D"]["hands"])),
            "restate_effect_hands": 0.5 * ((s["B"]["hands"] - s["D"]["hands"]) + (s["A"]["hands"] - s["C"]["hands"])),
            "mass": {v: s[v]["mass"] for v in VARIANTS}, "hands": {v: s[v]["hands"] for v in VARIANTS}}


def run(cfg, root: str, ent_dir: str, model_key: str, ids: List[str], image_dirs: Optional[List[str]] = None,
        label: str = "") -> dict:
    from .. import attn
    from .conflict import _engine, claim_of
    from .entropy import _batch
    from .generate import build_inputs
    from .reading_groups import image_grid, think_skip
    from .sources import dial_geometry
    from .trigger_regions import hand_shape, load_entropy, resolve_image, resolve_source
    os.makedirs(root, exist_ok=True)
    ent = load_entropy(ent_dir, model_key, ["ClockQA"])
    keys = [k for k in sorted(ent) if k[0] in ids]
    if not keys:
        raise SystemExit(f"熵差实验里没有这些题：{ids}")
    eng = _engine(cfg)
    b = eng.backend
    torch, tok = b.torch, b.tokenizer
    targets = attn.find_targets(b.model)
    all_l = sorted(int(m.layer_idx) for _, m in targets)
    sliding = np.array([bool(getattr(m, "is_sliding", False)) for _, m in sorted(targets, key=lambda x: int(x[1].layer_idx))])
    attn.install(b.model, verbose=False)
    kinds = {"全局层": ~sliding, "滑动窗口层": sliding} if sliding.any() else {"全部层": np.ones(len(all_l), bool)}
    res = {}
    for iid, c in keys:
        e = ent[(iid, c)]
        job, item = e["job"], e["job"]["item"]
        rec = json.load(open(resolve_source(job), encoding="utf-8"))
        image = resolve_image(job, rec, image_dirs)
        out = [int(x) for x in rec["output_ids"]]
        pieces = [tok.decode([x], skip_special_tokens=False) for x in out[:MAX_RESTATE + 20]]
        skip = think_skip(tok, cfg, out)
        claim = claim_of(item, c)
        end = find_restatement(pieces, skip, claim)
        if end is None or end + W > len(out):
            print(f"  {iid}__{c}：开头 {MAX_RESTATE} 个 token 里没找到含用户时间的复述段（或之后不够 {W} 个 token），跳过", file=sys.stderr)
            res[f"{iid}__{c}"] = {"skipped": True}
            continue
        restated = tok.decode(out[skip:end], skip_special_tokens=False)
        kept = out[:skip] + out[end:]                    # 删掉复述段，保留思考开头标记
        variants = {"A": (item["messages"][c], out, end), "B": (item["messages"]["neutral"], out, end),
                    "C": (item["messages"][c], kept, skip), "D": (item["messages"]["neutral"], kept, skip)}
        stats, secs, hands = {k: {} for k in kinds}, {}, None
        for v, (msg, seq, r0) in variants.items():
            _, inp = build_inputs(b, image, msg)
            ids_in = inp["input_ids"][0].tolist()
            img_tok = attn.image_token_id(b)
            image_idx = torch.tensor([i for i, x in enumerate(ids_in) if x == img_tok])
            if hands is None:
                grid = image_grid(b, inp)
                hands = hand_shape(dial_geometry(image), grid, item["gold"])
            r = RowRecorder(torch, r0, W, len(ids_in), image_idx, all_l, chunk=cfg.attn_chunk)
            r.passthrough, r.rel_checked = True, {}
            inner = getattr(b.model, "model", None)
            if inner is not None and hasattr(inner, "rope_deltas"):
                inner.rope_deltas = None
            t0 = time.time()
            with torch.inference_mode(), attn.recording(r):
                try:
                    b.model(**_batch(b, inp, seq[: r0 + W]), use_cache=False, return_dict=True, logits_to_keep=1)
                except TypeError:
                    b.model(**_batch(b, inp, seq[: r0 + W]), use_cache=False, return_dict=True)
            secs[v] = round(time.time() - t0, 2)
            for kind, lm in kinds.items():
                stats[kind][v] = summarize(r.dist, r.mass, lm, hands)
        reading = tok.decode(out[end:end + W], skip_special_tokens=False)
        res[f"{iid}__{c}"] = {"id": iid, "condition": c, "group": e["group"], "restated": restated, "reading": reading,
                              "restate_tokens": end - skip, "seconds": secs,
                              "effects": {k: effects(s) for k, s in stats.items()}}
        print(f"  {iid}__{c}（{e['group']}）复述段 {end - skip} 个 token；用时 {secs}", file=sys.stderr)
    with open(os.path.join(root, "restate_ablation.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    gname = {"flip": "被带偏", "hold": "守住"}
    cname = {"belief": "用户说错", "belief_right": "用户说对"}
    L = [f"# 读图区域的偏移来自提示里的用户那句话，还是思考开头的复述：{label}（试跑）", "",
         f"同一条受压回答，取开头复述段（思考开头到第一个空行、其中出现了用户给的时间）之后的 {W} 个读图 token。"
         "在这同一批 token 上比较四种上下文：", ""] + [f"- {VNAME[v]}" for v in VARIANTS] + [
         "", "**指标**：",
         "- 与 D 的差别：看图分布与 D 的 JSD，越大说明偏得越多；",
         "- 看图总量：全部注意力里落在图像上的比例；",
         "- 指针区域：看图注意力里落在真实指针上的比例。",
         "**用户那句话的作用** = 加上这句话带来的变化（B→A 和 D→C 的平均）；**复述段的作用** = 加上复述段带来的变化（D→B 和 C→A 的平均）。", ""]
    for key, v in res.items():
        if v.get("skipped"):
            L += [f"## {key}：跳过（没找到复述段）", ""]
            continue
        L += [f"## {v['id']} · {cname[v['condition']]}（{gname.get(v['group'], v['group'])}）", "",
              f"删掉的复述段（{v['restate_tokens']} 个 token）：{v['restated']!r}", "",
              f"比较的读图 token：{v['reading']!r}", ""]
        for kind, ef in v["effects"].items():
            L += [f"**{kind}**", "", "| 上下文 | 与 D 的差别（JSD） | 看图总量 | 指针区域 |", "|---|---:|---:|---:|"]
            for x in VARIANTS:
                L.append(f"| {VNAME[x]} | {ef['jsd_to_D'][x]:.4f} | {ef['mass'][x] * 100:.1f}% | {ef['hands'][x] * 100:.1f}% |")
            L += ["", "| | 用户那句话的作用 | 复述段的作用 |", "|---|---:|---:|",
                  f"| 看图分布变化（JSD） | {ef['prompt_effect_jsd']:.4f} | {ef['restate_effect_jsd']:.4f} |",
                  f"| 看图总量变化（百分点） | {ef['prompt_effect_mass'] * 100:+.2f} | {ef['restate_effect_mass'] * 100:+.2f} |",
                  f"| 指针区域变化（百分点） | {ef['prompt_effect_hands'] * 100:+.2f} | {ef['restate_effect_hands'] * 100:+.2f} |", ""]
    with open(os.path.join(root, "restate_ablation_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"[fp-restate] {len(res)} 条 → {root}/restate_ablation_report.md", file=sys.stderr)
    return res
