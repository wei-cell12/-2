# -*- coding: utf-8 -*-
"""熵谷出现之后，后面的词把注意力分给了谁（离线回放熵差实验的原回答，不生成）。

对每条回答，取开头复述之后第一次熵谷报警的位置 t0（每段连续负熵差里第一次 ≤ −1）。回放完整提示 + 原回答（到思考结束），
记录从 t0 − PRE 到思考结束每个词、每个标准注意力层（Gemma 分全局 / 滑动窗口层）、所有头平均的注意力，按键的来源分组求和：
  图像            提示里的图像 token
  用户那句话      提示里问题之后用户说的那句话（按"问题 + 一句话"的格式找出，只用于分析）
  开头的熵差片段  开头复述里的负熵差片段（含 ≤ −1 的连续负值段）
  之后的熵差片段  复述之后的负熵差片段（同上；只统计当前词之前已经写出的部分）
  其余            以上之外（提示里的其他部分、模型自己写的其他文字）
熵差取自熵差实验的逐词记录（与原回答逐词对齐）。
"""

import json
import os
import sys
from typing import List, Optional

import numpy as np

PRE = 64
GROUPS = ["image", "user", "echo_open", "echo_later", "rest"]
GROUP_ZH = {"image": "图像", "user": "用户那句话", "echo_open": "开头的熵差片段", "echo_later": "之后的熵差片段", "rest": "其余"}


def neg_runs(dH: List[float], start: int, stop: int, thresh: float = -1.0) -> List[List[int]]:
    """[start, stop) 里含 ≤ thresh 的连续负值段（每段的位置列表）。"""
    out, run = [], []
    for t in range(start, stop):
        if dH[t] < 0:
            run.append(t)
        else:
            if any(dH[x] <= thresh for x in run):
                out.append(run)
            run = []
    if any(dH[x] <= thresh for x in run):
        out.append(run)
    return out


def first_alarm(dH: List[float], start: int, stop: int, thresh: float = -1.0) -> Optional[int]:
    inseg, fired = False, False
    for t in range(len(dH)):
        v = dH[t]
        if v < 0:
            if not inseg:
                inseg, fired = True, False
            if not fired and v <= thresh and start <= t < stop:
                return t
        else:
            inseg = False
    return None


class GroupRecorder:
    """记录输出行 [r0, r1) 上，每层、头平均的注意力按键的来源分组求和：shares[row, layer, group]。"""

    def __init__(self, torch, n_in: int, r0: int, r1: int, labels: np.ndarray, layers: List[int], chunk: int = 256):
        self.torch, self.n_in, self.r0, self.r1, self.layers, self.chunk = torch, n_in, r0, r1, list(layers), chunk
        self.labels = labels                               # 每个键位置的组号（GROUPS 下标；"其余"不单列）
        self.selfcheck_left, self.selfcheck_diffs = 0, []
        self.shares = np.zeros((r1 - r0, len(layers), len(GROUPS)))
        self._oh = {}

    def stages_for(self, q_pos):
        return self.torch.full_like(q_pos, -1)

    def _onehot(self, kL, device):
        key = (kL, str(device))
        if key not in self._oh:
            t = self.torch
            lab = t.as_tensor(self.labels[:kL], device=device, dtype=t.long)
            oh = t.zeros(kL, len(GROUPS), device=device)
            oh[t.arange(kL, device=device), lab] = 1.0
            self._oh[key] = oh
        return self._oh[key]

    def accumulate(self, layer_idx, probs, stages, q_pos):
        if layer_idx not in self.layers:
            return
        rows = (q_pos - (self.n_in - 1)).cpu().numpy()
        hit = np.flatnonzero((rows >= self.r0) & (rows < self.r1))
        if not len(hit):
            return
        p = probs[0][:, hit, :].float()                    # [H, r, kL]
        g = p.mean(0) @ self._onehot(p.shape[-1], p.device)  # [r, G]
        self.shares[rows[hit] - self.r0, self.layers.index(layer_idx)] = g.cpu().numpy()


def run(cfg, root: str, ent_dir: str, model_key: str, cases: List[str], image_dirs: Optional[List[str]] = None) -> None:
    from .. import attn
    from .conflict import _engine
    from .entropy import _batch
    from .generate import build_inputs
    from .mask_query import claim_span
    from .read_anchor import first_paragraph
    from .trigger_regions import load_entropy, read_steps, resolve_image, resolve_source
    os.makedirs(root, exist_ok=True)
    ent = load_entropy(ent_dir, model_key, ["ClockQA", "PathVQA", "PENDULUM"])
    eng = _engine(cfg)
    b = eng.backend
    torch, tok = b.torch, b.tokenizer
    targets = attn.find_targets(b.model)
    layers = sorted(int(m.layer_idx) for _, m in targets)
    sliding = {int(m.layer_idx): bool(getattr(m, "is_sliding", False)) for _, m in targets}
    attn.install(b.model, verbose=False)
    res = {}
    for case in cases:
        iid, c = case.split("__", 1)
        e = ent[(iid, c)]
        job, item = e["job"], e["job"]["item"]
        rec = json.load(open(resolve_source(job), encoding="utf-8"))
        image = resolve_image(job, rec, image_dirs)
        out = [int(x) for x in rec["output_ids"]]
        dH, ok, toks, n_think = read_steps(e["csv"])
        acc, end = "", None
        st = next((t for t in range(len(toks)) if toks[t].strip() not in ("", "<|channel>", "thought", "<think>")), 0)
        for t in range(st, min(len(toks), st + 200)):
            acc += toks[t]
            if first_paragraph(acc.lstrip("\n")) is not None:
                end = t + 1
                break
        if end is None:
            raise SystemExit(f"{case}：前 200 个词里找不到开头一段")
        t0 = first_alarm(dH, end, n_think)
        if t0 is None:
            raise SystemExit(f"{case}：复述之后没有熵谷报警")
        msg = item["messages"][c]
        _, inp = build_inputs(b, image, msg)
        ids = inp["input_ids"][0].tolist()
        n_in = len(ids)
        q = item["messages"]["neutral"] if msg.startswith(item["messages"]["neutral"]) else item["question"]
        span = claim_span(lambda x: tok.decode(x, skip_special_tokens=False), ids, msg, q)
        img = [i for i, x in enumerate(ids) if x == attn.image_token_id(b)]
        runs_open = neg_runs(dH, 0, end)
        runs_later = neg_runs(dH, end, n_think)
        total = n_in + n_think
        labels = np.full(total, GROUPS.index("rest"), np.int64)
        labels[img] = GROUPS.index("image")
        labels[span] = GROUPS.index("user")
        for r in runs_open:
            labels[[n_in + x for x in r]] = GROUPS.index("echo_open")
        for r in runs_later:
            labels[[n_in + x for x in r]] = GROUPS.index("echo_later")
        r0, r1 = max(0, t0 - PRE), n_think
        R = GroupRecorder(torch, n_in, r0, r1, labels, layers, chunk=cfg.attn_chunk)
        R.passthrough, R.rel_checked = True, {}
        inner = getattr(b.model, "model", None)
        if inner is not None and hasattr(inner, "rope_deltas"):
            inner.rope_deltas = None
        with torch.inference_mode(), attn.recording(R):
            try:
                b.model(**_batch(b, inp, out[:n_think]), use_cache=False, return_dict=True, logits_to_keep=1)
            except TypeError:
                b.model(**_batch(b, inp, out[:n_think]), use_cache=False, return_dict=True)
        sl = np.array([sliding[l] for l in layers])
        kinds = {"全局层": ~sl, "滑动窗口层": sl} if sl.any() else {"全部层": np.ones(len(layers), bool)}
        run_end = t0
        while run_end < len(dH) and dH[run_end] < 0:
            run_end += 1
        wins = {"熵谷之前 64 个词": (max(r0, t0 - PRE), t0), "熵谷那一段（连续负熵差）": (t0, run_end),
                "熵谷之后 50 个词": (run_end, min(r1, run_end + 50)), "熵谷之后到思考结束": (run_end, r1)}
        summ = {}
        for kind, lm in kinds.items():
            summ[kind] = {}
            for wn, (a, z) in wins.items():
                if z <= a:
                    continue
                v = R.shares[a - r0:z - r0][:, lm].mean((0, 1))
                summ[kind][wn] = {g: float(v[k]) for k, g in enumerate(GROUPS[:-1])}
                summ[kind][wn]["rest"] = float(1 - v[:-1].sum())
        curve = {kind: R.shares[:, lm].mean(1)[:, :-1].round(5).tolist() for kind, lm in kinds.items()}
        res[case] = {"id": iid, "condition": c, "group": e["group"], "dataset": e["dataset"], "t0": t0, "run_end": run_end,
                     "opening_end": end, "n_think": n_think, "span_text": tok.decode([ids[i] for i in span]),
                     "runs_open": ["".join(toks[x] for x in r) for r in runs_open],
                     "runs_later": ["".join(toks[x] for x in r) for r in runs_later],
                     "context": "".join(toks[max(0, t0 - 40):t0]) + "⟪" + toks[t0] + "⟫" + "".join(toks[t0 + 1:t0 + 40]),
                     "windows": summ, "curve_rows_from": r0, "curve": curve}
        print(f"  {case}（{e['group']}）熵谷在第 {t0} 个词，思考共 {n_think} 个词；开头熵差片段 {len(runs_open)} 段，之后 {len(runs_later)} 段", file=sys.stderr)
        with open(os.path.join(root, "valley_attn.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False)
    report(root)


def report(root: str) -> None:
    res = json.load(open(os.path.join(root, "valley_attn.json"), encoding="utf-8"))
    gname = {"flip": "被带偏", "hold": "守住"}
    L = ["# 熵谷出现之后，后面的词把注意力分给了谁", "",
         "回放熵差实验的原回答（不生成）。数字是所有头平均、所选层平均后，落在各来源上的注意力占全部注意力的百分比：",
         "- 图像：提示里的图像；用户那句话：提示里用户说的那句话；",
         "- 开头的熵差片段：开头复述里抄来的用户答案（含 ≤ −1 的连续负熵差段）；之后的熵差片段：复述之后出现的同类片段（只算当前词之前已写出的）；",
         "- 其余：提示里的其他部分、模型自己写的其他文字。", ""]
    for case, v in res.items():
        L += [f"## {v['id']}（{gname.get(v['group'], v['group'])}）", "",
              f"熵谷在第 {v['t0']} 个词（思考共 {v['n_think']} 个词）：{v['context']!r}", "",
              f"用户那句话：{v['span_text']!r}", "",
              f"开头的熵差片段：{v['runs_open']}", "", f"之后的熵差片段：{v['runs_later'][:8]}", ""]
        for kind, ws in v["windows"].items():
            L += [f"**{kind}**", "", "| 时段 | " + " | ".join(GROUP_ZH[g] for g in GROUPS) + " |", "|---|" + "---:|" * len(GROUPS)]
            for wn, s in ws.items():
                L.append(f"| {wn} | " + " | ".join(f"{s[g] * 100:.2f}%" for g in GROUPS) + " |")
            L.append("")
    with open(os.path.join(root, "valley_attn_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"[fp-valleyattn] {len(res)} 条 → {root}/valley_attn_report.md", file=sys.stderr)
