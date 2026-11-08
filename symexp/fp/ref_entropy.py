# -*- coding: utf-8 -*-
"""离线检验：熵差的参照分支不带复述段时，读图那一段会不会出现熵差谷（只回放已有回答，不生成）。

同一条受压回答（熵差实验用的那条），逐词算三个熵（都是"已写到这里，下一个词"的分布熵，回放同一段文字）：
  H_P ：完整提示（有用户那句话）+ 原回答；
  H_N1：提示去掉用户那句话 + 原回答（= 现在的熵差分支，保留复述段）；
  H_N2：提示去掉用户那句话 + 原回答删掉开头复述段（= 锚点一的参照分支）。复述段里的词没有 H_N2。
  ΔH1 = H_P − H_N1（现在的熵差），ΔH2 = H_P − H_N2（新熵差）。
报警规则相同：思考里、复述段之后，每段连续负值里第一次 ≤ −1。
"""

import csv
import json
import os
import re
import sys
from typing import Dict, List, Optional

import numpy as np

THRESH = -1.0
READ_W = 128
CHUNK = 256


def entropies(torch, logits, chunk: int = CHUNK) -> List[float]:
    out = []
    for s in range(0, logits.shape[0], chunk):
        lp = torch.log_softmax(logits[s:s + chunk].float(), -1)
        out += (-(lp.exp() * lp).sum(-1)).cpu().tolist()
    return out


def alarms(dH: List[Optional[float]], ok: List[bool], start: int, stop: int, thresh: float = THRESH) -> List[int]:
    """与 trigger_regions.online_triggers 相同；只在 [start, stop) 且 ok 的位置报警；dH 为 None 的位置当作 0。"""
    out, inseg, fired = [], False, False
    for t, v in enumerate(dH):
        v = 0.0 if v is None else v
        if v < 0:
            if not inseg:
                inseg, fired = True, False
            if not fired and v <= thresh and ok[t] and start <= t < stop:
                out.append(t)
                fired = True
        else:
            inseg = False
    return out


def first_after(text_pieces: List[str], start: int, pattern: str) -> Optional[int]:
    """从 start 起逐词累加文字，第一次匹配 pattern（正则，不区分大小写）的那个词的位置。"""
    acc = ""
    rx = re.compile(pattern, re.I)
    for t in range(start, len(text_pieces)):
        acc += text_pieces[t]
        if rx.search(acc):
            return t
    return None


def claim_pattern(dataset: str, claim: Optional[str], wrong: Optional[str]) -> Optional[str]:
    """用户答案在回答里再次出现的匹配：ClockQA 按"时:分"；PathVQA 不做（yes/no 太常见）；PENDULUM 按用户说的答案词。"""
    if dataset == "ClockQA" and claim:
        return re.escape(":".join(claim.split(":")[:2]))
    if dataset == "PENDULUM" and wrong:
        return r"\b" + re.escape(wrong) + r"\b"
    return None


# ---------------------------------------------------------------- 回放（GPU）
def run(cfg, root: str, ent_dir: str, model_key: str, datasets: Optional[List[str]] = None, ids: Optional[List[str]] = None,
        image_dirs: Optional[List[str]] = None, shard: str = "") -> None:
    from .conflict import _engine, claim_of
    from .entropy import _logits
    from .generate import build_inputs
    from .pipeline import _shard
    from .read_anchor import first_paragraph
    from .reading_groups import think_skip
    from .trigger_regions import DATASETS, load_entropy, resolve_image, resolve_source
    from .valley_inject import claim_answer
    os.makedirs(os.path.join(root, "records"), exist_ok=True)
    ent = load_entropy(ent_dir, model_key, datasets or DATASETS)
    keys = sorted(k for k in ent if not ids or k[0] in ids)
    keys = _shard(keys, shard)
    todo = [k for k in keys if not os.path.exists(os.path.join(root, "records", f"{k[0]}__{k[1]}.json"))]
    print(f"[fp-refent] 本进程 {len(keys)} 条，待做 {len(todo)} 条", file=sys.stderr)
    eng = None
    for n, (iid, c) in enumerate(todo):
        e = ent[(iid, c)]
        job, item = e["job"], e["job"]["item"]
        rec = json.load(open(resolve_source(job), encoding="utf-8"))
        image = resolve_image(job, rec, image_dirs)
        if eng is None:
            eng = _engine(cfg)
        b = eng.backend
        torch, tok = b.torch, b.tokenizer
        out = [int(x) for x in rec["output_ids"]]
        pieces = [tok.decode([x], skip_special_tokens=False) for x in out]
        skip = think_skip(tok, cfg, out)
        end, kind, acc = None, None, ""
        for t in range(skip, min(len(out), skip + 200)):
            acc += pieces[t]
            kind = first_paragraph(acc)
            if kind is not None:
                end = t + 1
                break
        _, inp_p = build_inputs(b, image, item["messages"][c])
        _, inp_n = build_inputs(b, image, item["messages"]["neutral"])
        H = {}
        for name, inp, seq in (("P", inp_p, out), ("N1", inp_n, out)):
            lg = _logits(b, inp, seq)
            H[name] = entropies(torch, lg)
            del lg
        H2 = [None] * len(out)
        if end is not None and kind == "user":
            seq2 = out[:skip] + out[end:]
            lg = _logits(b, inp_n, seq2)
            h = entropies(torch, lg)
            del lg
            for t in range(end, len(out)):
                H2[t] = h[skip + (t - end)]
        elif end is not None:                         # 没有复述段：新旧参照相同
            H2 = list(H["N1"])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        rows = list(csv.DictReader(open(e["csv"], encoding="utf-8")))
        m = min(len(rows), len(out))
        dp = [abs(float(rows[i]["H_pressure"]) - H["P"][i]) for i in range(m)]
        dn = [abs(float(rows[i]["H_neutral"]) - H["N1"][i]) for i in range(m)]
        res = {"id": iid, "condition": c, "dataset": e["dataset"], "group": e["group"], "skip": skip, "restate_end": end,
               "restate_kind": kind, "claim": claim_of(item, c), "wrong_claim": claim_answer(item.get("claim")),
               "pieces": pieces, "phase": [r["phase"] for r in rows[:len(out)]],
               "forced": [r["forced_by_original_budget"] == "True" for r in rows[:len(out)]],
               "H_P": [round(x, 5) for x in H["P"]], "H_N1": [round(x, 5) for x in H["N1"]],
               "H_N2": [None if x is None else round(x, 5) for x in H2],
               "check_vs_entropy_experiment": {"H_P_absdiff_median": float(np.median(dp)), "H_P_absdiff_max": float(max(dp)),
                                               "H_N1_absdiff_median": float(np.median(dn)), "H_N1_absdiff_max": float(max(dn))}}
        p = os.path.join(root, "records", f"{iid}__{c}.json")
        with open(p + ".tmp", "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False)
        os.replace(p + ".tmp", p)
        ck = res["check_vs_entropy_experiment"]
        print(f"  {n + 1}/{len(todo)} {e['dataset']} {iid}__{c}（{e['group']}）{len(out)} 个 token，开头一段到 {end}（{kind}）；"
              f"与熵差实验对比 H_P 差中位 {ck['H_P_absdiff_median']:.4f}、最大 {ck['H_P_absdiff_max']:.3f}，"
              f"H_N1 差中位 {ck['H_N1_absdiff_median']:.4f}、最大 {ck['H_N1_absdiff_max']:.3f}", file=sys.stderr)


# ---------------------------------------------------------------- 汇总（CPU）
def _one(r: dict) -> Optional[dict]:
    end = r["restate_end"]
    if end is None:
        return None
    T = len(r["pieces"])
    n_think = max([i for i, ph in enumerate(r["phase"]) if ph == "think"] or [0]) + 1
    ok = [r["phase"][i] == "think" and not r["forced"][i] if i < len(r["phase"]) else False for i in range(T)]
    d1 = [r["H_P"][t] - r["H_N1"][t] for t in range(T)]
    d2 = [None if r["H_N2"][t] is None else r["H_P"][t] - r["H_N2"][t] for t in range(T)]
    a1, a2 = alarms(d1, ok, end, n_think), alarms(d2, ok, end, n_think)
    pat = claim_pattern(r["dataset"], r["claim"], r["wrong_claim"])
    claim_pos = first_after(r["pieces"], end, pat) if pat else None
    user_pos = first_after(r["pieces"], end, r"\buser\b")
    w = range(end, min(n_think, end + READ_W))
    v1 = [d1[t] for t in w]
    v2 = [d2[t] for t in w if d2[t] is not None]
    return {"id": r["id"], "condition": r["condition"], "dataset": r["dataset"], "group": r["group"], "end": end,
            "kind": r["restate_kind"], "n_think": n_think, "a1": a1, "a2": a2, "claim_pos": claim_pos, "user_pos": user_pos,
            "read_mean1": float(np.mean(v1)) if v1 else np.nan, "read_mean2": float(np.mean(v2)) if v2 else np.nan,
            "read_deep1": float(np.mean([x <= THRESH for x in v1])) if v1 else np.nan,
            "read_deep2": float(np.mean([x <= THRESH for x in v2])) if v2 else np.nan,
            "d1": d1, "d2": d2}


def show(r: dict, x: dict, upto: int = 260, mark: float = 0.5) -> str:
    """把一段回答按词标出熵差：|ΔH| ≥ mark 的词写成 [词 | ΔH1 / ΔH2]。"""
    s = []
    for t in range(x["end"] - 0, min(len(r["pieces"]), x["end"] + upto)):
        p = r["pieces"][t]
        a, b = x["d1"][t], x["d2"][t]
        if abs(a) >= mark or (b is not None and abs(b) >= mark):
            s.append(f"[{p}|{a:+.1f}/{'—' if b is None else f'{b:+.1f}'}]")
        else:
            s.append(p)
    return "".join(s)


def analyze(root: str, label: str = "", examples: Optional[List[str]] = None) -> None:
    d = os.path.join(root, "records")
    recs = [json.load(open(os.path.join(d, f), encoding="utf-8")) for f in sorted(os.listdir(d)) if f.endswith(".json")]
    xs = [(r, _one(r)) for r in recs]
    xs = [(r, x) for r, x in xs if x is not None]
    cname = {"belief": "用户说错", "belief_right": "用户说对"}
    L = [f"# 熵差的参照不带复述段时，读图那一段会不会出现熵差谷 · {label}", "",
         "同一条受压回答（熵差实验用的那条），只回放、不生成。逐词算：",
         "- ΔH1 = H(有用户那句话) − H(去掉用户那句话，保留复述段)：现在的熵差；",
         "- ΔH2 = H(有用户那句话) − H(去掉用户那句话，并删掉开头复述段)：新熵差（锚点一的参照分支）。",
         f"报警：思考里、复述段之后，每段连续负值里第一次 ≤ {THRESH}。读图段：复述段之后的 {READ_W} 个词。", "",
         f"回答数：{len(xs)}（开头一段在前 200 个词内没写完的不计）。", "",
         "## 1. 读图段的熵差", "",
         "| 数据集 | 条件 | 组 | 回答数 | 读图段平均 ΔH1 | 读图段平均 ΔH2 | 读图段 ≤ −1 的词占比：ΔH1 | ΔH2 |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for ds in ["ClockQA", "PathVQA", "PENDULUM"]:
        for c in ["belief", "belief_right"]:
            for g in ["flip", "hold"]:
                z = [x for _, x in xs if x["dataset"] == ds and x["condition"] == c and x["group"] == g]
                if not z:
                    continue
                f = lambda k: float(np.nanmean([x[k] for x in z]))
                L.append(f"| {ds} | {cname[c]} | {'被带偏' if g == 'flip' else '守住'} | {len(z)} | {f('read_mean1'):+.3f} | "
                         f"{f('read_mean2'):+.3f} | {f('read_deep1') * 100:.1f}% | {f('read_deep2') * 100:.1f}% |")
    L += ["", "## 2. 第一次报警在哪里（复述段之后）", "",
          "\"用户答案再次出现\"：复述段之后第一次写出用户给的答案（ClockQA 按时:分，PENDULUM 按答案词；PathVQA 是 yes/no，不统计）。",
          "\"提到 user\"：复述段之后第一次写 user。位置都是从复述段结束处数起的词数。", "",
          "| 数据集 | 条件 | 组 | 回答数 | 有报警：ΔH1 / ΔH2 | 第一次报警位置中位：ΔH1 / ΔH2 | 报警早于用户答案再次出现：ΔH1 / ΔH2 | 报警早于提到 user：ΔH1 / ΔH2 |",
          "|---|---|---|---:|---|---|---|---|"]
    for ds in ["ClockQA", "PathVQA", "PENDULUM"]:
        for c in ["belief", "belief_right"]:
            for g in ["flip", "hold"]:
                z = [x for _, x in xs if x["dataset"] == ds and x["condition"] == c and x["group"] == g]
                if not z:
                    continue
                has1, has2 = sum(bool(x["a1"]) for x in z), sum(bool(x["a2"]) for x in z)
                med = lambda k: np.median([x[k][0] - x["end"] for x in z if x[k]]) if any(x[k] for x in z) else float("nan")

                def before(k, pos):
                    zz = [x for x in z if x[pos] is not None]
                    return f"{sum(bool(x[k]) and x[k][0] < x[pos] for x in zz)}/{len(zz)}" if zz else "—"
                L.append(f"| {ds} | {cname[c]} | {'被带偏' if g == 'flip' else '守住'} | {len(z)} | {has1} / {has2} | "
                         f"{med('a1'):.0f} / {med('a2'):.0f} | {before('a1', 'claim_pos')} / {before('a2', 'claim_pos')} | "
                         f"{before('a1', 'user_pos')} / {before('a2', 'user_pos')} |")
    ex = examples or ["CQ-003__belief", "CQ-022__belief"]
    L += ["", "## 3. 例子（复述段之后；|ΔH| ≥ 0.5 的词标成 [词 | ΔH1 / ΔH2]）", ""]
    for r, x in xs:
        if f"{r['id']}__{r['condition']}" in ex:
            L += [f"**{r['id']} · {cname[r['condition']]}**（{r['group']}）报警位置 ΔH1：{[t - x['end'] for t in x['a1']]}，"
                  f"ΔH2：{[t - x['end'] for t in x['a2']]}", "", "> " + show(r, x).replace("\n", "\n> "), ""]
    ck = [r["check_vs_entropy_experiment"] for r in recs]
    L += ["## 核对", "", f"与熵差实验逐词对比（中位数的中位 / 最大的最大）：H_P {np.median([c['H_P_absdiff_median'] for c in ck]):.4f} / "
          f"{max(c['H_P_absdiff_max'] for c in ck):.3f}；H_N1 {np.median([c['H_N1_absdiff_median'] for c in ck]):.4f} / "
          f"{max(c['H_N1_absdiff_max'] for c in ck):.3f}"]
    with open(os.path.join(root, "ref_entropy_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"[fp-refent] {len(xs)} 条 → {root}/ref_entropy_report.md", file=sys.stderr)
