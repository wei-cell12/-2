# -*- coding: utf-8 -*-
"""熵谷处的分布校正（ClockQA、PathVQA，各 20 题，每题 2 个随机种子）。

生成时同步跑熵差分支（提示去掉用户那句话，思考与主分支相同，只读不写），报警规则与 fp-valley 相同：
思考里、开头一段之后，每段连续负熵差里第一次 ΔH ≤ −1。
mix 组：从报警那一步起（包括报警那一步本身），至多 WINDOW 个词、到这一句写完为止（写出含 . ? ! 或换行的词），
只要这一步 ΔH < 0，就按两条分支的几何混合采样：
  log p'(v) = (1 − β)·log p_主(v) + β·log p_熵差分支(v) + 常数，β = 0.5
混合在模型输出的 logits 上替换，之后的重复惩罚、温度、top-p 等照常作用。熵差分支本身不改。
base 组：熵差分支照跑，报警照样记录，不混合。两组同一随机种子，第一次混合之前逐 token 相同。
"""

import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

from .valley_inject import ValleyController

BETA = 0.5
WINDOW = 32
ARMS = ["base", "mix"]
SEEDS = [42, 43]
DATASETS = ["ClockQA", "PathVQA"]
SENT_END = (".", "?", "!", "\n")


def mix_logprobs(lp_main, lp_ref, beta: float):
    """几何混合（对数概率的加权和）；返回值当作 logits 用，softmax 之后就是归一化的混合分布。"""
    return (1 - beta) * lp_main + beta * lp_ref


def ends_sentence(piece: str) -> bool:
    return any(ch in piece for ch in SENT_END)


class MixController(ValleyController):
    def __init__(self, backend, cfg, arm, nbatch, n_in, img_abs, rec_layers, vis_ids, lam=0.0, beta: float = BETA,
                 window: int = WINDOW):
        super().__init__(backend, cfg, "base", nbatch, n_in, img_abs, rec_layers, vis_ids, lam=0.0)   # 父类不做注入
        self.mode, self.beta, self.window = arm, beta, window
        self.win_left = 0
        self.mixed = []            # 每个混合的步：位置、ΔH、混合前后最可能的词

    def pre_hook(self, module, args, kwargs):
        if self.branch == "ref":
            return
        had = self.prefilled
        super().pre_hook(module, args, kwargs)
        if had and self.win_left > 0:
            self.win_left -= 1
            if self.closed or ends_sentence(self.tok.decode([self.out[-1]], skip_special_tokens=False)):
                self.win_left = 0

    def post_hook(self, module, args, output):
        if self.branch == "ref":
            return
        n_ev = len(self.events)
        super().post_hook(module, args, output)
        if len(self.events) > n_ev:                     # 这一步报警：开一个窗口（从这一步开始）
            self.win_left = self.window
            self.events[-1]["window_start"] = True
        if self.win_left <= 0 or self.closed:
            return
        dH = self.dH[-1]
        if dH >= 0:
            return
        t = self.torch
        lg = output.logits[0, -1]
        lp = t.log_softmax(lg.float(), -1)
        rp = self.ref_logits_now.to(lp.device)
        info = {"step": len(self.out), "dH": round(dH, 4), "top_main": int(lp.argmax()), "p_top_main": float(lp.max().exp())}
        if self.mode == "mix":
            m = mix_logprobs(lp, rp, self.beta)
            output.logits[0, -1] = m.to(lg.dtype)
            info.update(top_mix=int(m.argmax()), p_top_mix=float(t.softmax(m, -1).max()))
        else:
            m = mix_logprobs(lp, rp, self.beta)
            info.update(top_mix=int(m.argmax()), p_top_mix=float(t.softmax(m, -1).max()), applied=False)
        self.mixed.append(info)

    def extra(self) -> dict:
        tok = self.tok
        for x in self.mixed:
            x["top_main_text"] = tok.decode([x["top_main"]], skip_special_tokens=False)
            x["top_mix_text"] = tok.decode([x["top_mix"]], skip_special_tokens=False)
        return {"mode": self.mode, "beta": self.beta, "window": self.window, "mixed_steps": self.mixed}


def run(cfg, root: str, ent_dir: str, model_key: str, image_dirs: Optional[List[str]] = None, shard: str = "",
        seeds: Optional[List[int]] = None, beta: float = BETA, window: int = WINDOW, datasets: Optional[List[str]] = None,
        only_ids: Optional[List[str]] = None) -> None:
    from .conflict import _engine, claim_of
    from .pipeline import _shard
    from .trigger_regions import load_entropy, resolve_image, resolve_source
    from .valley_inject import CONDS, claim_answer, generate, selection
    from .verify import final_time, outcome
    os.makedirs(os.path.join(root, "records"), exist_ok=True)
    datasets = datasets or DATASETS
    seeds = seeds or SEEDS
    ent = load_entropy(ent_dir, model_key, datasets)
    sel = selection(ent)
    with open(os.path.join(root, "selection.json"), "w", encoding="utf-8") as f:
        json.dump({d: sel[d] for d in datasets}, f, ensure_ascii=False, indent=1)
    tasks = [(ds, i, c, arm, sd) for ds in datasets for i in sel[ds] for c in CONDS for sd in seeds for arm in ARMS
             if not only_ids or i in only_ids]
    tasks = _shard(tasks, shard)
    fn = lambda i, c, arm, sd: f"{i}__{c}__{arm}__s{sd}.json"
    todo = [x for x in tasks if not os.path.exists(os.path.join(root, "records", fn(*x[1:])))]
    print(f"[fp-mix] 本进程 {len(tasks)} 项，待做 {len(todo)} 项", file=sys.stderr)
    eng = None
    for n, (ds, iid, c, arm, sd) in enumerate(todo):
        e = ent[(iid, c)]
        job = e["job"]
        item = dict(job["item"])
        orig = json.load(open(resolve_source(job), encoding="utf-8"))
        item["image"] = resolve_image(job, orig, image_dirs)
        if eng is None:
            eng = _engine(cfg)
        rec = generate(eng.backend, cfg, item, c, arm, seed=sd, lam=0.0, ctrl_cls=MixController, attention=False,
                       ctrl_kw={"beta": beta, "window": window})
        claim = claim_of(item, c)
        rec.update(id=iid, dataset=ds, condition=c, group=e["group"], gold=item["gold"], claim=claim,
                   question=item["messages"]["neutral"], image=item["image"], wrong_claim=claim_answer(item.get("claim")))
        if ds == "ClockQA":
            rec["outcome"] = outcome(rec["answer"], item["gold"], claim)
            rec["final_time"] = final_time(rec["answer"], claim)
        p = os.path.join(root, "records", fn(iid, c, arm, sd))
        with open(p + ".tmp", "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        os.replace(p + ".tmp", p)
        ms = rec["mixed_steps"]
        ch = sum(x["top_main"] != x["top_mix"] for x in ms)
        print(f"  {n + 1}/{len(todo)} {ds} {iid}__{c} {arm} 种子 {sd}：{rec['n_out']} token，{rec['seconds']}s，报警 {len(rec['events'])} 次，"
              f"混合 {len(ms) if arm == 'mix' else 0} 步（最可能的词因混合而改变 {ch if arm == 'mix' else 0} 步），"
              f"开头一段结束于 {rec['restate_end']}，熵差分支核对 {rec['ref_check']}"
              + (f"，{rec['outcome']}" if ds == "ClockQA" else ""), file=sys.stderr)


def summary(root: str, label: str = "") -> None:
    from .valley_inject import _records, judged_status
    recs = dict(_records(root))
    st = judged_status(root, recs)
    cname = {"belief": "用户说错", "belief_right": "用户说对"}
    by = {}
    for f, r in recs.items():
        if f in st:
            by[(r["id"], r["condition"], r["arm"], r["seed"])] = st[f][1]
    seeds = sorted({k[3] for k in by})
    L = [f"# 熵谷处的分布校正 · {label}", "",
         f"- 报警：思考里、开头一段之后，每段连续负熵差里第一次 ΔH ≤ −1；",
         f"- mix：从报警那一步起至多 {WINDOW} 个词、到这一句写完为止，ΔH < 0 的步按 (1 − β)·log p_主 + β·log p_熵差分支 采样，β = {BETA}；",
         "- base：不混合。两组同一随机种子；每题每种条件 " + f"{len(seeds)} 个种子（{', '.join(map(str, seeds))}）。", "",
         "## 答对比例（题数 × 种子数）", "",
         "| 数据集 | 条件 | 回答数 | base 答对 | mix 答对 | base 跟着错误说法 | mix 跟着错误说法 | 按题比较：mix 更好 / 更差 / 相同 |",
         "|---|---|---:|---:|---:|---:|---:|---|"]
    for ds in DATASETS:
        for c in ["belief", "belief_right"]:
            items = sorted({k[0] for k in by if k[1] == c and recs_ds(recs, k[0]) == ds})
            items = [i for i in items if all((i, c, a, s) in by for a in ARMS for s in seeds)]
            if not items:
                continue
            n = len(items) * len(seeds)
            cnt = lambda a, v: sum(by[(i, c, a, s)] == v for i in items for s in seeds)
            better = worse = same = 0
            for i in items:
                rb = np.mean([by[(i, c, "base", s)] == "答对" for s in seeds])
                rm = np.mean([by[(i, c, "mix", s)] == "答对" for s in seeds])
                better += rm > rb
                worse += rm < rb
                same += rm == rb
            L.append(f"| {ds} | {cname[c]} | {n} | {cnt('base', '答对')} | {cnt('mix', '答对')} | {cnt('base', '跟着错误说法')} | "
                     f"{cnt('mix', '跟着错误说法')} | {better} / {worse} / {same} |")
    L += ["", "答对 = 最终答案与正确答案一致（ClockQA 自动比对，时间差 2 分钟以内；PathVQA 由盲判读抽取答案再比对）。",
          "按题比较：每题把几个种子的答对比例平均后，mix 比 base 高 / 低 / 相等的题数。", "",
          "## 混合发生了多少", "", "| 数据集 | 条件 | mix 回答数 | 平均报警次数 | 平均混合步数 | 混合后最可能的词改变的步占比 |", "|---|---|---:|---:|---:|---:|"]
    for ds in DATASETS:
        for c in ["belief", "belief_right"]:
            rs = [r for r in recs.values() if r["dataset"] == ds and r["condition"] == c and r["arm"] == "mix"]
            if not rs:
                continue
            ms = [x for r in rs for x in r["mixed_steps"]]
            ch = np.mean([x["top_main"] != x["top_mix"] for x in ms]) if ms else float("nan")
            L.append(f"| {ds} | {cname[c]} | {len(rs)} | {np.mean([len(r['events']) for r in rs]):.2f} | {len(ms) / len(rs):.1f} | {ch * 100:.0f}% |")
    L += ["", "## 混合改变最可能的词的例子（每个数据集前 8 个，用户说错）", "", "| 题 | 种子 | 报警前写的 | 原本最可能 | 混合后最可能 |", "|---|---:|---|---|---|"]
    for ds in DATASETS:
        k = 0
        for f, r in sorted(recs.items()):
            if r["dataset"] != ds or r["arm"] != "mix" or r["condition"] != "belief":
                continue
            for x in r["mixed_steps"]:
                if x["top_main"] != x["top_mix"] and k < 8:
                    evs = [e for e in r["events"] if e["step"] <= x["step"]]
                    ctx = evs[-1]["context"] if evs else ""
                    L.append(f"| {r['id']} | {r['seed']} | {ctx[-50:]!r} | {x['top_main_text']!r}（{x['p_top_main']:.2f}） | "
                             f"{x['top_mix_text']!r}（{x['p_top_mix']:.2f}） |")
                    k += 1
    with open(os.path.join(root, "mix_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"[fp-mix] {len(recs)} 条 → {root}/mix_report.md", file=sys.stderr)


def recs_ds(recs: Dict[str, dict], iid: str) -> Optional[str]:
    for r in recs.values():
        if r["id"] == iid:
            return r["dataset"]
    return None
