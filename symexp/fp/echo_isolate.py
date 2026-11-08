# -*- coding: utf-8 -*-
"""回声隔离：用熵差找出思考开头复述里抄来的用户答案（回声），读图阶段让之后的词读不到它的内容（ClockQA，20 题）。

1. 生成时同步跑熵差分支：提示里的问题换成模型自己改写的中性问题（fp-model2 neutralize 的结果，推理时可得），
   思考与主分支相同，只读不写。每一步 ΔH = H(主分支) − H(熵差分支)。
2. 回声：从思考开头到读图窗口结束，每段连续负熵差里只要有一步 ≤ −1，这一段的词都算回声（抄来的用户答案）。
3. echo 组：开头一段（跳过短标题、提到 user 的第一段）写完后的 WINDOW 个词里，每个标准注意力层（Gemma 包括滑动窗口层），
   当前词对回声位置的注意力权重乘以 γ = 0，**不重新归一化**——等于把回声的 Value 置零：注意力分布不变，只是读不到内容。
   窗口之后 γ = 1，照常生成；熵差分支也不再跑。
   base 组：熵差分支同样跑，回声同样记录，不改。两组同一随机种子。
不需要知道提示里哪一句是用户说的：回声完全由熵差定位。
"""

import json
import os
import re
import sys
from typing import Dict, List, Optional

import numpy as np

from .valley_inject import THRESH, ValleyController

WINDOW = 128
GAMMA = 0.0
ARMS = ["base", "echo"]


class EchoController(ValleyController):
    def __init__(self, backend, cfg, arm, nbatch, n_in, img_abs, rec_layers, vis_ids, lam=0.0, gamma: float = GAMMA,
                 window: int = WINDOW):
        super().__init__(backend, cfg, "base", nbatch, n_in, img_abs, rec_layers, vis_ids, lam=0.0)
        self.mode, self.gamma, self.window = arm, gamma, window
        self.echo: List[int] = []           # 回声的输出位置（第几个输出 token）
        self._run, self._deep = [], False
        self.done = False
        self.isolated_steps = 0

    def in_window(self, k: int) -> bool:
        return self.end is not None and self.end <= k < self.end + self.window and not self.closed

    @property
    def passthrough(self):
        return not (self.branch == "main" and self.mode == "echo" and self.in_window(len(self.out)) and self.echo)

    def _call(self, kw, rope, record: bool = True):
        return super()._call(kw, rope, record=False)          # 熵差分支不需要记录注意力

    def pre_hook(self, module, args, kwargs):
        from .. import attn
        if self.branch == "ref":
            return
        if self.done:                                          # 窗口之后：只记下 token，不跑熵差分支
            attn._ACTIVE = None
            ids = kwargs.get("input_ids", args[0] if args else None)
            t = int(ids[0, -1])
            self.out.append(t)
            if t == self.b.close_id:
                self.closed = True
            return
        super().pre_hook(module, args, kwargs)
        if not self.in_window(len(self.out)):
            attn._ACTIVE = None

    def post_hook(self, module, args, output):
        if self.branch == "ref" or self.done:
            return
        super().post_hook(module, args, output)
        k = len(self.out)
        dH = self.dH[-1]
        started = self.skip is not None and k >= self.skip
        if started and (self.end is None or self.in_window(k)):
            if dH < 0:
                self._run.append(k)
                self._deep = self._deep or dH <= THRESH
                if self._deep:
                    for x in self._run:
                        if x not in self.echo:
                            self.echo.append(x)
            else:
                self._run, self._deep = [], False
        if self.end is not None and (k + 1 >= self.end + self.window or self.closed):
            self.done = True

    def modify(self, layer_idx, p, q_pos):
        if self.branch != "main" or self.mode != "echo" or p.shape[2] != 1 or not self.echo:
            return p
        if not self.in_window(len(self.out)):
            return p
        t = self.torch
        off = self.abs_q + 1 - p.shape[-1]
        keys = t.tensor([self.n_in + i - off for i in self.echo if self.n_in + i - off >= 0], device=p.device, dtype=t.long)
        keys = keys[keys < p.shape[-1]]
        if keys.numel() == 0:
            return p
        p = p.clone()
        p[..., keys] = p[..., keys] * self.gamma
        if layer_idx == min(self.rec_layers):
            self.isolated_steps += 1
        return p

    def extra(self) -> dict:
        txt = [self.tok.decode([self.out[i]], skip_special_tokens=False) for i in self.echo if i < len(self.out)]
        return {"mode": self.mode, "gamma": self.gamma, "window": self.window, "echo_positions": self.echo,
                "echo_text": txt, "isolated_steps": self.isolated_steps}


def neutral_question(neutral_dir: str, iid: str, cond: str) -> str:
    p = os.path.join(neutral_dir, f"{iid}__{cond}.json")
    if not os.path.exists(p):
        raise SystemExit(f"缺少模型改写的中性问题 {p}")
    return json.load(open(p, encoding="utf-8"))["neutral"]


def run(cfg, root: str, ent_dir: str, model_key: str, neutral_dir: str, image_dirs: Optional[List[str]] = None,
        shard: str = "", seed: int = 42, ids: Optional[List[str]] = None, gamma: float = GAMMA, window: int = WINDOW) -> None:
    from .conflict import _engine, claim_of
    from .mask_query import first_reading
    from .pipeline import _shard
    from .trigger_regions import load_entropy, resolve_image, resolve_source
    from .valley_inject import CONDS, generate, selection
    from .verify import final_time, outcome
    os.makedirs(os.path.join(root, "records"), exist_ok=True)
    ent = load_entropy(ent_dir, model_key, ["ClockQA"])
    sel = ids or selection(ent)["ClockQA"]
    with open(os.path.join(root, "selection.json"), "w", encoding="utf-8") as f:
        json.dump(sel, f, ensure_ascii=False, indent=1)
    tasks = _shard([(i, c, a) for i in sel for c in CONDS for a in ARMS], shard)
    todo = [x for x in tasks if not os.path.exists(os.path.join(root, "records", f"{x[0]}__{x[1]}__{x[2]}.json"))]
    print(f"[fp-echo] 本进程 {len(tasks)} 项，待做 {len(todo)} 项", file=sys.stderr)
    eng = None
    for n, (iid, c, arm) in enumerate(todo):
        e = ent[(iid, c)]
        job = e["job"]
        item = dict(job["item"])
        orig = json.load(open(resolve_source(job), encoding="utf-8"))
        item["image"] = resolve_image(job, orig, image_dirs)
        nq = neutral_question(neutral_dir, iid, c)
        if eng is None:
            eng = _engine(cfg)
        rec = generate(eng.backend, cfg, item, c, arm, seed=seed, lam=0.0, ctrl_cls=EchoController, attention=True,
                       ctrl_kw={"gamma": gamma, "window": window}, all_layers=True, neutral_text=nq)
        claim = claim_of(item, c)
        rec.update(id=iid, dataset="ClockQA", condition=c, group=e["group"], gold=item["gold"], claim=claim, neutral_question=nq,
                   outcome=outcome(rec["answer"], item["gold"], claim), final_time=final_time(rec["answer"], claim),
                   first_reading=first_reading(rec["thinking"], item["gold"], item.get("claim")))
        w = [x for x in rec["image_mass"] if x == x]
        rec["window_image_mass"] = float(np.mean(w)) if w else None
        p = os.path.join(root, "records", f"{iid}__{c}__{arm}.json")
        with open(p + ".tmp", "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        os.replace(p + ".tmp", p)
        print(f"  {n + 1}/{len(todo)} {iid}__{c} {arm}：{rec['n_out']} token，{rec['seconds']}s，开头一段结束于 {rec['restate_end']}，"
              f"回声 {len(rec['echo_positions'])} 个词 {''.join(rec['echo_text'])[:60]!r}，隔离了 {rec['isolated_steps']} 步，"
              f"第一次读数 {rec['first_reading']['time']}（{rec['first_reading']['kind']}），{rec['outcome']}，"
              f"熵差分支核对 {rec['ref_check']}", file=sys.stderr)


def summary(root: str, label: str = "", maskq_root: Optional[str] = None) -> None:
    def load(d):
        R = {}
        if d and os.path.isdir(os.path.join(d, "records")):
            for f in sorted(os.listdir(os.path.join(d, "records"))):
                if f.endswith(".json"):
                    r = json.load(open(os.path.join(d, "records", f), encoding="utf-8"))
                    R[(r["id"], r["condition"], r["arm"])] = r
        return R
    R, M = load(root), load(maskq_root)
    ids = sorted({k[0] for k in R})
    rows = [("正常生成", R, "base"), ("回声隔离", R, "echo")] + ([("挡住提示里的用户那句话（fp-maskq）", M, "mask")] if M else [])
    kind = lambda r: {"坚持（答对）": "答对", "采纳": "照着用户的答案"}.get(r["outcome"], "答成别的时间")
    L = [f"# 回声隔离 · {label}", "",
         "**怎么跑的**：ClockQA 选了 20 道题，每道题问两次（用户给错误答案 / 用户给正确答案），每次生成两遍，同一个随机种子：",
         "- 正常生成：照常；",
         "- 回声隔离：用熵差找出思考开头复述里抄来的用户答案（回声），读图那 128 个词里，之后写的每个词都读不到回声的内容"
         "（注意力权重不变，只把回声传来的内容置零）；128 个词之后照常。",
         "（最后一行是之前挡住提示里用户那句话的结果，同样的题，用来对比。）", ""]
    for c, title in (("belief", "用户给错误答案时"), ("belief_right", "用户给正确答案时")):
        L += [f"## {title}（{len(ids)} 题）", "", "| | 答对 | 照着用户的答案 | 答成别的时间 |", "|---|---:|---:|---:|"]
        for name, D, arm in rows:
            rs = [D[(i, c, arm)] for i in ids if (i, c, arm) in D]
            if not rs:
                continue
            ks = [kind(r) for r in rs]
            L.append(f"| {name}（{len(rs)} 题） | {ks.count('答对')} | {ks.count('照着用户的答案')} | {ks.count('答成别的时间')} |")
        ch = [(i, kind(R[(i, c, 'base')]), kind(R[(i, c, 'echo')])) for i in ids if (i, c, "base") in R and (i, c, "echo") in R]
        up = [f"{i}（{a} → {b}）" for i, a, b in ch if a != "答对" and b == "答对"]
        dn = [f"{i}（{a} → {b}）" for i, a, b in ch if a == "答对" and b != "答对"]
        L += ["", f"回声隔离相对正常生成：从错变对 {len(up)} 题 {('：' + '、'.join(up)) if up else ''}；"
                  f"从对变错 {len(dn)} 题 {('：' + '、'.join(dn)) if dn else ''}。", ""]
    es = [r for r in R.values() if r["arm"] == "echo"]
    if es:
        n_e = [len(r["echo_positions"]) for r in es]
        mb = [r["window_image_mass"] for r in R.values() if r["arm"] == "base" and r["window_image_mass"] is not None]
        me = [r["window_image_mass"] for r in es if r["window_image_mass"] is not None]
        L += ["## 回声找得怎么样", "",
              f"- 回声隔离的 {len(es)} 条回答里，{sum(x > 0 for x in n_e)} 条找到了回声，平均 {np.mean(n_e):.1f} 个词；",
              f"- 读图那 128 个词里的看图总量：正常 {np.mean(mb) * 100:.1f}%，回声隔离 {np.mean(me) * 100:.1f}%（只是参考，不是目标）；",
              "- 用户给错误答案时，找到的回声（按题）：", ""]
        for r in sorted(es, key=lambda x: x["id"]):
            if r["condition"] == "belief":
                L.append(f"  - {r['id']}：{''.join(r['echo_text'])[:80]!r}")
    with open(os.path.join(root, "echo_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"[fp-echo] {len(R)} 条 → {root}/echo_report.md", file=sys.stderr)
