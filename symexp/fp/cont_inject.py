# -*- coding: utf-8 -*-
"""从第一个熵差谷开始，连续注入视觉向量（ClockQA，20 题）。

熵差分支与 fp-valley 相同（提示里的问题用模型自己改写的中性问题；思考与主分支相同，只读不写）。
起点：思考开始后第一次 ΔH ≤ −1 的那一步（通常是开头复述里抄写用户答案的地方）。
cont 组：从起点开始的 WINDOW 步里，每一步主分支下一步读入的向量都换成
  ẽ = (1 − λ)·E_{v∼p_t}[e(v)] + λ·e_img
  e_img：图像位置第一层输入向量，按熵差分支这一步对各图块的注意力加权平均（去掉固定位置），缩放到与 E[e(v)] 一样长（同 fp-valley 的 img）。
  λ 默认 0.2（连续注入，比单步注入的 0.4 小）。窗口之后照常生成，熵差分支也停掉。
base 组：熵差分支同样跑、起点同样记录，不注入。两组同一随机种子。
"""

import json
import os
import sys
from typing import List, Optional

import numpy as np

from .valley_inject import THRESH, ValleyController, img_weights, mix

WINDOW = 128
LAM = 0.2
NO_START_LIMIT = 600      # 思考开始后这么多步还没有熵差谷，就不再找（不注入）
ARMS = ["base", "cont"]


class ContController(ValleyController):
    def __init__(self, backend, cfg, arm, nbatch, n_in, img_abs, rec_layers, vis_ids, lam=LAM, window: int = WINDOW):
        super().__init__(backend, cfg, "base", nbatch, n_in, img_abs, rec_layers, vis_ids, lam=lam)
        self.mode, self.cont_lam, self.window = arm, lam, window
        self.start = None
        self.n_inj = 0
        self.done = False

    def pre_hook(self, module, args, kwargs):
        from .. import attn
        if self.branch == "ref":
            return
        if self.done:
            attn._ACTIVE = None
            ids = kwargs.get("input_ids", args[0] if args else None)
            t = int(ids[0, -1])
            self.out.append(t)
            if t == self.b.close_id:
                self.closed = True
            return
        super().pre_hook(module, args, kwargs)

    def post_hook(self, module, args, output):
        if self.branch == "ref" or self.done:
            return
        super().post_hook(module, args, output)
        k = len(self.out)
        dH = self.dH[-1]
        if self.start is None and self.skip is not None and k > self.skip and not self.closed:
            if dH <= THRESH:
                self.start = k
            elif k > self.skip + NO_START_LIMIT:
                self.done = True
                return
        if self.start is None:
            return
        if self.closed or k >= self.start + self.window:
            self.done = True
            return
        if self.mode != "cont" or self.V is None or "main" not in self.acc or "ref" not in self.acc:
            return
        t = self.torch
        edev = self.embed.weight.device
        lp = t.log_softmax(output.logits[0, -1].float(), -1)
        top = t.topk(lp.exp(), 100)
        w = (top.values / top.values.sum()).to(edev)
        E = self.embed(top.indices[None].to(edev))[0].float()
        e_hat = (w[:, None] * E).sum(0)
        am, ar = self.acc["main"], self.acc["ref"]
        qm, qr = am["share"] / am["share"].sum(), ar["share"] / ar["share"].sum()
        wj = img_weights(self.xp, qm, qr)
        v = (wj[:, None].to(self.V.device) * self.V).sum(0)
        self.pending = mix(e_hat, v.to(e_hat.device), self.cont_lam, align=True)
        self.n_inj += 1

    def extra(self) -> dict:
        ctx = self.tok.decode(self.out[max(0, self.start - 24):self.start], skip_special_tokens=False) if self.start else None
        return {"mode": self.mode, "cont_lam": self.cont_lam, "window": self.window, "start": self.start,
                "start_context": ctx, "injected_steps": self.n_inj}


def run(cfg, root: str, ent_dir: str, model_key: str, neutral_dir: str, image_dirs: Optional[List[str]] = None,
        shard: str = "", seed: int = 42, ids: Optional[List[str]] = None, lam: float = LAM, window: int = WINDOW) -> None:
    from .conflict import _engine, claim_of
    from .echo_isolate import neutral_question
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
    print(f"[fp-cont] 本进程 {len(tasks)} 项，待做 {len(todo)} 项", file=sys.stderr)
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
        rec = generate(eng.backend, cfg, item, c, arm, seed=seed, lam=lam, ctrl_cls=ContController, attention=True,
                       ctrl_kw={"window": window}, neutral_text=nq)
        claim = claim_of(item, c)
        rec.update(id=iid, dataset="ClockQA", condition=c, group=e["group"], gold=item["gold"], claim=claim, neutral_question=nq,
                   outcome=outcome(rec["answer"], item["gold"], claim), final_time=final_time(rec["answer"], claim),
                   first_reading=first_reading(rec["thinking"], item["gold"], item.get("claim")))
        p = os.path.join(root, "records", f"{iid}__{c}__{arm}.json")
        with open(p + ".tmp", "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        os.replace(p + ".tmp", p)
        print(f"  {n + 1}/{len(todo)} {iid}__{c} {arm}：{rec['n_out']} token，{rec['seconds']}s，起点第 {rec['start']} 个词 "
              f"{(rec['start_context'] or '')[-40:]!r}，注入 {rec['injected_steps']} 步，第一次读数 {rec['first_reading']['time']}"
              f"（{rec['first_reading']['kind']}），{rec['outcome']}，熵差分支核对 {rec['ref_check']}", file=sys.stderr)


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
    rows = [("正常生成", R, "base"), ("连续注入视觉向量", R, "cont")] + ([("挡住提示里的用户那句话（fp-maskq）", M, "mask")] if M else [])
    kind = lambda r: {"坚持（答对）": "答对", "采纳": "照着用户的答案"}.get(r["outcome"], "答成别的时间")
    lam = next((r.get("cont_lam") for r in R.values() if r["arm"] == "cont"), LAM)
    L = [f"# 从第一个熵差谷开始连续注入视觉向量 · {label}", "",
         "**怎么跑的**：ClockQA 选了 20 道题，每道题问两次（用户给错误答案 / 用户给正确答案），每次生成两遍，同一个随机种子：",
         "- 正常生成：照常；",
         f"- 连续注入：从思考里第一次出现熵差谷（≤ −1）的那一步起，接下来 {WINDOW} 步，每一步模型读入的向量里混入 {lam:.0%} 的图像向量"
         "（按没有用户那句话时这一步看图的位置加权）；之后照常。",
         "（最后一行是之前挡住提示里用户那句话的结果，同样的题，用来对比。）", ""]
    for c, title in (("belief", "用户给错误答案时"), ("belief_right", "用户给正确答案时")):
        L += [f"## {title}（{len(ids)} 题）", "", "| | 答对 | 照着用户的答案 | 答成别的时间 | 第一次读数就是用户的时间 |", "|---|---:|---:|---:|---:|"]
        for name, D, arm in rows:
            rs = [D[(i, c, arm)] for i in ids if (i, c, arm) in D]
            if not rs:
                continue
            ks = [kind(r) for r in rs]
            fu = sum(r.get("first_reading", {}).get("kind") == "用户的时间" for r in rs)
            L.append(f"| {name}（{len(rs)} 题） | {ks.count('答对')} | {ks.count('照着用户的答案')} | {ks.count('答成别的时间')} | {fu} |")
        ch = [(i, kind(R[(i, c, 'base')]), kind(R[(i, c, 'cont')])) for i in ids if (i, c, "base") in R and (i, c, "cont") in R]
        up = [f"{i}（{a} → {b}）" for i, a, b in ch if a != "答对" and b == "答对"]
        dn = [f"{i}（{a} → {b}）" for i, a, b in ch if a == "答对" and b != "答对"]
        L += ["", f"连续注入相对正常生成：从错变对 {len(up)} 题{('：' + '、'.join(up)) if up else ''}；"
                  f"从对变错 {len(dn)} 题{('：' + '、'.join(dn)) if dn else ''}。", ""]
    cs = [r for r in R.values() if r["arm"] == "cont"]
    if cs:
        L += ["## 起点在哪里、注入了多少", "",
              f"- {len(cs)} 条连续注入的回答里，{sum(r['start'] is not None for r in cs)} 条找到了起点，平均注入 {np.mean([r['injected_steps'] for r in cs]):.0f} 步；",
              "- 用户给错误答案时，起点之前写的（按题）：", ""]
        for r in sorted(cs, key=lambda x: x["id"]):
            if r["condition"] == "belief":
                L.append(f"  - {r['id']}：第 {r['start']} 个词，{(r['start_context'] or '')[-60:]!r}")
        L += ["", "- 连续注入后的回答开头（用户给错误答案，前 300 个字，用来看文字是否通顺）：", ""]
        for r in [x for x in sorted(cs, key=lambda x: x["id"]) if x["condition"] == "belief"][:6]:
            L.append(f"  - {r['id']}：{r['thinking'][:300]!r}")
    with open(os.path.join(root, "cont_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"[fp-cont] {len(R)} 条 → {root}/cont_report.md", file=sys.stderr)
