# -*- coding: utf-8 -*-
"""熵差片段隔离：整段思考里，之后写的每个词都读不到"靠用户那句话写出来的片段"的内容（ClockQA，20 题）。
片段完全由熵差决定，不依赖数据集的答案格式。

1. 熵差分支：提示里的问题换成模型自己改写的中性问题（推理时可得），思考与主分支相同，只读不写；
   每一步 ΔH = H(主分支) − H(熵差分支)，从思考开头一直算到思考结束。
2. 熵差片段：每段连续负熵差里只要有一步 ≤ −1，这一整段就是熵差片段。**一段写完（熵差回到 ≥ 0）之后才开始隔离**，
   让模型先把这一句话写完整。
3. seg 组：思考里每个词、每个标准注意力层（Gemma 包括滑动窗口层），对写完的熵差片段的注意力权重乘以 γ = 0，**不重新归一化**
   ——等于把这些位置的 Value 置零：注意力分布不变，只是读不到内容。思考结束后照常生成答案，熵差分支也停掉。
   提示（包括用户那句话）和图像都不动。
segc 组（方案 C）：与 seg 组相同，另外熵差分支也读不到已隔离的片段（同样乘以 0、不重新归一化）。
   这样熵差分支代表"完全没有用户信息、连思考里抄来的也读不到"；模型之后再引用用户的答案时，熵差分支预测不出来，
   熵差变负，这些引用自动成为新的片段。
base 组：熵差分支同样跑、片段同样记录，不改（默认沿用回声隔离实验里同一随机种子的 base，不重复生成）。
"""

import json
import os
import sys
from typing import Dict, List, Optional, Set

import numpy as np

from .valley_inject import THRESH, ValleyController

GAMMA = 0.0
ARMS = ["base", "seg", "segc"]
ISO = ("seg", "segc")


class SegController(ValleyController):
    def __init__(self, backend, cfg, arm, nbatch, n_in, img_abs, rec_layers, vis_ids, lam=0.0, gamma: float = GAMMA):
        super().__init__(backend, cfg, "base", nbatch, n_in, img_abs, rec_layers, vis_ids, lam=0.0)
        self.mode, self.gamma = arm, gamma
        self.segs: List[List[int]] = []          # 熵差片段（输出位置）
        self.seg_set: Set[int] = set()
        self._run, self._deep = [], False
        self.pieces: List[str] = []
        self.done = False
        self.isolated_steps = 0
        self._keys_t = None

    @property
    def marked(self) -> List[int]:
        return sorted(self.seg_set)

    def _iso_here(self) -> bool:
        """当前这次前向要不要隔离：主分支在 seg / segc 组隔离；熵差分支只在 segc 组隔离。"""
        return self.mode in ISO and (self.branch == "main" or self.mode == "segc")

    @property
    def passthrough(self):
        return not (self._iso_here() and not self.closed and self.seg_set)

    def _call(self, kw, rope, record: bool = True):
        # 熵差分支不记录注意力；segc 组有片段之后，熵差分支也要经过注意力包装，好在里面隔离
        return super()._call(kw, rope, record=self.mode == "segc" and bool(self.seg_set))

    def accumulate(self, layer_idx, probs, stages, q_pos):
        if self.branch == "main":
            super().accumulate(layer_idx, probs, stages, q_pos)

    def verify_ref(self, output):
        """熵差分支逐步算出的分布与整段重算核对：segc 组只核对第一段片段开始隔离之前的那些步（之后熵差分支被改过）。"""
        if self.mode == "segc" and self.segs:
            self.ref_logits = self.ref_logits[: self.segs[0][-1] + 2]
        return super().verify_ref(output)

    def pre_hook(self, module, args, kwargs):
        from .. import attn
        if self.branch == "ref":
            return
        if self.done:                                          # 思考结束之后：只记下 token，不跑熵差分支、不改
            attn._ACTIVE = None
            ids = kwargs.get("input_ids", args[0] if args else None)
            self.out.append(int(ids[0, -1]))
            return
        super().pre_hook(module, args, kwargs)
        if self.prefilled and len(self.out) > len(self.pieces):
            self.pieces.append(self.tok.decode([self.out[-1]], skip_special_tokens=False))
        self._keys_t = None

    def _close_run(self):
        if self._deep and self._run:
            self.segs.append(list(self._run))
            self.seg_set.update(self._run)
        self._run, self._deep = [], False

    def post_hook(self, module, args, output):
        if self.branch == "ref" or self.done:
            return
        super().post_hook(module, args, output)
        k = len(self.out)                                      # 这一步预测第 k 个输出 token
        dH = self.dH[-1]
        started = self.skip is not None and k >= self.skip and not self.closed
        if started:
            if dH < 0:
                self._run.append(k)
                self._deep = self._deep or dH <= THRESH
            else:
                self._close_run()                              # 一段写完（熵差回到 ≥ 0）：从下一步起隔离
        if self.closed:
            self._close_run()
            self.done = True

    def modify(self, layer_idx, p, q_pos):
        if not self._iso_here() or p.shape[2] != 1 or self.closed:
            return p
        m = self.marked
        if not m:
            return p
        t = self.torch
        off = self.abs_q + 1 - p.shape[-1]
        ck = (self.branch, len(m), p.shape[-1], str(p.device))
        if self._keys_t is None or self._keys_t[0] != ck:
            base = self.n_in if self.branch == "main" else self.n_ref0      # 两条分支提示长度不同
            keys = t.tensor([base + i - off for i in m if 0 <= base + i - off < p.shape[-1] - 1],
                            device=p.device, dtype=t.long)        # 不含当前这个词自己
            self._keys_t = (ck, keys)
        keys = self._keys_t[1]
        if keys.numel() == 0:
            return p
        p = p.clone()
        p[..., keys] = p[..., keys] * self.gamma
        if layer_idx == min(self.rec_layers) and self.branch == "main":
            self.isolated_steps += 1
        return p

    def extra(self) -> dict:
        dec = lambda xs: "".join(self.pieces[i] for i in xs if i < len(self.pieces))
        return {"mode": self.mode, "gamma": self.gamma, "segments": [dec(r) for r in self.segs],
                "segment_positions": self.segs, "isolated_steps": self.isolated_steps}


def run(cfg, root: str, ent_dir: str, model_key: str, neutral_dir: str, image_dirs: Optional[List[str]] = None,
        shard: str = "", seed: int = 42, ids: Optional[List[str]] = None, arms: Optional[List[str]] = None,
        gamma: float = GAMMA) -> None:
    from .conflict import _engine, claim_of
    from .echo_isolate import neutral_question
    from .mask_query import first_reading
    from .pipeline import _shard
    from .trigger_regions import load_entropy, resolve_image, resolve_source
    from .valley_inject import CONDS, generate, selection
    from .verify import final_time, outcome
    arms = arms or ["seg"]
    os.makedirs(os.path.join(root, "records"), exist_ok=True)
    ent = load_entropy(ent_dir, model_key, ["ClockQA"])
    sel = ids or selection(ent)["ClockQA"]
    with open(os.path.join(root, "selection.json"), "w", encoding="utf-8") as f:
        json.dump(sel, f, ensure_ascii=False, indent=1)
    tasks = _shard([(i, c, a) for i in sel for c in CONDS for a in arms], shard)
    todo = [x for x in tasks if not os.path.exists(os.path.join(root, "records", f"{x[0]}__{x[1]}__{x[2]}.json"))]
    print(f"[fp-seg] 本进程 {len(tasks)} 项，待做 {len(todo)} 项", file=sys.stderr)
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
        rec = generate(eng.backend, cfg, item, c, arm, seed=seed, lam=0.0, ctrl_cls=SegController, attention=True,
                       ctrl_kw={"gamma": gamma}, all_layers=True, neutral_text=nq)
        claim = claim_of(item, c)
        rec.update(id=iid, dataset="ClockQA", condition=c, group=e["group"], gold=item["gold"], claim=claim, neutral_question=nq,
                   outcome=outcome(rec["answer"], item["gold"], claim), final_time=final_time(rec["answer"], claim),
                   first_reading=first_reading(rec["thinking"], item["gold"], item.get("claim")))
        p = os.path.join(root, "records", f"{iid}__{c}__{arm}.json")
        with open(p + ".tmp", "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        os.replace(p + ".tmp", p)
        print(f"  {n + 1}/{len(todo)} {iid}__{c} {arm}：{rec['n_out']} token，{rec['seconds']}s，熵差片段 {len(rec['segments'])} 段 "
              f"{''.join(rec['segments'])[:60]!r}，隔离了 {rec['isolated_steps']} 步，"
              f"第一次读数 {rec['first_reading']['time']}（{rec['first_reading']['kind']}），{rec['outcome']}，"
              f"熵差分支核对 {rec['ref_check']}", file=sys.stderr)


def _load(d: Optional[str]) -> Dict[tuple, dict]:
    R = {}
    if d and os.path.isdir(os.path.join(d, "records")):
        for f in sorted(os.listdir(os.path.join(d, "records"))):
            if f.endswith(".json"):
                r = json.load(open(os.path.join(d, "records", f), encoding="utf-8"))
                R[(r["id"], r["condition"], r["arm"])] = r
    return R


def _rekind(r: dict) -> str:
    """按修正后的判定重新算（旧记录里的 outcome 可能用了旧的时间比较）。"""
    from .verify import outcome
    o = outcome(r["answer"], r["gold"], r["claim"]) if r.get("answer") is not None and "gold" in r else r["outcome"]
    return {"坚持（答对）": "答对", "采纳": "照着用户的答案"}.get(o, "答成别的时间")


def summary(root: str, label: str = "", echo_root: Optional[str] = None, maskq_root: Optional[str] = None) -> None:
    S, E, M = _load(root), _load(echo_root), _load(maskq_root)
    B = {k: v for k, v in S.items() if k[2] == "base"} or {k: v for k, v in E.items() if k[2] == "base"}
    base_src = "本实验" if any(k[2] == "base" for k in S) else "回声隔离实验（同一随机种子）"
    ids = sorted({k[0] for k in S})
    rows = [("正常生成", B, "base"), ("回声隔离（只隔离开头片段、只在读图的 128 个词里）", E, "echo"),
            ("熵差片段隔离（整段思考、所有熵差片段）", S, "seg"),
            ("熵差片段隔离 C（熵差分支也读不到已隔离的片段）", S, "segc"),
            ("挡住提示里的用户那句话（fp-maskq，参考上限）", M, "mask")]
    kind = _rekind
    L = [f"# 熵差片段隔离 · {label}", "",
         "**怎么跑的**：ClockQA 20 道题（与回声隔离、fp-maskq 同一批），每道题问两次（用户给错误答案 / 用户给正确答案）：",
         "- 熵差片段隔离：思考里每段靠用户那句话写出来的话（连续负熵差、其中至少一步 ≤ −1）写完之后，"
         "之后写的每个词都读不到它们的内容（注意力权重不变，只把内容置零）。"
         "提示里用户那句话、图像都不动；思考结束后照常写答案。",
         "- 熵差片段隔离 C：同上，另外熵差分支也读不到已隔离的片段，模型之后再引用用户的答案时熵差会变负，自动被隔离。",
         f"- 正常生成取自{base_src}；回声隔离、挡住用户那句话是之前的结果，同样的题、同一随机种子，用来对比。",
         "- 答案判定按修正后的规则重新算（以前把 12:00:05 和 12:05:00 当成同一个时间）。", ""]
    for c, title in (("belief", "用户给错误答案时"), ("belief_right", "用户给正确答案时")):
        L += [f"## {title}（{len(ids)} 题）", "", "| | 答对 | 照着用户的答案 | 答成别的时间 |", "|---|---:|---:|---:|"]
        for name, D, arm in rows:
            rs = [D[(i, c, arm)] for i in ids if (i, c, arm) in D]
            if not rs:
                continue
            ks = [kind(r) for r in rs]
            L.append(f"| {name}（{len(rs)} 题） | {ks.count('答对')} | {ks.count('照着用户的答案')} | {ks.count('答成别的时间')} |")
        L.append("")
        for arm, nm in (("seg", "熵差片段隔离"), ("segc", "熵差片段隔离 C")):
            ch = [(i, kind(B[(i, c, 'base')]), kind(S[(i, c, arm)])) for i in ids if (i, c, "base") in B and (i, c, arm) in S]
            if not ch:
                continue
            up = [f"{i}（{a} → {b}）" for i, a, b in ch if a != "答对" and b == "答对"]
            dn = [f"{i}（{a} → {b}）" for i, a, b in ch if a == "答对" and b != "答对"]
            L.append(f"{nm}相对正常生成（{len(ch)} 题）：从错变对 {len(up)} 题{('：' + '、'.join(up)) if up else ''}；"
                     f"从对变错 {len(dn)} 题{('：' + '、'.join(dn)) if dn else ''}。")
            L.append("")
    for arm, nm in (("seg", "熵差片段隔离"), ("segc", "熵差片段隔离 C")):
        ss = [r for r in S.values() if r["arm"] == arm]
        if not ss:
            continue
        def frac(r):
            n = (r.get("budget") or {}).get("close_position") or r.get("n_out") or 1
            return sum(len(x) for x in r.get("segment_positions", [])) / max(n, 1)
        L += [f"## {nm}隔离了什么", "",
              f"- {len(ss)} 条回答里，{sum(bool(r['segments']) for r in ss)} 条找到了熵差片段，平均每条 "
              f"{np.mean([len(r['segments']) for r in ss]):.1f} 段；被隔离的词平均占思考的 {np.mean([frac(r) for r in ss]) * 100:.1f}%；",
              "- 每条回答隔离的片段（前 6 段）：", ""]
        for r in sorted(ss, key=lambda x: (x["condition"], x["id"])):
            segs = " / ".join(repr(x[:60]) for x in r["segments"][:6])
            L.append(f"  - {r['id']} · {'用户说错' if r['condition'] == 'belief' else '用户说对'} · {kind(r)} · "
                     f"{len(r['segments'])} 段、占思考 {frac(r) * 100:.1f}%：{segs}")
        L.append("")
    with open(os.path.join(root, "seg_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"[fp-seg] {len(S)} 条 → {root}/seg_report.md", file=sys.stderr)
