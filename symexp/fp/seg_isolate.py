# -*- coding: utf-8 -*-
"""熵差片段隔离：整段思考里，之后写的每个词都读不到"靠用户那句话写出来的片段"的内容（ClockQA，20 题）。

1. 熵差分支：提示里的问题换成模型自己改写的中性问题（推理时可得），思考与主分支相同，只读不写；
   每一步 ΔH = H(主分支) − H(熵差分支)，从思考开头一直算到思考结束。
2. 熵差片段：每段连续负熵差里只要有一步 ≤ −1，这一整段就是熵差片段。**一段写完（熵差回到 ≥ 0）之后才开始隔离**，
   让模型先把这一句话写完整。
3. 重复的内容：熵差片段里出现的时间（时:分，忽略秒），模型之后（或之前）在思考里任何地方再写出同一个时:分，
   这几个词也一起隔离。只用到模型自己写过的文字，不需要知道用户的答案。
4. seg 组：思考里每个词、每个标准注意力层（Gemma 包括滑动窗口层），对以上位置的注意力权重乘以 γ = 0，**不重新归一化**
   ——等于把这些位置的 Value 置零：注意力分布不变，只是读不到内容。思考结束后照常生成答案，熵差分支也停掉。
   提示（包括用户那句话）和图像都不动。
base 组：熵差分支同样跑、片段同样记录，不改（默认沿用回声隔离实验里同一随机种子的 base，不重复生成）。
"""

import json
import os
import re
import sys
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .valley_inject import THRESH, ValleyController

GAMMA = 0.0
ARMS = ["base", "seg"]
_TIME = re.compile(r"(?<![\d:])(\d{1,2}):(\d{2})(?::\d{2})?(?![\d:])")


def time_keys(text: str) -> Set[Tuple[int, int]]:
    return {(int(m.group(1)) % 12, int(m.group(2))) for m in _TIME.finditer(text)}


def time_spans(text: str) -> List[Tuple[int, int, Tuple[int, int]]]:
    """写完整的时间（后面已经跟了别的字符）：(起, 止, (时 mod 12, 分))。"""
    out = []
    for m in _TIME.finditer(text):
        if m.end() < len(text):
            out.append((m.start(), m.end(), (int(m.group(1)) % 12, int(m.group(2)))))
    return out


def tokens_in(offsets: List[int], a: int, z: int) -> List[int]:
    """字符区间 [a, z) 覆盖到的 token 下标；offsets[i] 是第 i 个 token 的起始字符位置（末尾多一个总长度）。"""
    return [i for i in range(len(offsets) - 1) if offsets[i] < z and offsets[i + 1] > a]


class SegController(ValleyController):
    def __init__(self, backend, cfg, arm, nbatch, n_in, img_abs, rec_layers, vis_ids, lam=0.0, gamma: float = GAMMA):
        super().__init__(backend, cfg, "base", nbatch, n_in, img_abs, rec_layers, vis_ids, lam=0.0)
        self.mode, self.gamma = arm, gamma
        self.segs: List[List[int]] = []          # 熵差片段（输出位置）
        self.seg_set: Set[int] = set()
        self.prop: Set[int] = set()              # 重复内容（输出位置）
        self.keys: Set[Tuple[int, int]] = set()  # 片段里出现过的时:分
        self._run, self._deep = [], False
        self.pieces: List[str] = []
        self.done = False
        self.isolated_steps = 0
        self._keys_t = None

    @property
    def marked(self) -> List[int]:
        return sorted(self.seg_set | self.prop)

    @property
    def passthrough(self):
        return not (self.branch == "main" and self.mode == "seg" and not self.closed and (self.seg_set or self.prop))

    def _call(self, kw, rope, record: bool = True):
        return super()._call(kw, rope, record=False)          # 熵差分支不需要记录注意力

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
            self.keys |= time_keys("".join(self.pieces[i] for i in self._run if i < len(self.pieces)))
        self._run, self._deep = [], False

    def _propagate(self):
        if not self.keys:
            return
        s = self.skip or 0
        text, offs = "", []
        for p in self.pieces[s:]:
            offs.append(len(text))
            text += p
        offs.append(len(text))
        for a, z, key in time_spans(text):
            if key in self.keys:
                for i in tokens_in(offs, a, z):
                    if s + i not in self.seg_set:
                        self.prop.add(s + i)

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
            self._propagate()
        if self.closed:
            self._close_run()
            self.done = True

    def modify(self, layer_idx, p, q_pos):
        if self.branch != "main" or self.mode != "seg" or p.shape[2] != 1 or self.closed:
            return p
        m = self.marked
        if not m:
            return p
        t = self.torch
        off = self.abs_q + 1 - p.shape[-1]
        if self._keys_t is None or self._keys_t[0] != (len(m), p.shape[-1], str(p.device)):
            keys = t.tensor([self.n_in + i - off for i in m if 0 <= self.n_in + i - off < p.shape[-1] - 1],
                            device=p.device, dtype=t.long)        # 不含当前这个词自己
            self._keys_t = ((len(m), p.shape[-1], str(p.device)), keys)
        keys = self._keys_t[1]
        if keys.numel() == 0:
            return p
        p = p.clone()
        p[..., keys] = p[..., keys] * self.gamma
        if layer_idx == min(self.rec_layers):
            self.isolated_steps += 1
        return p

    def extra(self) -> dict:
        dec = lambda xs: "".join(self.pieces[i] for i in xs if i < len(self.pieces))
        return {"mode": self.mode, "gamma": self.gamma, "segments": [dec(r) for r in self.segs],
                "segment_positions": self.segs, "seg_times": sorted(f"{h}:{m:02d}" for h, m in self.keys),
                "propagated_positions": sorted(self.prop), "propagated_text": dec(sorted(self.prop)),
                "isolated_steps": self.isolated_steps}


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
        print(f"  {n + 1}/{len(todo)} {iid}__{c} {arm}：{rec['n_out']} token，{rec['seconds']}s，熵差片段 {len(rec['segments'])} 段，"
              f"片段里的时间 {rec['seg_times']}，重复内容 {len(rec['propagated_positions'])} 个词，隔离了 {rec['isolated_steps']} 步，"
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


def summary(root: str, label: str = "", echo_root: Optional[str] = None, maskq_root: Optional[str] = None) -> None:
    S, E, M = _load(root), _load(echo_root), _load(maskq_root)
    B = {k: v for k, v in S.items() if k[2] == "base"} or {k: v for k, v in E.items() if k[2] == "base"}
    base_src = "本实验" if any(k[2] == "base" for k in S) else "回声隔离实验（同一随机种子）"
    ids = sorted({k[0] for k in S})
    rows = [("正常生成", B, "base"), ("回声隔离（只隔离开头片段、只在读图的 128 个词里）", E, "echo"),
            ("熵差片段隔离（整段思考、所有片段 + 重复内容）", S, "seg"),
            ("挡住提示里的用户那句话（fp-maskq，参考上限）", M, "mask")]
    kind = lambda r: {"坚持（答对）": "答对", "采纳": "照着用户的答案"}.get(r["outcome"], "答成别的时间")
    L = [f"# 熵差片段隔离 · {label}", "",
         "**怎么跑的**：ClockQA 20 道题（与回声隔离、fp-maskq 同一批），每道题问两次（用户给错误答案 / 用户给正确答案）：",
         "- 熵差片段隔离：思考里每段靠用户那句话写出来的话（连续负熵差、其中至少一步 ≤ −1）写完之后，"
         "以及思考里再次写出片段中同一个时:分的地方，之后写的每个词都读不到它们的内容（注意力权重不变，只把内容置零）。"
         "提示里用户那句话、图像都不动；思考结束后照常写答案。",
         f"- 正常生成取自{base_src}；回声隔离、挡住用户那句话是之前的结果，同样的题、同一随机种子，用来对比。", ""]
    for c, title in (("belief", "用户给错误答案时"), ("belief_right", "用户给正确答案时")):
        L += [f"## {title}（{len(ids)} 题）", "", "| | 答对 | 照着用户的答案 | 答成别的时间 |", "|---|---:|---:|---:|"]
        for name, D, arm in rows:
            rs = [D[(i, c, arm)] for i in ids if (i, c, arm) in D]
            if not rs:
                continue
            ks = [kind(r) for r in rs]
            L.append(f"| {name}（{len(rs)} 题） | {ks.count('答对')} | {ks.count('照着用户的答案')} | {ks.count('答成别的时间')} |")
        ch = [(i, kind(B[(i, c, 'base')]), kind(S[(i, c, 'seg')])) for i in ids if (i, c, "base") in B and (i, c, "seg") in S]
        up = [f"{i}（{a} → {b}）" for i, a, b in ch if a != "答对" and b == "答对"]
        dn = [f"{i}（{a} → {b}）" for i, a, b in ch if a == "答对" and b != "答对"]
        L += ["", f"熵差片段隔离相对正常生成：从错变对 {len(up)} 题{('：' + '、'.join(up)) if up else ''}；"
                  f"从对变错 {len(dn)} 题{('：' + '、'.join(dn)) if dn else ''}。", ""]
    ss = [r for r in S.values() if r["arm"] == "seg"]
    if ss:
        L += ["## 隔离了什么", "",
              f"- {len(ss)} 条回答里，{sum(bool(r['segments']) for r in ss)} 条找到了熵差片段，平均每条 "
              f"{np.mean([len(r['segments']) for r in ss]):.1f} 段；{sum(bool(r['propagated_positions']) for r in ss)} 条有重复内容被一起隔离；",
              "- 每条回答（片段里的时间 → 片段原文，前 4 段；重复内容的原文）：", ""]
        for r in sorted(ss, key=lambda x: (x["condition"], x["id"])):
            segs = " / ".join(repr(x[:60]) for x in r["segments"][:4])
            L.append(f"  - {r['id']} · {'用户说错' if r['condition'] == 'belief' else '用户说对'} · {kind(r)}："
                     f"时间 {r['seg_times']}；片段 {segs}；重复内容 {r['propagated_text'][:60]!r}")
    with open(os.path.join(root, "seg_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"[fp-seg] {len(S)} 条 → {root}/seg_report.md", file=sys.stderr)
