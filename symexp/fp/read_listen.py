# -*- coding: utf-8 -*-
"""先看再听（read-then-listen），ClockQA / PathVQA / PENDULUM 各 20 题，用户说错 / 说对。

阶段一（读图阶段）：思考开头由熵差分支来写——它的提示里问题换成模型自己改写的中性问题，看不到用户的话。
  每一步从熵差分支的分布采样（同一套温度、top-p 等设置；不允许它结束思考），主分支只把同样的词读进自己的缓存。
  读图阶段什么时候结束——看模型自己是不是还在看图：记录熵差分支每一步对图像的注意力（标准注意力层、所有头平均），
  按段落（空行分段）平均；一段写完时，如果它的看图量降到此前各段最高值的 RATIO 以下（且此前至少有一段），
  说明模型已经从看图转到推理，读图阶段在这一段写完后结束（已写出的词不能撤回，所以切换发生在这一段之后）。
  只计入至少 MIN_PAR 个词的段落；读图阶段最多 R_MAX 个词。不依赖任何数据集的答案格式。
  另外（规则 2，2026-11-09 起）：如果熵差分支这一步最可能写的是"结束思考"或"结束回答"的标记，说明中性模型认为已经读完，
  读图阶段在这一步结束、由主分支接着写（以前是禁止它结束，题目短时会写出乱码）。两个条件先到哪个算哪个。
  旧规则下撞到上限的记录（受影响的只有这些）重跑时会自动重新生成，旧文件移到 records_superseded/。
阶段二：切回主分支，模型带着自己读图的结果第一次面对用户的话，接着往下写。
  rl 组：之后每段熵差片段（连续负熵差、其中至少一步 ≤ −1）写完之后，后面的词读不到它的内容（权重乘 0，不重新归一化）；
  rl0 组：不隔离（只看读图阶段的作用）。提示（包括用户那句话）和图像都不动。
对照：base（正常生成）、itsm（ITSM：中性分支对比解码，fp-model2 的实现）、preread（先读后听：两次调用、预读 64–128 个词，
  preread.py 的实现）、reframe（Ask Don't Tell 的两步改写：直接回答模型改写出的中性问题，看不到用户的话）、
  mmsy（MM-SY，fp-model2 的实现，须单独一个进程跑）、s2va（S2VA，s2va.py 的实现），同一随机种子。
"""

import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

from .seg_isolate import SegController
from .valley_inject import CONDS, DATASETS, THRESH

RATIO = 0.5
MIN_PAR = 8
R_MAX = 512
ARMS = ["base", "reframe", "itsm", "mmsy", "s2va", "preread", "rl", "rl0"]
RL_RULE = 2


def wants_end(ref_top: int, end_ids) -> bool:
    """熵差分支这一步最可能写的词是不是"结束思考 / 结束回答"的标记。"""
    return int(ref_top) in {int(x) for x in end_ids if x is not None}


def needs_redo(rec: dict) -> bool:
    """旧规则（禁止熵差分支结束思考）下撞到读图上限的记录：按新规则重新生成。"""
    return (rec.get("arm") in ("rl", "rl0") and int(rec.get("rl_rule") or 1) < RL_RULE
            and str(rec.get("switch_reason") or "").startswith("读图阶段达到上限"))
DEFAULT_ARMS = ["base", "itsm", "rl"]


def para_switch(masses: List[float], ratio: float = RATIO) -> bool:
    """masses：已写完、计入的各段平均看图量（按顺序）。最后一段是否降到此前最高值的 ratio 以下。"""
    if len(masses) < 2:
        return False
    return masses[-1] < ratio * max(masses[:-1])


class RLController(SegController):
    def __init__(self, backend, cfg, arm, nbatch, n_in, img_abs, rec_layers, vis_ids, lam=0.0, gamma: float = 0.0,
                 ratio: float = RATIO, min_par: int = MIN_PAR, r_max: int = R_MAX):
        super().__init__(backend, cfg, arm, nbatch, n_in, img_abs, rec_layers, vis_ids, gamma=gamma)
        self.ratio, self.min_par, self.r_max = ratio, min_par, r_max
        self.reading = True
        self.switch_at: Optional[int] = None          # 第一个由主分支写的输出位置
        self.switch_reason = None
        self.ref_mass: List[float] = []                 # 每个输出词作为当前词时，熵差分支的看图量
        self._par, self.par_masses, self.par_info = [], [], []
        self.eos = set()
        gc = self.model.generation_config
        for e in (gc.eos_token_id if isinstance(gc.eos_token_id, list) else [gc.eos_token_id]):
            if e is not None:
                self.eos.add(int(e))

    def _iso_here(self) -> bool:
        return self.mode == "rl" and self.branch == "main"

    def _call(self, kw, rope, record: bool = True):
        from .valley_inject import ValleyController
        return ValleyController._call(self, kw, rope, record=self.reading)   # 读图阶段记录熵差分支的看图注意力

    def accumulate(self, layer_idx, probs, stages, q_pos):
        from .valley_inject import ValleyController
        if self.branch == "main" or self.reading:
            ValleyController.accumulate(self, layer_idx, probs, stages, q_pos)

    def _end_reading(self, k: int, reason: str):
        self.reading = False
        self.switch_at, self.switch_reason = k, reason

    def post_hook(self, module, args, output):
        from .valley_inject import ValleyController
        if self.branch == "ref" or self.done:
            return
        ValleyController.post_hook(self, module, args, output)
        k = len(self.out)                                       # 这一步预测第 k 个输出词
        if self.reading and self.closed:                         # 思考被总长度上限强制结束
            self._end_reading(k, "思考被长度上限截断")
        if self.reading and k > 0:
            a = self.acc.get("ref")
            m = float(a["mass"] / a["n"]) if a and a["n"] else float("nan")
            self.ref_mass.append(m)                             # 属于第 k − 1 个词
            started = self.skip is not None and k - 1 >= self.skip
            if started:
                self._par.append(k - 1)
                if "\n\n" in self.pieces[k - 1]:
                    v = [self.ref_mass[i] for i in self._par if self.ref_mass[i] == self.ref_mass[i]]
                    if len(self._par) >= self.min_par and v:
                        self.par_masses.append(float(np.mean(v)))
                        self.par_info.append({"start": self._par[0], "end": self._par[-1], "mass": float(np.mean(v)),
                                              "text": "".join(self.pieces[i] for i in self._par)[:200]})
                        if para_switch(self.par_masses, self.ratio):
                            self._end_reading(k, "看图量降到此前最高段的一半以下")
                    self._par = []
            if self.reading and self.skip is not None and k - self.skip >= self.r_max:
                self._end_reading(k, f"读图阶段达到上限 {self.r_max} 个词")
        if self.reading and wants_end(int(self.ref_logits_now.argmax()), [self.b.close_id, *self.eos]):
            self._end_reading(k, "熵差分支要结束思考")                 # 这一步起由主分支来写
        if self.reading:                                        # 这一步从熵差分支采样（不是最可能结束时，不让它偶然采到结束）
            t = self.torch
            lg = output.logits[0, -1]
            rp = self.ref_logits_now.to(lg.device).clone()
            for x in [self.b.close_id, *self.eos]:
                if x is not None and 0 <= x < rp.shape[-1]:
                    rp[x] = -float("inf")
            output.logits[0, -1] = rp.to(lg.dtype)
            return
        started = self.switch_at is not None and k >= self.switch_at and not self.closed
        if started:
            dH = self.dH[-1]
            if dH < 0:
                self._run.append(k)
                self._deep = self._deep or dH <= THRESH
            else:
                self._close_run()
        if self.closed:
            self._close_run()
            self.done = True

    def extra(self) -> dict:
        e = super().extra()
        s = self.skip or 0
        e.update(rl_rule=RL_RULE, switch_at=self.switch_at, switch_reason=self.switch_reason, reading_tokens=(self.switch_at or 0) - s,
                 reading_text="".join(self.pieces[s:self.switch_at or s]), paragraphs=self.par_info,
                 ref_mass=[round(x, 5) for x in self.ref_mass])
        return e


# ---------------------------------------------------------------- 中性问题（模型自己改写）
def neutralize(cfg, root: str, ent_dir: str, model_key: str, shard: str = "") -> None:
    from ..backends import GenRequest
    from .conflict import _engine
    from .model2 import NEUTRALIZE_PROMPT, neutral_leak
    from .pipeline import _shard
    from .trigger_regions import load_entropy
    from .valley_inject import selection
    d = os.path.join(root, "neutralized")
    os.makedirs(d, exist_ok=True)
    ent = load_entropy(ent_dir, model_key, DATASETS)
    sel = selection(ent)
    tasks = _shard([(i, c) for ds in DATASETS for i in sel[ds] for c in CONDS], shard)
    todo = [t for t in tasks if not os.path.exists(os.path.join(d, f"{t[0]}__{t[1]}.json"))]
    print(f"[fp-rl neutralize] 待做 {len(todo)} 条", file=sys.stderr)
    if not todo:
        return
    eng = _engine(cfg)
    b = eng.backend
    torch, model, tok = b.torch, b.model, b.tokenizer
    for n, (iid, c) in enumerate(todo):
        it = ent[(iid, c)]["job"]["item"]
        prompt = b._template(GenRequest(user_text=NEUTRALIZE_PROMPT.format(msg=it["messages"][c]), images=[]), thinking=False)
        inp = b.processor(text=[prompt], return_tensors="pt")
        batch = {k: v.to(model.device) for k, v in inp.items() if hasattr(v, "to")}
        with torch.inference_mode():
            o = model.generate(**batch, do_sample=False, max_new_tokens=200, use_cache=True)
        text = tok.decode(o[0, batch["input_ids"].shape[1]:].tolist(), skip_special_tokens=True).strip()
        if not text:
            raise SystemExit(f"{iid}__{c}：改写结果为空")
        q = it["messages"]["neutral"].strip()
        rec = {"id": iid, "condition": c, "original": it["messages"][c], "neutral": text,
               "exactly_original_question": text.strip() == q}
        try:
            rec.update(neutral_leak(text, dict(it, question=it.get("question", q)), c))
        except Exception:
            pass
        p = os.path.join(d, f"{iid}__{c}.json")
        with open(p + ".tmp", "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        os.replace(p + ".tmp", p)
        print(f"  {n + 1}/{len(todo)} {iid}__{c} → {text[:80]!r}，与原问题一字不差 = {rec['exactly_original_question']}",
              file=sys.stderr)


# ---------------------------------------------------------------- 生成
def run(cfg, root: str, ent_dir: str, model_key: str, image_dirs: Optional[List[str]] = None, shard: str = "",
        seed: int = 42, arms: Optional[List[str]] = None, datasets: Optional[List[str]] = None,
        ids: Optional[List[str]] = None, per_dataset: int = 20) -> None:
    from .conflict import _engine, claim_of
    from .mask_query import first_reading
    from .model2 import generate2
    from .pipeline import _shard
    from .trigger_regions import load_entropy, resolve_image, resolve_source
    from .valley_inject import claim_answer, generate, select_items
    from .verify import final_time, outcome
    arms = arms or DEFAULT_ARMS
    datasets = datasets or DATASETS
    if "mmsy" in arms and len(arms) > 1:
        raise SystemExit("mmsy 会另装一套注意力包装，请单独一个进程跑（--arms mmsy）")
    os.makedirs(os.path.join(root, "records"), exist_ok=True)
    ent = load_entropy(ent_dir, model_key, DATASETS)
    sel = {ds: select_items({k[0]: v["group"] for k, v in ent.items() if v["dataset"] == ds and k[1] == "belief"}, per_dataset)
           for ds in DATASETS}                                  # per_dataset = 20 时与熵谷注入、主实验的选题完全相同
    with open(os.path.join(root, "selection.json" if per_dataset == 20 else f"selection_n{per_dataset}.json"), "w",
              encoding="utf-8") as f:
        json.dump(sel, f, ensure_ascii=False, indent=1)
    tasks = [(ds, i, c, a) for ds in datasets for i in sel[ds] for c in CONDS for a in arms if not ids or i in ids]
    tasks = _shard(tasks, shard)
    todo, redo = [], 0
    for x in tasks:
        p = os.path.join(root, "records", f"{x[1]}__{x[2]}__{x[3]}.json")
        if not os.path.exists(p):
            todo.append(x)
        elif needs_redo(json.load(open(p, encoding="utf-8"))):
            os.makedirs(os.path.join(root, "records_superseded"), exist_ok=True)
            os.replace(p, os.path.join(root, "records_superseded", os.path.basename(p)))
            todo.append(x)
            redo += 1
    print(f"[fp-rl] 本进程 {len(tasks)} 项，待做 {len(todo)} 项（其中旧规则下撞到读图上限、按新规则重做 {redo} 项，"
          f"旧文件移到 records_superseded/）", file=sys.stderr)
    eng = None
    for n, (ds, iid, c, arm) in enumerate(todo):
        e = ent[(iid, c)]
        job = e["job"]
        item = dict(job["item"])
        orig = json.load(open(resolve_source(job), encoding="utf-8"))
        item["image"] = resolve_image(job, orig, image_dirs)
        item.setdefault("id", iid)
        npath = os.path.join(root, "neutralized", f"{iid}__{c}.json")
        if arm in ("itsm", "preread", "rl", "rl0", "reframe", "s2va"):
            if not os.path.exists(npath):
                raise SystemExit(f"缺少模型改写的中性问题 {npath}（先跑 --phase neutralize）")
            nq = json.load(open(npath, encoding="utf-8"))["neutral"]
        else:
            nq = None
        if eng is None:
            eng = _engine(cfg)
            if "mmsy" in arms:
                from .model2 import install
                layer_info = install(eng.backend.model)
        if arm in ("base", "itsm"):
            rec = generate2(eng.backend, cfg, item, c, arm, seed=seed, neutral_text=nq)
        elif arm == "mmsy":
            rec = generate2(eng.backend, cfg, item, c, "mmsy", seed=seed, layer_info=layer_info)
        elif arm == "reframe":                                  # 直接回答模型改写出的中性问题（看不到用户的话）
            it2 = dict(item, messages=dict(item["messages"], **{c: nq}))
            rec = generate2(eng.backend, cfg, it2, c, "base", seed=seed)
            rec["message"] = item["messages"][c]
        elif arm == "s2va":
            from .s2va import generate_s2va
            rec = generate_s2va(eng.backend, cfg, item, c, seed, neutral_text=nq)
        elif arm == "preread":
            from .preread import generate_preread
            rec = generate_preread(eng.backend, cfg, item, c, seed, neutral_text=nq)
        else:
            rec = generate(eng.backend, cfg, item, c, arm, seed=seed, lam=0.0, ctrl_cls=RLController, attention=True,
                           all_layers=True, neutral_text=nq)
        claim = claim_of(item, c)
        rec.update(id=iid, dataset=ds, condition=c, arm=arm, group=e["group"], gold=item["gold"], claim=claim,
                   question=item["messages"]["neutral"], image=item["image"], wrong_claim=claim_answer(item.get("claim")),
                   neutral_question=nq)
        if ds == "ClockQA":
            rec["outcome"] = outcome(rec["answer"], item["gold"], claim)
            rec["final_time"] = final_time(rec["answer"], claim)
        p = os.path.join(root, "records", f"{iid}__{c}__{arm}.json")
        with open(p + ".tmp", "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        os.replace(p + ".tmp", p)
        extra = ""
        if arm in ("rl", "rl0"):
            extra = (f"，读图阶段 {rec['reading_tokens']} 个词（{rec['switch_reason']}），熵差片段 {len(rec['segments'])} 段，"
                     f"熵差分支核对 {rec['ref_check']}")
        print(f"  {n + 1}/{len(todo)} {ds} {iid}__{c} {arm}：{rec['n_out']} token，{rec['seconds']}s{extra}"
              + (f"，{rec['outcome']}" if ds == "ClockQA" else ""), file=sys.stderr)


def summary(root: str, label: str = "") -> None:
    from .valley_inject import summary as vsum
    notes = [
        "- base：正常生成；reframe：直接回答模型改写出的中性问题（Ask Don't Tell 两步改写，看不到用户的话）；"
        "itsm：ITSM（中性分支对比解码）；mmsy：MM-SY；s2va：S2VA；preread：先读后听（两次调用，预读 64–128 个词，贴在思考开头）；",
        f"- rl（先看再听）：思考开头由看不到用户的话的熵差分支来写，写到某一段的看图量降到此前最高段的 {RATIO} 以下"
        f"、或中性模型最可能要结束思考（或满 {R_MAX} 个词）为止；之后切回主分支，熵差片段写完后隔离；rl0：同样的读图阶段，之后不隔离；",
        "- 中性问题都是模型自己改写的；各组同一随机种子。"]
    vsum(root, label, title="先看再听（读图阶段看不到用户的话，之后用熵差隔离）", notes=notes, arms_order=ARMS)
    recs = [json.load(open(os.path.join(root, "records", f), encoding="utf-8"))
            for f in sorted(os.listdir(os.path.join(root, "records"))) if f.endswith(".json")]
    L = ["", "## 读图阶段", "", "| 数据集 | 组 | 回答数 | 读图阶段平均词数 | 因看图量下降而切换 | 因中性模型要结束思考而切换 | 因达到上限而切换 |",
         "|---|---|---:|---:|---:|---:|---:|"]
    for ds in DATASETS:
        for a in ("rl", "rl0"):
            rs = [r for r in recs if r["dataset"] == ds and r["arm"] == a]
            if not rs:
                continue
            drop = sum(1 for r in rs if (r.get("switch_reason") or "").startswith("看图量"))
            cap = sum(1 for r in rs if (r.get("switch_reason") or "").startswith("读图阶段达到上限"))
            end = sum(1 for r in rs if (r.get("switch_reason") or "").startswith("熵差分支要结束"))
            L.append(f"| {ds} | {a} | {len(rs)} | {np.mean([r['reading_tokens'] for r in rs]):.0f} | {drop} | {end} | {cap} |")
    L += ["", "## 耗时（秒 / 条，平均）", "", "| 数据集 | " + " | ".join(ARMS) + " |", "|---|" + "---:|" * len(ARMS)]
    for ds in DATASETS:
        cells = []
        for a in ARMS:
            rs = [r["seconds"] for r in recs if r["dataset"] == ds and r["arm"] == a]
            cells.append(f"{np.mean(rs):.0f}" if rs else "")
        L.append(f"| {ds} | " + " | ".join(cells) + " |")
    with open(os.path.join(root, "summary_report.md"), "a", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
