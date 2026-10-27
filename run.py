#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""命令行入口。用法见 README.md。"""

import argparse
import json
import os
import sys

from symexp.config import Config
from symexp.data import build_conditions, prepare_pathvqa, prepare_sycophancy, prepare_synthetic
from symexp.evaluate import evaluate
from symexp.judge import export_requests, import_judgments
from symexp.runner import run


def _overrides(pairs):
    out = {}
    for kv in pairs or []:
        k, v = kv.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser(description="抗谄媚实验方案 v1：plain / SRT 提示版 / 本方法")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="下载并整理题目，生成 base.jsonl 与 conditions.jsonl")
    p.add_argument("--source", choices=["sycophancy", "pathvqa", "synthetic"], default="sycophancy")
    p.add_argument("--out", required=True, help="数据目录")
    p.add_argument("--n", type=int, default=0, help="按答案分层抽几题；0 = 全部（pathvqa 来源默认 300）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--subset-filter", default="path", help="sycophancy 来源里 dataset 字段要包含的字符串")
    p.add_argument("--pathvqa-split", default="test")

    b = sub.add_parser("conditions", help="从 base.jsonl 生成条件")
    b.add_argument("--base", required=True)
    b.add_argument("--out", required=True)

    for q in (p, b):
        q.add_argument("--ptype-mode", choices=["assign", "all"], default="assign",
                       help="assign：每题平均分配一种压力方式（3 个条件）；all：每题 5 种全跑（11 个条件）")
        q.add_argument("--ptype-seed", type=int, default=0, help="分配压力方式用的随机种子")

    r = sub.add_parser("run", help="跑一种方法")
    r.add_argument("--method", choices=["plain", "srt", "ours", "ours_direct", "lqcd"], required=True)
    r.add_argument("--config", default=None)
    r.add_argument("--data", required=True, help="conditions.jsonl")
    r.add_argument("--out", required=True, help="这一次运行的输出目录")
    r.add_argument("--seeds", type=int, nargs="+", default=[42], help="默认 42，与复用的 plain 相同")
    r.add_argument("--conditions", nargs="*", default=None, help="只跑某些条件，例如 neutral")
    r.add_argument("--ptypes", nargs="*", default=None)
    r.add_argument("--ids", nargs="*", default=None)
    r.add_argument("--limit", type=int, default=0, help="只跑前 N 题")
    r.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="临时覆盖配置，值按 JSON 解析，例如 --set workers=8 answer_budget=1024")

    je = sub.add_parser("judge-export", help="导出最终回复给 Luna 判读")
    je.add_argument("--runs", nargs="+", required=True)
    je.add_argument("--out", required=True)

    ji = sub.add_parser("judge-import", help="导回 Luna 的判读结果")
    ji.add_argument("--runs", nargs="+", required=True)
    ji.add_argument("--judgments", required=True)

    ev = sub.add_parser("eval", help="评测")
    ev.add_argument("--runs", nargs="+", required=True)
    ev.add_argument("--out", required=True)
    ev.add_argument("--bootstrap", type=int, default=2000)

    ar = sub.add_parser("attn-replay", help="只读前向回放已保存的序列，按阶段记录图像注意力（不生成）")
    ar.add_argument("--config", default=None)
    ar.add_argument("--manifest", required=True, help="序列清单 JSONL，格式见 symexp/attn_replay.py")
    ar.add_argument("--out", required=True)
    ar.add_argument("--limit", type=int, default=0, help="只跑前 N 题")
    ar.add_argument("--ids", nargs="*", default=None)
    ar.add_argument("--no-positions", action="store_true", help="不保存逐位置图像总量（校验要用，默认保存）")
    ar.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")

    av = sub.add_parser("attn-validate", help="把回放结果和预实验保存的 generation_masses 对比")
    av.add_argument("--replay", required=True)
    av.add_argument("--saved", required=True, help="文件路径模板，用 {id} 和 {condition} 占位")
    av.add_argument("--array", default=None, help=".npz 里的数组名")
    av.add_argument("--class-index", type=int, default=0, help="类别维里“图像”那一类的序号")
    av.add_argument("--out", default=None)

    aa = sub.add_parser("attn-analyze", help="比较两组的图像注意力区域")
    aa.add_argument("--ref", required=True, help="参照组 NAME=DIR@TAG，文件为 DIR/{id}__{TAG}.npz")
    aa.add_argument("--cmp", nargs="+", required=True, help="比较组，格式同上，可多个")
    aa.add_argument("--out", required=True)
    aa.add_argument("--runs", nargs="*", default=None, help="plain 中性和受压的运行目录，用来标出哪些题改口")
    aa.add_argument("--crit", default="answer", choices=["answer", "full"])
    aa.add_argument("--label-method", default="plain", help="用哪个方法的结果标“改口”（plain 或 lqcd 等）")
    aa.add_argument("--heatmaps", nargs="*", default=None, help="画热图的题号（prefill 末尾、思考、回答三个阶段各一张）")
    aa.add_argument("--heatmap-shared-scale", action="store_true", help="同一阶段所有热图用统一色阶，便于题间比较")
    aa.add_argument("--bootstrap", type=int, default=2000)

    # ---- 五压力实验（PathVQA 重做），说明见 docs/服务器任务_PathVQA五压力_重做.md ----
    fp = sub.add_parser("fp-prepare", help="五压力：取前 N 题，生成六个条件的消息")
    fp.add_argument("--root", required=True)
    fp.add_argument("--n", type=int, default=100)
    fp.add_argument("--source", choices=["sycophancy", "synthetic"], default="sycophancy")
    fo = sub.add_parser("fp-observe", help="五压力：中性 + 5 种压力各生成一次，在线记录注意力")
    fo.add_argument("--conditions", nargs="*", default=None)
    fo.add_argument("--limit", type=int, default=0, help="只跑前 N 题")
    fx = sub.add_parser("fp-export", help="五压力：导出判读请求（判读者看不到条件和组别）")
    fx.add_argument("--phase", choices=["obs", "int"], required=True)
    fm = sub.add_parser("fp-import", help="五压力：导回判读")
    fm.add_argument("--phase", choices=["obs", "int"], required=True)
    fm.add_argument("--judgments", required=True)
    fc = sub.add_parser("fp-candidates", help="五压力：选出全部“中性答对、受压答错”的观测")
    fc.add_argument("--criterion", choices=["answer", "joint"], default="answer", help="answer = 只看最终答案（原论文）；joint = 答案和解释都对")
    fi = sub.add_parser("fp-intervene", help="五压力：干预（先 pilot，阅读确认后 full）")
    fi.add_argument("--phase", choices=["pilot", "full"], required=True)
    fi.add_argument("--arms", nargs="*", default=None, help="默认 restore_E rand_matched")
    fpub = sub.add_parser("fp-public", help="五压力：整理可上传 GitHub 的内容（去掉 token 列表，数组只列清单）")
    fpub.add_argument("--out", required=True)
    fa = sub.add_parser("fp-analyze", help="五压力：汇总 A 注意力变化、B 干预效果")
    fa.add_argument("--out", default=None)
    fa.add_argument("--heatmaps", type=int, default=12, help="每种压力画前几题的热图")
    fa.add_argument("--criterion", choices=["answer", "joint"], default=None, help="默认沿用 candidates.json 的标准")
    fss = sub.add_parser("fp-subset-setup", help="五压力：从主实验目录挑出指定观测，放进新目录单独干预")
    fss.add_argument("--src", required=True, help="主实验目录（只读）")
    fss.add_argument("--dst", required=True, help="新目录")
    fss.add_argument("--cases", required=True, help="观测清单 JSON，见 symexp/fp/subset.py")
    frg = sub.add_parser("fp-region", help="五压力：只用 CPU，比较中性与受压时关键区域份额（按分组）")
    frg.add_argument("--src", required=True)
    frg.add_argument("--cases", required=True)
    frg.add_argument("--out", required=True)
    fpu = sub.add_parser("fp-push", help="五压力：推力分析（重读已保存的记录，不生成）")
    fpu.add_argument("--cases", required=True, help="观测清单 JSON，见 docs/pathvqa_push10_cases.json")
    fpu.add_argument("--out", required=True)
    fdu = sub.add_parser("fp-dual", help="五压力：锚点一 v2（逐步参照，双分支同步），先 pilot，阅读确认后 full")
    fdu.add_argument("--phase", choices=["pilot", "full"], required=True)
    fdu.add_argument("--lam", type=float, default=1.0, help="混合系数 λ（默认 1：换成中性分支此刻的分布）")
    fas = sub.add_parser("fp-agree-setup", help="五压力：建“用户说对”（主张 = 正确答案）的对照目录")
    fas.add_argument("--src", required=True, help="已有实验目录（只读）")
    fas.add_argument("--dst", required=True, help="新目录")
    fas.add_argument("--cases", required=True, help="题目清单 JSON（intervene: [[id, 压力], ...]）")
    fag = sub.add_parser("fp-agree", help="五压力：生成“用户说对”条件（与观察阶段相同的设置）")
    fen = sub.add_parser("fp-entropy", help="五压力：熵差分析（重读已保存的轨迹，受压/中性提示各前向一次，不生成）")
    fen.add_argument("--out", required=True)
    for q in (fo, fi, fpu, fdu, fen, fag):
        q.add_argument("--config", default=None)
        q.add_argument("--shard", default="", help="k/n：只做第 k 份（共 n 份），多张卡并行时用")
        q.add_argument("--seed", type=int, default=42)
        q.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    for q in (fx, fm, fc, fi, fa, fo, fpub, fpu, fdu, fen, fag):
        q.add_argument("--root", required=True)

    a = ap.parse_args()
    if a.cmd == "prepare":
        if a.source == "sycophancy":
            prepare_sycophancy(a.out, n=a.n, seed=a.seed, subset_filter=a.subset_filter)
        elif a.source == "pathvqa":
            prepare_pathvqa(a.out, n=a.n or 300, seed=a.seed, split=a.pathvqa_split)
        else:
            prepare_synthetic(a.out, n=a.n or 8, seed=a.seed)
        build_conditions(os.path.join(a.out, "base.jsonl"), os.path.join(a.out, "conditions.jsonl"),
                         ptype_mode=a.ptype_mode, ptype_seed=a.ptype_seed)
    elif a.cmd == "conditions":
        build_conditions(a.base, a.out, ptype_mode=a.ptype_mode, ptype_seed=a.ptype_seed)
    elif a.cmd == "run":
        cfg = Config.load(a.config, _overrides(a.set))
        run(cfg, a.method, a.data, a.out, a.seeds, a.conditions, a.ptypes, a.limit, a.ids)
    elif a.cmd == "judge-export":
        n = export_requests(a.runs, a.out)
        print(f"exported {n} replies -> {a.out}")
    elif a.cmd == "judge-import":
        n = import_judgments(a.runs, a.judgments)
        print(f"imported {n} judgments")
    elif a.cmd == "attn-replay":
        from symexp.attn_replay import run_replay
        cfg = Config.load(a.config, _overrides(a.set))
        run_replay(cfg, a.manifest, a.out, a.limit, not a.no_positions, a.ids)
    elif a.cmd == "attn-validate":
        from symexp.attn_analysis import validate
        res = validate(a.replay, a.saved, a.array, a.class_index)
        print(json.dumps(res["summary"], ensure_ascii=False, indent=2))
        out = a.out or os.path.join(a.replay, "validate.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
    elif a.cmd == "attn-analyze":
        from symexp.attn_analysis import analyze, flip_labels_from_runs
        labels = flip_labels_from_runs(a.runs, a.crit, a.label_method) if a.runs else None
        print(analyze(a.ref, a.cmp, a.out, labels, a.heatmaps, a.bootstrap, shared_scale=a.heatmap_shared_scale))
    elif a.cmd == "eval":
        print(evaluate(a.runs, a.out, reps=a.bootstrap))
    elif a.cmd == "fp-agree-setup":
        from symexp.fp import agree
        agree.setup(a.src, a.dst, a.cases)
    elif a.cmd == "fp-agree":
        from symexp.fp import agree
        agree.run(Config.load(a.config, _overrides(a.set)), a.root, a.shard, a.seed)
    elif a.cmd in ("fp-subset-setup", "fp-region"):
        from symexp.fp import subset
        if a.cmd == "fp-subset-setup":
            subset.setup(a.src, a.dst, a.cases)
        else:
            subset.region(a.src, a.cases, a.out)
    elif a.cmd.startswith("fp-"):
        from symexp.fp import pipeline as fpp
        if a.cmd == "fp-prepare":
            fpp.prepare(a.root, a.n, a.source)
        elif a.cmd == "fp-observe":
            fpp.observe(Config.load(a.config, _overrides(a.set)), a.root, a.shard, a.limit, a.conditions, a.seed)
        elif a.cmd == "fp-export":
            print(f"exported {fpp.judge_export(a.root, a.phase)} replies -> {a.root}/{a.phase}/judge/requests.jsonl")
        elif a.cmd == "fp-import":
            print(f"imported {fpp.judge_import(a.root, a.phase, a.judgments)} judgments")
        elif a.cmd == "fp-candidates":
            fpp.candidates(a.root, a.criterion)
        elif a.cmd == "fp-intervene":
            fpp.intervene(Config.load(a.config, _overrides(a.set)), a.root, a.phase, a.arms, a.shard, a.seed)
        elif a.cmd == "fp-entropy":
            from symexp.fp.entropy import run as fp_entropy
            fp_entropy(Config.load(a.config, _overrides(a.set)), a.root, a.out)
        elif a.cmd == "fp-dual":
            from symexp.fp.dual import run as fp_dual
            fp_dual(Config.load(a.config, _overrides(a.set)), a.root, a.phase, a.shard, a.seed, a.lam)
        elif a.cmd == "fp-push":
            from symexp.fp.push import run as fp_push
            fp_push(Config.load(a.config, _overrides(a.set)), a.root, a.cases, a.out)
        elif a.cmd == "fp-public":
            print(f"{fpp.public(a.root, a.out)} records -> {a.out}")
        elif a.cmd == "fp-analyze":
            from symexp.fp.analyze import analyze as fp_analyze
            print(fp_analyze(a.root, a.out, a.heatmaps, a.criterion))


if __name__ == "__main__":
    sys.exit(main())
