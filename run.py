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


if __name__ == "__main__":
    sys.exit(main())
