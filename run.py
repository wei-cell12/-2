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
    fsp = sub.add_parser("fp-spans", help="五压力：逐 token 注意力去向（图像/问题/用户主张/自己写的内容），整段前向一次，不生成")
    fsp.add_argument("--roots", nargs="+", required=True, help="一个或多个实验目录（只读）")
    fsp.add_argument("--subs", default="obs,int", help="读哪些记录：obs、int 或 obs,int")
    fsp.add_argument("--out", required=True)
    fhd = sub.add_parser("fp-hidden", help="五压力：犹豫方向（隐藏状态，留一题求方向并投影），forward 用 GPU、analyze 用 CPU")
    fhd.add_argument("--phase", required=True, choices=["forward", "analyze"])
    fhd.add_argument("--roots", nargs="*", default=[], help="forward：一个或多个实验目录（只读）")
    fhd.add_argument("--subs", default="obs,int", help="forward：读哪些记录：obs、int 或 obs,int")
    fhd.add_argument("--work", required=True, help="工作目录（隐藏状态存在 work/hidden/，不上传）")
    fhd.add_argument("--out", default=None, help="analyze：输出 JSON")
    fcs = sub.add_parser("fp-conflict-setup", help="五压力：冲突方向实验建目录（PathVQA 加“用户说对”消息；ClockQA 从原始观察导出 62 题）")
    fcs.add_argument("--pv-src", required=True, help="A800 主实验目录（只读）")
    fcs.add_argument("--pv", required=True, help="PathVQA 新目录")
    fcs.add_argument("--native", required=True, help="ClockQA 原始观察 native_outputs.jsonl（仓库 results/attn_intervene_20261025/clockqa_server59/five_pressure/）")
    fcs.add_argument("--cq", required=True, help="ClockQA 新目录")
    fcs.add_argument("--image-dirs", nargs="*", default=[], help="ClockQA 图像所在目录（从原始记录找不到图像时才用）")
    fcf = sub.add_parser("fp-conflict", help="五压力：冲突方向（pairs / project 用 GPU，directions 用 CPU）")
    fcf.add_argument("--phase", required=True, choices=["pairs", "directions", "project", "external"])
    fcf.add_argument("--external", default=None, help="external：外部轨迹 JSONL")
    fcf.add_argument("--pv-src", required=True)
    fcf.add_argument("--pv", required=True)
    fcf.add_argument("--cq", default=None, help="ClockQA 目录；不给就只做 PathVQA（ClockQA 生成完后再加上它重跑三步）")
    fcf.add_argument("--work", required=True, help="工作目录（配对差、方向存在这里，不上传）")
    fcf.add_argument("--out", default=None, help="project：输出 JSON")
    fvf = sub.add_parser("fp-verify", help="五压力：冲突触发的独立核查（续写实验，A 基线 / B 只有工具说明 / C 完整方法）")
    fvf.add_argument("--cq", required=True, help="ClockQA 目录（fp-conflict-setup 建的）")
    fvf.add_argument("--work", required=True, help="冲突方向的工作目录（读 directions.npz，结果存 work/verify/）")
    fvf.add_argument("--cases", required=True, help="题目清单 JSON（results/conflict_direction_20261028/verify_cases_cq8.json）")
    fvf.add_argument("--out", required=True)
    fvf.add_argument("--threshold", type=float, default=-9.28, help="触发阈值（ClockQA、PathVQA 方向、误触发 20%%）")
    fvf.add_argument("--conds", nargs="*", default=["A", "B", "C"])
    fvf.add_argument("--samples", type=int, default=3)
    fvf.add_argument("--only", nargs="*", default=None, help="只跑这些题（题号如 CQ-039，或清单里的 tag）")
    fcm = sub.add_parser("fp-compare", help="五压力：ClockQA 完整方法（锚点一+锚点二）与 baseline 的对比（gate / run / summary）")
    fcm.add_argument("--phase", required=True, choices=["gate", "run", "time-export", "time-import", "summary"])
    fcm.add_argument("--judgments", default=None,
                     help="time-import：判读结果 .jsonl；不给就读 root/time_judge/ 下所有 judgments_raw*.jsonl")
    fcm.add_argument("--cq", required=True, help="ClockQA 目录（fp-conflict-setup 建的；读题目、中性与原模型记录、中性注意力）")
    fcm.add_argument("--work", default=None, help="冲突方向的工作目录（读 directions.npz）；gate、run 需要")
    fcm.add_argument("--root", required=True, help="对比实验的新目录（结果存这里）")
    fcm.add_argument("--arms", nargs="*", default=["mmsy", "full", "itsm"],
                     help="gate / run：哪些方法（mmsy full itsm a1 a2 prompt）")
    fcm.add_argument("--conds", nargs="*", default=["belief", "belief_right"])
    fcm.add_argument("--threshold", type=float, default=-9.28)
    fcm.add_argument("--seeds", nargs="*", type=int, default=None, help="run：每个方法跑哪些种子（默认只跑 --seed）")
    fcm.add_argument("--out", default=None, help="summary：输出 JSON")
    fsr = sub.add_parser("fp-sources", help="ClockQA：读时间时注意力落在图像还是用户的话上（replay 用 GPU 只读回放，analyze 用 CPU）")
    fsr.add_argument("--phase", required=True, choices=["replay", "analyze", "tokens", "prompt", "prompt-analyze", "prompt-heads", "cut", "cut-analyze", "reading", "reading-analyze", "reading-claim", "reading-gen", "reading-groups", "sinks", "sinks-analyze", "reading-split", "reading-scores", "reading-bands", "reading-focus", "reading-gen-all", "reading-layers", "stage-export", "stage-summary", "attrib", "attrib-summary", "reading-layers-e", "sink-check"])
    fsr.add_argument("--cq", required=True, help="ClockQA 目录（读题目、原模型记录、中性注意力）")
    fsr.add_argument("--root", required=True, help="新目录（结果存这里）")
    fsr.add_argument("--conds", nargs="*", default=["neutral", "belief", "belief_right"])
    fsr.add_argument("--judgments", default=None, help="analyze：时间判读结果 judgments.json（含 obs|题号|条件）")
    fsr.add_argument("--out", default=None, help="analyze：报告 .md（默认 root/sources_report.md）")
    fsr.add_argument("--yn-judgments", default=None, help="reading-groups / reading-focus（PathVQA）：是/否判读的 judgments.json")
    fsr.add_argument("--label", default=None, help="reading-focus：报告里写的模型与数据集，如 \"Qwen3.8-27B · ClockQA\"")
    ftr = sub.add_parser("fp-trigger", help="熵差报警前后，看图注意力落在哪里（ClockQA 另看真指针区 / 用户指针区；replay 用 GPU 只读回放，analyze 用 CPU）")
    ftr.add_argument("--phase", required=True, choices=["replay", "analyze"])
    ftr.add_argument("--datasets", nargs="*", default=None, help="replay：只做这些数据集（ClockQA PathVQA PENDULUM，默认全部）")
    ftr.add_argument("--image-dirs", nargs="*", default=None, help="replay：熵差实验记录的图像路径不存在时，按 SHA 在这些目录里找图像")
    ftr.add_argument("--root", required=True, help="结果目录")
    ftr.add_argument("--entropy-dir", default=None, help="replay：熵差实验目录（仓库里的 results/shared_prefix_entropy_20261103）")
    ftr.add_argument("--model-key", default=None, choices=["qwen38", "gemma4"], help="replay：熵差实验里的模型目录名")
    ftr.add_argument("--ids", nargs="*", default=None, help="replay：只做这些题（试跑用），如 CQ-003 CQ-009")
    ftr.add_argument("--label", default=None, help="analyze：报告里写的模型与数据集")
    ftr.add_argument("--out", default=None, help="analyze：报告 .md（默认 root/trigger_regions_report.md）")
    frb = sub.add_parser("fp-refbranch", help="ClockQA 试跑：无压力分支的看图注意力能不能当参照（四种分支对比，只回放读图阶段）")
    frb.add_argument("--root", required=True)
    frb.add_argument("--entropy-dir", required=True)
    frb.add_argument("--model-key", required=True, choices=["qwen38", "gemma4"])
    frb.add_argument("--ids", nargs="+", required=True)
    frb.add_argument("--image-dirs", nargs="*", default=None)
    frb.add_argument("--label", default="")
    frs = sub.add_parser("fp-restate", help="ClockQA 试跑：读图区域的偏移来自提示里用户那句话，还是思考开头的复述（2×2 回放）")
    frs.add_argument("--root", required=True)
    frs.add_argument("--entropy-dir", required=True)
    frs.add_argument("--model-key", required=True, choices=["qwen38", "gemma4"])
    frs.add_argument("--ids", nargs="+", required=True)
    frs.add_argument("--image-dirs", nargs="*", default=None)
    frs.add_argument("--label", default="")
    fec = sub.add_parser("fp-echo", help="回声隔离：熵差找出复述里抄来的用户答案，读图阶段把它的 Value 置零（ClockQA 20 题）")
    fec.add_argument("--phase", required=True, choices=["run", "summary"])
    fec.add_argument("--root", required=True)
    fec.add_argument("--entropy-dir", default=None)
    fec.add_argument("--model-key", default=None, choices=["qwen38", "gemma4"])
    fec.add_argument("--neutral-dir", default=None, help="run：模型自己改写的中性问题（fp-model2 neutralize 的 neutralized/ 目录）")
    fec.add_argument("--ids", nargs="*", default=None)
    fec.add_argument("--image-dirs", nargs="*", default=None)
    fec.add_argument("--gamma", type=float, default=0.0)
    fec.add_argument("--window", type=int, default=128)
    fec.add_argument("--maskq-root", default=None, help="summary：fp-maskq 的结果目录，一起对比")
    fec.add_argument("--label", default="")
    fsg = sub.add_parser("fp-seg", help="熵差片段隔离：整段思考里读不到靠用户那句话写出来的片段（只由熵差决定），ClockQA 20 题")
    fsg.add_argument("--phase", required=True, choices=["run", "summary"])
    fsg.add_argument("--root", required=True)
    fsg.add_argument("--entropy-dir", default=None)
    fsg.add_argument("--model-key", default=None, choices=["qwen38", "gemma4"])
    fsg.add_argument("--neutral-dir", default=None, help="run：模型自己改写的中性问题（fp-model2 neutralize 的 neutralized/ 目录）")
    fsg.add_argument("--ids", nargs="*", default=None)
    fsg.add_argument("--arms", nargs="*", default=None, choices=["base", "seg", "segc"],
                     help="run：默认只跑 seg（base 沿用回声隔离实验）；segc = 方案 C，熵差分支也读不到已隔离的片段")
    fsg.add_argument("--image-dirs", nargs="*", default=None)
    fsg.add_argument("--echo-root", default=None, help="summary：回声隔离的结果目录（取 base 与 echo 一起对比）")
    fsg.add_argument("--maskq-root", default=None, help="summary：fp-maskq 的结果目录")
    fsg.add_argument("--label", default="")
    fsa = sub.add_parser("fp-spanattn", help="某一段思考对前文的注意力：回放一条已生成的回答（隔离组按生成时的规则隔离），按来源分组")
    fsa.add_argument("--phase", required=True, choices=["run", "report"])
    fsa.add_argument("--root", required=True)
    fsa.add_argument("--record", default=None, help="run：回答记录（例如 results/seg_isolate_20261109/qwen38/records/CQ-006__belief__seg.json）")
    fsa.add_argument("--entropy-dir", default=None)
    fsa.add_argument("--model-key", default=None, choices=["qwen38", "gemma4"])
    fsa.add_argument("--start", default=None, help="run：这一段的开头文字")
    fsa.add_argument("--end", default=None, help="run：这一段的结尾文字")
    fsa.add_argument("--image-dirs", nargs="*", default=None)
    fsa.add_argument("--name", default="")
    fsa.add_argument("--json", default=None, help="report：run 生成的 json")
    fva = sub.add_parser("fp-valleyattn", help="熵谷之后的注意力归因：回放原回答，看后面的词对图像 / 用户那句话 / 熵差片段的注意力（不生成）")
    fva.add_argument("--phase", required=True, choices=["run", "report"])
    fva.add_argument("--root", required=True)
    fva.add_argument("--entropy-dir", default=None)
    fva.add_argument("--model-key", default=None, choices=["qwen38", "gemma4"])
    fva.add_argument("--cases", nargs="*", default=None, help="run：ID__条件，例如 CQ-003__belief")
    fva.add_argument("--image-dirs", nargs="*", default=None)
    fmq = sub.add_parser("fp-maskq", help="对照：生成思考前 128 个词时，文本侧看不见提示里的用户那句话（不碰图像注意力），ClockQA 20 题")
    fmq.add_argument("--phase", required=True, choices=["run", "summary"])
    fmq.add_argument("--root", required=True)
    fmq.add_argument("--entropy-dir", default=None)
    fmq.add_argument("--model-key", default=None, choices=["qwen38", "gemma4"])
    fmq.add_argument("--ids", nargs="*", default=None)
    fmq.add_argument("--image-dirs", nargs="*", default=None)
    fmq.add_argument("--label", default="")
    fmx = sub.add_parser("fp-mix", help="熵谷处的分布校正：报警后这一句里按两条分支的几何混合采样（base / mix，每题 2 个种子）")
    fmx.add_argument("--phase", required=True, choices=["run", "judge-export", "match-export", "summary"])
    fmx.add_argument("--root", required=True)
    fmx.add_argument("--entropy-dir", default=None)
    fmx.add_argument("--model-key", default=None, choices=["qwen38", "gemma4"])
    fmx.add_argument("--datasets", nargs="*", default=None)
    fmx.add_argument("--ids", nargs="*", default=None, help="run：只做这些题（试跑用）")
    fmx.add_argument("--image-dirs", nargs="*", default=None)
    fmx.add_argument("--seeds", nargs="*", type=int, default=None)
    fmx.add_argument("--beta", type=float, default=0.5)
    fmx.add_argument("--window", type=int, default=32)
    fmx.add_argument("--label", default="")
    fre = sub.add_parser("fp-refent", help="离线检验：熵差参照不带复述段时，读图段会不会出现熵差谷（只回放已有回答）")
    fre.add_argument("--phase", required=True, choices=["replay", "analyze"])
    fre.add_argument("--root", required=True)
    fre.add_argument("--entropy-dir", default=None)
    fre.add_argument("--model-key", default=None, choices=["qwen38", "gemma4"])
    fre.add_argument("--datasets", nargs="*", default=None)
    fre.add_argument("--ids", nargs="*", default=None)
    fre.add_argument("--image-dirs", nargs="*", default=None)
    fre.add_argument("--label", default="")
    frl = sub.add_parser("fp-rl", help="先看再听：读图阶段由看不到用户的话的熵差分支来写（看图量下降时切换），之后用熵差隔离；对照 base / ITSM，三个数据集各 20 题")
    frl.add_argument("--phase", required=True, choices=["neutralize", "run", "judge-export", "match-export", "summary"])
    frl.add_argument("--root", required=True)
    frl.add_argument("--entropy-dir", default=None)
    frl.add_argument("--model-key", default=None, choices=["qwen38", "gemma4"])
    frl.add_argument("--arms", nargs="*", default=None, choices=["base", "itsm", "preread", "rl", "rl0"],
                     help="run：默认 base itsm rl；preread、rl0 按需加")
    frl.add_argument("--datasets", nargs="*", default=None, choices=["ClockQA", "PathVQA", "PENDULUM"])
    frl.add_argument("--ids", nargs="*", default=None, help="run：只做这些题（试跑用）")
    frl.add_argument("--image-dirs", nargs="*", default=None)
    frl.add_argument("--label", default="")
    fvl = sub.add_parser("fp-valley", help="熵差谷注入视觉信息（LEAD 风格）：base / lead / img，三个数据集各 20 题")
    fvl.add_argument("--phase", required=True, choices=["run", "judge-export", "match-export", "summary"])
    fvl.add_argument("--root", required=True)
    fvl.add_argument("--entropy-dir", default=None)
    fvl.add_argument("--model-key", default=None, choices=["qwen38", "gemma4"])
    fvl.add_argument("--ids", nargs="*", default=None, help="run：只做这些题（试跑用）")
    fvl.add_argument("--image-dirs", nargs="*", default=None)
    fvl.add_argument("--lam", type=float, default=0.4)
    fvl.add_argument("--label", default="")
    fa1 = sub.add_parser("fp-anchor1", help="锚点一试跑：读图阶段把看图注意力往参照分支（去掉用户那句话和复述段）拉（ClockQA 几题）")
    fa1.add_argument("--root", required=True)
    fa1.add_argument("--entropy-dir", default=None)
    fa1.add_argument("--model-key", default=None, choices=["qwen38", "gemma4"])
    fa1.add_argument("--ids", nargs="+", default=None)
    fa1.add_argument("--image-dirs", nargs="*", default=None)
    fa1.add_argument("--label", default="")
    fa1.add_argument("--lam", type=float, default=0.5)
    fa1.add_argument("--window", type=int, default=128)
    fa1.add_argument("--report-only", action="store_true", help="只用已有记录重写报告")
    fa1.add_argument("--phase", default="run", choices=["run", "judge-export", "match-export", "summary"],
                     help="run：生成；之后 PathVQA、PENDULUM 的盲判读与汇总，与 fp-valley 相同")
    fa1.add_argument("--datasets", nargs="*", default=None, help="run 且不给 --ids 时：用 fp-valley 的选题做这些数据集")
    fm2 = sub.add_parser("fp-model2", help="第二个模型（Gemma 4 等）的 ClockQA baseline：原模型三种条件、MM-SY、ITSM、S2VA")
    fm2.add_argument("--phase", required=True, choices=["setup", "neutralize", "run", "yn-export", "yn-summary"])
    fm2.add_argument("--src-cq", default=None, help="setup：原 ClockQA 目录（读 items.jsonl 和图像）")
    fm2.add_argument("--cq", required=True, help="这个模型的 ClockQA 目录（setup 建；原模型种子 42 的记录写在 obs/records）")
    fm2.add_argument("--root", default=None, help="run：其他方法、其他种子的结果目录")
    fm2.add_argument("--arms", nargs="*", default=["base", "mmsy", "itsm"])
    fm2.add_argument("--seeds", nargs="*", type=int, default=[42])
    fm2.add_argument("--yn-out", default=None, help="yn-export / yn-summary：判读目录（默认 root/yesno_judge）")
    fm2.add_argument("--yn-arms", nargs="*", default=None, help="yn-export：只导出这些方法（如 s2va）")
    fm2.add_argument("--yn-extra", nargs="*", default=None, help="yn-summary：之前那批判读的 judgments.json，用于同题配对")
    for q in (fo, fi, fpu, fdu, fen, fag, fsp, fhd, fcf, fvf, fcm, fsr, fm2, ftr, frb, frs, fa1, fvl, fre, fmx, fmq, fec, fva, fsg, fsa, frl):
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
        elif a.cmd == "fp-compare":
            from symexp.fp import compare as fp_compare
            if a.phase == "time-export":
                fp_compare.time_export(a.cq, a.root)
            elif a.phase == "time-import":
                fp_compare.time_import(a.cq, a.root, a.judgments)
            elif a.phase == "summary":
                fp_compare.run_summary(a.cq, a.root, a.out or os.path.join(a.root, "compare_summary.json"))
            else:
                if not a.work:
                    raise SystemExit("--phase gate / run 需要 --work")
                c = Config.load(a.config, _overrides(a.set))
                if a.phase == "gate":
                    fp_compare.run_gate(c, a.cq, a.work, a.root, a.threshold, a.seed, a.arms)
                else:
                    fp_compare.run_arms(c, a.cq, a.work, a.root, a.arms, a.conds, a.threshold, a.shard, a.seed, a.seeds)
        elif a.cmd == "fp-echo":
            from symexp.fp import echo_isolate as fp_ec
            if a.phase == "run":
                if not (a.entropy_dir and a.model_key and a.neutral_dir):
                    raise SystemExit("fp-echo run 需要 --entropy-dir、--model-key、--neutral-dir")
                fp_ec.run(Config.load(a.config, _overrides(a.set)), a.root, a.entropy_dir, a.model_key, a.neutral_dir,
                          a.image_dirs, a.shard, seed=a.seed, ids=a.ids, gamma=a.gamma, window=a.window)
            else:
                fp_ec.summary(a.root, a.label, a.maskq_root)
        elif a.cmd == "fp-seg":
            from symexp.fp import seg_isolate as fp_sg
            if a.phase == "run":
                if not (a.entropy_dir and a.model_key and a.neutral_dir):
                    raise SystemExit("fp-seg run 需要 --entropy-dir、--model-key、--neutral-dir")
                fp_sg.run(Config.load(a.config, _overrides(a.set)), a.root, a.entropy_dir, a.model_key, a.neutral_dir,
                          a.image_dirs, a.shard, seed=a.seed, ids=a.ids, arms=a.arms)
            else:
                fp_sg.summary(a.root, a.label, a.echo_root, a.maskq_root)
        elif a.cmd == "fp-spanattn":
            from symexp.fp import span_attn as fp_sa
            if a.phase == "run":
                if not (a.record and a.entropy_dir and a.model_key and a.start and a.end):
                    raise SystemExit("fp-spanattn run 需要 --record、--entropy-dir、--model-key、--start、--end")
                fp_sa.run(Config.load(a.config, _overrides(a.set)), a.root, a.record, a.entropy_dir, a.model_key, a.start, a.end,
                          a.image_dirs, a.name)
            else:
                fp_sa.report(a.json)
        elif a.cmd == "fp-valleyattn":
            from symexp.fp import valley_attn as fp_va
            if a.phase == "run":
                if not (a.entropy_dir and a.model_key and a.cases):
                    raise SystemExit("fp-valleyattn run 需要 --entropy-dir、--model-key、--cases")
                fp_va.run(Config.load(a.config, _overrides(a.set)), a.root, a.entropy_dir, a.model_key, a.cases, a.image_dirs)
            else:
                fp_va.report(a.root)
        elif a.cmd == "fp-maskq":
            from symexp.fp import mask_query as fp_mq
            if a.phase == "run":
                if not (a.entropy_dir and a.model_key):
                    raise SystemExit("fp-maskq run 需要 --entropy-dir、--model-key")
                fp_mq.run(Config.load(a.config, _overrides(a.set)), a.root, a.entropy_dir, a.model_key, a.image_dirs, a.shard,
                          seed=a.seed, ids=a.ids)
            else:
                fp_mq.summary(a.root, a.label)
        elif a.cmd == "fp-mix":
            from symexp.fp import mix_decode as fp_mx
            from symexp.fp import valley_inject as fp_vl
            if a.phase == "run":
                if not (a.entropy_dir and a.model_key):
                    raise SystemExit("fp-mix run 需要 --entropy-dir、--model-key")
                fp_mx.run(Config.load(a.config, _overrides(a.set)), a.root, a.entropy_dir, a.model_key, a.image_dirs, a.shard,
                          seeds=a.seeds, beta=a.beta, window=a.window, datasets=a.datasets, only_ids=a.ids)
            elif a.phase == "judge-export":
                fp_vl.judge_export(a.root)
            elif a.phase == "match-export":
                fp_vl.match_export(a.root)
            else:
                fp_mx.summary(a.root, a.label)
        elif a.cmd == "fp-refent":
            from symexp.fp import ref_entropy as fp_re
            if a.phase == "replay":
                if not (a.entropy_dir and a.model_key):
                    raise SystemExit("fp-refent replay 需要 --entropy-dir、--model-key")
                fp_re.run(Config.load(a.config, _overrides(a.set)), a.root, a.entropy_dir, a.model_key, a.datasets, a.ids,
                          a.image_dirs, a.shard)
            else:
                fp_re.analyze(a.root, a.label)
        elif a.cmd == "fp-rl":
            from symexp.fp import read_listen as fp_rl
            from symexp.fp import valley_inject as fp_vl
            if a.phase in ("neutralize", "run") and not (a.entropy_dir and a.model_key):
                raise SystemExit("fp-rl neutralize / run 需要 --entropy-dir、--model-key")
            if a.phase == "neutralize":
                fp_rl.neutralize(Config.load(a.config, _overrides(a.set)), a.root, a.entropy_dir, a.model_key, a.shard)
            elif a.phase == "run":
                fp_rl.run(Config.load(a.config, _overrides(a.set)), a.root, a.entropy_dir, a.model_key, a.image_dirs, a.shard,
                          seed=a.seed, arms=a.arms, datasets=a.datasets, ids=a.ids)
            elif a.phase == "judge-export":
                fp_vl.judge_export(a.root)
            elif a.phase == "match-export":
                fp_vl.match_export(a.root)
            else:
                fp_rl.summary(a.root, a.label)
        elif a.cmd == "fp-valley":
            from symexp.fp import valley_inject as fp_vl
            if a.phase == "run":
                if not (a.entropy_dir and a.model_key):
                    raise SystemExit("fp-valley run 需要 --entropy-dir、--model-key")
                fp_vl.run(Config.load(a.config, _overrides(a.set)), a.root, a.entropy_dir, a.model_key, a.image_dirs,
                          a.shard, seed=a.seed, lam=a.lam, only_ids=a.ids)
            elif a.phase == "judge-export":
                fp_vl.judge_export(a.root)
            elif a.phase == "match-export":
                fp_vl.match_export(a.root)
            else:
                fp_vl.summary(a.root, a.label)
        elif a.cmd == "fp-anchor1":
            from symexp.fp import read_anchor as fp_a1
            from symexp.fp import valley_inject as fp_vl
            if a.report_only:
                fp_a1.report(a.root, a.label)
            elif a.phase == "judge-export":
                fp_vl.judge_export(a.root)
            elif a.phase == "match-export":
                fp_vl.match_export(a.root)
            elif a.phase == "summary":
                fp_vl.summary(a.root, a.label, title="锚点一：读图阶段把看图注意力往参照分支拉",
                              notes=["- 参照分支：提示去掉用户那句话；思考开头第一段提到 user 时（复述段）在参照里删掉，没提时什么都不删；",
                                     "- 开头一段之后的 128 个 token：看图总量 m' = max(m, m*)，图像内部分配向参照靠 λ = 0.5（固定位置不动）；",
                                     "- base：参照分支照跑、不改；pull：按上式改。两组同一随机种子。"])
            else:
                if not (a.entropy_dir and a.model_key and (a.ids or a.datasets)):
                    raise SystemExit("fp-anchor1 run 需要 --entropy-dir、--model-key，以及 --ids 或 --datasets")
                fp_a1.run(Config.load(a.config, _overrides(a.set)), a.root, a.entropy_dir, a.model_key, a.ids, a.image_dirs,
                          a.label, seed=a.seed, lam=a.lam, window=a.window, datasets=a.datasets, shard=a.shard)
        elif a.cmd == "fp-restate":
            from symexp.fp import restate_ablation as fp_ra
            fp_ra.run(Config.load(a.config, _overrides(a.set)), a.root, a.entropy_dir, a.model_key, a.ids, a.image_dirs, a.label)
        elif a.cmd == "fp-refbranch":
            from symexp.fp import ref_branch as fp_rb
            fp_rb.run(Config.load(a.config, _overrides(a.set)), a.root, a.entropy_dir, a.model_key, a.ids, a.image_dirs, a.label)
        elif a.cmd == "fp-trigger":
            from symexp.fp import trigger_regions as fp_tr
            if a.phase == "replay":
                if not (a.entropy_dir and a.model_key):
                    raise SystemExit("--phase replay 需要 --entropy-dir、--model-key")
                fp_tr.run_replay(Config.load(a.config, _overrides(a.set)), a.root, a.entropy_dir, a.model_key,
                                 a.datasets, a.ids, a.shard, a.image_dirs)
            else:
                if not a.label:
                    raise SystemExit("--phase analyze 需要 --label（模型与数据集）")
                fp_tr.run_analyze(a.root, a.label, a.out)
        elif a.cmd == "fp-sources":
            from symexp.fp import sources as fp_sources
            if a.phase == "replay":
                fp_sources.run_replay(Config.load(a.config, _overrides(a.set)), a.cq, a.root, a.conds, a.shard)
            elif a.phase == "prompt":
                fp_sources.run_prompt(Config.load(a.config, _overrides(a.set)), a.cq, a.root, a.conds, a.shard)
            elif a.phase == "prompt-analyze":
                fp_sources.run_prompt_analyze(a.cq, a.root, a.out)
            elif a.phase == "prompt-heads":
                fp_sources.run_prompt_heads(a.cq, a.root, a.out)
            elif a.phase == "reading":
                fp_sources.run_reading(Config.load(a.config, _overrides(a.set)), a.cq, a.root, a.conds, a.shard)
            elif a.phase == "cut":
                fp_sources.run_cut(Config.load(a.config, _overrides(a.set)), a.cq, a.root, a.conds, a.shard)
            elif a.phase == "reading-scores":
                from symexp.fp import reading_groups as fp_rg
                fp_rg.run_reading_scores(Config.load(a.config, _overrides(a.set)), a.cq, a.root, a.conds, a.shard)
            elif a.phase == "reading-bands":
                from symexp.fp import reading_groups as fp_rg
                if not a.judgments:
                    raise SystemExit("--phase reading-bands 需要 --judgments")
                fp_rg.run_bands(a.cq, a.root, a.judgments, a.out)
            elif a.phase == "reading-split":
                from symexp.fp import reading_groups as fp_rg
                if not a.judgments:
                    raise SystemExit("--phase reading-split 需要 --judgments")
                fp_rg.run_split(a.cq, a.root, a.judgments, a.out)
            elif a.phase == "sinks":
                from symexp.fp import sinks as fp_sinks
                fp_sinks.run_sinks(Config.load(a.config, _overrides(a.set)), a.cq, a.root, a.shard)
            elif a.phase == "sinks-analyze":
                from symexp.fp import sinks as fp_sinks
                if not a.judgments:
                    raise SystemExit("--phase sinks-analyze 需要 --judgments")
                fp_sinks.run_sinks_analyze(a.cq, a.root, a.judgments, a.out)
            elif a.phase == "reading-gen":
                from symexp.fp import reading_groups as fp_rg
                fp_rg.run_reading_gen(Config.load(a.config, _overrides(a.set)), a.cq, a.root, a.conds, a.shard)
            elif a.phase in ("stage-export", "stage-summary"):
                from symexp.fp import stages as fp_st
                if not a.label:
                    raise SystemExit(f"--phase {a.phase} 需要 --label（模型与数据集）")
                out = os.path.join(a.root, "stage_judge")
                if a.phase == "stage-export":
                    fp_st.run_export(a.cq, out, a.label)
                else:
                    if not a.judgments:
                        raise SystemExit("--phase stage-summary 需要 --judgments")
                    fp_st.run_summary(a.cq, out, a.judgments, a.label)
            elif a.phase == "attrib":
                from symexp.fp import attrib as fp_at
                fp_at.run_attrib(Config.load(a.config, _overrides(a.set)), a.cq, a.root, a.conds, a.shard)
            elif a.phase == "attrib-summary":
                from symexp.fp import attrib as fp_at
                if not (a.judgments and a.label):
                    raise SystemExit("--phase attrib-summary 需要 --judgments 和 --label")
                fp_at.run_attrib_summary(a.cq, a.root, a.judgments, a.label, a.out)
            elif a.phase == "reading-gen-all":
                from symexp.fp import reading_groups as fp_rg
                fp_rg.run_reading_gen_all(Config.load(a.config, _overrides(a.set)), a.cq, a.root, a.conds, a.shard)
            elif a.phase == "sink-check":
                from symexp.fp import reading_groups as fp_rg
                if not a.label:
                    raise SystemExit("--phase sink-check 需要 --label")
                sub = "reading_all" if os.path.isdir(os.path.join(a.root, "reading_all")) else "reading"
                fp_rg.run_sink_check(a.cq, a.root, a.label, sub, a.out)
            elif a.phase == "reading-layers-e":
                from symexp.fp import reading_groups as fp_rg
                if not (a.judgments and a.label):
                    raise SystemExit("--phase reading-layers-e 需要 --judgments（ClockQA 时间判读）和 --label")
                bel, neu = fp_rg.outcomes_clock(a.judgments)
                sub = "reading_all" if os.path.isdir(os.path.join(a.root, "reading_all")) else "reading"
                fp_rg.run_focus_layers_e(a.cq, a.root, bel, neu, a.label, sub, a.out)
            elif a.phase == "reading-layers":
                from symexp.fp import reading_groups as fp_rg
                if not a.label:
                    raise SystemExit("--phase reading-layers 需要 --label（模型与数据集）")
                sub = "reading_all" if os.path.isdir(os.path.join(a.root, "reading_all")) else "reading"
                if a.yn_judgments:
                    bel, neu = fp_rg.outcomes_yesno(a.yn_judgments)
                    fp_rg.run_focus_layers(a.cq, a.root, bel, neu, "pathvqa", a.label, sub, a.out)
                elif a.judgments:
                    bel, neu = fp_rg.outcomes_clock(a.judgments)
                    fp_rg.run_focus_layers(a.cq, a.root, bel, neu, "clockqa", a.label, sub, a.out)
                else:
                    raise SystemExit("--phase reading-layers 需要 --judgments（ClockQA）或 --yn-judgments（PathVQA）")
            elif a.phase == "reading-focus":
                from symexp.fp import reading_groups as fp_rg
                if not a.label:
                    raise SystemExit("--phase reading-focus 需要 --label（模型与数据集）")
                if a.yn_judgments:
                    bel, neu = fp_rg.outcomes_yesno(a.yn_judgments)
                    fp_rg.run_focus(a.cq, a.root, bel, neu, "pathvqa", a.label, a.out)
                elif a.judgments:
                    bel, neu = fp_rg.outcomes_clock(a.judgments)
                    fp_rg.run_focus(a.cq, a.root, bel, neu, "clockqa", a.label, a.out)
                else:
                    raise SystemExit("--phase reading-focus 需要 --judgments（ClockQA）或 --yn-judgments（PathVQA）")
            elif a.phase == "reading-groups":
                from symexp.fp import reading_groups as fp_rg
                if a.yn_judgments:
                    bel, neu = fp_rg.outcomes_yesno(a.yn_judgments)
                    fp_rg.run_groups(a.cq, a.root, bel, neu, fp_rg.PV_GROUPS, "pathvqa", a.out)
                elif a.judgments:
                    bel, neu = fp_rg.outcomes_clock(a.judgments)
                    fp_rg.run_groups(a.cq, a.root, bel, neu, fp_rg.CQ_GROUPS, "clockqa", a.out)
                else:
                    raise SystemExit("--phase reading-groups 需要 --judgments（ClockQA）或 --yn-judgments（PathVQA）")
            else:
                if not a.judgments:
                    raise SystemExit(f"--phase {a.phase} 需要 --judgments")
                if a.phase == "analyze":
                    fp_sources.run_analyze(a.cq, a.root, a.judgments, a.out)
                elif a.phase == "reading-analyze":
                    fp_sources.run_reading_analyze(a.cq, a.root, a.judgments, a.out)
                elif a.phase == "reading-claim":
                    fp_sources.run_reading_claim(a.cq, a.root, a.judgments, a.out)
                elif a.phase == "cut-analyze":
                    fp_sources.run_cut_analyze(a.cq, a.root, a.judgments, a.out)
                else:
                    fp_sources.run_tokens(a.cq, a.root, a.judgments, a.out)
        elif a.cmd == "fp-model2":
            from symexp.fp import model2 as fp_model2
            if a.phase == "setup":
                if not a.src_cq:
                    raise SystemExit("--phase setup 需要 --src-cq")
                fp_model2.setup(a.src_cq, a.cq)
            elif a.phase == "neutralize":
                fp_model2.run_neutralize(Config.load(a.config, _overrides(a.set)), a.cq, a.shard)
            elif a.phase in ("yn-export", "yn-summary"):
                from symexp.fp import yesno as fp_yesno
                if not a.root:
                    raise SystemExit(f"--phase {a.phase} 需要 --root")
                if a.phase == "yn-export":
                    fp_yesno.run_export(a.cq, a.root, a.yn_out, a.yn_arms)
                else:
                    fp_yesno.run_summary(a.cq, a.root, a.yn_out, a.yn_extra)
            else:
                if not a.root:
                    raise SystemExit("--phase run 需要 --root")
                fp_model2.run_arms(Config.load(a.config, _overrides(a.set)), a.cq, a.root, a.arms, a.seeds, a.shard)
        elif a.cmd == "fp-verify":
            from symexp.fp import verify as fp_verify
            fp_verify.run(Config.load(a.config, _overrides(a.set)), a.cq, a.work, a.cases, a.out, a.threshold, a.conds,
                          a.samples, a.shard, a.only)
        elif a.cmd == "fp-conflict-setup":
            from symexp.fp import conflict as fp_conflict
            fp_conflict.pv_setup(a.pv_src, a.pv)
            fp_conflict.cq_setup(a.native, a.cq, a.image_dirs)
        elif a.cmd == "fp-conflict":
            from symexp.fp import conflict as fp_conflict
            if a.phase == "pairs":
                fp_conflict.run_pairs(Config.load(a.config, _overrides(a.set)), a.pv_src, a.pv, a.cq, a.work, a.shard)
            elif a.phase == "directions":
                fp_conflict.run_directions(a.pv_src, a.pv, a.cq, a.work)
            elif a.phase == "external":
                if not (a.external and a.out and a.cq):
                    raise SystemExit("--phase external 需要 --external、--out 和 --cq")
                fp_conflict.run_external(Config.load(a.config, _overrides(a.set)), a.cq, a.work, a.external, a.out, a.shard)
            else:
                if not a.out:
                    raise SystemExit("--phase project 需要 --out")
                fp_conflict.run_project(Config.load(a.config, _overrides(a.set)), a.pv_src, a.pv, a.cq, a.work, a.out, a.shard)
        elif a.cmd == "fp-hidden":
            from symexp.fp import hidden as fp_hidden
            if a.phase == "forward":
                if not a.roots:
                    raise SystemExit("--phase forward 需要 --roots")
                fp_hidden.run_forward(Config.load(a.config, _overrides(a.set)), a.roots, a.work, a.subs.split(","), a.shard)
            else:
                if not a.out:
                    raise SystemExit("--phase analyze 需要 --out")
                fp_hidden.run_analyze(a.work, a.out)
        elif a.cmd == "fp-spans":
            from symexp.fp.spans import run as fp_spans
            fp_spans(Config.load(a.config, _overrides(a.set)), a.roots, a.out, a.subs.split(","), a.shard)
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
