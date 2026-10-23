"""用 mock 后端把整个流程跑一遍：准备数据 → 三种方法 → 判读导出导入 → 评测。"""

import json
import os
from collections import Counter

import pytest

from symexp.config import Config
from symexp.data import assign_ptypes, build_conditions, prepare_synthetic
from symexp.evaluate import evaluate
from symexp.judge import export_requests, import_judgments
from symexp.parsing import rule_answer
from symexp.prompts import PRESSURE_TEMPLATES
from symexp.runner import run
from symexp.store import read_jsonl, write_jsonl


def test_assign_ptypes_balanced():
    base = [{"id": f"X-{i:03d}", "answer": "yes" if i < 75 else "no"} for i in range(150)]
    a = assign_ptypes(base, seed=0)
    per = Counter(a.values())
    assert set(per) == set(PRESSURE_TEMPLATES) and all(v == 30 for v in per.values())
    for p in PRESSURE_TEMPLATES:
        n_yes = sum(1 for it in base if a[it["id"]] == p and it["answer"] == "yes")
        assert n_yes in (14, 15, 16)


@pytest.mark.parametrize("mode,per_item", [("assign", 3), ("all", 11)])
def test_end_to_end(tmp_path, mode, per_item):
    n_items = 10
    data = tmp_path / "data"
    prepare_synthetic(str(data), n=n_items)
    cond = str(data / "conditions.jsonl")
    build_conditions(str(data / "base.jsonl"), cond, ptype_mode=mode)
    rows = read_jsonl(cond)
    assert len(rows) == n_items * per_item
    if mode == "assign":  # 同一题的用户说错、说对用同一种压力方式
        for i in {r["id"] for r in rows}:
            assert len({r["ptype"] for r in rows if r["id"] == i and r["condition"] != "neutral"}) == 1
    n_cond = len(rows)

    cfg = Config.load(None, {"backend": "mock", "workers": 3, "crop_target_long_side": 256})
    runs = {m: str(tmp_path / "runs" / m) for m in ("plain", "srt", "ours")}
    run(cfg, "plain", cond, runs["plain"], [0], conditions=["neutral"])
    # 模拟复用的 plain 受压结果：另一个目录、同为 plain，评测时合并
    press = str(tmp_path / "runs" / "plain_pressure_reference")
    run(cfg, "plain", cond, press, [0], conditions=["user_wrong"])
    run(cfg, "srt", cond, runs["srt"], [0])
    run(cfg, "ours", cond, runs["ours"], [0])

    finals = {m: read_jsonl(os.path.join(d, "final.jsonl")) for m, d in runs.items()}
    assert len(finals["plain"]) == n_items
    assert len(finals["srt"]) == n_cond and len(finals["ours"]) == n_cond
    assert not any(r.get("error") for rs in finals.values() for r in rs)

    # 第 2 步每题只跑一次；第 3 步：结论为 yes/no 的题最多一次，undecidable 的题 yes、no 各一次
    s2 = read_jsonl(os.path.join(runs["ours"], "step2.jsonl"))
    s3 = read_jsonl(os.path.join(runs["ours"], "step3.jsonl"))
    assert len({r["key"] for r in s2}) == n_items
    concl = {r["id"]: r["parsed"]["conclusion"] for r in s2}
    per_claim = {}
    for r in s3:
        per_claim.setdefault(r["id"], set()).add(r["claim"])
    for i, claims in per_claim.items():
        assert len(claims) <= (2 if concl[i] == "undecidable" else 1)
        assert concl[i] not in claims
    calls = read_jsonl(os.path.join(runs["ours"], "calls.jsonl"))
    assert sum(c["step"] == "step4" for c in calls) == n_cond
    assert sum(c["step"] == "step2" for c in calls) >= n_items

    # 中性条件不触发第 3 步
    assert not any(r["triggered"] for r in finals["ours"] if r["condition"] == "neutral")

    # 续跑：不应再有新调用
    n_before = len(calls)
    run(cfg, "ours", cond, runs["ours"], [0])
    assert len(read_jsonl(os.path.join(runs["ours"], "calls.jsonl"))) == n_before

    # 本方法（无核查）：复用 ours 的第 2 步，只跑第 4 步
    direct = str(tmp_path / "runs" / "ours_direct")
    cfg_d = Config.load(None, {"backend": "mock", "workers": 3, "crop_target_long_side": 256,
                               "step2_source": runs["ours"]})
    run(cfg_d, "ours_direct", cond, direct, [0])
    fd = read_jsonl(os.path.join(direct, "final.jsonl"))
    cd = read_jsonl(os.path.join(direct, "calls.jsonl"))
    assert len(fd) == n_cond and not any(r.get("error") for r in fd)
    assert not any(r["triggered"] for r in fd)
    assert {c["step"] for c in cd} == {"step4"} and len(cd) == n_cond  # 第 2 步全部复用
    assert not read_jsonl(os.path.join(direct, "step3.jsonl"))

    # LQCD：mock 后端，记录注意力，并用注意力分析比较两条分支
    lq = str(tmp_path / "runs" / "lqcd")
    cfg_l = Config.load(None, {"backend": "mock", "workers": 2, "lqcd_record_attention": True})
    run(cfg_l, "lqcd", cond, lq, [0], conditions=["user_wrong"])
    fl = read_jsonl(os.path.join(lq, "final.jsonl"))
    assert len(fl) == n_cond // per_item * (per_item - 1) // 2 and not any(r.get("error") for r in fl)
    assert all(set(r["attn_files"]) == {"lead", "neutral"} for r in fl)
    from symexp.attn_analysis import analyze
    tag = "user_wrong__*__{}" if mode == "assign" else "user_wrong__statement__{}"
    rep = analyze(f"lqcd_neutral={lq}/attn@" + tag.format("neutral"), [f"lqcd_lead={lq}/attn@" + tag.format("lead")],
                  str(tmp_path / "attn_report"), reps=50)
    assert "lqcd_lead vs lqcd_neutral（10 题配对）" in rep

    # 判读：用规则抽取模拟 Luna
    req = str(tmp_path / "judge_requests.jsonl")
    dirs = list(runs.values()) + [press, direct]
    n = export_requests(dirs, req)
    assert n == n_items + 3 * n_cond + n_cond // per_item * (per_item - 1) // 2
    assert os.path.exists(str(tmp_path / "judge_requests_判读说明.md"))
    js = [{"key": r["key"], "semantic_answer": rule_answer(r["reply"]), "addresses_user": True,
           "explanation_correct": (k % 4 != 0) if k % 7 else None}
          for k, r in enumerate(read_jsonl(req))]
    jp = str(tmp_path / "judgments.jsonl")
    write_jsonl(jp, js)
    assert import_judgments(dirs, jp) == n

    report = evaluate(dirs, str(tmp_path / "report"), reps=200)
    assert "主要指标：综合" in report and "主要指标：仅语义答案" in report
    assert "阶段诊断：ours" in report and "阶段诊断：ours_direct" in report and "成本" in report
    summary = json.load(open(tmp_path / "report" / "summary.json", encoding="utf-8"))
    assert "ignore_user" in summary["metrics"]
    # 两个 plain 目录合并：plain 有谄媚率
    assert summary["metrics"]["plain"]["full"]["core"]["syc"][1] + \
        summary["metrics"]["plain"]["full"]["core"]["syc"][2] >= 0
    assert summary["metrics"]["plain"]["answer"]["core"]["clean_acc"][1] + \
        summary["metrics"]["plain"]["answer"]["core"]["clean_acc"][2] == n_items


def test_statuses_and_reference_sets():
    from symexp.evaluate import _statuses, core_metrics, reference_sets
    def row(i, cond, pred, ec, claim=None, gold="yes"):
        r = {"id": i, "condition": cond, "ptype": "statement", "seed": 42, "pred": pred, "gold": gold,
             "claim": claim, "explanation_correct": ec}
        _statuses(r)
        return r
    neu = [row("a", "neutral", "yes", True), row("b", "neutral", "yes", False), row("c", "neutral", "no", None),
           row("d", "neutral", None, None), row("e", "neutral", "yes", None)]
    assert [r["full_status"] for r in neu] == ["correct", "wrong", "wrong", "pending", "pending"]
    assert [r["answer_status"] for r in neu] == ["correct", "correct", "wrong", "pending", "correct"]
    runs = [{"method": "plain", "rows": neu}]
    C, W, P = reference_sets(runs, "full")
    assert (C, W, P) == ({"a"}, {"b", "c"}, {"d", "e"})
    C2, W2, P2 = reference_sets(runs, "answer")
    assert (C2, W2, P2) == ({"a", "b", "e"}, {"c"}, {"d"})
    uw = [row("a", "user_wrong", "no", True, claim="no"), row("b", "user_wrong", None, None, claim="no"),
          row("e", "user_wrong", "yes", True, claim="no")]
    m = core_metrics(uw, C2, W2, "answer")
    assert m["syc"] == (1, 2, 1)
