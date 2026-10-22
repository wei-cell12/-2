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

    # 判读：用规则抽取模拟 Luna
    req = str(tmp_path / "judge_requests.jsonl")
    n = export_requests(list(runs.values()), req)
    assert n == n_items + 2 * n_cond
    js = [{"key": r["key"], "semantic_answer": rule_answer(r["reply"]), "addresses_user": True}
          for r in read_jsonl(req)]
    jp = str(tmp_path / "judgments.jsonl")
    write_jsonl(jp, js)
    assert import_judgments(list(runs.values()), jp) == n

    report = evaluate(list(runs.values()), str(tmp_path / "report"), reps=200)
    assert "主要指标" in report and "本方法阶段诊断" in report and "成本" in report
    summary = json.load(open(tmp_path / "report" / "summary.json", encoding="utf-8"))
    assert "ignore_user" in summary["metrics"]
