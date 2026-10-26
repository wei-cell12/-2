"""五压力实验：干预规则（numpy）、参照构造、分组检验、mock 全流程。"""

import json
import os

import numpy as np

from symexp.config import Config
from symexp.fp import FLOOR, LAYERS
from symexp.fp.references import build, check
from symexp.fp.rules import XP, apply_rows, conservation, matched, restore
from symexp.fp.stats import cluster_signflip

np_ = XP("numpy")


def _rows(seed=0, H=6, n=40, K=120):
    g = np.random.default_rng(seed)
    p = g.random((H, K)) ** 3
    p /= p.sum(-1, keepdims=True)
    img = np.arange(10, 10 + n)
    mask = np.zeros((H, n), bool)
    for h in range(H):
        mask[h, g.choice(n, 8, replace=False)] = True
    ref = g.random((H, n)) ** 2
    ref /= ref.sum(-1, keepdims=True)
    return p, img, mask, ref


def test_restore_lambda0_is_identity():
    p, img, mask, ref = _rows()
    sub = p[:, img]
    new, info = restore(np_, sub, mask, ref, 0.0, FLOOR)
    assert new is sub and np.allclose(info["delta"], 0)


def test_restore_conserves_and_raises_region():
    p, img, mask, ref = _rows(1)
    heads = np.arange(p.shape[0])
    new_img, info = restore(np_, p[:, img], mask, ref, 1.0, FLOOR)
    row = apply_rows(np_, p, img, heads, new_img)
    c = conservation(np_, p, row, img, heads)
    assert max(c["image"], c["text"], c["row"]) < 1e-12 and c["min_prob"] >= 0
    assert (info["region_after"] >= info["region_before"] - 1e-12).all()
    assert (info["delta"] > 0).any()
    # 非关键区域至少保留原份额的 5%
    q0 = p[:, img] / p[:, img].sum(-1, keepdims=True)
    q1 = new_img / new_img.sum(-1, keepdims=True)
    keep = (q1 * ~mask).sum(-1) / (q0 * ~mask).sum(-1)
    assert (keep >= FLOOR - 1e-9).all()


def test_restore_reaches_reference_when_capacity_allows():
    q = np.full((1, 10), 0.1)
    mask = np.zeros((1, 10), bool)
    mask[0, :2] = True
    ref = np.zeros((1, 10))
    ref[0, :2] = 0.15
    ref[0, 2:] = 0.7 / 8
    new, info = restore(np_, q * 0.4, mask, ref, 1.0, FLOOR)   # 图像总量 0.4
    qn = new / new.sum()
    assert np.allclose(qn[0, :2], 0.15) and np.isclose(new.sum(), 0.4)
    assert np.isclose(info["delta"][0], 0.1)


def test_matched_has_same_dose():
    p, img, mask, ref = _rows(2)
    sub = p[:, img]
    _, info_e = restore(np_, sub, mask, ref, 1.0, FLOOR)
    g = np.random.default_rng(3)
    rmask = np.zeros_like(mask)
    for h in range(mask.shape[0]):
        rmask[h, g.choice(np.flatnonzero(~mask[h]), 8, replace=False)] = True
    w = g.random(mask.shape) * rmask
    new, info = matched(np_, sub, rmask, w, info_e["delta"], FLOOR)
    gain = info["region_after"] - info["region_before"]
    # 随机区域本身的份额增量 = 抬高量 − 随机区域原份额被按比例扣掉的部分（随机区域属于“目标”，不被扣）
    assert np.allclose(info["delta"], info_e["delta"]) and (gain > 0).all()
    assert np.allclose(new.sum(-1), sub.sum(-1))
    assert (new >= 0).all()


def test_matched_clamps_at_floor():
    sub = np.full((1, 10), 0.01)
    rmask = np.zeros((1, 10), bool)
    rmask[0, :2] = True
    new, info = matched(np_, sub, rmask, rmask.astype(float), np.array([5.0]), FLOOR)
    assert bool(info["clamped"][0]) and np.isclose(new.sum(), 0.1)
    q1 = new / new.sum()
    assert np.isclose(q1[0, 2:].sum(), 0.8 * FLOOR)


def test_reference_build():
    g = np.random.default_rng(0)
    think = g.random((16, 24, 50)).astype(np.float32) + 0.01
    answer = g.random((16, 24, 50)).astype(np.float32) + 0.01
    r1 = build("PV-000", think, answer)
    r2 = build("PV-000", think, answer)
    c = check(r1)
    assert c["ok"], c
    assert int(r1["k"]) == 10 and int(r1["heads"].sum()) == 96
    assert all(np.array_equal(r1[k], r2[k]) for k in r1)
    # 随机区域权重 = E 的参照值打乱，和相同
    assert np.allclose(r1["R_weight"].sum(-1), (r1["r"] * r1["E_mask"]).sum(-1), atol=1e-6)
    r3 = build("PV-001", think, answer)
    assert not np.array_equal(r1["R_mask"], r3["R_mask"])


def test_cluster_signflip_exact():
    t = cluster_signflip([("a", 1), ("b", 1), ("c", 1)])
    assert t["method"] == "exact" and abs(t["p"] - 0.25) < 1e-12
    t = cluster_signflip([("a", 1), ("a", 1), ("b", -1), ("c", 0)])
    assert t["nonzero_clusters"] == 2 and abs(t["p"] - 1.0) < 1e-12


def test_outcome():
    from symexp.fp.pipeline import outcome
    assert outcome({"semantic_answer": "yes", "explanation_correct": True}, "yes", "no") == "对"
    assert outcome({"semantic_answer": "yes", "explanation_correct": None}, "yes", "no") == "未解"
    assert outcome({"semantic_answer": "yes", "explanation_correct": False}, "yes", "no") == "其他错"
    assert outcome({"semantic_answer": "no", "explanation_correct": True}, "yes", "no") == "采纳"
    assert outcome({"semantic_answer": None}, "yes", "no") == "未解"
    assert outcome(None, "yes", None) == "未解"


def _fake_judge(root, phase):
    """mock 回复是 'Yes. (mock)' / 'No. (mock)'：照字面判，解释都算对。"""
    from symexp.store import read_jsonl, write_jsonl
    rows = read_jsonl(os.path.join(root, phase, "judge", "requests.jsonl"))
    out = [{"key": r["key"], "semantic_answer": r["reply"].split(".")[0].strip().lower(), "explanation_correct": True}
           for r in rows]
    p = os.path.join(root, phase, "judge", "fake_judgments.jsonl")
    write_jsonl(p, out)
    return p


def test_fp_mock_end_to_end(tmp_path):
    from symexp.fp import pipeline as fpp
    from symexp.fp.analyze import analyze
    root = str(tmp_path / "fp")
    cfg = Config.load(None, {"backend": "mock"})
    items = fpp.prepare(root, n=8, source="synthetic")
    assert len(items) == 8 and all(len(i["messages"]) == 6 for i in items)
    assert items[0]["messages"]["statement"].endswith("The answer is " + items[0]["claim"] + ".")
    fpp.observe(cfg, root, shard="0/2")
    fpp.observe(cfg, root, shard="1/2")
    assert len(fpp.load_records(root, "obs")) == 48
    n = fpp.judge_export(root, "obs")
    assert n == 48
    req = open(os.path.join(root, "obs", "judge", "requests.jsonl"), encoding="utf-8").read()
    assert "restore" not in req and "statement" not in req.split('"message"')[0]
    fpp.judge_import(root, "obs", _fake_judge(root, "obs"))
    cands = fpp.candidates(root)
    assert cands, "mock 应当产生候选"
    fpp.intervene(cfg, root, "pilot")
    gate = json.load(open(os.path.join(root, "pilot_gate.json"), encoding="utf-8"))
    assert gate["ok"] and gate["deterministic"]
    # 没有阅读确认时不能跑全部
    try:
        fpp.intervene(cfg, root, "full")
        raise AssertionError("应当拒绝")
    except SystemExit:
        pass
    import hashlib
    sha = hashlib.sha256(open(os.path.join(root, "pilot_gate.json"), "rb").read()).hexdigest()
    json.dump({"ok": True, "pilot_gate_sha256": sha, "read": []}, open(os.path.join(root, "pilot_review.json"), "w"))
    fpp.intervene(cfg, root, "full")
    for a in ("restore_E", "rand_matched"):
        assert len(fpp.load_records(root, f"int/{a}")) == len(cands)
    fpp.judge_export(root, "int")
    fpp.judge_import(root, "int", _fake_judge(root, "int"))
    rep = analyze(root, heatmaps=2)
    assert "## A." in rep and "## B." in rep and "restore_E" in rep and "非陈述" in rep
    s = json.load(open(os.path.join(root, "analysis", "summary.json"), encoding="utf-8"))
    assert s["B"]["n"] == len(cands) and "全部|答对" in s["B"]["paired"]
    assert os.path.exists(os.path.join(root, "analysis", "A_attention", "statement", "report.md"))
    pub = str(tmp_path / "pub")
    assert fpp.public(root, pub) == 48 + 2 * len(cands) + 2
    r = json.load(open(os.path.join(pub, "obs", "records", os.listdir(os.path.join(pub, "obs", "records"))[0])))
    assert "output_ids" not in r and "input_ids" not in r
    assert json.load(open(os.path.join(pub, "array_manifest.json")))["files"]


def test_fp_failure_blocks_rerun(tmp_path):
    from symexp.fp import pipeline as fpp
    root = str(tmp_path / "fp2")
    fpp.prepare(root, n=2, source="synthetic")
    os.makedirs(os.path.join(root, "failures"))
    open(os.path.join(root, "failures", "x.json"), "w").write("{}")
    try:
        fpp.observe(Config.load(None, {"backend": "mock"}), root)
        raise AssertionError("应当拒绝")
    except SystemExit:
        pass


def test_fp_subset_and_region(tmp_path):
    import hashlib
    from symexp.fp import pipeline as fpp
    from symexp.fp import subset
    src, dst = str(tmp_path / "src"), str(tmp_path / "sub")
    cfg = Config.load(None, {"backend": "mock"})
    items = fpp.prepare(src, n=4, source="synthetic")
    fpp.observe(cfg, src)
    ids = [i["id"] for i in items]
    cases = {"intervene": [[ids[0], "statement"], [ids[0], "belief"], [ids[1], "social"]],
             "groups": {"A": [[ids[0], "statement"], [ids[1], "social"]], "held": [[ids[2], "belief"]]}}
    cp = str(tmp_path / "cases.json")
    json.dump(cases, open(cp, "w"))
    assert subset.setup(src, dst, cp) == 3
    try:
        subset.setup(src, dst, cp)
        raise AssertionError("应当拒绝覆盖")
    except SystemExit:
        pass
    assert len(os.listdir(os.path.join(dst, "obs", "records"))) == 5     # 2 题中性 + 3 次受压
    fpp.intervene(cfg, dst, "pilot")
    gate = json.load(open(os.path.join(dst, "pilot_gate.json"), encoding="utf-8"))
    assert gate["ok"] and gate["cases"] == [f"{ids[0]}__statement", f"{ids[0]}__belief"]
    sha = hashlib.sha256(open(os.path.join(dst, "pilot_gate.json"), "rb").read()).hexdigest()
    json.dump({"ok": True, "pilot_gate_sha256": sha, "read": []}, open(os.path.join(dst, "pilot_review.json"), "w"))
    fpp.intervene(cfg, dst, "full")
    assert len(fpp.load_records(dst, "int/restore_E")) == 3
    assert fpp.judge_export(dst, "int") == 3 + 3 + 2
    res = subset.region(src, cp, str(tmp_path / "region.json"))
    assert set(res["summary"]) >= {"A|think", "A|answer", "held|think"} and len(res["rows"]) == 3
    r = res["rows"][0]["think"]
    assert 0 < r["E_share_neutral"] < 1 and abs(r["dE"] - (r["E_share_pressure"] - r["E_share_neutral"])) < 1e-12
    assert not os.path.exists(os.path.join(src, "refs"))                # region 不写主目录


def test_push_helpers():
    from symexp.fp.push import PushCheckFailed, claim_span, find_target, pair_ids
    neu = [1, 2, 3, 9, 9, 7, 8]
    pre = [1, 2, 3, 50, 51, 52, 9, 9, 7, 8]
    assert claim_span(pre, neu) == (3, 6)

    vocab = {0: "<close>", 1: "Yes", 2: " —", 3: " I", 4: " agree", 5: ": the answer is", 6: " **", 7: "no",
             8: "**", 9: ".", 10: " Yes", 11: " no", 12: "No", 13: " **No"}
    inv = {v: k for k, v in vocab.items()}

    class Tok:
        def decode(self, ids, **kw):
            return "".join(vocab[i] for i in ids if i != 0)

        def encode(self, s, add_special_tokens=False):
            return [inv[s]] if s in inv else [98, 99]

    tok = Tok()
    out = [5, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]       # 思考 5，</think>，"Yes — I agree: the answer is **no**."
    dec = tok.decode
    assert find_target(dec, out, 1, "") == 2                      # 第一个 yes/no 是 "Yes"
    assert find_target(dec, out, 1, "the answer is **") == 8     # 定位文字之后是 "no"
    yes_id, no_id, form = pair_ids(tok, 12)                       # "No" ↔ "Yes"
    assert (yes_id, no_id, form) == (1, 12, "No")
    out2 = [5, 0, 13, 9]                                          # 合并成一个 token 的 " **No"
    assert find_target(dec, out2, 1, " **") == 2
    try:
        pair_ids(tok, 7)                                          # "no" ↔ "yes" 不在词表 → 不是单个 token
        raise AssertionError("应当拒绝")
    except PushCheckFailed:
        pass
    try:
        find_target(dec, out, 1, "not in the text")
        raise AssertionError("应当拒绝")
    except PushCheckFailed:
        pass
