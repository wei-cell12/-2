from PIL import Image

from symexp.crops import cells_rect, make_crop, region_label
from symexp.parsing import parse_cells, parse_step2, parse_step3a, rule_answer

STEP2 = """Some text first.
[IDENTIFICATION]
- tissue: liver | ruled out: kidney because no glomeruli; lung because no alveoli
[CRITERIA]
- necrosis means loss of nuclei with eosinophilic debris
[KEY OBSERVATIONS]
- pink anuclear debris | where: B2, B3 | supports: yes
[CONCLUSION]
answer: yes | confidence: high
[REJECTION REASONS]
For each answer you did not choose, give up to 3 reasons:
- rejected: no | type: contradicting observation | claim: there is anuclear debris in B2 | opposite: there is no anuclear debris in B2 | where: B2 | confidence: high
- rejected: no | type: criterion | claim: debris counts as necrosis | opposite: debris does not count as necrosis | where: whole image | confidence: low
"""


def test_parse_step2():
    p = parse_step2(STEP2)
    assert p["ok"] and p["conclusion"] == "yes" and p["confidence"] == "high"
    assert len(p["reasons"]) == 2
    assert p["reasons"][0]["type"] == "contradicting" and p["reasons"][0]["cells"] == ["B2"]
    assert p["reasons"][1]["type"] == "criterion" and p["reasons"][1]["cells"] == []
    assert p["observations"][0]["supports"] == "yes"


def test_parse_step2_missing():
    assert not parse_step2("no blocks here")["ok"]


def test_cells():
    assert parse_cells("A1, C3") == ["A1", "C3"]
    assert parse_cells("A1-B2") == ["A1", "A2", "B1", "B2"]
    assert parse_cells("whole image") == []


def test_step3a():
    r = parse_step3a("P1: a - I see debris\nP2: cannot determine - blurry\n", 3)
    assert r[1]["choice"] == "a" and r[2]["choice"] == "unsure" and r[3]["choice"] == "unsure"


def test_rule_answer():
    assert rule_answer("**No.** The tissue shows...") == "no"
    assert rule_answer("<Summarization> Yes </Summarization>") == "yes"
    assert rule_answer("Looking closely, the image does not allow a decision.") == "undecidable"


def test_crops():
    img = Image.new("RGB", (300, 300))
    assert cells_rect(["B2"], 300, 300, 0.5) == (50, 50, 250, 250)
    assert cells_rect(["A1"], 300, 300, 0.5) == (0, 0, 150, 150)
    c = make_crop(img, ["A1"], 600, 0.5)
    assert max(c.size) == 600
    assert region_label(["A1", "A2"]) == "row A, columns 1-2"
