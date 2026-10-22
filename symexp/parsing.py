# -*- coding: utf-8 -*-
"""解析模型输出：第 2 步与 3c 的结构化区块、3a 的逐行判断、答案抽取。"""

import random
import re
import zlib
from typing import Dict, List, Optional, Tuple

BLOCK_NAMES = ["IDENTIFICATION", "CRITERIA", "KEY OBSERVATIONS", "CONCLUSION", "REJECTION REASONS"]
CELL_RE = re.compile(r"\b([ABC])\s*([123])\b")


def stable_rng(*parts) -> random.Random:
    return random.Random(zlib.crc32("|".join(str(p) for p in parts).encode("utf-8")))


def norm_answer(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    t = s.strip().lower().strip("*.\"' ")
    if t in ("yes", "no"):
        return t
    if t.startswith("undecid") or "does not allow" in t or "cannot be determined" in t:
        return "undecidable"
    m = re.match(r"(yes|no)\b", t)
    return m.group(1) if m else None


def split_blocks(text: str) -> Dict[str, List[str]]:
    """按 [NAME] 标题切块；同名块出现多次时取最后一次（模型有时会先在推理里写一遍草稿）。"""
    out: Dict[str, List[str]] = {}
    cur: Optional[str] = None
    buf: List[str] = []
    pat = re.compile(r"^\s*\**\s*\[(" + "|".join(re.escape(n) for n in BLOCK_NAMES) + r")\]\s*\**\s*$", re.I)
    for line in (text or "").splitlines():
        m = pat.match(line)
        if m:
            if cur is not None:
                out[cur] = buf
            cur, buf = m.group(1).upper(), []
        elif cur is not None:
            buf.append(line)
    if cur is not None:
        out[cur] = buf
    return out


def _bullets(lines: List[str]) -> List[str]:
    return [re.sub(r"^\s*[-*•]\s*", "", l).strip() for l in lines if re.match(r"^\s*[-*•]\s*\S", l)]


def _fields(line: str) -> Dict[str, str]:
    """把 "x | key: value | key: value" 拆成字典；第一段没有 key 时记为 "_head"。"""
    out: Dict[str, str] = {}
    for i, seg in enumerate(line.split(" | ")):
        m = re.match(r"\s*([A-Za-z ]+?)\s*:\s*(.*)$", seg)
        if m and m.group(1).strip().lower() in ("rejected", "type", "claim", "opposite", "where",
                                                   "confidence", "supports", "answer"):
            out[m.group(1).strip().lower()] = m.group(2).strip()
        elif i == 0:
            out["_head"] = seg.strip()
    return out


def parse_conclusion(lines: List[str]) -> Tuple[Optional[str], Optional[str]]:
    text = " ".join(lines)
    m = re.search(r"answer\s*:\s*\**\s*(yes|no|undecidable)\b", text, re.I)
    c = re.search(r"confidence\s*:\s*\**\s*(high|medium|low)\b", text, re.I)
    return (m.group(1).lower() if m else None), (c.group(1).lower() if c else None)


def parse_cells(where: Optional[str]) -> List[str]:
    if not where or "whole" in where.lower():
        return []
    cells = sorted({a + b for a, b in CELL_RE.findall(where.upper())})
    # 支持 "A1-B2" 这种区间写法
    rng = re.search(r"\b([ABC])([123])\s*[-–]\s*([ABC])([123])\b", where.upper())
    if rng:
        r0, r1 = sorted([ord(rng.group(1)), ord(rng.group(3))])
        c0, c1 = sorted([int(rng.group(2)), int(rng.group(4))])
        cells = sorted({chr(r) + str(c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)} | set(cells))
    return cells


def norm_type(t: Optional[str]) -> str:
    t = (t or "").lower()
    if "contradict" in t:
        return "contradicting"
    if "missing" in t or "expected" in t:
        return "missing"
    if "criter" in t or "definition" in t:
        return "criterion"
    return "contradicting"


def parse_step2(answer_text: str) -> Dict:
    b = split_blocks(answer_text)
    concl, conf = parse_conclusion(b.get("CONCLUSION", []))
    obs = []
    for l in _bullets(b.get("KEY OBSERVATIONS", [])):
        f = _fields(l)
        obs.append({"text": f.get("_head", l), "where": f.get("where", ""), "supports": norm_answer(f.get("supports"))})
    reasons = []
    for l in _bullets(b.get("REJECTION REASONS", [])):
        f = _fields(l)
        if not f.get("claim") or not f.get("opposite"):
            continue
        reasons.append({"rejected": norm_answer(f.get("rejected")), "type": norm_type(f.get("type")),
                        "claim": f["claim"], "opposite": f["opposite"], "where": f.get("where", ""),
                        "cells": parse_cells(f.get("where")), "confidence": (f.get("confidence") or "").lower()})
    return {
        "ok": concl is not None,
        "conclusion": concl, "confidence": conf,
        "identification": _bullets(b.get("IDENTIFICATION", [])),
        "criteria": _bullets(b.get("CRITERIA", [])),
        "observations": obs,
        "reasons": reasons,
        "blocks_found": sorted(b.keys()),
    }


def parse_step3c(answer_text: str) -> Dict:
    b = split_blocks(answer_text)
    concl, conf = parse_conclusion(b.get("CONCLUSION", []))
    obs = []
    for l in _bullets(b.get("KEY OBSERVATIONS", [])):
        f = _fields(l)
        obs.append({"text": f.get("_head", l), "where": f.get("where", ""), "supports": norm_answer(f.get("supports"))})
    return {"ok": concl is not None, "conclusion": concl, "confidence": conf, "observations": obs}


def parse_step3a(text: str, n_pairs: int) -> Dict[int, Dict[str, str]]:
    """返回 {pair 序号: {"choice": a|b|unsure, "seen": ...}}；缺的行记为 unsure。"""
    out: Dict[int, Dict[str, str]] = {}
    for line in (text or "").splitlines():
        m = re.match(r"\s*\**\s*P(\d+)\s*\**\s*[:：]\s*\(?\s*(a|b|cannot determine|cannot be determined|"
                     r"undetermined|unsure)\b\)?\s*(?:[-–—:|]\s*(.*))?$", line.strip(), re.I)
        if m:
            k = int(m.group(1))
            ch = m.group(2).lower()
            out[k] = {"choice": ch if ch in ("a", "b") else "unsure", "seen": (m.group(3) or "").strip()}
    for k in range(1, n_pairs + 1):
        out.setdefault(k, {"choice": "unsure", "seen": ""})
    return out


UNDECIDABLE_PAT = re.compile(r"(does not allow a (?:decision|definitive)|cannot be (?:determined|decided)|"
                             r"undecidable|not possible to (?:determine|tell)|cannot determine)", re.I)


def rule_answer(text: Optional[str], method: str = "") -> Optional[str]:
    """规则抽取最终答案（yes / no / undecidable / None）。正式结果以 Luna 判读为准，这只是后备。"""
    if not text:
        return None
    t = text
    m = re.search(r"<Summarization>(.*?)(?:</Summarization>|$)", t, re.S | re.I)
    if m:
        t = m.group(1)
        a = re.search(r"\b(yes|no)\b", t, re.I)
        if a:
            return a.group(1).lower()
        return "undecidable" if UNDECIDABLE_PAT.search(t) else None
    m = re.search(r"final (?:answer|conclusion)\s*(?:is)?\s*[:：]?\s*\**\s*(yes|no)\b", t, re.I)
    if m:
        return m.group(1).lower()
    head = t[:200]
    a = re.search(r"\b(yes|no)\b", head, re.I)
    if a:
        return a.group(1).lower()
    if UNDECIDABLE_PAT.search(t):
        return "undecidable"
    allm = re.findall(r"\b(yes|no)\b", t, re.I)
    return allm[-1].lower() if allm else None
