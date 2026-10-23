# -*- coding: utf-8 -*-
"""三种方法：plain、SRT 提示版、本方法（对称推理 + 否定理由核查 + 思考前缀接续）。

第 2、3 步的存档带 prompt_version；提示词版本变了就重新生成，不复用旧存档。"""

import json
import os
import re
import threading
import time
from typing import Dict, List, Optional, Tuple

from PIL import Image

from . import prompts as P
from .backends import Backend, GenRequest, GenResult
from .config import Config
from .crops import make_crop, region_label
from .parsing import (norm_answer, parse_cells, parse_step2, parse_step3a, parse_step3c, rule_answer,
                      stable_rng)
from .store import KeyedStore


class Context:
    def __init__(self, cfg: Config, backend: Backend, run_dir: str, data_dir: str, method: str):
        self.cfg = cfg
        self.backend = backend
        self.run_dir = run_dir
        self.data_dir = data_dir
        self.method = method
        os.makedirs(run_dir, exist_ok=True)
        self.calls_path = os.path.join(run_dir, "calls.jsonl")
        self.calls_lock = threading.Lock()
        self.final = KeyedStore(os.path.join(run_dir, "final.jsonl"))
        if method.startswith("ours"):
            self.step2 = KeyedStore(os.path.join(run_dir, "step2.jsonl"))
            # 从另一个本方法目录复用第 2 步（ours 与 ours_direct 的第 2 步完全相同）
            src = cfg.step2_source
            self.step2_src = KeyedStore(os.path.join(src, "step2.jsonl")) if src else None
            self.step3 = KeyedStore(os.path.join(run_dir, "step3.jsonl"))
            self.unit_locks: Dict[str, threading.Lock] = {}
            self.unit_locks_lock = threading.Lock()
        self._img_cache: Dict[str, Image.Image] = {}

    def image(self, rel: str) -> Image.Image:
        root = self.cfg.image_root or self.data_dir
        path = rel if os.path.isabs(rel) else os.path.join(root, rel)
        if path not in self._img_cache:
            self._img_cache[path] = Image.open(path).convert("RGB")
        return self._img_cache[path]

    def system(self) -> Optional[str]:
        return P.SYSTEM_PROMPT if self.cfg.use_system_prompt else None

    def call(self, step: str, unit: str, req: GenRequest) -> GenResult:
        res = self.backend.generate(req)
        row = {"method": self.method, "step": step, "unit": unit, "mode": req.mode, "seed": req.seed,
               "greedy": req.greedy, "n_images": len(req.images), "system": req.system, "user_text": req.user_text,
               "prefill_thinking": req.prefill_thinking, "prompt_text": res.prompt_text, "thinking": res.thinking,
               "answer": res.answer, "thinking_closed": res.thinking_closed, "prompt_tokens": res.prompt_tokens,
               "gen_tokens": res.gen_tokens, "seconds": round(res.seconds, 3), "time": time.time()}
        with self.calls_lock:
            with open(self.calls_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return res

    def lock_for(self, key: str) -> threading.Lock:
        with self.unit_locks_lock:
            return self.unit_locks.setdefault(key, threading.Lock())


def final_key(it: dict, seed: int) -> str:
    return f"{it['id']}|{it['condition']}|{it['ptype']}|{seed}"


def _base_row(ctx: Context, it: dict, seed: int) -> dict:
    return {"key": final_key(it, seed), "method": ctx.method, "id": it["id"], "condition": it["condition"],
            "ptype": it["ptype"], "seed": seed, "claim": it.get("claim"), "gold": it["answer"],
            "question": it["question"], "message": it["message"], "image": it["image"]}


def pressure_sentence(it: dict) -> str:
    m, q = it["message"], it["question"]
    return m[len(q):].strip() if m.startswith(q) else m


# =============================================================================================
# plain
# =============================================================================================
def run_plain(ctx: Context, it: dict, seed: int) -> dict:
    c = ctx.cfg
    req = GenRequest(user_text=it["message"], images=[ctx.image(it["image"])], system=ctx.system(), mode="think",
                     think_budget=c.think_budget, answer_budget=c.answer_budget, seed=seed)
    r = ctx.call("plain", final_key(it, seed), req)
    row = _base_row(ctx, it, seed)
    row.update(reply=r.answer, reply_thinking=r.thinking, thinking_closed=r.thinking_closed,
               rule_pred=rule_answer(r.answer, "plain"),
               query_prompt_tokens=r.prompt_tokens, query_gen_tokens=r.gen_tokens, query_seconds=r.seconds,
               n_calls=1)
    return row


# =============================================================================================
# SRT 提示版
# =============================================================================================
def run_srt(ctx: Context, it: dict, seed: int) -> dict:
    c = ctx.cfg
    text = P.SRT_TEMPLATE.format(question=it["question"])
    if it["condition"] != "neutral":
        text += "\n" + pressure_sentence(it)
    req = GenRequest(user_text=text, images=[ctx.image(it["image"])], system=ctx.system(), mode="think",
                     think_budget=c.think_budget, answer_budget=c.answer_budget, seed=seed)
    r = ctx.call("srt", final_key(it, seed), req)
    row = _base_row(ctx, it, seed)
    has3 = all(tag in r.answer for tag in ("<Image Textualization>", "<Reflection>", "<Summarization>"))
    row.update(reply=r.answer, reply_thinking=r.thinking, thinking_closed=r.thinking_closed, srt_format_ok=has3,
               rule_pred=rule_answer(r.answer, "srt"),
               query_prompt_tokens=r.prompt_tokens, query_gen_tokens=r.gen_tokens, query_seconds=r.seconds,
               n_calls=1)
    return row


# =============================================================================================
# 本方法
# =============================================================================================
def _answer_order(ctx: Context, item_id: str, seed: int, tag: str) -> List[str]:
    ans = ["yes", "no"]
    stable_rng(ctx.cfg.order_salt, item_id, seed, tag).shuffle(ans)
    return ans


def ours_step2(ctx: Context, it: dict, seed: int) -> dict:
    """每题每个 seed 只跑一次，该题所有条件共用。"""
    key = f"{it['id']}|{seed}"
    with ctx.lock_for("s2|" + key):
        got = ctx.step2.get(key)
        if got and not got.get("error") and got.get("prompt_version") == P.PROMPT_VERSION:
            return got
        if ctx.step2_src is not None:
            src = ctx.step2_src.get(key)
            if src and not src.get("error") and src.get("prompt_version") == P.PROMPT_VERSION:
                row = {**src, "reused_from": ctx.cfg.step2_source}
                ctx.step2.put(row)
                return row
        c = ctx.cfg
        order = _answer_order(ctx, it["id"], seed, "step2")
        text = P.STEP2_TEMPLATE.format(question=it["question"], answers=", ".join(order))
        attempts = []
        parsed, r = None, None
        ptok = gtok = 0
        secs = 0.0
        for k in range(max(1, c.step2_max_attempts)):
            s = seed if k == 0 else seed + 100003 * k
            req = GenRequest(user_text=text, images=[ctx.image(it["image"])], system=None, mode="think",
                             think_budget=c.think_budget, answer_budget=c.step2_answer_budget, seed=s)
            r = ctx.call("step2", key, req)
            ptok += r.prompt_tokens
            gtok += r.gen_tokens
            secs += r.seconds
            parsed = parse_step2(r.answer)
            attempts.append({"seed": s, "ok": parsed["ok"], "blocks_found": parsed["blocks_found"]})
            if parsed["ok"]:
                break
        row = {"key": key, "prompt_version": P.PROMPT_VERSION, "id": it["id"], "seed": seed,
               "answer_order": order, "thinking": r.thinking,
               "answer": r.answer, "thinking_closed": r.thinking_closed, "parsed": parsed, "attempts": attempts,
               "parse_ok": parsed["ok"], "prompt_tokens": ptok, "gen_tokens": gtok, "seconds": secs}
        ctx.step2.put(row)
        return row


def needs_step3(s2: dict, claim: Optional[str]) -> bool:
    if claim is None or not s2["parse_ok"]:
        return False
    return norm_answer(claim) != s2["parsed"]["conclusion"]


def _jaccard(a: str, b: str) -> float:
    x = set(re.findall(r"[a-z]+", a.lower()))
    y = set(re.findall(r"[a-z]+", b.lower()))
    return len(x & y) / max(1, len(x | y))


def ours_step3(ctx: Context, it: dict, seed: int, s2: dict, claim: str) -> dict:
    """每题每个 seed 每个主张只跑一次（--ptype-mode all 时 5 种压力共用）。"""
    claim = norm_answer(claim)
    key = f"{it['id']}|{seed}|{claim}"
    with ctx.lock_for("s3|" + key):
        got = ctx.step3.get(key)
        if got and not got.get("error") and got.get("prompt_version") == P.PROMPT_VERSION:
            return got
        c = ctx.cfg
        img = ctx.image(it["image"])
        p2 = s2["parsed"]
        rng = stable_rng(c.order_salt, it["id"], seed, claim, "3a")
        ptok = gtok = 0
        secs = 0.0
        n_calls = 0

        # ---- 准备裁切 ---------------------------------------------------------------------------
        crops: List[Tuple[Tuple[str, ...], Image.Image]] = []

        def crop_index(cells: List[str]) -> int:
            t = tuple(cells)
            for i, (k_, _) in enumerate(crops):
                if k_ == t:
                    return i + 2
            crops.append((t, make_crop(img, cells, c.crop_target_long_side, c.crop_expand_cells)))
            return len(crops) + 1

        all_reasons = [r for r in p2["reasons"] if r["rejected"] == claim]
        dropped = [{**r} for r in all_reasons if r.get("invalid")]
        reasons = [r for r in all_reasons if not r.get("invalid")][:c.max_reasons]
        skip_3a = p2["conclusion"] == "undecidable" or not reasons
        checks = []
        go_3c = skip_3a
        decision = "skip_3a" if skip_3a else None

        # ---- 3a 核查 -------------------------------------------------------------------------
        if not skip_3a:
            pair_lines = []
            for i, rs in enumerate(reasons, 1):
                swap = rng.random() < 0.5
                a, b = (rs["opposite"], rs["claim"]) if swap else (rs["claim"], rs["opposite"])
                if rs["type"] == "criterion":
                    tag = "[definition]"
                elif rs["cells"]:
                    tag = f"[see Image {crop_index(rs['cells'])}]"
                else:
                    tag = "[see Image 1]"
                pair_lines.append(f"P{i} {tag}: (a) {a} (b) {b}")
                checks.append({**rs, "pair": i, "a_is_claim": not swap, "tag": tag})
            crop_lines = ""
            if crops:
                crop_lines = P.CROP_HEADER + "\n" + "\n".join(
                    P.CROP_LINE.format(k=i + 2, region=region_label(list(k_))) for i, (k_, _) in enumerate(crops))
            text = P.STEP3A_TEMPLATE.format(question=it["question"], crop_lines=crop_lines,
                                            pairs="\n".join(pair_lines))
            req = GenRequest(user_text=text, images=[img] + [im for _, im in crops], system=None, mode="nothink",
                             answer_budget=c.step3a_answer_budget, greedy=True, seed=seed)
            r = ctx.call("step3a", key, req)
            n_calls += 1
            ptok += r.prompt_tokens
            gtok += r.gen_tokens
            secs += r.seconds
            res = parse_step3a(r.answer, len(checks))
            for ch in checks:
                o = res[ch["pair"]]
                if o["choice"] == "unsure":
                    ch["result"] = "unsure"
                else:
                    picked_claim = (o["choice"] == "a") == ch["a_is_claim"]
                    ch["result"] = "held" if picked_claim else "overturned"
                ch["seen"] = o["seen"]
                ch["echo"] = bool(o.get("echo"))
            raw_3a = r.answer
            # ---- 3b 判定 ---------------------------------------------------------------------
            n_over = sum(ch["result"] == "overturned" for ch in checks)
            n_held = sum(ch["result"] == "held" for ch in checks)
            if n_over == 0 and n_held > 0:
                decision = "kept"
            else:
                decision = "overturned" if n_over else "all_unsure"
                go_3c = True
        else:
            raw_3a = None

        # ---- 3c 重新判断 --------------------------------------------------------------------
        c3 = None
        final_concl, final_conf = p2["conclusion"], p2["confidence"]
        if go_3c:
            if not crops:  # 跳过 3a 时用关键观察的位置裁切
                for ob in p2["observations"]:
                    cells = parse_cells(ob.get("where"))
                    if cells and len(crops) < c.max_reasons:
                        crop_index(cells)
            crit = list(p2["criteria"])
            obs_lines = [f"- {o['text']} | where: {o.get('where') or 'whole image'}" for o in p2["observations"]]
            for ch in checks:
                where = ch.get("where") or "whole image"
                if ch["type"] == "criterion":
                    if ch["result"] == "overturned":
                        crit = [x for x in crit if _jaccard(x, ch["claim"]) < 0.6] + [f"(re-checked) {ch['opposite']}"]
                    continue
                if ch["result"] == "held":
                    obs_lines.append(f"- (re-checked) {ch['claim']} | where: {where}")
                elif ch["result"] == "overturned":
                    obs_lines.append(f"- (re-checked) {ch['opposite']} | where: {where}")
                else:
                    obs_lines.append(f"- (re-checked) uncertain whether: {ch['claim']} / {ch['opposite']} | where: {where}")
            order = _answer_order(ctx, it["id"], seed, "3c|" + claim)
            text = P.STEP3C_TEMPLATE.format(
                question=it["question"], answers=", ".join(order),
                criteria="\n".join(f"- {x}" for x in crit) or "- (none stated; use the standard meaning of the terms)",
                observations="\n".join(obs_lines) or "- (none recorded)",
                image_note=P.IMAGE_NOTE_WITH_CROPS if crops else P.IMAGE_NOTE_FULL_ONLY)
            req = GenRequest(user_text=text, images=[img] + [im for _, im in crops], system=None, mode="think",
                             think_budget=c.step3c_think_budget, answer_budget=c.step3c_answer_budget, seed=seed)
            r = ctx.call("step3c", key, req)
            n_calls += 1
            ptok += r.prompt_tokens
            gtok += r.gen_tokens
            secs += r.seconds
            p3 = parse_step3c(r.answer)
            c3 = {"thinking": r.thinking, "answer": r.answer, "parsed": p3, "answer_order": order,
                  "criteria_used": crit, "observations_used": obs_lines}
            if p3["ok"]:
                final_concl, final_conf = p3["conclusion"], p3["confidence"]
            else:
                decision = (decision or "") + "|3c_parse_failed"

        row = {"key": key, "prompt_version": P.PROMPT_VERSION, "id": it["id"], "seed": seed, "claim": claim,
               "step2_conclusion": p2["conclusion"],
               "decision": decision, "ran_3a": not skip_3a, "ran_3c": go_3c, "checks": checks, "raw_3a": raw_3a,
               "dropped_reasons": dropped,
               "step3c": c3, "crop_regions": [list(k_) for k_, _ in crops],
               "final_conclusion": final_concl, "final_confidence": final_conf,
               "n_calls": n_calls, "prompt_tokens": ptok, "gen_tokens": gtok, "seconds": secs}
        ctx.step3.put(row)
        return row


def build_prefix(s2: dict, s3: Optional[dict], reminder: bool = True) -> Tuple[str, Optional[str], Optional[str]]:
    parts = [s2["thinking"].strip(), s2["answer"].strip()]
    concl, conf = s2["parsed"]["conclusion"], s2["parsed"]["confidence"]
    if s3 is not None:
        if s3["ran_3a"] and s3["checks"]:
            lines = [P.CHECK_HEADER]
            for ch in s3["checks"]:
                seen = ch.get("seen") or "no further detail"
                if ch["result"] == "held":
                    lines.append(P.CHECK_HELD.format(claim=ch["claim"], seen=seen))
                elif ch["result"] == "overturned":
                    lines.append(P.CHECK_OVERTURNED.format(claim=ch["claim"], opposite=ch["opposite"], seen=seen))
                else:
                    lines.append(P.CHECK_UNSURE.format(claim=ch["claim"], seen=seen))
            parts.append("\n".join(lines))
        if s3["step3c"] is not None:
            parts += [s3["step3c"]["thinking"].strip(), s3["step3c"]["answer"].strip()]
        concl, conf = s3["final_conclusion"], s3["final_confidence"]
    if concl == "undecidable":
        parts.append(P.FINAL_LINE_UNDECIDABLE)
    elif concl in ("yes", "no"):
        parts.append(P.FINAL_LINE.format(answer=concl, confidence=conf or "medium"))
    if reminder:
        parts.append(P.FINAL_RESPONSE_REMINDER)
    return "\n\n".join(p for p in parts if p), concl, conf


def run_ours(ctx: Context, it: dict, seed: int, use_step3: bool = True) -> dict:
    c = ctx.cfg
    s2 = ours_step2(ctx, it, seed)
    s3 = ours_step3(ctx, it, seed, s2, it["claim"]) if (use_step3 and needs_step3(s2, it.get("claim"))) else None
    prefix, concl, conf = build_prefix(s2, s3, reminder=c.step4_reminder)
    req = GenRequest(user_text=it["message"], images=[ctx.image(it["image"])], system=ctx.system(), mode="prefill",
                     prefill_thinking=prefix, answer_budget=c.answer_budget, seed=seed)
    r = ctx.call("step4", final_key(it, seed), req)
    row = _base_row(ctx, it, seed)
    q_ptok = s2["prompt_tokens"] + r.prompt_tokens + (s3["prompt_tokens"] if s3 else 0)
    q_gtok = s2["gen_tokens"] + r.gen_tokens + (s3["gen_tokens"] if s3 else 0)
    q_sec = s2["seconds"] + r.seconds + (s3["seconds"] if s3 else 0)
    n_calls = len(s2["attempts"]) + 1 + (s3["n_calls"] if s3 else 0)  # 单条消息成本，含共用的第 2 步
    row.update(reply=r.answer, rule_pred=rule_answer(r.answer, "ours"), step2_reused=bool(s2.get("reused_from")),
               step2_key=s2["key"], step2_parse_ok=s2["parse_ok"], step2_conclusion=s2["parsed"]["conclusion"],
               step3_key=s3["key"] if s3 else None, step3_decision=s3["decision"] if s3 else None,
               triggered=s3 is not None, final_conclusion=concl, final_confidence=conf,
               prefix_chars=len(prefix), step4_prompt_tokens=r.prompt_tokens, step4_gen_tokens=r.gen_tokens,
               query_prompt_tokens=q_ptok, query_gen_tokens=q_gtok, query_seconds=q_sec, n_calls=n_calls)
    return row


def run_ours_direct(ctx: Context, it: dict, seed: int) -> dict:
    """本方法（无核查）：第 2 步对称推理之后直接第 4 步前缀接续，不跑第 3 步。
    第 3 步只在用户主张与结论冲突时触发，触发本身由用户决定，复查只会把结论推向用户；去掉后用户立场不再影响计算流程。"""
    return run_ours(ctx, it, seed, use_step3=False)


# =============================================================================================
# LQCD（对比解码 baseline）
# =============================================================================================
def run_lqcd(ctx: Context, it: dict, seed: int) -> dict:
    c = ctx.cfg
    row = _base_row(ctx, it, seed)
    img = ctx.image(it["image"])
    if it["condition"] == "neutral":
        # 两条分支输入相同，组合后就是普通解码，等同 plain 中性
        req = GenRequest(user_text=it["message"], images=[img], system=ctx.system(), mode="think",
                         think_budget=c.think_budget, answer_budget=c.answer_budget, seed=seed)
        r = ctx.call("lqcd_neutral_shortcut", final_key(it, seed), req)
        row.update(reply=r.answer, reply_thinking=r.thinking, thinking_closed=r.thinking_closed,
                   rule_pred=rule_answer(r.answer, "lqcd"), lqcd_shortcut=True,
                   query_prompt_tokens=r.prompt_tokens, query_gen_tokens=r.gen_tokens, query_seconds=r.seconds,
                   n_calls=1)
        return row
    if c.lqcd_neutral == "llm":
        nreq = GenRequest(user_text=P.NEUTRALIZE_TEMPLATE.format(message=it["message"]), images=[],
                          mode="nothink", answer_budget=256, greedy=True, seed=seed)
        neutral_q = ctx.call("lqcd_neutralize", final_key(it, seed), nreq).answer.strip() or it["question"]
    else:
        neutral_q = it["question"]
    req_lead = GenRequest(user_text=it["message"], images=[img], system=ctx.system(), mode="think",
                          think_budget=c.think_budget, answer_budget=c.answer_budget, seed=seed)
    req_neu = GenRequest(user_text=neutral_q, images=[img], system=ctx.system(), mode="think",
                         think_budget=c.think_budget, answer_budget=c.answer_budget, seed=seed)
    if not hasattr(ctx.backend, "generate_lqcd"):
        raise RuntimeError("LQCD 只支持 hf 和 mock 后端（需要逐步读取两条分支的 logits）")
    r, info = ctx.backend.generate_lqcd(req_lead, req_neu, alpha0=c.lqcd_alpha0, lam=c.lqcd_lambda,
                                        beta0=c.lqcd_beta0, mu=c.lqcd_mu, record=c.lqcd_record_attention)
    with ctx.calls_lock:
        with open(ctx.calls_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"method": "lqcd", "step": "lqcd", "unit": final_key(it, seed), "seed": seed,
                                "user_text": it["message"], "neutral_text": neutral_q, "prompt_text": r.prompt_text,
                                "thinking": r.thinking, "answer": r.answer, "thinking_closed": r.thinking_closed,
                                "prompt_tokens": r.prompt_tokens, "gen_tokens": r.gen_tokens,
                                "seconds": round(r.seconds, 3), "alpha_mean": info.get("alpha_mean"),
                                "jsd_mean": info.get("jsd_mean"), "time": time.time()}, ensure_ascii=False) + "\n")
    attn_files = {}
    if info.get("attn"):
        from .attn import save_npz
        for b, d in info["attn"].items():
            path = os.path.join(ctx.run_dir, "attn", f"{it['id']}__{it['condition']}__{it['ptype']}__{b}.npz")
            save_npz(path, d, image=it["image"])
            attn_files[b] = path
    row.update(reply=r.answer, reply_thinking=r.thinking, thinking_closed=r.thinking_closed,
               rule_pred=rule_answer(r.answer, "lqcd"), neutral_query=neutral_q,
               lqcd_alpha_mean=info.get("alpha_mean"), lqcd_jsd_mean=info.get("jsd_mean"),
               lqcd_forced_close=info.get("forced_close"), attn_files=attn_files,
               query_prompt_tokens=r.prompt_tokens, query_gen_tokens=r.gen_tokens, query_seconds=r.seconds,
               n_calls=1, forward_passes_per_token=2)
    return row


METHODS = {"plain": run_plain, "srt": run_srt, "ours": run_ours, "ours_direct": run_ours_direct, "lqcd": run_lqcd}
