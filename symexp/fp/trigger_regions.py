# -*- coding: utf-8 -*-
"""熵差报警前后，看图注意力落在哪里（ClockQA、PathVQA、PENDULUM）。

报警：熵差实验（results/shared_prefix_entropy_20261103）里每条回答逐 token 的 ΔH = H(有用户) − H(去掉用户)。
每段连续负值里第一次 ΔH ≤ −1 的 token 报警一次（只用当前和之前的值）。只取普通思考 token（不含预算强制写入的）。

三类窗口（每个窗口：报警 token t 之前 PRE 个、t 本身、之后 POST 个 token）：
- later：开头之后的报警（开头 = 熵差实验的 opening_cutoff，即前 64 个 token 与第一段中较长者）；
- opening：开头里的最后一次报警（模型抄写用户答案之后、开始读图之前）；
- control：同一条回答里熵差接近 0（|ΔH| ≤ 0.05）的普通位置，离任何报警至少 PRE + POST 个 token。

两类指标：
1. 图像内的注意力焦点（三个数据集都有）：报警前窗口里注意力最集中的 20% 图块记为"前焦点"；看报警时、报警后
   注意力还有多少落在前焦点上，以及前后两个窗口图像内分布的 JSD。
2. 区域（只有 ClockQA，按图上的钟面算，两种条件用同一套区域）：
- 真指针区：正确时间的分针（方向 ±15°、0.15–1.05 个半径）和时针（方向 ±15°、0.15–0.6 个半径）所在的图块，
  去掉与用户指针区重叠的部分；
- 用户指针区：如果用户说错时给的那个错误时间是对的，分针、时针应在的图块，同样去掉重叠部分。
用户说对那条回答里，用户说的是正确时间；"用户指针区"仍指那个错误时间的方向，作对照（用户这次没提它）。

回放：把已有回答原样喂回去（不生成），前向用原注意力函数的输出（只记录）。记录层：Qwen 系列全部标准注意力层；
Gemma 4 只用全局层（报警多在第 500 个 token 之后，滑动窗口层这时可能已看不到图像）。
"""

import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

THRESH = -1.0
ORD_ABS = 0.05
PRE, POST = 32, 32
N_CTRL = 3
TOP_FRAC = 0.25          # 每题按两种条件的平均看图量取前 25% 的头（Qwen 96/384，与之前一致）
REGIONS = ["true", "claim"]
FOCUS_FRAC = 0.20        # 前焦点：报警前窗口里注意力最集中的 20% 图块
DATASETS = ["ClockQA", "PathVQA", "PENDULUM"]
KINDS = ["later", "opening", "control"]
CONDS = ["belief", "belief_right"]
BOOT = 4000


class TriggerCheckFailed(RuntimeError):
    pass


# ---------------------------------------------------------------- 纯函数
def online_triggers(dH: List[float], ok: List[bool], thresh: float = THRESH) -> List[int]:
    """每段连续负值里第一次 ≤ thresh 的位置（输出行号，0 起）。ok[t] 为 False 的位置不报警，但不打断负值段。"""
    out, inseg, fired = [], False, False
    for t, v in enumerate(dH):
        if v < 0:
            if not inseg:
                inseg, fired = True, False
            if not fired and v <= thresh and ok[t]:
                out.append(t)
                fired = True
        else:
            inseg = False
    return out


MIN_PRE_OPENING = 4


def pick_events(dH: List[float], ok: List[bool], cutoff: int, n_think: int, key: str,
                pre: int = PRE, post: int = POST, n_ctrl: int = N_CTRL) -> List[Tuple[str, int, int]]:
    """[(类别, 位置, 前窗口长度)]：开头之后的全部报警、开头里的最后一次报警、至多 n_ctrl 个普通位置。
    窗口必须落在思考里。开头里的报警离输出开头很近，前窗口取能取到的部分（至少 MIN_PRE_OPENING 个 token）。"""
    trig = online_triggers(dH, ok)
    fits = lambda t: t - pre >= 0 and t + post < n_think
    ev = [("later", t, pre) for t in trig if t >= cutoff and fits(t)]
    op = [t for t in trig if t < cutoff and t >= MIN_PRE_OPENING and t + post < n_think]
    if op:
        ev.append(("opening", op[-1], min(pre, op[-1])))
    far = lambda t: all(abs(t - x) > pre + post for x in trig)
    cand = [t for t in range(cutoff + pre, n_think - post) if ok[t] and abs(dH[t]) <= ORD_ABS and far(t)]
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    chosen = []
    for t in rng.permutation(cand).tolist() if cand else []:
        if all(abs(t - c) > pre + post for c in chosen):
            chosen.append(int(t))
        if len(chosen) >= n_ctrl:
            break
    ev += [("control", t, pre) for t in sorted(chosen)]
    return ev


def window_rows(events: List[Tuple[str, int, int]], post: int = POST) -> List[int]:
    rows = set()
    for _, t, pre in events:
        rows.update(range(t - pre, t + post + 1))
    return sorted(rows)


HAND_SPLIT = 0.6          # 短针只画到 0.6 个半径（与 reading_groups.BAND_SPLIT 相同）


def hand_shape(geom, grid_hw, t: str) -> np.ndarray:
    """时间 t 下两根针所在的图块：分针方向 ±15°、0.15–1.05 个半径；时针方向 ±15°、0.15–0.6 个半径。
    只用时针、分针（不用秒针：时间写成 :00 时会多出一个指向 12 的秒针方向，许多钟面并没有秒针）。"""
    from .reading_groups import band_mask
    from .sources import hand_angles
    a = hand_angles(t)
    return band_mask(geom, grid_hw, a[1], 0.15, 1.05) | band_mask(geom, grid_hw, a[0], 0.15, HAND_SPLIT)


def region_masks(geom, grid_hw, gold: str, claim: str) -> Dict[str, np.ndarray]:
    """真指针区 = 正确时间两根针所在的图块；用户指针区 = 如果用户的时间是对的，两根针应在的图块；各自去掉重叠部分。
    短针只画到 0.6 个半径，所以"时针、分针对调"的错误时间（如正确 7:00、用户 12:35）两块区域仍然分得开。"""
    t = hand_shape(geom, grid_hw, gold)
    c = hand_shape(geom, grid_hw, claim)
    return {"true": t & ~c, "claim": c & ~t}


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(np.where(a > 0, a * np.log(np.maximum(a, 1e-300) / np.maximum(b, 1e-300)), 0.0)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def focus_values(dist: np.ndarray, row_index: Dict[int, int], t: int, pre: int = PRE, post: int = POST,
                 frac: float = FOCUS_FRAC) -> dict:
    """dist [N, n]：每行图像内部分布。前焦点 = 前窗口平均分布里最高的 frac 图块。
    focus：前 / 报警 / 后 三段平均分布落在前焦点上的比例；shift：前窗口与报警行、与后窗口的 JSD（nats）。"""
    norm = lambda x: x / max(float(x.sum()), 1e-12)
    avg = lambda rows: norm(dist[[row_index[r] for r in rows]].astype(np.float64).mean(0))
    b, a, f = avg(range(t - pre, t)), avg([t]), avg(range(t + 1, t + post + 1))
    k = max(1, int(round(len(b) * frac)))
    E = np.argsort(-b, kind="stable")[:k]
    return {"focus": {"before": float(b[E].sum()), "at": float(a[E].sum()), "after": float(f[E].sum())},
            "shift": {"before": 0.0, "at": _jsd(b, a), "after": _jsd(b, f)}}


def event_values(shares: np.ndarray, mass: np.ndarray, row_index: Dict[int, int], heads: np.ndarray, t: int,
                 pre: int = PRE, post: int = POST) -> dict:
    """shares [N, L, H, R]（图像内部归一化后各区域的份额）、mass [N, L, H]（看图总量）；heads [L, H] 布尔，选中的头。
    返回 前窗口 / 报警 token / 后窗口 的平均（对选中的头取平均）。"""
    def avg(rows):
        idx = [row_index[r] for r in rows]
        s = shares[idx][:, heads].mean(axis=(0, 1))                    # [R]
        m = mass[idx][:, heads].mean()
        return s, m
    b, mb = avg(range(t - pre, t))
    a, ma = avg([t])
    f, mf = avg(range(t + 1, t + post + 1))
    out = {"mass": {"before": float(mb), "at": float(ma), "after": float(mf)}}
    for k, name in enumerate(REGIONS):
        out[name] = {"before": float(b[k]), "at": float(a[k]), "after": float(f[k])}
    return out


def pick_heads(mass_by_cond: List[np.ndarray], frac: float = TOP_FRAC) -> np.ndarray:
    """mass_by_cond：每种条件一个 [N, L, H]。各条件先在行上平均，再在条件间平均；取前 frac 的头。"""
    m = np.mean([x.mean(0) for x in mass_by_cond], 0)                  # [L, H]
    k = max(1, int(round(m.size * frac)))
    thr = np.sort(m.ravel())[::-1][k - 1]
    sel = m >= thr
    if int(sel.sum()) != k:                                             # 并列时按顺序截断
        flat = np.argsort(-m.ravel(), kind="stable")[:k]
        sel = np.zeros(m.size, bool)
        sel[flat] = True
        sel = sel.reshape(m.shape)
    return sel


def boot_ci(vals: np.ndarray, reps: int = BOOT, seed: int = 0) -> Tuple[float, float]:
    if len(vals) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bs = vals[rng.integers(0, len(vals), (reps, len(vals)))].mean(1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


# ---------------------------------------------------------------- 记录器
class RegionRecorder:
    """只在指定的输出行上记录：每层每头图像内部各区域份额 [N, L, H, R]、看图总量 [N, L, H]，
    以及全部头按看图量加权的图像内部分布 [N, n]（各层平均；算焦点、JSD、画热图用）。"""

    def __init__(self, torch, rows: List[int], n_in: int, image_idx, masks: np.ndarray, layers: List[int],
                 chunk: int = 256):
        self.torch, self.chunk, self.layers = torch, chunk, list(layers)
        self.selfcheck_left, self.selfcheck_diffs = 0, []
        self.n_in, self.image_idx = n_in, image_idx
        self.rows = list(rows)
        self.index = {r: i for i, r in enumerate(self.rows)}
        self.rows_t = torch.as_tensor(self.rows, dtype=torch.long)
        self.masks = masks.astype(np.float32)                           # [R, n]
        self.shares, self.mass, self.filled = {}, {}, {}
        self.dist = np.zeros((len(self.rows), masks.shape[1]), np.float64)

    def stages_for(self, q_pos):
        return self.torch.full_like(q_pos, -1)

    def accumulate(self, layer_idx: int, probs, stages, q_pos):
        t = self.torch
        if layer_idx not in self.layers:
            return
        k = (q_pos - (self.n_in - 1)).cpu()
        hit = t.isin(k, self.rows_t)
        if not bool(hit.any()):
            return
        idx = hit.nonzero().flatten()
        p = probs[0][:, idx.to(probs.device), :][:, :, self.image_idx.to(probs.device)].float()   # [H, r, n]
        m = p.sum(-1)                                                                           # [H, r]
        d = p / m.clamp_min(1e-12).unsqueeze(-1)
        M = t.as_tensor(self.masks, device=p.device)
        s = t.einsum("hrn,gn->hrg", d, M)                                                       # [H, r, R]
        H = p.shape[0]
        if layer_idx not in self.shares:
            self.shares[layer_idx] = np.zeros((len(self.rows), H, M.shape[0]), np.float32)
            self.mass[layer_idx] = np.zeros((len(self.rows), H), np.float32)
            self.filled[layer_idx] = 0
        ps = p.sum(0)                                                                           # [r, n]：各头按看图量加权
        dw = ps / ps.sum(-1, keepdim=True).clamp_min(1e-12)
        sc, mc, dc = s.cpu().numpy(), m.cpu().numpy(), dw.cpu().numpy()
        for j, kk in enumerate(k[idx].tolist()):
            i = self.index[int(kk)]
            self.shares[layer_idx][i] = sc[:, j, :]
            self.mass[layer_idx][i] = mc[:, j]
            self.dist[i] += dc[j]
            self.filled[layer_idx] += 1

    def result(self) -> Dict[str, np.ndarray]:
        layers = sorted(self.shares)
        if layers != sorted(self.layers):
            raise TriggerCheckFailed(f"记录到的层 {layers} 与要记录的层 {sorted(self.layers)} 不一致")
        for l in layers:
            if self.filled[l] != len(self.rows):
                raise TriggerCheckFailed(f"第 {l} 层记录了 {self.filled[l]} 行，应为 {len(self.rows)}")
        return {"rows": np.array(self.rows), "shares": np.stack([self.shares[l] for l in layers], 1),
                "mass": np.stack([self.mass[l] for l in layers], 1),
                "dist": (self.dist / len(layers)).astype(np.float16), "layer_ids": np.array(layers)}


# ---------------------------------------------------------------- 读熵差实验的数据
def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_entropy(ent_dir: str, model_key: str, datasets: Optional[List[str]] = None) -> Dict[Tuple[str, str], dict]:
    """熵差实验的回答：{(题号, 条件): {dataset, group, cutoff, csv, json, job}}。job 是熵差实验 manifest 里这条回答的记录
    （含题目、图像路径、各条件的消息原文、原回答路径和 SHA）。"""
    import csv
    datasets = datasets or DATASETS
    d = os.path.join(ent_dir, model_key)
    man = json.load(open(os.path.join(d, "manifest.json"), encoding="utf-8"))
    jobs = {j["key"]: j for j in man["jobs"]}
    out = {}
    for r in csv.DictReader(open(os.path.join(d, "analysis", "metrics.csv"), encoding="utf-8")):
        if r["dataset"] not in datasets or r["scope"] != "after_opening" or r["condition"] not in CONDS:
            continue
        key = f"{r['id']}__{r['condition']}"
        if key not in jobs:
            raise TriggerCheckFailed(f"{d}/manifest.json 里没有 {key}")
        out[(r["id"], r["condition"])] = {"dataset": r["dataset"], "group": r["group"], "cutoff": int(r["opening_cutoff"]),
                                          "csv": os.path.join(d, "records", key + ".csv"),
                                          "json": os.path.join(d, "records", key + ".json"), "job": jobs[key]}
    if not out:
        raise TriggerCheckFailed(f"{d}/analysis/metrics.csv 里没有 {datasets} 的回答")
    return out


def read_steps(csv_path: str) -> Tuple[List[float], List[bool], List[str], int]:
    """(ΔH 逐行, 可报警, token 文字, 思考长度)。行号 r = prefix_output_tokens，即预测 output_ids[r] 的那一步。"""
    import csv
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    for i, r in enumerate(rows):
        if int(r["prefix_output_tokens"]) != i:
            raise TriggerCheckFailed(f"{csv_path} 第 {i} 行 prefix_output_tokens={r['prefix_output_tokens']}，行号不连续")
    dH = [float(r["delta_H"]) for r in rows]
    ok = [r["phase"] == "think" and r["forced_by_original_budget"] == "False" for r in rows]
    toks = [r["token_piece"] for r in rows]
    th = [i for i, r in enumerate(rows) if r["phase"] == "think"]
    n_think = th[-1] + 1 if th else 0            # Gemma 4 第 0 行是思考开头的特殊 token（phase = special）
    return dH, ok, toks, n_think


def resolve_source(job: dict) -> str:
    """原回答文件：先用熵差实验读的路径，没有就用它记下的原始路径；SHA 必须与熵差实验记录的一致。"""
    for p in (job.get("source"), job.get("source_original_path")):
        if p and os.path.exists(p):
            if _sha(p) != job["source_sha256"]:
                raise TriggerCheckFailed(f"{p} 与熵差实验用的原回答 SHA 不一致")
            return p
    raise TriggerCheckFailed(f"{job['key']}：找不到原回答（试过 {job.get('source')}、{job.get('source_original_path')}）")


_IMG_INDEX: Dict[str, str] = {}


def resolve_image(job: dict, rec: dict, image_dirs: Optional[List[str]] = None) -> str:
    """图像：manifest 里的路径 → 原回答里记的路径 → 在 image_dirs 里按 SHA 找；SHA 必须与熵差实验记录的一致。"""
    want = job["image_sha256"]
    for p in (job["item"].get("image"), rec.get("image")):
        if p and os.path.exists(p) and _sha(p) == want:
            return p
    for d in image_dirs or []:
        if want in _IMG_INDEX:
            break
        for dp, _, fs in os.walk(d):
            for f in fs:
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                    q = os.path.join(dp, f)
                    _IMG_INDEX.setdefault(_sha(q), q)
    if want in _IMG_INDEX:
        return _IMG_INDEX[want]
    raise TriggerCheckFailed(f"{job['key']}：找不到 SHA 为 {want[:12]}… 的图像（可用 --image-dirs 指定图像目录）")


# ---------------------------------------------------------------- GPU：回放
def replay_one(backend, cfg, image: str, message: str, rec: dict, n_in_expect: Optional[int], rows: List[int],
               masks: np.ndarray, layers, only, name: str) -> Tuple[dict, Dict[str, np.ndarray], np.ndarray]:
    from .. import attn
    from .entropy import _batch
    from .generate import build_inputs
    from .reading_groups import image_grid
    torch, model = backend.torch, backend.model
    got = attn.install(model, verbose=False, only=only)
    if sorted(got) != sorted(layers):
        raise TriggerCheckFailed(f"装上的记录层 {got} 与要记录的层 {layers} 不一致")
    _, inp = build_inputs(backend, image, message)
    in_ids = inp["input_ids"][0].tolist()
    if rec.get("input_ids") is not None and in_ids != [int(x) for x in rec["input_ids"]]:
        raise TriggerCheckFailed(f"{name}：重新构造的提示与原回答里的 input_ids 不一致")
    if n_in_expect is not None and len(in_ids) != int(n_in_expect):
        raise TriggerCheckFailed(f"{name}：提示长度 {len(in_ids)}，熵差实验记录的是 {n_in_expect}")
    grid = image_grid(backend, inp)
    out_ids = [int(x) for x in rec["output_ids"]][: rows[-1] + 1]      # 因果注意力：后面的 token 不影响前面
    img_tok = attn.image_token_id(backend)
    image_idx = torch.tensor([i for i, x in enumerate(in_ids) if x == img_tok])
    if int(grid.prod()) != int(image_idx.numel()):
        raise TriggerCheckFailed(f"{name}：图像 token 排列 {grid.tolist()} 与图像 token 数 {int(image_idx.numel())} 不一致")
    if masks is None:
        masks = np.zeros((len(REGIONS), int(image_idx.numel())), bool)
    r = RegionRecorder(torch, rows, len(in_ids), image_idx, masks, layers, chunk=cfg.attn_chunk)
    r.passthrough, r.rel_checked = True, {}
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "rope_deltas"):
        inner.rope_deltas = None
    t0 = time.time()
    with torch.inference_mode(), attn.recording(r):
        try:
            model(**_batch(backend, inp, out_ids), use_cache=False, return_dict=True, logits_to_keep=1)
        except TypeError:
            model(**_batch(backend, inp, out_ids), use_cache=False, return_dict=True)
    arrays = r.result()
    arrays["masks"] = masks
    return {"seconds": round(time.time() - t0, 2), "output_check": r.rel_checked, "n_in": len(in_ids),
            "grid_hw": grid.tolist()}, arrays, grid


def run_replay(cfg, root: str, ent_dir: str, model_key: str, datasets: Optional[List[str]] = None,
               ids: Optional[List[str]] = None, shard: str = "", image_dirs: Optional[List[str]] = None) -> int:
    from .. import attn
    from .conflict import _engine
    from .generate import build_inputs
    from .pipeline import _atomic_json, _atomic_npz, _check_no_failure, _fail, _shard
    from .reading_groups import image_grid, record_layers
    from .sources import dial_geometry
    _check_no_failure(root)
    ent = load_entropy(ent_dir, model_key, datasets)
    keys = sorted(k for k in ent if (not ids or k[0] in ids))
    if ids:
        miss = [i for i in ids if not any(k[0] == i for k in keys)]
        if miss:
            raise TriggerCheckFailed(f"熵差实验里没有这些题：{miss}")
    d = os.path.join(root, "trigger_regions")
    os.makedirs(d, exist_ok=True)
    tasks = _shard(keys, shard)
    todo = [k for k in tasks if not os.path.exists(os.path.join(d, f"{k[0]}__{k[1]}.json"))]
    print(f"[fp-trigger replay] {model_key}：本进程 {len(tasks)} 条，待回放 {len(todo)} 条 → {d}", file=sys.stderr)
    if not todo:
        return 0
    eng = _engine(cfg)
    b = eng.backend
    keep_layers, only = record_layers(b.model)
    layers = keep_layers if keep_layers is not None else sorted(int(m.layer_idx) for _, m in attn.find_targets(b.model))
    print(f"  记录的层：{layers}", file=sys.stderr)
    for n, (iid, c) in enumerate(todo):
        name = f"{iid}__{c}"
        try:
            e = ent[(iid, c)]
            job, item = e["job"], e["job"]["item"]
            rec = json.load(open(resolve_source(job), encoding="utf-8"))
            image = resolve_image(job, rec, image_dirs)
            dH, ok, toks, n_think = read_steps(e["csv"])
            out_ids = [int(x) for x in rec["output_ids"]]
            if len(dH) != len(out_ids):
                raise TriggerCheckFailed(f"{name}：熵差 {len(dH)} 行，原回答 {len(out_ids)} 个 token")
            events = pick_events(dH, ok, e["cutoff"], n_think, name)
            meta = {"id": iid, "condition": c, "dataset": e["dataset"], "group": e["group"], "cutoff": e["cutoff"],
                    "n_think": n_think, "image": image,
                    "events": [{"kind": k, "t": t, "pre": p, "dH": dH[t], "token": toks[t]} for k, t, p in events],
                    "gold": item.get("gold"), "claim_wrong": item.get("claim")}
            if not events:
                meta.update(rows=0, note="没有完整窗口的报警或普通位置")
                _atomic_json(os.path.join(d, name + ".json"), meta)
                print(f"  {n + 1}/{len(todo)} {name}：没有可用窗口", file=sys.stderr)
                continue
            for k, t, _ in events:                                # 对齐核对：熵差的 token 文字 = 原回答的 token
                piece = b.tokenizer.decode([out_ids[t]], skip_special_tokens=False)
                if piece != toks[t] and "�" not in piece + toks[t]:      # 半个 UTF-8 字符单独解码不出，跳过
                    raise TriggerCheckFailed(f"{name} 第 {t} 行：熵差记录的 token {toks[t]!r} 与原回答 {piece!r} 不一致")
            masks = None
            if e["dataset"] == "ClockQA":
                _, inp = build_inputs(b, image, item["messages"][c])
                rm = region_masks(dial_geometry(image), image_grid(b, inp), item["gold"], item["claim"])
                masks = np.stack([rm[r] for r in REGIONS])
            rows = window_rows(events)
            m, arrays, grid = replay_one(b, cfg, image, item["messages"][c], rec, job.get("n_in"), rows, masks, layers,
                                         only, name)
            meta.update(m, rows=len(rows), layer_ids=layers,
                        region_blocks={r: int(arrays["masks"][i].sum()) for i, r in enumerate(REGIONS)})
            _atomic_npz(os.path.join(d, name + ".npz"), arrays)
            _atomic_json(os.path.join(d, name + ".json"), meta)
            kinds = {k: sum(1 for x, _, _ in events if x == k) for k in KINDS}
            chk = "；".join(f"{k} 层相对差 {v['rel']:.1e}" for k, v in m["output_check"].items())
            reg = (f"；真指针区 {meta['region_blocks']['true']} 块、用户指针区 {meta['region_blocks']['claim']} 块"
                   if e["dataset"] == "ClockQA" else "")
            print(f"  {n + 1}/{len(todo)} {e['dataset']} {name}（{e['group']}）报警窗口 {kinds['later']} 个、开头 {kinds['opening']} 个、"
                  f"普通 {kinds['control']} 个，记录 {len(rows)} 行，{m['seconds']}s{reg}；{chk}", file=sys.stderr)
        except Exception as ex:
            _fail(root, f"trigger_{name}", ex)
            raise
    return len(todo)


# ---------------------------------------------------------------- CPU：汇总
GROUP_NAME = {"flip": "被带偏", "hold": "守住"}
COND_NAME = {"belief": "用户说错", "belief_right": "用户说对"}
KIND_NAME = {"later": "开头之后的报警", "opening": "开头里最后一次报警（抄写用户答案之后）", "control": "普通位置（|ΔH| ≤ 0.05）"}
FIELD_NAME = {"focus": "落在前焦点上的比例", "shift": "图像内分布与前窗口的 JSD（nats）", "true": "真指针区份额",
              "claim": "用户指针区份额", "mass": "看图总量"}


def collect(root: str) -> List[dict]:
    d = os.path.join(root, "trigger_regions")
    metas = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".json"):
            m = json.load(open(os.path.join(d, f), encoding="utf-8"))
            metas[(m["id"], m["condition"])] = m
    out = []
    for iid in sorted({k[0] for k in metas}):
        have = [c for c in CONDS if (iid, c) in metas and metas[(iid, c)].get("rows")]
        if not have:
            continue
        Z = {}
        for c in have:
            with np.load(os.path.join(d, f"{iid}__{c}.npz")) as z:
                Z[c] = {k: z[k] for k in ("rows", "shares", "mass", "dist")}
        heads = pick_heads([Z[c]["mass"] for c in have])
        for c in have:
            m = metas[(iid, c)]
            clock = m.get("dataset", "ClockQA") == "ClockQA" and all(m["region_blocks"][r] > 0 for r in REGIONS)
            idx = {int(r): i for i, r in enumerate(Z[c]["rows"].tolist())}
            for ev in m["events"]:
                v = event_values(Z[c]["shares"], Z[c]["mass"].astype(np.float64), idx, heads, ev["t"], pre=ev["pre"])
                if not clock:
                    v = {"mass": v["mass"]}
                v.update(focus_values(Z[c]["dist"], idx, ev["t"], pre=ev["pre"]))
                out.append({"id": iid, "condition": c, "dataset": m.get("dataset", "ClockQA"), "group": m["group"],
                            "kind": ev["kind"], "t": ev["t"], "dH": ev["dH"], "token": ev["token"], **v})
    return out


def summarize(events: List[dict]) -> dict:
    """每个（数据集, 条件, 组, 类别, 指标）：按题平均（同一题多个窗口先平均）后的 前 / 报警 / 后 和 后 − 前，加按题重抽样区间。"""
    res = {}
    for ds in DATASETS:
        for c in CONDS:
            for g in ("flip", "hold"):
                for k in KINDS:
                    ev = [e for e in events if e.get("dataset", "ClockQA") == ds and e["condition"] == c
                          and e["group"] == g and e["kind"] == k]
                    if not ev:
                        continue
                    items = sorted({e["id"] for e in ev})
                    cell = {"events": len(ev), "items": len(items)}
                    for f in ["focus", "shift", "true", "claim", "mass"]:
                        if any(f not in e for e in ev):
                            continue
                        per = {s: np.array([np.mean([e[f][s] for e in ev if e["id"] == i]) for i in items])
                               for s in ("before", "at", "after")}
                        diff = per["after"] - per["before"]
                        cell[f] = {s: float(per[s].mean()) for s in per}
                        cell[f]["after_minus_before"] = float(diff.mean())
                        cell[f]["ci"] = boot_ci(diff)
                    res[f"{ds}|{c}|{g}|{k}"] = cell
    return res


def run_analyze(root: str, label: str, out_md: Optional[str] = None) -> dict:
    from .pipeline import _atomic_json
    events = collect(root)
    if not events:
        raise SystemExit("没有可汇总的窗口（先跑 --phase replay）")
    res = summarize(events)
    _atomic_json(os.path.join(root, "trigger_regions_events.json"), events)
    _atomic_json(os.path.join(root, "trigger_regions_summary.json"), res)
    pp = lambda x: f"{100 * x:.1f}%"
    dd = lambda x: f"{100 * x:+.2f}"
    L = [f"# 熵差报警前后，看图注意力落在哪里：{label}", "",
         "## 数据是什么", "",
         "- **回答**：熵差实验（results/shared_prefix_entropy_20261103）里这个模型的已有回答，用户说错、用户说对两种条件；"
         "原样喂回去记录注意力，没有生成新回答。",
         "- **分组**（只看中性答对的题，按用户说错那条回答的结果分；用户说对那条回答沿用同一组名）：",
         "  - 被带偏：用户说错后答错；",
         "  - 守住：用户说错后仍答对。",
         "- **报警**：每段连续负熵差里第一次 ΔH ≤ −1 的 token（ΔH = H(有用户) − H(去掉用户)）。",
         f"- **窗口**：报警 token 之前 {PRE} 个 token 为\"前\"（开头里的报警离输出开头近，取能取到的部分，至少 "
         f"{MIN_PRE_OPENING} 个），报警 token 本身为\"报警\"，之后 {POST} 个 token 为\"后\"。",
         "- **三类窗口**：",
         f"  - {KIND_NAME['later']}；",
         f"  - {KIND_NAME['opening']}；",
         f"  - {KIND_NAME['control']}：同一条回答里离任何报警至少 {PRE + POST} 个 token，每条回答至多 {N_CTRL} 个，作对照。",
         "- **指标**：",
         f"  - 落在前焦点上的比例：\"前\"窗口里注意力最集中的 {int(FOCUS_FRAC * 100)}% 图块记为前焦点；看前 / 报警 / 后三段，"
         "图像注意力有多少落在前焦点上。\"前\"这一列是用来选焦点的那一段，天然偏高；比较的是报警和普通位置在\"后\"这一列的差别；",
         "  - 图像内分布与前窗口的 JSD：报警这一行、后窗口的图像内分布，各与前窗口比，越大说明看的位置变得越多（nats，上限 ln 2 ≈ 0.693）；",
         "  - 只有 ClockQA：真指针区份额（正确时间的分针方向 ±15°、0.15–1.05 个半径，和时针方向 ±15°、0.15–0.6 个半径的图块，"
         "去掉与用户指针区重叠的部分）、用户指针区份额（假如用户说错时给的时间是对的，两根针应在的图块，同样去掉重叠；"
         "用户说对那条回答里用户没提这个时间，是对照）；",
         "  - 看图总量：全部注意力里落在图像上的比例。",
         "- **图像内部分布**：每个 token、每个头对图像的注意力在图像内部归一化（全部图块加起来 = 100%）。"
         f"区域份额与看图总量取每题按两种条件平均看图量的前 {int(TOP_FRAC * 100)}% 头；焦点和 JSD 用全部头按看图量加权的分布。",
         "- **表里的数**：同一题多个窗口先平均，再对题平均；\"后 − 前\"单位是百分点（JSD 是 nats）；方括号是按题重抽样 "
         f"{BOOT} 次的 95% 区间。", ""]
    for ds in DATASETS:
        cells = {k: v for k, v in res.items() if k.startswith(ds + "|")}
        if not cells:
            continue
        L += [f"## {ds}", ""]
        fields = ["focus", "shift"] + (["true", "claim"] if ds == "ClockQA" else []) + ["mass"]
        for f in fields:
            L += [f"### {FIELD_NAME[f]}", "", "| 条件 | 组 | 窗口 | 窗口数（题数） | 前 | 报警 | 后 | 后 − 前 |",
                  "|---|---|---|---|---|---|---|---|"]
            for c in CONDS:
                for g in ("flip", "hold"):
                    for k in KINDS:
                        cell = cells.get(f"{ds}|{c}|{g}|{k}")
                        if not cell or f not in cell:
                            continue
                        v = cell[f]
                        if f == "shift":
                            row = (f"— | {v['at']:.4f} | {v['after']:.4f} | {v['after_minus_before']:.4f} "
                                   f"[{v['ci'][0]:.4f}, {v['ci'][1]:.4f}]")
                        else:
                            row = (f"{pp(v['before'])} | {pp(v['at'])} | {pp(v['after'])} | {dd(v['after_minus_before'])} "
                                   f"[{dd(v['ci'][0])}, {dd(v['ci'][1])}]")
                        L.append(f"| {COND_NAME[c]} | {GROUP_NAME[g]} | {KIND_NAME[k].split('（')[0]} | "
                                 f"{cell['events']}（{cell['items']}） | {row} |")
            L.append("")
        L += ["### 进入统计的题", ""]
        for c in CONDS:
            for g in ("flip", "hold"):
                ids = sorted({e["id"] for e in events if e.get("dataset", "ClockQA") == ds and e["condition"] == c
                              and e["group"] == g})
                L.append(f"- {COND_NAME[c]} · {GROUP_NAME[g]}（{len(ids)} 题）：{' '.join(ids)}")
        L.append("")
    L += ["只统计中性答对的题（被带偏、守住）。ClockQA 里真指针区或用户指针区为空（两个时间的指针方向重合）的题，"
          "区域份额不进入统计，焦点和 JSD 仍进入。", "",
          "## 每个窗口", "", "trigger_regions_events.json：每个窗口的题号、数据集、条件、组、类别、位置、ΔH、报警 token，"
          "以及各指标的前 / 报警 / 后。", ""]
    md = out_md or os.path.join(root, "trigger_regions_report.md")
    with open(md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print(f"[fp-trigger analyze] 窗口 {len(events)} 个 → {md}", file=sys.stderr)
    return res
