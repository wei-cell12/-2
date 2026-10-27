# -*- coding: utf-8 -*-
"""逐 token 注意力去向（离线，不生成）：对已保存的生成轨迹，用生成时的那份提示把“提示 + 已生成内容”
整段前向一次（模型不做任何修改），在 16 个标准注意力层上显式算出每个已生成 token 位置的注意力，
按被关注的 token 分成五类求和：

  image     图像 token
  question  问题那一行
  claim     用户主张那一行（例如 "I believe the correct answer is no."；中性条件没有这一类，恒为 0）
  output    模型自己已经写出的内容（思考 + 回答）
  other     其余（对话模板、<think> 等）

第 t 行 = 预测第 t 个输出 token 时（查询位置 n_in−1+t）的注意力，与生成时记录的 pos_image_mass 第 t 行对应。

输出（--out 指定的 JSON，可以上传）：每条轨迹一行，含
  pieces        逐 token 文字（用来切句子）
  mass_all      [n_out][5]   五类注意力，16 层 × 全部头平均
  img_layer     [n_out][16]  图像注意力，逐层（头平均）
  claim_layer   [n_out][16]  用户主张注意力，逐层（头平均）
另外把逐（层, 头）的完整数组 [n_out, 16, H, 5]（float16）存到 --out 同目录的 spans_heads/ 下，留在服务器，不上传。

检查（不通过就停下）：
  - 重新构造的提示与记录里的 input_ids 一致；问题行、主张行在提示里各恰好出现一次
  - 前 8 个查询位置上，显式计算的注意力乘 V 与原注意力函数的输出一致（与生成时的自检同一标准），
    防止掩码、缩放、GQA 头顺序写错
  - 观察轨迹（mode = observe）：与生成时记录的逐 token 图像注意力 pos_image_mass 对照：
    逐（token, 层, 头）差的中位数 < 0.005，逐 token 平均值的相关系数 > 0.98，
    且错开 ±1 个 token 时相关系数都更低（防止行号错位）；干预轨迹只报告差值，不作为检查
"""

import glob
import hashlib
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

CATS = ["image", "question", "claim", "output", "other"]
CHUNK = 256
SELFCHECK_ROWS = 8
TOL_MEDIAN = 0.005
TOL_CORR = 0.98


class SpanCheckFailed(RuntimeError):
    pass


# ---------------------------------------------------------------- 纯函数（可在没有 torch 的环境里测试）
def find_once(text: str, sub: str, what: str) -> Tuple[int, int]:
    n = text.count(sub)
    if n != 1:
        raise SpanCheckFailed(f"{what} 在提示里出现 {n} 次（应为 1 次）：{sub!r}")
    a = text.index(sub)
    return a, a + len(sub)


def char_span_tokens(pieces: List[str], span: Tuple[int, int]) -> List[int]:
    """与字符区间 [a, b) 有重叠的 token 下标。"""
    a, b = span
    out, pos = [], 0
    for i, p in enumerate(pieces):
        s, e = pos, pos + len(p)
        if e > a and s < b and len(p) > 0:
            out.append(i)
        pos = e
    return out


def split_message(message: str, question: str) -> Optional[str]:
    """返回主张那一行；中性条件（消息就是问题本身）返回 None。"""
    if message == question:
        return None
    if not message.startswith(question + "\n"):
        raise SpanCheckFailed(f"消息不是“问题 + 换行 + 主张”的格式：{message!r}")
    claim = message[len(question) + 1:]
    if not claim.strip() or "\n" in claim:
        raise SpanCheckFailed(f"主张行格式不对：{claim!r}")
    return claim


def key_labels(n_in: int, n_out: int, img_idx: List[int], q_idx: List[int], c_idx: List[int]) -> np.ndarray:
    """每个 key 位置的类别编号（CATS 的下标）。"""
    lab = np.full(n_in + n_out, CATS.index("other"), np.int64)
    lab[n_in:] = CATS.index("output")
    for idx, name in ((q_idx, "question"), (c_idx, "claim"), (img_idx, "image")):
        if any(i >= n_in for i in idx):
            raise SpanCheckFailed(f"{name} 的 token 落在提示之外")
        lab[list(idx)] = CATS.index(name)
    if len(set(img_idx) & (set(q_idx) | set(c_idx))) or len(set(q_idx) & set(c_idx)):
        raise SpanCheckFailed("图像、问题、主张三类 token 有重叠")
    return lab


def compare_recorded(rec_mass: np.ndarray, new_img: np.ndarray) -> Dict[str, float]:
    """rec_mass、new_img：[n_out, L, H]。返回对照统计。"""
    if rec_mass.shape != new_img.shape:
        raise SpanCheckFailed(f"记录的 pos_image_mass 形状 {rec_mass.shape} 与重算的 {new_img.shape} 不同")
    a, b = rec_mass.astype(np.float64), new_img.astype(np.float64)
    ok = np.isfinite(a)
    d = np.abs(a - b)[ok]
    ta, tb = np.nanmean(a, axis=(1, 2)), b.mean(axis=(1, 2))

    def corr(x, y):
        return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and x.std() > 0 and y.std() > 0 else float("nan")
    return {"median_abs_diff": float(np.median(d)), "p99_abs_diff": float(np.quantile(d, 0.99)),
            "max_abs_diff": float(d.max()), "corr_token_mean": corr(ta, tb),
            "corr_shift_minus1": corr(ta[1:], tb[:-1]), "corr_shift_plus1": corr(ta[:-1], tb[1:])}


# ---------------------------------------------------------------- GPU 部分
class SpanCollector:
    """装在 16 个标准注意力层的注意力函数上：原样返回原函数的输出，另外显式算出指定查询行的注意力并按类别求和。"""

    def __init__(self, torch, labels: np.ndarray, q0: int, nq: int):
        self.torch = torch
        self.labels = labels
        self.q0, self.nq = q0, nq
        self.out: Dict[int, np.ndarray] = {}
        self.selfcheck: Dict[int, float] = {}
        self._oh = {}

    def _onehot(self, device):
        if str(device) not in self._oh:
            t = self.torch
            lab = t.as_tensor(self.labels, device=device)
            self._oh[str(device)] = t.nn.functional.one_hot(lab, len(CATS)).float()
        return self._oh[str(device)]

    def attention(self, orig, module, query, key, value, attention_mask, *args, **kwargs):
        t = self.torch
        res = orig(module, query, key, value, attention_mask, *args, **kwargs)
        layer = int(module.layer_idx)
        B, H, Q, D = query.shape
        K = key.shape[2]
        if B != 1 or Q != K or K != len(self.labels):
            raise SpanCheckFailed(f"第 {layer} 层：只支持 batch=1、不带缓存的整段前向（Q={Q} K={K} 标签={len(self.labels)}）")
        rep = H // key.shape[1]
        scale = kwargs.get("scaling", args[1] if len(args) > 1 else None)
        if scale is None:
            scale = getattr(module, "scaling", D ** -0.5)
        kk = (key[0].repeat_interleave(rep, dim=0) if rep > 1 else key[0]).float()      # [H, K, D]
        vv = (value[0].repeat_interleave(rep, dim=0) if rep > 1 else value[0]).float()
        oh = self._onehot(query.device)
        parts = []
        for a in range(self.q0, self.q0 + self.nq, CHUNK):
            b = min(a + CHUNK, self.q0 + self.nq)
            sc = t.matmul(query[0, :, a:b].float(), kk[:, :b].transpose(-1, -2)) * scale       # [H, b-a, b]
            qi = t.arange(a, b, device=sc.device)[:, None]
            kj = t.arange(b, device=sc.device)[None, :]
            sc = sc.masked_fill(kj > qi, float("-inf"))
            p = t.softmax(sc, dim=-1)
            if a == self.q0:
                n = min(SELFCHECK_ROWS, b - a)
                mine = t.matmul(p[:, :n], vv[:, :b]).to(query.dtype).float()               # [H, n, D]
                ref = res[0][0, a:a + n].float().transpose(0, 1)                           # [B, Q, H, D] → [H, n, D]
                allowed = t.clamp(ref.abs() * 2.0 ** -6, min=0.05)
                ratio = float(((mine - ref).abs() / allowed).max())
                self.selfcheck[layer] = ratio
                if not ratio <= 1.0:
                    raise SpanCheckFailed(f"第 {layer} 层：显式注意力与原函数输出不一致（差/允许差 = {ratio:.3f}）")
            parts.append(t.matmul(p, oh[:b]).transpose(0, 1))                                # [b-a, H, C]
            del sc, p
        self.out[layer] = t.cat(parts, 0).cpu().numpy()
        return res


def analyze_one(backend, item: dict, rec: dict) -> Tuple[dict, np.ndarray]:
    from . import LAYERS
    from . import controller as fc
    from ..attn import image_token_id
    from .entropy import _batch
    from .generate import build_inputs
    torch, tok, model = backend.torch, backend.tokenizer, backend.model
    message = item["messages"][rec["condition"]]
    _, inp = build_inputs(backend, item["image"], message)
    ids = inp["input_ids"][0].tolist()
    if ids != rec["input_ids"]:
        raise SpanCheckFailed(f"{rec['id']}：重新构造的提示与记录里的 input_ids 不一致")
    out_ids = rec["output_ids"]
    n_in, n_out = len(ids), len(out_ids)
    pin = [tok.decode([x], skip_special_tokens=False, clean_up_tokenization_spaces=False) for x in ids]
    text = "".join(pin)
    q_idx = char_span_tokens(pin, find_once(text, item["question"], "问题"))
    claim = split_message(message, item["question"])
    c_idx = char_span_tokens(pin, find_once(text, claim, "主张")) if claim else []
    img_id = image_token_id(backend)
    img_idx = [i for i, x in enumerate(ids) if x == img_id]
    if len(img_idx) != rec["n_img"]:
        raise SpanCheckFailed(f"{rec['id']}：图像 token 数 {len(img_idx)} 与记录 {rec['n_img']} 不一致")
    lab = key_labels(n_in, n_out, img_idx, q_idx, c_idx)

    fc.install(model)
    col = SpanCollector(torch, lab, n_in - 1, n_out)
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "rope_deltas"):
        inner.rope_deltas = None
    with torch.inference_mode(), fc.active(col):
        model(**_batch(backend, inp, out_ids), use_cache=False, return_dict=True)
    if sorted(col.out) != LAYERS:
        raise SpanCheckFailed(f"{rec['id']}：收集到的层 {sorted(col.out)} 与 16 个标准注意力层不一致")
    M = np.stack([col.out[l] for l in LAYERS], 1)                                           # [n_out, L, H, C]
    if not np.isfinite(M).all() or np.abs(M.sum(-1) - 1).max() > 1e-3:
        raise SpanCheckFailed(f"{rec['id']}：五类注意力之和不为 1（最大偏差 {np.abs(M.sum(-1) - 1).max():.2e}）")
    if not claim and M[..., CATS.index("claim")].max() > 0:
        raise SpanCheckFailed("中性条件不应有主张类注意力")

    check = {"selfcheck_ratio_max": max(col.selfcheck.values())}
    npz = rec.get("npz")
    if npz and os.path.exists(npz):
        if rec.get("npz_sha256") and hashlib.sha256(open(npz, "rb").read()).hexdigest() != rec["npz_sha256"]:
            raise SpanCheckFailed(f"{npz} 的 sha256 与记录不一致")
        cmp = compare_recorded(np.load(npz)["pos_image_mass"].astype(np.float32), M[..., CATS.index("image")])
        check["vs_recorded"] = cmp
        if rec.get("mode") == "observe":
            bad = (not cmp["median_abs_diff"] < TOL_MEDIAN) or (not cmp["corr_token_mean"] > TOL_CORR) or \
                  (not cmp["corr_token_mean"] > max(cmp["corr_shift_minus1"], cmp["corr_shift_plus1"]))
            if bad:
                raise SpanCheckFailed(f"{rec['id']}__{rec['condition']}：与生成时记录的图像注意力对不上 {cmp}")
    else:
        check["vs_recorded"] = None
    close = out_ids.index(backend.close_id) if backend.close_id in out_ids else n_out
    pieces = [tok.decode([x], skip_special_tokens=False, clean_up_tokenization_spaces=False) for x in out_ids]
    r4 = lambda a: [[round(float(x), 5) for x in row] for row in a]
    row = {"n_in": n_in, "n_out": n_out, "close": close, "n_img": len(img_idx), "n_question": len(q_idx),
           "n_claim": len(c_idx), "cats": CATS, "layers": LAYERS, "pieces": pieces,
           "mass_all": r4(M.mean(axis=(1, 2))),
           "img_layer": r4(M[..., CATS.index("image")].mean(2)),
           "claim_layer": r4(M[..., CATS.index("claim")].mean(2)),
           "check": check}
    return row, M.astype(np.float16)


def _records(root: str, subs: List[str]) -> List[Tuple[str, str]]:
    out = []
    for s in subs:
        pat = os.path.join(root, "obs", "records", "*.json") if s == "obs" else os.path.join(root, "int", "*", "records", "*.json")
        for p in sorted(glob.glob(pat)):
            out.append((os.path.relpath(os.path.dirname(os.path.dirname(p)), root), p))
    return out


def run(cfg, roots: List[str], out_path: str, subs: List[str] = ("obs", "int"), shard: str = "") -> dict:
    from .pipeline import Engine, _atomic_json, _check_no_failure, _fail, _shard, load_items
    work = os.path.dirname(os.path.abspath(out_path))
    _check_no_failure(work)
    todo = []
    for root in roots:
        items = {i["id"]: i for i in load_items(root)}
        for sub, p in _records(root, list(subs)):
            todo.append((root, items, sub, p))
    if not todo:
        raise SystemExit("没有找到任何记录")
    todo = _shard(todo, shard)
    eng = Engine(cfg)
    if eng.backend is None:
        raise SystemExit("fp-spans 需要 hf 后端")
    heads_dir = os.path.join(work, "spans_heads")
    os.makedirs(heads_dir, exist_ok=True)
    res = []
    for k, (root, items, sub, p) in enumerate(todo):
        tag = os.path.splitext(os.path.basename(p))[0]
        task = f"spans/{os.path.basename(root.rstrip('/'))}/{sub}/{tag}"
        try:
            rec = json.load(open(p, encoding="utf-8"))
            row, M = analyze_one(eng.backend, items[rec["id"]], rec)
        except BaseException as e:
            _fail(work, task, e)
            raise
        hp = os.path.join(heads_dir, task.replace("/", "_") + ".npz")
        np.savez_compressed(hp, mass=M, cats=np.array(CATS))
        row.update(id=rec["id"], condition=rec["condition"], mode=rec.get("mode"), source=sub,
                   root=os.path.basename(root.rstrip("/")), heads_npz=hp)
        res.append(row)
        if eng.backend.torch.cuda.is_available():
            eng.backend.torch.cuda.empty_cache()
        cv = row["check"]["vs_recorded"]
        m = np.array(row["mass_all"])[: row["close"]].mean(0)
        print(f"  {k + 1}/{len(todo)} {task} tokens={row['n_out']} 思考段 图像={m[0]:.3f} 问题={m[1]:.3f} "
              f"主张={m[2]:.4f} 自己={m[3]:.3f} | 对照记录：" +
              (f"中位差={cv['median_abs_diff']:.4f} r={cv['corr_token_mean']:.3f}" if cv else "无"), file=sys.stderr)
    out = {"roots": list(roots), "shard": shard, "note": __doc__.split("\n\n")[0], "rows": res}
    _atomic_json(out_path, out)
    print(f"[fp-spans] {len(res)} 条轨迹 → {out_path}", file=sys.stderr)
    return out
