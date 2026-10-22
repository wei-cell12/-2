# -*- coding: utf-8 -*-
"""3×3 网格裁切：取格子的外接矩形，四边各外扩半格，放大到目标长边（不缩小）。"""

from typing import List, Tuple

from PIL import Image

ROWS = "ABC"


def cells_rect(cells: List[str], w: int, h: int, expand: float) -> Tuple[int, int, int, int]:
    rs = [ROWS.index(c[0]) for c in cells]
    cs = [int(c[1]) - 1 for c in cells]
    cw, ch = w / 3.0, h / 3.0
    x0 = (min(cs) - expand) * cw
    x1 = (max(cs) + 1 + expand) * cw
    y0 = (min(rs) - expand) * ch
    y1 = (max(rs) + 1 + expand) * ch
    return max(0, int(x0)), max(0, int(y0)), min(w, int(round(x1))), min(h, int(round(y1)))


def region_label(cells: List[str]) -> str:
    rs = sorted({c[0] for c in cells})
    cs = sorted({c[1] for c in cells})
    r = f"row {rs[0]}" if len(rs) == 1 else f"rows {rs[0]}-{rs[-1]}"
    c = f"column {cs[0]}" if len(cs) == 1 else f"columns {cs[0]}-{cs[-1]}"
    return f"{r}, {c}"


def make_crop(img: Image.Image, cells: List[str], target_long: int, expand: float) -> Image.Image:
    box = cells_rect(cells, img.width, img.height, expand)
    crop = img.crop(box)
    scale = max(1.0, target_long / max(crop.width, crop.height))
    if scale > 1.0:
        crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))), Image.BICUBIC)
    return crop
