# -*- coding: utf-8 -*-
"""JSONL 读写，支持中断后续跑。"""

import json
import os
import threading
from typing import Dict, Iterable, List


def read_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # 中断时写了半行
    return out


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class KeyedStore:
    """按 key 去重的追加式存储；重复 key 以最后一条为准。"""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.lock = threading.Lock()
        self.rows: Dict[str, dict] = {}
        for r in read_jsonl(path):
            if "key" in r:
                self.rows[r["key"]] = r

    def get(self, key: str):
        return self.rows.get(key)

    def put(self, row: dict) -> None:
        with self.lock:
            self.rows[row["key"]] = row
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def values(self) -> List[dict]:
        return list(self.rows.values())
