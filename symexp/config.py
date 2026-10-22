# -*- coding: utf-8 -*-
"""实验配置。YAML 文件里没写的项目用这里的默认值（与《抗谄媚实验方案 v1》一致）。"""

from dataclasses import dataclass, field, fields, asdict
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class Config:
    # ---- 后端 -------------------------------------------------------------------------------
    backend: str = "hf"                       # hf | openai | mock
    model_path: str = ""                      # hf：本地权重目录；openai：服务端的模型名
    model_class: str = "AutoModelForImageTextToText"   # 预实验加载的是 Qwen3_5ForConditionalGeneration
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    attn_implementation: Optional[str] = None
    base_url: str = "http://localhost:8000/v1"  # openai 后端
    api_key: str = "EMPTY"
    workers: int = 1                          # hf 后端请保持 1；openai 后端可调大
    processor_kwargs: Dict[str, Any] = field(default_factory=dict)  # 传给 AutoProcessor，例如图像像素范围
    hf_single_pass: bool = True               # hf：开思考时一次 generate 完成思考与回答（同预实验）

    # 聊天模板参数。思考模式用 enable_thinking 开关；预实验的“medium 思考强度”
    # 如果也是模板参数，写在 thinking_template_kwargs 里（例如 {"reasoning_effort": "medium"}，请按你的模板改）
    thinking_template_kwargs: Dict[str, Any] = field(default_factory=lambda: {"enable_thinking": True})
    nothinking_template_kwargs: Dict[str, Any] = field(default_factory=lambda: {"enable_thinking": False})
    think_open: str = "<think>"
    think_close: str = "</think>"
    end_markers: List[str] = field(default_factory=lambda: ["<|im_end|>", "<|endoftext|>"])

    # ---- 解码（同预实验）-----------------------------------------------------------------------
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    repetition_penalty: float = 1.0

    # ---- 长度上限 ---------------------------------------------------------------------------
    think_budget: int = 4096          # 剩余思考
    answer_budget: int = 512          # 可见回答（同预实验，复用的 plain 也是 512）
    step2_answer_budget: int = 2048   # 第 2 步可见回答（含结构化区块）
    step3a_answer_budget: int = 512
    step3c_think_budget: int = 2048
    step3c_answer_budget: int = 1024
    step2_max_attempts: int = 2       # 区块解析失败时重跑一次

    # ---- 第 3 步 ----------------------------------------------------------------------------
    max_reasons: int = 3
    crop_target_long_side: int = 1024  # 裁切图放大到的长边（请设成模型实际输入的最大边长）
    crop_expand_cells: float = 0.5     # 四边各外扩半格

    # ---- 系统提示 ---------------------------------------------------------------------------
    use_system_prompt: bool = False    # 预实验没有 system 消息；复用 plain 时必须保持 False

    # ---- 杂项 -------------------------------------------------------------------------------
    order_salt: str = "symexp-v1"      # 候选顺序与 a/b 分配的随机盐
    image_root: str = ""               # 数据里图像路径的根目录；空 = 相对数据文件所在目录

    @classmethod
    def load(cls, path: Optional[str], overrides: Optional[Dict[str, Any]] = None) -> "Config":
        cfg = cls()
        data: Dict[str, Any] = {}
        if path:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        data.update({k: v for k, v in (overrides or {}).items() if v is not None})
        names = {f.name for f in fields(cls)}
        unknown = set(data) - names
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        for k, v in data.items():
            setattr(cfg, k, v)
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
