# -*- coding: utf-8 -*-
"""
通用工具模块。

提供：
1. YAML / JSON 读写（统一 UTF-8，避免中文乱码）
2. 推理设备自动选择（CUDA / Apple MPS / CPU）
3. 图像文件枚举与目录创建
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """
    加载 YAML 配置文件。

    参数:
        path: 配置路径，如 configs/default.yaml

    返回:
        解析后的字典；文件为空时返回 {}
    """
    with open(path, "r", encoding="utf-8") as f:
        # safe_load：只解析数据，不执行任意代码
        return yaml.safe_load(f) or {}


def flatten_default_config(path: str | Path) -> dict[str, Any]:
    """
    将 configs/default.yaml 展平为 CLI / 流水线可用的扁平参数。

    说明:
        - 文件不存在时返回空字典（由调用方保留代码内默认值）
        - 仅抽取已知字段，忽略无关键
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}

    cfg = load_yaml(cfg_path)
    model = cfg.get("model") or {}
    pipeline = cfg.get("pipeline") or {}
    assist = cfg.get("assist") or {}
    llm = cfg.get("llm") or {}

    flat: dict[str, Any] = {}
    if "name" in model:
        flat["weights"] = model["name"]
    for key in ("imgsz", "conf", "iou", "device"):
        if key in model:
            flat[key] = model[key]
    for key in (
        "source",
        "save_dir",
        "conf_low",
        "conf_high",
        "hard_score_thresh",
        "skip_sam",
        "skip_llm",
    ):
        if key in pipeline:
            flat[key] = pipeline[key]
    if "sam_checkpoint" in assist:
        flat["sam_checkpoint"] = assist["sam_checkpoint"]
    if "sam_type" in assist:
        flat["sam_type"] = assist["sam_type"]
    if "base_url" in llm:
        flat["llm_base_url"] = llm["base_url"]
    if "model" in llm:
        flat["llm_model"] = llm["model"]
    if "temperature" in llm:
        flat["llm_temperature"] = llm["temperature"]
    return flat


def save_json(data: Any, path: str | Path) -> None:
    """
    将数据保存为 JSON。

    说明:
        - ensure_ascii=False：中文类别名、路径可读
        - 自动创建父目录，避免首次写入失败
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def pick_device(prefer: str = "") -> str:
    """
    自动选择计算设备。

    优先级:
        1. 调用方显式指定 prefer（如 "0" / "cpu" / "mps"）
        2. NVIDIA CUDA
        3. Apple Silicon MPS
        4. CPU

    返回值与 ultralytics 约定一致：
        - "0" 表示第一块 GPU
        - "mps" / "cpu" 为设备名字符串
    """
    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def list_images(folder: str | Path) -> list[Path]:
    """
    列出图像文件。

    行为:
        - 传入单文件：校验后缀后返回单元素列表
        - 传入目录：递归收集常见图像后缀

    支持后缀: jpg / jpeg / png / bmp / tif / tiff / webp
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    folder = Path(folder)
    if folder.is_file():
        return [folder] if folder.suffix.lower() in exts else []
    # rglob：适配按地块/日期分子目录存放的田间数据
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in exts)


def ensure_dir(path: str | Path) -> Path:
    """
    确保目录存在并返回 Path。

    用于 runs/、labels/ 等输出目录的统一创建入口。
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
