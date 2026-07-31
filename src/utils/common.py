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
