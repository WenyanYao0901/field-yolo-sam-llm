# -*- coding: utf-8 -*-
"""
YOLO 主模型训练。

说明:
    用田间标注数据（maize / broadleaf / grass）微调 YOLO。
    训练完成后用 best.pt 做 detect / run。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics import YOLO

from src.utils.common import pick_device


def train_yolo(
    data_yaml: str = "configs/field.yaml",
    weights: str = "yolov8s.pt",
    epochs: int = 50,
    batch: int = 8,
    imgsz: int = 640,
    device: str = "",
    name: str = "field_yolo",
    project: str = "runs/train",
) -> Path:
    """
    启动 YOLO 训练。

    返回:
        best.pt 路径
    """
    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"未找到数据配置: {data_yaml}")

    model = YOLO(weights)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=pick_device(device),
        project=project,
        name=name,
        exist_ok=True,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"[训练完成] best 权重: {best}")
    return best
