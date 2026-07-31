# -*- coding: utf-8 -*-
"""
SAM 半自动精炼模块（视觉大模型）。

用途:
    用 Segment Anything 根据 YOLO 粗框生成精细掩膜，再取外接矩形，
    提升难例定位质量，并导出可入库的 YOLO 标签。

注意:
    - SAM 本身不分类；类别与置信度始终沿用 YOLO
    - SAM 不进入最终 mAP 验收口径
    - 权重缺失时直接报错，不使用假数据顶替
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from src.utils.common import ensure_dir, pick_device


def _mask_to_yolo_bbox(mask: np.ndarray, img_w: int, img_h: int) -> list[float] | None:
    """
    将二值掩膜转换为 YOLO 归一化边界框。

    参数:
        mask: 二维数组，前景 > 0
        img_w / img_h: 原图像宽高

    返回:
        [cx, cy, w, h]（相对宽高归一化到 0~1）；无效掩膜返回 None
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    bw = x2 - x1 + 1
    bh = y2 - y1 + 1
    # 过小区域通常是噪声，直接丢弃
    if bw <= 1 or bh <= 1:
        return None
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    return [cx, cy, bw / img_w, bh / img_h]


def yolo_bbox_to_xyxy(yolo_bbox: list[float], img_w: int, img_h: int) -> list[float]:
    """
    YOLO 归一化框 [cx, cy, w, h] -> 像素坐标 [x1, y1, x2, y2]。

    供可视化与 final_results.json 使用。
    """
    cx, cy, bw, bh = yolo_bbox
    w = bw * img_w
    h = bh * img_h
    x1 = cx * img_w - w / 2.0
    y1 = cy * img_h - h / 2.0
    x2 = x1 + w
    y2 = y1 + h
    return [float(x1), float(y1), float(x2), float(y2)]


def load_sam(checkpoint: str, model_type: str = "vit_b", device: str = ""):
    """
    加载 Segment Anything 模型与预测器。

    参数:
        checkpoint: 权重路径，例如 weights/sam_vit_b_01ec64.pth
        model_type: vit_b / vit_l / vit_h（需与权重匹配）
        device: 空则自动选择
    """
    from segment_anything import SamPredictor, sam_model_registry

    if not Path(checkpoint).exists():
        raise FileNotFoundError(
            f"未找到 SAM 权重: {checkpoint}\n"
            "请下载后放到 weights/，例如 sam_vit_b_01ec64.pth"
        )
    dev = pick_device(device)
    # ultralytics 用 "0" 表示 CUDA，SAM 需要 "cuda" 字符串
    torch_device = "cuda" if dev == "0" else dev
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=torch_device)
    return SamPredictor(sam)


def annotate_with_box_prompt(
    predictor,
    image_bgr: np.ndarray,
    box_xyxy: list[float],
    class_id: int,
) -> dict[str, Any] | None:
    """
    使用矩形框提示 SAM，生成掩膜并导出精炼框。

    参数:
        predictor: load_sam 返回的预测器
        image_bgr: OpenCV 读取的 BGR 图像
        box_xyxy: [x1, y1, x2, y2] 提示框（通常来自 YOLO）
        class_id: 类别编号（由 YOLO/标注员给定）

    返回:
        含 class_id / yolo_bbox / bbox_xyxy / sam_score / mask；失败返回 None
    """
    # SAM 期望 RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    box = np.array(box_xyxy, dtype=np.float32)
    # multimask_output=False：只要最置信的一个掩膜，便于批量处理
    masks, scores, _ = predictor.predict(box=box, multimask_output=False)
    mask = masks[0].astype(np.uint8)
    h, w = image_bgr.shape[:2]
    yolo_box = _mask_to_yolo_bbox(mask, w, h)
    if yolo_box is None:
        return None
    return {
        "class_id": class_id,
        "yolo_bbox": yolo_box,
        "bbox_xyxy": yolo_bbox_to_xyxy(yolo_box, w, h),
        "sam_score": float(scores[0]),
        "mask": mask,
    }


def save_yolo_label(label_path: str | Path, objects: list[dict[str, Any]]) -> None:
    """
    将目标列表写成 YOLO 格式 txt。

    每行格式:
        <class_id> <cx> <cy> <w> <h>
    坐标均为相对图像宽高的归一化值。
    """
    ensure_dir(Path(label_path).parent)
    lines = []
    for obj in objects:
        cid = int(obj["class_id"])
        cx, cy, bw, bh = obj["yolo_bbox"]
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    # 无目标时写空文件，表示本图无 SAM 精炼框
    Path(label_path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
