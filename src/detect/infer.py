# -*- coding: utf-8 -*-
"""
田间图像 YOLO 实时检测模块。

角色定位（论文方案）:
    - 主模型：负责输出每个目标的位置、类别、置信度
    - SAM / LLM 不参与本模块线上推理
    - 验收指标（mAP 等）也以 YOLO 检测结果为准
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

from src.utils.common import (
    ensure_dir,
    list_images,
    pick_device,
    relative_image_path,
    save_json,
    validate_inference_params,
)


def load_detector(weights: str = "yolov8s.pt", device: str = "") -> YOLO:
    """
    加载 YOLO 检测模型。

    参数:
        weights: 权重文件
            - yolov8s.pt / yolo11s.pt：官方预训练（首次自动下载）
            - 自训 best.pt：田间微调后的业务权重
        device: 设备字符串，空则自动选择
    """
    model = YOLO(weights)
    model.to(pick_device(device))
    return model


def detect_image(
    model: YOLO,
    image_path: str | Path,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
) -> list[dict[str, Any]]:
    """
    对单张田间图像做目标检测。

    参数:
        model: 已加载的 YOLO 模型
        image_path: 图像路径
        conf: 置信度阈值，低于此值的框会被过滤
        iou: NMS IoU 阈值，用于抑制重复框
        imgsz: 推理输入边长（方形缩放）

    返回:
        检测列表，每项包含：
            class_id / class_name / confidence / bbox_xyxy
    """
    results = model.predict(
        source=str(image_path),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,  # 关闭逐图冗余日志，由外层统一打印
    )
    names = model.names
    detections: list[dict[str, Any]] = []
    for r in results:
        # 无目标时 boxes 可能为空
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls.item())
            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": names.get(cls_id, str(cls_id)),
                    "confidence": float(box.conf.item()),
                    # xyxy：便于可视化与后续送入 SAM
                    "bbox_xyxy": [float(x) for x in box.xyxy[0].tolist()],
                }
            )
    return detections


def draw_detections(
    image_path: str | Path,
    detections: list[dict[str, Any]],
    save_path: str | Path,
) -> None:
    """
    将检测框与类别文字绘制到图像并保存。

    颜色约定:
        - 绿色：YOLO 原始框（refined_by=yolo）
        - 橙色：经 SAM 精炼后的框（refined_by=sam）
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
        tag = det.get("refined_by", "yolo")
        label = f"{det['class_name']} {det['confidence']:.2f} [{tag}]"
        # BGR：绿=YOLO，橙=SAM
        color = (40, 180, 80) if tag == "yolo" else (40, 140, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            label,
            (x1, max(20, y1 - 8)),  # 避免文字画出上边界
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    ensure_dir(Path(save_path).parent)
    cv2.imwrite(str(save_path), img)


def run_detect(
    source: str,
    weights: str = "yolov8s.pt",
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    device: str = "",
    save_dir: str = "runs/detect",
) -> list[dict[str, Any]]:
    """
    批量检测田间图像，并落盘可视化与 JSON。

    输出结构:
        save_dir/
          vis/               # 画框图像
          predictions.json   # 结构化检测结果
    """
    validate_inference_params(conf, iou, imgsz)
    images = list_images(source)
    if not images:
        raise FileNotFoundError(f"未找到图像: {source}")

    # Validate input before loading a potentially large model.
    model = load_detector(weights=weights, device=device)
    out_dir = ensure_dir(save_dir)
    vis_dir = ensure_dir(out_dir / "vis")
    all_results: list[dict[str, Any]] = []

    for img_path in images:
        dets = detect_image(model, img_path, conf=conf, iou=iou, imgsz=imgsz)
        rel_path = relative_image_path(img_path, source)
        draw_detections(img_path, dets, vis_dir / rel_path)
        all_results.append(
            {"image": str(img_path), "count": len(dets), "detections": dets}
        )
        print(f"[检测] {img_path.name}: {len(dets)} 个目标")

    save_json(all_results, out_dir / "predictions.json")
    print(f"结果已保存: {out_dir}")
    return all_results
