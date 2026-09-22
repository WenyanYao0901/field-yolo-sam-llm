# -*- coding: utf-8 -*-
"""
闭环流水线编排模块。

完整链路:
    1. YOLO 初检（类别 + 置信度 + 初检框）
    2. 难例打分（复用初检结果，避免二次推理）
    3. SAM 精炼难例图 / 不确定框（可用 --skip-sam 跳过）
    4. 合并最终检测结果并可视化
    5. LLM（DeepSeek/GPT）质检与中文报告（可用 --skip-llm 跳过）

合并规则（论文可复述）:
    - 类别、置信度：始终使用 YOLO
    - 框：难例/不确定且 SAM 成功 -> 用 SAM 外接框；否则保留 YOLO 框
    - 每条检测增加 refined_by: "yolo" | "sam"
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from src.assist.hard_mining import is_uncertain_box, score_hard_example
from src.assist.llm_review import review_with_llm
from src.assist.sam_annotate import (
    load_sam,
    predict_box_prompt,
    save_yolo_label,
    set_sam_image,
)
from src.detect.infer import detect_image, draw_detections, load_detector
from src.utils.common import (
    ensure_dir,
    list_images,
    relative_image_path,
    save_json,
    validate_inference_params,
)


def _should_refine_box(
    det: dict[str, Any],
    image_is_hard: bool,
    conf_low: float,
    conf_high: float,
) -> bool:
    """
    判断该框是否送入 SAM。

    条件（满足其一即可）:
        1. 单框置信度落在不确定区间
        2. 该图被判定为难例图（整图框都尝试精炼）
    """
    if is_uncertain_box(det["confidence"], conf_low, conf_high):
        return True
    return image_is_hard


def run_pipeline(
    source: str,
    weights: str = "yolov8s.pt",
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    device: str = "",
    save_dir: str = "runs/pipeline/run",
    sam_checkpoint: str = "weights/sam_vit_b_01ec64.pth",
    sam_type: str = "vit_b",
    conf_low: float = 0.25,
    conf_high: float = 0.6,
    hard_score_thresh: float = 0.45,
    llm_base_url: str = "https://api.deepseek.com",
    llm_model: str = "deepseek-chat",
    llm_temperature: float = 0.2,
    skip_sam: bool = False,
    skip_llm: bool = False,
) -> dict[str, Any]:
    """
    执行完整闭环，返回 summary 字典。

    输出目录结构:
        save_dir/
          predictions_yolo.json   # YOLO 初检
          hard_examples.json      # 难例清单
          final_results.json      # 合并后最终结果
          vis/                    # 最终可视化
          labels_sam/             # SAM 精炼 YOLO txt（可入库）
          sam_masks/              # 掩膜可视化
          qc_report.json          # LLM 质检（未跳过时）
          report.md               # LLM 中文报告（未跳过时）
          summary.json            # 流水线摘要
    """
    validate_inference_params(
        conf,
        iou,
        imgsz,
        conf_low=conf_low,
        conf_high=conf_high,
        hard_score_thresh=hard_score_thresh,
    )
    images = list_images(source)
    if not images:
        raise FileNotFoundError(f"未找到图像: {source}")

    out_dir = ensure_dir(save_dir)
    vis_dir = ensure_dir(out_dir / "vis")
    label_dir = ensure_dir(out_dir / "labels_sam")
    mask_dir = ensure_dir(out_dir / "sam_masks")

    print(f"[流水线] 图像数={len(images)} 输出={out_dir}")
    if skip_sam:
        print("[流水线] 已跳过 SAM")
    if skip_llm:
        print("[流水线] 已跳过 LLM")

    # ---------- 1. YOLO 初检 ----------
    model = load_detector(weights=weights, device=device)
    yolo_results: list[dict[str, Any]] = []
    for img_path in images:
        dets = detect_image(model, img_path, conf=conf, iou=iou, imgsz=imgsz)
        yolo_results.append(
            {"image": str(img_path), "count": len(dets), "detections": dets}
        )
        print(f"[YOLO] {img_path.name}: {len(dets)} 个目标")
    save_json(yolo_results, out_dir / "predictions_yolo.json")

    # ---------- 2. 难例打分 ----------
    hard_examples: list[dict[str, Any]] = []
    scored_items: list[dict[str, Any]] = []
    for item in yolo_results:
        hard = score_hard_example(
            item["detections"], conf_low=conf_low, conf_high=conf_high
        )
        scored = {**item, **hard}
        scored_items.append(scored)
        if hard["hard_score"] >= hard_score_thresh:
            hard_examples.append(scored)
    hard_examples.sort(key=lambda x: x["hard_score"], reverse=True)
    save_json(hard_examples, out_dir / "hard_examples.json")
    print(f"[难例] 筛出 {len(hard_examples)} 张")

    hard_image_set = {h["image"] for h in hard_examples}

    # ---------- 3. SAM 精炼 + 结果合并 ----------
    # 仅在确有框需要精炼时再加载权重，避免无难例时白白读大模型
    need_sam = (not skip_sam) and any(
        _should_refine_box(det, item["image"] in hard_image_set, conf_low, conf_high)
        for item in scored_items
        for det in item["detections"]
    )
    predictor = load_sam(sam_checkpoint, sam_type, device=device) if need_sam else None
    if skip_sam:
        pass
    elif not need_sam:
        print("[SAM] 本批次无需精炼框，跳过加载权重")

    final_results: list[dict[str, Any]] = []
    sam_box_count = 0

    for item in scored_items:
        img_path = Path(item["image"])
        rel_path = relative_image_path(img_path, source)
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {img_path}")

        image_is_hard = item["image"] in hard_image_set
        merged: list[dict[str, Any]] = []
        sam_labels: list[dict[str, Any]] = []

        # 同一张图只编码一次 embedding
        refine_indices = [
            i
            for i, det in enumerate(item["detections"])
            if predictor is not None
            and _should_refine_box(det, image_is_hard, conf_low, conf_high)
        ]
        img_h = img_w = 0
        if refine_indices:
            img_h, img_w = set_sam_image(predictor, image)

        for i, det in enumerate(item["detections"]):
            # 默认保留 YOLO 结果
            out_det = {
                "class_id": det["class_id"],
                "class_name": det["class_name"],
                "confidence": det["confidence"],
                "bbox_xyxy": list(det["bbox_xyxy"]),
                "refined_by": "yolo",
            }
            if i in refine_indices:
                refined = predict_box_prompt(
                    predictor,
                    det["bbox_xyxy"],
                    int(det["class_id"]),
                    img_w,
                    img_h,
                )
                if refined is not None:
                    # 只替换框；类别与置信度仍用 YOLO
                    out_det["bbox_xyxy"] = refined["bbox_xyxy"]
                    out_det["refined_by"] = "sam"
                    out_det["sam_score"] = refined["sam_score"]
                    sam_box_count += 1
                    sam_labels.append(
                        {
                            "class_id": det["class_id"],
                            "yolo_bbox": refined["yolo_bbox"],
                        }
                    )
                    mask_path = mask_dir / rel_path.parent / f"{img_path.stem}_{i}.png"
                    ensure_dir(mask_path.parent)
                    cv2.imwrite(str(mask_path), refined["mask"] * 255)

            merged.append(out_det)

        if sam_labels:
            save_yolo_label(label_dir / rel_path.with_suffix(".txt"), sam_labels)

        draw_detections(img_path, merged, vis_dir / rel_path)
        final_results.append(
            {
                "image": str(img_path),
                "count": len(merged),
                "detections": merged,
                "hard_score": item["hard_score"],
                "hard_reason": item["reason"],
            }
        )
        print(
            f"[合并] {img_path.name}: {len(merged)} 目标, "
            f"hard={item['hard_score']:.2f}"
        )

    save_json(final_results, out_dir / "final_results.json")

    # ---------- 4. LLM 质检与报告 ----------
    llm_artifacts: dict[str, str] = {}
    if skip_llm:
        print("[LLM] 已跳过，不生成质检报告")
    else:
        llm_out = review_with_llm(
            final_results=final_results,
            hard_examples=hard_examples,
            base_url=llm_base_url,
            model=llm_model,
            temperature=llm_temperature,
        )
        save_json(llm_out["qc"], out_dir / "qc_report.json")
        (out_dir / "report.md").write_text(llm_out["report_md"], encoding="utf-8")
        save_json(llm_out["summary_sent"], out_dir / "llm_input_summary.json")
        llm_artifacts = {
            "qc_report": str(out_dir / "qc_report.json"),
            "report_md": str(out_dir / "report.md"),
        }

    summary = {
        "source": source,
        "weights": weights,
        "num_images": len(images),
        "num_hard_images": len(hard_examples),
        "total_detections": sum(x["count"] for x in final_results),
        "sam_refined_boxes": sam_box_count,
        "skip_sam": skip_sam,
        "skip_llm": skip_llm,
        "llm_model": None if skip_llm else llm_model,
        "output_dir": str(out_dir),
        "artifacts": {
            "final_results": str(out_dir / "final_results.json"),
            "vis": str(vis_dir),
            **llm_artifacts,
        },
    }
    save_json(summary, out_dir / "summary.json")
    print(f"[完成] 最终结果: {out_dir / 'final_results.json'}")
    if not skip_llm:
        print(f"[完成] 质检报告: {out_dir / 'qc_report.json'}")
        print(f"[完成] 中文报告: {out_dir / 'report.md'}")
    return summary
