# -*- coding: utf-8 -*-
"""
难例挖掘模块。

目标:
    从 YOLO 初检结果中筛出“更值得花人工 / 交给 SAM”的样本，
    把预算集中在不确定、拥挤、易漏检的图片上。

常见难例信号:
    - 置信度落在中间带（模型犹豫）
    - 低置信框较多（需把 YOLO conf 设得低于 conf_low 才会出现）
    - 单图目标过密（遮挡/粘连风险高）
    - 完全无检测（可能漏检，也需复核）
"""
from __future__ import annotations

from typing import Any

# 无检测时的难例分数：需高于默认 hard_score_thresh(0.45)，保证进入复核
EMPTY_DETECTION_HARD_SCORE = 1.0


def score_hard_example(
    detections: list[dict[str, Any]],
    conf_low: float = 0.25,
    conf_high: float = 0.6,
) -> dict[str, Any]:
    """
    根据一图检测结果计算难例分数。

    打分逻辑:
        - 无检测: 固定给 EMPTY_DETECTION_HARD_SCORE（漏检复核）
        - 每个“不确定区”框 +1.0
        - 每个低置信框 +0.5（仅当存在 conf < conf_low 的框）
        - 目标拥挤（>=8）额外 +2.0

    返回:
        hard_score / reason / uncertain / dense
    """
    if not detections:
        # 无框不一定简单：田间可能存在漏检，仍建议抽检
        return {
            "hard_score": float(EMPTY_DETECTION_HARD_SCORE),
            "reason": "无检测，可能漏检",
            "uncertain": 0,
            "dense": False,
        }

    confs = [d["confidence"] for d in detections]
    # 中间置信：模型不够确定，最值得人工/SAM 看
    uncertain = sum(1 for c in confs if conf_low <= c < conf_high)
    # 低置信：当 YOLO 过滤阈值低于 conf_low 时才会进入此路径
    low = sum(1 for c in confs if c < conf_low)
    dense = len(detections) >= 8

    hard_score = uncertain * 1.0 + low * 0.5 + (2.0 if dense else 0.0)
    reasons = []
    if uncertain:
        reasons.append(f"不确定框{uncertain}个")
    if low:
        reasons.append(f"低置信{low}个")
    if dense:
        reasons.append("目标拥挤")
    if not reasons:
        reasons.append("相对简单")

    return {
        "hard_score": float(hard_score),
        "reason": "；".join(reasons),
        "uncertain": uncertain,
        "dense": dense,
    }


def is_uncertain_box(
    confidence: float,
    conf_low: float = 0.25,
    conf_high: float = 0.6,
) -> bool:
    """
    判断单框是否落在不确定置信区间 [conf_low, conf_high)。

    这类框优先送入 SAM 做边界精炼。
    """
    return conf_low <= confidence < conf_high
