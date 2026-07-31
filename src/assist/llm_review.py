# -*- coding: utf-8 -*-
"""
LLM 质检与报告模块（语言大模型：DeepSeek / GPT）。

角色定位（论文方案）:
    - 输入：结构化检测摘要（不上传原图，稳定、省成本、易复现）
    - 输出：质检 JSON + 中文 Markdown 报告
    - 不改检测框、不改类别、不进 mAP 验收口径

接口约定:
    OpenAI 兼容 Chat Completions
    默认 DeepSeek；可通过 base_url / model 切换到 GPT
密钥:
    直接写在下方 LLM_API_KEY 常量中（不再读环境变量）
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# LLM API Key（写死在此；把引号内换成你的真实密钥后即可调用）
# DeepSeek 一般以 sk- 开头；OpenAI 同理
# ---------------------------------------------------------------------------
LLM_API_KEY = "REDACTED_REVOKED_KEY"


def _basename(path: str) -> str:
    """只保留文件名，缩短发给 LLM 的 prompt。"""
    return Path(path).name


def resolve_api_key() -> str:
    """
    返回代码中写死的 API Key。

    若仍为占位字符串或为空，直接报错提示去改 LLM_API_KEY。
    """
    key = (LLM_API_KEY or "").strip()
    if not key or key == "在此填入你的密钥":
        raise RuntimeError(
            "请先在 src/assist/llm_review.py 中把 LLM_API_KEY 改成你的真实密钥。"
        )
    return key


def build_detection_summary(
    final_results: list[dict[str, Any]],
    hard_examples: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    汇总检测结果，供 LLM 质检/写报告。

    汇总内容包括:
        - 全局类别计数、SAM 精炼框数量
        - 难例清单（截断前 50 张）
        - 每图统计（截断前 100 张）
    """
    class_counter: Counter[str] = Counter()
    total_dets = 0
    sam_refined = 0
    per_image: list[dict[str, Any]] = []

    for item in final_results:
        local: Counter[str] = Counter()
        local_sam = 0
        for d in item.get("detections", []):
            name = str(d.get("class_name", d.get("class_id")))
            local[name] += 1
            class_counter[name] += 1
            total_dets += 1
            if d.get("refined_by") == "sam":
                sam_refined += 1
                local_sam += 1
        per_image.append(
            {
                "image": _basename(item["image"]),
                "count": item.get("count", len(item.get("detections", []))),
                "class_counts": dict(local),
                "sam_refined": local_sam,
                "hard_score": item.get("hard_score", 0.0),
                "hard_reason": item.get("hard_reason", ""),
            }
        )

    return {
        "num_images": len(final_results),
        "total_detections": total_dets,
        "class_totals": dict(class_counter),
        "sam_refined_boxes": sam_refined,
        "num_hard_images": len(hard_examples),
        "hard_images": [
            {
                "image": _basename(h["image"]),
                "hard_score": h.get("hard_score", 0.0),
                "reason": h.get("reason", ""),
            }
            for h in hard_examples[:50]
        ],
        "per_image": per_image[:100],
    }


def _extract_json_object(text: str) -> dict[str, Any]:
    """
    从模型回复中提取 JSON 对象。

    兼容：
        - 纯 JSON
        - ```json ... ``` 代码块包裹
    """
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence:
        return json.loads(fence.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("LLM 回复中未找到可解析的 JSON 对象")


def review_with_llm(
    final_results: list[dict[str, Any]],
    hard_examples: list[dict[str, Any]],
    base_url: str = "https://api.deepseek.com",
    model: str = "deepseek-chat",
    temperature: float = 0.2,
    timeout: int = 120,
) -> dict[str, Any]:
    """
    调用 LLM，一次生成质检 JSON 与中文报告。

    参数:
        final_results: 合并后的最终检测结果
        hard_examples: 难例清单
        base_url: API 根地址（DeepSeek 或 OpenAI）
        model: 模型名
        temperature: 采样温度（质检建议偏低）
        timeout: HTTP 超时秒数

    返回:
        {
          "qc": {...},           # 质检结构化结果
          "report_md": "...",    # 中文 Markdown 报告
          "raw": "...",          # 模型原始回复
          "summary_sent": {...}  # 实际发给模型的摘要
        }
    """
    api_key = resolve_api_key()
    summary = build_detection_summary(final_results, hard_examples)

    system_prompt = (
        "你是田间视觉检测质检助手。任务：根据结构化检测结果做质检，并写中文报告。\n"
        "约束：\n"
        "1. 不要修改或编造检测框坐标与类别；\n"
        "2. 类别语境为 maize（玉米苗）、broadleaf（阔叶杂草）、grass（禾本科杂草）；\n"
        "3. 必须严格输出一个 JSON 对象，字段仅有 qc 与 report_md；\n"
        "4. qc 含：suspicious_images（数组，含 image/risk/suggestion）、"
        "risk_points（字符串数组）、review_suggestions（字符串数组）；\n"
        "5. report_md 为中文 Markdown，含检测概况、类别统计、风险与农事建议摘要。"
    )
    user_prompt = (
        "以下是本批次田间检测结构化摘要（JSON）。请完成质检与报告：\n"
        + json.dumps(summary, ensure_ascii=False, indent=2)
    )

    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    print(f"[LLM] 请求 {model} @ {base_url}")
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if resp.status_code >= 400:
        raise RuntimeError(f"LLM API 失败 HTTP {resp.status_code}: {resp.text[:800]}")

    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"LLM 响应结构异常: {data}") from exc

    parsed = _extract_json_object(content)
    if "qc" not in parsed or "report_md" not in parsed:
        raise ValueError("LLM JSON 缺少必要字段 qc / report_md")
    if not isinstance(parsed["qc"], dict) or not isinstance(parsed["report_md"], str):
        raise ValueError("LLM JSON 字段类型错误：qc 应为对象，report_md 应为字符串")

    return {
        "qc": parsed["qc"],
        "report_md": parsed["report_md"],
        "raw": content,
        "summary_sent": summary,
    }
