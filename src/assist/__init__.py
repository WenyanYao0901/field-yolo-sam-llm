# -*- coding: utf-8 -*-
"""
辅助子包。

- hard_mining : 难例打分，筛出优先复核样本
- sam_annotate: 视觉大模型 SAM，精炼难例/不确定框
- llm_review  : 语言大模型 DeepSeek/GPT，质检与中文报告

说明:
    辅助流程不替代线上 YOLO 推理；最终验收指标仍以 YOLO 为准。
"""
