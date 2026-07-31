# -*- coding: utf-8 -*-
"""
田间目标检测源码包。

职责划分:
- detect   : YOLO 实时检测（位置 + 类别 + 置信度）
- assist   : SAM 精炼框 + LLM 质检/报告 + 难例打分
- pipeline : 闭环编排（YOLO -> 难例 -> SAM -> LLM）
- utils    : 通用工具函数
"""
