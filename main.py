# -*- coding: utf-8 -*-
"""
田间目标检测统一命令行入口。

架构叙事（论文方案 A 扩展）:
    YOLO     : 实时检测（位置 + 类别 + 置信度），验收主体
    SAM      : 视觉大模型，仅精炼难例/不确定框，提升定位质量
    LLM      : DeepSeek / GPT，基于结构化结果做质检与中文报告

子命令:
    run           一键闭环：YOLO -> 难例 -> SAM -> LLM
    detect        仅 YOLO 检测
    train         用标注数据训练 YOLO
    make-samples  生成田间风格示例图到 data/raw
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.common import flatten_default_config


def build_parser(defaults: dict | None = None) -> argparse.ArgumentParser:
    """
    构建命令行参数解析器。

    说明:
        默认值优先来自 configs/default.yaml（经 flatten_default_config），
        命令行显式参数可覆盖配置文件。
    """
    d = defaults or {}

    def _get(key: str, fallback):
        return d[key] if key in d else fallback

    parser = argparse.ArgumentParser(
        description="田间 YOLO + SAM + LLM 闭环检测"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="默认参数配置文件（可被命令行覆盖）",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---------- 闭环：业务主入口 ----------
    p_run = sub.add_parser("run", help="一键闭环：YOLO + SAM + LLM")
    p_run.add_argument(
        "--source", type=str, default=_get("source", "data/raw"), help="图像路径或目录"
    )
    p_run.add_argument(
        "--weights",
        type=str,
        default=_get("weights", "yolov8s.pt"),
        help="YOLO 权重；田间三类检测请用自训 best.pt",
    )
    p_run.add_argument(
        "--conf", type=float, default=float(_get("conf", 0.25)), help="置信度阈值"
    )
    p_run.add_argument(
        "--iou", type=float, default=float(_get("iou", 0.45)), help="NMS IoU"
    )
    p_run.add_argument(
        "--imgsz", type=int, default=int(_get("imgsz", 640)), help="推理尺寸"
    )
    p_run.add_argument(
        "--device", type=str, default=str(_get("device", "")), help="设备，空则自动"
    )
    p_run.add_argument(
        "--save-dir",
        type=str,
        default=_get("save_dir", "runs/pipeline/run"),
        help="输出目录",
    )
    p_run.add_argument(
        "--sam-checkpoint",
        type=str,
        default=_get("sam_checkpoint", "weights/sam_vit_b_01ec64.pth"),
        help="SAM 权重路径",
    )
    p_run.add_argument(
        "--sam-type", type=str, default=_get("sam_type", "vit_b"), help="SAM 结构"
    )
    p_run.add_argument(
        "--conf-low",
        type=float,
        default=float(_get("conf_low", 0.25)),
        help="不确定区间下界",
    )
    p_run.add_argument(
        "--conf-high",
        type=float,
        default=float(_get("conf_high", 0.6)),
        help="不确定区间上界",
    )
    p_run.add_argument(
        "--thresh",
        type=float,
        default=float(_get("hard_score_thresh", 0.45)),
        help="难例分数阈值",
    )
    p_run.add_argument(
        "--llm-base-url",
        type=str,
        default=_get("llm_base_url", "https://api.deepseek.com"),
        help="LLM API 地址（DeepSeek 或 OpenAI）",
    )
    p_run.add_argument(
        "--llm-model",
        type=str,
        default=_get("llm_model", "deepseek-chat"),
        help="LLM 模型名，如 deepseek-chat / gpt-4o-mini",
    )
    p_run.add_argument(
        "--llm-temperature",
        type=float,
        default=float(_get("llm_temperature", 0.2)),
        help="LLM 温度",
    )
    p_run.add_argument(
        "--skip-sam",
        action=argparse.BooleanOptionalAction,
        default=bool(_get("skip_sam", False)),
        help="跳过 SAM 精炼，仅保留 YOLO 框（可用 --no-skip-sam 关闭）",
    )
    p_run.add_argument(
        "--skip-llm",
        action=argparse.BooleanOptionalAction,
        default=bool(_get("skip_llm", False)),
        help="跳过 LLM 质检与报告（可用 --no-skip-llm 关闭）",
    )

    # ---------- 仅检测 ----------
    p_det = sub.add_parser("detect", help="仅 YOLO 检测")
    p_det.add_argument(
        "--source", type=str, default=_get("source", "data/raw"), help="图像路径或目录"
    )
    p_det.add_argument(
        "--weights",
        type=str,
        default=_get("weights", "yolov8s.pt"),
        help="YOLO 权重；田间三类检测请用自训 best.pt",
    )
    p_det.add_argument(
        "--conf", type=float, default=float(_get("conf", 0.25)), help="置信度阈值"
    )
    p_det.add_argument(
        "--iou", type=float, default=float(_get("iou", 0.45)), help="NMS IoU"
    )
    p_det.add_argument(
        "--imgsz", type=int, default=int(_get("imgsz", 640)), help="推理尺寸"
    )
    p_det.add_argument(
        "--device", type=str, default=str(_get("device", "")), help="设备，空则自动"
    )
    p_det.add_argument("--save-dir", type=str, default="runs/detect", help="输出目录")

    # ---------- 训练 ----------
    p_train = sub.add_parser("train", help="用标注数据训练 YOLO")
    p_train.add_argument("--data", type=str, default="configs/field.yaml", help="数据集配置")
    p_train.add_argument(
        "--weights", type=str, default=_get("weights", "yolov8s.pt"), help="预训练起点"
    )
    p_train.add_argument("--epochs", type=int, default=50, help="训练轮数")
    p_train.add_argument("--batch", type=int, default=8, help="batch size")
    p_train.add_argument(
        "--imgsz", type=int, default=int(_get("imgsz", 640)), help="输入尺寸"
    )
    p_train.add_argument(
        "--device", type=str, default=str(_get("device", "")), help="设备，空则自动"
    )
    p_train.add_argument("--name", type=str, default="field_yolo", help="实验名")

    # ---------- 生成示例图 ----------
    p_samples = sub.add_parser("make-samples", help="生成田间风格示例图到 data/raw")
    p_samples.add_argument(
        "--out-dir",
        type=str,
        default="data/raw",
        help="示例图输出目录",
    )
    p_samples.add_argument("--width", type=int, default=640, help="图像宽度")
    p_samples.add_argument("--height", type=int, default=480, help="图像高度")

    return parser


def main() -> None:
    """解析命令行并分发到对应业务模块。"""
    # 先读 --config（可出现在子命令前后），再用其中默认值构建完整解析器
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default="configs/default.yaml")
    pre_args, remaining = pre.parse_known_args()

    cfg_path = Path(pre_args.config)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).resolve().parent / cfg_path
    defaults = flatten_default_config(cfg_path)
    if defaults:
        print(f"[配置] 已加载 {cfg_path}")

    parser = build_parser(defaults)
    # remaining 已去掉 --config，避免子命令后携带时重复解析报错
    args = parser.parse_args(remaining)
    args.config = pre_args.config

    if args.cmd == "run":
        # 主路径：图片进 -> 最终检测 + 质检报告出
        from src.pipeline.run import run_pipeline

        run_pipeline(
            source=args.source,
            weights=args.weights,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            save_dir=args.save_dir,
            sam_checkpoint=args.sam_checkpoint,
            sam_type=args.sam_type,
            conf_low=args.conf_low,
            conf_high=args.conf_high,
            hard_score_thresh=args.thresh,
            llm_base_url=args.llm_base_url,
            llm_model=args.llm_model,
            llm_temperature=args.llm_temperature,
            skip_sam=args.skip_sam,
            skip_llm=args.skip_llm,
        )
    elif args.cmd == "detect":
        # 仅 YOLO，便于单独检查检测效果
        from src.detect.infer import run_detect

        run_detect(
            source=args.source,
            weights=args.weights,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            save_dir=args.save_dir,
        )
    elif args.cmd == "train":
        # 用标注好的田间数据训练 YOLO
        from src.train.train_yolo import train_yolo

        train_yolo(
            data_yaml=args.data,
            weights=args.weights,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            name=args.name,
        )
    elif args.cmd == "make-samples":
        # 优先安装已生成的真实风格示例图；否则本地合成
        from scripts.install_sample_images import DST, main as install_samples
        from scripts.make_sample_images import generate_samples

        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = Path(__file__).resolve().parent / out_dir

        if out_dir.resolve() == DST.resolve():
            install_samples()
        else:
            generate_samples(out_dir, width=args.width, height=args.height)


if __name__ == "__main__":
    main()
