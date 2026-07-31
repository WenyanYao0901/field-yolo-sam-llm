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


def build_parser() -> argparse.ArgumentParser:
    """
    构建命令行参数解析器。

    说明:
        各子命令参数尽量与 configs/default.yaml 对齐，
        便于脚本调用与配置文件两种用法并存。
    """
    parser = argparse.ArgumentParser(
        description="田间 YOLO + SAM + LLM 闭环检测"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---------- 闭环：业务主入口 ----------
    p_run = sub.add_parser("run", help="一键闭环：YOLO + SAM + LLM")
    p_run.add_argument("--source", type=str, default="data/raw", help="图像路径或目录")
    p_run.add_argument("--weights", type=str, default="yolov8s.pt", help="YOLO 权重")
    p_run.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    p_run.add_argument("--iou", type=float, default=0.45, help="NMS IoU")
    p_run.add_argument("--imgsz", type=int, default=640, help="推理尺寸")
    p_run.add_argument("--device", type=str, default="", help="设备，空则自动")
    p_run.add_argument(
        "--save-dir", type=str, default="runs/pipeline/run", help="输出目录"
    )
    p_run.add_argument(
        "--sam-checkpoint",
        type=str,
        default="weights/sam_vit_b_01ec64.pth",
        help="SAM 权重路径",
    )
    p_run.add_argument("--sam-type", type=str, default="vit_b", help="SAM 结构")
    p_run.add_argument("--conf-low", type=float, default=0.25, help="不确定区间下界")
    p_run.add_argument("--conf-high", type=float, default=0.6, help="不确定区间上界")
    p_run.add_argument("--thresh", type=float, default=0.45, help="难例分数阈值")
    p_run.add_argument(
        "--llm-base-url",
        type=str,
        default="https://api.deepseek.com",
        help="LLM API 地址（DeepSeek 或 OpenAI）",
    )
    p_run.add_argument(
        "--llm-model",
        type=str,
        default="deepseek-chat",
        help="LLM 模型名，如 deepseek-chat / gpt-4o-mini",
    )
    p_run.add_argument("--llm-temperature", type=float, default=0.2, help="LLM 温度")

    # ---------- 仅检测 ----------
    p_det = sub.add_parser("detect", help="仅 YOLO 检测")
    p_det.add_argument("--source", type=str, default="data/raw", help="图像路径或目录")
    p_det.add_argument("--weights", type=str, default="yolov8s.pt", help="YOLO 权重")
    p_det.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    p_det.add_argument("--iou", type=float, default=0.45, help="NMS IoU")
    p_det.add_argument("--imgsz", type=int, default=640, help="推理尺寸")
    p_det.add_argument("--device", type=str, default="", help="设备，空则自动")
    p_det.add_argument("--save-dir", type=str, default="runs/detect", help="输出目录")

    # ---------- 训练 ----------
    p_train = sub.add_parser("train", help="用标注数据训练 YOLO")
    p_train.add_argument("--data", type=str, default="configs/field.yaml", help="数据集配置")
    p_train.add_argument("--weights", type=str, default="yolov8s.pt", help="预训练起点")
    p_train.add_argument("--epochs", type=int, default=50, help="训练轮数")
    p_train.add_argument("--batch", type=int, default=8, help="batch size")
    p_train.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    p_train.add_argument("--device", type=str, default="", help="设备，空则自动")
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
    parser = build_parser()
    args = parser.parse_args()

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
        from pathlib import Path

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
