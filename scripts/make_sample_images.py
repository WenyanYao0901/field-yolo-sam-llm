# -*- coding: utf-8 -*-
"""
生成田间风格示例图，便于本地跑通 detect / run 流程。

说明:
    - 图像为程序合成（土壤底 + 秸秆纹理 + 苗/草色块），不是真实田间拍摄
    - 用途是验证「读图 -> 落盘 -> 流水线编排」是否通畅
    - 若使用官方 COCO 预训练 YOLO，类别未必对得上 maize/broadleaf/grass；
      完整业务检测请换田间微调权重
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _soil_canvas(h: int, w: int, seed: int, brightness: float = 1.0) -> np.ndarray:
    """生成带噪声的土壤/麦茬背景（BGR）。"""
    rng = np.random.default_rng(seed)
    # 棕色土壤底
    base = np.zeros((h, w, 3), dtype=np.uint8)
    base[:, :, 0] = int(35 * brightness)   # B
    base[:, :, 1] = int(70 * brightness)   # G
    base[:, :, 2] = int(95 * brightness)   # R
    noise = rng.integers(-25, 26, size=(h, w, 3), dtype=np.int16)
    img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 稀疏秸秆短线，模拟麦茬干扰
    for _ in range(180):
        x1 = int(rng.integers(0, w))
        y1 = int(rng.integers(0, h))
        x2 = int(np.clip(x1 + rng.integers(-40, 41), 0, w - 1))
        y2 = int(np.clip(y1 + rng.integers(-8, 9), 0, h - 1))
        color = (
            int(rng.integers(40, 80)),
            int(rng.integers(90, 140)),
            int(rng.integers(120, 180)),
        )
        cv2.line(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    return img


def _draw_maize(img: np.ndarray, cx: int, cy: int, scale: float, rng: np.random.Generator) -> None:
    """画近似玉米苗：竖直叶片簇。"""
    color = (
        int(rng.integers(20, 50)),
        int(rng.integers(120, 190)),
        int(rng.integers(40, 90)),
    )
    for dx in (-10, -4, 2, 8):
        tip = (int(cx + dx * scale), int(cy - 55 * scale))
        base = (int(cx + dx * 0.2 * scale), int(cy + 20 * scale))
        cv2.line(img, base, tip, color, max(2, int(3 * scale)), cv2.LINE_AA)
    # 茎部小椭圆
    cv2.ellipse(
        img,
        (cx, int(cy + 10 * scale)),
        (int(8 * scale), int(14 * scale)),
        0,
        0,
        360,
        color,
        -1,
        cv2.LINE_AA,
    )


def _draw_broadleaf(img: np.ndarray, cx: int, cy: int, scale: float, rng: np.random.Generator) -> None:
    """画近似阔叶杂草：圆叶团块。"""
    color = (
        int(rng.integers(10, 40)),
        int(rng.integers(140, 210)),
        int(rng.integers(30, 80)),
    )
    for _ in range(5):
        ox = int(rng.integers(-18, 19) * scale)
        oy = int(rng.integers(-14, 15) * scale)
        axes = (int(rng.integers(12, 22) * scale), int(rng.integers(9, 16) * scale))
        cv2.ellipse(img, (cx + ox, cy + oy), axes, int(rng.integers(0, 180)), 0, 360, color, -1, cv2.LINE_AA)


def _draw_grass(img: np.ndarray, cx: int, cy: int, scale: float, rng: np.random.Generator) -> None:
    """画近似禾本科杂草：细长散叶。"""
    color = (
        int(rng.integers(15, 45)),
        int(rng.integers(100, 160)),
        int(rng.integers(40, 90)),
    )
    for _ in range(8):
        ang = float(rng.uniform(-0.7, 0.7))
        length = int(rng.integers(35, 70) * scale)
        x2 = int(cx + length * np.sin(ang))
        y2 = int(cy - length * np.cos(ang))
        cv2.line(img, (cx, cy), (x2, y2), color, 1, cv2.LINE_AA)


def _compose(
    name: str,
    h: int,
    w: int,
    seed: int,
    plants: list[tuple[str, int, int, float]],
    brightness: float = 1.0,
) -> np.ndarray:
    """按植物列表合成一张示例图。"""
    rng = np.random.default_rng(seed)
    img = _soil_canvas(h, w, seed, brightness=brightness)
    drawers = {
        "maize": _draw_maize,
        "broadleaf": _draw_broadleaf,
        "grass": _draw_grass,
    }
    for kind, cx, cy, scale in plants:
        drawers[kind](img, cx, cy, scale, rng)
    # 角落标注场景名，方便人工对照（不影响流程）
    cv2.putText(
        img,
        name,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    return img


def generate_samples(out_dir: str | Path, width: int = 640, height: int = 480) -> list[Path]:
    """
    生成一组覆盖不同场景的示例图。

    返回:
        写出的文件路径列表
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    specs = [
        (
            "sample_01_mixed",
            11,
            1.0,
            [
                ("maize", 160, 260, 1.2),
                ("maize", 280, 240, 1.0),
                ("broadleaf", 420, 300, 1.1),
                ("grass", 520, 280, 1.0),
            ],
        ),
        (
            "sample_02_maize_heavy",
            22,
            1.0,
            [
                ("maize", 120, 250, 1.3),
                ("maize", 220, 230, 1.1),
                ("maize", 330, 260, 1.2),
                ("maize", 450, 240, 1.0),
                ("grass", 540, 320, 0.8),
            ],
        ),
        (
            "sample_03_weed_heavy",
            33,
            1.0,
            [
                ("broadleaf", 140, 280, 1.2),
                ("broadleaf", 260, 300, 1.0),
                ("grass", 360, 260, 1.1),
                ("grass", 460, 290, 1.0),
                ("grass", 540, 250, 0.9),
                ("maize", 200, 200, 0.9),
            ],
        ),
        (
            "sample_04_occlusion",
            44,
            0.95,
            [
                ("maize", 300, 250, 1.4),
                ("broadleaf", 320, 270, 1.3),
                ("grass", 290, 280, 1.2),
            ],
        ),
        (
            "sample_05_bright",
            55,
            1.35,
            [
                ("maize", 180, 240, 1.1),
                ("broadleaf", 400, 280, 1.0),
                ("grass", 500, 260, 1.0),
            ],
        ),
        (
            "sample_06_sparse",
            66,
            1.0,
            [
                ("maize", 320, 250, 1.0),
            ],
        ),
    ]

    saved: list[Path] = []
    for name, seed, bright, plants in specs:
        img = _compose(name, height, width, seed, plants, brightness=bright)
        path = out / f"{name}.jpg"
        # 高质量 JPEG，便于目视检查
        cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        saved.append(path)
        print(f"[示例图] 已生成: {path}")
    return saved


def main() -> None:
    """命令行入口：默认输出到 data/raw。"""
    parser = argparse.ArgumentParser(description="生成田间风格示例图")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/raw",
        help="输出目录（相对项目根或绝对路径）",
    )
    parser.add_argument("--width", type=int, default=640, help="图像宽度")
    parser.add_argument("--height", type=int, default=480, help="图像高度")
    args = parser.parse_args()

    # 若相对路径，则以本脚本所在项目根为基准
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        root = Path(__file__).resolve().parents[1]
        out_dir = root / out_dir

    paths = generate_samples(out_dir, width=args.width, height=args.height)
    print(f"[示例图] 共 {len(paths)} 张 -> {out_dir}")


if __name__ == "__main__":
    main()
