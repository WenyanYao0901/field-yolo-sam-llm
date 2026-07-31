# -*- coding: utf-8 -*-
"""
把已生成的示例图安装到 data/raw。

优先顺序:
    1. 若 Cursor assets 目录已有 sample_*.png，则复制过来
    2. 否则调用 make_sample_images 生成本地合成图

用法（在项目根目录）:
    python3 scripts/install_sample_images.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / "data" / "raw"

# 本会话生成的示例图所在目录（Cursor assets）
ASSETS = Path(
    "/Users/hanchongchong/.cursor/projects/Users-hanchongchong-Desktop/assets"
)

SAMPLE_NAMES = [
    "sample_01_mixed.png",
    "sample_02_maize_heavy.png",
    "sample_03_weed_heavy.png",
    "sample_04_occlusion.png",
    "sample_05_bright.png",
    "sample_06_sparse.png",
]


def main() -> None:
    """安装示例图到 data/raw。"""
    DST.mkdir(parents=True, exist_ok=True)
    copied = 0
    for name in SAMPLE_NAMES:
        src = ASSETS / name
        if src.exists():
            shutil.copy2(src, DST / name)
            print(f"[安装] 复制 {name}")
            copied += 1

    if copied == len(SAMPLE_NAMES):
        print(f"[完成] 已安装 {copied} 张示例图 -> {DST}")
        return

    # assets 不齐时，退回程序合成图（需 opencv）
    print(f"[提示] assets 仅找到 {copied} 张，改为本地合成...")
    from scripts.make_sample_images import generate_samples

    generate_samples(DST)
    print(f"[完成] 示例图已就绪 -> {DST}")


if __name__ == "__main__":
    main()
