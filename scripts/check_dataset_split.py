# -*- coding: utf-8 -*-
"""Validate a YOLO dataset split before training.

The checker intentionally uses content hashes instead of filenames so an exact
copy renamed between train/val/test is still detected. Near-duplicate burst
frames require a domain-specific perceptual check and human review.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.common import list_images, load_yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_dataset(data_yaml: str | Path) -> dict[str, Any]:
    """Return split statistics, exact overlaps, and missing-label errors."""
    yaml_path = Path(data_yaml).resolve()
    cfg = load_yaml(yaml_path)
    root = Path(cfg.get("path", "."))
    if not root.is_absolute():
        # Ultralytics resolves the dataset path from the current working
        # directory for this project configuration.
        root = (Path.cwd() / root).resolve()

    split_hashes: dict[str, dict[str, Path]] = {}
    missing_labels: list[str] = []
    counts: dict[str, int] = {}

    for split in ("train", "val", "test"):
        rel = cfg.get(split)
        if not rel:
            continue
        image_dir = root / str(rel)
        images = list_images(image_dir)
        counts[split] = len(images)
        split_hashes[split] = {_sha256(path): path for path in images}
        for image in images:
            relative = image.relative_to(image_dir)
            label = root / "labels" / split / relative.with_suffix(".txt")
            if not label.is_file():
                missing_labels.append(str(image))

    overlaps: list[dict[str, str]] = []
    names = list(split_hashes)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            common = set(split_hashes[left]) & set(split_hashes[right])
            for digest in sorted(common):
                overlaps.append(
                    {
                        "left_split": left,
                        "left_image": str(split_hashes[left][digest]),
                        "right_split": right,
                        "right_image": str(split_hashes[right][digest]),
                        "sha256": digest,
                    }
                )

    return {
        "counts": counts,
        "overlaps": overlaps,
        "missing_labels": missing_labels,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check YOLO dataset split hygiene")
    parser.add_argument("--data", default="configs/field.yaml", help="dataset YAML")
    args = parser.parse_args()
    report = inspect_dataset(args.data)

    print("[dataset] " + ", ".join(f"{k}={v}" for k, v in report["counts"].items()))
    for item in report["overlaps"]:
        print(
            f"[error] duplicate across {item['left_split']}/{item['right_split']}: "
            f"{item['left_image']} == {item['right_image']}"
        )
    for image in report["missing_labels"]:
        print(f"[error] missing label: {image}")

    if report["overlaps"] or report["missing_labels"]:
        return 1
    if any(count == 0 for count in report["counts"].values()):
        print("[warning] one or more splits are empty; add independently collected data")
    print("[ok] no exact cross-split duplicate or missing label found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
