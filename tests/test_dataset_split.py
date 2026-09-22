from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.check_dataset_split import inspect_dataset


class DatasetSplitTests(unittest.TestCase):
    def _make_layout(self, root: Path) -> Path:
        for split in ("train", "val"):
            (root / "data/images" / split).mkdir(parents=True)
            (root / "data/labels" / split).mkdir(parents=True)
        yaml_path = root / "field.yaml"
        yaml_path.write_text(
            "path: data\ntrain: images/train\nval: images/val\n",
            encoding="utf-8",
        )
        return yaml_path

    def test_detects_renamed_cross_split_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch("pathlib.Path.cwd", return_value=Path(tmp)):
            root = Path(tmp)
            yaml_path = self._make_layout(root)
            (root / "data/images/train/a.png").write_bytes(b"same-image")
            (root / "data/images/val/renamed.png").write_bytes(b"same-image")
            (root / "data/labels/train/a.txt").write_text("", encoding="utf-8")
            (root / "data/labels/val/renamed.txt").write_text("", encoding="utf-8")

            report = inspect_dataset(yaml_path)
            self.assertEqual(len(report["overlaps"]), 1)

    def test_reports_missing_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch("pathlib.Path.cwd", return_value=Path(tmp)):
            root = Path(tmp)
            yaml_path = self._make_layout(root)
            (root / "data/images/train/a.jpg").write_bytes(b"image")

            report = inspect_dataset(yaml_path)
            self.assertEqual(
                report["missing_labels"],
                [str((root / "data/images/train/a.jpg").resolve())],
            )


if __name__ == "__main__":
    unittest.main()
