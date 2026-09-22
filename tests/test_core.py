from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.assist.llm_review import _chat_completions_url, _extract_json_object
from src.assist.sam_annotate import yolo_bbox_to_xyxy
from src.utils.common import (
    relative_image_path,
    validate_inference_params,
)


class CommonTests(unittest.TestCase):
    def test_relative_image_path_preserves_nested_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "images"
            image = source / "plot_a" / "frame.jpg"
            image.parent.mkdir(parents=True)
            image.touch()
            self.assertEqual(
                relative_image_path(image, source), Path("plot_a/frame.jpg")
            )

    def test_validate_inference_params(self) -> None:
        validate_inference_params(
            0.25, 0.45, 640, conf_low=0.25, conf_high=0.6,
            hard_score_thresh=0.45,
        )
        with self.assertRaises(ValueError):
            validate_inference_params(1.1, 0.45, 640)
        with self.assertRaises(ValueError):
            validate_inference_params(
                0.25, 0.45, 640, conf_low=0.7, conf_high=0.6
            )


class LlmTests(unittest.TestCase):
    def test_chat_url_accepts_root_or_v1_base(self) -> None:
        expected = "https://example.test/v1/chat/completions"
        self.assertEqual(_chat_completions_url("https://example.test"), expected)
        self.assertEqual(_chat_completions_url("https://example.test/v1/"), expected)

    def test_extract_fenced_json(self) -> None:
        self.assertEqual(_extract_json_object('```json\n{"ok": true}\n```'), {"ok": True})


class SamTests(unittest.TestCase):
    def test_yolo_bbox_to_xyxy(self) -> None:
        self.assertEqual(
            yolo_bbox_to_xyxy([0.5, 0.5, 0.2, 0.4], 100, 200),
            [40.0, 60.0, 60.0, 140.0],
        )


if __name__ == "__main__":
    unittest.main()
