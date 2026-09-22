from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.check_secrets import scan_paths


class SecretScanTests(unittest.TestCase):
    def test_detects_likely_token_without_exposing_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            safe = Path(tmp) / "safe.py"
            unsafe = Path(tmp) / "unsafe.py"
            safe.write_text("LLM_API_KEY = getenv('LLM_API_KEY')", encoding="utf-8")
            unsafe.write_text("token = 's" + "k-exampletoken1234567890'", encoding="utf-8")
            self.assertEqual(scan_paths([safe, unsafe]), [unsafe])


if __name__ == "__main__":
    unittest.main()
