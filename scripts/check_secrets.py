# -*- coding: utf-8 -*-
"""Fail when likely API credentials appear in tracked text files."""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


# Construct the prefix in pieces so the scanner does not flag its own source.
TOKEN_PATTERN = re.compile("s" + "k-" + r"[A-Za-z0-9_-]{16,}")
TEXT_SUFFIXES = {
    "",
    ".cff",
    ".json",
    ".md",
    ".py",
    ".txt",
    ".yaml",
    ".yml",
}


def scan_paths(paths: list[Path]) -> list[Path]:
    """Return text files containing likely provider tokens without printing them."""
    hits: list[Path] = []
    for path in paths:
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if TOKEN_PATTERN.search(text):
            hits.append(path)
    return hits


def tracked_paths(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [root / item.decode() for item in result.stdout.split(b"\0") if item]


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan tracked text for API tokens")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    hits = scan_paths(tracked_paths(root))
    if hits:
        for path in hits:
            print(f"[error] possible credential: {path.relative_to(root)}")
        return 1
    print("[ok] no likely API token found in tracked text files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
