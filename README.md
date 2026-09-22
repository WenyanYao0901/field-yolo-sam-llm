# field-yolo-sam-llm

Hard-example-driven closed-loop pipeline for **field seedling crop–weed detection** in precision agriculture.

> **Research status:** this is a reproducibility-oriented research prototype,
> not a production agronomic decision system. The repository currently contains
> no validated field benchmark or publishable quantitative result.

## Overview

This repository implements a decoupled YOLO + SAM + LLM workflow:

- **YOLO**: real-time detection (class, confidence, bounding boxes)
- **SAM**: refines uncertain / hard examples only (box-prompt segmentation)
- **LLM**: structured quality review and Chinese agronomic report (does not modify detections)

Target classes: maize seedling, broadleaf weed, grass weed.

> For field three-class detection, use your fine-tuned `best.pt`. The default `yolov8s.pt` is COCO pretrained and is only for smoke-testing the pipeline.

## Requirements

- Python 3.10+
- See `requirements.txt`
- Download YOLO / SAM weights locally (not included in repo)
- LLM API key via environment variable (never commit secrets):

```bash
export LLM_API_KEY='your_key'
# or
export DEEPSEEK_API_KEY='your_key'
```

All relative input, weight, and output paths are resolved from the current
working directory. Run commands from the repository root for reproducible
results.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Place weights locally, e.g. yolov8s.pt and weights/sam_vit_b_01ec64.pth
export LLM_API_KEY='your_key'

# Full pipeline（默认读取 configs/default.yaml，命令行可覆盖）
python main.py run --source data/raw

# 仅 YOLO + 难例，跳过 SAM / LLM
python main.py run --source data/raw --skip-sam --skip-llm

# YOLO detect only
python main.py detect --source data/raw

# Train YOLO（需补齐 data/images 与 data/labels）
python scripts/check_dataset_split.py --data configs/field.yaml
python main.py train --data configs/field.yaml

# Run lightweight unit tests (no model/API call required)
python -m unittest discover -s tests -v
```

## Project structure

```
configs/          dataset and default configs
src/detect/       YOLO inference
src/train/        YOLO training
src/assist/       SAM hard-mining, LLM review
src/pipeline/     end-to-end run
scripts/          sample image utilities
docs/             manuscript draft (in preparation)
```

## Dataset policy

The repository does not distribute the field dataset. Put locally cleared data
under `data/images/{train,val,test}` and matching YOLO labels under
`data/labels/{train,val,test}`. These directories are ignored by Git.

Split by field, date, or acquisition batch before training. Do not place the
same image, re-encoded copy, or adjacent burst frames in different splits. The
provided checker detects exact cross-split duplicates and missing labels:

```bash
python scripts/check_dataset_split.py --data configs/field.yaml
```

## Status

- Research prototype developed by Wenyan Yao (Liaocheng University)
- Project created: **2026-07-31**
- Related journal manuscript **in preparation** (not yet published)
- Field dataset and full quantitative evaluation are ongoing

## Contributions

- **Wenyan Yao:** research conception, agricultural problem definition,
  pipeline design, manuscript preparation, and project stewardship.
- **OpenAI Codex (under the author's direction):** implementation review,
  configuration/CLI integration, robustness fixes, tests, security hygiene,
  and documentation improvements. All generated changes remain subject to
  author review and repository licensing.

## Research-use boundaries

- YOLO detections, SAM refinements, and LLM reports require independent field
  validation before agronomic or operational use.
- Synthetic sample images are only for software smoke tests.
- Manuscript placeholders must never be reported as measured results.
- See [docs/PUBLICATION_STATUS.md](docs/PUBLICATION_STATUS.md) before
  redistributing manuscripts, images, datasets, or model weights.

## License

The repository source code and original software diagrams are licensed under
the [Apache License 2.0](LICENSE). Third-party weights, datasets, services, and
unpublished manuscript rights are excluded unless separately stated.

## Author

- Wenyan Yao — Agricultural Engineering / Precision Agriculture
- Liaocheng University
- Contact: [GitHub Issues](https://github.com/WenyanYao0901/field-yolo-sam-llm/issues)
- Repository: https://github.com/WenyanYao0901/field-yolo-sam-llm
