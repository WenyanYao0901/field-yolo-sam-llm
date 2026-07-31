# field-yolo-sam-llm

Hard-example-driven closed-loop pipeline for **field seedling crop–weed detection** in precision agriculture.

## Overview

This repository implements a decoupled YOLO + SAM + LLM workflow:

- **YOLO**: real-time detection (class, confidence, bounding boxes)
- **SAM**: refines uncertain / hard examples only (box-prompt segmentation)
- **LLM**: structured quality review and Chinese agronomic report (does not modify detections)

Target classes: maize seedling, broadleaf weed, grass weed.

## Requirements

- Python 3.10+
- See `requirements.txt`
- Download YOLO / SAM weights locally (not included in repo)

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Place weights locally, e.g. yolov8s.pt and weights/sam_vit_b_01ec64.pth

# Full pipeline
python main.py run --source data/raw

# YOLO detect only
python main.py detect --source data/raw

# Train YOLO
python main.py train --data configs/field.yaml
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

## Status

- Research prototype developed by Wenyan Yao (Liaocheng University)
- Related journal manuscript **in preparation** (not yet published)
- Field dataset and full quantitative evaluation are ongoing

## License

TBD. Contact the author before commercial use.

## Author

Wenyan Yao — Agricultural Engineering / Precision Agriculture
