# Sports Video Analytics Pipeline

An end-to-end computer vision pipeline that ingests raw match video and
detects, tracks, identifies, and spots key events (goals, fouls, passes, etc.)
with precise timestamps.

## Pipeline Phases

| Phase | Description |
|---|---|
| 0 | Project setup |
| 1 | Video ingestion |
| 2 | Frame sampling + timestamps |
| 3 | YOLO object detection |
| 4 | ByteTrack / BoT-SORT tracking |
| 5 | Player/team identification |
| 6 | Temporal data preparation |
| 7 | Baseline action spotting |
| 8 | T-DEED (precise event localization) |
| 9 | Event disambiguation |
| 10 | Evaluation / VisionScore |
| 11 | Optimization |
| 12 | TensorRT FP16 |
| 13 | FastAPI serving |
| 14 | Docker deployment |

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configuration

All paths, thresholds, and model settings live in `configs/config.yaml`.

## Logging

All modules should use the shared logger:

```python
from src.utils.logger import get_logger
logger = get_logger(__name__)
```

## Status

Work in progress -- currently on Phase 1 (video ingestion).
