"""
Stage 6 - Multi-Object Tracking with ByteTrack

Uses your fine-tuned YOLO model + Ultralytics' built-in ByteTrack to follow
each player across frames, assigning a persistent track ID.

Detection (Stage 5): "There is a player here" - one frame, no memory.
Tracking (Stage 6):  "This is player #7, and I've been following him" -
                      across frames, with identity.

Usage:
    python src/tracking/track_players.py
"""

from pathlib import Path
import sys
import json
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger
from ultralytics import YOLO

logger = get_logger(__name__)

MODEL_PATH = "models/yolov8_football_finetune_v1.pt"
TEST_VIDEO = "data/soccernet/videos-224p/england_epl/2016-2017/2016-08-14 - 18-00 Arsenal 3 - 4 Liverpool/1_224p.mkv"
OUTPUT_DIR = Path("data/analysis/tracking")

# How much footage to track - keep SHORT for a first local CPU test.
# 10 seconds at 25fps = 250 frames, which is plenty to see tracking behavior
# without a long CPU-bound wait.
START_SEC = 90.0
DURATION_SEC = 10.0


def run_tracking(tracker_config: str = "bytetrack.yaml"):
    """
    Run tracking and return structured per-frame, per-track data.

    WHY tracker_config is a parameter: this is what lets us later swap in
    'botsort.yaml' with zero other code changes, to directly compare the
    two trackers on the same footage (your checklist's "experiment with
    BoT-SORT" step).
    """
    model = YOLO(MODEL_PATH)
    fps = 25.0  # confirmed uniform across your dataset in Stage 3

    start_frame = int(START_SEC * fps)
    num_frames = int(DURATION_SEC * fps)

    logger.info(f"Tracking with {tracker_config} | frames {start_frame} to {start_frame + num_frames}")

    # model.track() with stream=True processes frame-by-frame and REMEMBERS
    # track identities between calls - this internal memory is what turns
    # detection into tracking. persist=True keeps that memory across the
    # whole loop below.
    all_records = []

    import cv2
    cap = cv2.VideoCapture(TEST_VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for frame_offset in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # iou controls Non-Max Suppression: how much two boxes can overlap
        # before being merged into one. Default (0.7) is permissive - analysis
        # showed most duplicate detections were NOT low-confidence (median
        # conf 0.721), so confidence filtering alone wouldn't fix ghost tracks.
        # Tightening iou forces heavily-overlapping boxes on the same player
        # to collapse into a single detection instead of surviving as two.
        results = model.track(frame, tracker=tracker_config, persist=True, verbose=False, iou=0.5)[0]

        if results.boxes.id is None:
            # No tracks confirmed yet this frame (can happen in the first
            # few frames while the tracker is still initializing)
            continue

        for box, track_id, conf, cls in zip(
            results.boxes.xyxy, results.boxes.id, results.boxes.conf, results.boxes.cls
        ):
            class_name = model.names[int(cls)]
            if class_name != "Player":
                continue  # this stage focuses on PLAYER tracking specifically

            x1, y1, x2, y2 = box.tolist()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            all_records.append({
                "frame": start_frame + frame_offset,
                "timestamp_sec": round((start_frame + frame_offset) / fps, 3),
                "track_id": int(track_id),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "confidence": round(float(conf), 3),
                "center_x": round(center_x, 1),
                "center_y": round(center_y, 1),
            })

    cap.release()
    logger.info(f"Collected {len(all_records)} track records across {len(set(r['track_id'] for r in all_records))} unique track IDs")
    return all_records


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run BOTH trackers on the exact same footage for a fair comparison.
    # botsort.yaml uses appearance (visual) features in addition to motion -
    # may handle close/overlapping players better than ByteTrack's motion-only approach.
    trackers = {
        "bytetrack": "configs/trackers/bytetrack_tuned.yaml",
        "botsort": "configs/trackers/botsort_tuned.yaml",
    }

    for name, config in trackers.items():
        print(f"\n{'='*20} {name.upper()} {'='*20}")
        records = run_tracking(config)

        output_path = OUTPUT_DIR / f"{name}_raw_records.json"
        with open(output_path, "w") as f:
            json.dump(records, f, indent=2)

        print(f"Saved {len(records)} tracking records to: {output_path.resolve()}")

        unique_ids = sorted(set(r["track_id"] for r in records))
        print(f"Unique track IDs: {len(unique_ids)}")


if __name__ == "__main__":
    main()
