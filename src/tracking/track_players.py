"""
Stage 6 - Multi-Object Tracking with ByteTrack / BoT-SORT

Uses your fine-tuned YOLO model + Ultralytics' tracker to follow each
player across frames, assigning a persistent track ID.

Usage:
    python src/tracking/track_players.py
    python src/tracking/track_players.py --video "data/ball_action/.../1_720p.mkv" --output-prefix 720p_ --start 120 --duration 15
"""

from pathlib import Path
import sys
import json
import argparse
import cv2

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger
from ultralytics import YOLO

logger = get_logger(__name__)

MODEL_PATH = "models/yolov8_football_finetune_v1.pt"
DEFAULT_VIDEO = "data/soccernet/videos-224p/england_epl/2016-2017/2016-08-14 - 18-00 Arsenal 3 - 4 Liverpool/1_224p.mkv"
OUTPUT_DIR = Path("data/analysis/tracking")
FPS = 25.0  # confirmed uniform across your dataset in Stage 3


def run_tracking(video_path: str, tracker_config: str, start_sec: float, duration_sec: float):
    """Run tracking and return structured per-frame, per-track data."""
    model = YOLO(MODEL_PATH)

    start_frame = int(start_sec * FPS)
    num_frames = int(duration_sec * FPS)

    logger.info(f"Tracking with {tracker_config} | video={Path(video_path).name} | frames {start_frame} to {start_frame + num_frames}")

    all_records = []
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for frame_offset in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # iou=0.5 (tightened NMS) + tuned tracker config (track_buffer=75) -
        # the settings that fixed both the duplicate-box and ID-switch
        # problems found during Stage 6 evaluation.
        results = model.track(frame, tracker=tracker_config, persist=True, verbose=False, iou=0.5)[0]

        if results.boxes.id is None:
            continue

        for box, track_id, conf, cls in zip(
            results.boxes.xyxy, results.boxes.id, results.boxes.conf, results.boxes.cls
        ):
            class_name = model.names[int(cls)]
            if class_name != "Player":
                continue

            x1, y1, x2, y2 = box.tolist()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            all_records.append({
                "frame": start_frame + frame_offset,
                "timestamp_sec": round((start_frame + frame_offset) / FPS, 3),
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="Path to video file")
    parser.add_argument("--output-prefix", default="", help="Prefix for output filenames, e.g. '720p_'")
    parser.add_argument("--start", type=float, default=600.0, help="Start timestamp in seconds")
    parser.add_argument("--duration", type=float, default=15.0, help="Duration in seconds")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trackers = {
        "bytetrack": "configs/trackers/bytetrack_tuned.yaml",
        "botsort": "configs/trackers/botsort_tuned.yaml",
    }

    for name, config in trackers.items():
        print(f"\n{'='*20} {name.upper()} {'='*20}")
        records = run_tracking(args.video, config, args.start, args.duration)

        output_path = OUTPUT_DIR / f"{args.output_prefix}{name}_raw_records.json"
        with open(output_path, "w") as f:
            json.dump(records, f, indent=2)

        print(f"Saved {len(records)} tracking records to: {output_path.resolve()}")
        unique_ids = sorted(set(r["track_id"] for r in records))
        print(f"Unique track IDs: {len(unique_ids)}")


if __name__ == "__main__":
    main()
