"""
Stage 7 - Diagnostic: save actual torso crop images so we can SEE what
the team classifier is working with, rather than just trusting the numbers.

Usage:
    python src/context/inspect_crops.py
"""

import json
from pathlib import Path
from collections import defaultdict
import sys
import cv2

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger
from src.pipeline.video_pipeline import VideoPipeline

logger = get_logger(__name__)

TRACKS_PATH = Path("data/analysis/tracking/bytetrack_raw_records.json")

# Testing resolution theory: does 720p give usable crops where 224p didn't?
# Reusing the already-downloaded Ball Action Spotting 720p file - this is a
# DIFFERENT match than the tracks above, so this run is purely a visual
# resolution test, not matched to specific track IDs from before.
TEST_VIDEO = "data/ball_action/spotting-ball-2023/valid/england_efl/2019-2020/2019-10-01 - Middlesbrough - Preston North End/1_720p.mkv"
OUTPUT_DIR = Path("data/analysis/context/crop_samples_720p")

# Arbitrary sample timestamps - just need SOME players to crop from a video
# we don't have existing tracking data for, so run a fresh quick detection pass
TEST_TIMESTAMPS = [120.0, 180.0, 240.0]


def save_crop_pair(frame, bbox, label, index):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    full_crop = frame[max(0, y1):y2, max(0, x1):x2]

    box_height, box_width = y2 - y1, x2 - x1
    torso_y1 = y1 + int(box_height * 0.15)
    torso_y2 = y1 + int(box_height * 0.55)
    torso_x1 = x1 + int(box_width * 0.20)
    torso_x2 = x1 + int(box_width * 0.80)
    torso_crop = frame[max(0, torso_y1):torso_y2, max(0, torso_x1):torso_x2]

    if full_crop.size > 0:
        full_upscaled = cv2.resize(full_crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(OUTPUT_DIR / f"{label}_{index}_full.jpg"), full_upscaled)
    if torso_crop.size > 0:
        torso_upscaled = cv2.resize(torso_crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(OUTPUT_DIR / f"{label}_{index}_torso.jpg"), torso_upscaled)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO("models/yolov8_football_finetune_v1.pt")

    pipeline = VideoPipeline(TEST_VIDEO)
    pipeline.load()
    pipeline.extract_metadata()

    crop_count = 0
    for ts in TEST_TIMESTAMPS:
        frame = pipeline.get_frame_at_timestamp(ts)
        if frame is None:
            continue

        results = model(frame, verbose=False)[0]
        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            if model.names[int(cls)] != "Player":
                continue
            save_crop_pair(frame, box.tolist(), "720p", crop_count)
            crop_count += 1
            if crop_count >= 8:
                break
        if crop_count >= 8:
            break

    pipeline.release()
    print(f"\nSaved {crop_count} crop pairs to: {OUTPUT_DIR.resolve()}")
    print("Open the '_full' and '_torso' images - are jersey colors actually")
    print("distinguishable now, compared to the 224p crops?")


if __name__ == "__main__":
    main()
