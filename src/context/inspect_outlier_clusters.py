"""
Stage 7 - Spot-check DBSCAN's small clusters: are track IDs 40, 185, 91, 108
actually goalkeepers (distinct kit color from their own team)?

Usage:
    python src/context/inspect_outlier_clusters.py
"""

import json
from pathlib import Path
from collections import defaultdict
import sys
import cv2

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.pipeline.video_pipeline import VideoPipeline

TRACKS_PATH = Path("data/analysis/tracking/720p_bytetrack_raw_records.json")
VIDEO = "data/ball_action/spotting-ball-2023/valid/england_efl/2019-2020/2019-10-01 - Middlesbrough - Preston North End/1_720p.mkv"
OUTPUT_DIR = Path("data/analysis/context/outlier_crops")

# Checking the goalkeeper-identification candidates
CLUSTER_2 = [110]  # smallest positional range by far - strongest position signal
CLUSTER_3 = [65]   # flagged by BOTH signals (range + DBSCAN outlier)


def save_crop(pipeline, by_track, track_id, label):
    if track_id not in by_track:
        print(f"Track {track_id} not found in data.")
        return
    record = by_track[track_id][0]
    frame = pipeline.get_frame(record["frame"])
    if frame is None:
        return
    x1, y1, x2, y2 = [int(v) for v in record["bbox"]]
    crop = frame[max(0, y1):y2, max(0, x1):x2]
    if crop.size > 0:
        upscaled = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(OUTPUT_DIR / f"{label}_track{track_id}.jpg"), upscaled)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(TRACKS_PATH) as f:
        records = json.load(f)
    by_track = defaultdict(list)
    for r in records:
        by_track[r["track_id"]].append(r)

    pipeline = VideoPipeline(VIDEO)
    pipeline.load()
    pipeline.extract_metadata()

    for tid in CLUSTER_2:
        save_crop(pipeline, by_track, tid, "cluster2")
    for tid in CLUSTER_3:
        save_crop(pipeline, by_track, tid, "cluster3")

    pipeline.release()
    print(f"Saved crops to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
