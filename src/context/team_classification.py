"""
Stage 7 - Team Classification

Pipeline: player crop -> visual features -> clustering -> home/away

Usage:
    python src/context/team_classification.py
    python src/context/team_classification.py --video "path/to/720p.mkv" --tracks data/analysis/tracking/720p_bytetrack_raw_records.json
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import sys
import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger
from src.pipeline.video_pipeline import VideoPipeline
from sklearn.cluster import DBSCAN, KMeans

logger = get_logger(__name__)

DEFAULT_TRACKS = Path("data/analysis/tracking/bytetrack_raw_records.json")
DEFAULT_VIDEO = "data/soccernet/videos-224p/england_epl/2016-2017/2016-08-14 - 18-00 Arsenal 3 - 4 Liverpool/1_224p.mkv"
OUTPUT_DIR = Path("data/analysis/context")


def extract_jersey_color(frame, bbox):
    """
    Crop the player's TORSO region and compute its average HSV color.
    See Stage 7 notes: torso-only isolates jersey from legs/background;
    HSV separates color from lighting/shadow brightness.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    box_height = y2 - y1
    box_width = x2 - x1

    torso_y1 = y1 + int(box_height * 0.15)
    torso_y2 = y1 + int(box_height * 0.55)
    torso_x1 = x1 + int(box_width * 0.20)
    torso_x2 = x1 + int(box_width * 0.80)

    torso_x1, torso_y1 = max(0, torso_x1), max(0, torso_y1)
    torso_x2 = min(frame.shape[1], torso_x2)
    torso_y2 = min(frame.shape[0], torso_y2)

    if torso_x2 <= torso_x1 or torso_y2 <= torso_y1:
        return None

    crop = frame[torso_y1:torso_y2, torso_x1:torso_x2]
    crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mean_hsv = crop_hsv.reshape(-1, 3).mean(axis=0)
    return mean_hsv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--tracks", default=str(DEFAULT_TRACKS))
    args = parser.parse_args()

    tracks_path = Path(args.tracks)
    if not tracks_path.exists():
        logger.error(f"{tracks_path} not found - run track_players.py first.")
        return

    with open(tracks_path) as f:
        records = json.load(f)

    pipeline = VideoPipeline(args.video)
    if not pipeline.load():
        return
    pipeline.extract_metadata()

    by_track = defaultdict(list)
    for r in records:
        by_track[r["track_id"]].append(r)

    logger.info(f"Extracting jersey colors for {len(by_track)} tracked players from {Path(args.video).name}...")

    track_colors = {}
    for track_id, track_records in by_track.items():
        sampled = track_records[::10] if len(track_records) > 10 else track_records
        colors = []
        for record in sampled:
            frame = pipeline.get_frame(record["frame"])
            if frame is None:
                continue
            color = extract_jersey_color(frame, record["bbox"])
            if color is not None:
                colors.append(color)
        if colors:
            track_colors[track_id] = np.mean(colors, axis=0)

    pipeline.release()

    track_ids = list(track_colors.keys())
    features = np.array([track_colors[tid] for tid in track_ids])

    logger.info(f"Extracted color features for {len(track_ids)} players.")

    dbscan = DBSCAN(eps=15, min_samples=2)
    dbscan_labels = dbscan.fit_predict(features)

    n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = list(dbscan_labels).count(-1)

    print("\n=== DBSCAN RESULTS ===")
    print(f"Clusters found: {n_clusters_dbscan} (plus {n_noise} noise/outlier points)")
    for cluster_id in sorted(set(dbscan_labels)):
        members = [tid for tid, lbl in zip(track_ids, dbscan_labels) if lbl == cluster_id]
        label_name = "NOISE (outliers)" if cluster_id == -1 else f"Cluster {cluster_id}"
        print(f"  {label_name}: {len(members)} players - track IDs {members}")

    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    kmeans_labels = kmeans.fit_predict(features)

    print("\n=== KMEANS (2-cluster) RESULTS ===")
    for cluster_id in [0, 1]:
        members = [tid for tid, lbl in zip(track_ids, kmeans_labels) if lbl == cluster_id]
        print(f"  Team {cluster_id}: {len(members)} players - track IDs {members}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "video": args.video,
        "dbscan": {
            "n_clusters": n_clusters_dbscan,
            "n_noise": n_noise,
            "assignments": {str(tid): int(lbl) for tid, lbl in zip(track_ids, dbscan_labels)},
        },
        "kmeans": {
            "assignments": {str(tid): int(lbl) for tid, lbl in zip(track_ids, kmeans_labels)},
        },
    }
    output_path = OUTPUT_DIR / "team_classification.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
