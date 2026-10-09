"""
Stage 7 - Goalkeeper Identification (heuristic, since no Goalkeeper class
exists in the training data - a known gap documented in Stage 5).

Approach: combine two weak signals into one stronger one:
  1. Positional range - goalkeepers move far less than outfield players,
     staying near their own goal for nearly the whole match.
  2. DBSCAN outlier signal from team_classification.py - goalkeepers often
     wear a different kit color than their own team.

Usage:
    python src/context/identify_goalkeeper.py
"""

import json
from pathlib import Path
from collections import defaultdict
import sys
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger

logger = get_logger(__name__)

TRACKS_PATH = Path("data/analysis/tracking/720p_bytetrack_raw_records.json")
TEAM_CLASSIFICATION_PATH = Path("data/analysis/context/team_classification.json")
OUTPUT_DIR = Path("data/analysis/context")


def compute_positional_range(records: list) -> dict:
    """
    How much ground did this track cover, in both x and y?

    THE MATH: just max minus min for each axis - the simplest possible
    'range' measure. A goalkeeper camped near their own goal has a tiny
    x-range (barely moves left-right across the match); an outfield
    player covering the pitch has a much larger one.
    """
    xs = [r["center_x"] for r in records]
    ys = [r["center_y"] for r in records]
    return {
        "x_range": round(max(xs) - min(xs), 1),
        "y_range": round(max(ys) - min(ys), 1),
        "total_range": round(((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5, 1),
    }


def main():
    if not TRACKS_PATH.exists():
        logger.error(f"{TRACKS_PATH} not found.")
        return

    with open(TRACKS_PATH) as f:
        records = json.load(f)

    by_track = defaultdict(list)
    for r in records:
        by_track[r["track_id"]].append(r)

    # Only consider tracks with enough detections to measure range meaningfully -
    # a track with just 2-3 frames can't tell us much about typical movement.
    MIN_DETECTIONS = 20
    ranges = {}
    for tid, track_records in by_track.items():
        if len(track_records) >= MIN_DETECTIONS:
            ranges[tid] = compute_positional_range(track_records)

    logger.info(f"Computed positional range for {len(ranges)} tracks (>= {MIN_DETECTIONS} detections each).")

    # Sort by total range, ascending - smallest movers first (most GK-like)
    sorted_by_range = sorted(ranges.items(), key=lambda kv: kv[1]["total_range"])

    print("\n=== SMALLEST POSITIONAL RANGE (most goalkeeper-like) ===")
    for tid, r in sorted_by_range[:5]:
        print(f"  Track #{tid}: total_range={r['total_range']}px (x={r['x_range']}, y={r['y_range']})")

    print("\n=== LARGEST POSITIONAL RANGE (clearly outfield players) ===")
    for tid, r in sorted_by_range[-5:]:
        print(f"  Track #{tid}: total_range={r['total_range']}px (x={r['x_range']}, y={r['y_range']})")

    # Cross-reference with DBSCAN's outlier flags from team_classification.py,
    # if that file exists - combining both signals is stronger than either alone.
    dbscan_outliers = set()
    if TEAM_CLASSIFICATION_PATH.exists():
        with open(TEAM_CLASSIFICATION_PATH) as f:
            team_data = json.load(f)
        dbscan_assignments = team_data["dbscan"]["assignments"]
        dbscan_outliers = {int(tid) for tid, label in dbscan_assignments.items() if label == -1}
        logger.info(f"Loaded {len(dbscan_outliers)} DBSCAN outlier track IDs for cross-reference.")

    print("\n=== COMBINED SIGNAL: low movement range AND flagged by DBSCAN as outlier ===")
    likely_goalkeepers = []
    for tid, r in sorted_by_range[:10]:  # check the 10 smallest-range tracks
        is_dbscan_outlier = tid in dbscan_outliers
        marker = " <-- FLAGGED BY BOTH SIGNALS" if is_dbscan_outlier else ""
        print(f"  Track #{tid}: range={r['total_range']}px, dbscan_outlier={is_dbscan_outlier}{marker}")
        if is_dbscan_outlier:
            likely_goalkeepers.append(tid)

    output = {
        "positional_ranges": {str(tid): r for tid, r in ranges.items()},
        "likely_goalkeepers_combined_signal": likely_goalkeepers,
        "note": "Heuristic identification only - no Goalkeeper class exists in training data (Stage 5 gap). Combines low positional range + DBSCAN color-outlier signal. Recommend manual visual confirmation before treating as ground truth.",
    }
    output_path = OUTPUT_DIR / "goalkeeper_identification.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to: {output_path.resolve()}")
    if likely_goalkeepers:
        print(f"Players flagged by BOTH signals (strongest candidates): {likely_goalkeepers}")
    else:
        print("No track was flagged by both signals - inspect the smallest-range list manually instead.")


if __name__ == "__main__":
    main()
