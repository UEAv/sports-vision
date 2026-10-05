"""
Stage 6 - Compute Player Trajectories

Turns raw per-frame tracking records into clean per-player trajectories:
position history, velocity, and visibility - the actual Stage 6 deliverable
("stable player trajectories"), not just raw frame-by-frame detections.

Usage:
    python src/tracking/compute_trajectories.py --tracker bytetrack
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger

logger = get_logger(__name__)

OUTPUT_DIR = Path("data/analysis/tracking")


def compute_velocity(prev_record, curr_record):
    """
    Velocity = distance moved / time elapsed, in pixels per second.

    THE MATH:
        dx = curr_x - prev_x
        dy = curr_y - prev_y
        distance = sqrt(dx^2 + dy^2)          <- Pythagorean theorem
        velocity = distance / time_elapsed

    Plain English: if a player's center moved 20 pixels over 0.08 seconds,
    they're moving at 20/0.08 = 250 pixels/second in whatever direction
    that movement pointed. This is a 2D speed, not yet broken into
    separate left-right / up-down components (we keep it simple here -
    direction could be added later as dx/dt and dy/dt separately if needed).
    """
    dx = curr_record["center_x"] - prev_record["center_x"]
    dy = curr_record["center_y"] - prev_record["center_y"]
    distance = (dx ** 2 + dy ** 2) ** 0.5

    dt = curr_record["timestamp_sec"] - prev_record["timestamp_sec"]
    if dt <= 0:
        return 0.0

    return distance / dt


def build_trajectory(track_id: int, records: list, fps: float = 25.0):
    """
    Build one player's full trajectory from their raw per-frame records.

    VISIBILITY: a track's records aren't always perfectly consecutive frames -
    there can be gaps (brief occlusion the tracker bridged via track_buffer).
    visibility = (frames actually detected) / (total frames in the track's
    time span) - a value near 1.0 means nearly every frame had a confident
    detection; lower means the tracker "coasted" through gaps more often.
    """
    records_sorted = sorted(records, key=lambda r: r["frame"])

    first_frame = records_sorted[0]["frame"]
    last_frame = records_sorted[-1]["frame"]
    expected_frame_count = last_frame - first_frame + 1
    actual_frame_count = len(records_sorted)
    visibility = round(actual_frame_count / expected_frame_count, 3) if expected_frame_count > 0 else 1.0

    trajectory_points = []
    velocities = []

    for i, record in enumerate(records_sorted):
        point = {
            "frame": record["frame"],
            "timestamp_sec": record["timestamp_sec"],
            "position": [record["center_x"], record["center_y"]],
            "bbox": record["bbox"],
            "confidence": record["confidence"],
        }

        if i > 0:
            velocity = compute_velocity(records_sorted[i - 1], record)
            point["velocity_px_per_sec"] = round(velocity, 1)
            velocities.append(velocity)
        else:
            point["velocity_px_per_sec"] = None  # no previous point to compare

        trajectory_points.append(point)

    avg_velocity = round(sum(velocities) / len(velocities), 1) if velocities else 0.0
    max_velocity = round(max(velocities), 1) if velocities else 0.0

    return {
        "track_id": track_id,
        "start_time_sec": records_sorted[0]["timestamp_sec"],
        "end_time_sec": records_sorted[-1]["timestamp_sec"],
        "duration_sec": round(records_sorted[-1]["timestamp_sec"] - records_sorted[0]["timestamp_sec"], 2),
        "num_detections": actual_frame_count,
        "visibility": visibility,
        "avg_velocity_px_per_sec": avg_velocity,
        "max_velocity_px_per_sec": max_velocity,
        "trajectory": trajectory_points,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker", default="bytetrack", choices=["bytetrack", "botsort"])
    args = parser.parse_args()

    records_path = Path(f"data/analysis/tracking/{args.tracker}_raw_records.json")
    if not records_path.exists():
        logger.error(f"{records_path} not found - run track_players.py first.")
        return

    with open(records_path) as f:
        records = json.load(f)

    by_track = defaultdict(list)
    for r in records:
        by_track[r["track_id"]].append(r)

    logger.info(f"Building trajectories for {len(by_track)} tracks from {args.tracker}...")

    trajectories = [build_trajectory(tid, recs) for tid, recs in by_track.items()]
    trajectories.sort(key=lambda t: t["start_time_sec"])

    output_path = OUTPUT_DIR / f"{args.tracker}_trajectories.json"
    with open(output_path, "w") as f:
        json.dump(trajectories, f, indent=2)

    print(f"\n=== TRAJECTORY SUMMARY ({args.tracker}) ===")
    print(f"Total player trajectories: {len(trajectories)}")

    visibilities = [t["visibility"] for t in trajectories]
    avg_visibility = sum(visibilities) / len(visibilities)
    print(f"Average visibility: {avg_visibility:.3f} (1.0 = detected every frame, no gaps)")

    avg_velocities = [t["avg_velocity_px_per_sec"] for t in trajectories if t["num_detections"] > 1]
    if avg_velocities:
        print(f"Average player velocity: {sum(avg_velocities)/len(avg_velocities):.1f} px/sec")
        print(f"Fastest single movement recorded: {max(t['max_velocity_px_per_sec'] for t in trajectories):.1f} px/sec")

    print(f"\nSaved full trajectories to: {output_path.resolve()}")
    print("\nSample trajectory (first player):")
    sample = trajectories[0]
    print(f"  Track #{sample['track_id']}: {sample['duration_sec']}s duration, "
          f"{sample['num_detections']} detections, visibility={sample['visibility']}, "
          f"avg velocity={sample['avg_velocity_px_per_sec']} px/sec")


if __name__ == "__main__":
    main()
