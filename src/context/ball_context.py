"""
Stage 7 - Ball Context

Determines: ball location, trajectory, velocity, nearest player,
ball-player distance, potential possession.

Simpler than player tracking: there's only ONE ball, so no multi-object
ID-matching needed - just take the highest-confidence Ball detection per
frame as "the ball" for that frame.

Usage:
    python src/context/ball_context.py
"""

import json
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger
from src.pipeline.video_pipeline import VideoPipeline
from ultralytics import YOLO

logger = get_logger(__name__)

MODEL_PATH = "models/yolov8_football_finetune_v1.pt"
DEFAULT_VIDEO = "data/ball_action/spotting-ball-2023/valid/england_efl/2019-2020/2019-10-01 - Middlesbrough - Preston North End/1_720p.mkv"
DEFAULT_PLAYER_TRACKS = "data/analysis/tracking/720p_bytetrack_raw_records.json"
OUTPUT_DIR = Path("data/analysis/context")
FPS = 25.0

# Within this many pixels, a player is considered "close enough" to
# plausibly be in possession - not a certainty, just a reasonable proxy.
POSSESSION_DISTANCE_THRESHOLD = 80

# Ball detections below this confidence are discarded entirely (treated as
# "not detected" rather than trusted). Diagnostic check showed the most
# extreme false "velocity spikes" came almost exclusively from low-confidence
# detections (likely false positives elsewhere in frame), not real fast kicks.
BALL_CONFIDENCE_THRESHOLD = 0.5


def detect_ball_positions(video_path: str, start_sec: float, duration_sec: float):
    """Detect the ball in every frame, keeping only the highest-confidence
    detection per frame (there's only one real ball)."""
    model = YOLO(MODEL_PATH)
    start_frame = int(start_sec * FPS)
    num_frames = int(duration_sec * FPS)

    import cv2
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ball_positions = []

    for frame_offset in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]
        ball_boxes = [
            (box, conf) for box, cls, conf in
            zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf)
            if model.names[int(cls)] == "Ball"
        ]

        if not ball_boxes:
            continue  # ball not detected this frame - a real gap, not an error

        # Keep only the single highest-confidence ball detection this frame
        best_box, best_conf = max(ball_boxes, key=lambda bc: bc[1])

        if float(best_conf) < BALL_CONFIDENCE_THRESHOLD:
            continue  # too unreliable to trust - treat as a gap, not a real position
        x1, y1, x2, y2 = best_box.tolist()
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

        ball_positions.append({
            "frame": start_frame + frame_offset,
            "timestamp_sec": round((start_frame + frame_offset) / FPS, 3),
            "position": [round(center_x, 1), round(center_y, 1)],
            "confidence": round(float(best_conf), 3),
        })

    cap.release()
    return ball_positions, start_frame, num_frames


def compute_ball_velocity(ball_positions: list):
    """Same distance/time math as Stage 6's player velocity - just applied
    to ball positions, which may have GAPS (frames where ball wasn't
    detected), so dt between consecutive entries can be larger than 1 frame."""
    for i, point in enumerate(ball_positions):
        if i == 0:
            point["velocity_px_per_sec"] = None
            continue
        prev = ball_positions[i - 1]
        dx = point["position"][0] - prev["position"][0]
        dy = point["position"][1] - prev["position"][1]
        distance = (dx ** 2 + dy ** 2) ** 0.5
        dt = point["timestamp_sec"] - prev["timestamp_sec"]
        point["velocity_px_per_sec"] = round(distance / dt, 1) if dt > 0 else 0.0


def find_nearest_player(ball_position: list, players_at_frame: list):
    """For one ball position, find the closest player and the distance to them."""
    if not players_at_frame:
        return None, None

    nearest = min(
        players_at_frame,
        key=lambda p: ((p["center_x"] - ball_position[0]) ** 2 + (p["center_y"] - ball_position[1]) ** 2) ** 0.5
    )
    distance = ((nearest["center_x"] - ball_position[0]) ** 2 + (nearest["center_y"] - ball_position[1]) ** 2) ** 0.5
    return nearest["track_id"], round(distance, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--player-tracks", default=DEFAULT_PLAYER_TRACKS)
    parser.add_argument("--start", type=float, default=120.0)
    parser.add_argument("--duration", type=float, default=15.0)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Detecting ball across all frames in the window...")
    ball_positions, start_frame, num_frames = detect_ball_positions(args.video, args.start, args.duration)

    visibility = round(len(ball_positions) / num_frames, 3) if num_frames > 0 else 0.0
    logger.info(f"Ball detected in {len(ball_positions)}/{num_frames} frames (visibility={visibility})")

    compute_ball_velocity(ball_positions)

    # Load player positions for nearest-player / possession lookup
    player_tracks_path = Path(args.player_tracks)
    players_by_frame = {}
    if player_tracks_path.exists():
        with open(player_tracks_path) as f:
            player_records = json.load(f)
        from collections import defaultdict
        players_by_frame = defaultdict(list)
        for r in player_records:
            players_by_frame[r["frame"]].append(r)
    else:
        logger.warning(f"{player_tracks_path} not found - nearest-player/possession will be skipped.")

    for point in ball_positions:
        players_here = players_by_frame.get(point["frame"], [])
        nearest_id, distance = find_nearest_player(point["position"], players_here)
        point["nearest_player_track_id"] = nearest_id
        point["ball_player_distance_px"] = distance
        point["potential_possession"] = (distance is not None and distance <= POSSESSION_DISTANCE_THRESHOLD)

    output_path = OUTPUT_DIR / "ball_context.json"
    with open(output_path, "w") as f:
        json.dump(ball_positions, f, indent=2)

    print(f"\n=== BALL CONTEXT SUMMARY ===")
    print(f"Ball visibility: {visibility} ({len(ball_positions)}/{num_frames} frames)")

    velocities = [p["velocity_px_per_sec"] for p in ball_positions if p["velocity_px_per_sec"] is not None]
    if velocities:
        print(f"Avg ball velocity: {sum(velocities)/len(velocities):.1f} px/sec")
        print(f"Max ball velocity: {max(velocities):.1f} px/sec")

    possession_frames = [p for p in ball_positions if p["potential_possession"]]
    print(f"Frames with potential possession (player within {POSSESSION_DISTANCE_THRESHOLD}px): "
          f"{len(possession_frames)}/{len(ball_positions)}")

    print(f"\nSaved to: {output_path.resolve()}")
    print("\nSample entries:")
    for p in ball_positions[:5]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
