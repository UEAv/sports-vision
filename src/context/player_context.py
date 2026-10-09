"""
Stage 7 - Player Context

Extracts per-player, per-frame: position, speed, direction, proximity to
ball, proximity to nearest other player - plus CAMERA-COMPENSATED speed and
direction.

Why compensation: all positions are in IMAGE pixels, so when the broadcast
camera pans, every player appears to move. The per-frame MEDIAN velocity
across all players is used as a camera-motion estimate and subtracted out.
This removes panning only (not zoom or camera cuts).

Usage:
    python src/context/player_context.py
"""

import json
import argparse
import statistics
from pathlib import Path
from collections import defaultdict
import math
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_PLAYER_TRACKS = "data/analysis/tracking/720p_bytetrack_raw_records.json"
DEFAULT_BALL_CONTEXT = "data/analysis/context/ball_context.json"
OUTPUT_DIR = Path("data/analysis/context")

# Need at least this many moving players in a frame before trusting the
# median as a camera-motion estimate.
MIN_PLAYERS_FOR_CAMERA_ESTIMATE = 4
CAMERA_MOVING_THRESHOLD = 300  # px/sec of shared motion treated as "camera moving"

# A box touching the frame border is only PART of a player (the rest is
# off-screen), and a box pinned at the border does not move with the scene.
# Both break the "everyone moves with the camera" assumption, so these
# detections are flagged and left out of the camera estimate and out of the
# compensated speed.
EDGE_MARGIN_PX = 15


def is_edge_clipped(bbox, frame_w, frame_h, margin=EDGE_MARGIN_PX):
    """True if the box lies within `margin` pixels of any frame border."""
    x1, y1, x2, y2 = bbox
    return x1 <= margin or y1 <= margin or x2 >= frame_w - margin or y2 >= frame_h - margin


def velocity_vector(prev_record, curr_record):
    """Velocity as a vector (vx, vy) in px/sec, or None if dt is not positive."""
    dt = curr_record["timestamp_sec"] - prev_record["timestamp_sec"]
    if dt <= 0:
        return None
    vx = (curr_record["center_x"] - prev_record["center_x"]) / dt
    vy = (curr_record["center_y"] - prev_record["center_y"]) / dt
    return vx, vy


def heading_degrees(vx, vy):
    """
    Direction = atan2(vy, vx) in degrees. atan2 (not plain atan) uses the
    signs of BOTH components so it can tell left from right and up from down.
    0 = right, 90 = down (image y grows downward), +-180 = left, -90 = up.
    """
    if abs(vx) < 1e-9 and abs(vy) < 1e-9:
        return None
    return round(math.degrees(math.atan2(vy, vx)), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player-tracks", default=DEFAULT_PLAYER_TRACKS)
    parser.add_argument("--ball-context", default=DEFAULT_BALL_CONTEXT)
    parser.add_argument("--frame-width", type=int, default=1280)
    parser.add_argument("--frame-height", type=int, default=720)
    args = parser.parse_args()

    with open(args.player_tracks) as f:
        player_records = json.load(f)

    ball_by_frame = {}
    ball_path = Path(args.ball_context)
    if ball_path.exists():
        with open(ball_path) as f:
            ball_positions = json.load(f)
        ball_by_frame = {p["frame"]: p["position"] for p in ball_positions}
        logger.info(f"Loaded ball positions for {len(ball_by_frame)} frames.")
    else:
        logger.warning(f"{ball_path} not found - proximity_to_ball will be null for all players.")

    by_frame = defaultdict(list)
    for r in player_records:
        by_frame[r["frame"]].append(r)

    by_track = defaultdict(list)
    for r in player_records:
        by_track[r["track_id"]].append(r)
    for tid in by_track:
        by_track[tid].sort(key=lambda r: r["frame"])

    prev_lookup = {}
    for tid, recs in by_track.items():
        for i in range(1, len(recs)):
            prev_lookup[(tid, recs[i]["frame"])] = recs[i - 1]

    # --- Pass 1: apparent velocity vector for every detection ---
    vel_lookup = {}
    vectors_by_frame = defaultdict(list)
    for r in player_records:
        prev = prev_lookup.get((r["track_id"], r["frame"]))
        if prev is None:
            continue
        v = velocity_vector(prev, r)
        if v is not None:
            vel_lookup[(r["track_id"], r["frame"])] = v
            # Edge-clipped boxes keep their apparent speed, but must not vote
            # on what the camera is doing.
            if not is_edge_clipped(r["bbox"], args.frame_width, args.frame_height):
                vectors_by_frame[r["frame"]].append(v)

    # --- Camera motion per frame = median velocity across players ---
    camera_motion = {}
    for frame, vecs in vectors_by_frame.items():
        if len(vecs) >= MIN_PLAYERS_FOR_CAMERA_ESTIMATE:
            camera_motion[frame] = (statistics.median(v[0] for v in vecs),
                                    statistics.median(v[1] for v in vecs))

    logger.info(f"Computing context for {len(player_records)} detections across {len(by_frame)} frames "
                f"({len(camera_motion)} frames have a camera-motion estimate)...")

    # --- Pass 2: assemble enriched records ---
    enriched = []
    for record in player_records:
        frame, track_id = record["frame"], record["track_id"]
        context = dict(record)
        clipped = is_edge_clipped(record["bbox"], args.frame_width, args.frame_height)
        context["edge_clipped"] = clipped

        vel = vel_lookup.get((track_id, frame))
        if vel is not None:
            context["direction_degrees"] = heading_degrees(*vel)
            context["speed_px_per_sec"] = round(math.hypot(*vel), 1)
        else:
            context["direction_degrees"] = None
            context["speed_px_per_sec"] = None

        cam = camera_motion.get(frame)
        if cam is not None:
            context["camera_motion_px_per_sec"] = round(math.hypot(*cam), 1)
        else:
            context["camera_motion_px_per_sec"] = None

        if vel is not None and cam is not None and not clipped:
            cvx, cvy = vel[0] - cam[0], vel[1] - cam[1]
            context["speed_compensated_px_per_sec"] = round(math.hypot(cvx, cvy), 1)
            context["direction_compensated_degrees"] = heading_degrees(cvx, cvy)
        else:
            context["speed_compensated_px_per_sec"] = None
            context["direction_compensated_degrees"] = None

        ball_pos = ball_by_frame.get(frame)
        if ball_pos is not None:
            d = ((record["center_x"] - ball_pos[0]) ** 2 + (record["center_y"] - ball_pos[1]) ** 2) ** 0.5
            context["proximity_to_ball_px"] = round(d, 1)
        else:
            context["proximity_to_ball_px"] = None

        others = [p for p in by_frame[frame] if p["track_id"] != track_id]
        if others:
            nearest = min(others, key=lambda p: (p["center_x"] - record["center_x"]) ** 2 + (p["center_y"] - record["center_y"]) ** 2)
            d = ((nearest["center_x"] - record["center_x"]) ** 2 + (nearest["center_y"] - record["center_y"]) ** 2) ** 0.5
            context["nearest_player_track_id"] = nearest["track_id"]
            context["proximity_to_nearest_player_px"] = round(d, 1)
        else:
            context["nearest_player_track_id"] = None
            context["proximity_to_nearest_player_px"] = None

        enriched.append(context)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "player_context.json"
    with open(output_path, "w") as f:
        json.dump(enriched, f, indent=2)

    print("\n=== PLAYER CONTEXT SUMMARY ===")
    print(f"Total enriched records: {len(enriched)}")
    n_clipped = sum(1 for r in enriched if r["edge_clipped"])
    print(f"Edge-clipped detections (excluded from compensation): {n_clipped} ({n_clipped/len(enriched)*100:.1f}%)")

    apparent = [r["speed_px_per_sec"] for r in enriched if r["speed_px_per_sec"] is not None]
    comp = [r["speed_compensated_px_per_sec"] for r in enriched if r["speed_compensated_px_per_sec"] is not None]
    if apparent:
        print(f"\nApparent speed (includes camera):  avg {sum(apparent)/len(apparent):.1f} | max {max(apparent):.1f} px/sec")
    if comp:
        print(f"Compensated speed (camera removed): avg {sum(comp)/len(comp):.1f} | max {max(comp):.1f} px/sec")

    cam_mags = [m for m in {r["frame"]: r["camera_motion_px_per_sec"] for r in enriched}.values() if m is not None]
    if cam_mags:
        moving = sum(1 for m in cam_mags if m >= CAMERA_MOVING_THRESHOLD)
        print(f"\nFrames with a camera estimate: {len(cam_mags)}")
        print(f"Frames with camera motion >= {CAMERA_MOVING_THRESHOLD} px/sec: {moving} ({moving/len(cam_mags)*100:.0f}%)")

    ball_prox = [r["proximity_to_ball_px"] for r in enriched if r["proximity_to_ball_px"] is not None]
    if ball_prox:
        print(f"\nAvg proximity to ball: {sum(ball_prox)/len(ball_prox):.1f}px | closest: {min(ball_prox):.1f}px")

    print(f"\nSaved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
