"""
Stage 7 - Combined Frame-Level Representation (the Stage 7 deliverable)

Merges everything extracted so far into ONE structure per frame:

    Frame
     |-- quality (camera motion, what is available this frame)
     |-- players  (each with team, tracking, spatial, pose)
     |-- ball
     |-- goalkeeper (unresolved - documented gap)
     |-- teams (track IDs grouped by team cluster)
     `-- spatial_relationships (possession candidate, close player pairs)

Nothing new is detected here - this script only JOINS existing outputs on
(frame, track_id) and checks that nothing was silently dropped.

Usage:
    python src/context/frame_representation.py
"""

import json
import math
import argparse
from pathlib import Path
from collections import defaultdict
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger

logger = get_logger(__name__)

CONTEXT_DIR = Path("data/analysis/context")
FPS = 25.0
CAMERA_HIGH_MOTION_PX_PER_SEC = 300   # same threshold used in player_context.py
CLOSE_PAIR_DISTANCE_PX = 80           # players closer than this are listed as a close pair
SAMPLE_FRAMES = [3000, 3095, 3250]    # frames that have pose; written to a readable sample file

LIMITATIONS = [
    "All positions are image pixels (origin top-left, y grows downward), not pitch coordinates.",
    "Speeds are px/sec. 'speed_px_per_sec' includes camera motion; 'speed_compensated_px_per_sec' removes panning only (not zoom or cuts) and is null for edge-clipped boxes or frames with fewer than 4 players.",
    "Ball velocity is apparent (image-space) speed and is NOT camera-compensated. Ball is only present on frames where it was detected with confidence >= 0.5.",
    "Team cluster is an unnamed KMeans cluster (0 or 1), computed once per TRACK from average torso colour. It is not mapped to home/away. dbscan_label -1 means outlier (possible referee or goalkeeper, unreliable).",
    "Goalkeeper is unresolved: no Goalkeeper class exists in the training data and the position/colour heuristic failed on a 15 s clip.",
    "Pose exists only on a few sampled frames and only for players where the model returned a skeleton. Keypoints are in frame pixel coordinates; null means that keypoint was not detected.",
    "potential_possession is a distance proxy (nearest player within 80 px of the ball), not a true possession decision.",
    "Built from one 15 second clip of one match.",
]


def load_json(path, required=False):
    p = Path(path)
    if not p.exists():
        if required:
            raise SystemExit(f"Required file not found: {p}")
        logger.warning(f"Optional file not found, continuing without it: {p}")
        return None
    with open(p) as f:
        return json.load(f)


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def build_player_entry(rec, ball, kmeans, dbscan, pose):
    tid = rec["track_id"]
    entry = {
        "track_id": tid,
        "bbox": rec["bbox"],
        "confidence": rec["confidence"],
        "position": [rec["center_x"], rec["center_y"]],
        "team": {
            "cluster": kmeans.get(str(tid)),
            "dbscan_label": dbscan.get(str(tid)),
        },
        "tracking": {
            "speed_px_per_sec": rec.get("speed_px_per_sec"),
            "direction_degrees": rec.get("direction_degrees"),
            "speed_compensated_px_per_sec": rec.get("speed_compensated_px_per_sec"),
            "direction_compensated_degrees": rec.get("direction_compensated_degrees"),
            "edge_clipped": rec.get("edge_clipped"),
        },
        "spatial": {
            "proximity_to_ball_px": rec.get("proximity_to_ball_px"),
            "is_ball_nearest": bool(ball is not None and ball.get("nearest_player_track_id") == tid),
            "nearest_player_track_id": rec.get("nearest_player_track_id"),
            "proximity_to_nearest_player_px": rec.get("proximity_to_nearest_player_px"),
        },
        "pose": None,
    }
    if pose is not None:
        entry["pose"] = {
            "usable": pose.get("usable"),
            "confident_keypoints": pose.get("confident_keypoints"),
            "keypoints_xy": pose.get("keypoints_xy"),
            "keypoints_confidence": pose.get("keypoints_confidence"),
            "body_orientation_degrees": pose.get("body_orientation_degrees"),
            "limb_configuration": pose.get("limb_configuration"),
        }
    return entry


def build_frames(player_records, ball_records, team_data, pose_records):
    players_by_frame = defaultdict(list)
    for r in player_records:
        players_by_frame[r["frame"]].append(r)

    ball_by_frame = {b["frame"]: b for b in ball_records}
    pose_by_key = {(p["frame"], p["track_id"]): p for p in pose_records}

    kmeans = (team_data or {}).get("kmeans", {}).get("assignments", {})
    dbscan = (team_data or {}).get("dbscan", {}).get("assignments", {})

    frames_out = []
    counts = {"players": 0, "ball": 0, "pose": 0}

    for frame in sorted(set(players_by_frame) | set(ball_by_frame)):
        recs = players_by_frame.get(frame, [])
        ball = ball_by_frame.get(frame)

        players = []
        for rec in recs:
            pose = pose_by_key.get((frame, rec["track_id"]))
            players.append(build_player_entry(rec, ball, kmeans, dbscan, pose))
            counts["players"] += 1
            counts["pose"] += 1 if pose is not None else 0

        ball_entry = None
        if ball is not None:
            counts["ball"] += 1
            ball_entry = {
                "position": ball["position"],
                "confidence": ball["confidence"],
                "velocity_px_per_sec": ball.get("velocity_px_per_sec"),
                "nearest_player_track_id": ball.get("nearest_player_track_id"),
                "ball_player_distance_px": ball.get("ball_player_distance_px"),
                "potential_possession": ball.get("potential_possession"),
            }

        camera = next((r["camera_motion_px_per_sec"] for r in recs
                       if r.get("camera_motion_px_per_sec") is not None), None)

        teams = {"cluster_0": [], "cluster_1": [], "unassigned": []}
        for p in players:
            c = p["team"]["cluster"]
            teams[f"cluster_{c}" if c in (0, 1) else "unassigned"].append(p["track_id"])

        close_pairs = []
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                d = distance((recs[i]["center_x"], recs[i]["center_y"]),
                             (recs[j]["center_x"], recs[j]["center_y"]))
                if d <= CLOSE_PAIR_DISTANCE_PX:
                    close_pairs.append([recs[i]["track_id"], recs[j]["track_id"], round(d, 1)])

        possession_track = None
        if ball is not None and ball.get("potential_possession"):
            possession_track = ball.get("nearest_player_track_id")

        timestamp = recs[0]["timestamp_sec"] if recs else (ball["timestamp_sec"] if ball else round(frame / FPS, 3))

        frames_out.append({
            "frame": frame,
            "timestamp_sec": timestamp,
            "quality": {
                "camera_motion_px_per_sec": camera,
                "camera_high_motion": (camera >= CAMERA_HIGH_MOTION_PX_PER_SEC) if camera is not None else None,
                "ball_detected": ball is not None,
                "num_players": len(players),
                "pose_available": any(p["pose"] is not None for p in players),
            },
            "players": players,
            "ball": ball_entry,
            "goalkeeper": None,
            "teams": teams,
            "spatial_relationships": {
                "ball_nearest_player_track_id": ball.get("nearest_player_track_id") if ball else None,
                "ball_nearest_distance_px": ball.get("ball_player_distance_px") if ball else None,
                "potential_possession_track_id": possession_track,
                "close_player_pairs": close_pairs,
            },
        })

    return frames_out, counts


def summarize_frame(fr):
    q = fr["quality"]
    print(f"  frame {fr['frame']} ({fr['timestamp_sec']}s): players={q['num_players']}, "
          f"ball={'yes' if q['ball_detected'] else 'no'}, pose={'yes' if q['pose_available'] else 'no'}, "
          f"camera={q['camera_motion_px_per_sec']} px/sec, "
          f"close pairs={len(fr['spatial_relationships']['close_player_pairs'])}, "
          f"possession candidate=#{fr['spatial_relationships']['potential_possession_track_id']}")
    print(f"      teams: cluster_0={fr['teams']['cluster_0']} cluster_1={fr['teams']['cluster_1']} unassigned={fr['teams']['unassigned']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--player-context", default=str(CONTEXT_DIR / "player_context.json"))
    ap.add_argument("--ball-context", default=str(CONTEXT_DIR / "ball_context.json"))
    ap.add_argument("--teams", default=str(CONTEXT_DIR / "team_classification.json"))
    ap.add_argument("--pose", default=str(CONTEXT_DIR / "pose_per_player.json"))
    ap.add_argument("--out-dir", default=str(CONTEXT_DIR))
    args = ap.parse_args()

    player_records = load_json(args.player_context, required=True)
    ball_records = load_json(args.ball_context) or []
    team_data = load_json(args.teams)
    pose_records = load_json(args.pose) or []

    frames, counts = build_frames(player_records, ball_records, team_data, pose_records)

    # --- Integrity checks: the join must not silently drop anything ---
    print("\n=== INTEGRITY CHECKS ===")
    checks = [
        ("player detections", counts["players"], len(player_records)),
        ("ball detections", counts["ball"], len(ball_records)),
        ("pose entries", counts["pose"], len(pose_records)),
    ]
    all_ok = True
    for name, placed, expected in checks:
        ok = placed == expected
        all_ok &= ok
        print(f"  {'OK  ' if ok else 'FAIL'} {name}: placed {placed} of {expected}")
    if not all_ok:
        print("  WARNING: some inputs were not placed in any frame - check that all files come from the SAME clip.")

    # --- Write outputs ---
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "video": (team_data or {}).get("video"),
        "fps": FPS,
        "frame_range": [frames[0]["frame"], frames[-1]["frame"]] if frames else None,
        "num_frames": len(frames),
        "sources": {
            "players": args.player_context, "ball": args.ball_context,
            "teams": args.teams, "pose": args.pose,
        },
        "goalkeeper_status": "unresolved - see goalkeeper_identification_evaluation.yaml",
        "limitations": LIMITATIONS,
    }

    full_path = out_dir / "frame_representation.json"
    with open(full_path, "w") as f:
        json.dump({"metadata": metadata, "frames": frames}, f)

    sample = [fr for fr in frames if fr["frame"] in SAMPLE_FRAMES]
    sample_path = out_dir / "frame_representation_sample.json"
    with open(sample_path, "w") as f:
        json.dump({"metadata": metadata, "frames": sample}, f, indent=2)

    # --- Summary ---
    n = len(frames)
    total_players = sum(fr["quality"]["num_players"] for fr in frames)
    with_team = sum(1 for fr in frames for p in fr["players"] if p["team"]["cluster"] is not None)
    cam_known = [fr for fr in frames if fr["quality"]["camera_motion_px_per_sec"] is not None]
    high_motion = sum(1 for fr in cam_known if fr["quality"]["camera_high_motion"])

    print("\n=== FRAME REPRESENTATION SUMMARY ===")
    print(f"Frames: {n}")
    print(f"Avg players per frame: {total_players / n:.1f}")
    print(f"Frames with ball detected: {sum(fr['quality']['ball_detected'] for fr in frames)} ({sum(fr['quality']['ball_detected'] for fr in frames) / n * 100:.0f}%)")
    print(f"Frames with pose: {sum(fr['quality']['pose_available'] for fr in frames)}")
    print(f"Frames with a camera estimate: {len(cam_known)} | high camera motion: {high_motion}")
    print(f"Players with a team cluster: {with_team} of {total_players}")
    print(f"Frames with a possession candidate: {sum(1 for fr in frames if fr['spatial_relationships']['potential_possession_track_id'] is not None)}")
    print(f"Goalkeeper: unresolved on all frames (documented gap)")

    print("\nSample frames:")
    for fr in sample:
        summarize_frame(fr)

    print(f"\nSaved full file:   {full_path.resolve()}")
    print(f"Saved readable sample (open this one): {sample_path.resolve()}")


if __name__ == "__main__":
    main()
