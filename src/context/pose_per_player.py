"""
Stage 7 - Per-Player Pose Estimation

Instead of running pose on the whole frame (low recall: small players get
missed), crop each TRACKED player using the boxes from the fine-tuned
detector, upscale the crop, and run pose on it. Each pose is then tied to a
track ID.

Usage:
    python src/context/pose_per_player.py
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import sys
import cv2

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger
from src.pipeline.video_pipeline import VideoPipeline
from src.context.pose_estimation import compute_body_orientation, compute_limb_configuration
from ultralytics import YOLO

logger = get_logger(__name__)

DEFAULT_VIDEO = "data/ball_action/spotting-ball-2023/valid/england_efl/2019-2020/2019-10-01 - Middlesbrough - Preston North End/1_720p.mkv"
DEFAULT_TRACKS = "data/analysis/tracking/720p_bytetrack_raw_records.json"
OUTPUT_DIR = Path("data/analysis/context")
FPS = 25.0

TEST_TIMESTAMPS = [120.0, 123.8, 130.0]

CROP_PADDING = 0.15          # extra margin around the box so limbs aren't clipped
TARGET_CROP_HEIGHT = 480     # upscale every crop to roughly this height
KEYPOINT_CONF_THRESHOLD = 0.5
MIN_CONFIDENT_KEYPOINTS = 8  # of 17 - below this, the pose isn't "usable"

# COCO skeleton connections, used only to draw the verification image
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def crop_player(frame, bbox):
    """Padded crop + upscale. Returns (crop, scale, offset_x, offset_y)."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    px1 = max(0, int(x1 - w * CROP_PADDING))
    py1 = max(0, int(y1 - h * CROP_PADDING))
    px2 = min(frame.shape[1], int(x2 + w * CROP_PADDING))
    py2 = min(frame.shape[0], int(y2 + h * CROP_PADDING))

    crop = frame[py1:py2, px1:px2]
    if crop.size == 0:
        return None, None, None, None

    scale = TARGET_CROP_HEIGHT / crop.shape[0]
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop, scale, px1, py1


def pick_main_person(results, crop_shape):
    """The crop may contain a neighbour too. Keep the pose whose centre is
    closest to the crop's centre - that's the player we cropped around."""
    if results.keypoints is None or len(results.keypoints.xy) == 0:
        return None
    cx, cy = crop_shape[1] / 2, crop_shape[0] / 2
    best_idx, best_dist = None, None
    for i, box in enumerate(results.boxes.xyxy):
        bx = (box[0].item() + box[2].item()) / 2
        by = (box[1].item() + box[3].item()) / 2
        dist = (bx - cx) ** 2 + (by - cy) ** 2
        if best_dist is None or dist < best_dist:
            best_idx, best_dist = i, dist
    return best_idx


def map_to_frame(kpts_xy, kpts_conf, scale, off_x, off_y):
    """Convert crop-space keypoints back into full-frame pixel coordinates."""
    mapped = []
    for (x, y), c in zip(kpts_xy, kpts_conf):
        if x == 0 and y == 0:
            mapped.append(None)  # keypoint not detected
        else:
            mapped.append([round(x / scale + off_x, 1), round(y / scale + off_y, 1)])
    return mapped


def draw_pose(frame, track_id, mapped_kpts, kpts_conf):
    for a, b in SKELETON:
        if mapped_kpts[a] and mapped_kpts[b] and kpts_conf[a] >= KEYPOINT_CONF_THRESHOLD and kpts_conf[b] >= KEYPOINT_CONF_THRESHOLD:
            cv2.line(frame, tuple(int(v) for v in mapped_kpts[a]), tuple(int(v) for v in mapped_kpts[b]), (0, 255, 0), 1)
    for kp, c in zip(mapped_kpts, kpts_conf):
        if kp and c >= KEYPOINT_CONF_THRESHOLD:
            cv2.circle(frame, (int(kp[0]), int(kp[1])), 2, (0, 0, 255), -1)
    head = next((kp for kp in mapped_kpts[:5] if kp), None)
    if head:
        cv2.putText(frame, f"#{track_id}", (int(head[0]) - 8, int(head[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--tracks", default=DEFAULT_TRACKS)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(args.tracks) as f:
        records = json.load(f)
    by_frame = defaultdict(list)
    for r in records:
        by_frame[r["frame"]].append(r)

    model = YOLO("yolov8n-pose.pt")
    pipeline = VideoPipeline(args.video)
    if not pipeline.load():
        return
    pipeline.extract_metadata()

    all_poses = []
    totals = {"players": 0, "pose_found": 0, "usable": 0}

    for ts in TEST_TIMESTAMPS:
        frame_idx = int(round(ts * FPS))
        frame = pipeline.get_frame(frame_idx)
        players = by_frame.get(frame_idx, [])
        if frame is None or not players:
            logger.warning(f"No frame or no tracked players at {ts}s (frame {frame_idx}) - skipping.")
            continue

        annotated = frame.copy()
        found, usable = 0, 0

        for p in players:
            crop, scale, off_x, off_y = crop_player(frame, p["bbox"])
            if crop is None:
                continue

            results = model(crop, verbose=False)[0]
            idx = pick_main_person(results, crop.shape)
            if idx is None:
                continue

            kpts_xy = results.keypoints.xy[idx].tolist()
            kpts_conf = results.keypoints.conf[idx].tolist()
            n_confident = sum(1 for c in kpts_conf if c >= KEYPOINT_CONF_THRESHOLD)

            # Orientation/limb maths run on CROP-space points, where (0,0)
            # still means "not detected".
            orientation = compute_body_orientation(kpts_xy)
            limb_config = compute_limb_configuration(kpts_xy)
            mapped = map_to_frame(kpts_xy, kpts_conf, scale, off_x, off_y)

            is_usable = n_confident >= MIN_CONFIDENT_KEYPOINTS
            found += 1
            usable += 1 if is_usable else 0

            all_poses.append({
                "frame": frame_idx,
                "timestamp_sec": ts,
                "track_id": p["track_id"],
                "keypoints_xy": mapped,
                "keypoints_confidence": [round(c, 3) for c in kpts_conf],
                "confident_keypoints": n_confident,
                "usable": is_usable,
                "body_orientation_degrees": orientation,
                "limb_configuration": limb_config,
            })

            draw_pose(annotated, p["track_id"], mapped, kpts_conf)

        totals["players"] += len(players)
        totals["pose_found"] += found
        totals["usable"] += usable
        print(f"[{ts}s] tracked players: {len(players)} | pose found: {found} | usable (>= {MIN_CONFIDENT_KEYPOINTS} keypoints): {usable}")

        cv2.imwrite(str(OUTPUT_DIR / f"pose_per_player_{int(ts)}s.jpg"), annotated)

    pipeline.release()

    with open(OUTPUT_DIR / "pose_per_player.json", "w") as f:
        json.dump(all_poses, f, indent=2)

    print("\n=== PER-PLAYER POSE SUMMARY ===")
    if totals["players"]:
        print(f"Tracked players across frames: {totals['players']}")
        print(f"Pose found:  {totals['pose_found']} ({totals['pose_found'] / totals['players'] * 100:.0f}%)")
        print(f"Usable pose: {totals['usable']} ({totals['usable'] / totals['players'] * 100:.0f}%)")
    print(f"\nSaved to: {(OUTPUT_DIR / 'pose_per_player.json').resolve()}")
    print("Skeleton images: data/analysis/context/pose_per_player_*.jpg")


if __name__ == "__main__":
    main()
