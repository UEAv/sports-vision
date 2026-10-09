"""
Stage 7 - Pose Estimation

Extracts: keypoints, body orientation, limb configuration.

IMPORTANT: uses YOLOv8-Pose, a DIFFERENT pretrained model than your
fine-tuned Stage 5 model - trained on COCO's generic 17-keypoint human
skeleton, not fine-tuned on football specifically. This is pretrained,
off-the-shelf - same starting point as your very first Stage 5 test.

COCO 17 keypoints (standard order):
  0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear,
  5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow,
  9=left_wrist, 10=right_wrist, 11=left_hip, 12=right_hip,
  13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle

Usage:
    python src/context/pose_estimation.py
"""

from pathlib import Path
import sys
import json
import math
import cv2

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger
from src.pipeline.video_pipeline import VideoPipeline
from ultralytics import YOLO

logger = get_logger(__name__)

TEST_VIDEO = "data/ball_action/spotting-ball-2023/valid/england_efl/2019-2020/2019-10-01 - Middlesbrough - Preston North End/1_720p.mkv"
OUTPUT_DIR = Path("data/analysis/context")
TEST_TIMESTAMPS = [120.0, 123.8, 130.0]  # a few sample frames, not the whole clip

LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_ANKLE, RIGHT_ANKLE = 15, 16


def compute_body_orientation(keypoints_xy):
    """
    Body orientation, approximated from the SHOULDER LINE angle.

    WHY SHOULDERS: the line between left and right shoulder rotates as a
    person turns - facing the camera gives a roughly horizontal line,
    turning sideways makes it foreshorten/rotate. This is an approximation
    of facing direction, not a precise 3D measurement - genuinely turning
    toward/away from camera can't be fully captured from 2D keypoints alone.
    """
    left = keypoints_xy[LEFT_SHOULDER]
    right = keypoints_xy[RIGHT_SHOULDER]
    if left[0] == 0 and left[1] == 0:
        return None  # keypoint not detected
    if right[0] == 0 and right[1] == 0:
        return None

    dx = right[0] - left[0]
    dy = right[1] - left[1]
    return round(math.degrees(math.atan2(dy, dx)), 1)


def compute_limb_configuration(keypoints_xy):
    """
    Simple limb configuration proxies - not a full biomechanical analysis,
    just two useful, easy-to-compute signals:

    LEG SPREAD: distance between ankles, relative to hip width. A high
    ratio suggests a running/lunging stance; near 0 suggests standing
    with feet together.

    ARM RAISE: whether wrists are ABOVE shoulders (image y is smaller =
    higher on screen) - useful for spotting goalkeeper dives, headers,
    celebrations, etc. later.
    """
    hip_width = abs(keypoints_xy[LEFT_HIP][0] - keypoints_xy[RIGHT_HIP][0])
    ankle_distance = ((keypoints_xy[LEFT_ANKLE][0] - keypoints_xy[RIGHT_ANKLE][0]) ** 2 +
                       (keypoints_xy[LEFT_ANKLE][1] - keypoints_xy[RIGHT_ANKLE][1]) ** 2) ** 0.5

    leg_spread_ratio = round(ankle_distance / hip_width, 2) if hip_width > 1 else None

    left_wrist_y = keypoints_xy[9][1]
    right_wrist_y = keypoints_xy[10][1]
    left_shoulder_y = keypoints_xy[LEFT_SHOULDER][1]
    right_shoulder_y = keypoints_xy[RIGHT_SHOULDER][1]

    left_arm_raised = left_wrist_y < left_shoulder_y if left_wrist_y > 0 else None
    right_arm_raised = right_wrist_y < right_shoulder_y if right_wrist_y > 0 else None

    return {
        "leg_spread_ratio": leg_spread_ratio,
        "left_arm_raised": left_arm_raised,
        "right_arm_raised": right_arm_raised,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading YOLOv8-Pose (pretrained on COCO, downloads on first run)...")
    model = YOLO("yolov8n-pose.pt")

    pipeline = VideoPipeline(TEST_VIDEO)
    if not pipeline.load():
        return
    pipeline.extract_metadata()

    all_poses = []

    for ts in TEST_TIMESTAMPS:
        frame = pipeline.get_frame_at_timestamp(ts)
        if frame is None:
            continue

        results = model(frame, verbose=False)[0]

        if results.keypoints is None or len(results.keypoints.xy) == 0:
            logger.warning(f"No poses detected at {ts}s")
            continue

        for person_idx, (kpts_xy, kpts_conf) in enumerate(zip(results.keypoints.xy, results.keypoints.conf)):
            kpts_xy = kpts_xy.tolist()
            kpts_conf = kpts_conf.tolist()

            orientation = compute_body_orientation(kpts_xy)
            limb_config = compute_limb_configuration(kpts_xy)

            all_poses.append({
                "timestamp_sec": ts,
                "person_index": person_idx,
                "keypoints_xy": kpts_xy,
                "keypoints_confidence": [round(c, 3) for c in kpts_conf],
                "avg_keypoint_confidence": round(sum(kpts_conf) / len(kpts_conf), 3),
                "body_orientation_degrees": orientation,
                "limb_configuration": limb_config,
            })

        # Save an annotated frame showing the skeleton, so you can visually verify
        annotated = results.plot()
        out_path = OUTPUT_DIR / f"pose_sample_{int(ts)}s.jpg"
        cv2.imwrite(str(out_path), annotated)

    pipeline.release()

    output_path = OUTPUT_DIR / "pose_estimation.json"
    with open(output_path, "w") as f:
        json.dump(all_poses, f, indent=2)

    print(f"\n=== POSE ESTIMATION SUMMARY ===")
    print(f"Total poses detected: {len(all_poses)}")

    avg_confs = [p["avg_keypoint_confidence"] for p in all_poses]
    if avg_confs:
        print(f"Avg keypoint confidence: {sum(avg_confs)/len(avg_confs):.3f}")

    print(f"\nSaved to: {output_path.resolve()}")
    print(f"Saved annotated skeleton images to: {OUTPUT_DIR.resolve()} (pose_sample_*.jpg)")
    print("\nSample pose:")
    if all_poses:
        sample = all_poses[0]
        print(f"  timestamp={sample['timestamp_sec']}s, orientation={sample['body_orientation_degrees']}°, "
              f"limb_config={sample['limb_configuration']}, avg_conf={sample['avg_keypoint_confidence']}")


if __name__ == "__main__":
    main()
