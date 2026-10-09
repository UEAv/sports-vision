"""
Stage 5 - Resolution Comparison: Can we even SEE the ball at 224p vs 720p?

Tests the same moment in a Ball Action Spotting video at both resolutions
to determine whether poor ball detection is a YOLO problem or a resolution
problem - these need completely different fixes.

Usage:
    python src/detection/compare_resolution_ball_visibility.py
"""

from pathlib import Path
import sys
import cv2

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger
from src.pipeline.video_pipeline import VideoPipeline
from ultralytics import YOLO

logger = get_logger(__name__)

# Ball Action Spotting gives us BOTH resolutions of the SAME match -
# a direct, fair comparison (unlike comparing two different matches)
VIDEO_224P = "data/ball_action/spotting-ball-2023/valid/england_efl/2019-2020/2019-10-01 - Middlesbrough - Preston North End/1_224p.mkv"
VIDEO_720P = "data/ball_action/spotting-ball-2023/valid/england_efl/2019-2020/2019-10-01 - Middlesbrough - Preston North End/1_720p.mkv"

OUTPUT_DIR = Path("data/analysis/resolution_comparison")
TEST_TIMESTAMPS = [120.0, 300.0, 600.0]


def test_video(model, video_path, label):
    pipeline = VideoPipeline(video_path)
    if not pipeline.load():
        logger.error(f"Could not load: {video_path}")
        return []

    pipeline.extract_metadata()
    results_log = []

    for ts in TEST_TIMESTAMPS:
        frame = pipeline.get_frame_at_timestamp(ts)
        if frame is None:
            continue

        h, w = frame.shape[:2]
        results = model(frame, verbose=False)[0]

        ball_detections = [b for b in results.boxes if model.names[int(b.cls[0])] == "sports ball"]
        ball_found = len(ball_detections) > 0
        ball_confidence = float(ball_detections[0].conf[0]) if ball_found else 0.0

        logger.info(f"[{label} @ {ts}s] Resolution: {w}x{h} | Ball detected: {ball_found} | Confidence: {ball_confidence:.2f}")
        results_log.append({"timestamp": ts, "resolution": f"{w}x{h}", "ball_found": ball_found, "confidence": ball_confidence})

        raw_path = OUTPUT_DIR / f"{label}_{int(ts)}s_raw.jpg"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(raw_path), frame)

        annotated_path = OUTPUT_DIR / f"{label}_{int(ts)}s_annotated.jpg"
        cv2.imwrite(str(annotated_path), results.plot())

    pipeline.release()
    return results_log


def main():
    model = YOLO("yolov8n.pt")

    print("\n=== Testing 224p ===")
    results_224p = test_video(model, VIDEO_224P, "224p")

    print("\n=== Testing 720p (same match, same timestamps) ===")
    results_720p = test_video(model, VIDEO_720P, "720p")

    print("\n=== COMPARISON ===")
    print(f"{'Timestamp':<12} {'224p ball found':<18} {'224p conf':<12} {'720p ball found':<18} {'720p conf':<12}")
    for r224, r720 in zip(results_224p, results_720p):
        print(f"{r224['timestamp']:<12} {str(r224['ball_found']):<18} {r224['confidence']:<12.2f} {str(r720['ball_found']):<18} {r720['confidence']:<12.2f}")

    print(f"\nImages saved to: {OUTPUT_DIR.resolve()}")
    print("Compare *_raw.jpg files side by side (224p vs 720p) for the same timestamp.")


if __name__ == "__main__":
    main()
