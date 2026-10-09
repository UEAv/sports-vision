"""
Stage 5 - Test Pretrained YOLOv8 on Football Footage

Runs the standard pretrained YOLOv8 model (trained on COCO - a general
80-class dataset, NOT football-specific) against a few sample frames,
to see what it detects out of the box before any fine-tuning.

Usage:
    python src/detection/test_pretrained_yolo.py
"""

from pathlib import Path
import sys
import cv2

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger
from src.pipeline.video_pipeline import VideoPipeline
from ultralytics import YOLO

logger = get_logger(__name__)

TEST_VIDEO = "data/soccernet/videos-224p/england_epl/2016-2017/2016-08-14 - 18-00 Arsenal 3 - 4 Liverpool/1_224p.mkv"
OUTPUT_DIR = Path("data/analysis/yolo_pretrained_test")
TEST_TIMESTAMPS = [90.0, 300.0, 600.0, 1200.0, 1800.0]  # spread across the video

# What we're HOPING to see, vs what COCO (pretrained YOLO's training data)
# actually contains. This mismatch is the whole point of this test.
TARGET_CLASSES = ["player", "ball", "goalkeeper", "referee"]
COCO_RELEVANT_CLASSES = ["person", "sports ball"]  # COCO has no football-specific classes


def main():
    logger.info("Loading pretrained YOLOv8n (downloads automatically on first run)...")
    model = YOLO("yolov8n.pt")  # smallest/fastest variant - good for CPU testing

    logger.info(f"COCO training classes relevant to football: {COCO_RELEVANT_CLASSES}")
    logger.info(f"What we actually need: {TARGET_CLASSES}")
    logger.info("NOTE: COCO has no 'goalkeeper' or 'referee' class - both would just show as 'person'.\n")

    pipeline = VideoPipeline(TEST_VIDEO)
    if not pipeline.load():
        logger.error("Could not load test video.")
        return
    pipeline.extract_metadata()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_detections = []

    for ts in TEST_TIMESTAMPS:
        frame = pipeline.get_frame_at_timestamp(ts)
        if frame is None:
            logger.warning(f"Could not get frame at {ts}s - skipping.")
            continue

        # Run inference - YOLO handles its own resizing/normalization internally,
        # so we pass the raw frame directly (no need for our preprocess_frame() here)
        results = model(frame, verbose=False)[0]

        # Count detections by class name
        detected_classes = {}
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            detected_classes[class_name] = detected_classes.get(class_name, 0) + 1

        logger.info(f"[{ts}s] Detections: {detected_classes if detected_classes else 'NONE'}")
        all_detections.append({"timestamp": ts, "detections": detected_classes})

        # Save BOTH versions: raw (so boxes don't hide the ball) and annotated
        # (so you can see what YOLO actually detected)
        raw_path = OUTPUT_DIR / f"frame_{int(ts)}s_raw.jpg"
        cv2.imwrite(str(raw_path), frame)

        annotated = results.plot()  # draws boxes + labels on the frame
        annotated_path = OUTPUT_DIR / f"frame_{int(ts)}s_annotated.jpg"
        cv2.imwrite(str(annotated_path), annotated)

    pipeline.release()

    print("\n=== SUMMARY ===")
    for d in all_detections:
        print(f"  {d['timestamp']}s: {d['detections']}")
    print(f"\nAnnotated frames saved to: {OUTPUT_DIR.resolve()}")
    print("Open these images and look for: missed small/blurry ball, players")
    print("merged into one box, 'person' label on everyone (no player/goalkeeper/referee distinction).")


if __name__ == "__main__":
    main()
