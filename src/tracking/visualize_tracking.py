"""
Stage 6 - Visualize Tracking

Runs tracking on a short clip and saves an ANNOTATED VIDEO (boxes + track IDs
drawn on every frame) so you can actually watch tracking happen, rather than
reading raw numbers. This is the fastest way to visually judge whether
ID switches/ghost tracks are a real problem or just numbers on paper.

Usage:
    python src/tracking/visualize_tracking.py
"""

from pathlib import Path
import sys
import cv2

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger
from ultralytics import YOLO

logger = get_logger(__name__)

MODEL_PATH = "models/yolov8_football_finetune_v1.pt"
TEST_VIDEO = "data/soccernet/videos-224p/england_epl/2016-2017/2016-08-14 - 18-00 Arsenal 3 - 4 Liverpool/1_224p.mkv"
OUTPUT_DIR = Path("data/analysis/tracking")

# Try a few different timestamps if one lands on a broadcast cutaway
# (crowd/sideline/replay shots) instead of continuous on-pitch play.
START_SEC = 600.0  # 10 minutes in - avoids kickoff-area camera cuts
DURATION_SEC = 15.0
TRACKER_CONFIG = "configs/trackers/bytetrack_tuned.yaml"  # swap to botsort_tuned.yaml to compare
IOU_THRESHOLD = 0.5

# Ultralytics' default plot() auto-scales line thickness/font to image size -
# at this resolution that works out to roughly line_width=2, font_size~12.
# ~60% reduction from those defaults:
LINE_WIDTH = 1      # was ~2
FONT_SIZE = 5        # was ~12


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(MODEL_PATH)
    fps = 25.0

    start_frame = int(START_SEC * fps)
    num_frames = int(DURATION_SEC * fps)

    cap = cv2.VideoCapture(TEST_VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Upscale the output - 398x224 is too small to comfortably watch with
    # readable ID labels on screen. 2x makes it actually watchable.
    scale = 2
    out_width, out_height = width * scale, height * scale

    # Use just the filename stem (e.g. "bytetrack_tuned"), not the full config
    # path - using the full path here was creating a broken nested folder
    # structure instead of a clean output filename.
    tracker_label = Path(TRACKER_CONFIG).stem
    output_path = OUTPUT_DIR / f"tracking_visualization_{tracker_label}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_width, out_height))

    logger.info(f"Generating annotated video with {TRACKER_CONFIG} (iou={IOU_THRESHOLD})...")

    for frame_offset in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, tracker=TRACKER_CONFIG, persist=True, verbose=False, iou=IOU_THRESHOLD)[0]

        # plot() draws boxes + track IDs + class labels directly on the frame.
        # line_width/font_size override the auto-scaled defaults, which were
        # too thick/large and blocking the (already small) players on screen.
        annotated = results.plot(line_width=LINE_WIDTH, font_size=FONT_SIZE)
        annotated_upscaled = cv2.resize(annotated, (out_width, out_height))

        writer.write(annotated_upscaled)

        if frame_offset % 50 == 0:
            logger.info(f"  Processed {frame_offset}/{num_frames} frames...")

    cap.release()
    writer.release()

    print(f"\nSaved annotated video to: {output_path.resolve()}")
    print("Open it and watch for: boxes flickering/jumping between IDs,")
    print("two boxes on one player, boxes disappearing/reappearing with a new number.")


if __name__ == "__main__":
    main()
