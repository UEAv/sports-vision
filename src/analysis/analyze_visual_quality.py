"""
Stage 3 - Visual Quality Analysis

Computes quantitative proxies for video quality characteristics:
- lighting (mean brightness + variation)
- camera movement (optical flow magnitude between sampled frames)
- stabilization (consistency of that motion - high variance = shaky/erratic)

NOTE: zoom, occlusion, and ball visibility have no reliable automated proxy
and are intentionally left for manual inspection (see the companion
manual-inspection step). This script narrows down WHICH videos most need
that manual look, rather than replacing it.

Usage:
    python src/analysis/analyze_visual_quality.py
"""

import cv2
import numpy as np
from pathlib import Path
import pandas as pd
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger

logger = get_logger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi"}
DATA_ROOT = Path("data")
OUTPUT_CSV = Path("data/analysis/visual_quality.csv")

SAMPLE_FRAMES = 20          # how many frames to sample per video
RESIZE_WIDTH = 320          # downscale before optical flow, for speed


def find_video_files(root: Path) -> list[Path]:
    files = []
    for ext in VIDEO_EXTENSIONS:
        files.extend(root.rglob(f"*{ext}"))
    return [f for f in files if ".cache" not in f.parts and not any(p.startswith(".") for p in f.parts)]


def resize_frame(frame):
    h, w = frame.shape[:2]
    scale = RESIZE_WIDTH / w
    return cv2.resize(frame, (RESIZE_WIDTH, int(h * scale)))


def analyze_video_quality(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"file": str(path), "status": "FAILED_TO_OPEN"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < SAMPLE_FRAMES + 1:
        cap.release()
        return {"file": str(path), "status": "TOO_SHORT"}

    sample_indices = np.linspace(0, total_frames - 1, SAMPLE_FRAMES, dtype=int)

    brightness_values = []
    flow_magnitudes = []
    prev_gray = None

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame = resize_frame(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Lighting: mean pixel brightness (0-255 scale)
        brightness_values.append(np.mean(gray))

        # Camera movement: optical flow magnitude vs previous sampled frame
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=2, winsize=15,
                iterations=2, poly_n=5, poly_sigma=1.1, flags=0
            )
            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            flow_magnitudes.append(np.mean(magnitude))

        prev_gray = gray

    cap.release()

    if not brightness_values:
        return {"file": str(path), "status": "NO_FRAMES_READ"}

    avg_brightness = float(np.mean(brightness_values))
    brightness_std = float(np.std(brightness_values))
    avg_motion = float(np.mean(flow_magnitudes)) if flow_magnitudes else 0.0
    motion_std = float(np.std(flow_magnitudes)) if flow_magnitudes else 0.0

    # Rough qualitative labels from the numbers (thresholds are starting points -
    # adjust after comparing a few against your own manual viewing)
    lighting_label = "dark" if avg_brightness < 80 else "bright" if avg_brightness > 180 else "normal"
    movement_label = "high_movement" if avg_motion > 2.0 else "low_movement"
    stability_label = "unstable/shaky" if motion_std > 1.5 else "stable"

    return {
        "file": str(path),
        "status": "OK",
        "avg_brightness": round(avg_brightness, 1),
        "brightness_std": round(brightness_std, 1),
        "lighting_label": lighting_label,
        "avg_camera_motion": round(avg_motion, 3),
        "motion_std": round(motion_std, 3),
        "movement_label": movement_label,
        "stability_label": stability_label,
    }


def main():
    logger.info(f"Scanning for video files under: {DATA_ROOT.resolve()}")
    video_files = find_video_files(DATA_ROOT)
    logger.info(f"Found {len(video_files)} video files. Sampling {SAMPLE_FRAMES} frames each.")

    results = []
    for i, path in enumerate(video_files, 1):
        logger.info(f"[{i}/{len(video_files)}] Analyzing: {path.name}")
        results.append(analyze_video_quality(path))

    df = pd.DataFrame(results)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Saved report to: {OUTPUT_CSV.resolve()}")

    ok_df = df[df["status"] == "OK"]
    if not ok_df.empty:
        print("\n=== SUMMARY ===")
        print(f"Videos analyzed: {len(ok_df)}")
        print(f"\nLighting breakdown:\n{ok_df['lighting_label'].value_counts().to_string()}")
        print(f"\nMovement breakdown:\n{ok_df['movement_label'].value_counts().to_string()}")
        print(f"\nStability breakdown:\n{ok_df['stability_label'].value_counts().to_string()}")
        print("\nFlagged for manual review (unstable/shaky):")
        shaky = ok_df[ok_df["stability_label"] == "unstable/shaky"]
        for _, row in shaky.iterrows():
            print(f"  {row['file']}")
    else:
        print("No videos successfully analyzed.")

    print("\nNOTE: zoom, occlusion, and ball visibility require manual inspection -")
    print("no automated proxy is computed for these. Use this report to prioritize")
    print("WHICH videos to watch first (e.g. start with flagged unstable ones).")


if __name__ == "__main__":
    main()
