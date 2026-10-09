"""
Video Property Analysis

Scans all video files under the project's data/ folder and extracts:
resolution, FPS, duration, frame count, codec, and aspect ratio.

"""

import cv2
from pathlib import Path
import pandas as pd
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger

logger = get_logger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi"}
DATA_ROOT = Path("data")
OUTPUT_CSV = Path("data/analysis/video_properties.csv")


def fourcc_to_string(fourcc_int: int) -> str:
    """Convert OpenCV's numeric fourcc code into a readable 4-character codec string."""
    return "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).strip()


def analyze_video(path: Path) -> dict:
    """Extract core properties from a single video file."""
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        logger.warning(f"Could not open: {path}")
        return {
            "file": str(path),
            "status": "FAILED_TO_OPEN",
        }

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = fourcc_to_string(fourcc_int) if fourcc_int else "UNKNOWN"

    duration_sec = frame_count / fps if fps > 0 else 0
    aspect_ratio = round(width / height, 3) if height > 0 else 0

    cap.release()

    return {
        "file": str(path),
        "status": "OK",
        "resolution": f"{width}x{height}",
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "duration_sec": round(duration_sec, 2),
        "duration_min": round(duration_sec / 60, 2),
        "codec": codec,
        "aspect_ratio": aspect_ratio,
    }


def find_video_files(root: Path) -> list[Path]:
    """Recursively find all video files under root, skipping cache folders."""
    files = []
    for ext in VIDEO_EXTENSIONS:
        files.extend(root.rglob(f"*{ext}"))
    # Skip HF cache junk and hidden folders
    return [f for f in files if ".cache" not in f.parts and not any(p.startswith(".") for p in f.parts)]


def main():
    logger.info(f"Scanning for video files under: {DATA_ROOT.resolve()}")
    video_files = find_video_files(DATA_ROOT)
    logger.info(f"Found {len(video_files)} video files.")

    results = []
    for i, path in enumerate(video_files, 1):
        logger.info(f"[{i}/{len(video_files)}] Analyzing: {path.name}")
        results.append(analyze_video(path))

    df = pd.DataFrame(results)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Saved report to: {OUTPUT_CSV.resolve()}")

    # Quick summary printed to console
    ok_df = df[df["status"] == "OK"]
    if not ok_df.empty:
        print("\n=== SUMMARY ===")
        print(f"Total videos analyzed: {len(ok_df)}")
        print(f"Unique resolutions: {ok_df['resolution'].unique().tolist()}")
        print(f"Unique FPS values: {ok_df['fps'].unique().tolist()}")
        print(f"Unique codecs: {ok_df['codec'].unique().tolist()}")
        print(f"Total duration (all videos): {ok_df['duration_min'].sum():.1f} minutes")
    else:
        print("No videos successfully analyzed - check the log above for errors.")


if __name__ == "__main__":
    main()
