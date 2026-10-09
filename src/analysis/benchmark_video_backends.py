"""
Stage 4 - Video Backend Benchmark

Compares OpenCV, PyAV, and FFmpeg (CLI) on three tasks:
  1. Video loading (time to open/initialize)
  2. Frame decoding (time to decode N frames sequentially)
  3. Preprocessing (time to resize those N frames)

Usage:
    python src/analysis/benchmark_video_backends.py
"""

import time
import subprocess
import numpy as np
import cv2
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger

logger = get_logger(__name__)

N_FRAMES = 200
RESIZE_TO = (224, 224)

# Pick one real, representative video to benchmark against
TEST_VIDEO = Path("data/soccernet/videos-224p/england_epl/2016-2017/2016-08-14 - 18-00 Arsenal 3 - 4 Liverpool/1_224p.mkv")


def bench_opencv(path: Path):
    t0 = time.perf_counter()
    cap = cv2.VideoCapture(str(path))
    load_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    frames = []
    for _ in range(N_FRAMES):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    decode_time = time.perf_counter() - t0
    cap.release()

    t0 = time.perf_counter()
    for f in frames:
        cv2.resize(f, RESIZE_TO)
    preprocess_time = time.perf_counter() - t0

    return load_time, decode_time, preprocess_time, len(frames)


def bench_pyav(path: Path):
    import av

    t0 = time.perf_counter()
    container = av.open(str(path))
    stream = container.streams.video[0]
    load_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    frames = []
    for i, frame in enumerate(container.decode(stream)):
        if i >= N_FRAMES:
            break
        frames.append(frame.to_ndarray(format="bgr24"))
    decode_time = time.perf_counter() - t0
    container.close()

    t0 = time.perf_counter()
    for f in frames:
        cv2.resize(f, RESIZE_TO)
    preprocess_time = time.perf_counter() - t0

    return load_time, decode_time, preprocess_time, len(frames)


def bench_ffmpeg_cli(path: Path):
    # "Loading" for the CLI approach = probing the video for width/height (needed to parse raw output)
    t0 = time.perf_counter()
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True
    )
    width, height = map(int, probe.stdout.strip().split(","))
    load_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    cmd = [
        "ffmpeg", "-i", str(path), "-vframes", str(N_FRAMES),
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-"
    ]
    result = subprocess.run(cmd, capture_output=True)
    frame_size = width * height * 3
    raw = result.stdout
    n_decoded = len(raw) // frame_size
    frames = [
        np.frombuffer(raw[i * frame_size:(i + 1) * frame_size], dtype=np.uint8).reshape((height, width, 3))
        for i in range(n_decoded)
    ]
    decode_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    for f in frames:
        cv2.resize(f, RESIZE_TO)
    preprocess_time = time.perf_counter() - t0

    return load_time, decode_time, preprocess_time, len(frames)


def main():
    if not TEST_VIDEO.exists():
        logger.error(f"Test video not found: {TEST_VIDEO}")
        logger.error("Update TEST_VIDEO in this script to point at a real video file you have.")
        return

    logger.info(f"Benchmarking against: {TEST_VIDEO.name}")
    logger.info(f"Decoding {N_FRAMES} frames, resizing to {RESIZE_TO}\n")

    results = {}

    logger.info("Running OpenCV benchmark...")
    results["OpenCV"] = bench_opencv(TEST_VIDEO)

    logger.info("Running PyAV benchmark...")
    try:
        results["PyAV"] = bench_pyav(TEST_VIDEO)
    except Exception as e:
        logger.error(f"PyAV failed: {e}")
        results["PyAV"] = None

    logger.info("Running FFmpeg CLI benchmark...")
    try:
        results["FFmpeg"] = bench_ffmpeg_cli(TEST_VIDEO)
    except Exception as e:
        logger.error(f"FFmpeg failed: {e}")
        results["FFmpeg"] = None

    print("\n=== BENCHMARK RESULTS ===")
    print(f"{'Backend':<10} {'Load (s)':<12} {'Decode (s)':<12} {'Preprocess (s)':<16} {'Total (s)':<12} {'Frames':<8}")
    for name, r in results.items():
        if r is None:
            print(f"{name:<10} FAILED")
            continue
        load_t, decode_t, prep_t, n = r
        total = load_t + decode_t + prep_t
        print(f"{name:<10} {load_t:<12.4f} {decode_t:<12.4f} {prep_t:<16.4f} {total:<12.4f} {n:<8}")

    valid = {k: v for k, v in results.items() if v is not None}
    if valid:
        fastest = min(valid.items(), key=lambda kv: sum(kv[1][:3]))
        print(f"\nFastest overall: {fastest[0]}")


if __name__ == "__main__":
    main()
