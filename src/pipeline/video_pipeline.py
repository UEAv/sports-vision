"""
Stage 4 - Reusable Video Processing Pipeline

Chunk 1: Loading, Validation, Metadata Extraction
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VideoMetadata:
    """Container for a video's core properties. A dataclass is just a clean
    way to bundle related values together instead of passing 5 separate variables around."""
    path: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_sec: float
    is_valid: bool


class VideoPipeline:
    """
    Reusable video processing pipeline. Wraps OpenCV (chosen via benchmark:
    ~7x faster than PyAV, ~25x faster than FFmpeg CLI for frame-by-frame access).
    """

    def __init__(self, video_path: str):
        self.path = Path(video_path)
        self.cap = None
        self.metadata = None

    def load(self) -> bool:
        """
        Open the video file. Returns True if successful, False otherwise.

        WHY a separate load() method instead of doing this in __init__:
        opening a video can fail (missing file, corrupted file, unsupported
        codec). Separating load() from __init__ lets us handle that failure
        gracefully, log it clearly, and let the caller decide what to do -
        rather than crashing the moment someone creates a VideoPipeline object.
        """
        if not self.path.exists():
            logger.error(f"File does not exist: {self.path}")
            return False

        self.cap = cv2.VideoCapture(str(self.path))

        if not self.cap.isOpened():
            logger.error(f"OpenCV could not open: {self.path}")
            return False

        logger.info(f"Loaded: {self.path.name}")
        return True

    def validate(self) -> bool:
        """
        Check the video is actually usable, not just "opened".

        WHY this is separate from load(): a file can technically open
        (isOpened() == True) but still be broken - zero frames, zero fps,
        or a resolution of 0x0 from a corrupted header. load() answers
        "did OpenCV accept this file?" - validate() answers "is this data
        actually trustworthy enough to build a pipeline on?"
        """
        if self.cap is None:
            logger.error("Cannot validate - call load() first.")
            return False

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        problems = []
        if fps <= 0:
            problems.append(f"invalid fps ({fps})")
        if frame_count <= 0:
            problems.append(f"invalid frame_count ({frame_count})")
        if width <= 0 or height <= 0:
            problems.append(f"invalid resolution ({width}x{height})")

        if problems:
            logger.warning(f"Validation failed for {self.path.name}: {', '.join(problems)}")
            return False

        return True

    def extract_metadata(self) -> VideoMetadata:
        """
        Pull FPS, resolution, frame count, and duration into one clean object.

        THE KEY FORMULA (you'll use this constantly from here on):
            duration_seconds = frame_count / fps

        Plain English: if a video has 1500 frames and plays at 25 frames
        per second, it takes 1500/25 = 60 seconds to watch. This single
        formula is the bridge between "frame numbers" and "real time" -
        everything else in this pipeline (sampling, windows, clipping)
        builds on it.
        """
        if self.cap is None:
            raise RuntimeError("Call load() before extract_metadata().")

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = frame_count / fps if fps > 0 else 0.0
        is_valid = self.validate()

        self.metadata = VideoMetadata(
            path=str(self.path),
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
            duration_sec=duration_sec,
            is_valid=is_valid,
        )
        return self.metadata

    def release(self):
        """Free the video file handle. Always call this when done -
        leaving handles open can lock the file or leak memory over many videos."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # =====================================================
    # Chunk 2: Frame extraction, indexing, timestamp <-> frame
    # =====================================================

    def get_frame(self, frame_index: int):
        """
        Retrieve ONE specific frame by its index (0-based).

        HOW THIS WORKS UNDER THE HOOD:
        cv2.VideoCapture normally reads frames sequentially, one after another,
        like pressing "play" - it doesn't naturally jump around. CAP_PROP_POS_FRAMES
        is how we tell it "skip ahead (or back) to this exact frame number first."
        Internally OpenCV seeks to the nearest keyframe and decodes forward from
        there - so seeking is NOT free, it costs some time (more on this in the
        benchmark chunk later). For occasional single-frame lookups, this is fine.
        For extracting MANY frames in order, sequential reading (like Chunk 1's
        test) is faster than seeking to each one individually.
        """
        if self.cap is None:
            raise RuntimeError("Call load() before get_frame().")

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()

        if not ret:
            logger.warning(f"Could not read frame {frame_index} from {self.path.name}")
            return None

        return frame

    def frame_to_timestamp(self, frame_index: int) -> float:
        """
        Convert a frame index into a real-world timestamp, in seconds.

        THE FORMULA (same one from Chunk 1, just rearranged):
            timestamp_seconds = frame_index / fps

        Plain English: "frame number 1500, at 25 fps, happened at the
        60-second mark." This is literally duration math applied to a
        single frame instead of the whole video.
        """
        if self.metadata is None:
            raise RuntimeError("Call extract_metadata() first.")
        return frame_index / self.metadata.fps

    def timestamp_to_frame(self, timestamp_sec: float) -> int:
        """
        Convert a real-world timestamp (seconds) into the nearest frame index.
        This is the INVERSE of frame_to_timestamp() - same formula, solved
        for the other variable:

            frame_index = timestamp_seconds * fps

        Plain English: "I want the frame at 60 seconds. At 25 fps, that's
        60 * 25 = frame 1500." We round to the nearest whole frame since
        you can't have "frame 1500.5".

        WHY YOU NEED BOTH DIRECTIONS: annotation files (like SoccerNet's
        gameTime "1 - 23:45") give you TIME. Your video pipeline works in
        FRAMES. These two functions are the translation layer between
        "what a human/annotation says" and "what the video actually stores."
        """
        if self.metadata is None:
            raise RuntimeError("Call extract_metadata() first.")
        return round(timestamp_sec * self.metadata.fps)

    def get_frame_at_timestamp(self, timestamp_sec: float):
        """Convenience method chaining the two ideas above: give a timestamp,
        get the actual frame back directly."""
        frame_index = self.timestamp_to_frame(timestamp_sec)
        return self.get_frame(frame_index)

    # =====================================================
    # Chunk 3: Sampling, resizing, normalization, batching
    # =====================================================

    def sample_frames(self, interval: int = 5, max_frames: int = None):
        """
        Extract every Nth frame SEQUENTIALLY (not by seeking each one -
        remember from Chunk 2 that seeking has overhead; reading forward
        and just keeping every Nth frame is much faster for bulk sampling).

        WHY SAMPLE AT ALL: a 47-minute half at 25fps is 70,450 frames.
        Feeding every single frame into a model is wasteful - consecutive
        frames look almost identical. Sampling every 5th frame (interval=5)
        cuts your data to 20% of the size with minimal information loss,
        since 1/5th of a second rarely changes much in a football match.

        Returns a list of (frame_index, frame) tuples - keeping the index
        attached matters, because you'll need it later to convert back to
        a timestamp for labeling.
        """
        if self.cap is None:
            raise RuntimeError("Call load() before sample_frames().")

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # start from the beginning
        sampled = []
        frame_idx = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            if frame_idx % interval == 0:
                sampled.append((frame_idx, frame))
                if max_frames is not None and len(sampled) >= max_frames:
                    break

            frame_idx += 1

        logger.info(f"Sampled {len(sampled)} frames (every {interval}th frame)")
        return sampled

    @staticmethod
    def resize_frame(frame, target_size=(224, 224)):
        """
        Resize a frame to a fixed size.

        WHY THIS MATTERS: your videos come in at LEAST 4 different resolutions
        (confirmed in Stage 3 - 398x224, 1280x720, 1920x1080, 1920x1160).
        YOLO and most deep learning models need every input to be the EXACT
        same dimensions, because the model's internal math (matrix operations)
        is built around fixed-size tensors. This is the standardization step
        that makes mixed-resolution footage usable together.
        """
        return cv2.resize(frame, target_size)

    @staticmethod
    def normalize_frame(frame):
        """
        Scale pixel values from the 0-255 integer range down to 0.0-1.0 floats.

        THE MATH: normalized_pixel = raw_pixel / 255.0

        WHY: raw pixel values (0-255) are fine for humans/displays, but
        neural networks train far more reliably on small, consistent number
        ranges. Feeding in values up to 255 can cause unstable gradients
        during training - squeezing everything into 0.0-1.0 is a near-universal
        preprocessing step before data reaches a model.

        NOTE: OpenCV loads frames as BGR (Blue-Green-Red) by default, not RGB.
        Most pretrained models (including YOLO) expect RGB - we convert that
        here too, so this function handles BOTH standardizations at once.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb.astype("float32") / 255.0

    def preprocess_frame(self, frame, target_size=(224, 224)):
        """Convenience method: resize + normalize in one call, since you'll
        almost always want both done together before a frame reaches a model."""
        resized = self.resize_frame(frame, target_size)
        return self.normalize_frame(resized)

    @staticmethod
    def batch_frames(frames: list):
        """
        Stack a list of individual frames into one single array - a "batch".

        WHY BATCHING EXISTS: models process many frames at once far more
        efficiently than one at a time, because modern hardware (even CPUs,
        and especially GPUs) is built to do the same matrix operation on
        many pieces of data simultaneously rather than looping one-by-one.

        SHAPE EXPLANATION: if you have 32 frames, each 224x224 pixels with
        3 color channels, stacking them gives you one array of shape
        (32, 224, 224, 3) - read as "32 frames, each 224 tall, 224 wide,
        3 color channels." This 4D shape is the standard format most
        vision models expect as input.
        """
        return np.stack(frames, axis=0)

    # =====================================================
    # Chunk 4: Video clipping, temporal windows
    # =====================================================

    def extract_clip(self, start_sec: float, end_sec: float):
        """
        Extract every frame between two timestamps - a "clip".

        HOW: convert both timestamps to frame indices (Chunk 2's
        timestamp_to_frame), then read sequentially from start to end.
        Sequential reading is used here too, same reasoning as sample_frames -
        faster than seeking to every single frame in the range.

        USE CASE: SoccerNet's Ball Action Spotting data uses exactly this idea -
        "10-second clips centered on an event." This method is how you'd
        reproduce that yourself from any timestamp.
        """
        start_frame = self.timestamp_to_frame(start_sec)
        end_frame = self.timestamp_to_frame(end_sec)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        clip_frames = []

        for _ in range(end_frame - start_frame):
            ret, frame = self.cap.read()
            if not ret:
                break
            clip_frames.append(frame)

        return clip_frames

    def extract_temporal_window(self, center_sec: float, window_sec: float = 5.0):
        """
        Extract a clip CENTERED on a timestamp, spanning window_sec total.

        THE MATH:
            start = center_sec - (window_sec / 2)
            end   = center_sec + (window_sec / 2)

        Plain English: if an event happens at 34:57 and you want a 10-second
        window around it, you grab from 34:52 to 35:02 - 5 seconds before,
        5 seconds after. This is exactly what Phase 7-9 (action spotting)
        needs: the model doesn't just see the exact instant of a "goal",
        it sees the build-up and aftermath too, which is what makes the
        event recognizable.

        clamp to 0: if an event happens very early (e.g. center_sec=2),
        start would go negative - which doesn't exist in a video. We clamp
        it to 0 instead of letting it error out.
        """
        half_window = window_sec / 2
        start_sec = max(0, center_sec - half_window)
        end_sec = center_sec + half_window

        return self.extract_clip(start_sec, end_sec)


# --- Quick manual test, so you can SEE this working before we build more ---
if __name__ == "__main__":
    test_video = "data/soccernet/videos-224p/england_epl/2016-2017/2016-08-14 - 18-00 Arsenal 3 - 4 Liverpool/1_224p.mkv"

    pipeline = VideoPipeline(test_video)

    if pipeline.load():
        meta = pipeline.extract_metadata()
        print(f"\nMetadata for {Path(test_video).name}:")
        print(f"  FPS: {meta.fps}")
        print(f"  Frame count: {meta.frame_count}")
        print(f"  Resolution: {meta.width}x{meta.height}")
        print(f"  Duration: {meta.duration_sec:.1f} sec ({meta.duration_sec/60:.1f} min)")
        print(f"  Valid: {meta.is_valid}")

        # --- Chunk 2 test: timestamp <-> frame conversion ---
        print("\n--- Chunk 2: Timestamp <-> Frame Conversion ---")
        test_timestamp = 90.0  # 1 minute 30 seconds into the video
        frame_idx = pipeline.timestamp_to_frame(test_timestamp)
        print(f"  Timestamp {test_timestamp}s -> frame index {frame_idx}")

        back_to_time = pipeline.frame_to_timestamp(frame_idx)
        print(f"  Frame index {frame_idx} -> timestamp {back_to_time}s (should match {test_timestamp}s)")

        frame = pipeline.get_frame_at_timestamp(test_timestamp)
        if frame is not None:
            print(f"  Retrieved actual frame at {test_timestamp}s - shape: {frame.shape} (height, width, color channels)")

        # --- Chunk 3 test: sampling, resizing, normalization, batching ---
        print("\n--- Chunk 3: Sampling, Resizing, Normalization, Batching ---")
        sampled = pipeline.sample_frames(interval=50, max_frames=8)  # small sample for a quick test
        print(f"  Sampled {len(sampled)} frames, indices: {[idx for idx, _ in sampled]}")

        preprocessed = [pipeline.preprocess_frame(f) for _, f in sampled]
        print(f"  After preprocessing, each frame shape: {preprocessed[0].shape}, dtype: {preprocessed[0].dtype}")
        print(f"  Pixel value range after normalization: min={preprocessed[0].min():.3f}, max={preprocessed[0].max():.3f}")

        batch = pipeline.batch_frames(preprocessed)
        print(f"  Batched shape: {batch.shape} (batch_size, height, width, channels)")

        # --- Chunk 4 test: clipping and temporal windows ---
        print("\n--- Chunk 4: Clipping and Temporal Windows ---")
        clip = pipeline.extract_clip(start_sec=90.0, end_sec=93.0)
        print(f"  Clip from 90.0s-93.0s: {len(clip)} frames (expect ~{int((93.0-90.0)*meta.fps)})")

        window = pipeline.extract_temporal_window(center_sec=90.0, window_sec=4.0)
        print(f"  Temporal window centered at 90.0s, 4.0s wide: {len(window)} frames (expect ~{int(4.0*meta.fps)})")

        edge_window = pipeline.extract_temporal_window(center_sec=1.0, window_sec=4.0)
        print(f"  Edge case - window near video start (center=1.0s) clamps instead of erroring: {len(edge_window)} frames")

        pipeline.release()
