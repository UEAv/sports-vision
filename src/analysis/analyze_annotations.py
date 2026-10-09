"""
Annotation Analysis

Parses the Roots & Rise IQ JSON annotation files and computes:
- annotation format summary (fields present)
- event label / class distribution
- events per video
- timestamp distribution
- class imbalance ratio

Usage:
    python src/analysis/analyze_annotations.py
"""

import json
from pathlib import Path
import pandas as pd
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger

logger = get_logger(__name__)

ROOTS_RISE_DIR = Path("data/roots_rise")
OUTPUT_DIR = Path("data/analysis")

# Your project's target 15-class taxonomy
TARGET_CLASSES = [
    "pass", "pass_received", "recovery", "tackle", "interception",
    "ball_out_of_play", "clearance", "take_on", "substitution", "block",
    "aerial_duel", "shot", "save", "foul", "goal",
]


def load_all_events() -> pd.DataFrame:
    """Load every event from every Roots & Rise IQ JSON file into one DataFrame."""
    json_files = list(ROOTS_RISE_DIR.glob("*.json"))
    logger.info(f"Found {len(json_files)} annotation JSON files.")

    all_events = []
    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            events = json.load(f)
        for e in events:
            e["source_file"] = jf.name
            e["game_id_from_file"] = jf.stem
        all_events.extend(events)

    df = pd.DataFrame(all_events)
    logger.info(f"Loaded {len(df)} total events across {len(json_files)} files.")
    return df


def summarize_format(df: pd.DataFrame):
    """Print the annotation format: what fields exist, sample values, types."""
    print("\n=== ANNOTATION FORMAT ===")
    print(f"Fields present ({len(df.columns)}): {list(df.columns)}\n")
    for col in df.columns:
        sample = df[col].dropna().iloc[0] if df[col].notna().any() else "N/A"
        print(f"  {col:30s} | dtype={str(df[col].dtype):10s} | example={sample}")


def analyze_event_classes(df: pd.DataFrame):
    """Compute samples per class, percentages, and imbalance ratio."""
    print("\n=== EVENT TYPE DISTRIBUTION (raw 'type' field) ===")
    counts = df["type"].value_counts()
    percentages = (counts / counts.sum() * 100).round(2)

    summary = pd.DataFrame({"count": counts, "percentage": percentages})
    print(summary.to_string())

    print(f"\nTotal distinct event types found: {df['type'].nunique()}")
    print(f"Most common: {counts.idxmax()} ({counts.max()} events)")
    print(f"Least common: {counts.idxmin()} ({counts.min()} events)")
    imbalance_ratio = round(counts.max() / counts.min(), 1)
    print(f"Class imbalance ratio (max/min): {imbalance_ratio}:1")

    # Cross-check against your target 15-class taxonomy
    found_target_classes = [c for c in TARGET_CLASSES if c in counts.index]
    missing_target_classes = [c for c in TARGET_CLASSES if c not in counts.index]
    print(f"\nTarget classes FOUND in this dataset ({len(found_target_classes)}/15): {found_target_classes}")
    print(f"Target classes MISSING from this dataset: {missing_target_classes}")

    summary.to_csv(OUTPUT_DIR / "event_class_distribution.csv")
    return summary


def analyze_events_per_video(df: pd.DataFrame):
    """Compute how many events each video/game has."""
    print("\n=== EVENTS PER VIDEO ===")
    per_video = df.groupby("game_id_from_file").size().sort_values(ascending=False)
    print(per_video.to_string())
    print(f"\nAverage events/video: {per_video.mean():.1f}")
    print(f"Min: {per_video.min()}, Max: {per_video.max()}")
    return per_video


def analyze_timestamp_distribution(df: pd.DataFrame):
    """Summarize when events occur across match time (video_time field)."""
    print("\n=== TIMESTAMP DISTRIBUTION (video_time, ms) ===")
    if "video_time" not in df.columns:
        print("No 'video_time' field found - skipping.")
        return

    ts = df["video_time"].dropna()
    print(f"Earliest event: {ts.min()} ms ({ts.min()/60000:.1f} min)")
    print(f"Latest event:   {ts.max()} ms ({ts.max()/60000:.1f} min)")
    print(f"Median event time: {ts.median():.0f} ms ({ts.median()/60000:.1f} min)")

    # Bucket into 15-minute match segments for a rough distribution view
    bucket_min = 15
    ts_minutes = ts / 60000
    buckets = (ts_minutes // bucket_min * bucket_min).astype(int)
    bucket_counts = buckets.value_counts().sort_index()
    print(f"\nEvents per {bucket_min}-minute match segment:")
    for minute, count in bucket_counts.items():
        print(f"  {minute:3d}-{minute+bucket_min:3d} min: {'#' * min(count, 50)} ({count})")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_all_events()

    if df.empty:
        logger.warning("No events loaded - check that data/roots_rise/*.json files exist.")
        return

    summarize_format(df)
    analyze_event_classes(df)
    analyze_events_per_video(df)
    analyze_timestamp_distribution(df)

    # Save the full combined event table for later use
    full_output = OUTPUT_DIR / "all_events_combined.csv"
    df.to_csv(full_output, index=False)
    logger.info(f"\nSaved combined event table to: {full_output.resolve()}")


if __name__ == "__main__":
    main()
