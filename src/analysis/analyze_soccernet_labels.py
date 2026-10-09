"""
SoccerNet-v2 Label Analysis

Parses all Labels-v2.json files (across train/valid/test) and computes:
- annotation format summary
- event label distribution (real scale: 500 games)
- events per game
- timestamp distribution across match time
- class imbalance

Usage:
    python src/analysis/analyze_soccernet_labels.py
"""

import json
from pathlib import Path
import pandas as pd
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger

logger = get_logger(__name__)

SOCCERNET_DIR = Path("data/soccernet")
OUTPUT_DIR = Path("data/analysis")


def load_all_labels() -> pd.DataFrame:
    """Load every annotation from every Labels-v2.json file into one DataFrame."""
    label_files = list(SOCCERNET_DIR.rglob("Labels-v2.json"))
    logger.info(f"Found {len(label_files)} Labels-v2.json files.")

    all_annotations = []
    for lf in label_files:
        with open(lf, "r", encoding="utf-8") as f:
            data = json.load(f)

        game_path = str(lf.parent)
        annotations = data.get("annotations", [])
        for a in annotations:
            a["game_path"] = game_path
        all_annotations.extend(annotations)

    df = pd.DataFrame(all_annotations)
    logger.info(f"Loaded {len(df)} total annotations across {len(label_files)} games.")
    return df


def summarize_format(df: pd.DataFrame):
    print("\n=== ANNOTATION FORMAT ===")
    print(f"Fields present ({len(df.columns)}): {list(df.columns)}\n")
    for col in df.columns:
        sample = df[col].dropna().iloc[0] if df[col].notna().any() else "N/A"
        print(f"  {col:20s} | dtype={str(df[col].dtype):10s} | example={sample}")


def analyze_event_classes(df: pd.DataFrame):
    print("\n=== EVENT LABEL DISTRIBUTION (across all 500 games) ===")
    counts = df["label"].value_counts()
    percentages = (counts / counts.sum() * 100).round(2)
    summary = pd.DataFrame({"count": counts, "percentage": percentages})
    print(summary.to_string())

    print(f"\nTotal distinct event labels: {df['label'].nunique()}")
    print(f"Most common: {counts.idxmax()} ({counts.max()} events)")
    print(f"Least common: {counts.idxmin()} ({counts.min()} events)")
    imbalance_ratio = round(counts.max() / counts.min(), 1)
    print(f"Class imbalance ratio (max/min): {imbalance_ratio}:1")

    summary.to_csv(OUTPUT_DIR / "soccernet_label_distribution.csv")
    return summary


def analyze_events_per_game(df: pd.DataFrame):
    print("\n=== EVENTS PER GAME ===")
    per_game = df.groupby("game_path").size()
    print(f"Average events/game: {per_game.mean():.1f}")
    print(f"Min: {per_game.min()}, Max: {per_game.max()}")
    print(f"Median: {per_game.median():.1f}")
    return per_game


def analyze_timestamp_distribution(df: pd.DataFrame):
    print("\n=== TIMESTAMP DISTRIBUTION (gameTime field) ===")
    if "gameTime" not in df.columns:
        print("No 'gameTime' field found - skipping.")
        return

    # gameTime format is typically "1 - 23:45" (half - MM:SS)
    def parse_minute(gt):
        try:
            half, time_str = gt.split(" - ")
            minutes, seconds = time_str.split(":")
            return int(minutes) + (45 if half.strip() == "2" else 0)
        except Exception:
            return None

    df["match_minute"] = df["gameTime"].apply(parse_minute)
    valid_minutes = df["match_minute"].dropna()

    bucket_min = 15
    buckets = (valid_minutes // bucket_min * bucket_min).astype(int)
    bucket_counts = buckets.value_counts().sort_index()
    print(f"Events per {bucket_min}-minute match segment (both halves):")
    for minute, count in bucket_counts.items():
        bar_len = min(int(count / bucket_counts.max() * 50), 50)
        print(f"  {minute:3d}-{minute+bucket_min:3d} min: {'#' * bar_len} ({count})")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_all_labels()

    if df.empty:
        logger.warning("No labels loaded - check that data/soccernet/**/Labels-v2.json files exist.")
        return

    summarize_format(df)
    analyze_event_classes(df)
    analyze_events_per_game(df)
    analyze_timestamp_distribution(df)

    full_output = OUTPUT_DIR / "soccernet_all_labels_combined.csv"
    df.to_csv(full_output, index=False)
    logger.info(f"\nSaved combined label table to: {full_output.resolve()}")


if __name__ == "__main__":
    main()
