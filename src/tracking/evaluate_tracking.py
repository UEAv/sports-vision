"""
Stage 6 - Tracking Evaluation

Quantifies track quality from raw ByteTrack output:
- concurrent track count per frame (sanity check vs ~22 real players)
- track duration distribution (short tracks = likely fragments/noise)
- potential ID switches (heuristic: a track ending and a new one starting
  very close in position/time, suggesting the same player got a new ID)

NOTE: without ground-truth tracking labels (which we don't have), true ID
switch detection isn't possible with certainty - these are heuristic flags
for YOU to visually spot-check, not definitive proof.

Usage:
    python src/tracking/evaluate_tracking.py
"""

import json
from pathlib import Path
from collections import defaultdict
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger

logger = get_logger(__name__)

import argparse

DEFAULT_RECORDS_PATH = Path("data/analysis/tracking/bytetrack_raw_records.json")

# Realistic sanity bound: 22 players + ref/linesman, but usually fewer are
# actually IN FRAME at once depending on camera angle/zoom
MAX_REALISTIC_CONCURRENT = 23

# A track shorter than this is almost certainly noise/fragment, not a real
# player sighting (at 25fps, 5 frames = 0.2 seconds)
SHORT_TRACK_THRESHOLD = 5

# Distance (pixels) + time gap (seconds) under which a new track starting
# right after an old one ends is flagged as a POSSIBLE ID switch
SWITCH_DISTANCE_PX = 50
SWITCH_TIME_GAP_SEC = 0.5


def load_records(records_path: Path):
    with open(records_path) as f:
        return json.load(f)


def analyze_concurrent_tracks(records):
    print("\n=== CONCURRENT TRACKS PER FRAME ===")
    by_frame = defaultdict(list)
    for r in records:
        by_frame[r["frame"]].append(r["track_id"])

    counts = [len(ids) for ids in by_frame.values()]
    avg_count = sum(counts) / len(counts)
    max_count = max(counts)

    print(f"Average concurrent tracks/frame: {avg_count:.1f}")
    print(f"Max concurrent tracks/frame: {max_count}")
    print(f"Realistic upper bound (players+ref): ~{MAX_REALISTIC_CONCURRENT}")

    if max_count > MAX_REALISTIC_CONCURRENT:
        print(f"⚠ FLAG: max concurrent tracks ({max_count}) EXCEEDS realistic bound - "
              f"strongly suggests duplicate/ghost tracks on the same physical player.")
    else:
        print("✓ Concurrent track counts stay within a realistic range.")


def analyze_track_durations(records):
    print("\n=== TRACK DURATION DISTRIBUTION ===")
    by_track = defaultdict(list)
    for r in records:
        by_track[r["track_id"]].append(r)

    durations = {tid: len(recs) for tid, recs in by_track.items()}
    total_tracks = len(durations)
    short_tracks = {tid: d for tid, d in durations.items() if d < SHORT_TRACK_THRESHOLD}

    print(f"Total unique track IDs: {total_tracks}")
    print(f"Tracks shorter than {SHORT_TRACK_THRESHOLD} frames (likely fragments/noise): "
          f"{len(short_tracks)} ({len(short_tracks)/total_tracks*100:.1f}%)")
    print(f"Tracks lasting >= 50 frames (likely real, stable players): "
          f"{sum(1 for d in durations.values() if d >= 50)}")

    if len(short_tracks) / total_tracks > 0.3:
        print(f"⚠ FLAG: over 30% of tracks are very short-lived - significant fragmentation present.")

    return by_track


def detect_possible_id_switches(by_track):
    print("\n=== POSSIBLE ID SWITCHES (heuristic) ===")
    print(f"Looking for: a track ending, then a NEW track starting within "
          f"{SWITCH_TIME_GAP_SEC}s and {SWITCH_DISTANCE_PX}px of where it ended.\n")

    # Build (end_time, end_position, track_id) for every track
    track_ends = []
    track_starts = []
    for tid, recs in by_track.items():
        recs_sorted = sorted(recs, key=lambda r: r["frame"])
        track_ends.append((recs_sorted[-1]["timestamp_sec"], recs_sorted[-1]["center_x"], recs_sorted[-1]["center_y"], tid))
        track_starts.append((recs_sorted[0]["timestamp_sec"], recs_sorted[0]["center_x"], recs_sorted[0]["center_y"], tid))

    possible_switches = 0
    for end_time, end_x, end_y, end_tid in track_ends:
        for start_time, start_x, start_y, start_tid in track_starts:
            if start_tid == end_tid:
                continue
            time_gap = start_time - end_time
            if 0 <= time_gap <= SWITCH_TIME_GAP_SEC:
                distance = ((end_x - start_x) ** 2 + (end_y - start_y) ** 2) ** 0.5
                if distance <= SWITCH_DISTANCE_PX:
                    possible_switches += 1
                    print(f"  Track #{end_tid} ends at {end_time}s ({end_x:.0f},{end_y:.0f}) -> "
                          f"Track #{start_tid} starts at {start_time}s ({start_x:.0f},{start_y:.0f}) "
                          f"[gap={time_gap:.2f}s, dist={distance:.0f}px]")

    print(f"\nTotal possible ID switches flagged: {possible_switches}")
    print("(These are CANDIDATES for manual review, not confirmed switches - "
          "a player genuinely could have another player briefly nearby.)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker", default="bytetrack", choices=["bytetrack", "botsort"],
                         help="Which tracker's output to evaluate")
    args = parser.parse_args()

    records_path = Path(f"data/analysis/tracking/{args.tracker}_raw_records.json")

    if not records_path.exists():
        logger.error(f"{records_path} not found - run track_players.py first.")
        return

    print(f"\n{'#'*20} EVALUATING: {args.tracker.upper()} {'#'*20}")
    records = load_records(records_path)
    logger.info(f"Loaded {len(records)} tracking records.")

    analyze_concurrent_tracks(records)
    by_track = analyze_track_durations(records)
    detect_possible_id_switches(by_track)


if __name__ == "__main__":
    main()
