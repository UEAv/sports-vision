"""
Stage 7 - Sanity check on the fastest player movements, now able to look at
either APPARENT speed (includes camera) or CAMERA-COMPENSATED speed.

Usage:
    python src/context/check_player_speed_outliers.py
    python src/context/check_player_speed_outliers.py --column speed_px_per_sec
"""

import json
import argparse
from pathlib import Path

PLAYER_CONTEXT_PATH = Path("data/analysis/context/player_context.json")

parser = argparse.ArgumentParser()
parser.add_argument("--column", default="speed_compensated_px_per_sec",
                    choices=["speed_px_per_sec", "speed_compensated_px_per_sec"])
args = parser.parse_args()

with open(PLAYER_CONTEXT_PATH) as f:
    records = json.load(f)

valid = [r for r in records if r.get(args.column) is not None]
ranked = sorted(valid, key=lambda r: r[args.column], reverse=True)

print(f"=== TOP 8 FASTEST ({args.column}) ===\n")
for r in ranked[:8]:
    print(f"frame {r['frame']} ({r['timestamp_sec']}s) | track #{r['track_id']} | conf {r['confidence']}")
    print(f"   apparent speed: {r['speed_px_per_sec']} | camera motion: {r['camera_motion_px_per_sec']} "
          f"| compensated speed: {r['speed_compensated_px_per_sec']}")
    print(f"   bbox: {r['bbox']} | nearest player: #{r['nearest_player_track_id']} "
          f"at {r['proximity_to_nearest_player_px']}px\n")

values = sorted(r[args.column] for r in valid)
print(f"Median: {values[len(values)//2]:.1f} px/sec")
print(f"95th percentile: {values[int(len(values)*0.95)]:.1f} px/sec")
print(f"99th percentile: {values[int(len(values)*0.99)]:.1f} px/sec")
print(f"Max: {max(values):.1f} px/sec")
