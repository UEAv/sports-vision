"""
Stage 6 - Quick sanity check on the fastest recorded velocity:
is it a real sprint/pan, or a detection glitch?

Usage:
    python src/tracking/check_fastest.py
"""

import json
from pathlib import Path

RECORDS_PATH = Path("data/analysis/tracking/bytetrack_trajectories.json")

with open(RECORDS_PATH) as f:
    trajs = json.load(f)

fastest = max(trajs, key=lambda t: t["max_velocity_px_per_sec"])
print(f"Track #{fastest['track_id']}, max velocity: {fastest['max_velocity_px_per_sec']} px/sec\n")

for p in fastest["trajectory"]:
    if p["velocity_px_per_sec"] and p["velocity_px_per_sec"] > 300:
        print(p)
