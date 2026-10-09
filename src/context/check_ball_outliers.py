"""
Stage 7 - Sanity check: is the extreme ball velocity a real fast kick,
or a detection glitch (ball mis-detected somewhere else in frame)?

Usage:
    python src/context/check_ball_outliers.py
"""

import json
from pathlib import Path

BALL_CONTEXT_PATH = Path("data/analysis/context/ball_context.json")

with open(BALL_CONTEXT_PATH) as f:
    positions = json.load(f)

# Sort by velocity, show the top 5 fastest "movements"
sorted_by_velocity = sorted(
    [p for p in positions if p["velocity_px_per_sec"] is not None],
    key=lambda p: p["velocity_px_per_sec"],
    reverse=True
)

print("=== TOP 5 FASTEST BALL 'MOVEMENTS' ===\n")
for p in sorted_by_velocity[:5]:
    print(p)
    print()

velocities = [p["velocity_px_per_sec"] for p in positions if p["velocity_px_per_sec"] is not None]
velocities_sorted = sorted(velocities)
median = velocities_sorted[len(velocities_sorted) // 2]
print(f"Median velocity: {median:.1f} px/sec")
print(f"95th percentile: {velocities_sorted[int(len(velocities_sorted)*0.95)]:.1f} px/sec")
print(f"Max: {max(velocities):.1f} px/sec")
