"""
Stage 7 - Is the "player speed" really player speed, or camera panning?

All our speeds are measured in IMAGE pixels, so a camera pan makes every
player (and the ball) look like it is moving. If the camera is panning,
nearly all players in a frame move in the SAME direction at a similar speed.
The per-frame MEDIAN velocity across all players is therefore a simple
camera-motion estimate: real runners differ from each other, a pan shifts
everyone together.

Usage:
    python src/context/check_camera_motion.py
"""

import json
import math
import statistics
from pathlib import Path
from collections import defaultdict

PLAYER_CONTEXT_PATH = Path("data/analysis/context/player_context.json")
MIN_PLAYERS = 4          # need a few players in the frame for a meaningful median
PAN_THRESHOLD = 300      # px/sec of shared motion that we treat as "camera moving"

with open(PLAYER_CONTEXT_PATH) as f:
    records = json.load(f)

by_frame = defaultdict(list)
for r in records:
    if r["speed_px_per_sec"] is None or r["direction_degrees"] is None:
        continue
    angle = math.radians(r["direction_degrees"])
    by_frame[r["frame"]].append((r["speed_px_per_sec"] * math.cos(angle),
                                 r["speed_px_per_sec"] * math.sin(angle)))

frame_motion = {}
for frame, vels in by_frame.items():
    if len(vels) < MIN_PLAYERS:
        continue
    mvx = statistics.median(v[0] for v in vels)
    mvy = statistics.median(v[1] for v in vels)
    frame_motion[frame] = (mvx, mvy, math.hypot(mvx, mvy))

magnitudes = sorted(m[2] for m in frame_motion.values())
panning = [f for f, m in frame_motion.items() if m[2] >= PAN_THRESHOLD]

print(f"Frames analysed: {len(frame_motion)}")
print(f"Median shared motion: {magnitudes[len(magnitudes)//2]:.1f} px/sec")
print(f"95th percentile:      {magnitudes[int(len(magnitudes)*0.95)]:.1f} px/sec")
print(f"Frames with shared motion >= {PAN_THRESHOLD} px/sec: {len(panning)} "
      f"({len(panning)/len(frame_motion)*100:.0f}%)")

print("\n=== 8 FRAMES WITH THE MOST SHARED (CAMERA-LIKE) MOTION ===")
for frame, (mvx, mvy, mag) in sorted(frame_motion.items(), key=lambda kv: kv[1][2], reverse=True)[:8]:
    print(f"  frame {frame} ({frame/25:.2f}s): shared motion {mag:.0f} px/sec  (vx={mvx:.0f}, vy={mvy:.0f})")
