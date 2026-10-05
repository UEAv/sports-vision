"""
Stage 6 - Quick check: what confidence scores are the duplicate/ghost
detections actually carrying? This tells us where to set a stricter
confidence threshold BEFORE guessing a number blindly.

Usage:
    python src/tracking/check_confidence_distribution.py
"""

import json
from pathlib import Path
import numpy as np

RECORDS_PATH = Path("data/analysis/tracking/bytetrack_raw_records.json")

with open(RECORDS_PATH) as f:
    records = json.load(f)

confidences = np.array([r["confidence"] for r in records])

print(f"Total records: {len(confidences)}")
print(f"\nConfidence distribution:")
print(f"  Min:    {confidences.min():.3f}")
print(f"  25th %: {np.percentile(confidences, 25):.3f}")
print(f"  Median: {np.median(confidences):.3f}")
print(f"  75th %: {np.percentile(confidences, 75):.3f}")
print(f"  Max:    {confidences.max():.3f}")

print(f"\nHow many records would survive at different thresholds:")
for threshold in [0.25, 0.35, 0.45, 0.55, 0.65, 0.75]:
    survive = (confidences >= threshold).sum()
    pct = survive / len(confidences) * 100
    print(f"  conf >= {threshold}: {survive} records ({pct:.1f}% of total)")
