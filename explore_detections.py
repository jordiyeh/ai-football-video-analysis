#!/usr/bin/env python3
"""
Quick script to explore and analyze detections.parquet

Usage:
    python explore_detections.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    # Load detections
    df = pd.read_parquet("runs/arlington_fast/detections.parquet")

    print("=" * 60)
    print("DETECTION ANALYSIS")
    print("=" * 60)

    # Basic stats
    print(f"\nTotal detections: {len(df):,}")
    print(f"Frames analyzed: {df.frame_idx.nunique():,}")
    print(f"Time range: {df.timestamp.min():.1f}s - {df.timestamp.max():.1f}s ({df.timestamp.max()/60:.1f} min)")

    # Detections by type
    print("\n--- By Object Type ---")
    print(df.object_type.value_counts())

    # Average detections per frame
    print("\n--- Average Detections per Frame ---")
    per_frame = df.groupby('frame_idx').size()
    print(f"Players per frame: {per_frame.mean():.1f} (median: {per_frame.median():.0f})")

    # Confidence distribution
    print("\n--- Confidence Stats ---")
    print(df.groupby('object_type')['confidence'].describe())

    # Ball detections
    ball_df = df[df.object_type == 'ball']
    print(f"\n--- Ball Detection ---")
    print(f"Ball detected in {len(ball_df)} frames")
    print(f"Ball detection rate: {len(ball_df)/df.frame_idx.nunique()*100:.1f}%")

    # Export options
    print("\n" + "=" * 60)
    print("EXPORT OPTIONS")
    print("=" * 60)

    # 1. Export to CSV
    csv_path = "runs/arlington_fast/detections.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Exported to CSV: {csv_path}")

    # 2. Export ball-only data
    ball_csv = "runs/arlington_fast/ball_detections.csv"
    ball_df.to_csv(ball_csv, index=False)
    print(f"✓ Exported ball-only: {ball_csv}")

    # 3. Export time slices (example: first minute)
    first_min = df[df.timestamp <= 60]
    first_min.to_csv("runs/arlington_fast/first_minute.csv", index=False)
    print(f"✓ Exported first minute: runs/arlington_fast/first_minute.csv")

    # 4. Summary by frame
    summary = df.groupby('frame_idx').agg({
        'object_type': lambda x: f"{(x=='player').sum()} players, {(x=='ball').sum()} balls",
        'timestamp': 'first',
        'confidence': 'mean'
    }).reset_index()
    summary.to_csv("runs/arlington_fast/frame_summary.csv", index=False)
    print(f"✓ Exported frame summary: runs/arlington_fast/frame_summary.csv")

    print("\n" + "=" * 60)
    print("QUERY EXAMPLES")
    print("=" * 60)

    # Example queries
    print("\n1. High-confidence ball detections:")
    high_conf_ball = df[(df.object_type == 'ball') & (df.confidence > 0.8)]
    print(f"   Found {len(high_conf_ball)} high-confidence ball detections")

    print("\n2. Frames with many players (>20):")
    player_counts = df[df.object_type == 'player'].groupby('frame_idx').size()
    busy_frames = player_counts[player_counts > 20]
    print(f"   Found {len(busy_frames)} busy frames")
    if len(busy_frames) > 0:
        print(f"   Example timestamps: {df[df.frame_idx.isin(busy_frames.index[:5])].timestamp.unique()[:5]}")

    print("\n3. Large player bboxes (close to camera):")
    large_players = df[(df.object_type == 'player') & (df.area > 100000)]
    print(f"   Found {len(large_players)} large player detections")

    print("\n" + "=" * 60)
    print("\nAll exports saved to: runs/arlington_fast/")
    print("Open CSV files with Excel, Numbers, Google Sheets, etc.")


if __name__ == "__main__":
    main()
