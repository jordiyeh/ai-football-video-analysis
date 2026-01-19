#!/usr/bin/env python3
"""
Quick script to explore and analyze tracks.parquet

Usage:
    python explore_tracks.py [run_directory]
    python explore_tracks.py runs/arlington_fast
"""

import sys
from pathlib import Path

import pandas as pd


def main():
    # Get run directory from args or use default
    if len(sys.argv) > 1:
        run_dir = Path(sys.argv[1])
    else:
        run_dir = Path("runs/arlington_fast")

    tracks_path = run_dir / "tracks.parquet"

    if not tracks_path.exists():
        print(f"Error: {tracks_path} not found")
        print("Usage: python explore_tracks.py [run_directory]")
        return 1

    # Load tracks
    df = pd.read_parquet(tracks_path)

    print("=" * 60)
    print("TRACK ANALYSIS")
    print("=" * 60)

    # Basic stats
    print(f"\nRun directory: {run_dir}")
    print(f"Total track points: {len(df):,}")
    print(f"Unique tracks: {df.track_id.nunique():,}")
    print(f"Frames analyzed: {df.frame_idx.nunique():,}")
    print(f"Time range: {df.timestamp.min():.1f}s - {df.timestamp.max():.1f}s ({df.timestamp.max()/60:.1f} min)")

    # Tracks by type
    print("\n--- By Object Type ---")
    print(df.groupby('object_type')['track_id'].nunique())

    # Track lengths
    print("\n--- Track Length Statistics ---")
    track_lengths = df.groupby('track_id').size()
    print(f"Average track length: {track_lengths.mean():.1f} frames")
    print(f"Median track length: {track_lengths.median():.0f} frames")
    print(f"Max track length: {track_lengths.max()} frames")
    print(f"Min track length: {track_lengths.min()} frames")

    # Track quality
    print("\n--- Track Quality ---")
    print(f"Average confidence: {df.confidence.mean():.3f}")
    print(f"Average hits per track: {df.hits.mean():.1f}")
    print(f"Average age per track: {df.age.mean():.1f} frames")

    # Long tracks (likely consistent players)
    long_tracks = track_lengths[track_lengths > 100]
    print(f"\nLong tracks (>100 frames): {len(long_tracks)}")
    if len(long_tracks) > 0:
        print(f"  Track IDs: {sorted(long_tracks.index.tolist())[:10]}")

    # Short tracks (might be fragments or false detections)
    short_tracks = track_lengths[track_lengths < 10]
    print(f"\nShort tracks (<10 frames): {len(short_tracks)}")

    # Ball tracking
    ball_tracks = df[df.object_type == 'ball']
    if len(ball_tracks) > 0:
        print(f"\n--- Ball Tracking ---")
        print(f"Ball track points: {len(ball_tracks)}")
        print(f"Unique ball tracks: {ball_tracks.track_id.nunique()}")
        ball_track_lengths = ball_tracks.groupby('track_id').size()
        print(f"Ball track lengths: {ball_track_lengths.describe()}")

    print("\n" + "=" * 60)
    print("EXPORT OPTIONS")
    print("=" * 60)

    # 1. Export to CSV
    csv_path = run_dir / "tracks.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Exported to CSV: {csv_path}")

    # 2. Export track summary
    track_summary = df.groupby('track_id').agg({
        'object_type': 'first',
        'frame_idx': ['min', 'max', 'count'],
        'timestamp': ['min', 'max'],
        'confidence': 'mean',
        'hits': 'max',
        'age': 'max',
    }).reset_index()
    track_summary.columns = ['_'.join(col).rstrip('_') for col in track_summary.columns.values]
    track_summary_path = run_dir / "track_summary.csv"
    track_summary.to_csv(track_summary_path, index=False)
    print(f"✓ Exported track summary: {track_summary_path}")

    # 3. Export ball tracks only
    if len(ball_tracks) > 0:
        ball_csv = run_dir / "ball_tracks.csv"
        ball_tracks.to_csv(ball_csv, index=False)
        print(f"✓ Exported ball tracks: {ball_csv}")

    # 4. Export player trajectories (for heatmap/visualization)
    player_tracks = df[df.object_type == 'player']
    if len(player_tracks) > 0:
        # Extract center points
        player_centers = player_tracks.copy()
        player_centers['center_x'] = player_centers['bbox'].apply(lambda b: (b[0] + b[2]) / 2)
        player_centers['center_y'] = player_centers['bbox'].apply(lambda b: (b[1] + b[3]) / 2)

        trajectories_path = run_dir / "player_trajectories.csv"
        player_centers[['track_id', 'frame_idx', 'timestamp', 'center_x', 'center_y']].to_csv(
            trajectories_path, index=False
        )
        print(f"✓ Exported player trajectories: {trajectories_path}")

    print("\n" + "=" * 60)
    print("TRACK QUALITY INSIGHTS")
    print("=" * 60)

    # Fragmentation analysis
    avg_track_length = track_lengths.mean()
    total_frames = df.frame_idx.nunique()
    fragmentation_ratio = avg_track_length / total_frames

    print(f"\nFragmentation ratio: {fragmentation_ratio:.3f}")
    if fragmentation_ratio > 0.5:
        print("  ✓ Good: Tracks are relatively stable")
    elif fragmentation_ratio > 0.2:
        print("  ⚠ Moderate: Some track fragmentation present")
    else:
        print("  ✗ Poor: High track fragmentation - consider tuning parameters")

    # Coverage analysis
    avg_detections_per_frame = len(df) / total_frames
    print(f"\nAverage tracked objects per frame: {avg_detections_per_frame:.1f}")

    print("\n" + "=" * 60)
    print(f"\nAll exports saved to: {run_dir}/")
    print("Open CSV files with Excel, Numbers, Google Sheets, etc.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
