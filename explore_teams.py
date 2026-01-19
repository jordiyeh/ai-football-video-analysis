#!/usr/bin/env python3
"""
Quick script to explore and analyze team assignments

Usage:
    python explore_teams.py [run_directory]
    python explore_teams.py runs/arlington_fast
"""

import json
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
    teams_path = run_dir / "teams.json"

    if not tracks_path.exists():
        print(f"Error: {tracks_path} not found")
        print("Run analysis with tracking first")
        return 1

    if not teams_path.exists():
        print(f"Error: {teams_path} not found")
        print("Team assignment not available for this run")
        return 1

    # Load tracks
    df = pd.read_parquet(tracks_path)

    # Load team info
    with open(teams_path, "r") as f:
        team_info = json.load(f)

    print("=" * 60)
    print("TEAM ASSIGNMENT ANALYSIS")
    print("=" * 60)

    print(f"\nRun directory: {run_dir}")
    print(f"Teams identified: {team_info['n_teams']}")

    # Team names and colors
    print("\n--- Team Information ---")
    for team_id, team_name in team_info['team_names'].items():
        color = team_info['team_colors'][str(team_id)]
        print(f"Team {team_id} ({team_name}):")
        print(f"  Color (BGR): {color}")
        print(f"  Players: {len([tid for tid, t in team_info['track_assignments'].items() if t == int(team_id)])}")

    # Team distribution
    print("\n--- Team Distribution in Tracks ---")
    if 'team_name' in df.columns:
        team_counts = df.groupby('team_name')['track_id'].nunique()
        print(team_counts)

        print("\n--- Track Points by Team ---")
        track_point_counts = df.groupby('team_name').size()
        print(track_point_counts)

    else:
        print("No team_name column in tracks - team assignment may have failed")

    # Track-level analysis
    if 'team_id' in df.columns:
        print("\n--- Longest Tracks by Team ---")
        track_lengths = df.groupby(['track_id', 'team_name']).size().reset_index(name='length')
        track_lengths_sorted = track_lengths.sort_values('length', ascending=False)
        print(track_lengths_sorted.head(20))

        # Team consistency (how often does a track change teams? - should be 0)
        print("\n--- Team Consistency Check ---")
        team_changes = df.groupby('track_id')['team_id'].nunique()
        inconsistent = team_changes[team_changes > 1]
        if len(inconsistent) > 0:
            print(f"⚠ Warning: {len(inconsistent)} tracks have multiple team assignments")
            print(f"  This may indicate tracking errors or goalkeepers")
        else:
            print("✓ All tracks have consistent team assignments")

    print("\n" + "=" * 60)
    print("EXPORT OPTIONS")
    print("=" * 60)

    # Export by team
    if 'team_name' in df.columns:
        for team_name in df['team_name'].unique():
            if team_name == 'unknown':
                continue
            team_tracks = df[df.team_name == team_name]
            output_path = run_dir / f"tracks_{team_name}.csv"
            team_tracks.to_csv(output_path, index=False)
            print(f"✓ Exported {team_name} tracks: {output_path}")

    # Team summary
    if 'team_id' in df.columns and 'team_name' in df.columns:
        team_summary = df.groupby(['team_id', 'team_name']).agg({
            'track_id': 'nunique',
            'frame_idx': ['min', 'max', 'count'],
            'confidence': 'mean',
        }).reset_index()
        team_summary.columns = ['_'.join(col).rstrip('_') for col in team_summary.columns.values]
        summary_path = run_dir / "team_summary.csv"
        team_summary.to_csv(summary_path, index=False)
        print(f"✓ Exported team summary: {summary_path}")

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    # Analysis and recommendations
    if 'team_name' in df.columns:
        team_counts = df.groupby('team_name')['track_id'].nunique()

        if len(team_counts) == 2:
            teams = team_counts.index.tolist()
            counts = team_counts.values.tolist()
            ratio = max(counts) / min(counts) if min(counts) > 0 else 0

            if ratio > 1.5:
                print(f"\n⚠ Team imbalance detected:")
                print(f"  {teams[0]}: {counts[0]} players")
                print(f"  {teams[1]}: {counts[1]} players")
                print(f"  This may indicate:")
                print(f"    - Goalkeeper wearing different color")
                print(f"    - One team more visible in video")
                print(f"    - Referee being classified as a player")
            else:
                print(f"\n✓ Teams are roughly balanced:")
                print(f"  {teams[0]}: {counts[0]} players")
                print(f"  {teams[1]}: {counts[1]} players")

    print(f"\nAll exports saved to: {run_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
