#!/usr/bin/env python3
"""
Quick script to explore and analyze detected events (shots, goals)

Usage:
    python explore_events.py [run_directory]
    python explore_events.py runs/arlington_fast
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

    events_path = run_dir / "events.jsonl"
    timeline_path = run_dir / "score_timeline.json"
    video_metadata_path = run_dir / "video_metadata.json"

    if not events_path.exists():
        print(f"Error: {events_path} not found")
        print("Run analysis with event detection first")
        return 1

    # Load events
    events = []
    with open(events_path, "r") as f:
        for line in f:
            events.append(json.loads(line))

    # Convert to DataFrame
    df = pd.DataFrame(events)

    # Load video metadata for context
    video_duration = None
    if video_metadata_path.exists():
        with open(video_metadata_path, "r") as f:
            video_metadata = json.load(f)
            video_duration = video_metadata.get("duration")

    print("=" * 60)
    print("EVENT DETECTION ANALYSIS")
    print("=" * 60)

    print(f"\nRun directory: {run_dir}")
    print(f"Total events detected: {len(events)}")

    if video_duration:
        print(f"Video duration: {video_duration:.2f}s ({video_duration/60:.1f} min)")

    # Event type breakdown
    print("\n--- Event Types ---")
    event_counts = df.groupby("event_type").size()
    print(event_counts)

    # Confidence statistics
    print("\n--- Confidence Statistics ---")
    print(df.groupby("event_type")["confidence"].describe())

    # Shot events
    if "shot" in df["event_type"].values:
        shots = df[df.event_type == "shot"]
        print(f"\n--- Shot Events ({len(shots)}) ---")

        # Sort by confidence
        top_shots = shots.sort_values("confidence", ascending=False).head(10)
        print("\nTop 10 shots by confidence:")
        for idx, shot in top_shots.iterrows():
            time_str = f"{shot['timestamp']:.1f}s ({shot['timestamp']/60:.1f}min)"
            conf_str = f"{shot['confidence']:.2f}"
            speed = shot.get("metadata", {}).get("speed", "N/A")
            target = shot.get("metadata", {}).get("target_goal", "unknown")
            print(f"  {time_str:15s} confidence={conf_str} speed={speed:.1f}px/f target={target}")

        # Export shots
        output_path = run_dir / "shots.csv"
        shots.to_csv(output_path, index=False)
        print(f"\n✓ Exported shots to: {output_path}")

    # Goal events
    if "goal" in df["event_type"].values:
        goals = df[df.event_type == "goal"]
        print(f"\n--- Goal Events ({len(goals)}) ---")

        # Sort by timestamp
        for idx, goal in goals.iterrows():
            time_str = f"{goal['timestamp']:.1f}s ({goal['timestamp']/60:.1f}min)"
            conf_str = f"{goal['confidence']:.2f}"
            region = goal.get("metadata", {}).get("goal_region", "unknown")
            shot_time = goal.get("metadata", {}).get("shot_timestamp", None)
            if shot_time is not None:
                shot_str = f"shot@{shot_time:.1f}s"
            else:
                shot_str = "no shot"
            print(f"  {time_str:15s} confidence={conf_str} region={region} ({shot_str})")

        # Export goals
        output_path = run_dir / "goals.csv"
        goals.to_csv(output_path, index=False)
        print(f"\n✓ Exported goals to: {output_path}")

    # Score timeline
    if timeline_path.exists():
        with open(timeline_path, "r") as f:
            timeline_data = json.load(f)

        print("\n--- Score Timeline ---")
        print(f"Total goals: {timeline_data['goals']}")
        print(f"Final score: {timeline_data['final_score']}")

        if timeline_data['timeline']:
            print("\nTimeline:")
            for entry in timeline_data['timeline']:
                time_str = f"{entry['timestamp']:.1f}s ({entry['timestamp']/60:.1f}min)"
                score = entry['score']
                conf_str = f"{entry['confidence']:.2f}"
                print(f"  {time_str:15s} {score['team_a']}-{score['team_b']} (confidence={conf_str})")

    # Export all events
    print("\n" + "=" * 60)
    print("EXPORTS")
    print("=" * 60)

    # Full event export
    output_path = run_dir / "events.csv"
    df.to_csv(output_path, index=False)
    print(f"✓ Exported all events to: {output_path}")

    # Event timeline for visualization
    timeline_df = df[["event_type", "timestamp", "confidence", "frame_idx"]].copy()
    timeline_df = timeline_df.sort_values("timestamp")
    output_path = run_dir / "event_timeline.csv"
    timeline_df.to_csv(output_path, index=False)
    print(f"✓ Exported event timeline to: {output_path}")

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    # Analysis and recommendations
    if len(events) == 0:
        print("\n⚠ No events detected")
        print("  Possible reasons:")
        print("    - Ball not detected consistently")
        print("    - Ball speed below threshold")
        print("    - No ball movement toward goal regions")
        print("  Try:")
        print("    - Lower confidence threshold for ball detection")
        print("    - Adjust shot_velocity_threshold")
        print("    - Check ball detection quality with explore_tracks.py")
    else:
        # Check shot-to-goal ratio
        n_shots = len(df[df.event_type == "shot"]) if "shot" in df.event_type.values else 0
        n_goals = len(df[df.event_type == "goal"]) if "goal" in df.event_type.values else 0

        if n_shots > 0 and n_goals == 0:
            print("\n⚠ Shots detected but no goals")
            print("  This could indicate:")
            print("    - Goal regions not correctly estimated")
            print("    - Goal confidence threshold too high")
            print("    - Ball not tracked into goal region")

        if n_goals > 10:
            print("\n⚠ Many goals detected (may be false positives)")
            print("  Consider:")
            print("    - Raising goal_confidence_threshold")
            print("    - Improving ball tracking quality")
            print("    - Using field detection for better goal localization")

        # Confidence distribution
        low_conf = df[df.confidence < 0.5]
        if len(low_conf) > len(df) * 0.5:
            print("\n⚠ Many low-confidence events detected")
            print(f"  {len(low_conf)}/{len(df)} events have confidence < 0.5")
            print("  Consider filtering or manual review")

    print(f"\nAll exports saved to: {run_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
