"""Tests for per-player analytics aggregation across runs."""

from __future__ import annotations

import json
from pathlib import Path

from src.analytics import build_player_analytics_report


def _write_json(path: Path, payload: dict) -> None:
    """Write JSON payload to path, creating parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write rows to JSONL path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _seed_run(run_dir: Path, assignments: list[dict], tracks: list[dict], events: list[dict]) -> None:
    """Create minimal run artifacts used by player analytics aggregation."""
    _write_json(run_dir / "video_metadata.json", {"fps": 10.0})
    _write_json(
        run_dir / "player_assignments.json",
        {
            "schema_version": "1.2",
            "video_id": run_dir.name,
            "assignments": assignments,
        },
    )
    _write_jsonl(run_dir / "tracks.jsonl", tracks)
    _write_jsonl(run_dir / "events.jsonl", events)
    _write_json(run_dir / "run_manifest.json", {"end_time": f"2026-01-{run_dir.name[-1]}T10:00:00Z"})


def test_build_player_analytics_report_aggregates_minutes_distance_sprints_and_events(tmp_path: Path):
    """Per-player analytics should aggregate run-level motion + event totals."""
    runs_root = tmp_path / "runs"
    run1 = runs_root / "match1"
    run2 = runs_root / "match2"

    run1_tracks = [
        {
            "frame_idx": frame_idx,
            "timestamp": frame_idx / 10.0,
            "object_type": "player",
            "track_id": 1,
            "bbox": [100 + (30 * frame_idx), 200, 120 + (30 * frame_idx), 240],
        }
        for frame_idx in range(10)
    ]
    _seed_run(
        run1,
        assignments=[
            {
                "track_id": 1,
                "player_id": 10,
                "player_name": "Alex",
                "team_hint": "ours",
                "confidence": 0.95,
            }
        ],
        tracks=run1_tracks,
        events=[
            {"event_type": "shot", "metadata": {"track_id": 1}},
            {"event_type": "pass", "metadata": {"player_id": 10}},
        ],
    )

    run2_tracks = [
        {
            "frame_idx": frame_idx,
            "timestamp": frame_idx / 10.0,
            "object_type": "player",
            "track_id": 2,
            "bbox": [50 + (10 * frame_idx), 120, 70 + (10 * frame_idx), 160],
        }
        for frame_idx in range(6)
    ] + [
        {
            "frame_idx": frame_idx,
            "timestamp": frame_idx / 10.0,
            "object_type": "player",
            "track_id": 3,
            "bbox": [200 + (20 * frame_idx), 180, 220 + (20 * frame_idx), 220],
        }
        for frame_idx in range(6)
    ]
    _seed_run(
        run2,
        assignments=[
            {
                "track_id": 2,
                "player_id": 10,
                "player_name": "Alex",
                "team_hint": "ours",
                "confidence": 0.93,
            },
            {
                "track_id": 3,
                "player_id": 22,
                "player_name": "Blake",
                "team_hint": "opponent",
                "confidence": 0.91,
            },
        ],
        tracks=run2_tracks,
        events=[
            {"event_type": "goal", "metadata": {"player_id": 10}},
            {"event_type": "shot", "metadata": {"kick_track_id": 3}},
        ],
    )

    payload = build_player_analytics_report(
        runs_root=runs_root,
        current_run=run2,
        config={
            "include_current_run": True,
            "max_runs": 10,
            "min_assignment_confidence": 0.5,
            "sprint_speed_threshold_px_per_sec": 250.0,
            "sprint_min_duration_seconds": 0.4,
            "max_track_gap_frames": 2,
        },
    )

    summary = payload["summary"]
    assert summary["runs_analyzed"] == 2
    assert summary["players_detected"] == 2
    assert summary["events_total"] == 4

    players = {row["player_id"]: row for row in payload["players"]}
    assert set(players.keys()) == {10, 22}

    alex = players[10]
    assert alex["player_name"] == "Alex"
    assert alex["matches_played"] == 2
    assert abs(float(alex["totals"]["minutes_played"]) - (16.0 / 600.0)) < 1e-6
    assert abs(float(alex["totals"]["distance_pixels"]) - 320.0) < 1e-6
    assert int(alex["totals"]["sprints"]) == 1
    assert int(alex["totals"]["events_total"]) == 3
    assert alex["totals"]["events_by_type"] == {"goal": 1, "pass": 1, "shot": 1}

    blake = players[22]
    assert blake["player_name"] == "Blake"
    assert blake["matches_played"] == 1
    assert abs(float(blake["totals"]["distance_pixels"]) - 100.0) < 1e-6
    assert int(blake["totals"]["sprints"]) == 0
    assert int(blake["totals"]["events_total"]) == 1
    assert blake["totals"]["events_by_type"] == {"shot": 1}


def test_build_player_analytics_report_handles_empty_runs_root(tmp_path: Path):
    """Empty runs root should produce a valid zeroed analytics payload."""
    payload = build_player_analytics_report(
        runs_root=tmp_path / "runs",
        current_run=tmp_path / "runs" / "current",
        config={"include_current_run": True, "max_runs": 5},
    )

    assert payload["summary"]["runs_analyzed"] == 0
    assert payload["summary"]["players_detected"] == 0
    assert payload["players"] == []
    assert payload["runs"] == []
