"""Tests for unified match stats aggregation and artifact generation."""

from __future__ import annotations

from src.analytics import build_match_stats


def _player_track(
    *,
    frame_idx: int,
    track_id: int,
    team_name: str,
    x: float,
    y: float,
) -> dict:
    """Create a minimal player track row."""
    return {
        "frame_idx": frame_idx,
        "timestamp": frame_idx / 30.0,
        "track_id": track_id,
        "object_type": "player",
        "team_name": team_name,
        "bbox": [x - 8.0, y - 18.0, x + 8.0, y + 18.0],
        "image_x": x,
        "image_y": y,
        "confidence": 0.9,
    }


def test_build_match_stats_aggregates_team_totals_and_possession():
    """Match stats should combine event totals with possession summary."""
    tracks = [
        _player_track(frame_idx=10, track_id=1, team_name="team_A", x=110, y=120),
        _player_track(frame_idx=10, track_id=2, team_name="team_B", x=520, y=130),
        _player_track(frame_idx=50, track_id=1, team_name="team_A", x=120, y=120),
        _player_track(frame_idx=50, track_id=2, team_name="team_B", x=530, y=130),
    ]
    events = [
        {"event_type": "shot", "frame_idx": 10, "timestamp": 1.0, "location": [112, 121], "metadata": {}},
        {"event_type": "goal", "frame_idx": 12, "timestamp": 1.1, "location": None, "metadata": {"shot_frame": 10}},
        {"event_type": "pass", "frame_idx": 20, "timestamp": 2.0, "location": None, "metadata": {"team_id": "team_A"}},
        {"event_type": "corner_kick", "frame_idx": 30, "timestamp": 3.0, "location": None, "metadata": {"team_id": "team_B"}},
        {"event_type": "free_kick", "frame_idx": 40, "timestamp": 4.0, "location": None, "metadata": {"team_id": "team_B"}},
        {"event_type": "shot", "frame_idx": 50, "timestamp": 5.0, "location": [531, 130], "metadata": {}},
    ]
    team_analytics = {
        "possession": {
            "teams": {
                "team_A": {"frames": 90, "seconds": 30.0, "share": 0.60},
                "team_B": {"frames": 60, "seconds": 20.0, "share": 0.40},
            }
        }
    }

    stats = build_match_stats(
        events=events,
        team_analytics=team_analytics,
        tracks=tracks,
        fps=30.0,
    )

    assert stats["schema_version"] == "1.0"
    assert stats["summary"]["events_processed"] == 6
    assert stats["summary"]["events_without_team"] == 0

    team_a = stats["teams"]["team_A"]
    assert team_a["shots"] == 1
    assert team_a["goals"] == 1
    assert team_a["passes"] == 1
    assert team_a["set_pieces"] == 0
    assert team_a["possession_frames"] == 90
    assert team_a["possession_seconds"] == 30.0
    assert team_a["possession_share"] == 0.60

    team_b = stats["teams"]["team_B"]
    assert team_b["shots"] == 1
    assert team_b["goals"] == 0
    assert team_b["passes"] == 0
    assert team_b["set_pieces"] == 2
    assert team_b["possession_frames"] == 60
    assert team_b["possession_seconds"] == 20.0
    assert team_b["possession_share"] == 0.40

    totals = stats["totals"]
    assert totals["shots"] == 2
    assert totals["goals"] == 1
    assert totals["passes"] == 1
    assert totals["set_pieces"] == 2
    assert totals["possession_frames"] == 150
    assert totals["possession_seconds"] == 50.0


def test_build_match_stats_keeps_unattributed_events_in_unknown_bucket():
    """Events without team evidence should be tracked under unknown."""
    stats = build_match_stats(
        events=[
            {
                "event_type": "shot",
                "frame_idx": 8,
                "timestamp": 0.2,
                "location": None,
                "metadata": {},
            }
        ],
        team_analytics={},
        tracks=[],
    )

    assert stats["summary"]["events_processed"] == 1
    assert stats["summary"]["events_without_team"] == 1
    assert "unknown" in stats["teams"]
    assert stats["teams"]["unknown"]["shots"] == 1
