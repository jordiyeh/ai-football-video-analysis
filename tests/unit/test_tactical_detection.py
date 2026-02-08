"""Tests for deterministic tactical event inference."""

from __future__ import annotations

from src.events.tactical import TACTICAL_INFERENCE_ALGO_VERSION, infer_tactical_events


def _build_team_analytics_payload() -> dict:
    """Build synthetic possession/pressing timelines for tactical inference."""
    possession_timeline: list[dict] = []

    # Team ours build-up phase (long possession with carrier switch + field progression).
    for frame_idx in range(0, 20):
        possession_timeline.append(
            {
                "frame_idx": frame_idx,
                "timestamp": frame_idx / 10.0,
                "owner_team": "ours",
                "owner_track_id": 10 if frame_idx < 10 else 11,
                "owner_norm_x": 0.20 + (0.010 * frame_idx),
                "owner_norm_y": 0.75 - (0.012 * frame_idx),
            }
        )

    # Transition into opponent possession.
    for frame_idx in range(20, 31):
        possession_timeline.append(
            {
                "frame_idx": frame_idx,
                "timestamp": frame_idx / 10.0,
                "owner_team": "opponent",
                "owner_track_id": 21,
                "owner_norm_x": 0.46 + (0.007 * (frame_idx - 20)),
                "owner_norm_y": 0.45 + (0.004 * (frame_idx - 20)),
            }
        )

    pressing_timeline: list[dict] = []

    # Opponent high-press run while ours has possession.
    for frame_idx in range(4, 14):
        pressing_timeline.append(
            {
                "frame_idx": frame_idx,
                "timestamp": frame_idx / 10.0,
                "attacking_team": "ours",
                "defending_team": "opponent",
                "carrier_track_id": 10 if frame_idx < 10 else 11,
                "nearest_distance_norm": 0.05,
                "defenders_within_radius": 3,
                "pressure_score": 0.82,
                "high_press": True,
            }
        )

    # Ours low-block defending run while opponent has possession.
    for frame_idx in range(20, 34):
        pressing_timeline.append(
            {
                "frame_idx": frame_idx,
                "timestamp": frame_idx / 10.0,
                "attacking_team": "opponent",
                "defending_team": "ours",
                "carrier_track_id": 21,
                "nearest_distance_norm": 0.08,
                "defenders_within_radius": 2,
                "pressure_score": 0.48,
                "high_press": False,
            }
        )

    return {
        "possession_timeline": possession_timeline,
        "pressing_timeline": pressing_timeline,
    }


def test_tactical_inference_detects_all_primary_tactical_events() -> None:
    """Build-up, pressing, defending, and transition events should be inferred."""
    analytics = _build_team_analytics_payload()
    events = infer_tactical_events(
        tracks=[],
        team_analytics=analytics,
        fps=10.0,
        config={
            "build_up_min_frames": 12,
            "build_up_min_progress_norm": 0.08,
            "pressing_min_frames": 5,
            "pressing_min_pressure_score": 0.60,
            "defending_min_frames": 6,
            "transition_max_gap_frames": 3,
            "transition_min_displacement_norm": 0.08,
            "min_event_separation_seconds": 0.0,
        },
    )

    by_type = {event.event_type: event for event in events}
    assert set(by_type) >= {"build_up", "pressing", "defending", "transition"}

    for event_type in ("build_up", "pressing", "defending", "transition"):
        event = by_type[event_type]
        assert 0.0 <= event.confidence <= 1.0
        assert event.metadata is not None
        assert event.metadata["event_family"] == "tactical"
        assert event.metadata["tactical_type"] == event_type
        assert event.metadata["provenance"]["detector"] == "tactical_phase_heuristics"
        assert (
            event.metadata["provenance"]["algorithm_version"]
            == TACTICAL_INFERENCE_ALGO_VERSION
        )


def test_tactical_inference_returns_empty_without_team_analytics() -> None:
    """Missing team analytics should yield no tactical events."""
    assert infer_tactical_events(tracks=[], team_analytics=None, fps=30.0) == []


def test_tactical_inference_respects_pressing_threshold() -> None:
    """Pressing events should be filtered when pressure threshold is too strict."""
    analytics = _build_team_analytics_payload()
    events = infer_tactical_events(
        tracks=[],
        team_analytics=analytics,
        fps=10.0,
        config={
            "build_up_min_frames": 12,
            "build_up_min_progress_norm": 0.08,
            "pressing_min_frames": 5,
            "pressing_min_pressure_score": 0.95,
            "defending_min_frames": 6,
            "transition_max_gap_frames": 3,
            "transition_min_displacement_norm": 0.08,
            "min_event_separation_seconds": 0.0,
        },
    )

    event_types = {event.event_type for event in events}
    assert "pressing" not in event_types
