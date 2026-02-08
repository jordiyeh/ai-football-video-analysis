"""Tests for deterministic pass inference from possession handoffs."""

from __future__ import annotations

import pytest

from src.events.passes import PassInferencer, infer_pass_events


def _track_row(
    *,
    frame_idx: int,
    track_id: int,
    object_type: str,
    center_x: float,
    center_y: float,
    team_name: str | None = None,
) -> dict:
    """Create a minimal synthetic track row for pass inference tests."""
    box_w = 20.0
    box_h = 20.0

    row = {
        "frame_idx": frame_idx,
        "timestamp": frame_idx / 10.0,
        "track_id": track_id,
        "object_type": object_type,
        "confidence": 0.95,
        "bbox": [
            center_x - (box_w / 2.0),
            center_y - (box_h / 2.0),
            center_x + (box_w / 2.0),
            center_y + (box_h / 2.0),
        ],
        "image_x": center_x,
        "image_y": center_y,
        "image_xy": [center_x, center_y],
    }
    if team_name is not None:
        row["team_name"] = team_name
    return row


def test_pass_inference_detects_same_team_handoff():
    """A same-team ownership transfer should yield one pass event."""
    tracks: list[dict] = []
    owner_x = [300.0, 300.0, 450.0, 450.0, 700.0, 700.0]

    for frame_idx in range(6):
        tracks.append(
            _track_row(
                frame_idx=frame_idx,
                track_id=1,
                object_type="player",
                center_x=owner_x[frame_idx] if frame_idx < 2 else 250.0,
                center_y=250.0,
                team_name="team_A",
            )
        )
        tracks.append(
            _track_row(
                frame_idx=frame_idx,
                track_id=2,
                object_type="player",
                center_x=owner_x[frame_idx] if 2 <= frame_idx <= 3 else 520.0,
                center_y=250.0,
                team_name="team_A",
            )
        )
        tracks.append(
            _track_row(
                frame_idx=frame_idx,
                track_id=3,
                object_type="player",
                center_x=owner_x[frame_idx] if frame_idx >= 4 else 760.0,
                center_y=250.0,
                team_name="team_B",
            )
        )
        tracks.append(
            _track_row(
                frame_idx=frame_idx,
                track_id=90,
                object_type="ball",
                center_x=owner_x[frame_idx] + 8.0,
                center_y=250.0,
            )
        )

    inferencer = PassInferencer(
        config={
            "possession_max_ball_distance_px": 80.0,
            "possession_smoothing_frames": 1,
            "possession_min_stable_frames": 1,
            "possession_min_segment_frames": 1,
            "pass_min_gap_seconds": 0.0,
            "pass_max_gap_seconds": 1.0,
        }
    )
    events = inferencer.infer(tracks, fps=10.0)

    assert len(events) == 1
    event = events[0]
    assert event.event_type == "pass"
    assert event.frame_idx == 2
    assert event.metadata is not None
    assert event.metadata["team_id"] == "team_A"
    assert event.metadata["from_track_id"] == 1
    assert event.metadata["to_track_id"] == 2
    assert event.metadata["event_family"] == "pass"
    assert event.metadata["provenance"]["detector"] == "possession_handoff"
    assert set(event.metadata["confidence_factors"]) == {
        "proximity",
        "stability",
        "gap",
        "owner_confidence",
        "raw",
    }
    assert 0.0 <= event.confidence <= 1.0


def test_pass_inference_excludes_cross_team_handoffs():
    """Cross-team possession changes are not passes."""
    tracks: list[dict] = []
    owner_x = [300.0, 300.0, 720.0, 720.0]

    for frame_idx in range(4):
        tracks.append(
            _track_row(
                frame_idx=frame_idx,
                track_id=1,
                object_type="player",
                center_x=owner_x[frame_idx] if frame_idx < 2 else 200.0,
                center_y=200.0,
                team_name="team_A",
            )
        )
        tracks.append(
            _track_row(
                frame_idx=frame_idx,
                track_id=3,
                object_type="player",
                center_x=owner_x[frame_idx] if frame_idx >= 2 else 760.0,
                center_y=200.0,
                team_name="team_B",
            )
        )
        tracks.append(
            _track_row(
                frame_idx=frame_idx,
                track_id=90,
                object_type="ball",
                center_x=owner_x[frame_idx] + 4.0,
                center_y=200.0,
            )
        )

    events = infer_pass_events(
        tracks=tracks,
        fps=10.0,
        config={
            "possession_max_ball_distance_px": 80.0,
            "possession_smoothing_frames": 1,
            "possession_min_stable_frames": 1,
            "possession_min_segment_frames": 1,
            "pass_min_gap_seconds": 0.0,
            "pass_max_gap_seconds": 1.0,
        },
    )

    assert events == []


def test_pass_inference_respects_gap_thresholds():
    """Long handoff gaps should be filtered by max pass gap."""
    tracks: list[dict] = []

    for frame_idx in range(11):
        # Team A players are always present.
        tracks.append(
            _track_row(
                frame_idx=frame_idx,
                track_id=1,
                object_type="player",
                center_x=300.0 if frame_idx <= 1 else 200.0,
                center_y=220.0,
                team_name="team_A",
            )
        )
        tracks.append(
            _track_row(
                frame_idx=frame_idx,
                track_id=2,
                object_type="player",
                center_x=520.0 if frame_idx >= 9 else 620.0,
                center_y=220.0,
                team_name="team_A",
            )
        )

        # Ball is close to player 1, then away from everyone, then close to player 2.
        if frame_idx <= 1:
            ball_x = 306.0
        elif frame_idx >= 9:
            ball_x = 526.0
        else:
            ball_x = 900.0

        tracks.append(
            _track_row(
                frame_idx=frame_idx,
                track_id=90,
                object_type="ball",
                center_x=ball_x,
                center_y=220.0,
            )
        )

    strict_events = infer_pass_events(
        tracks=tracks,
        fps=10.0,
        config={
            "possession_max_ball_distance_px": 80.0,
            "possession_smoothing_frames": 1,
            "possession_min_stable_frames": 1,
            "possession_min_segment_frames": 1,
            "pass_min_gap_seconds": 0.0,
            "pass_max_gap_seconds": 0.3,
        },
    )
    assert strict_events == []

    permissive_events = infer_pass_events(
        tracks=tracks,
        fps=10.0,
        config={
            "possession_max_ball_distance_px": 80.0,
            "possession_smoothing_frames": 1,
            "possession_min_stable_frames": 1,
            "possession_min_segment_frames": 1,
            "pass_min_gap_seconds": 0.0,
            "pass_max_gap_seconds": 1.5,
        },
    )
    assert len(permissive_events) == 1
    assert permissive_events[0].metadata["gap_seconds"] == pytest.approx(0.8)
