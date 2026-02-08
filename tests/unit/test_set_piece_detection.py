"""Tests for deterministic set-piece inference heuristics."""

from __future__ import annotations

from src.events.set_pieces import SetPieceInferencer, infer_set_piece_events


def _track_row(
    *,
    frame_idx: int,
    track_id: int,
    object_type: str,
    center_x: float,
    center_y: float,
    team_name: str | None = None,
) -> dict:
    """Create a synthetic track row with stable schema fields."""
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


def _build_restart_tracks(
    *,
    origin_xy: tuple[float, float],
    restart_xy: tuple[float, float],
    acting_team: str = "ours",
    stationary_frames: int = 6,
) -> list[dict]:
    """Build one stationary->restart sequence with nearby players."""
    ox, oy = origin_xy
    rx, ry = restart_xy

    tracks: list[dict] = []
    for frame_idx in range(stationary_frames):
        tracks.append(
            _track_row(
                frame_idx=frame_idx,
                track_id=1,
                object_type="player",
                center_x=ox + 8.0,
                center_y=oy + 5.0,
                team_name=acting_team,
            )
        )
        tracks.append(
            _track_row(
                frame_idx=frame_idx,
                track_id=2,
                object_type="player",
                center_x=ox + 220.0,
                center_y=oy + 200.0,
                team_name="opponent",
            )
        )
        tracks.append(
            _track_row(
                frame_idx=frame_idx,
                track_id=90,
                object_type="ball",
                center_x=ox,
                center_y=oy,
            )
        )

    restart_frame = stationary_frames
    tracks.append(
        _track_row(
            frame_idx=restart_frame,
            track_id=1,
            object_type="player",
            center_x=rx + 10.0,
            center_y=ry + 8.0,
            team_name=acting_team,
        )
    )
    tracks.append(
        _track_row(
            frame_idx=restart_frame,
            track_id=2,
            object_type="player",
            center_x=rx + 260.0,
            center_y=ry + 160.0,
            team_name="opponent",
        )
    )
    tracks.append(
        _track_row(
            frame_idx=restart_frame,
            track_id=90,
            object_type="ball",
            center_x=rx,
            center_y=ry,
        )
    )
    return tracks


def test_kickoff_set_piece_inference() -> None:
    """Ball restart from center spot should classify as kickoff."""
    tracks = _build_restart_tracks(
        origin_xy=(500.0, 300.0),
        restart_xy=(670.0, 300.0),
    )

    inferencer = SetPieceInferencer(frame_width=1000, frame_height=600)
    events = inferencer.infer(tracks, fps=10.0)

    assert len(events) == 1
    event = events[0]
    assert event.event_type == "kickoff"
    assert event.metadata is not None
    assert event.metadata["event_family"] == "set_piece"
    assert event.metadata["set_piece_type"] == "kickoff"
    assert event.metadata["team_id"] == "ours"
    assert event.metadata["provenance"]["detector"] == "set_piece_heuristics"


def test_throw_in_set_piece_inference() -> None:
    """Ball restart from side edge should classify as throw-in."""
    tracks = _build_restart_tracks(
        origin_xy=(20.0, 300.0),
        restart_xy=(170.0, 306.0),
    )

    events = infer_set_piece_events(
        tracks=tracks,
        fps=10.0,
        frame_width=1000,
        frame_height=600,
    )

    assert len(events) == 1
    assert events[0].event_type == "throw_in"
    assert events[0].metadata["set_piece_type"] == "throw_in"


def test_corner_kick_set_piece_inference() -> None:
    """Ball restart from a corner should classify as corner kick."""
    tracks = _build_restart_tracks(
        origin_xy=(965.0, 30.0),
        restart_xy=(820.0, 150.0),
    )

    events = infer_set_piece_events(
        tracks=tracks,
        fps=10.0,
        frame_width=1000,
        frame_height=600,
    )

    assert len(events) == 1
    assert events[0].event_type == "corner_kick"
    assert events[0].metadata["set_piece_type"] == "corner_kick"


def test_goal_kick_set_piece_inference() -> None:
    """Ball restart from top center goal band should classify as goal kick."""
    tracks = _build_restart_tracks(
        origin_xy=(500.0, 42.0),
        restart_xy=(505.0, 200.0),
    )

    events = infer_set_piece_events(
        tracks=tracks,
        fps=10.0,
        frame_width=1000,
        frame_height=600,
    )

    assert len(events) == 1
    assert events[0].event_type == "goal_kick"
    assert events[0].metadata["set_piece_type"] == "goal_kick"


def test_free_kick_set_piece_inference() -> None:
    """Ball restart away from center/edge/corners should classify as free kick."""
    tracks = _build_restart_tracks(
        origin_xy=(740.0, 260.0),
        restart_xy=(860.0, 310.0),
    )

    events = infer_set_piece_events(
        tracks=tracks,
        fps=10.0,
        frame_width=1000,
        frame_height=600,
    )

    assert len(events) == 1
    assert events[0].event_type == "free_kick"
    assert events[0].metadata["set_piece_type"] == "free_kick"


def test_set_piece_thresholds_are_configurable() -> None:
    """Per-type confidence thresholds should be configurable."""
    tracks = _build_restart_tracks(
        origin_xy=(500.0, 300.0),
        restart_xy=(660.0, 300.0),
    )

    strict_events = infer_set_piece_events(
        tracks=tracks,
        fps=10.0,
        frame_width=1000,
        frame_height=600,
        config={
            "kickoff_min_confidence": 0.99,
            "throw_in_min_confidence": 0.99,
            "corner_kick_min_confidence": 0.99,
            "free_kick_min_confidence": 0.99,
            "goal_kick_min_confidence": 0.99,
        },
    )
    assert strict_events == []

    permissive_events = infer_set_piece_events(
        tracks=tracks,
        fps=10.0,
        frame_width=1000,
        frame_height=600,
        config={
            "kickoff_min_confidence": 0.40,
            "throw_in_min_confidence": 0.99,
            "corner_kick_min_confidence": 0.99,
            "free_kick_min_confidence": 0.99,
            "goal_kick_min_confidence": 0.99,
        },
    )
    assert len(permissive_events) == 1
    assert permissive_events[0].event_type == "kickoff"
