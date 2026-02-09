"""Tests for possession-by-minute and phase rollups."""

from __future__ import annotations

from src.analytics import build_team_analytics


def _track_row(
    *,
    frame_idx: int,
    track_id: int,
    object_type: str,
    center_x: float,
    center_y: float,
    frame_width: int,
    frame_height: int,
    team_name: str | None = None,
) -> dict:
    """Create a minimal synthetic track row."""
    box_w = 20.0
    box_h = 40.0
    row = {
        "frame_idx": frame_idx,
        "timestamp": frame_idx / 10.0,
        "track_id": track_id,
        "object_type": object_type,
        "confidence": 0.9,
        "bbox": [
            center_x - (box_w / 2.0),
            center_y - (box_h / 2.0),
            center_x + (box_w / 2.0),
            center_y + (box_h / 2.0),
        ],
        "image_x": center_x,
        "image_y": center_y,
        "image_xy": [center_x, center_y],
        "norm_x": center_x / max(1.0, frame_width - 1.0),
        "norm_y": center_y / max(1.0, frame_height - 1.0),
    }
    row["norm_xy"] = [row["norm_x"], row["norm_y"]]
    if team_name is not None:
        row["team_name"] = team_name
    return row


def test_team_analytics_builds_possession_by_minute_and_phase():
    """Team analytics should provide per-minute and phase possession breakdowns."""
    frame_width = 1000
    frame_height = 500
    tracks = []

    for frame_idx in range(30):
        if frame_idx < 10:
            owner_track = 1
            owner_x = 200 + (frame_idx * 20)
        elif frame_idx < 20:
            owner_track = 2
            owner_x = 450 + ((frame_idx - 10) * 40)
        else:
            owner_track = 3
            owner_x = 800 - ((frame_idx - 20) * 60)

        team_a_track_1_x = owner_x if owner_track == 1 else 110
        team_a_track_2_x = owner_x if owner_track == 2 else 520
        team_b_track_3_x = owner_x if owner_track == 3 else 890

        tracks.append(
            _track_row(
                frame_idx=frame_idx,
                track_id=1,
                object_type="player",
                center_x=team_a_track_1_x,
                center_y=220,
                frame_width=frame_width,
                frame_height=frame_height,
                team_name="team_A",
            )
        )
        tracks.append(
            _track_row(
                frame_idx=frame_idx,
                track_id=2,
                object_type="player",
                center_x=team_a_track_2_x,
                center_y=260,
                frame_width=frame_width,
                frame_height=frame_height,
                team_name="team_A",
            )
        )
        tracks.append(
            _track_row(
                frame_idx=frame_idx,
                track_id=3,
                object_type="player",
                center_x=team_b_track_3_x,
                center_y=240,
                frame_width=frame_width,
                frame_height=frame_height,
                team_name="team_B",
            )
        )
        tracks.append(
            _track_row(
                frame_idx=frame_idx,
                track_id=90,
                object_type="ball",
                center_x=owner_x + 4.0,
                center_y=245,
                frame_width=frame_width,
                frame_height=frame_height,
            )
        )

    analytics = build_team_analytics(
        tracks=tracks,
        assignments=[],
        fps=10.0,
        frame_width=frame_width,
        frame_height=frame_height,
        config={
            "use_norm_coordinates": True,
            "possession_max_ball_distance_px": 60.0,
            "possession_smoothing_frames": 1,
            "possession_min_stable_frames": 1,
            "possession_min_segment_frames": 1,
            "possession_minute_bucket_seconds": 1.0,
            "possession_phase_boundaries_norm_x": [0.34, 0.67],
        },
    )

    possession = analytics["possession"]
    assert possession["teams"]["team_A"]["frames"] == 20
    assert possession["teams"]["team_B"]["frames"] == 10

    by_minute = possession["by_minute"]
    assert [row["minute_index"] for row in by_minute] == [0, 1, 2]
    assert by_minute[0]["teams"]["team_A"]["frames"] == 10
    assert by_minute[1]["teams"]["team_A"]["frames"] == 10
    assert by_minute[2]["teams"]["team_B"]["frames"] == 10
    assert by_minute[2]["dominant_team"] == "team_B"

    by_phase = possession["by_phase"]
    assert by_phase["phase_boundaries_norm_x"] == [0.34, 0.67]
    assert by_phase["teams"]["team_A"]["direction_sign"] == 1
    assert by_phase["teams"]["team_A"]["direction_method"] == "segment_delta"
    assert by_phase["teams"]["team_B"]["direction_sign"] == -1
    assert by_phase["teams"]["team_B"]["direction_method"] == "segment_delta"

    team_a_phase_total = (
        by_phase["teams"]["team_A"]["phases"]["build_up"]["frames"]
        + by_phase["teams"]["team_A"]["phases"]["middle_third"]["frames"]
        + by_phase["teams"]["team_A"]["phases"]["final_third"]["frames"]
        + by_phase["teams"]["team_A"]["unknown_frames"]
    )
    team_b_phase_total = (
        by_phase["teams"]["team_B"]["phases"]["build_up"]["frames"]
        + by_phase["teams"]["team_B"]["phases"]["middle_third"]["frames"]
        + by_phase["teams"]["team_B"]["phases"]["final_third"]["frames"]
        + by_phase["teams"]["team_B"]["unknown_frames"]
    )
    assert team_a_phase_total == possession["teams"]["team_A"]["frames"]
    assert team_b_phase_total == possession["teams"]["team_B"]["frames"]
