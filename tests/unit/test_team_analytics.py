"""Tests for team-level tactical analytics."""

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
    include_norm: bool = True,
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
    }
    if include_norm:
        row["norm_x"] = center_x / max(1.0, frame_width - 1.0)
        row["norm_y"] = center_y / max(1.0, frame_height - 1.0)
        row["norm_xy"] = [row["norm_x"], row["norm_y"]]
    if team_name is not None:
        row["team_name"] = team_name
    return row


def test_team_analytics_infers_possession_passes_and_pressing():
    """Possession timeline should drive pass-network and pressing summaries."""
    frame_width = 1000
    frame_height = 500
    tracks = []

    # 6 frames, two teams, deterministic possession transfer:
    # team_A(track 1) -> team_A(track 2) -> team_B(track 3)
    owner_x = [300, 305, 450, 455, 700, 705]
    for frame_idx in range(6):
        tracks.append(
            _track_row(
                frame_idx=frame_idx,
                track_id=1,
                object_type="player",
                center_x=owner_x[frame_idx] if frame_idx < 2 else 280,
                center_y=250,
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
                center_x=owner_x[frame_idx] if 2 <= frame_idx <= 3 else 520,
                center_y=255,
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
                center_x=owner_x[frame_idx] if frame_idx >= 4 else 710,
                center_y=245,
                frame_width=frame_width,
                frame_height=frame_height,
                team_name="team_B",
            )
        )
        # Defender from opposite team stays near the carrier to trigger pressure.
        tracks.append(
            _track_row(
                frame_idx=frame_idx,
                track_id=4,
                object_type="player",
                center_x=owner_x[frame_idx] + 22.0,
                center_y=248,
                frame_width=frame_width,
                frame_height=frame_height,
                team_name="team_B" if frame_idx < 4 else "team_A",
            )
        )
        tracks.append(
            _track_row(
                frame_idx=frame_idx,
                track_id=90,
                object_type="ball",
                center_x=owner_x[frame_idx] + 8.0,
                center_y=250,
                frame_width=frame_width,
                frame_height=frame_height,
            )
        )

    assignments = [
        {"track_id": 1, "player_id": 101, "player_name": "A1", "confidence": 0.95},
        {"track_id": 2, "player_id": 102, "player_name": "A2", "confidence": 0.93},
        {"track_id": 3, "player_id": 201, "player_name": "B1", "confidence": 0.91},
    ]
    analytics = build_team_analytics(
        tracks=tracks,
        assignments=assignments,
        fps=10.0,
        frame_width=frame_width,
        frame_height=frame_height,
        config={
            "use_norm_coordinates": True,
            "possession_max_ball_distance_px": 80.0,
            "possession_smoothing_frames": 1,
            "possession_min_stable_frames": 1,
            "possession_min_segment_frames": 1,
            "pass_min_gap_seconds": 0.0,
            "pass_max_gap_seconds": 1.0,
            "pressure_radius_norm": 0.08,
            "high_press_threshold": 0.2,
            "high_press_min_frames": 1,
            "min_assignment_confidence": 0.6,
        },
    )

    summary = analytics["summary"]
    assert summary["frames_with_ball"] == 6
    assert summary["frames_with_possession"] == 6
    assert summary["teams_detected"] == ["team_A", "team_B"]

    possession = analytics["possession"]["teams"]
    assert possession["team_A"]["frames"] == 4
    assert possession["team_B"]["frames"] == 2

    pass_edges = analytics["pass_network_edges"]
    edge = next(
        (
            row for row in pass_edges
            if row["team"] == "team_A" and row["from_track_id"] == 1 and row["to_track_id"] == 2
        ),
        None,
    )
    assert edge is not None
    assert edge["pass_count"] >= 1
    assert edge["from_player_id"] == 101
    assert edge["to_player_id"] == 102

    pressing = analytics["pressing"]["teams"]
    assert pressing["team_B"]["frames_defending"] >= 4
    assert pressing["team_B"]["high_press_frames"] >= 1


def test_team_analytics_territory_works_without_norm_xy():
    """Territory metrics should fallback to image-space normalization when needed."""
    tracks = [
        _track_row(
            frame_idx=0,
            track_id=1,
            object_type="player",
            center_x=120,
            center_y=180,
            frame_width=640,
            frame_height=360,
            team_name="team_A",
            include_norm=False,
        ),
        _track_row(
            frame_idx=0,
            track_id=2,
            object_type="player",
            center_x=520,
            center_y=190,
            frame_width=640,
            frame_height=360,
            team_name="team_B",
            include_norm=False,
        ),
        _track_row(
            frame_idx=1,
            track_id=1,
            object_type="player",
            center_x=140,
            center_y=170,
            frame_width=640,
            frame_height=360,
            team_name="team_A",
            include_norm=False,
        ),
        _track_row(
            frame_idx=1,
            track_id=2,
            object_type="player",
            center_x=500,
            center_y=210,
            frame_width=640,
            frame_height=360,
            team_name="team_B",
            include_norm=False,
        ),
    ]

    analytics = build_team_analytics(
        tracks=tracks,
        assignments=[],
        fps=10.0,
        frame_width=640,
        frame_height=360,
        config={"use_norm_coordinates": True},
    )

    assert analytics["summary"]["frames_with_ball"] == 0
    territory = analytics["territory"]
    assert territory["samples"] == 4
    assert "team_A" in territory["teams"]
    assert "left" in territory["teams"]["team_A"]["x_bins"]


def test_team_analytics_builds_possession_by_minute_and_phase():
    """Possession output should include per-minute windows and phase rollups."""
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
