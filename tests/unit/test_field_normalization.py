"""Tests for zoom-aware field normalization helpers."""

from __future__ import annotations

from src.vision.field.normalization import normalize_tracks_to_field_view


def _track_row(
    frame_idx: int,
    track_id: int,
    object_type: str,
    center_x: float,
    center_y: float,
    box_w: float = 20.0,
    box_h: float = 40.0,
) -> dict:
    """Create a minimal track row with bbox centered at (x, y)."""
    half_w = box_w * 0.5
    half_h = box_h * 0.5
    return {
        "frame_idx": frame_idx,
        "track_id": track_id,
        "object_type": object_type,
        "bbox": [
            center_x - half_w,
            center_y - half_h,
            center_x + half_w,
            center_y + half_h,
        ],
    }


def test_normalization_adds_norm_fields():
    """Every valid track row should contain image_xy and norm_xy fields."""
    tracks = [
        _track_row(0, 1, "player", 100, 200),
        _track_row(0, 2, "player", 300, 240),
        _track_row(0, 3, "player", 520, 260),
        _track_row(0, 99, "ball", 340, 220, box_w=12.0, box_h=12.0),
    ]

    normalized, viewports, summary = normalize_tracks_to_field_view(
        tracks=tracks,
        frame_width=1280,
        frame_height=720,
        config={
            "min_players_per_frame": 3,
            "smoothing_alpha": 1.0,
        },
    )

    assert len(normalized) == len(tracks)
    assert len(viewports) == 1
    assert summary["track_points_normalized"] == len(tracks)

    for row in normalized:
        assert isinstance(row.get("image_xy"), list)
        assert isinstance(row.get("norm_xy"), list)
        assert 0.0 <= float(row["norm_x"]) <= 1.0
        assert 0.0 <= float(row["norm_y"]) <= 1.0


def test_dynamic_viewport_reflects_zoom_change():
    """Viewport width should shrink when players occupy a tighter region (zoom-in effect)."""
    tracks = []

    # Frame 0: wider spread.
    frame0_x = [90, 180, 270, 360, 450, 540]
    for idx, cx in enumerate(frame0_x, start=1):
        tracks.append(_track_row(0, idx, "player", cx, 220 + (idx * 8)))
    tracks.append(_track_row(0, 99, "ball", 320, 260, box_w=12.0, box_h=12.0))

    # Frame 1: tighter spread (zoom-in style framing).
    frame1_x = [220, 260, 300, 340, 380, 420]
    for idx, cx in enumerate(frame1_x, start=10):
        tracks.append(_track_row(1, idx, "player", cx, 230 + ((idx - 10) * 8)))
    tracks.append(_track_row(1, 100, "ball", 320, 270, box_w=12.0, box_h=12.0))

    normalized, viewports, summary = normalize_tracks_to_field_view(
        tracks=tracks,
        frame_width=1280,
        frame_height=720,
        config={
            "min_players_per_frame": 4,
            "smoothing_alpha": 1.0,
            "margin_ratio": 0.05,
            "min_viewport_width_ratio": 0.15,
            "min_viewport_height_ratio": 0.15,
        },
    )

    viewport_by_frame = {row["frame_idx"]: row for row in viewports}
    assert viewport_by_frame[1]["width_ratio"] < viewport_by_frame[0]["width_ratio"]
    assert summary["frames_dynamic"] == 2

    ball_rows = [row for row in normalized if row["object_type"] == "ball"]
    assert len(ball_rows) == 2
    # Ball remains near the center under both zoom states.
    assert abs(ball_rows[0]["norm_x"] - 0.5) < 0.25
    assert abs(ball_rows[1]["norm_x"] - 0.5) < 0.25


def test_fallback_full_frame_when_players_insufficient():
    """If too few players are visible, normalization should fallback to full-frame space."""
    tracks = [
        _track_row(0, 1, "player", 640, 360),
        _track_row(0, 99, "ball", 960, 180, box_w=12.0, box_h=12.0),
    ]

    normalized, viewports, summary = normalize_tracks_to_field_view(
        tracks=tracks,
        frame_width=1280,
        frame_height=720,
        config={
            "min_players_per_frame": 5,
            "smoothing_alpha": 1.0,
        },
    )

    assert len(viewports) == 1
    assert viewports[0]["method"] == "frame_full"
    assert summary["frames_fallback"] == 1

    ball_row = next(row for row in normalized if row["object_type"] == "ball")
    assert ball_row["norm_source"] == "frame_full"
    assert 0.70 <= ball_row["norm_x"] <= 0.80  # ~960/1279
    assert 0.20 <= ball_row["norm_y"] <= 0.30  # ~180/719
