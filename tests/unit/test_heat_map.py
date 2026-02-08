"""Tests for heat-map visualization renderer."""

from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest

from src.export.visualizations import VisualizationQuery
from src.export.visualizations.heat_map import HeatMapRenderer


def _decode_png(encoded: str) -> np.ndarray:
    """Decode a base64 PNG payload into an OpenCV image."""
    raw = base64.b64decode(encoded.encode("ascii"))
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    return image


def _sample_tracks() -> list[dict]:
    """Build deterministic sample tracks for heat map tests."""
    return [
        {
            "frame_idx": 0,
            "track_id": 10,
            "object_type": "player",
            "team_name": "ours",
            "norm_xy": [0.20, 0.30],
            "confidence": 0.95,
        },
        {
            "frame_idx": 1,
            "track_id": 10,
            "object_type": "player",
            "team_name": "ours",
            "norm_x": 0.25,
            "norm_y": 0.36,
            "confidence": 0.90,
        },
        {
            "frame_idx": 2,
            "track_id": 22,
            "object_type": "player",
            "team_name": "opponent",
            "norm_xy": [0.72, 0.64],
            "confidence": 0.85,
        },
        {
            "frame_idx": 3,
            "track_id": 99,
            "object_type": "ball",
            "norm_xy": [0.50, 0.50],
        },
    ]


def test_heat_map_renderer_returns_schema_and_image_payload():
    """Heat map artifacts should include schema metadata + PNG payload."""
    renderer = HeatMapRenderer()
    artifact = renderer.render(
        tracks=_sample_tracks(),
        query=VisualizationQuery(team_id="ours"),
        context={"canvas_width": 640, "canvas_height": 360},
    )

    payload = artifact.to_dict()

    assert payload["schema_version"] == "1.0"
    assert payload["visualization_type"] == "heat_map"
    assert payload["width"] == 640
    assert payload["height"] == 360
    assert payload["metadata"]["mode"] == "team"
    assert payload["metadata"]["tracks_rendered"] == 2
    assert payload["payload"]["totals"]["samples"] == 2
    assert payload["payload"]["totals"]["max_cell_weight"] > 0.0
    assert len(payload["payload"]["image_png_base64"]) > 128

    image = _decode_png(payload["payload"]["image_png_base64"])
    assert image.shape[:2] == (360, 640)


def test_heat_map_player_filter_and_image_coordinate_fallback():
    """Player mode should filter to one player and normalize image-space fallback."""
    tracks = [
        {
            "frame_idx": 6,
            "track_id": 42,
            "player_id": 7,
            "object_type": "player",
            "team_name": "ours",
            "image_xy": [64.0, 32.0],
            "confidence": 0.80,
        },
        {
            "frame_idx": 7,
            "track_id": 43,
            "player_id": 8,
            "object_type": "player",
            "team_name": "ours",
            "image_xy": [140.0, 90.0],
            "confidence": 0.80,
        },
    ]

    renderer = HeatMapRenderer()
    artifact = renderer.render(
        tracks=tracks,
        query=VisualizationQuery(
            player_id=7,
            extra={"include_points": True},
        ),
        context={"frame_width": 160, "frame_height": 100},
    )

    payload = artifact.to_dict()
    assert payload["metadata"]["mode"] == "player"
    assert payload["metadata"]["tracks_rendered"] == 1
    assert payload["payload"]["totals"]["samples"] == 1

    points = payload["payload"]["points"]
    assert len(points) == 1
    assert points[0]["player_id"] == 7
    assert points[0]["team_id"] == "ours"
    assert points[0]["provenance"] == "track.image_xy"
    assert points[0]["norm_xy"][0] == pytest.approx(0.4, abs=1e-3)
    assert points[0]["norm_xy"][1] == pytest.approx(0.32, abs=1e-3)


def test_heat_map_defaults_to_player_objects_only():
    """Ball tracks should be skipped unless include_ball is explicitly enabled."""
    renderer = HeatMapRenderer()
    tracks = _sample_tracks()

    artifact_default = renderer.render(
        tracks=tracks,
        query=VisualizationQuery(),
    ).to_dict()
    assert artifact_default["payload"]["totals"]["samples"] == 3
    assert artifact_default["metadata"]["object_filter"] == ["player"]

    artifact_with_ball = renderer.render(
        tracks=tracks,
        query=VisualizationQuery(extra={"include_ball": True}),
    ).to_dict()
    assert artifact_with_ball["payload"]["totals"]["samples"] == 4
    assert "ball" in artifact_with_ball["metadata"]["object_filter"]
