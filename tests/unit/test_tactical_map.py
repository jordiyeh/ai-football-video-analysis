"""Tests for tactical-map visualization renderer."""

from __future__ import annotations

import base64

import cv2
import numpy as np

from src.export.visualizations import VisualizationQuery
from src.export.visualizations.tactical_map import TacticalMapRenderer


def _decode_png(encoded: str) -> np.ndarray:
    """Decode base64 PNG payload into OpenCV image."""
    raw = base64.b64decode(encoded.encode("ascii"))
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    return image


def _sample_tracks() -> list[dict]:
    """Build deterministic player tracks for tactical-map tests."""
    tracks: list[dict] = []
    for frame_idx in range(6):
        tracks.extend(
            [
                {
                    "frame_idx": frame_idx,
                    "track_id": 10,
                    "object_type": "player",
                    "team_name": "ours",
                    "player_id": 7,
                    "norm_xy": [0.20 + (frame_idx * 0.01), 0.35],
                    "confidence": 0.94,
                },
                {
                    "frame_idx": frame_idx,
                    "track_id": 11,
                    "object_type": "player",
                    "team_name": "ours",
                    "player_id": 9,
                    "norm_xy": [0.42 + (frame_idx * 0.01), 0.55],
                    "confidence": 0.91,
                },
                {
                    "frame_idx": frame_idx,
                    "track_id": 20,
                    "object_type": "player",
                    "team_name": "opponent",
                    "player_id": 4,
                    "norm_xy": [0.70 - (frame_idx * 0.01), 0.38],
                    "confidence": 0.90,
                },
                {
                    "frame_idx": frame_idx,
                    "track_id": 21,
                    "object_type": "player",
                    "team_name": "opponent",
                    "player_id": 6,
                    "norm_xy": [0.78 - (frame_idx * 0.01), 0.62],
                    "confidence": 0.88,
                },
            ]
        )
    return tracks


def test_tactical_map_renderer_returns_schema_and_team_shapes():
    """Tactical map should return schema payload with per-team shapes."""
    renderer = TacticalMapRenderer()
    artifact = renderer.render(
        tracks=_sample_tracks(),
        query=VisualizationQuery(extra={"min_samples_per_track": 1}),
        context={
            "canvas_width": 640,
            "canvas_height": 360,
            "team_analytics": {
                "territory": {
                    "teams": {
                        "ours": {"x_zone_control_share": {"left": 0.6, "center": 0.3, "right": 0.1}},
                        "opponent": {"x_zone_control_share": {"left": 0.2, "center": 0.3, "right": 0.5}},
                    }
                },
                "pressing": {
                    "teams": {
                        "ours": {"avg_pressure_score": 0.56, "high_press_rate": 0.31},
                        "opponent": {"avg_pressure_score": 0.48, "high_press_rate": 0.27},
                    }
                },
            },
        },
    )

    payload = artifact.to_dict()

    assert payload["schema_version"] == "1.0"
    assert payload["visualization_type"] == "tactical_map"
    assert payload["width"] == 640
    assert payload["height"] == 360
    assert payload["payload"]["totals"]["teams"] == 2
    assert payload["payload"]["totals"]["tracks"] == 4
    assert payload["metadata"]["has_team_analytics"] is True
    assert len(payload["payload"]["teams"]) == 2
    assert len(payload["payload"]["image_png_base64"]) > 128

    image = _decode_png(payload["payload"]["image_png_base64"])
    assert image.shape[:2] == (360, 640)


def test_tactical_map_player_filter_and_include_points():
    """Player filter should isolate one team shape and include point payload when requested."""
    renderer = TacticalMapRenderer()
    artifact = renderer.render(
        tracks=_sample_tracks(),
        query=VisualizationQuery(
            player_id=7,
            extra={
                "include_points": True,
                "min_samples_per_track": 1,
                "max_tracks_per_team": 5,
                "include_territory": False,
                "include_pressing": False,
            },
        ),
    )

    payload = artifact.to_dict()

    assert payload["metadata"]["mode"] == "player"
    assert payload["metadata"]["include_points"] is True
    assert payload["metadata"]["include_territory"] is False
    assert payload["metadata"]["include_pressing"] is False
    assert payload["payload"]["totals"]["teams"] == 1
    assert payload["payload"]["totals"]["tracks"] == 1
    assert len(payload["payload"].get("points", [])) > 0

    team_shape = payload["payload"]["teams"][0]
    assert team_shape["team_id"] == "ours"
    assert team_shape["unique_tracks"] == 1
    assert team_shape["nodes"][0]["track_id"] == 10
