"""Tests for shot-map visualization renderer."""

from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest

from src.events.detection import Event
from src.export.visualizations import VisualizationQuery
from src.export.visualizations.shot_map import ShotMapRenderer


def _decode_png(encoded: str) -> np.ndarray:
    """Decode a base64 PNG payload into an OpenCV image."""
    raw = base64.b64decode(encoded.encode("ascii"))
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    return image


def _sample_events() -> list[dict]:
    """Build deterministic sample events for shot map tests."""
    return [
        {
            "event_type": "shot",
            "frame_idx": 90,
            "timestamp": 3.0,
            "confidence": 0.9,
            "location": [100.0, 50.0],
            "metadata": {
                "team_id": "ours",
                "player_id": 9,
                "provenance": "detector",
            },
        },
        {
            "event_type": "goal",
            "frame_idx": 120,
            "timestamp": 4.0,
            "confidence": 0.92,
            "location": [104.0, 28.0],
            "metadata": {"shot_frame": 90},
        },
        {
            "event_type": "shot",
            "frame_idx": 150,
            "timestamp": 5.0,
            "confidence": 0.7,
            "location": [160.0, 68.0],
            "metadata": {
                "team_id": "opponent",
                "player_id": 4,
                "provenance": "detector",
            },
        },
    ]


def test_shot_map_renderer_returns_schema_and_image_payload():
    """Shot map artifacts should include schema metadata + PNG payload."""
    renderer = ShotMapRenderer()
    artifact = renderer.render(
        tracks=[],
        events=_sample_events(),
        context={
            "frame_width": 200,
            "frame_height": 100,
            "canvas_width": 640,
            "canvas_height": 360,
        },
    )

    payload = artifact.to_dict()

    assert payload["schema_version"] == "1.0"
    assert payload["visualization_type"] == "shot_map"
    assert payload["width"] == 640
    assert payload["height"] == 360
    assert payload["metadata"]["shots_rendered"] == 2
    assert payload["payload"]["totals"]["shots"] == 2
    assert payload["payload"]["totals"]["goals"] == 1
    assert len(payload["payload"]["image_png_base64"]) > 128

    image = _decode_png(payload["payload"]["image_png_base64"])
    assert image.shape[:2] == (360, 640)


def test_shot_map_filters_by_team_and_player():
    """Team/player filters should narrow the shot points deterministically."""
    renderer = ShotMapRenderer()
    artifact = renderer.render(
        tracks=[],
        events=_sample_events(),
        query=VisualizationQuery(team_id="ours", player_id=9),
        context={"frame_width": 200, "frame_height": 100},
    )

    points = artifact.to_dict()["payload"]["points"]

    assert len(points) == 1
    assert points[0]["team_id"] == "ours"
    assert points[0]["player_id"] == 9
    assert points[0]["is_goal"] is True


def test_shot_map_event_dataclass_and_pixel_fallback():
    """Dataclass events should render via pixel-to-normalized fallback."""
    shot = Event(
        event_type="shot",
        frame_idx=12,
        timestamp=0.4,
        confidence=0.65,
        location=(50.0, 25.0),
        metadata={"track_id": 11},
    )
    renderer = ShotMapRenderer()
    artifact = renderer.render(
        tracks=[{"track_id": 11, "object_type": "player", "team_id": "opponent"}],
        events=[shot],
        query=VisualizationQuery(start_t=0.1, end_t=1.0),
        context={"frame_width": 100, "frame_height": 50},
    )

    points = artifact.to_dict()["payload"]["points"]
    assert len(points) == 1

    point = points[0]
    assert point["team_id"] == "opponent"
    assert point["player_id"] == 11
    assert point["provenance"] == "event.location"
    assert point["norm_xy"][0] == pytest.approx(0.5, abs=1e-3)
    assert point["norm_xy"][1] == pytest.approx(0.5, abs=1e-3)
