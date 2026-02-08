"""Tests for pass-map visualization renderer."""

from __future__ import annotations

import base64

import cv2
import numpy as np

from src.events.detection import Event
from src.export.visualizations import VisualizationQuery
from src.export.visualizations.pass_map import PassMapRenderer


def _decode_png(encoded: str) -> np.ndarray:
    """Decode base64 PNG payload into OpenCV image."""
    raw = base64.b64decode(encoded.encode("ascii"))
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    return image


def _sample_tracks() -> list[dict]:
    """Build deterministic player tracks for pass-map tests."""
    return [
        {
            "frame_idx": 10,
            "track_id": 10,
            "object_type": "player",
            "team_name": "ours",
            "player_id": 7,
            "norm_xy": [0.20, 0.35],
            "confidence": 0.95,
        },
        {
            "frame_idx": 10,
            "track_id": 11,
            "object_type": "player",
            "team_name": "ours",
            "player_id": 9,
            "norm_xy": [0.48, 0.52],
            "confidence": 0.93,
        },
        {
            "frame_idx": 20,
            "track_id": 21,
            "object_type": "player",
            "team_name": "opponent",
            "player_id": 4,
            "norm_xy": [0.76, 0.42],
            "confidence": 0.90,
        },
    ]


def _sample_events() -> list[dict]:
    """Build deterministic pass events for pass-map tests."""
    return [
        {
            "event_type": "pass",
            "frame_idx": 10,
            "timestamp": 1.0,
            "confidence": 0.92,
            "metadata": {
                "team_id": "ours",
                "from_track_id": 10,
                "to_track_id": 11,
                "from_player_id": 7,
                "to_player_id": 9,
            },
        },
        {
            "event_type": "pass",
            "frame_idx": 12,
            "timestamp": 1.2,
            "confidence": 0.87,
            "metadata": {
                "team_id": "ours",
                "from_track_id": 10,
                "to_track_id": 11,
                "from_player_id": 7,
                "to_player_id": 9,
            },
        },
        {
            "event_type": "pass",
            "frame_idx": 20,
            "timestamp": 2.0,
            "confidence": 0.78,
            "metadata": {
                "team_id": "opponent",
                "from_track_id": 21,
                "to_track_id": 10,
                "from_player_id": 4,
                "to_player_id": 7,
            },
        },
    ]


def test_pass_map_renderer_returns_schema_and_image_payload():
    """Pass map should return schema-versioned artifact with encoded image payload."""
    renderer = PassMapRenderer()
    artifact = renderer.render(
        tracks=_sample_tracks(),
        events=_sample_events(),
        context={"canvas_width": 640, "canvas_height": 360},
    )

    payload = artifact.to_dict()

    assert payload["schema_version"] == "1.0"
    assert payload["visualization_type"] == "pass_map"
    assert payload["width"] == 640
    assert payload["height"] == 360
    assert payload["payload"]["totals"]["passes"] == 3
    assert payload["payload"]["totals"]["edges"] == 2
    assert payload["metadata"]["passes_rendered"] == 3
    assert len(payload["payload"]["image_png_base64"]) > 128

    image = _decode_png(payload["payload"]["image_png_base64"])
    assert image.shape[:2] == (360, 640)


def test_pass_map_filters_player_team_and_min_pass_count():
    """Team/player filters and edge count threshold should narrow output."""
    dataclass_event = Event(
        event_type="pass",
        frame_idx=13,
        timestamp=1.3,
        confidence=0.70,
        location=(90.0, 60.0),
        metadata={
            "team_id": "ours",
            "from_track_id": 10,
            "to_track_id": 11,
            "from_player_id": 7,
            "to_player_id": 9,
        },
    )

    renderer = PassMapRenderer()
    artifact = renderer.render(
        tracks=_sample_tracks(),
        events=_sample_events() + [dataclass_event],
        query=VisualizationQuery(
            team_id="ours",
            player_id=9,
            extra={"min_pass_count": 2},
        ),
    )

    payload = artifact.to_dict()
    edges = payload["payload"]["edges"]
    totals = payload["payload"]["totals"]

    assert len(edges) == 1
    assert edges[0]["team_id"] == "ours"
    assert edges[0]["to_player_id"] == 9
    assert edges[0]["pass_count"] == 3
    assert totals["passes"] == 3
    assert totals["edges"] == 1
    assert totals["nodes"] == 2
