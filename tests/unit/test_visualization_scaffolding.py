"""Tests for visualization scaffolding and shared field canvas helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.export.visualizations import (
    FieldCanvas,
    FieldCanvasConfig,
    VisualizationQuery,
    VisualizationRenderer,
    accumulate_heat_grid,
    build_field_canvas,
    normalize_heat_grid,
)


class _DummyRenderer(VisualizationRenderer):
    """Simple concrete renderer used to validate base contracts."""

    visualization_type = "dummy_map"

    def render(
        self,
        *,
        tracks: list[dict[str, Any]],
        events: list[dict[str, Any]] | None = None,
        query: VisualizationQuery | None = None,
        context: dict[str, Any] | None = None,
    ):
        del tracks, events, context
        return self.build_artifact(
            title="Dummy Map",
            width=640,
            height=360,
            query=query,
            metadata={"points": 0},
            payload={"image_path": "dummy.png"},
        )


def test_visualization_renderer_artifact_has_schema_version():
    """Renderer artifacts should include stable schema + query serialization."""
    renderer = _DummyRenderer()
    query = VisualizationQuery(team_id="ours", player_id=9, start_t=5.0, end_t=15.0)

    artifact = renderer.render(tracks=[], query=query).to_dict()

    assert artifact["schema_version"] == "1.0"
    assert artifact["visualization_type"] == "dummy_map"
    assert artifact["query"]["team_id"] == "ours"
    assert artifact["query"]["player_id"] == 9
    assert artifact["width"] == 640
    assert artifact["height"] == 360


def test_field_canvas_draws_pitch_and_projects_points():
    """Canvas utility should draw pitch and provide invertible mapping."""
    config = FieldCanvasConfig(width=640, height=360, padding=24)
    canvas = build_field_canvas(config=config, include_markings=True)
    mapper = FieldCanvas(config=config)

    assert canvas.shape == (360, 640, 3)
    assert canvas.dtype == np.uint8

    center_x, center_y = mapper.norm_to_pixel(0.5, 0.5)
    assert tuple(canvas[center_y, center_x]) == config.line_color

    round_trip_x, round_trip_y = mapper.pixel_to_norm(center_x, center_y)
    assert abs(round_trip_x - 0.5) < 0.01
    assert abs(round_trip_y - 0.5) < 0.01

    assert mapper.norm_to_pixel(-0.2, 1.1, clip=True) == mapper.norm_to_pixel(0.0, 1.0, clip=True)


def test_heat_grid_accumulates_weights_and_clipping():
    """Heat grid should accumulate weighted bins with consistent clipping."""
    points = [
        (-0.1, 0.1),  # clipped to x=0
        (0.1, 0.1),
        (0.5, 0.5, 0.5),
        (1.0, 1.0, 2.0),
        (0.99, 0.99),
    ]

    clipped = accumulate_heat_grid(points, bins_x=4, bins_y=4, clip=True)
    unclipped = accumulate_heat_grid(points, bins_x=4, bins_y=4, clip=False)

    assert clipped.shape == (4, 4)
    assert unclipped.shape == (4, 4)
    assert clipped[0, 0] == 2.0
    assert unclipped[0, 0] == 1.0
    assert clipped[2, 2] == 0.5
    assert clipped[3, 3] == 3.0

    normalized = normalize_heat_grid(clipped)
    assert float(np.max(normalized)) == 1.0
    assert normalized[3, 3] == 1.0
