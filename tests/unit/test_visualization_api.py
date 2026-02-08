"""Unit tests for pass-map and tactical-map API endpoints."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pandas as pd

from src.ui.server import create_app


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _get_route_endpoint(app, path: str, method: str = "GET"):
    """Find one route endpoint callable by path and HTTP method."""
    for route in app.routes:
        if getattr(route, "path", None) != path:
            continue
        methods = getattr(route, "methods", set())
        if method.upper() in methods:
            return route.endpoint
    raise AssertionError(f"Route not found: {method} {path}")


def test_pass_map_visualization_endpoint(tmp_path: Path):
    """Pass-map endpoint should render pass map artifact from run tracks/events."""
    runs_dir = tmp_path / "runs"
    run_path = runs_dir / "viz_run"
    run_path.mkdir(parents=True)

    tracks_df = pd.DataFrame(
        [
            {
                "frame_idx": 10,
                "track_id": 10,
                "object_type": "player",
                "team_name": "ours",
                "player_id": 7,
                "norm_xy": [0.20, 0.30],
                "confidence": 0.95,
            },
            {
                "frame_idx": 10,
                "track_id": 11,
                "object_type": "player",
                "team_name": "ours",
                "player_id": 9,
                "norm_xy": [0.48, 0.55],
                "confidence": 0.92,
            },
        ]
    )
    tracks_df.to_parquet(run_path / "tracks.parquet")

    events = [
        {
            "event_type": "pass",
            "frame_idx": 10,
            "timestamp": 1.0,
            "confidence": 0.91,
            "metadata": {
                "team_id": "ours",
                "from_track_id": 10,
                "to_track_id": 11,
                "from_player_id": 7,
                "to_player_id": 9,
            },
        }
    ]
    with open(run_path / "events.jsonl", "w") as f:
        for row in events:
            f.write(json.dumps(row) + "\n")

    _write_json(run_path / "video_metadata.json", {"width": 1920, "height": 1080, "fps": 30})

    app = create_app(runs_dir)
    endpoint = _get_route_endpoint(app, "/api/runs/{run_name}/visualizations/pass_map")
    result = asyncio.run(
        endpoint(
            run_name="viz_run",
            team_id="ours",
            min_pass_count=1,
            canvas_width=800,
            canvas_height=500,
        )
    )

    assert result["visualization_type"] == "pass_map"
    assert result["width"] == 800
    assert result["height"] == 500
    assert result["payload"]["totals"]["passes"] == 1
    assert result["payload"]["totals"]["edges"] == 1
    assert len(result["payload"]["image_png_base64"]) > 128


def test_tactical_map_visualization_endpoint(tmp_path: Path):
    """Tactical-map endpoint should render tactical artifact and include points when requested."""
    runs_dir = tmp_path / "runs"
    run_path = runs_dir / "tactical_run"
    run_path.mkdir(parents=True)

    tracks_df = pd.DataFrame(
        [
            {
                "frame_idx": 0,
                "track_id": 10,
                "object_type": "player",
                "team_name": "ours",
                "player_id": 7,
                "norm_xy": [0.20, 0.30],
                "confidence": 0.95,
            },
            {
                "frame_idx": 1,
                "track_id": 10,
                "object_type": "player",
                "team_name": "ours",
                "player_id": 7,
                "norm_xy": [0.22, 0.32],
                "confidence": 0.94,
            },
            {
                "frame_idx": 0,
                "track_id": 20,
                "object_type": "player",
                "team_name": "opponent",
                "player_id": 4,
                "norm_xy": [0.70, 0.42],
                "confidence": 0.90,
            },
            {
                "frame_idx": 1,
                "track_id": 20,
                "object_type": "player",
                "team_name": "opponent",
                "player_id": 4,
                "norm_xy": [0.68, 0.43],
                "confidence": 0.89,
            },
        ]
    )
    tracks_df.to_parquet(run_path / "tracks.parquet")
    _write_json(run_path / "video_metadata.json", {"width": 1280, "height": 720, "fps": 30})
    _write_json(
        run_path / "team_analytics.json",
        {
            "territory": {
                "teams": {
                    "ours": {"x_zone_control_share": {"left": 0.62, "center": 0.25, "right": 0.13}},
                    "opponent": {"x_zone_control_share": {"left": 0.18, "center": 0.29, "right": 0.53}},
                }
            },
            "pressing": {
                "teams": {
                    "ours": {"avg_pressure_score": 0.58, "high_press_rate": 0.33},
                    "opponent": {"avg_pressure_score": 0.49, "high_press_rate": 0.24},
                }
            },
        },
    )

    app = create_app(runs_dir)
    endpoint = _get_route_endpoint(app, "/api/runs/{run_name}/visualizations/tactical_map")
    result = asyncio.run(
        endpoint(
            run_name="tactical_run",
            include_points=True,
            min_samples_per_track=1,
            canvas_width=820,
            canvas_height=520,
        )
    )

    assert result["visualization_type"] == "tactical_map"
    assert result["width"] == 820
    assert result["height"] == 520
    assert result["metadata"]["include_points"] is True
    assert result["payload"]["totals"]["teams"] == 2
    assert result["payload"]["totals"]["tracks"] == 2
    assert len(result["payload"].get("points", [])) == 4
    assert len(result["payload"]["image_png_base64"]) > 128
