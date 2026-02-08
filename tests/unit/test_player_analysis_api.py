"""Unit tests for player analytics API endpoints in the local UI server."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from src.ui.server import create_app


def _write_json(path: Path, payload: dict) -> None:
    """Write JSON payload to path."""
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


def test_player_analytics_endpoint_returns_artifact(tmp_path: Path):
    """GET /api/runs/{run_name}/player_analytics should return artifact payload."""
    runs_dir = tmp_path / "runs"
    run_path = runs_dir / "analytics_run"
    run_path.mkdir(parents=True)

    payload = {
        "schema_version": "1.0",
        "run_name": "analytics_run",
        "summary": {
            "runs_analyzed": 3,
            "players_detected": 4,
            "events_total": 21,
            "sprints_total": 12,
        },
        "players": [],
        "runs": [],
    }
    _write_json(run_path / "player_analytics.json", payload)

    app = create_app(runs_dir)
    endpoint = _get_route_endpoint(app, "/api/runs/{run_name}/player_analytics")
    result = asyncio.run(endpoint("analytics_run"))

    assert result["summary"]["players_detected"] == 4
    assert result["summary"]["events_total"] == 21


def test_player_analytics_endpoint_returns_404_when_missing(tmp_path: Path):
    """GET /api/runs/{run_name}/player_analytics should return 404 without artifact."""
    runs_dir = tmp_path / "runs"
    (runs_dir / "missing_analytics").mkdir(parents=True)

    app = create_app(runs_dir)
    endpoint = _get_route_endpoint(app, "/api/runs/{run_name}/player_analytics")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(endpoint("missing_analytics"))
    assert exc_info.value.status_code == 404


def test_list_runs_includes_player_analytics_flag_and_summary(tmp_path: Path):
    """GET /api/runs should expose player analytics availability and summary."""
    runs_dir = tmp_path / "runs"
    run_with_analytics = runs_dir / "with_analytics"
    run_without_analytics = runs_dir / "without_analytics"
    run_with_analytics.mkdir(parents=True)
    run_without_analytics.mkdir(parents=True)

    _write_json(
        run_with_analytics / "player_analytics.json",
        {
            "schema_version": "1.0",
            "summary": {
                "runs_analyzed": 5,
                "players_detected": 11,
                "events_total": 47,
                "sprints_total": 19,
            },
            "players": [],
            "runs": [],
        },
    )

    app = create_app(runs_dir)
    endpoint = _get_route_endpoint(app, "/api/runs")
    result = asyncio.run(endpoint())
    run_rows = {row["name"]: row for row in result["runs"]}

    assert run_rows["with_analytics"]["has_player_analytics"] is True
    assert run_rows["without_analytics"]["has_player_analytics"] is False
    assert run_rows["with_analytics"]["player_analytics_summary"]["players_detected"] == 11
    assert run_rows["with_analytics"]["player_analytics_summary"]["events_total"] == 47
