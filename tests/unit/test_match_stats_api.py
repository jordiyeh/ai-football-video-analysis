"""Unit tests for match-stats API endpoints in the local UI server."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastapi import HTTPException

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


def test_match_stats_endpoint_returns_artifact(tmp_path: Path):
    """GET /api/runs/{run_name}/match_stats should return match_stats.json content."""
    runs_dir = tmp_path / "runs"
    run_path = runs_dir / "stats_run"
    run_path.mkdir(parents=True)

    payload = {
        "schema_version": "1.0",
        "summary": {"events_processed": 14, "events_without_team": 1},
        "teams": {
            "ours": {"shots": 6, "goals": 2, "passes": 41, "set_pieces": 3, "possession_share": 0.58},
            "opponent": {"shots": 4, "goals": 1, "passes": 33, "set_pieces": 2, "possession_share": 0.42},
        },
        "totals": {"shots": 10, "goals": 3, "passes": 74, "set_pieces": 5},
    }
    _write_json(run_path / "match_stats.json", payload)

    app = create_app(runs_dir)
    endpoint = _get_route_endpoint(app, "/api/runs/{run_name}/match_stats")
    result = asyncio.run(endpoint("stats_run"))

    assert result["summary"]["events_processed"] == 14
    assert result["teams"]["ours"]["shots"] == 6
    assert result["totals"]["passes"] == 74


def test_match_stats_endpoint_returns_404_when_missing(tmp_path: Path):
    """GET /api/runs/{run_name}/match_stats should return 404 without artifact."""
    runs_dir = tmp_path / "runs"
    (runs_dir / "no_stats_run").mkdir(parents=True)

    app = create_app(runs_dir)
    endpoint = _get_route_endpoint(app, "/api/runs/{run_name}/match_stats")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(endpoint("no_stats_run"))
    assert exc_info.value.status_code == 404


def test_list_runs_includes_match_stats_flag(tmp_path: Path):
    """GET /api/runs should expose has_match_stats per run."""
    runs_dir = tmp_path / "runs"
    run_with_stats = runs_dir / "with_stats"
    run_without_stats = runs_dir / "without_stats"
    run_with_stats.mkdir(parents=True)
    run_without_stats.mkdir(parents=True)

    _write_json(
        run_with_stats / "match_stats.json",
        {"schema_version": "1.0", "teams": {}, "totals": {}, "summary": {}},
    )

    app = create_app(runs_dir)
    endpoint = _get_route_endpoint(app, "/api/runs")
    result = asyncio.run(endpoint())
    run_rows = {row["name"]: row for row in result["runs"]}

    assert run_rows["with_stats"]["has_match_stats"] is True
    assert run_rows["without_stats"]["has_match_stats"] is False
