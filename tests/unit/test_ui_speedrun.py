"""Unit tests for UI speedrun playback window endpoint."""

import asyncio
import json
from pathlib import Path

from fastapi import HTTPException
import pytest

from src.ui.server import create_app


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _get_route_endpoint(app, path: str, method: str = "GET"):
    for route in app.routes:
        if getattr(route, "path", None) != path:
            continue
        methods = getattr(route, "methods", set())
        if method.upper() in methods:
            return route.endpoint
    raise AssertionError(f"Route not found: {method} {path}")


def test_speedrun_endpoint_builds_high_and_low_action_windows(tmp_path: Path):
    """Speedrun endpoint should merge event windows and expose low-action spans."""
    runs_dir = tmp_path / "runs"
    run_path = runs_dir / "speedrun_run"
    run_path.mkdir(parents=True)

    _write_json(
        run_path / "video_metadata.json",
        {
            "fps": 30.0,
            "duration": 120.0,
            "width": 1920,
            "height": 1080,
        },
    )

    events = [
        {"event_type": "shot", "timestamp": 10.0, "confidence": 0.91},
        {"event_type": "goal", "timestamp": 14.0, "confidence": 0.95},
        {"event_type": "pass", "timestamp": 70.0, "confidence": 0.84},
        {"event_type": "throw_in", "timestamp": 40.0, "confidence": 0.2},
    ]
    with open(run_path / "events.jsonl", "w") as f:
        for row in events:
            f.write(json.dumps(row) + "\n")

    app = create_app(runs_dir)
    endpoint = _get_route_endpoint(app, "/api/runs/{run_name}/playback/speedrun")
    result = asyncio.run(
        endpoint(
            run_name="speedrun_run",
            pre_padding=4.0,
            post_padding=6.0,
            min_window=2.0,
            merge_gap=0.25,
            min_confidence=0.5,
        )
    )

    assert result["schema_version"] == "1.0"
    assert result["run_name"] == "speedrun_run"
    assert result["events_considered"] == 3
    assert result["fallback_full_match_window"] is False
    assert result["speedrun_ready"] is True

    high_action = result["high_action_windows"]
    assert len(high_action) == 2
    assert high_action[0]["start"] == pytest.approx(6.0)
    assert high_action[0]["end"] == pytest.approx(20.0)
    assert high_action[0]["event_count"] == 2
    assert high_action[1]["start"] == pytest.approx(66.0)
    assert high_action[1]["end"] == pytest.approx(76.0)
    assert high_action[1]["event_count"] == 1

    low_action = result["low_action_windows"]
    assert len(low_action) == 3
    assert low_action[0]["start"] == pytest.approx(0.0)
    assert low_action[0]["end"] == pytest.approx(6.0)
    assert low_action[1]["start"] == pytest.approx(20.0)
    assert low_action[1]["end"] == pytest.approx(66.0)
    assert low_action[2]["start"] == pytest.approx(76.0)
    assert low_action[2]["end"] == pytest.approx(120.0)


def test_speedrun_endpoint_falls_back_to_full_match_window_when_events_missing(tmp_path: Path):
    """When no events exist, endpoint should expose one full-match action window."""
    runs_dir = tmp_path / "runs"
    run_path = runs_dir / "quiet_run"
    run_path.mkdir(parents=True)

    _write_json(
        run_path / "video_metadata.json",
        {
            "fps": 25.0,
            "duration": 90.0,
            "width": 1280,
            "height": 720,
        },
    )

    app = create_app(runs_dir)
    endpoint = _get_route_endpoint(app, "/api/runs/{run_name}/playback/speedrun")
    result = asyncio.run(endpoint(run_name="quiet_run"))

    assert result["events_considered"] == 0
    assert result["fallback_full_match_window"] is True
    assert result["speedrun_ready"] is True
    assert len(result["high_action_windows"]) == 1
    assert result["high_action_windows"][0]["start"] == pytest.approx(0.0)
    assert result["high_action_windows"][0]["end"] == pytest.approx(90.0)
    assert result["low_action_windows"] == []


def test_speedrun_endpoint_returns_404_for_missing_run(tmp_path: Path):
    """Speedrun endpoint should return HTTP 404 when run directory is absent."""
    runs_dir = tmp_path / "runs"
    app = create_app(runs_dir)
    endpoint = _get_route_endpoint(app, "/api/runs/{run_name}/playback/speedrun")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(endpoint(run_name="missing_run"))
    assert exc_info.value.status_code == 404
