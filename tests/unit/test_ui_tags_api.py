"""Unit tests for run tag endpoints in the UI server."""

import asyncio
from pathlib import Path

import pytest
from fastapi import HTTPException

from src.identity import PlayerDatabase
from src.ui.server import CreateTagBody, UpdateTagBody, create_app


def _get_route_endpoint(app, path: str, method: str = "GET"):
    """Find a route endpoint callable by path and HTTP method."""
    for route in app.routes:
        if getattr(route, "path", None) != path:
            continue
        methods = getattr(route, "methods", set())
        if method.upper() in methods:
            return route.endpoint
    raise AssertionError(f"Route not found: {method} {path}")


def test_run_tags_api_crud_and_filters(tmp_path: Path):
    """Run tag endpoints should support create/list/filter/update/delete flows."""
    runs_dir = tmp_path / "runs"
    run_path = runs_dir / "match_run"
    run_path.mkdir(parents=True, exist_ok=True)

    db_path = runs_dir / "players.db"
    with PlayerDatabase(db_path) as db:
        team = db.create_team(name="FC Tags")
        player = db.create_player(name="Ava Nine", team_hint="ours")
        db.set_player_team(player.player_id, team.team_id)

    app = create_app(runs_dir)
    create_endpoint = _get_route_endpoint(app, "/api/runs/{run_name}/tags", "POST")
    list_endpoint = _get_route_endpoint(app, "/api/runs/{run_name}/tags")
    update_endpoint = _get_route_endpoint(app, "/api/runs/{run_name}/tags/{tag_id}", "PATCH")
    delete_endpoint = _get_route_endpoint(app, "/api/runs/{run_name}/tags/{tag_id}", "DELETE")

    created_a = asyncio.run(
        create_endpoint(
            run_name="match_run",
            body=CreateTagBody(
                label="press_trigger",
                category="tactical",
                start_time=12.5,
                end_time=13.0,
                frame_idx=375,
                track_id=9,
                player_id=player.player_id,
                team_id=team.team_id,
                confidence=0.81,
                notes="manual marker",
                metadata={"zone": "left"},
            ),
        )
    )
    assert created_a["success"] is True
    assert created_a["tag"]["label"] == "press_trigger"
    assert created_a["tag"]["player_name"] == "Ava Nine"
    assert created_a["tag"]["team_name"] == "FC Tags"
    tag_a_id = int(created_a["tag"]["tag_id"])

    created_b = asyncio.run(
        create_endpoint(
            run_name="match_run",
            body=CreateTagBody(
                label="set_piece",
                category="event",
                start_time=42.0,
                source="imported",
            ),
        )
    )
    assert created_b["success"] is True
    tag_b_id = int(created_b["tag"]["tag_id"])

    listed_all = asyncio.run(list_endpoint(run_name="match_run"))
    assert listed_all["count"] == 2
    assert len(listed_all["tags"]) == 2

    listed_tactical = asyncio.run(list_endpoint(run_name="match_run", category="tactical"))
    assert listed_tactical["count"] == 1
    assert listed_tactical["tags"][0]["tag_id"] == tag_a_id

    listed_imported = asyncio.run(list_endpoint(run_name="match_run", source="imported"))
    assert listed_imported["count"] == 1
    assert listed_imported["tags"][0]["tag_id"] == tag_b_id

    listed_window = asyncio.run(list_endpoint(run_name="match_run", min_time=40.0, max_time=50.0))
    assert listed_window["count"] == 1
    assert listed_window["tags"][0]["tag_id"] == tag_b_id

    updated = asyncio.run(
        update_endpoint(
            run_name="match_run",
            tag_id=tag_a_id,
            body=UpdateTagBody(
                notes="updated note",
                confidence=0.92,
                end_time=13.8,
                metadata={"zone": "center"},
            ),
        )
    )
    assert updated["success"] is True
    assert updated["tag"]["notes"] == "updated note"
    assert float(updated["tag"]["confidence"]) == pytest.approx(0.92)
    assert float(updated["tag"]["end_time"]) == pytest.approx(13.8)
    assert updated["tag"]["metadata"]["zone"] == "center"

    deleted = asyncio.run(delete_endpoint(run_name="match_run", tag_id=tag_a_id))
    assert deleted["success"] is True
    assert deleted["tag_id"] == tag_a_id

    listed_after_delete = asyncio.run(list_endpoint(run_name="match_run"))
    assert listed_after_delete["count"] == 1
    assert listed_after_delete["tags"][0]["tag_id"] == tag_b_id


def test_run_tags_api_validation_and_run_scope(tmp_path: Path):
    """Tag API should enforce payload validation and keep tags scoped to run."""
    runs_dir = tmp_path / "runs"
    (runs_dir / "run_a").mkdir(parents=True, exist_ok=True)
    (runs_dir / "run_b").mkdir(parents=True, exist_ok=True)

    app = create_app(runs_dir)
    create_endpoint = _get_route_endpoint(app, "/api/runs/{run_name}/tags", "POST")
    update_endpoint = _get_route_endpoint(app, "/api/runs/{run_name}/tags/{tag_id}", "PATCH")
    delete_endpoint = _get_route_endpoint(app, "/api/runs/{run_name}/tags/{tag_id}", "DELETE")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            create_endpoint(
                run_name="run_a",
                body=CreateTagBody(label=" ", category="general"),
            )
        )
    assert exc_info.value.status_code == 400

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            create_endpoint(
                run_name="run_a",
                body=CreateTagBody(label="goal_mouth", confidence=1.2),
            )
        )
    assert exc_info.value.status_code == 400

    created = asyncio.run(
        create_endpoint(
            run_name="run_a",
            body=CreateTagBody(label="good_tag", category="event"),
        )
    )
    tag_id = int(created["tag"]["tag_id"])

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            update_endpoint(
                run_name="run_b",
                tag_id=tag_id,
                body=UpdateTagBody(notes="should fail across runs"),
            )
        )
    assert exc_info.value.status_code == 404

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            update_endpoint(
                run_name="run_a",
                tag_id=tag_id,
                body=UpdateTagBody(category=None),
            )
        )
    assert exc_info.value.status_code == 400

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(delete_endpoint(run_name="run_b", tag_id=tag_id))
    assert exc_info.value.status_code == 404
