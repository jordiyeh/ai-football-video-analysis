"""Unit tests for UI server player reel endpoints."""

import asyncio
import json
import threading
import time
import zipfile
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.responses import FileResponse
import pandas as pd
import pytest

from src.identity import PlayerDatabase
from src.ui.server import (
    ApplyIdentitySuggestionsAndRecomputeBody,
    ApplyIdentitySuggestionsBody,
    ApprovePlayerReelsPreviewBody,
    AssignTrackBody,
    BulkAssignAppearancesBody,
    CreatePlayerBody,
    CreateTagBody,
    CreateTeamBody,
    UpdateTeamBody,
    SetRunTeamsBody,
    UpdateTagBody,
    RemapRunTeamsBody,
    ExportCrossMatchPackageBody,
    ExportPlayerReelsPackageBody,
    QueuePipelineJobsBody,
    RecomputePlayerReelsBody,
    create_app,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _get_route_endpoint(app, path: str, method: str = "GET"):
    """Find a route endpoint callable by path and HTTP method."""
    for route in app.routes:
        if getattr(route, "path", None) != path:
            continue
        methods = getattr(route, "methods", set())
        if method.upper() in methods:
            return route.endpoint
    raise AssertionError(f"Route not found: {method} {path}")


def _wait_for_job_status(list_jobs_endpoint, job_id: str, expected_status: str, timeout: float = 6.0):
    """Poll jobs endpoint until one job reaches expected status."""
    deadline = time.time() + timeout
    last_payload = {}
    while time.time() < deadline:
        payload = asyncio.run(list_jobs_endpoint(limit=200, include_logs=False))
        last_payload = payload
        for job in payload.get("jobs", []):
            if job.get("job_id") == job_id and job.get("status") == expected_status:
                return job
        time.sleep(0.05)
    raise AssertionError(
        f"Timed out waiting for {job_id} to reach status={expected_status}. Last payload: {last_payload}"
    )


def test_list_runs_includes_player_reel_summary(tmp_path: Path):
    """Runs API should expose player reel availability and summary metrics."""
    runs_dir = tmp_path / "runs"
    run_path = runs_dir / "match_run"
    run_path.mkdir(parents=True)

    _write_json(
        run_path / "player_highlights.json",
        {
            "schema_version": "1.0",
            "video_id": "match",
            "players": [
                {
                    "player_id": 1,
                    "player_name": "Nick",
                    "segment_count": 1,
                    "segments": [
                        {
                            "segment_id": "highlight_001",
                            "start_time": 10.0,
                            "end_time": 18.0,
                            "player_segment_score": 0.88,
                        }
                    ],
                }
            ],
            "summary": {
                "players_with_reels": 1,
                "player_segments_total": 1,
            },
        },
    )

    app = create_app(runs_dir)
    list_runs = _get_route_endpoint(app, "/api/runs")
    payload = asyncio.run(list_runs())
    runs = payload["runs"]
    assert len(runs) == 1
    assert runs[0]["has_player_reels"] is True
    assert runs[0]["player_reel_summary"]["players_with_reels"] == 1
    assert runs[0]["player_reel_summary"]["player_segments_total"] == 1


def test_list_runs_includes_cross_match_summary(tmp_path: Path):
    """Runs API should expose cross-match availability and summary metrics."""
    runs_dir = tmp_path / "runs"
    run_path = runs_dir / "match_run"
    run_path.mkdir(parents=True)

    _write_json(
        run_path / "cross_match_report.json",
        {
            "schema_version": "1.0",
            "summary": {
                "matches_analyzed": 12,
                "unique_players": 24,
            },
        },
    )

    app = create_app(runs_dir)
    list_runs = _get_route_endpoint(app, "/api/runs")
    payload = asyncio.run(list_runs())
    runs = payload["runs"]
    assert len(runs) == 1
    assert runs[0]["has_cross_match_report"] is True
    assert runs[0]["cross_match_summary"]["matches_analyzed"] == 12
    assert runs[0]["cross_match_summary"]["unique_players"] == 24


def test_player_reels_endpoint_enriches_player_metadata(tmp_path: Path):
    """Player reels endpoint should fill missing name/number/team from identity DB."""
    runs_dir = tmp_path / "runs"
    run_path = runs_dir / "match_run"
    run_path.mkdir(parents=True)

    with PlayerDatabase(runs_dir / "players.db") as db:
        player = db.create_player(name="Nicholas Oestringer", jersey_number=10, team_hint="ours")

    _write_json(
        run_path / "player_highlights.json",
        {
            "schema_version": "1.0",
            "video_id": "match",
            "players": [
                {
                    "player_id": player.player_id,
                    "player_name": None,
                    "segments": [
                        {
                            "segment_id": "highlight_003",
                            "start_time": 80.0,
                            "end_time": 92.0,
                            "player_segment_score": 0.91,
                        }
                    ],
                    "segment_count": 1,
                }
            ],
            "summary": {
                "players_with_reels": 1,
                "player_segments_total": 1,
            },
        },
    )

    app = create_app(runs_dir)
    get_player_reels = _get_route_endpoint(app, "/api/runs/{run_name}/player_reels")
    payload = asyncio.run(get_player_reels(run_name="match_run"))
    assert payload["count"] == 1
    reel_player = payload["players"][0]
    assert reel_player["player_name"] == "Nicholas Oestringer"
    assert reel_player["jersey_number"] == 10
    assert reel_player["team_hint"] == "ours"
    assert reel_player["segments"][0]["has_clip"] is False

    get_player_highlights = _get_route_endpoint(app, "/api/runs/{run_name}/player_highlights")
    alias_payload = asyncio.run(get_player_highlights(run_name="match_run"))
    assert alias_payload["players"][0]["player_name"] == "Nicholas Oestringer"

    get_player_reel = _get_route_endpoint(app, "/api/runs/{run_name}/player_reels/{player_id}")
    single_payload = asyncio.run(
        get_player_reel(run_name="match_run", player_id=player.player_id)
    )
    assert single_payload["player_id"] == player.player_id

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(get_player_reel(run_name="match_run", player_id=999))
    assert exc_info.value.status_code == 404


def test_player_reel_clip_endpoint_streams_segment_clip(tmp_path: Path):
    """Clip endpoint should stream per-segment player clip when clip_path exists."""
    runs_dir = tmp_path / "runs"
    run_path = runs_dir / "match_run"
    clip_path = run_path / "player_clips" / "player_7" / "highlight_001_10.0_18.0.mp4"
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    clip_path.write_bytes(b"fake-mp4")

    _write_json(
        run_path / "player_highlights.json",
        {
            "schema_version": "1.0",
            "video_id": "match",
            "players": [
                {
                    "player_id": 7,
                    "player_name": "Alex",
                    "segment_count": 1,
                    "segments": [
                        {
                            "segment_id": "highlight_001",
                            "start_time": 10.0,
                            "end_time": 18.0,
                            "player_segment_score": 0.86,
                            "clip_path": str(clip_path),
                        }
                    ],
                }
            ],
            "summary": {
                "players_with_reels": 1,
                "player_segments_total": 1,
            },
        },
    )

    app = create_app(runs_dir)
    get_player_reels = _get_route_endpoint(app, "/api/runs/{run_name}/player_reels")
    reels_payload = asyncio.run(get_player_reels(run_name="match_run"))
    assert reels_payload["players"][0]["segments"][0]["has_clip"] is True

    get_clip = _get_route_endpoint(
        app,
        "/api/runs/{run_name}/player_reels/{player_id}/segments/{segment_id}/clip",
    )
    clip_response = asyncio.run(
        get_clip(run_name="match_run", player_id=7, segment_id="highlight_001")
    )
    assert isinstance(clip_response, FileResponse)
    assert Path(clip_response.path) == clip_path

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(get_clip(run_name="match_run", player_id=7, segment_id="missing"))
    assert exc_info.value.status_code == 404


def test_player_reels_export_package_endpoint(tmp_path: Path):
    """Export endpoint should build filtered ZIP package and include clips when requested."""
    runs_dir = tmp_path / "runs"
    run_path = runs_dir / "match_run"
    run_path.mkdir(parents=True)

    clip_a = run_path / "player_clips" / "player_7" / "highlight_001.mp4"
    clip_b = run_path / "player_clips" / "player_8" / "highlight_002.mp4"
    clip_a.parent.mkdir(parents=True, exist_ok=True)
    clip_b.parent.mkdir(parents=True, exist_ok=True)
    clip_a.write_bytes(b"clip-a")
    clip_b.write_bytes(b"clip-b")

    _write_json(
        run_path / "player_highlights.json",
        {
            "schema_version": "1.0",
            "video_id": "match",
            "players": [
                {
                    "player_id": 7,
                    "player_name": "Alex",
                    "team_hint": "ours",
                    "segment_count": 2,
                    "segments": [
                        {
                            "segment_id": "highlight_001",
                            "start_time": 10.0,
                            "end_time": 18.0,
                            "player_segment_score": 0.92,
                            "clip_path": str(clip_a),
                        },
                        {
                            "segment_id": "highlight_010",
                            "start_time": 70.0,
                            "end_time": 82.0,
                            "player_segment_score": 0.55,
                        },
                    ],
                },
                {
                    "player_id": 8,
                    "player_name": "Blake",
                    "team_hint": "opponent",
                    "segment_count": 1,
                    "segments": [
                        {
                            "segment_id": "highlight_002",
                            "start_time": 25.0,
                            "end_time": 35.0,
                            "player_segment_score": 0.88,
                            "clip_path": str(clip_b),
                        }
                    ],
                },
            ],
            "summary": {
                "players_with_reels": 2,
                "player_segments_total": 3,
            },
        },
    )

    app = create_app(runs_dir)
    export_package = _get_route_endpoint(
        app, "/api/runs/{run_name}/player_reels/actions/export_package", method="POST"
    )
    download_export = _get_route_endpoint(
        app, "/api/runs/{run_name}/player_reels/exports/{export_name}"
    )

    # Filter to our team, keep only top-1 >= 0.6, skip clips.
    export_payload = asyncio.run(
        export_package(
            run_name="match_run",
            body=ExportPlayerReelsPackageBody(
                team_filter="ours",
                min_score=0.6,
                top_n=1,
                sort_by="best_score_desc",
                include_clips=False,
                player_ids=[7],
            ),
        )
    )
    assert export_payload["success"] is True
    assert export_payload["summary"]["players_with_reels"] == 1
    assert export_payload["summary"]["player_segments_total"] == 1
    assert export_payload["summary"]["clip_files_included"] == 0

    export_path = Path(export_payload["export_path"])
    assert export_path.exists()
    with zipfile.ZipFile(export_path, "r") as zf:
        names = set(zf.namelist())
        assert "player_reels/player_highlights_filtered.json" in names
        assert "player_reels/player_highlights_filtered.csv" in names
        assert "player_reels/export_manifest.json" in names
        assert not any(name.endswith(".mp4") for name in names)

        filtered = json.loads(zf.read("player_reels/player_highlights_filtered.json").decode("utf-8"))
        assert filtered["count"] == 1
        assert filtered["players"][0]["player_id"] == 7
        assert filtered["players"][0]["segment_count"] == 1

    # Include clips this time and verify downloadable endpoint points to the zip.
    export_with_clips = asyncio.run(
        export_package(
            run_name="match_run",
            body=ExportPlayerReelsPackageBody(
                team_filter="all",
                min_score=0.0,
                top_n=2,
                sort_by="best_score_desc",
                include_clips=True,
            ),
        )
    )
    assert export_with_clips["summary"]["clip_files_included"] >= 2

    with zipfile.ZipFile(Path(export_with_clips["export_path"]), "r") as zf:
        assert any(name.endswith(".mp4") for name in zf.namelist())

    download_response = asyncio.run(
        download_export(
            run_name="match_run",
            export_name=export_with_clips["export_name"],
        )
    )
    assert isinstance(download_response, FileResponse)
    assert Path(download_response.path) == Path(export_with_clips["export_path"])


def test_cross_match_report_endpoints(tmp_path: Path):
    """Cross-match endpoints should expose report payload and downloadable artifacts."""
    runs_dir = tmp_path / "runs"
    run_path = runs_dir / "match_run"
    run_path.mkdir(parents=True)

    _write_json(
        run_path / "cross_match_report.json",
        {
            "schema_version": "1.0",
            "generated_at": "2026-02-07T12:00:00Z",
            "summary": {
                "matches_analyzed": 8,
                "unique_players": 19,
            },
            "season_trends": {
                "match_aggregates": {"goals_per_match": 1.75},
                "team_trends": {"ours": {"avg_possession_share": 0.56}},
                "window": {"last_n": 3, "goals": [1, 2, 0], "shots": [6, 9, 5], "highlights": [4, 6, 3]},
            },
            "players": {
                "top_players": [
                    {"player_id": 10, "player_name": "Nick", "total_segments": 11}
                ]
            },
        },
    )
    (run_path / "cross_match_match_trends.csv").write_text("run_name,goals\nmatch1,2\n")
    (run_path / "cross_match_player_trends.csv").write_text("player_id,total_segments\n10,11\n")
    (run_path / "coach_report_template.md").write_text("# Coach Template\n")
    (run_path / "player_report_templates.md").write_text("# Player Templates\n")

    app = create_app(runs_dir)
    get_cross_match = _get_route_endpoint(app, "/api/runs/{run_name}/cross_match")
    download_artifact = _get_route_endpoint(
        app, "/api/runs/{run_name}/cross_match/artifacts/{artifact_id}"
    )
    export_package = _get_route_endpoint(
        app, "/api/runs/{run_name}/cross_match/actions/export_package", method="POST"
    )
    download_export = _get_route_endpoint(
        app, "/api/runs/{run_name}/cross_match/exports/{export_name}"
    )

    payload = asyncio.run(get_cross_match(run_name="match_run"))
    assert payload["summary"]["matches_analyzed"] == 8
    assert "report_json" in payload["available_artifacts"]
    assert payload["available_artifacts"]["report_json"]["file_name"] == "cross_match_report.json"

    artifact_response = asyncio.run(
        download_artifact(run_name="match_run", artifact_id="coach_template_md")
    )
    assert isinstance(artifact_response, FileResponse)
    assert Path(artifact_response.path) == (run_path / "coach_report_template.md")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(download_artifact(run_name="match_run", artifact_id="invalid"))
    assert exc_info.value.status_code == 400

    export_payload = asyncio.run(
        export_package(
            run_name="match_run",
            body=ExportCrossMatchPackageBody(include_templates=False),
        )
    )
    assert export_payload["success"] is True
    assert export_payload["summary"]["matches_analyzed"] == 8
    assert export_payload["summary"]["artifact_files"] == 3

    export_path = Path(export_payload["export_path"])
    assert export_path.exists()
    with zipfile.ZipFile(export_path, "r") as zf:
        names = set(zf.namelist())
        assert "cross_match/cross_match_report.json" in names
        assert "cross_match/cross_match_match_trends.csv" in names
        assert "cross_match/cross_match_player_trends.csv" in names
        assert "cross_match/coach_report_template.md" not in names
        assert "cross_match/player_report_templates.md" not in names
        assert "cross_match/export_manifest.json" in names

    download_response = asyncio.run(
        download_export(
            run_name="match_run",
            export_name=export_payload["export_name"],
        )
    )
    assert isinstance(download_response, FileResponse)
    assert Path(download_response.path) == export_path


def test_identity_review_and_create_player_endpoint(tmp_path: Path):
    """Identity review endpoint should expose players + assignments and allow player creation."""
    runs_dir = tmp_path / "runs"
    run_path = runs_dir / "match_run"
    run_path.mkdir(parents=True)

    with PlayerDatabase(runs_dir / "players.db") as db:
        existing = db.create_player(name="Sam", jersey_number=9, team_hint="ours")
        db.create_appearance(
            video_id="match",
            run_name="match_run",
            track_id=42,
            player_id=existing.player_id,
            match_confidence=0.93,
            match_method="auto",
            frame_start=100,
            frame_end=200,
        )

    _write_json(
        run_path / "player_assignments.json",
        {
            "schema_version": "1.2",
            "video_id": "match",
            "assignments": [
                {
                    "track_id": 42,
                    "player_id": existing.player_id,
                    "confidence": 0.93,
                    "match_method": "auto",
                    "lock_state": "locked",
                    "lock_reason": "initial_lock",
                    "lock_conflict_with_track_id": None,
                    "fusion": {
                        "strategy": "agreement_boost",
                        "multimodal": {
                            "face": {
                                "player_id": existing.player_id,
                                "confidence": 0.91,
                                "support_frames": 3,
                                "backend": "facenet512",
                            },
                            "jersey_ocr": {
                                "jersey_number": 9,
                                "player_id": existing.player_id,
                                "confidence": 0.78,
                                "support_frames": 2,
                                "ambiguous": False,
                            },
                            "applied": ["face_agreement_boost", "jersey_agreement_boost"],
                        },
                    },
                }
            ],
        },
    )

    app = create_app(runs_dir)

    get_identity_review = _get_route_endpoint(app, "/api/runs/{run_name}/identity_review")
    review_payload = asyncio.run(get_identity_review(run_name="match_run"))
    assert review_payload["video_id"] == "match"
    assert review_payload["summary"]["assigned"] == 1
    assert review_payload["summary"]["locked"] == 1
    assert review_payload["assignments"][0]["track_id"] == 42
    assert review_payload["assignments"][0]["player_name"] == "Sam"
    assert review_payload["assignments"][0]["lock_state"] == "locked"
    assert review_payload["assignments"][0]["lock_reason"] == "initial_lock"
    assert review_payload["assignments"][0]["fusion_strategy"] == "agreement_boost"
    assert review_payload["assignments"][0]["face_backend"] == "facenet512"
    assert review_payload["assignments"][0]["jersey_number_detected"] == 9

    create_player = _get_route_endpoint(app, "/api/players", method="POST")
    created_payload = asyncio.run(
        create_player(
            body=CreatePlayerBody(name="Alex", jersey_number=18, team_hint="opponent")
        )
    )
    assert created_payload["success"] is True
    assert created_payload["player"]["name"] == "Alex"


def test_recompute_player_reels_endpoint(tmp_path: Path):
    """Recompute endpoint should regenerate player_highlights artifacts from DB assignments."""
    runs_dir = tmp_path / "runs"
    run_path = runs_dir / "match_run"
    run_path.mkdir(parents=True)

    _write_json(
        run_path / "highlights.json",
        {
            "schema_version": "1.0",
            "video_id": "match",
            "segments": [
                {
                    "segment_id": "highlight_001",
                    "start_time": 10.0,
                    "end_time": 20.0,
                    "duration": 10.0,
                    "score": 0.9,
                    "reasons": ["goal"],
                    "sources": ["event"],
                }
            ],
            "summary": {"segments_selected": 1},
        },
    )
    _write_json(
        run_path / "video_metadata.json",
        {"fps": 30.0},
    )
    _write_json(
        run_path / "player_assignments.json",
        {
            "schema_version": "1.1",
            "video_id": "match",
            "assignments": [
                {"track_id": 10, "player_id": 1, "confidence": 0.95, "match_method": "auto"}
            ],
        },
    )

    tracks = pd.DataFrame(
        [
            {
                "frame_idx": frame_idx,
                "timestamp": frame_idx / 30.0,
                "object_type": "player",
                "track_id": 10,
            }
            for frame_idx in range(300, 451)
        ]
    )
    tracks.to_parquet(run_path / "tracks.parquet", index=False)

    with PlayerDatabase(runs_dir / "players.db") as db:
        player = db.create_player(name="Nick", jersey_number=10, team_hint="ours")
        db.create_appearance(
            video_id="match",
            run_name="match_run",
            track_id=10,
            player_id=player.player_id,
            match_confidence=0.95,
            match_method="manual",
            frame_start=300,
            frame_end=450,
        )

    app = create_app(runs_dir)
    recompute = _get_route_endpoint(
        app, "/api/runs/{run_name}/player_reels/actions/recompute", method="POST"
    )
    payload = asyncio.run(
        recompute(
            run_name="match_run",
            body=RecomputePlayerReelsBody(preserve_existing_clips=True),
        )
    )
    assert payload["success"] is True
    assert payload["summary"]["players_with_reels"] == 1
    assert (run_path / "player_highlights.json").exists()
    assert (run_path / "player_highlights.csv").exists()
    assert (run_path / "player_highlights_manifest.json").exists()


def test_bulk_assign_endpoint_updates_multiple_tracks(tmp_path: Path):
    """Bulk assign endpoint should reassign tracks and support bulk unassign."""
    runs_dir = tmp_path / "runs"
    run_path = runs_dir / "match_run"
    run_path.mkdir(parents=True)

    with PlayerDatabase(runs_dir / "players.db") as db:
        target = db.create_player(name="Target", jersey_number=7, team_hint="ours")
        db.create_appearance(
            video_id="match",
            run_name="match_run",
            track_id=10,
            player_id=None,
            match_confidence=0.2,
            match_method="suggested",
            frame_start=100,
            frame_end=140,
        )
        db.create_appearance(
            video_id="match",
            run_name="match_run",
            track_id=11,
            player_id=None,
            match_confidence=0.3,
            match_method="suggested",
            frame_start=150,
            frame_end=190,
        )

    _write_json(
        run_path / "player_assignments.json",
        {
            "schema_version": "1.1",
            "video_id": "match",
            "assignments": [
                {"track_id": 10, "player_id": None, "confidence": 0.2, "match_method": "suggested"},
                {"track_id": 11, "player_id": None, "confidence": 0.3, "match_method": "suggested"},
            ],
        },
    )

    app = create_app(runs_dir)
    bulk_assign = _get_route_endpoint(
        app, "/api/runs/{run_name}/identity_review/actions/bulk_assign", method="POST"
    )
    payload = asyncio.run(
        bulk_assign(
            run_name="match_run",
            body=BulkAssignAppearancesBody(
                track_ids=[10, 11, 999],
                player_id=target.player_id,
                confidence=1.0,
                method="manual",
            ),
        )
    )

    assert payload["success"] is True
    assert payload["updated_count"] == 2
    assert payload["missing_count"] == 1
    assert payload["missing_track_ids"] == [999]
    assert payload["operation_id"]

    with PlayerDatabase(runs_dir / "players.db") as db:
        updated_10 = db.get_appearance("match", 10)
        updated_11 = db.get_appearance("match", 11)
        assert updated_10 is not None and updated_10.player_id == target.player_id
        assert updated_11 is not None and updated_11.player_id == target.player_id

    # Bulk unassign both tracks.
    unassign_payload = asyncio.run(
        bulk_assign(
            run_name="match_run",
            body=BulkAssignAppearancesBody(
                track_ids=[10, 11],
                player_id=None,
                confidence=1.0,
                method="manual",
            ),
        )
    )
    assert unassign_payload["success"] is True
    assert unassign_payload["updated_count"] == 2
    assert unassign_payload["missing_count"] == 0
    assert unassign_payload["operation_id"]

    with PlayerDatabase(runs_dir / "players.db") as db:
        cleared_10 = db.get_appearance("match", 10)
        cleared_11 = db.get_appearance("match", 11)
        assert cleared_10 is not None and cleared_10.player_id is None
        assert cleared_11 is not None and cleared_11.player_id is None


def test_identity_suggestions_endpoints(tmp_path: Path):
    """Identity suggestions should be generated and apply selected pending suggestions."""
    runs_dir = tmp_path / "runs"
    run_path = runs_dir / "match_run"
    run_path.mkdir(parents=True)

    with PlayerDatabase(runs_dir / "players.db") as db:
        player_a = db.create_player(name="Alice", jersey_number=8, team_hint="ours")
        player_b = db.create_player(name="Bob", jersey_number=5, team_hint="opponent")
        db.create_appearance(
            video_id="match",
            run_name="match_run",
            track_id=10,
            player_id=None,
            match_confidence=0.2,
            match_method="suggested",
            frame_start=100,
            frame_end=140,
        )
        db.create_appearance(
            video_id="match",
            run_name="match_run",
            track_id=11,
            player_id=None,
            match_confidence=0.2,
            match_method="suggested",
            frame_start=150,
            frame_end=190,
        )

    _write_json(
        run_path / "player_assignments.json",
        {
            "schema_version": "1.1",
            "video_id": "match",
            "assignments": [
                {
                    "track_id": 10,
                    "player_id": player_a.player_id,
                    "player_name": "Alice",
                    "match_method": "suggested",
                    "confidence": 0.79,
                    "fusion": {
                        "strategy": "agreement_boost",
                        "body_match": {
                            "player_id": player_a.player_id,
                            "confidence": 0.74,
                            "method": "suggested",
                        },
                        "profile_match": {
                            "profile_id": "8_Alice",
                            "player_id": player_a.player_id,
                            "confidence": 0.81,
                        },
                    },
                },
                {
                    "track_id": 11,
                    "player_id": player_b.player_id,
                    "player_name": "Bob",
                    "match_method": "auto",
                    "confidence": 0.93,
                    "fusion": {
                        "strategy": "body_only",
                        "body_match": {
                            "player_id": player_b.player_id,
                            "confidence": 0.93,
                            "method": "auto",
                        },
                        "profile_match": None,
                    },
                },
            ],
        },
    )

    app = create_app(runs_dir)
    get_suggestions = _get_route_endpoint(app, "/api/runs/{run_name}/identity_suggestions")
    apply_suggestions = _get_route_endpoint(
        app, "/api/runs/{run_name}/identity_suggestions/actions/apply", method="POST"
    )

    suggestions_payload = asyncio.run(get_suggestions(run_name="match_run", refresh=True))
    assert suggestions_payload["count"] == 2
    assert suggestions_payload["summary"]["pending"] == 1
    first = suggestions_payload["suggestions"][0]
    assert first["track_id"] == 10
    assert first["needs_review"] is True
    assert first["recommended"]["player_name"] == "Alice"
    assert first["candidates"][0]["player_id"] == player_a.player_id

    apply_payload = asyncio.run(
        apply_suggestions(
            run_name="match_run",
            body=ApplyIdentitySuggestionsBody(
                track_ids=[10, 11],
                min_confidence=0.75,
                suggested_only=True,
            ),
        )
    )
    assert apply_payload["success"] is True
    assert apply_payload["applied_count"] == 1
    assert apply_payload["failed_count"] == 0

    with PlayerDatabase(runs_dir / "players.db") as db:
        assigned = db.get_appearance("match", 10)
        untouched = db.get_appearance("match", 11)
        assert assigned is not None and assigned.player_id == player_a.player_id
        assert assigned.match_method == "suggested"
        assert untouched is not None and untouched.player_id is None

    with open(run_path / "profile_match_suggestions.json") as f:
        updated_suggestions = json.load(f)
    updated_track_10 = next(row for row in updated_suggestions["suggestions"] if row["track_id"] == 10)
    assert updated_track_10["status"] == "applied"
    assert updated_track_10.get("applied_operation_id")


def test_apply_and_recompute_identity_suggestions_endpoint(tmp_path: Path):
    """Combined endpoint should apply selected suggestions then recompute reel artifacts."""
    runs_dir = tmp_path / "runs"
    run_path = runs_dir / "match_run"
    run_path.mkdir(parents=True)

    with PlayerDatabase(runs_dir / "players.db") as db:
        player = db.create_player(name="Alice", jersey_number=8, team_hint="ours")
        db.create_appearance(
            video_id="match",
            run_name="match_run",
            track_id=10,
            player_id=None,
            match_confidence=0.2,
            match_method="suggested",
            frame_start=300,
            frame_end=450,
        )

    _write_json(
        run_path / "highlights.json",
        {
            "schema_version": "1.0",
            "video_id": "match",
            "segments": [
                {
                    "segment_id": "highlight_001",
                    "start_time": 10.0,
                    "end_time": 20.0,
                    "duration": 10.0,
                    "score": 0.9,
                    "reasons": ["goal"],
                    "sources": ["event"],
                }
            ],
            "summary": {"segments_selected": 1},
        },
    )
    _write_json(run_path / "video_metadata.json", {"fps": 30.0})
    _write_json(
        run_path / "player_assignments.json",
        {
            "schema_version": "1.1",
            "video_id": "match",
            "assignments": [
                {
                    "track_id": 10,
                    "player_id": player.player_id,
                    "player_name": "Alice",
                    "match_method": "suggested",
                    "confidence": 0.82,
                    "fusion": {
                        "strategy": "agreement_boost",
                        "body_match": {
                            "player_id": player.player_id,
                            "confidence": 0.78,
                            "method": "suggested",
                        },
                        "profile_match": {
                            "profile_id": "8_Alice",
                            "player_id": player.player_id,
                            "confidence": 0.8,
                        },
                    },
                }
            ],
        },
    )

    tracks = pd.DataFrame(
        [
            {
                "frame_idx": frame_idx,
                "timestamp": frame_idx / 30.0,
                "object_type": "player",
                "track_id": 10,
            }
            for frame_idx in range(300, 451)
        ]
    )
    tracks.to_parquet(run_path / "tracks.parquet", index=False)

    app = create_app(runs_dir)
    apply_and_recompute = _get_route_endpoint(
        app,
        "/api/runs/{run_name}/identity_suggestions/actions/apply_and_recompute",
        method="POST",
    )

    payload = asyncio.run(
        apply_and_recompute(
            run_name="match_run",
            body=ApplyIdentitySuggestionsAndRecomputeBody(
                track_ids=[10],
                min_confidence=0.75,
                suggested_only=True,
                preserve_existing_clips=True,
            ),
        )
    )

    assert payload["success"] is True
    assert payload["apply"]["applied_count"] == 1
    assert payload["apply"]["failed_count"] == 0
    assert payload["recompute"]["summary"]["players_with_reels"] == 1
    assert payload["recompute"]["summary"]["player_segments_total"] >= 1
    assert payload["recompute"]["inputs"]["source"] == "ui_apply_suggestions_recompute"
    assert (run_path / "player_highlights.json").exists()
    assert (run_path / "player_highlights.csv").exists()
    assert (run_path / "player_highlights_manifest.json").exists()

    with PlayerDatabase(runs_dir / "players.db") as db:
        assigned = db.get_appearance("match", 10)
        assert assigned is not None
        assert assigned.player_id == player.player_id


def test_apply_and_preview_identity_suggestions_endpoint(tmp_path: Path):
    """Combined preview endpoint should apply selected suggestions and return a recompute diff only."""
    runs_dir = tmp_path / "runs"
    run_path = runs_dir / "match_run"
    run_path.mkdir(parents=True)

    with PlayerDatabase(runs_dir / "players.db") as db:
        player = db.create_player(name="Alice", jersey_number=8, team_hint="ours")
        db.create_appearance(
            video_id="match",
            run_name="match_run",
            track_id=10,
            player_id=None,
            match_confidence=0.2,
            match_method="suggested",
            frame_start=300,
            frame_end=450,
        )

    _write_json(
        run_path / "highlights.json",
        {
            "schema_version": "1.0",
            "video_id": "match",
            "segments": [
                {
                    "segment_id": "highlight_001",
                    "start_time": 10.0,
                    "end_time": 20.0,
                    "duration": 10.0,
                    "score": 0.9,
                    "reasons": ["goal"],
                    "sources": ["event"],
                }
            ],
            "summary": {"segments_selected": 1},
        },
    )
    _write_json(run_path / "video_metadata.json", {"fps": 30.0})
    _write_json(
        run_path / "player_assignments.json",
        {
            "schema_version": "1.1",
            "video_id": "match",
            "assignments": [
                {
                    "track_id": 10,
                    "player_id": player.player_id,
                    "player_name": "Alice",
                    "match_method": "suggested",
                    "confidence": 0.82,
                    "fusion": {
                        "strategy": "agreement_boost",
                        "body_match": {
                            "player_id": player.player_id,
                            "confidence": 0.78,
                            "method": "suggested",
                        },
                        "profile_match": {
                            "profile_id": "8_Alice",
                            "player_id": player.player_id,
                            "confidence": 0.8,
                        },
                    },
                }
            ],
        },
    )

    tracks = pd.DataFrame(
        [
            {
                "frame_idx": frame_idx,
                "timestamp": frame_idx / 30.0,
                "object_type": "player",
                "track_id": 10,
            }
            for frame_idx in range(300, 451)
        ]
    )
    tracks.to_parquet(run_path / "tracks.parquet", index=False)

    app = create_app(runs_dir)
    apply_and_preview = _get_route_endpoint(
        app,
        "/api/runs/{run_name}/identity_suggestions/actions/apply_and_preview",
        method="POST",
    )

    payload = asyncio.run(
        apply_and_preview(
            run_name="match_run",
            body=ApplyIdentitySuggestionsAndRecomputeBody(
                track_ids=[10],
                min_confidence=0.75,
                suggested_only=True,
                preserve_existing_clips=False,
            ),
        )
    )

    assert payload["success"] is True
    assert payload["apply"]["applied_count"] == 1
    assert payload["apply"]["failed_count"] == 0
    assert payload["preview"]["summary"]["players_with_reels"] == 1
    assert payload["preview"]["diff"]["delta"]["player_segments_total"] > 0
    assert payload["preview"]["inputs"]["preserve_existing_clips"] is False

    # Preview flow should not write reel artifacts.
    assert not (run_path / "player_highlights.json").exists()
    assert not (run_path / "player_highlights.csv").exists()
    assert not (run_path / "player_highlights_manifest.json").exists()

    with PlayerDatabase(runs_dir / "players.db") as db:
        assigned = db.get_appearance("match", 10)
        assert assigned is not None
        assert assigned.player_id == player.player_id


def test_assign_and_undo_identity_edits(tmp_path: Path):
    """Single assign endpoint should audit edits and undo should restore prior state."""
    runs_dir = tmp_path / "runs"
    run_path = runs_dir / "match_run"
    run_path.mkdir(parents=True)

    with PlayerDatabase(runs_dir / "players.db") as db:
        target = db.create_player(name="Target", jersey_number=7, team_hint="ours")

    _write_json(
        run_path / "player_assignments.json",
        {
            "schema_version": "1.1",
            "video_id": "match",
            "assignments": [
                {"track_id": 11, "player_id": None, "confidence": 0.2, "match_method": "suggested"},
            ],
        },
    )

    app = create_app(runs_dir)
    assign = _get_route_endpoint(
        app, "/api/runs/{run_name}/identity_review/actions/assign", method="POST"
    )
    undo = _get_route_endpoint(
        app, "/api/runs/{run_name}/identity_review/actions/undo", method="POST"
    )
    undo_by_op = _get_route_endpoint(
        app, "/api/runs/{run_name}/identity_review/actions/undo/{op_id}", method="POST"
    )
    get_edits = _get_route_endpoint(
        app, "/api/runs/{run_name}/identity_review/edits"
    )

    # Assign missing appearance (track 11) to exercise undo delete (previous.exists = False).
    assign_created_payload = asyncio.run(
        assign(
            run_name="match_run",
            body=AssignTrackBody(
                track_id=11,
                player_id=target.player_id,
                confidence=1.0,
                method="manual",
            ),
        )
    )
    assert assign_created_payload["success"] is True
    assert assign_created_payload["updated_count"] == 0
    assert assign_created_payload["created_count"] == 1
    created_op_id = assign_created_payload["operation_id"]
    assert created_op_id

    with PlayerDatabase(runs_dir / "players.db") as db:
        assigned_11 = db.get_appearance("match", 11)
        assert assigned_11 is not None and assigned_11.player_id == target.player_id

    edits_payload = asyncio.run(get_edits(run_name="match_run"))
    assert edits_payload["count"] >= 1
    assert edits_payload["edits"][0]["op_id"] == created_op_id
    assert edits_payload["edits"][0]["undoable"] is True
    assert edits_payload["edits"][0]["action"] == "assign"

    # Undo newest op (track 11 create), should remove appearance row.
    undo_created_payload = asyncio.run(undo(run_name="match_run"))
    assert undo_created_payload["success"] is True
    assert undo_created_payload["deleted_count"] == 1
    assert undo_created_payload["reverted_count"] == 0

    with PlayerDatabase(runs_dir / "players.db") as db:
        after_undo_11 = db.get_appearance("match", 11)
        assert after_undo_11 is None

    edits_after_first_undo = asyncio.run(get_edits(run_name="match_run"))
    matching_created_rows = [
        row for row in edits_after_first_undo["edits"]
        if row.get("op_id") == created_op_id and row.get("action") in {"assign", "bulk_assign"}
    ]
    assert matching_created_rows
    assert matching_created_rows[0]["undoable"] is False
    assert any(
        row.get("action") == "undo" and row.get("target_op_id") == created_op_id
        for row in edits_after_first_undo["edits"]
    )

    # Seed an existing appearance and update it to exercise previous.exists = True undo path.
    with PlayerDatabase(runs_dir / "players.db") as db:
        db.create_appearance(
            video_id="match",
            run_name="match_run",
            track_id=10,
            player_id=None,
            match_confidence=0.25,
            match_method="suggested",
            frame_start=100,
            frame_end=140,
        )

    assign_existing_payload = asyncio.run(
        assign(
            run_name="match_run",
            body=AssignTrackBody(
                track_id=10,
                player_id=target.player_id,
                confidence=1.0,
                method="manual",
            ),
        )
    )
    assert assign_existing_payload["success"] is True
    assert assign_existing_payload["updated_count"] == 1
    assert assign_existing_payload["created_count"] == 0
    existing_op_id = assign_existing_payload["operation_id"]
    assert existing_op_id

    undo_existing_payload = asyncio.run(undo_by_op(run_name="match_run", op_id=existing_op_id))
    assert undo_existing_payload["success"] is True
    assert undo_existing_payload["reverted_count"] == 1
    assert undo_existing_payload["deleted_count"] == 0

    with PlayerDatabase(runs_dir / "players.db") as db:
        restored_10 = db.get_appearance("match", 10)
        assert restored_10 is not None
        assert restored_10.player_id is None
        assert restored_10.match_method == "suggested"
        assert restored_10.match_confidence == pytest.approx(0.25)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(undo_by_op(run_name="match_run", op_id=existing_op_id))
    assert exc_info.value.status_code == 404

    # No further edits should be undoable.
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(undo(run_name="match_run"))
    assert exc_info.value.status_code == 404


def test_recompute_preview_diff_endpoint(tmp_path: Path):
    """Preview endpoint should return diff without overwriting persisted player_highlights.json."""
    runs_dir = tmp_path / "runs"
    run_path = runs_dir / "match_run"
    run_path.mkdir(parents=True)

    _write_json(
        run_path / "highlights.json",
        {
            "schema_version": "1.0",
            "video_id": "match",
            "segments": [
                {
                    "segment_id": "highlight_001",
                    "start_time": 10.0,
                    "end_time": 20.0,
                    "duration": 10.0,
                    "score": 0.9,
                    "reasons": ["goal"],
                    "sources": ["event"],
                }
            ],
            "summary": {"segments_selected": 1},
        },
    )
    _write_json(run_path / "video_metadata.json", {"fps": 30.0})
    _write_json(
        run_path / "player_assignments.json",
        {
            "schema_version": "1.1",
            "video_id": "match",
            "assignments": [
                {"track_id": 10, "player_id": 1, "confidence": 0.95, "match_method": "auto"}
            ],
        },
    )

    existing_payload = {
        "schema_version": "1.0",
        "video_id": "match",
        "players": [],
        "summary": {
            "players_with_reels": 0,
            "player_segments_total": 0,
        },
    }
    _write_json(run_path / "player_highlights.json", existing_payload)

    tracks = pd.DataFrame(
        [
            {
                "frame_idx": frame_idx,
                "timestamp": frame_idx / 30.0,
                "object_type": "player",
                "track_id": 10,
            }
            for frame_idx in range(300, 451)
        ]
    )
    tracks.to_parquet(run_path / "tracks.parquet", index=False)

    with PlayerDatabase(runs_dir / "players.db") as db:
        player = db.create_player(name="Nick", jersey_number=10, team_hint="ours")
        db.create_appearance(
            video_id="match",
            run_name="match_run",
            track_id=10,
            player_id=player.player_id,
            match_confidence=0.95,
            match_method="manual",
            frame_start=300,
            frame_end=450,
        )

    app = create_app(runs_dir)
    preview_recompute = _get_route_endpoint(
        app, "/api/runs/{run_name}/player_reels/actions/recompute_preview", method="POST"
    )
    preview_payload = asyncio.run(
        preview_recompute(
            run_name="match_run",
            body=RecomputePlayerReelsBody(preserve_existing_clips=True),
        )
    )
    assert preview_payload["success"] is True
    assert preview_payload.get("preview_id")
    assert preview_payload["summary"]["players_with_reels"] == 1
    assert preview_payload["diff"]["delta"]["player_segments_total"] > 0

    # Preview call should not overwrite existing persisted reels artifact.
    with open(run_path / "player_highlights.json") as f:
        persisted_after_preview = json.load(f)
    assert persisted_after_preview == existing_payload


def test_approve_preview_endpoint_persists_and_rejects_stale_preview(tmp_path: Path):
    """Approve endpoint should persist a stored preview and reject stale preview IDs/signatures."""
    runs_dir = tmp_path / "runs"
    run_path = runs_dir / "match_run"
    run_path.mkdir(parents=True)

    _write_json(
        run_path / "highlights.json",
        {
            "schema_version": "1.0",
            "video_id": "match",
            "segments": [
                {
                    "segment_id": "highlight_001",
                    "start_time": 10.0,
                    "end_time": 20.0,
                    "duration": 10.0,
                    "score": 0.9,
                    "reasons": ["goal"],
                    "sources": ["event"],
                }
            ],
            "summary": {"segments_selected": 1},
        },
    )
    _write_json(run_path / "video_metadata.json", {"fps": 30.0})
    _write_json(
        run_path / "player_assignments.json",
        {
            "schema_version": "1.1",
            "video_id": "match",
            "assignments": [
                {"track_id": 10, "player_id": 1, "confidence": 0.95, "match_method": "auto"}
            ],
        },
    )

    tracks = pd.DataFrame(
        [
            {
                "frame_idx": frame_idx,
                "timestamp": frame_idx / 30.0,
                "object_type": "player",
                "track_id": 10,
            }
            for frame_idx in range(300, 451)
        ]
    )
    tracks.to_parquet(run_path / "tracks.parquet", index=False)

    with PlayerDatabase(runs_dir / "players.db") as db:
        player = db.create_player(name="Nick", jersey_number=10, team_hint="ours")
        db.create_appearance(
            video_id="match",
            run_name="match_run",
            track_id=10,
            player_id=player.player_id,
            match_confidence=0.95,
            match_method="manual",
            frame_start=300,
            frame_end=450,
        )

    app = create_app(runs_dir)
    preview_recompute = _get_route_endpoint(
        app, "/api/runs/{run_name}/player_reels/actions/recompute_preview", method="POST"
    )
    approve_preview = _get_route_endpoint(
        app, "/api/runs/{run_name}/player_reels/actions/approve_preview", method="POST"
    )
    assign_track = _get_route_endpoint(
        app, "/api/runs/{run_name}/identity_review/actions/assign", method="POST"
    )

    preview_payload = asyncio.run(
        preview_recompute(
            run_name="match_run",
            body=RecomputePlayerReelsBody(preserve_existing_clips=False),
        )
    )
    preview_id = preview_payload.get("preview_id")
    assert isinstance(preview_id, str) and preview_id

    approved = asyncio.run(
        approve_preview(
            run_name="match_run",
            body=ApprovePlayerReelsPreviewBody(preview_id=preview_id),
        )
    )
    assert approved["success"] is True
    assert approved.get("preview_id") == preview_id
    assert approved["summary"]["players_with_reels"] == 1
    assert approved["inputs"]["source"] == "ui_approve_preview"
    assert (run_path / "player_highlights.json").exists()
    assert (run_path / "player_highlights.csv").exists()
    assert (run_path / "player_highlights_manifest.json").exists()

    # Generate a new preview, mutate assignments, and ensure stale preview is rejected.
    stale_preview_payload = asyncio.run(
        preview_recompute(
            run_name="match_run",
            body=RecomputePlayerReelsBody(preserve_existing_clips=True),
        )
    )
    stale_preview_id = stale_preview_payload.get("preview_id")
    assert isinstance(stale_preview_id, str) and stale_preview_id

    asyncio.run(
        assign_track(
            run_name="match_run",
            body=AssignTrackBody(
                track_id=10,
                player_id=None,
                confidence=1.0,
                method="manual",
            ),
        )
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            approve_preview(
                run_name="match_run",
                body=ApprovePlayerReelsPreviewBody(preview_id=stale_preview_id),
            )
        )
    assert exc_info.value.status_code == 409


def test_pipeline_configs_endpoint_lists_available_configs(tmp_path: Path):
    """Pipeline config endpoint should expose built-in default plus YAML config files."""
    app = create_app(tmp_path / "runs")
    list_configs = _get_route_endpoint(app, "/api/pipeline/configs")

    payload = asyncio.run(list_configs())
    assert payload["count"] >= 1
    assert payload["configs"][0]["path"] is None
    assert payload["configs"][0]["label"] == "Built-in default"
    assert any(
        isinstance(row.get("path"), str) and str(row.get("path")).endswith(".yaml")
        for row in payload["configs"][1:]
    )


def test_pipeline_job_queue_supports_multiple_videos(tmp_path: Path, monkeypatch):
    """Queue endpoint should accept multiple videos and complete jobs in background."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    video_a = tmp_path / "match_a.mp4"
    video_b = tmp_path / "match_b.mp4"
    video_a.write_bytes(b"fake-mp4-a")
    video_b.write_bytes(b"fake-mp4-b")

    def _fake_pipeline_run(self, video_path, output_dir, resume=False):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        _write_json(
            output_path / "run_manifest.json",
            {
                "schema_version": "1.1",
                "video_path": str(video_path),
                "original_video_path": str(video_path),
                "output_dir": str(output_dir),
            },
        )
        _write_json(
            output_path / "video_metadata.json",
            {
                "fps": 30.0,
                "duration": 120.0,
                "width": 1920,
                "height": 1080,
            },
        )
        (output_path / "events.jsonl").write_text("")
        _write_json(
            output_path / "score_timeline.json",
            {
                "schema_version": "1.0",
                "final_score": {"team_a": 0, "team_b": 0},
                "goals": 0,
            },
        )
        return {"events": [], "tracks": []}

    monkeypatch.setattr("src.pipeline.base.Pipeline.run", _fake_pipeline_run)

    app = create_app(runs_dir)
    queue_jobs = _get_route_endpoint(app, "/api/pipeline/jobs", method="POST")
    list_jobs = _get_route_endpoint(app, "/api/pipeline/jobs")
    get_job = _get_route_endpoint(app, "/api/pipeline/jobs/{job_id}")
    list_runs = _get_route_endpoint(app, "/api/runs")

    queue_payload = asyncio.run(
        queue_jobs(
            body=QueuePipelineJobsBody(
                video_paths=[str(video_a), str(video_b)],
                run_name_prefix="weekend",
                resume=False,
                no_overlay=True,
            )
        )
    )
    assert queue_payload["accepted_count"] == 2
    queued_jobs = queue_payload["jobs"]
    assert len(queued_jobs) == 2
    assert queued_jobs[0]["run_name"] != queued_jobs[1]["run_name"]
    assert all(job["run_name"].startswith("weekend_") for job in queued_jobs)

    # Poll until both jobs finish.
    latest_jobs = []
    for _ in range(80):
        jobs_payload = asyncio.run(list_jobs(limit=10, include_logs=False))
        latest_jobs = jobs_payload["jobs"]
        if len(latest_jobs) >= 2 and all(job["status"] == "succeeded" for job in latest_jobs[:2]):
            break
        time.sleep(0.05)

    assert len(latest_jobs) >= 2
    assert all(job["status"] == "succeeded" for job in latest_jobs[:2])

    # Detail endpoint should include stage metadata and logs for each completed job.
    for queued in queued_jobs:
        detail_payload = asyncio.run(
            get_job(job_id=queued["job_id"], include_logs=True)
        )
        assert detail_payload["status"] == "succeeded"
        assert detail_payload["run_name"] == queued["run_name"]
        assert detail_payload["stage_total"] >= 12
        assert isinstance(detail_payload.get("logs"), list)
        assert detail_payload["logs"]

    runs_payload = asyncio.run(list_runs())
    run_names = {run["name"] for run in runs_payload["runs"]}
    assert queued_jobs[0]["run_name"] in run_names
    assert queued_jobs[1]["run_name"] in run_names


def test_pipeline_job_queue_rejects_single_run_name_for_batch(tmp_path: Path):
    """Batch queue requests should reject run_name in favor of run_name_prefix."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    video_a = tmp_path / "match_a.mp4"
    video_b = tmp_path / "match_b.mp4"
    video_a.write_bytes(b"a")
    video_b.write_bytes(b"b")

    app = create_app(runs_dir)
    queue_jobs = _get_route_endpoint(app, "/api/pipeline/jobs", method="POST")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            queue_jobs(
                body=QueuePipelineJobsBody(
                    video_paths=[str(video_a), str(video_b)],
                    run_name="batch_name",
                )
            )
        )
    assert exc_info.value.status_code == 400


def test_pipeline_job_cancel_for_queued_job(tmp_path: Path, monkeypatch):
    """Cancel endpoint should cancel queued jobs while allowing current run to finish."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    video_a = tmp_path / "queue_a.mp4"
    video_b = tmp_path / "queue_b.mp4"
    video_a.write_bytes(b"a")
    video_b.write_bytes(b"b")

    gate = threading.Event()
    counter_lock = threading.Lock()
    call_count = {"value": 0}

    def _fake_pipeline_run(self, video_path, output_dir, resume=False):
        with counter_lock:
            call_count["value"] += 1
            current = call_count["value"]

        if current == 1:
            gate.wait(timeout=2.0)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        _write_json(
            output_path / "run_manifest.json",
            {
                "schema_version": "1.1",
                "video_path": str(video_path),
                "original_video_path": str(video_path),
                "output_dir": str(output_dir),
            },
        )
        return {}

    monkeypatch.setattr("src.pipeline.base.Pipeline.run", _fake_pipeline_run)

    app = create_app(runs_dir)
    queue_jobs = _get_route_endpoint(app, "/api/pipeline/jobs", method="POST")
    list_jobs = _get_route_endpoint(app, "/api/pipeline/jobs")
    cancel_job = _get_route_endpoint(app, "/api/pipeline/jobs/{job_id}/cancel", method="POST")

    queued = asyncio.run(
        queue_jobs(
            body=QueuePipelineJobsBody(
                video_paths=[str(video_a), str(video_b)],
                run_name_prefix="cancel_case",
            )
        )
    )
    jobs = queued["jobs"]
    assert len(jobs) == 2

    # Wait until one job is running and one is queued.
    queued_job_id = None
    for _ in range(80):
        payload = asyncio.run(list_jobs(limit=10, include_logs=False))
        statuses = {row["job_id"]: row["status"] for row in payload["jobs"]}
        running_ids = [job_id for job_id, status in statuses.items() if status == "running"]
        queued_ids = [job_id for job_id, status in statuses.items() if status == "queued"]
        if running_ids and queued_ids:
            queued_job_id = queued_ids[0]
            break
        time.sleep(0.05)
    assert queued_job_id is not None

    cancel_payload = asyncio.run(cancel_job(job_id=queued_job_id))
    assert cancel_payload["success"] is True
    assert cancel_payload["job"]["status"] == "cancelled"

    gate.set()
    _wait_for_job_status(list_jobs, queued_job_id, "cancelled")


def test_pipeline_retry_and_duplicate_endpoints(tmp_path: Path, monkeypatch):
    """Retry + duplicate endpoints should queue new jobs with copied settings."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    video_path = tmp_path / "retry_source.mp4"
    video_path.write_bytes(b"video")

    call_counter = {"value": 0}

    def _fake_pipeline_run(self, video_path, output_dir, resume=False):
        call_counter["value"] += 1
        if call_counter["value"] == 1:
            raise RuntimeError("forced failure")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        _write_json(
            output_path / "run_manifest.json",
            {
                "schema_version": "1.1",
                "video_path": str(video_path),
                "original_video_path": str(video_path),
                "output_dir": str(output_dir),
            },
        )
        return {}

    monkeypatch.setattr("src.pipeline.base.Pipeline.run", _fake_pipeline_run)

    app = create_app(runs_dir)
    queue_jobs = _get_route_endpoint(app, "/api/pipeline/jobs", method="POST")
    list_jobs = _get_route_endpoint(app, "/api/pipeline/jobs")
    retry_job = _get_route_endpoint(app, "/api/pipeline/jobs/{job_id}/retry", method="POST")
    duplicate_job = _get_route_endpoint(app, "/api/pipeline/jobs/{job_id}/duplicate", method="POST")

    queue_payload = asyncio.run(
        queue_jobs(
            body=QueuePipelineJobsBody(
                video_paths=[str(video_path)],
                run_name="retry_case",
            )
        )
    )
    source_job = queue_payload["jobs"][0]
    source_job_id = source_job["job_id"]

    _wait_for_job_status(list_jobs, source_job_id, "failed")

    retry_payload = asyncio.run(retry_job(job_id=source_job_id))
    assert retry_payload["success"] is True
    retry_job_info = retry_payload["job"]
    assert retry_job_info["source_job_id"] == source_job_id

    retry_final = _wait_for_job_status(list_jobs, retry_job_info["job_id"], "succeeded")
    assert str(retry_final["run_name"]).startswith("retry_case_retry")

    duplicate_payload = asyncio.run(duplicate_job(job_id=retry_job_info["job_id"]))
    assert duplicate_payload["success"] is True
    duplicate_info = duplicate_payload["job"]
    assert duplicate_info["source_job_id"] == retry_job_info["job_id"]

    duplicate_final = _wait_for_job_status(list_jobs, duplicate_info["job_id"], "succeeded")
    assert str(duplicate_final["run_name"]).startswith(str(retry_final["run_name"]) + "_copy")


def test_pipeline_jobs_are_persisted_between_app_instances(tmp_path: Path, monkeypatch):
    """Persisted jobs should be available after creating a new app instance."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    video_path = tmp_path / "persist_case.mp4"
    video_path.write_bytes(b"persist")

    def _fake_pipeline_run(self, video_path, output_dir, resume=False):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        _write_json(
            output_path / "run_manifest.json",
            {
                "schema_version": "1.1",
                "video_path": str(video_path),
                "original_video_path": str(video_path),
                "output_dir": str(output_dir),
            },
        )
        return {}

    monkeypatch.setattr("src.pipeline.base.Pipeline.run", _fake_pipeline_run)

    app_one = create_app(runs_dir)
    queue_jobs = _get_route_endpoint(app_one, "/api/pipeline/jobs", method="POST")
    list_jobs_one = _get_route_endpoint(app_one, "/api/pipeline/jobs")

    queue_payload = asyncio.run(
        queue_jobs(
            body=QueuePipelineJobsBody(
                video_paths=[str(video_path)],
                run_name="persist_case_run",
            )
        )
    )
    queued_job = queue_payload["jobs"][0]
    _wait_for_job_status(list_jobs_one, queued_job["job_id"], "succeeded")

    app_two = create_app(runs_dir)
    list_jobs_two = _get_route_endpoint(app_two, "/api/pipeline/jobs")
    persisted_payload = asyncio.run(list_jobs_two(limit=50, include_logs=False))
    persisted_ids = {row["job_id"] for row in persisted_payload["jobs"]}
    assert queued_job["job_id"] in persisted_ids


def test_team_analytics_endpoint(tmp_path: Path):
    """Team analytics endpoint returns the content of team_analytics.json."""
    runs_dir = tmp_path / "runs"
    run_path = runs_dir / "analytics_run"
    run_path.mkdir(parents=True)

    analytics_data = {
        "schema_version": "1.0",
        "possession": {"teams": {"team_A": {"share": 0.55, "seconds": 120}}, "dominant_team": "team_A"},
        "territory": {"teams": {}, "x_bins": ["left", "center", "right"], "y_bins": ["top", "middle", "bottom"]},
        "pass_network": {"passes_inferred": 42, "top_edges": []},
        "pressing": {"teams": {}},
    }
    _write_json(run_path / "team_analytics.json", analytics_data)

    app = create_app(runs_dir)
    endpoint = _get_route_endpoint(app, "/api/runs/{run_name}/team_analytics")
    result = asyncio.run(endpoint("analytics_run"))

    assert result["possession"]["dominant_team"] == "team_A"
    assert result["pass_network"]["passes_inferred"] == 42


def test_team_analytics_endpoint_returns_404_when_missing(tmp_path: Path):
    """Team analytics endpoint returns 404 when file doesn't exist."""
    runs_dir = tmp_path / "runs"
    run_path = runs_dir / "no_analytics_run"
    run_path.mkdir(parents=True)

    app = create_app(runs_dir)
    endpoint = _get_route_endpoint(app, "/api/runs/{run_name}/team_analytics")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(endpoint("no_analytics_run"))
    assert exc_info.value.status_code == 404


def test_match_stats_endpoint(tmp_path: Path):
    """Match stats endpoint returns the content of match_stats.json."""
    runs_dir = tmp_path / "runs"
    run_path = runs_dir / "stats_run"
    run_path.mkdir(parents=True)

    stats_data = {
        "schema_version": "1.0",
        "summary": {"events_processed": 14, "events_without_team": 1},
        "teams": {
            "ours": {"shots": 6, "goals": 2, "passes": 41, "set_pieces": 3, "possession_share": 0.58},
            "opponent": {"shots": 4, "goals": 1, "passes": 33, "set_pieces": 2, "possession_share": 0.42},
        },
        "totals": {"shots": 10, "goals": 3, "passes": 74, "set_pieces": 5},
    }
    _write_json(run_path / "match_stats.json", stats_data)

    app = create_app(runs_dir)
    endpoint = _get_route_endpoint(app, "/api/runs/{run_name}/match_stats")
    result = asyncio.run(endpoint("stats_run"))

    assert result["summary"]["events_processed"] == 14
    assert result["teams"]["ours"]["shots"] == 6
    assert result["totals"]["passes"] == 74


def test_match_stats_endpoint_returns_404_when_missing(tmp_path: Path):
    """Match stats endpoint returns 404 when file doesn't exist."""
    runs_dir = tmp_path / "runs"
    run_path = runs_dir / "no_stats_run"
    run_path.mkdir(parents=True)

    app = create_app(runs_dir)
    endpoint = _get_route_endpoint(app, "/api/runs/{run_name}/match_stats")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(endpoint("no_stats_run"))
    assert exc_info.value.status_code == 404


def test_list_runs_includes_match_stats_flag(tmp_path: Path):
    """Runs API should expose match-stats availability per run."""
    runs_dir = tmp_path / "runs"
    run_with_stats = runs_dir / "with_stats"
    run_without_stats = runs_dir / "without_stats"
    run_with_stats.mkdir(parents=True)
    run_without_stats.mkdir(parents=True)

    _write_json(run_with_stats / "match_stats.json", {"schema_version": "1.0", "teams": {}, "totals": {}, "summary": {}})

    app = create_app(runs_dir)
    endpoint = _get_route_endpoint(app, "/api/runs")
    result = asyncio.run(endpoint())
    run_rows = {row["name"]: row for row in result["runs"]}

    assert run_rows["with_stats"]["has_match_stats"] is True
    assert run_rows["without_stats"]["has_match_stats"] is False


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


def test_metadata_endpoint_includes_summary(tmp_path: Path):
    """Metadata endpoint returns summary.json content when available."""
    runs_dir = tmp_path / "runs"
    run_path = runs_dir / "summary_run"
    run_path.mkdir(parents=True)

    summary_data = {
        "schema_version": "1.0",
        "video": {"duration_seconds": 300, "resolution": {"width": 1920, "height": 1080}},
        "counts": {"detections_total": 50000, "tracks_unique": 30, "events_total": 5, "shots": 3, "goals": 2},
    }
    _write_json(run_path / "summary.json", summary_data)

    app = create_app(runs_dir)
    endpoint = _get_route_endpoint(app, "/api/runs/{run_name}/metadata")
    result = asyncio.run(endpoint("summary_run"))

    assert "summary" in result
    assert result["summary"]["counts"]["tracks_unique"] == 30
    assert result["summary"]["video"]["duration_seconds"] == 300


def test_speedrun_endpoint_builds_high_and_low_action_windows(tmp_path: Path):
    """Speedrun endpoint should merge event windows and expose complementary low-action spans."""
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
    """When no events exist, speedrun endpoint should expose one full-match action window."""
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


# ── Run tags API tests ───────────────────────────────────────────────────


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

# ── Team CRUD API tests ─────────────────────────────────────────────────


def test_create_team_api(tmp_path: Path):
    """POST /api/teams should create a team."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    # Pre-create database so get_player_db_path finds it
    db_path = runs_dir / "players.db"
    PlayerDatabase(db_path).close()

    app = create_app(runs_dir)
    endpoint = _get_route_endpoint(app, "/api/teams", "POST")
    body = CreateTeamBody(name="FC Test", short_name="TST")
    result = asyncio.run(endpoint(body))
    assert result["success"] is True
    assert result["team"]["name"] == "FC Test"
    assert result["team"]["short_name"] == "TST"


def test_list_teams_api_empty(tmp_path: Path):
    """GET /api/teams should return empty list when no DB exists."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    app = create_app(runs_dir)
    endpoint = _get_route_endpoint(app, "/api/teams")
    result = asyncio.run(endpoint())
    assert result["teams"] == []
    assert result["count"] == 0


def test_list_teams_api_with_data(tmp_path: Path):
    """GET /api/teams should list all teams with kits."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    db_path = runs_dir / "players.db"
    with PlayerDatabase(db_path) as db:
        t = db.create_team(name="FC Test")
        db.upsert_kit(t.team_id, "home", color_hex="#FF0000")

    app = create_app(runs_dir)
    endpoint = _get_route_endpoint(app, "/api/teams")
    result = asyncio.run(endpoint())
    assert result["count"] == 1
    assert result["teams"][0]["name"] == "FC Test"
    assert len(result["teams"][0]["kits"]) == 1


def test_get_team_api(tmp_path: Path):
    """GET /api/teams/{team_id} should return team with kits and players."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    db_path = runs_dir / "players.db"
    with PlayerDatabase(db_path) as db:
        t = db.create_team(name="FC Test")
        db.upsert_kit(t.team_id, "home", color_hex="#FF0000")
        p = db.create_player(name="Player 1")
        db.set_player_team(p.player_id, t.team_id)

    app = create_app(runs_dir)
    endpoint = _get_route_endpoint(app, "/api/teams/{team_id}")
    result = asyncio.run(endpoint(t.team_id))
    assert result["name"] == "FC Test"
    assert len(result["kits"]) == 1
    assert len(result["players"]) == 1


def test_get_team_api_not_found(tmp_path: Path):
    """GET /api/teams/{team_id} should return 404 for missing team."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    db_path = runs_dir / "players.db"
    PlayerDatabase(db_path).close()

    app = create_app(runs_dir)
    endpoint = _get_route_endpoint(app, "/api/teams/{team_id}")
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(endpoint(9999))
    assert exc_info.value.status_code == 404


def test_update_team_api(tmp_path: Path):
    """PATCH /api/teams/{team_id} should update team."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    db_path = runs_dir / "players.db"
    with PlayerDatabase(db_path) as db:
        t = db.create_team(name="FC Old")

    app = create_app(runs_dir)
    endpoint = _get_route_endpoint(app, "/api/teams/{team_id}", "PATCH")
    body = UpdateTeamBody(name="FC New")
    result = asyncio.run(endpoint(t.team_id, body))
    assert result["success"] is True
    assert result["team"]["name"] == "FC New"


def test_delete_team_api(tmp_path: Path):
    """DELETE /api/teams/{team_id} should delete team."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    db_path = runs_dir / "players.db"
    with PlayerDatabase(db_path) as db:
        t = db.create_team(name="FC Delete")

    app = create_app(runs_dir)
    endpoint = _get_route_endpoint(app, "/api/teams/{team_id}", "DELETE")
    result = asyncio.run(endpoint(t.team_id))
    assert result["success"] is True


def test_set_run_teams_api(tmp_path: Path):
    """POST /api/runs/{run_name}/teams should create associations."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    db_path = runs_dir / "players.db"
    with PlayerDatabase(db_path) as db:
        t1 = db.create_team(name="Home")
        t2 = db.create_team(name="Away")

    app = create_app(runs_dir)
    endpoint = _get_route_endpoint(app, "/api/runs/{run_name}/teams", "POST")
    body = SetRunTeamsBody(home_team_id=t1.team_id, away_team_id=t2.team_id)
    result = asyncio.run(endpoint("test_run", body))
    assert result["success"] is True
    assert len(result["associations"]) == 2


def test_get_run_teams_api(tmp_path: Path):
    """GET /api/runs/{run_name}/teams should return associations."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    db_path = runs_dir / "players.db"
    with PlayerDatabase(db_path) as db:
        t1 = db.create_team(name="Home")
        t2 = db.create_team(name="Away")
        db.set_run_teams("test_run", t1.team_id, t2.team_id)

    app = create_app(runs_dir)
    endpoint = _get_route_endpoint(app, "/api/runs/{run_name}/teams")
    result = asyncio.run(endpoint("test_run"))
    assert len(result["associations"]) == 2
    home = [a for a in result["associations"] if a["role"] == "home"][0]
    assert home["team_name"] == "Home"


def test_remap_run_teams_api(tmp_path: Path):
    """POST /api/runs/{run_name}/teams/remap should swap cluster-to-team mapping."""
    runs_dir = tmp_path / "runs"
    run_path = runs_dir / "test_run"
    run_path.mkdir(parents=True)
    db_path = runs_dir / "players.db"
    with PlayerDatabase(db_path) as db:
        t1 = db.create_team(name="Home FC")
        t2 = db.create_team(name="Away FC")
        db.set_run_teams("test_run", t1.team_id, t2.team_id)

    teams_json = {
        "n_teams": 2,
        "team_names": {"0": "Home FC", "1": "Away FC"},
        "cluster_to_role": {"0": "home", "1": "away"},
        "db_team_ids": {"home": t1.team_id, "away": t2.team_id},
    }
    _write_json(run_path / "teams.json", teams_json)

    app = create_app(runs_dir)
    endpoint = _get_route_endpoint(app, "/api/runs/{run_name}/teams/remap", "POST")
    body = RemapRunTeamsBody()
    result = asyncio.run(endpoint("test_run", body))
    assert result["success"] is True
    # Roles should be swapped
    assert result["cluster_to_role"]["0"] == "away"
    assert result["cluster_to_role"]["1"] == "home"


def test_pipeline_job_body_includes_team_fields(tmp_path: Path):
    """QueuePipelineJobsBody should accept team selection fields."""
    body = QueuePipelineJobsBody(
        video_paths=["test.mp4"],
        home_team_id=1,
        away_team_id=2,
        home_kit="away",
        away_kit="third",
    )
    assert body.home_team_id == 1
    assert body.away_team_id == 2
    assert body.home_kit == "away"
    assert body.away_kit == "third"


# ── Team logo endpoint tests ──────────────────────────────────────────

class _FakeUploadFile:
    """Minimal UploadFile stand-in for endpoint tests."""
    def __init__(self, content: bytes, filename: str = "photo.jpg"):
        self.filename = filename
        self._content = content

    async def read(self):
        return self._content


def test_upload_team_logo(tmp_path: Path):
    """POST /api/teams/{team_id}/logo should save file and set logo_path."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    db_path = runs_dir / "players.db"
    with PlayerDatabase(db_path) as db:
        t = db.create_team(name="FC Logo")

    app = create_app(runs_dir)
    endpoint = _get_route_endpoint(app, "/api/teams/{team_id}/logo", "POST")
    fake_file = _FakeUploadFile(b"\x89PNG fake logo", "crest.png")
    result = asyncio.run(endpoint(t.team_id, fake_file))
    assert result["success"] is True
    assert "team_logos" in result["logo_path"]


def test_serve_team_logo(tmp_path: Path):
    """GET /api/teams/{team_id}/logo should return FileResponse."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    db_path = runs_dir / "players.db"
    with PlayerDatabase(db_path) as db:
        t = db.create_team(name="FC Logo")

    app = create_app(runs_dir)
    # Upload first
    upload_ep = _get_route_endpoint(app, "/api/teams/{team_id}/logo", "POST")
    fake_file = _FakeUploadFile(b"\x89PNG fake logo", "crest.png")
    asyncio.run(upload_ep(t.team_id, fake_file))

    # Serve
    serve_ep = _get_route_endpoint(app, "/api/teams/{team_id}/logo", "GET")
    result = asyncio.run(serve_ep(t.team_id))
    assert isinstance(result, FileResponse)


def test_serve_team_logo_not_found(tmp_path: Path):
    """GET /api/teams/{team_id}/logo should 404 when no logo."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    db_path = runs_dir / "players.db"
    with PlayerDatabase(db_path) as db:
        t = db.create_team(name="FC NoLogo")

    app = create_app(runs_dir)
    endpoint = _get_route_endpoint(app, "/api/teams/{team_id}/logo", "GET")
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(endpoint(t.team_id))
    assert exc_info.value.status_code == 404


def test_delete_team_logo(tmp_path: Path):
    """DELETE /api/teams/{team_id}/logo should remove logo."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    db_path = runs_dir / "players.db"
    with PlayerDatabase(db_path) as db:
        t = db.create_team(name="FC Logo")

    app = create_app(runs_dir)
    # Upload first
    upload_ep = _get_route_endpoint(app, "/api/teams/{team_id}/logo", "POST")
    fake_file = _FakeUploadFile(b"\x89PNG fake logo", "crest.png")
    asyncio.run(upload_ep(t.team_id, fake_file))

    # Delete
    delete_ep = _get_route_endpoint(app, "/api/teams/{team_id}/logo", "DELETE")
    result = asyncio.run(delete_ep(t.team_id))
    assert result["success"] is True

    # Verify logo is gone
    serve_ep = _get_route_endpoint(app, "/api/teams/{team_id}/logo", "GET")
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(serve_ep(t.team_id))
    assert exc_info.value.status_code == 404


def test_serve_team_logo_extension_mismatch_autoheals(tmp_path: Path):
    """GET /api/teams/{team_id}/logo should find file even when DB ext differs."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    db_path = runs_dir / "players.db"
    with PlayerDatabase(db_path) as db:
        t = db.create_team(name="FC Heal")

    app = create_app(runs_dir)
    # Upload a .png logo
    upload_ep = _get_route_endpoint(app, "/api/teams/{team_id}/logo", "POST")
    fake_file = _FakeUploadFile(b"\x89PNG fake logo", "crest.png")
    asyncio.run(upload_ep(t.team_id, fake_file))

    # Manually corrupt the DB path to a .jpg extension
    with PlayerDatabase(db_path) as db:
        db.set_team_logo(t.team_id, f"data/team_logos/{t.team_id}/logo.jpg")

    # Serve should still find the .png file and auto-heal the DB
    serve_ep = _get_route_endpoint(app, "/api/teams/{team_id}/logo", "GET")
    result = asyncio.run(serve_ep(t.team_id))
    assert isinstance(result, FileResponse)

    # Verify DB was healed
    with PlayerDatabase(db_path) as db:
        team = db.get_team(t.team_id)
        assert team.logo_path.endswith(".png")


def test_serve_player_photo_extension_mismatch_autoheals(tmp_path: Path):
    """GET /api/players/{id}/photo should find file even when DB ext differs."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    db_path = runs_dir / "players.db"
    with PlayerDatabase(db_path) as db:
        p = db.create_player(name="Heal Player")

    app = create_app(runs_dir)
    # Upload a .png photo
    upload_ep = _get_route_endpoint(app, "/api/players/{player_id}/photo", "POST")
    fake_file = _FakeUploadFile(b"\x89PNG fake photo", "pic.png")
    asyncio.run(upload_ep(p.player_id, fake_file))

    # Manually corrupt the DB path to a .jpg extension
    with PlayerDatabase(db_path) as db:
        db.set_player_photo(p.player_id, f"data/player_photos/{p.player_id}/photo.jpg")

    # Serve should still find the .png file and auto-heal the DB
    serve_ep = _get_route_endpoint(app, "/api/players/{player_id}/photo", "GET")
    result = asyncio.run(serve_ep(p.player_id))
    assert isinstance(result, FileResponse)

    # Verify DB was healed
    with PlayerDatabase(db_path) as db:
        player = db.get_player(p.player_id)
        assert player.photo_path.endswith(".png")


# ── Player photo endpoint tests ────────────────────────────────────────

def test_upload_player_photo(tmp_path: Path):
    """POST /api/players/{player_id}/photo should save file and set photo_path."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    db_path = runs_dir / "players.db"
    with PlayerDatabase(db_path) as db:
        p = db.create_player(name="Photo Player")

    app = create_app(runs_dir)
    endpoint = _get_route_endpoint(app, "/api/players/{player_id}/photo", "POST")
    fake_file = _FakeUploadFile(b"\x89PNG fake image data", "headshot.png")
    result = asyncio.run(endpoint(p.player_id, fake_file))
    assert result["success"] is True
    assert result["player"]["photo_path"] is not None
    assert "player_photos" in result["player"]["photo_path"]


def test_serve_player_photo(tmp_path: Path):
    """GET /api/players/{player_id}/photo should return FileResponse."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    db_path = runs_dir / "players.db"
    with PlayerDatabase(db_path) as db:
        p = db.create_player(name="Photo Player")

    app = create_app(runs_dir)
    # Upload first
    upload_ep = _get_route_endpoint(app, "/api/players/{player_id}/photo", "POST")
    fake_file = _FakeUploadFile(b"\x89PNG fake image data", "photo.jpg")
    asyncio.run(upload_ep(p.player_id, fake_file))

    # Serve
    serve_ep = _get_route_endpoint(app, "/api/players/{player_id}/photo")
    result = asyncio.run(serve_ep(p.player_id))
    assert isinstance(result, FileResponse)


def test_serve_player_photo_not_found(tmp_path: Path):
    """GET /api/players/{player_id}/photo should 404 when no photo."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    db_path = runs_dir / "players.db"
    with PlayerDatabase(db_path) as db:
        p = db.create_player(name="No Photo")

    app = create_app(runs_dir)
    endpoint = _get_route_endpoint(app, "/api/players/{player_id}/photo")
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(endpoint(p.player_id))
    assert exc_info.value.status_code == 404


def test_delete_player_photo(tmp_path: Path):
    """DELETE /api/players/{player_id}/photo should remove file and clear path."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    db_path = runs_dir / "players.db"
    with PlayerDatabase(db_path) as db:
        p = db.create_player(name="Photo Player")

    app = create_app(runs_dir)
    # Upload first
    upload_ep = _get_route_endpoint(app, "/api/players/{player_id}/photo", "POST")
    fake_file = _FakeUploadFile(b"\x89PNG fake image data", "photo.jpg")
    asyncio.run(upload_ep(p.player_id, fake_file))

    # Delete
    delete_ep = _get_route_endpoint(app, "/api/players/{player_id}/photo", "DELETE")
    result = asyncio.run(delete_ep(p.player_id))
    assert result["success"] is True

    # Verify photo is gone from DB
    with PlayerDatabase(db_path) as db:
        updated = db.get_player(p.player_id)
        assert updated.photo_path is None


def test_list_players_includes_team_name(tmp_path: Path):
    """GET /api/players should include team_name in each player."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    db_path = runs_dir / "players.db"
    with PlayerDatabase(db_path) as db:
        t = db.create_team(name="FC Test")
        p = db.create_player(name="Linked Player")
        db.set_player_team(p.player_id, t.team_id)
        p2 = db.create_player(name="Unlinked Player")

    app = create_app(runs_dir)
    endpoint = _get_route_endpoint(app, "/api/players")
    result = asyncio.run(endpoint())
    assert result["count"] == 2
    players_by_id = {p["player_id"]: p for p in result["players"]}
    assert players_by_id[p.player_id]["team_name"] == "FC Test"
    assert players_by_id[p2.player_id]["team_name"] is None


# ── Training image endpoint tests ──────────────────────────────────────

# The server computes project_root from its own __file__ location.
_SERVER_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _training_dir_for(player_id: int) -> Path:
    """Return the project-root-relative training directory for a player."""
    return _SERVER_PROJECT_ROOT / "data" / "player_photos" / str(player_id) / "training"


def _cleanup_training_dir(player_id: int) -> None:
    """Remove training dir created by tests (best-effort)."""
    d = _SERVER_PROJECT_ROOT / "data" / "player_photos" / str(player_id)
    import shutil
    if d.exists():
        shutil.rmtree(d)


def test_upload_training_images(tmp_path: Path):
    """POST /api/players/{player_id}/training-images should generate embeddings and delete images."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    db_path = runs_dir / "players.db"
    # Use a high player_id unlikely to collide with real data
    with PlayerDatabase(db_path) as db:
        p = db.create_player(name="Train Player")
    pid = p.player_id

    # Pre-clean to avoid stale data
    _cleanup_training_dir(pid)

    app = create_app(runs_dir)
    endpoint = _get_route_endpoint(app, "/api/players/{player_id}/training-images", "POST")

    fake_files = [
        _FakeUploadFile(b"\x89PNG fake image 1", "face1.png"),
        _FakeUploadFile(b"\x89PNG fake image 2", "face2.png"),
    ]

    try:
        with patch("src.identity.embedding_generator.add_embeddings_from_images") as mock_add:
            import numpy as _np
            mock_add.return_value = {
                "player_id": str(pid),
                "num_encodings": 2,
                "averaged_encoding": _np.ones(512, dtype=_np.float32) / _np.sqrt(512),
                "stats": {
                    "total_images_processed": 2,
                    "successful_extractions": 2,
                    "failed_extractions": 0,
                    "existing_encodings_kept": 0,
                },
            }
            result = asyncio.run(endpoint(pid, fake_files))

        assert result["success"] is True
        assert result["embedding_count"] == 2
        assert result["training_image_count"] == 2

        # Images should be deleted after training
        training_dir = _training_dir_for(pid)
        if training_dir.exists():
            assert len(list(training_dir.iterdir())) == 0
    finally:
        _cleanup_training_dir(pid)


def test_upload_training_images_player_not_found(tmp_path: Path):
    """POST /api/players/{player_id}/training-images should 404 for missing player."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    db_path = runs_dir / "players.db"
    PlayerDatabase(db_path).close()

    app = create_app(runs_dir)
    endpoint = _get_route_endpoint(app, "/api/players/{player_id}/training-images", "POST")

    fake_files = [_FakeUploadFile(b"\x89PNG fake", "face.png")]
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(endpoint(9999, fake_files))
    assert exc_info.value.status_code == 404


def test_list_training_images(tmp_path: Path):
    """GET /api/players/{player_id}/training-images should list embedding info from pkl."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    db_path = runs_dir / "players.db"
    with PlayerDatabase(db_path) as db:
        p = db.create_player(name="List Player")

    # Images are no longer kept — create a pkl file with embedding metadata
    player_dir = Path("data/player_photos") / str(p.player_id)
    player_dir.mkdir(parents=True, exist_ok=True)
    try:
        import numpy as _np
        import pickle
        payload = {
            "encodings": [
                {"image_name": "img_000.jpg", "timestamp": "2026-01-01T00:00:00", "model": "Facenet512",
                 "encoding": _np.ones(512, dtype=_np.float32)},
                {"image_name": "img_001.png", "timestamp": "2026-01-01T00:00:01", "model": "Facenet512",
                 "encoding": _np.ones(512, dtype=_np.float32)},
            ],
        }
        with open(player_dir / "embeddings.pkl", "wb") as f:
            pickle.dump(payload, f)

        app = create_app(runs_dir)
        endpoint = _get_route_endpoint(app, "/api/players/{player_id}/training-images", "GET")
        result = asyncio.run(endpoint(p.player_id))

        # Images list is empty (not kept), but encodings info is returned
        assert result["count"] == 0
        assert result["images"] == []
        assert result["has_embeddings"] is True
        assert len(result["encodings"]) == 2
        names = {e["image_name"] for e in result["encodings"]}
        assert "img_000.jpg" in names
        assert "img_001.png" in names
    finally:
        import shutil
        shutil.rmtree(player_dir, ignore_errors=True)


def test_serve_training_image(tmp_path: Path):
    """GET /api/players/{player_id}/training-images/{filename} should serve file."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    db_path = runs_dir / "players.db"
    PlayerDatabase(db_path).close()

    # Use a unique player_id to avoid collision
    training_dir = _training_dir_for(99990)
    training_dir.mkdir(parents=True, exist_ok=True)
    try:
        (training_dir / "img_000.jpg").write_bytes(b"\x89PNG fake")

        app = create_app(runs_dir)
        endpoint = _get_route_endpoint(app, "/api/players/{player_id}/training-images/{filename}", "GET")
        result = asyncio.run(endpoint(99990, "img_000.jpg"))
        assert isinstance(result, FileResponse)
    finally:
        _cleanup_training_dir(99990)


def test_serve_training_image_not_found(tmp_path: Path):
    """GET /api/players/{player_id}/training-images/{filename} 404 for missing."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)

    app = create_app(runs_dir)
    endpoint = _get_route_endpoint(app, "/api/players/{player_id}/training-images/{filename}", "GET")
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(endpoint(1, "nonexistent.jpg"))
    assert exc_info.value.status_code == 404


def test_delete_training_image(tmp_path: Path):
    """DELETE /api/players/{player_id}/training-images/{filename} should remove embedding from pkl."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    db_path = runs_dir / "players.db"
    with PlayerDatabase(db_path) as db:
        p = db.create_player(name="Delete Player")

    player_dir = Path("data/player_photos") / str(p.player_id)
    player_dir.mkdir(parents=True, exist_ok=True)
    try:
        import numpy as _np
        import pickle
        payload = {
            "player_id": str(p.player_id),
            "num_encodings": 2,
            "averaged_encoding": _np.ones(512, dtype=_np.float32) / _np.sqrt(512),
            "encodings": [
                {"image_name": "img_000.jpg", "encoding": _np.ones(512, dtype=_np.float32) / _np.sqrt(512)},
                {"image_name": "img_001.jpg", "encoding": _np.ones(512, dtype=_np.float32) / _np.sqrt(512)},
            ],
        }
        with open(player_dir / "embeddings.pkl", "wb") as f:
            pickle.dump(payload, f)

        app = create_app(runs_dir)
        endpoint = _get_route_endpoint(app, "/api/players/{player_id}/training-images/{filename}", "DELETE")
        result = asyncio.run(endpoint(p.player_id, "img_000.jpg"))

        assert result["success"] is True
        assert result["embedding_count"] == 1

        # Verify pkl was updated
        with open(player_dir / "embeddings.pkl", "rb") as f:
            updated = pickle.load(f)
        assert len(updated["encodings"]) == 1
        assert updated["encodings"][0]["image_name"] == "img_001.jpg"
    finally:
        import shutil
        shutil.rmtree(player_dir, ignore_errors=True)


def test_delete_training_image_not_found(tmp_path: Path):
    """DELETE /api/players/{player_id}/training-images/{filename} 404 for missing embedding."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    db_path = runs_dir / "players.db"
    with PlayerDatabase(db_path) as db:
        p = db.create_player(name="NotFound Player")

    app = create_app(runs_dir)
    endpoint = _get_route_endpoint(app, "/api/players/{player_id}/training-images/{filename}", "DELETE")
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(endpoint(p.player_id, "nonexistent.jpg"))
    assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# Pipeline job deletion tests
# ---------------------------------------------------------------------------

def test_delete_pipeline_job_removes_metadata_and_files(tmp_path: Path, monkeypatch):
    """DELETE /api/pipeline/jobs/{job_id} removes job metadata and run files."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    video_a = tmp_path / "del_a.mp4"
    video_b = tmp_path / "del_b.mp4"
    video_a.write_bytes(b"a")
    video_b.write_bytes(b"b")

    gate = threading.Event()

    def _fake_pipeline_run(self, video_path, output_dir, resume=False):
        gate.wait(timeout=4.0)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        _write_json(
            output_path / "run_manifest.json",
            {"schema_version": "1.1", "video_path": str(video_path),
             "original_video_path": str(video_path), "output_dir": str(output_dir)},
        )
        return {}

    monkeypatch.setattr("src.pipeline.base.Pipeline.run", _fake_pipeline_run)

    app = create_app(runs_dir)
    queue_jobs = _get_route_endpoint(app, "/api/pipeline/jobs", method="POST")
    list_jobs = _get_route_endpoint(app, "/api/pipeline/jobs")
    cancel_job = _get_route_endpoint(app, "/api/pipeline/jobs/{job_id}/cancel", method="POST")
    delete_job = _get_route_endpoint(app, "/api/pipeline/jobs/{job_id}", method="DELETE")

    queued = asyncio.run(
        queue_jobs(
            body=QueuePipelineJobsBody(
                video_paths=[str(video_a), str(video_b)],
                run_name_prefix="del_case",
            )
        )
    )
    jobs = queued["jobs"]
    assert len(jobs) == 2

    # Wait until one is running and one is queued
    queued_job_id = None
    for _ in range(80):
        payload = asyncio.run(list_jobs(limit=10, include_logs=False))
        statuses = {row["job_id"]: row["status"] for row in payload["jobs"]}
        running_ids = [jid for jid, s in statuses.items() if s == "running"]
        queued_ids = [jid for jid, s in statuses.items() if s == "queued"]
        if running_ids and queued_ids:
            queued_job_id = queued_ids[0]
            break
        time.sleep(0.05)
    assert queued_job_id is not None

    # Cancel the queued job (synchronous — instant)
    cancel_result = asyncio.run(cancel_job(job_id=queued_job_id))
    assert cancel_result["job"]["status"] == "cancelled"

    # Find the output dir for the cancelled job
    cancelled_job = cancel_result["job"]
    output_dir = Path(cancelled_job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "artifact.json").write_text("{}")

    # Delete the cancelled job with file cleanup
    result = asyncio.run(delete_job(job_id=queued_job_id, clean_files=True))
    assert result["success"] is True
    assert result["job_id"] == queued_job_id
    assert result["files_removed"] is True

    # Verify job is gone from listing
    payload = asyncio.run(list_jobs(limit=200, include_logs=False))
    job_ids = [j["job_id"] for j in payload["jobs"]]
    assert queued_job_id not in job_ids

    # Verify output directory was removed
    assert not output_dir.exists()

    gate.set()


def test_delete_pipeline_job_running_returns_409(tmp_path: Path, monkeypatch):
    """DELETE should return 409 for a running job."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    video_path = tmp_path / "running.mp4"
    video_path.write_bytes(b"video")

    gate = threading.Event()

    def _fake_pipeline_run(self, video_path, output_dir, resume=False):
        gate.wait(timeout=4.0)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        _write_json(
            output_path / "run_manifest.json",
            {"schema_version": "1.1", "video_path": str(video_path),
             "original_video_path": str(video_path), "output_dir": str(output_dir)},
        )
        return {}

    monkeypatch.setattr("src.pipeline.base.Pipeline.run", _fake_pipeline_run)

    app = create_app(runs_dir)
    queue_jobs = _get_route_endpoint(app, "/api/pipeline/jobs", method="POST")
    list_jobs = _get_route_endpoint(app, "/api/pipeline/jobs")
    delete_job = _get_route_endpoint(app, "/api/pipeline/jobs/{job_id}", method="DELETE")

    queued = asyncio.run(
        queue_jobs(
            body=QueuePipelineJobsBody(
                video_paths=[str(video_path)],
                run_name="running_delete",
            )
        )
    )
    job_id = queued["jobs"][0]["job_id"]

    # Wait for running status
    _wait_for_job_status(list_jobs, job_id, "running")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(delete_job(job_id=job_id, clean_files=True))
    assert exc_info.value.status_code == 409

    gate.set()


def test_delete_pipeline_job_not_found(tmp_path: Path):
    """DELETE should return 404 for unknown job id."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    app = create_app(runs_dir)
    delete_job = _get_route_endpoint(app, "/api/pipeline/jobs/{job_id}", method="DELETE")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(delete_job(job_id="nonexistent_job", clean_files=True))
    assert exc_info.value.status_code == 404


def test_delete_pipeline_job_without_file_cleanup(tmp_path: Path, monkeypatch):
    """DELETE with clean_files=false should keep run directory."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    video_a = tmp_path / "keep_a.mp4"
    video_b = tmp_path / "keep_b.mp4"
    video_a.write_bytes(b"a")
    video_b.write_bytes(b"b")

    gate = threading.Event()

    def _fake_pipeline_run(self, video_path, output_dir, resume=False):
        gate.wait(timeout=4.0)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        _write_json(
            output_path / "run_manifest.json",
            {"schema_version": "1.1", "video_path": str(video_path),
             "original_video_path": str(video_path), "output_dir": str(output_dir)},
        )
        return {}

    monkeypatch.setattr("src.pipeline.base.Pipeline.run", _fake_pipeline_run)

    app = create_app(runs_dir)
    queue_jobs = _get_route_endpoint(app, "/api/pipeline/jobs", method="POST")
    list_jobs = _get_route_endpoint(app, "/api/pipeline/jobs")
    cancel_job = _get_route_endpoint(app, "/api/pipeline/jobs/{job_id}/cancel", method="POST")
    delete_job = _get_route_endpoint(app, "/api/pipeline/jobs/{job_id}", method="DELETE")

    queued = asyncio.run(
        queue_jobs(
            body=QueuePipelineJobsBody(
                video_paths=[str(video_a), str(video_b)],
                run_name_prefix="keep_case",
            )
        )
    )
    assert len(queued["jobs"]) == 2

    # Wait until one is queued
    queued_job_id = None
    for _ in range(80):
        payload = asyncio.run(list_jobs(limit=10, include_logs=False))
        statuses = {row["job_id"]: row["status"] for row in payload["jobs"]}
        queued_ids = [jid for jid, s in statuses.items() if s == "queued"]
        running_ids = [jid for jid, s in statuses.items() if s == "running"]
        if queued_ids and running_ids:
            queued_job_id = queued_ids[0]
            break
        time.sleep(0.05)
    assert queued_job_id is not None

    # Cancel the queued job
    asyncio.run(cancel_job(job_id=queued_job_id))

    # Find the output dir and put files in it
    payload = asyncio.run(list_jobs(limit=10, include_logs=False))
    cancelled_job = next(j for j in payload["jobs"] if j["job_id"] == queued_job_id)
    output_dir = Path(cancelled_job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data.json").write_text("{}")

    result = asyncio.run(delete_job(job_id=queued_job_id, clean_files=False))
    assert result["success"] is True
    assert result["files_removed"] is False

    # Directory should still exist
    assert output_dir.exists()
    assert (output_dir / "data.json").exists()

    gate.set()


# ---------------------------------------------------------------------------
# Run deletion tests
# ---------------------------------------------------------------------------

def test_delete_run_removes_directory_and_job_metadata(tmp_path: Path, monkeypatch):
    """DELETE /api/runs/{run_name} removes run dir and associated pipeline job records."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    video_a = tmp_path / "rundelA.mp4"
    video_b = tmp_path / "rundelB.mp4"
    video_a.write_bytes(b"a")
    video_b.write_bytes(b"b")

    gate = threading.Event()

    def _fake_pipeline_run(self, video_path, output_dir, resume=False):
        gate.wait(timeout=4.0)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        _write_json(
            output_path / "run_manifest.json",
            {"schema_version": "1.1", "video_path": str(video_path),
             "original_video_path": str(video_path), "output_dir": str(output_dir)},
        )
        return {}

    monkeypatch.setattr("src.pipeline.base.Pipeline.run", _fake_pipeline_run)

    app = create_app(runs_dir)
    queue_jobs = _get_route_endpoint(app, "/api/pipeline/jobs", method="POST")
    list_jobs = _get_route_endpoint(app, "/api/pipeline/jobs")
    list_runs = _get_route_endpoint(app, "/api/runs")
    cancel_job = _get_route_endpoint(app, "/api/pipeline/jobs/{job_id}/cancel", method="POST")
    delete_run = _get_route_endpoint(app, "/api/runs/{run_name}", method="DELETE")

    # Queue two jobs so one stays queued
    queued = asyncio.run(
        queue_jobs(
            body=QueuePipelineJobsBody(
                video_paths=[str(video_a), str(video_b)],
                run_name_prefix="rundel",
            )
        )
    )
    assert len(queued["jobs"]) == 2

    # Wait for one running + one queued
    queued_job_id = None
    queued_run_name = None
    for _ in range(80):
        payload = asyncio.run(list_jobs(limit=10, include_logs=False))
        statuses = {row["job_id"]: row for row in payload["jobs"]}
        queued_ids = [jid for jid, j in statuses.items() if j["status"] == "queued"]
        running_ids = [jid for jid, j in statuses.items() if j["status"] == "running"]
        if queued_ids and running_ids:
            queued_job_id = queued_ids[0]
            queued_run_name = statuses[queued_job_id]["run_name"]
            break
        time.sleep(0.05)
    assert queued_job_id is not None

    # Cancel the queued job so it becomes deletable
    asyncio.run(cancel_job(job_id=queued_job_id))

    # Create the run directory with some artifacts
    run_dir = runs_dir / queued_run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_manifest.json").write_text('{"schema_version": "1.1"}')
    (run_dir / "summary.json").write_text('{}')
    assert run_dir.exists()

    # Verify it shows up in the runs list
    runs_before = asyncio.run(list_runs())
    run_names_before = [r["name"] for r in runs_before["runs"]]
    assert queued_run_name in run_names_before

    # Delete the run
    result = asyncio.run(delete_run(run_name=queued_run_name))
    assert result["success"] is True
    assert result["run_name"] == queued_run_name
    assert result["jobs_removed"] >= 1

    # Verify directory is gone
    assert not run_dir.exists()

    # Verify run is gone from listing
    runs_after = asyncio.run(list_runs())
    run_names_after = [r["name"] for r in runs_after["runs"]]
    assert queued_run_name not in run_names_after

    # Verify job metadata is gone
    jobs_after = asyncio.run(list_jobs(limit=200, include_logs=False))
    job_ids_after = [j["job_id"] for j in jobs_after["jobs"]]
    assert queued_job_id not in job_ids_after

    gate.set()


def test_delete_run_not_found(tmp_path: Path):
    """DELETE /api/runs/{run_name} returns 404 for unknown run."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    app = create_app(runs_dir)
    delete_run = _get_route_endpoint(app, "/api/runs/{run_name}", method="DELETE")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(delete_run(run_name="nonexistent_run"))
    assert exc_info.value.status_code == 404


def test_delete_run_blocks_running_job(tmp_path: Path, monkeypatch):
    """DELETE /api/runs/{run_name} returns 409 if a pipeline job is running for that run."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    video_path = tmp_path / "blocked.mp4"
    video_path.write_bytes(b"video")

    gate = threading.Event()

    def _fake_pipeline_run(self, video_path, output_dir, resume=False):
        gate.wait(timeout=4.0)
        return {}

    monkeypatch.setattr("src.pipeline.base.Pipeline.run", _fake_pipeline_run)

    app = create_app(runs_dir)
    queue_jobs = _get_route_endpoint(app, "/api/pipeline/jobs", method="POST")
    list_jobs = _get_route_endpoint(app, "/api/pipeline/jobs")
    delete_run = _get_route_endpoint(app, "/api/runs/{run_name}", method="DELETE")

    queued = asyncio.run(
        queue_jobs(
            body=QueuePipelineJobsBody(
                video_paths=[str(video_path)],
                run_name="blocked_run",
            )
        )
    )
    job_id = queued["jobs"][0]["job_id"]

    # Wait for running status
    _wait_for_job_status(list_jobs, job_id, "running")

    # Create the run dir
    (runs_dir / "blocked_run").mkdir(parents=True, exist_ok=True)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(delete_run(run_name="blocked_run"))
    assert exc_info.value.status_code == 409

    gate.set()
