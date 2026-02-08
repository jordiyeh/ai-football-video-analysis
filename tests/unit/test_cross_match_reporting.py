"""Tests for cross-match reporting/export helpers."""

from __future__ import annotations

import json
from pathlib import Path

from src.export.cross_match import build_cross_match_report


def _write_json(path: Path, payload: dict) -> None:
    """Write JSON payload to path, creating parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _seed_run(
    run_path: Path,
    *,
    goals: int,
    shots: int,
    highlights: int,
    passes: int,
    possession_ours: float,
    possession_opponent: float,
    high_press_ours: float,
    high_press_opponent: float,
    players: list[dict],
    match_type: str | None = None,
    formation: str | None = None,
    ours_goals: int | None = None,
    opponent_goals: int | None = None,
    score_style: str = "ours_opponent",
) -> None:
    """Create minimal run artifacts used by cross-match reporting."""
    _write_json(
        run_path / "summary.json",
        {
            "schema_version": "1.0",
            "generated_at": f"2026-01-{run_path.name[-1]}T10:00:00Z",
            "counts": {
                "goals": goals,
                "shots": shots,
                "highlights_segments": highlights,
                "passes_inferred": passes,
                "players_with_reels": len(players),
                "player_reel_segments_total": sum(len(player.get("segments", [])) for player in players),
            },
        },
    )
    _write_json(
        run_path / "run_manifest.json",
        {
            "schema_version": "1.1",
            "end_time": f"2026-01-{run_path.name[-1]}T11:30:00Z",
        },
    )
    _write_json(
        run_path / "team_analytics.json",
        {
            "schema_version": "1.0",
            "summary": {
                "passes_inferred": passes,
                "frames_with_possession": 120,
            },
            "possession": {
                "dominant_team": "ours" if possession_ours >= possession_opponent else "opponent",
                "teams": {
                    "ours": {"share": possession_ours},
                    "opponent": {"share": possession_opponent},
                },
            },
            "pressing": {
                "teams": {
                    "ours": {"high_press_rate": high_press_ours},
                    "opponent": {"high_press_rate": high_press_opponent},
                }
            },
            "pass_network": {
                "teams": {
                    "ours": {"passes": int(round(passes * 0.58))},
                    "opponent": {"passes": int(round(passes * 0.42))},
                }
            },
        },
    )
    metadata_payload: dict[str, object] = {}
    if match_type is not None:
        metadata_payload["match_type"] = match_type
    if formation is not None:
        metadata_payload["formation"] = formation
    if metadata_payload:
        _write_json(run_path / "match_metadata.json", metadata_payload)

    if ours_goals is not None and opponent_goals is not None:
        if score_style == "team_a_team_b":
            final_score = {"team_a": ours_goals, "team_b": opponent_goals}
        elif score_style == "home_away":
            final_score = {"home": ours_goals, "away": opponent_goals}
        else:
            final_score = {"ours": ours_goals, "opponent": opponent_goals}
        _write_json(
            run_path / "score_timeline.json",
            {
                "goals": int(ours_goals + opponent_goals),
                "final_score": final_score,
                "timeline": [],
            },
        )

    _write_json(
        run_path / "player_highlights.json",
        {
            "schema_version": "1.0",
            "video_id": run_path.name,
            "players": players,
            "summary": {
                "players_with_reels": len(players),
                "player_segments_total": sum(len(player.get("segments", [])) for player in players),
            },
        },
    )


def test_build_cross_match_report_aggregates_matches_and_players(tmp_path: Path):
    """Cross-match report should aggregate season trends and top-player stats."""
    runs_root = tmp_path / "runs"

    _seed_run(
        runs_root / "match1",
        goals=2,
        shots=8,
        highlights=5,
        passes=54,
        possession_ours=0.57,
        possession_opponent=0.43,
        high_press_ours=0.48,
        high_press_opponent=0.33,
        players=[
            {
                "player_id": 10,
                "player_name": "Nicholas Oestringer",
                "segments": [
                    {
                        "segment_id": "seg_a",
                        "duration": 14.0,
                        "player_segment_score": 0.92,
                        "reasons": ["goal", "shot"],
                    },
                    {
                        "segment_id": "seg_b",
                        "duration": 11.0,
                        "player_segment_score": 0.70,
                        "reasons": ["shot"],
                    },
                ],
            }
        ],
    )
    _seed_run(
        runs_root / "match2",
        goals=1,
        shots=6,
        highlights=4,
        passes=49,
        possession_ours=0.52,
        possession_opponent=0.48,
        high_press_ours=0.44,
        high_press_opponent=0.37,
        players=[
            {
                "player_id": 10,
                "player_name": "Nicholas Oestringer",
                "segments": [
                    {
                        "segment_id": "seg_c",
                        "duration": 9.0,
                        "player_segment_score": 0.81,
                        "reasons": ["goal"],
                    }
                ],
            },
            {
                "player_id": 22,
                "player_name": "Player 22",
                "segments": [
                    {
                        "segment_id": "seg_d",
                        "duration": 8.5,
                        "player_segment_score": 0.60,
                        "reasons": ["shot"],
                    }
                ],
            },
        ],
    )

    payload = build_cross_match_report(
        runs_root=runs_root,
        current_run=runs_root / "match2",
        config={
            "include_current_run": True,
            "max_runs": 10,
            "top_players": 5,
            "min_player_segment_score": 0.2,
            "last_n_window": 5,
        },
    )

    report = payload["report"]
    summary = report["summary"]
    assert summary["matches_analyzed"] == 2
    assert summary["unique_players"] == 2

    aggregates = report["season_trends"]["match_aggregates"]
    assert aggregates["goals_total"] == 3
    assert aggregates["shots_total"] == 14
    assert round(float(aggregates["goals_per_match"]), 2) == 1.5

    team_trends = report["season_trends"]["team_trends"]
    assert "ours" in team_trends
    assert abs(float(team_trends["ours"]["avg_possession_share"]) - 0.545) < 1e-6

    player_rows = payload["player_rows"]
    top_row = player_rows[0]
    assert int(top_row["player_id"]) == 10
    assert int(top_row["total_segments"]) == 3
    assert int(top_row["goal_tagged_segments"]) == 2

    coach_template = payload["coach_template"]
    player_templates = payload["player_templates"]
    assert "Coach Report Template" in coach_template
    assert "Matches analyzed: 2" in coach_template
    assert "Player 1: Nicholas Oestringer" in player_templates


def test_build_cross_match_report_supports_filters_results_and_radar(tmp_path: Path):
    """Cross-match report should support W/L/D + metadata filters + radar aggregates."""
    runs_root = tmp_path / "runs"

    _seed_run(
        runs_root / "match1",
        goals=3,
        shots=10,
        highlights=6,
        passes=60,
        possession_ours=0.59,
        possession_opponent=0.41,
        high_press_ours=0.51,
        high_press_opponent=0.34,
        players=[],
        match_type="League",
        formation="4-3-3",
        ours_goals=3,
        opponent_goals=1,
        score_style="ours_opponent",
    )
    _seed_run(
        runs_root / "match2",
        goals=2,
        shots=7,
        highlights=4,
        passes=50,
        possession_ours=0.49,
        possession_opponent=0.51,
        high_press_ours=0.45,
        high_press_opponent=0.37,
        players=[],
        match_type="Friendly",
        formation="4-4-2",
        ours_goals=2,
        opponent_goals=2,
        score_style="home_away",
    )
    _seed_run(
        runs_root / "match3",
        goals=2,
        shots=5,
        highlights=3,
        passes=46,
        possession_ours=0.44,
        possession_opponent=0.56,
        high_press_ours=0.40,
        high_press_opponent=0.39,
        players=[],
        match_type="League",
        formation="4-3-3",
        ours_goals=0,
        opponent_goals=2,
        score_style="team_a_team_b",
    )

    payload = build_cross_match_report(
        runs_root=runs_root,
        current_run=runs_root / "match3",
        config={
            "include_current_run": True,
            "max_runs": 10,
            "top_players": 5,
            "min_player_segment_score": 0.2,
            "last_n_window": 2,
            "match_type_filter": ["league"],
            "formation_filter": ["4-3-3"],
        },
    )

    report = payload["report"]
    summary = report["summary"]
    assert summary["matches_available_before_filters"] == 3
    assert summary["matches_analyzed"] == 2

    filters = report["filters"]
    assert filters["match_type_filter"] == ["league"]
    assert filters["formation_filter"] == ["4-3-3"]
    assert filters["filtered_out_matches"] == 1

    result_tracking = report["season_trends"]["result_tracking"]
    assert result_tracking["wins"] == 1
    assert result_tracking["losses"] == 1
    assert result_tracking["draws"] == 0
    assert result_tracking["points"] == 3
    assert result_tracking["matches_with_result"] == 2

    possession_trend = report["season_trends"]["possession_trend"]
    assert possession_trend["last_n"] == 2
    assert len(possession_trend["series"]) == 2
    assert possession_trend["series"][0]["run_name"] == "match1"
    assert possession_trend["series"][1]["run_name"] == "match3"

    radar = report["season_trends"]["radar_ready_aggregates"]
    assert len(radar["metrics"]) >= 4
    assert "ours" in radar["teams"]
    assert "opponent" in radar["teams"]
    assert "normalized" in radar["teams"]["ours"]
    assert radar["teams"]["ours"]["normalized"]["possession_share"] is not None

    match_rows = payload["match_rows"]
    assert len(match_rows) == 2
    assert match_rows[0]["result"] == "win"
    assert match_rows[0]["match_type"] == "League"
    assert match_rows[0]["formation"] == "4-3-3"


def test_build_cross_match_report_handles_empty_runs_root(tmp_path: Path):
    """No runs should still produce a valid empty report payload."""
    payload = build_cross_match_report(
        runs_root=tmp_path / "runs",
        current_run=tmp_path / "runs" / "current",
        config={
            "include_current_run": True,
            "max_runs": 5,
            "top_players": 3,
            "min_player_segment_score": 0.5,
            "last_n_window": 3,
        },
    )

    report = payload["report"]
    assert report["summary"]["matches_analyzed"] == 0
    assert report["season_trends"]["match_aggregates"]["goals_total"] == 0
    assert payload["match_rows"] == []
    assert payload["player_rows"] == []
