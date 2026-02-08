"""Cross-match reporting helpers for season trends and report templates."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.analytics.season import build_season_analytics, resolve_match_result


def _cfg_value(config: Any, key: str, default: Any) -> Any:
    """Read config key from object or dict with fallback."""
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely cast to float with fallback."""
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int | None = 0) -> int | None:
    """Safely cast to int with fallback."""
    try:
        return int(value)
    except Exception:
        return default


def _load_json(path: Path) -> dict[str, Any]:
    """Load JSON from path or return empty dict."""
    if not path.exists():
        return {}
    try:
        import json

        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def _extract_timestamp(run_dir: Path, summary: dict[str, Any], manifest: dict[str, Any]) -> str:
    """Resolve best-effort run timestamp."""
    for key in ("generated_at", "end_time", "start_time"):
        value = summary.get(key)
        if isinstance(value, str) and value:
            return value

    for key in ("end_time", "start_time"):
        value = manifest.get(key)
        if isinstance(value, str) and value:
            return value

    try:
        stat = run_dir.stat()
        return datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()


def _normalize_text(value: Any) -> str | None:
    """Normalize scalar/list metadata values to a single string."""
    if value is None:
        return None
    if isinstance(value, list | tuple):
        for row in value:
            normalized = _normalize_text(row)
            if normalized:
                return normalized
        return None
    text = str(value).strip()
    return text or None


def _nested_dict_get(payload: dict[str, Any], dotted_key: str) -> Any:
    """Read dotted-key values from nested dictionaries."""
    current: Any = payload
    for key in dotted_key.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _collect_metadata_sources(
    summary: dict[str, Any],
    manifest: dict[str, Any],
    match_metadata: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Collect candidate metadata dictionaries with provenance."""
    sources: list[tuple[str, dict[str, Any]]] = []
    if match_metadata:
        sources.append(("match_metadata_json", match_metadata))
    manifest_metadata = manifest.get("match_metadata")
    if isinstance(manifest_metadata, dict):
        sources.append(("run_manifest", manifest_metadata))
    summary_metadata = summary.get("match_metadata")
    if isinstance(summary_metadata, dict):
        sources.append(("summary", summary_metadata))
    return sources


def _extract_metadata_value(
    sources: list[tuple[str, dict[str, Any]]],
    keys: list[str],
) -> tuple[Any, str | None]:
    """Resolve first available metadata value from source/key candidates."""
    for source_name, payload in sources:
        for key in keys:
            value = _nested_dict_get(payload, key)
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            if isinstance(value, list | tuple) and not value:
                continue
            return value, source_name
    return None, None


def _extract_home_is_ours(sources: list[tuple[str, dict[str, Any]]]) -> bool | None:
    """Resolve whether the home score should map to 'ours'."""
    raw_value, _ = _extract_metadata_value(
        sources,
        [
            "home_is_ours",
            "ours_is_home",
            "our_team_is_home",
            "homeTeamIsOurs",
        ],
    )
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        token = raw_value.strip().lower()
        if token in {"true", "1", "yes", "y"}:
            return True
        if token in {"false", "0", "no", "n"}:
            return False
    return None


def _extract_goals_from_score_payload(
    score_payload: dict[str, Any],
    *,
    home_is_ours: bool | None,
) -> tuple[int | None, int | None, str | None]:
    """Extract ours/opponent goals from score payload with key variants."""
    if not isinstance(score_payload, dict):
        return None, None, None

    pairs = [
        ("ours", "opponent", "ours_opponent"),
        ("our", "opponent", "our_opponent"),
        ("our_team", "opponent_team", "our_team_opponent_team"),
        ("team_ours", "team_opponent", "team_ours_team_opponent"),
    ]

    for ours_key, opponent_key, source in pairs:
        ours = _safe_int(score_payload.get(ours_key), default=None)
        opponent = _safe_int(score_payload.get(opponent_key), default=None)
        if ours is not None and opponent is not None:
            return ours, opponent, source

    home = _safe_int(score_payload.get("home"), default=None)
    away = _safe_int(score_payload.get("away"), default=None)
    if home is not None and away is not None:
        if home_is_ours is False:
            return away, home, "home_away_swapped"
        return home, away, "home_away"

    team_a = _safe_int(score_payload.get("team_a"), default=None)
    team_b = _safe_int(score_payload.get("team_b"), default=None)
    if team_a is not None and team_b is not None:
        return team_a, team_b, "team_a_team_b"

    return None, None, None


def _extract_score_payload(
    summary: dict[str, Any],
    manifest: dict[str, Any],
    timeline: dict[str, Any],
    metadata_sources: list[tuple[str, dict[str, Any]]],
) -> tuple[dict[str, Any] | None, str | None]:
    """Resolve final-score payload with provenance."""
    metadata_score, metadata_source = _extract_metadata_value(
        metadata_sources,
        ["final_score", "score"],
    )
    if isinstance(metadata_score, dict):
        return metadata_score, metadata_source

    timeline_score = timeline.get("final_score")
    if isinstance(timeline_score, dict):
        return timeline_score, "score_timeline"

    summary_score = summary.get("final_score")
    if isinstance(summary_score, dict):
        return summary_score, "summary"
    summary_score = summary.get("score")
    if isinstance(summary_score, dict):
        return summary_score, "summary"

    manifest_score = manifest.get("final_score")
    if isinstance(manifest_score, dict):
        return manifest_score, "run_manifest"
    manifest_score = manifest.get("score")
    if isinstance(manifest_score, dict):
        return manifest_score, "run_manifest"

    return None, None


def _parse_run_record(run_dir: Path) -> dict[str, Any] | None:
    """Parse one run directory into normalized cross-match metrics."""
    summary = _load_json(run_dir / "summary.json")
    manifest = _load_json(run_dir / "run_manifest.json")
    timeline = _load_json(run_dir / "score_timeline.json")
    match_metadata = _load_json(run_dir / "match_metadata.json")
    team_analytics = _load_json(run_dir / "team_analytics.json")
    player_highlights = _load_json(run_dir / "player_highlights.json")
    if (
        not summary
        and not manifest
        and not timeline
        and not team_analytics
        and not player_highlights
        and not match_metadata
    ):
        return None

    counts = summary.get("counts")
    counts = counts if isinstance(counts, dict) else {}

    goals = _safe_int(counts.get("goals"), default=0)
    shots = _safe_int(counts.get("shots"), default=0)
    highlights_segments = _safe_int(counts.get("highlights_segments"), default=0)
    players_with_reels = _safe_int(counts.get("players_with_reels"), default=0)
    player_reel_segments_total = _safe_int(counts.get("player_reel_segments_total"), default=0)
    passes_inferred = _safe_int(counts.get("passes_inferred"), default=0)
    possession_frames = _safe_int(counts.get("possession_frames"), default=0)

    if goals == 0 and timeline:
        goals = _safe_int(timeline.get("goals"), default=0)

    team_summary = team_analytics.get("summary")
    team_summary = team_summary if isinstance(team_summary, dict) else {}
    if passes_inferred == 0:
        passes_inferred = _safe_int(team_summary.get("passes_inferred"), default=0)
    if possession_frames == 0:
        possession_frames = _safe_int(team_summary.get("frames_with_possession"), default=0)

    possession = team_analytics.get("possession")
    possession = possession if isinstance(possession, dict) else {}
    possession_teams = possession.get("teams")
    possession_teams = possession_teams if isinstance(possession_teams, dict) else {}
    dominant_team = possession.get("dominant_team")
    dominant_team = dominant_team if isinstance(dominant_team, str) else None

    pressing = team_analytics.get("pressing")
    pressing = pressing if isinstance(pressing, dict) else {}
    pressing_teams = pressing.get("teams")
    pressing_teams = pressing_teams if isinstance(pressing_teams, dict) else {}

    pass_network = team_analytics.get("pass_network")
    pass_network = pass_network if isinstance(pass_network, dict) else {}
    pass_network_teams = pass_network.get("teams")
    pass_network_teams = pass_network_teams if isinstance(pass_network_teams, dict) else {}

    metadata_sources = _collect_metadata_sources(
        summary=summary,
        manifest=manifest,
        match_metadata=match_metadata,
    )
    match_type, _ = _extract_metadata_value(
        metadata_sources,
        ["match_type", "type", "game_type"],
    )
    formation, _ = _extract_metadata_value(
        metadata_sources,
        [
            "formation",
            "starting_formation",
            "formations.ours",
            "team_formations.ours",
            "lineup.formation",
        ],
    )
    explicit_result, explicit_result_source = _extract_metadata_value(
        metadata_sources,
        ["result", "outcome", "wld"],
    )

    home_is_ours = _extract_home_is_ours(metadata_sources)
    score_payload, score_source = _extract_score_payload(
        summary=summary,
        manifest=manifest,
        timeline=timeline,
        metadata_sources=metadata_sources,
    )
    ours_goals: int | None = None
    opponent_goals: int | None = None
    if score_payload is not None:
        ours_goals, opponent_goals, score_key_source = _extract_goals_from_score_payload(
            score_payload=score_payload,
            home_is_ours=home_is_ours,
        )
        if score_key_source is not None:
            score_source = f"{score_source}:{score_key_source}" if score_source else score_key_source

    if ours_goals is None or opponent_goals is None:
        metadata_ours_goals, _ = _extract_metadata_value(
            metadata_sources,
            ["ours_goals", "our_goals", "goals_for"],
        )
        metadata_opponent_goals, _ = _extract_metadata_value(
            metadata_sources,
            ["opponent_goals", "goals_against"],
        )
        ours_goals = _safe_int(metadata_ours_goals, default=ours_goals)
        opponent_goals = _safe_int(metadata_opponent_goals, default=opponent_goals)

    result, ours_goals, opponent_goals = resolve_match_result(
        ours_goals=ours_goals,
        opponent_goals=opponent_goals,
        explicit_result=explicit_result,
    )
    result_source = None
    if result is not None and score_source is not None:
        result_source = score_source
    elif result is not None and explicit_result_source is not None:
        result_source = f"{explicit_result_source}:result"

    if goals == 0 and ours_goals is not None and opponent_goals is not None:
        goals = max(0, ours_goals + opponent_goals)

    players_raw = player_highlights.get("players")
    players_raw = players_raw if isinstance(players_raw, list) else []
    players: list[dict[str, Any]] = []
    for player in players_raw:
        if not isinstance(player, dict):
            continue
        player_id = player.get("player_id")
        if player_id is None:
            continue
        try:
            player_id = int(player_id)
        except (TypeError, ValueError):
            continue

        segments_raw = player.get("segments")
        segments_raw = segments_raw if isinstance(segments_raw, list) else []
        segments = [segment for segment in segments_raw if isinstance(segment, dict)]
        players.append(
            {
                "player_id": player_id,
                "player_name": player.get("player_name"),
                "segments": segments,
                "segment_count": _safe_int(player.get("segment_count"), default=len(segments)),
            }
        )

    timestamp = _extract_timestamp(run_dir=run_dir, summary=summary, manifest=manifest)
    run_name = run_dir.name
    return {
        "run_name": run_name,
        "timestamp": timestamp,
        "goals": goals,
        "shots": shots,
        "highlights_segments": highlights_segments,
        "players_with_reels": players_with_reels,
        "player_reel_segments_total": player_reel_segments_total,
        "passes_inferred": passes_inferred,
        "possession_frames": possession_frames,
        "possession_teams": possession_teams,
        "dominant_team": dominant_team,
        "pressing_teams": pressing_teams,
        "pass_network_teams": pass_network_teams,
        "match_type": _normalize_text(match_type),
        "formation": _normalize_text(formation),
        "ours_goals": ours_goals,
        "opponent_goals": opponent_goals,
        "result": result,
        "result_source": result_source,
        "score_source": score_source,
        "players": players,
    }


def _to_float_map(value: Any, key: str) -> dict[str, float]:
    """Convert nested team dict payloads into team->float map."""
    if not isinstance(value, dict):
        return {}
    output: dict[str, float] = {}
    for team, payload in value.items():
        if not isinstance(team, str):
            continue
        if not isinstance(payload, dict):
            continue
        output[team] = _safe_float(payload.get(key), default=0.0)
    return output


def build_cross_match_report(
    runs_root: Path,
    current_run: Path,
    config: Any,
) -> dict[str, Any]:
    """
    Build cross-match season trends and coach/player report templates.

    Returns:
        {
          "report": dict,
          "match_rows": list[dict],
          "player_rows": list[dict],
          "coach_template": str,
          "player_templates": str,
          "summary": dict,
        }
    """
    include_current_run = bool(_cfg_value(config, "include_current_run", True))
    max_runs = max(1, int(_cfg_value(config, "max_runs", 60)))
    top_players = max(1, int(_cfg_value(config, "top_players", 15)))
    min_player_segment_score = float(_cfg_value(config, "min_player_segment_score", 0.25))
    last_n_window = max(1, int(_cfg_value(config, "last_n_window", 5)))
    match_type_filter = _cfg_value(config, "match_type_filter", [])
    formation_filter = _cfg_value(config, "formation_filter", [])

    run_dirs = []
    if runs_root.exists():
        for candidate in sorted(runs_root.iterdir(), key=lambda path: path.name):
            if not candidate.is_dir():
                continue
            if candidate.name.startswith("."):
                continue
            if not include_current_run and candidate.resolve() == current_run.resolve():
                continue
            run_dirs.append(candidate)

    run_records = []
    for run_dir in run_dirs:
        parsed = _parse_run_record(run_dir)
        if parsed is not None:
            run_records.append(parsed)

    run_records.sort(key=lambda row: (str(row.get("timestamp", "")), str(row.get("run_name", ""))))
    if len(run_records) > max_runs:
        run_records = run_records[-max_runs:]

    matches_available_before_filters = len(run_records)
    season_payload = build_season_analytics(
        run_records=run_records,
        last_n_window=last_n_window,
        match_type_filter=match_type_filter,
        formation_filter=formation_filter,
    )
    run_records = list(season_payload.get("run_records", []) or [])
    season_filters = season_payload.get("filters", {})
    result_tracking = season_payload.get("result_tracking", {})
    possession_trend = season_payload.get("possession_trend", {})
    radar_ready_aggregates = season_payload.get("radar_ready_aggregates", {})

    matches_analyzed = len(run_records)
    total_goals = sum(row["goals"] for row in run_records)
    total_shots = sum(row["shots"] for row in run_records)
    total_highlights = sum(row["highlights_segments"] for row in run_records)
    total_passes = sum(row["passes_inferred"] for row in run_records)
    total_player_segments = sum(row["player_reel_segments_total"] for row in run_records)
    avg_goals = (total_goals / matches_analyzed) if matches_analyzed > 0 else 0.0
    avg_shots = (total_shots / matches_analyzed) if matches_analyzed > 0 else 0.0
    avg_highlights = (total_highlights / matches_analyzed) if matches_analyzed > 0 else 0.0
    avg_passes = (total_passes / matches_analyzed) if matches_analyzed > 0 else 0.0

    possession_acc: dict[str, list[float]] = defaultdict(list)
    pressing_acc: dict[str, list[float]] = defaultdict(list)
    pass_acc: dict[str, list[int]] = defaultdict(list)

    match_rows: list[dict[str, Any]] = []
    for row in run_records:
        possession_map = _to_float_map(row.get("possession_teams"), "share")
        pressing_map = _to_float_map(row.get("pressing_teams"), "high_press_rate")
        pass_team_map: dict[str, int] = {}
        if isinstance(row.get("pass_network_teams"), dict):
            for team, payload in row["pass_network_teams"].items():
                if not isinstance(team, str) or not isinstance(payload, dict):
                    continue
                pass_team_map[team] = _safe_int(payload.get("passes"), default=0)

        for team, share in possession_map.items():
            possession_acc[team].append(share)
        for team, high_rate in pressing_map.items():
            pressing_acc[team].append(high_rate)
        for team, passes in pass_team_map.items():
            pass_acc[team].append(passes)

        match_rows.append(
            {
                "schema_version": "1.0",
                "run_name": row["run_name"],
                "timestamp": row["timestamp"],
                "goals": row["goals"],
                "shots": row["shots"],
                "highlights_segments": row["highlights_segments"],
                "players_with_reels": row["players_with_reels"],
                "player_reel_segments_total": row["player_reel_segments_total"],
                "passes_inferred": row["passes_inferred"],
                "possession_frames": row["possession_frames"],
                "dominant_team": row["dominant_team"],
                "possession_share_ours": possession_map.get("ours"),
                "possession_share_opponent": possession_map.get("opponent"),
                "high_press_rate_ours": pressing_map.get("ours"),
                "high_press_rate_opponent": pressing_map.get("opponent"),
                "match_type": row.get("match_type"),
                "formation": row.get("formation"),
                "ours_goals": row.get("ours_goals"),
                "opponent_goals": row.get("opponent_goals"),
                "result": row.get("result"),
                "result_source": row.get("result_source"),
            }
        )

    team_labels = sorted(set(possession_acc.keys()) | set(pressing_acc.keys()) | set(pass_acc.keys()))
    team_trends = {}
    for team in team_labels:
        poss = possession_acc.get(team, [])
        press = pressing_acc.get(team, [])
        passes = pass_acc.get(team, [])
        team_trends[team] = {
            "matches_seen": max(len(poss), len(press), len(passes)),
            "avg_possession_share": (sum(poss) / len(poss)) if poss else None,
            "avg_high_press_rate": (sum(press) / len(press)) if press else None,
            "avg_passes_inferred": (sum(passes) / len(passes)) if passes else None,
        }

    player_acc: dict[int, dict[str, Any]] = {}
    for row in run_records:
        for player in row["players"]:
            player_id = int(player["player_id"])
            entry = player_acc.setdefault(
                player_id,
                {
                    "player_id": player_id,
                    "player_name": player.get("player_name"),
                    "matches_with_reels": 0,
                    "total_segments": 0,
                    "total_highlight_seconds": 0.0,
                    "segment_score_sum": 0.0,
                    "segment_score_count": 0,
                    "best_segment_score": 0.0,
                    "goal_tagged_segments": 0,
                    "shot_tagged_segments": 0,
                },
            )
            if entry.get("player_name") is None and player.get("player_name"):
                entry["player_name"] = player.get("player_name")

            segments = player.get("segments") or []
            if segments:
                entry["matches_with_reels"] += 1

            for segment in segments:
                if not isinstance(segment, dict):
                    continue
                score = _safe_float(segment.get("player_segment_score"), default=0.0)
                if score < min_player_segment_score:
                    continue

                duration = _safe_float(segment.get("duration"), default=0.0)
                entry["total_segments"] += 1
                entry["total_highlight_seconds"] += max(0.0, duration)
                entry["segment_score_sum"] += score
                entry["segment_score_count"] += 1
                if score > entry["best_segment_score"]:
                    entry["best_segment_score"] = score

                reasons = segment.get("reasons")
                reasons = reasons if isinstance(reasons, list) else []
                reasons_text = " ".join(str(reason).lower() for reason in reasons)
                if "goal" in reasons_text:
                    entry["goal_tagged_segments"] += 1
                if "shot" in reasons_text:
                    entry["shot_tagged_segments"] += 1

    player_rows: list[dict[str, Any]] = []
    for player_id, entry in player_acc.items():
        score_count = max(1, int(entry["segment_score_count"]))
        avg_segment_score = float(entry["segment_score_sum"]) / score_count
        player_rows.append(
            {
                "schema_version": "1.0",
                "player_id": player_id,
                "player_name": entry.get("player_name"),
                "matches_with_reels": int(entry["matches_with_reels"]),
                "total_segments": int(entry["total_segments"]),
                "total_highlight_seconds": float(entry["total_highlight_seconds"]),
                "avg_segment_score": avg_segment_score,
                "best_segment_score": float(entry["best_segment_score"]),
                "goal_tagged_segments": int(entry["goal_tagged_segments"]),
                "shot_tagged_segments": int(entry["shot_tagged_segments"]),
            }
        )

    player_rows.sort(
        key=lambda row: (
            -int(row["total_segments"]),
            -float(row["best_segment_score"]),
            int(row["player_id"]),
        )
    )

    total_segment_pool = max(1, sum(int(row["total_segments"]) for row in player_rows))
    for row in player_rows:
        row["share_of_all_segments"] = float(row["total_segments"]) / total_segment_pool

    top_player_rows = player_rows[:top_players]

    goals_last_window = [int(row["goals"]) for row in run_records[-last_n_window:]]
    shots_last_window = [int(row["shots"]) for row in run_records[-last_n_window:]]
    highlights_last_window = [int(row["highlights_segments"]) for row in run_records[-last_n_window:]]

    generated_at = datetime.now(timezone.utc).isoformat()
    report = {
        "schema_version": "1.0",
        "generated_at": generated_at,
        "runs_root": str(runs_root),
        "summary": {
            "matches_available_before_filters": matches_available_before_filters,
            "matches_analyzed": matches_analyzed,
            "unique_players": len(player_rows),
            "player_segment_pool": total_segment_pool if player_rows else 0,
            "time_window_runs": last_n_window,
        },
        "filters": season_filters,
        "season_trends": {
            "match_aggregates": {
                "goals_total": total_goals,
                "shots_total": total_shots,
                "highlights_total": total_highlights,
                "passes_inferred_total": total_passes,
                "player_reel_segments_total": total_player_segments,
                "goals_per_match": avg_goals,
                "shots_per_match": avg_shots,
                "highlights_per_match": avg_highlights,
                "passes_inferred_per_match": avg_passes,
            },
            "team_trends": team_trends,
            "window": {
                "last_n": last_n_window,
                "goals": goals_last_window,
                "shots": shots_last_window,
                "highlights": highlights_last_window,
            },
            "result_tracking": result_tracking,
            "possession_trend": possession_trend,
            "radar_ready_aggregates": radar_ready_aggregates,
        },
        "matches": match_rows,
        "players": {
            "top_players": top_player_rows,
            "all_players_count": len(player_rows),
        },
    }

    coach_template_lines = [
        "# Coach Report Template",
        "",
        f"Schema Version: {report['schema_version']}",
        f"Generated At: {generated_at}",
        "",
        "## Season Snapshot",
        f"- Matches analyzed: {matches_analyzed}",
        f"- Matches available before filters: {matches_available_before_filters}",
        f"- W/L/D: {int(result_tracking.get('wins', 0))}/"
        f"{int(result_tracking.get('losses', 0))}/"
        f"{int(result_tracking.get('draws', 0))}",
        f"- Goals per match: {avg_goals:.2f}",
        f"- Shots per match: {avg_shots:.2f}",
        f"- Highlight segments per match: {avg_highlights:.2f}",
        f"- Inferred passes per match: {avg_passes:.2f}",
        "",
        "## Active Filters",
        f"- Match type filter: {season_filters.get('match_type_filter', [])}",
        f"- Formation filter: {season_filters.get('formation_filter', [])}",
        "",
        "## Team Trend Prompts",
        "- Possession: Which match contexts drove possession swings?",
        "- Territory: Which zones were consistently won or lost?",
        "- Pass Network: Which links are stable vs fragile under pressure?",
        "- Pressing: When did high press create turnovers or chances?",
        "",
        "## Last Window Check",
        f"- Goals trend (last {last_n_window}): {goals_last_window}",
        f"- Shots trend (last {last_n_window}): {shots_last_window}",
        f"- Highlights trend (last {last_n_window}): {highlights_last_window}",
        f"- Possession trend ours (last {last_n_window}): {possession_trend.get('ours', [])}",
        "",
        "## Match Review Template",
        "1. Match Context:",
        "2. Positive Patterns:",
        "3. Risk Patterns:",
        "4. Training Focus for Next Week:",
    ]
    coach_template = "\n".join(coach_template_lines) + "\n"

    player_template_lines = [
        "# Player Report Templates",
        "",
        f"Schema Version: {report['schema_version']}",
        f"Generated At: {generated_at}",
        "",
        "## How To Use",
        "- Use these prompts to finalize individual player reports.",
        "- Cross-check with `player_highlights.json` for clip-level evidence.",
        "",
    ]
    for index, row in enumerate(top_player_rows, start=1):
        player_heading = row["player_name"] if row.get("player_name") else f"Player {row['player_id']}"
        player_template_lines.extend(
            [
                f"## Player {index}: {player_heading}",
                f"- Player ID: {row['player_id']}",
                f"- Matches with reels: {row['matches_with_reels']}",
                f"- Total segments: {row['total_segments']}",
                f"- Total highlight seconds: {row['total_highlight_seconds']:.1f}",
                f"- Average segment score: {row['avg_segment_score']:.3f}",
                f"- Best segment score: {row['best_segment_score']:.3f}",
                f"- Goal-tagged segments: {row['goal_tagged_segments']}",
                f"- Shot-tagged segments: {row['shot_tagged_segments']}",
                "- Notes:",
                "- Development Focus:",
                "",
            ]
        )
    player_templates = "\n".join(player_template_lines).rstrip() + "\n"

    return {
        "report": report,
        "match_rows": match_rows,
        "player_rows": player_rows,
        "coach_template": coach_template,
        "player_templates": player_templates,
        "summary": report["summary"],
    }
