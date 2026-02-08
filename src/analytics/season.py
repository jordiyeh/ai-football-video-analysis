"""Season analytics helpers for cross-match reporting."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


SEASON_ANALYTICS_SCHEMA_VERSION = "1.0"


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely cast to float with fallback."""
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int | None = None) -> int | None:
    """Safely cast to int with fallback."""
    try:
        return int(value)
    except Exception:
        return default


def _normalize_text(value: Any) -> str | None:
    """Normalize scalar/list metadata values to a single display string."""
    if value is None:
        return None
    if isinstance(value, list | tuple):
        for candidate in value:
            normalized = _normalize_text(candidate)
            if normalized:
                return normalized
        return None
    text = str(value).strip()
    return text or None


def _normalize_token(value: Any) -> str | None:
    """Normalize text to lowercase token for filtering."""
    text = _normalize_text(value)
    if text is None:
        return None
    return text.casefold()


def _normalize_filter_values(value: Any) -> list[str]:
    """Normalize filter input to lowercase string list."""
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list | tuple | set):
        values = list(value)
    else:
        values = [value]

    normalized: list[str] = []
    seen: set[str] = set()
    for row in values:
        token = _normalize_token(row)
        if token is None or token in seen:
            continue
        normalized.append(token)
        seen.add(token)
    return normalized


def resolve_match_result(
    ours_goals: Any,
    opponent_goals: Any,
    explicit_result: Any = None,
) -> tuple[str | None, int | None, int | None]:
    """Resolve match result from score values or explicit metadata."""
    ours = _safe_int(ours_goals, default=None)
    opponent = _safe_int(opponent_goals, default=None)
    if ours is not None and opponent is not None:
        if ours > opponent:
            return "win", ours, opponent
        if ours < opponent:
            return "loss", ours, opponent
        return "draw", ours, opponent

    token = _normalize_token(explicit_result)
    if token in {"w", "win", "won"}:
        return "win", ours, opponent
    if token in {"d", "draw", "tied", "tie"}:
        return "draw", ours, opponent
    if token in {"l", "loss", "lost"}:
        return "loss", ours, opponent

    return None, ours, opponent


def _extract_team_metric(payload: Any, key: str) -> dict[str, float]:
    """Extract team->float metric from nested team payloads."""
    if not isinstance(payload, dict):
        return {}
    output: dict[str, float] = {}
    for team, values in payload.items():
        if not isinstance(team, str) or not isinstance(values, dict):
            continue
        output[team] = _safe_float(values.get(key), default=0.0)
    return output


def _apply_filters(
    run_records: list[dict[str, Any]],
    match_type_filter: list[str],
    formation_filter: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply match-type and formation filters to run records."""
    match_type_tokens = set(_normalize_filter_values(match_type_filter))
    formation_tokens = set(_normalize_filter_values(formation_filter))

    filtered_records: list[dict[str, Any]] = []
    filtered_out_match_type = 0
    filtered_out_formation = 0
    filtered_out_total = 0

    available_match_types = sorted(
        {
            token
            for row in run_records
            if (token := _normalize_token(row.get("match_type"))) is not None
        }
    )
    available_formations = sorted(
        {
            token
            for row in run_records
            if (token := _normalize_token(row.get("formation"))) is not None
        }
    )

    for row in run_records:
        match_type_token = _normalize_token(row.get("match_type"))
        formation_token = _normalize_token(row.get("formation"))

        match_type_ok = not match_type_tokens or match_type_token in match_type_tokens
        formation_ok = not formation_tokens or formation_token in formation_tokens

        if match_type_ok and formation_ok:
            filtered_records.append(row)
            continue

        filtered_out_total += 1
        if not match_type_ok:
            filtered_out_match_type += 1
        if not formation_ok:
            filtered_out_formation += 1

    return filtered_records, {
        "schema_version": SEASON_ANALYTICS_SCHEMA_VERSION,
        "match_type_filter": sorted(match_type_tokens),
        "formation_filter": sorted(formation_tokens),
        "available_match_types": available_match_types,
        "available_formations": available_formations,
        "filtered_out_matches": filtered_out_total,
        "filtered_out_by_match_type": filtered_out_match_type,
        "filtered_out_by_formation": filtered_out_formation,
    }


def _build_result_tracking(run_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build W/L/D result tracking summary and per-match rows."""
    wins = 0
    draws = 0
    losses = 0
    unknown = 0
    result_rows: list[dict[str, Any]] = []

    for row in run_records:
        result = _normalize_token(row.get("result"))
        if result == "win":
            wins += 1
        elif result == "draw":
            draws += 1
        elif result == "loss":
            losses += 1
        else:
            result = None
            unknown += 1

        result_rows.append(
            {
                "schema_version": SEASON_ANALYTICS_SCHEMA_VERSION,
                "run_name": row.get("run_name"),
                "timestamp": row.get("timestamp"),
                "ours_goals": row.get("ours_goals"),
                "opponent_goals": row.get("opponent_goals"),
                "result": result or "unknown",
                "result_source": row.get("result_source"),
            }
        )

    matches_with_result = wins + draws + losses
    win_rate = float(wins) / matches_with_result if matches_with_result > 0 else None
    points = wins * 3 + draws

    return {
        "schema_version": SEASON_ANALYTICS_SCHEMA_VERSION,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "unknown": unknown,
        "matches_with_result": matches_with_result,
        "points": points,
        "points_per_match": (float(points) / matches_with_result) if matches_with_result > 0 else None,
        "win_rate": win_rate,
        "results": result_rows,
    }


def _build_possession_trend(
    run_records: list[dict[str, Any]],
    last_n_window: int,
) -> dict[str, Any]:
    """Build possession trend payload over the trailing window."""
    window_records = run_records[-last_n_window:] if last_n_window > 0 else []

    series: list[dict[str, Any]] = []
    ours_series: list[float | None] = []
    opponent_series: list[float | None] = []
    dominant_series: list[str | None] = []
    run_names: list[str] = []

    for row in window_records:
        possession_map = _extract_team_metric(row.get("possession_teams"), "share")
        ours_share = possession_map.get("ours")
        opponent_share = possession_map.get("opponent")
        dominant_team = row.get("dominant_team")
        dominant_team = str(dominant_team) if isinstance(dominant_team, str) else None

        series.append(
            {
                "schema_version": SEASON_ANALYTICS_SCHEMA_VERSION,
                "run_name": row.get("run_name"),
                "timestamp": row.get("timestamp"),
                "ours": ours_share,
                "opponent": opponent_share,
                "dominant_team": dominant_team,
            }
        )
        ours_series.append(ours_share)
        opponent_series.append(opponent_share)
        dominant_series.append(dominant_team)
        run_names.append(str(row.get("run_name") or ""))

    return {
        "schema_version": SEASON_ANALYTICS_SCHEMA_VERSION,
        "last_n": last_n_window,
        "series": series,
        "run_names": run_names,
        "ours": ours_series,
        "opponent": opponent_series,
        "dominant_team": dominant_series,
    }


def _average(values: list[float]) -> float | None:
    """Compute average for non-empty lists."""
    if not values:
        return None
    return float(sum(values)) / float(len(values))


def _build_radar_ready_aggregates(run_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build radar-ready team aggregates with normalized values."""
    possession_acc: dict[str, list[float]] = defaultdict(list)
    pressing_acc: dict[str, list[float]] = defaultdict(list)
    passes_acc: dict[str, list[float]] = defaultdict(list)
    dominant_acc: dict[str, int] = defaultdict(int)

    for row in run_records:
        possession_map = _extract_team_metric(row.get("possession_teams"), "share")
        pressing_map = _extract_team_metric(row.get("pressing_teams"), "high_press_rate")
        passes_map = _extract_team_metric(row.get("pass_network_teams"), "passes")

        for team, value in possession_map.items():
            possession_acc[team].append(value)
        for team, value in pressing_map.items():
            pressing_acc[team].append(value)
        for team, value in passes_map.items():
            passes_acc[team].append(value)

        dominant_team = row.get("dominant_team")
        if isinstance(dominant_team, str) and dominant_team:
            dominant_acc[dominant_team] += 1

    teams = sorted(
        set(possession_acc.keys())
        | set(pressing_acc.keys())
        | set(passes_acc.keys())
        | set(dominant_acc.keys())
    )

    max_passes = max(
        [1.0]
        + [
            value
            for values in passes_acc.values()
            for value in values
        ]
    )

    metric_specs = [
        {"id": "possession_share", "label": "Possession Share", "min": 0.0, "max": 1.0},
        {"id": "high_press_rate", "label": "High Press Rate", "min": 0.0, "max": 1.0},
        {"id": "passes_inferred", "label": "Inferred Passes", "min": 0.0, "max": max_passes},
        {"id": "dominant_match_share", "label": "Dominant Match Share", "min": 0.0, "max": 1.0},
    ]

    teams_payload: dict[str, dict[str, Any]] = {}
    total_matches = max(1, len(run_records))
    for team in teams:
        raw = {
            "possession_share": _average(possession_acc.get(team, [])),
            "high_press_rate": _average(pressing_acc.get(team, [])),
            "passes_inferred": _average(passes_acc.get(team, [])),
            "dominant_match_share": float(dominant_acc.get(team, 0)) / float(total_matches),
        }

        normalized: dict[str, float | None] = {}
        for spec in metric_specs:
            metric_id = str(spec["id"])
            value = raw.get(metric_id)
            if value is None:
                normalized[metric_id] = None
                continue
            metric_min = _safe_float(spec.get("min"), default=0.0)
            metric_max = _safe_float(spec.get("max"), default=1.0)
            if metric_max <= metric_min:
                normalized[metric_id] = 0.0
                continue
            normalized_value = (float(value) - metric_min) / (metric_max - metric_min)
            normalized[metric_id] = max(0.0, min(1.0, normalized_value))

        teams_payload[team] = {
            "raw": raw,
            "normalized": normalized,
            "matches_seen": max(
                len(possession_acc.get(team, [])),
                len(pressing_acc.get(team, [])),
                len(passes_acc.get(team, [])),
            ),
        }

    return {
        "schema_version": SEASON_ANALYTICS_SCHEMA_VERSION,
        "metrics": metric_specs,
        "teams": teams_payload,
    }


def build_season_analytics(
    run_records: list[dict[str, Any]],
    *,
    last_n_window: int,
    match_type_filter: Any = None,
    formation_filter: Any = None,
) -> dict[str, Any]:
    """Build season analytics payload and return filtered run records."""
    filtered_records, filter_summary = _apply_filters(
        run_records=run_records,
        match_type_filter=_normalize_filter_values(match_type_filter),
        formation_filter=_normalize_filter_values(formation_filter),
    )

    return {
        "schema_version": SEASON_ANALYTICS_SCHEMA_VERSION,
        "run_records": filtered_records,
        "filters": filter_summary,
        "result_tracking": _build_result_tracking(filtered_records),
        "possession_trend": _build_possession_trend(filtered_records, last_n_window=last_n_window),
        "radar_ready_aggregates": _build_radar_ready_aggregates(filtered_records),
    }
