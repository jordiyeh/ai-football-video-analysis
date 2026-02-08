"""Unified match stats aggregation for team-level event and possession totals."""

from __future__ import annotations

from collections import Counter
from math import hypot
from typing import Any


UNKNOWN_TEAM = "unknown"
MATCH_STATS_SCHEMA_VERSION = "1.0"
SET_PIECE_EVENT_TYPES = frozenset(
    {"set_piece", "kickoff", "throw_in", "corner_kick", "free_kick", "goal_kick"}
)


def _safe_int(value: Any, default: int | None = None) -> int | None:
    """Safely cast arbitrary values to int."""
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely cast arbitrary values to float."""
    try:
        return float(value)
    except Exception:
        return default


def _is_unknown_team(value: str | None) -> bool:
    """Return True when a team label is missing/unknown."""
    if value is None:
        return True
    lowered = value.strip().lower()
    return lowered in {"", UNKNOWN_TEAM, "-1", "none", "null"}


def normalize_team_label(raw: Any) -> str:
    """Normalize team labels into stable string keys."""
    if raw is None:
        return UNKNOWN_TEAM

    if isinstance(raw, bool):
        return UNKNOWN_TEAM

    if isinstance(raw, int):
        return f"team_{raw}" if raw >= 0 else UNKNOWN_TEAM

    text = str(raw).strip()
    if _is_unknown_team(text):
        return UNKNOWN_TEAM

    if text.isdigit():
        numeric = _safe_int(text, default=None)
        if numeric is None or numeric < 0:
            return UNKNOWN_TEAM
        return f"team_{numeric}"

    return text


def _resolve_track_team(track: dict[str, Any]) -> str:
    """Resolve canonical team label from a player track row."""
    team_name = normalize_team_label(track.get("team_name"))
    if not _is_unknown_team(team_name):
        return team_name

    return normalize_team_label(track.get("team_id"))


def _center_from_bbox(track: dict[str, Any]) -> tuple[float, float] | None:
    """Extract bbox center from [x1, y1, x2, y2] boxes."""
    bbox = track.get("bbox")
    if not isinstance(bbox, list | tuple) or len(bbox) < 4:
        return None

    x1 = _safe_float(bbox[0], default=float("nan"))
    y1 = _safe_float(bbox[1], default=float("nan"))
    x2 = _safe_float(bbox[2], default=float("nan"))
    y2 = _safe_float(bbox[3], default=float("nan"))
    if x2 <= x1 or y2 <= y1:
        return None
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _resolve_track_xy(track: dict[str, Any]) -> tuple[float, float] | None:
    """Resolve image-space location from explicit fields or bbox fallback."""
    image_x = track.get("image_x")
    image_y = track.get("image_y")
    if image_x is not None and image_y is not None:
        return (_safe_float(image_x), _safe_float(image_y))

    image_xy = track.get("image_xy")
    if isinstance(image_xy, list | tuple) and len(image_xy) >= 2:
        return (_safe_float(image_xy[0]), _safe_float(image_xy[1]))

    return _center_from_bbox(track)


def _resolve_event_xy(event: dict[str, Any]) -> tuple[float, float] | None:
    """Resolve event location from event payload."""
    location = event.get("location")
    if isinstance(location, list | tuple) and len(location) >= 2:
        return (_safe_float(location[0]), _safe_float(location[1]))
    return None


def _event_to_dict(event: Any) -> dict[str, Any]:
    """Normalize dict/dataclass-like events to a dictionary."""
    if isinstance(event, dict):
        event_dict = dict(event)
    else:
        event_dict = {
            "event_type": getattr(event, "event_type", None),
            "frame_idx": getattr(event, "frame_idx", None),
            "timestamp": getattr(event, "timestamp", None),
            "confidence": getattr(event, "confidence", None),
            "location": getattr(event, "location", None),
            "metadata": getattr(event, "metadata", None),
        }

    metadata = event_dict.get("metadata")
    if not isinstance(metadata, dict):
        event_dict["metadata"] = {}

    return event_dict


def _team_sort_key(label: str) -> tuple[int, str]:
    """Sort team labels with ours/opponent first and unknown last."""
    lowered = label.lower()
    if lowered == "ours":
        return (0, lowered)
    if lowered == "opponent":
        return (1, lowered)
    if lowered == UNKNOWN_TEAM:
        return (99, lowered)
    return (2, lowered)


def _extract_team_from_metadata(metadata: dict[str, Any]) -> tuple[str, str]:
    """Resolve team from common metadata keys."""
    for key in (
        "team_id",
        "team_name",
        "owner_team",
        "attacking_team",
        "defending_team",
        "scoring_team",
    ):
        team = normalize_team_label(metadata.get(key))
        if team != UNKNOWN_TEAM:
            return team, "metadata"
    return UNKNOWN_TEAM, "unresolved"


def _build_track_indexes(
    tracks: list[dict[str, Any]],
) -> tuple[dict[int, str], dict[int, list[dict[str, Any]]]]:
    """Build lookup indexes for team attribution fallbacks."""
    votes_by_track_id: dict[int, Counter[str]] = {}
    players_by_frame: dict[int, list[dict[str, Any]]] = {}

    for track in tracks:
        if str(track.get("object_type", "")).strip().lower() != "player":
            continue

        track_id = _safe_int(track.get("track_id"), default=None)
        frame_idx = _safe_int(track.get("frame_idx"), default=None)
        if track_id is None or frame_idx is None:
            continue

        team = _resolve_track_team(track)
        xy = _resolve_track_xy(track)

        votes = votes_by_track_id.setdefault(track_id, Counter())
        votes[team] += 1

        if xy is None:
            continue
        players_by_frame.setdefault(frame_idx, []).append(
            {
                "track_id": track_id,
                "team": team,
                "xy": xy,
            }
        )

    team_by_track_id: dict[int, str] = {}
    for track_id, votes in votes_by_track_id.items():
        if not votes:
            continue
        known_votes = Counter({team: count for team, count in votes.items() if team != UNKNOWN_TEAM})
        if known_votes:
            team_by_track_id[track_id] = known_votes.most_common(1)[0][0]
        else:
            team_by_track_id[track_id] = UNKNOWN_TEAM

    return team_by_track_id, players_by_frame


def _nearest_player_team(
    frame_idx: int | None,
    xy: tuple[float, float] | None,
    players_by_frame: dict[int, list[dict[str, Any]]],
    frame_window: int,
) -> str:
    """Resolve team by nearest player around the event frame."""
    if frame_idx is None or xy is None:
        return UNKNOWN_TEAM

    best_team = UNKNOWN_TEAM
    best_distance = None
    for offset in range(-frame_window, frame_window + 1):
        rows = players_by_frame.get(frame_idx + offset, [])
        for row in rows:
            distance = hypot(xy[0] - row["xy"][0], xy[1] - row["xy"][1])
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_team = normalize_team_label(row["team"])

    return best_team


def _resolve_event_team(
    event: dict[str, Any],
    shot_team_by_frame: dict[int, str],
    team_by_track_id: dict[int, str],
    players_by_frame: dict[int, list[dict[str, Any]]],
    frame_window: int,
) -> tuple[str, str]:
    """Resolve event team using metadata, shot links, and track proximity fallbacks."""
    metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}

    team, method = _extract_team_from_metadata(metadata)
    if team != UNKNOWN_TEAM:
        return team, method

    event_type = str(event.get("event_type", ""))
    frame_idx = _safe_int(event.get("frame_idx"), default=None)

    if event_type == "goal":
        shot_frame = _safe_int(metadata.get("shot_frame"), default=None)
        if shot_frame is not None:
            linked_team = normalize_team_label(shot_team_by_frame.get(shot_frame))
            if linked_team != UNKNOWN_TEAM:
                return linked_team, "shot_link"

    kick_player_id = _safe_int(metadata.get("kick_player_id"), default=None)
    if kick_player_id is not None:
        team = normalize_team_label(team_by_track_id.get(kick_player_id))
        if team != UNKNOWN_TEAM:
            return team, "kick_player"

    event_xy = _resolve_event_xy(event)
    team = _nearest_player_team(
        frame_idx=frame_idx,
        xy=event_xy,
        players_by_frame=players_by_frame,
        frame_window=frame_window,
    )
    if team != UNKNOWN_TEAM:
        return team, "nearest_player"

    return UNKNOWN_TEAM, "unresolved"


def _event_count_template() -> dict[str, int]:
    """Create a counter payload for per-team event totals."""
    return {
        "shots": 0,
        "goals": 0,
        "passes": 0,
        "set_pieces": 0,
    }


def _event_bucket(event_type: str) -> str | None:
    """Map event type to stats bucket."""
    if event_type == "shot":
        return "shots"
    if event_type == "goal":
        return "goals"
    if event_type == "pass":
        return "passes"
    if event_type in SET_PIECE_EVENT_TYPES:
        return "set_pieces"
    return None


def build_match_stats(
    events: list[Any],
    team_analytics: dict[str, Any] | None,
    tracks: list[dict[str, Any]],
    fps: float = 30.0,
    frame_window: int = 2,
) -> dict[str, Any]:
    """
    Build team-level match stats from event outputs and possession summary.

    The output includes per-team event totals plus possession values from
    team analytics. Event team attribution uses metadata first, then nearest
    player/shot linkage fallbacks.
    """
    del fps  # Kept for API stability when caller provides video metadata fps.

    normalized_events = [_event_to_dict(event) for event in (events or [])]
    team_by_track_id, players_by_frame = _build_track_indexes(tracks or [])

    possession_summary = {}
    if isinstance(team_analytics, dict):
        possession_summary = team_analytics.get("possession", {}) or {}
    possession_teams = possession_summary.get("teams", {}) if isinstance(possession_summary, dict) else {}

    detected_teams: set[str] = set()
    for team_label in possession_teams.keys():
        team = normalize_team_label(team_label)
        if team != UNKNOWN_TEAM:
            detected_teams.add(team)
    for team_label in team_by_track_id.values():
        team = normalize_team_label(team_label)
        if team != UNKNOWN_TEAM:
            detected_teams.add(team)

    event_totals_by_team: dict[str, dict[str, int]] = {}
    attribution_counter: Counter[str] = Counter()
    shot_team_by_frame: dict[int, str] = {}
    events_processed = 0
    events_without_team = 0

    for event in sorted(normalized_events, key=lambda row: _safe_float(row.get("timestamp"), default=0.0)):
        event_type = str(event.get("event_type", "")).strip()
        bucket = _event_bucket(event_type)
        if bucket is None:
            continue

        events_processed += 1
        team, method = _resolve_event_team(
            event=event,
            shot_team_by_frame=shot_team_by_frame,
            team_by_track_id=team_by_track_id,
            players_by_frame=players_by_frame,
            frame_window=max(0, int(frame_window)),
        )
        attribution_counter[method] += 1
        if team == UNKNOWN_TEAM:
            events_without_team += 1
        else:
            detected_teams.add(team)

        per_team_counts = event_totals_by_team.setdefault(team, _event_count_template())
        per_team_counts[bucket] += 1

        if event_type == "shot":
            frame_idx = _safe_int(event.get("frame_idx"), default=None)
            if frame_idx is not None and team != UNKNOWN_TEAM:
                shot_team_by_frame[frame_idx] = team

    if events_without_team > 0:
        detected_teams.add(UNKNOWN_TEAM)

    ordered_teams = sorted(detected_teams, key=_team_sort_key)
    if not ordered_teams:
        ordered_teams = [UNKNOWN_TEAM]

    teams_payload: dict[str, dict[str, Any]] = {}
    totals = _event_count_template()
    total_possession_frames = 0.0
    total_possession_seconds = 0.0

    for team in ordered_teams:
        team_events = event_totals_by_team.get(team, _event_count_template())
        possession_row = possession_teams.get(team, {}) if isinstance(possession_teams, dict) else {}

        possession_frames = _safe_float(possession_row.get("frames"), default=0.0)
        possession_seconds = _safe_float(possession_row.get("seconds"), default=0.0)
        possession_share = _safe_float(possession_row.get("share"), default=0.0)

        teams_payload[team] = {
            **team_events,
            "possession_frames": int(possession_frames),
            "possession_seconds": possession_seconds,
            "possession_share": possession_share,
        }

        for key in totals.keys():
            totals[key] += int(team_events.get(key, 0))
        total_possession_frames += possession_frames
        total_possession_seconds += possession_seconds

    return {
        "schema_version": MATCH_STATS_SCHEMA_VERSION,
        "summary": {
            "events_processed": events_processed,
            "events_without_team": events_without_team,
            "teams_detected": ordered_teams,
            "attribution_methods": dict(attribution_counter),
        },
        "teams": teams_payload,
        "totals": {
            **totals,
            "possession_frames": int(total_possession_frames),
            "possession_seconds": total_possession_seconds,
        },
    }
