"""Per-player analytics across runs (minutes, distance, sprints, events)."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from math import hypot
from pathlib import Path
from typing import Any


PLAYER_ANALYTICS_SCHEMA_VERSION = "1.0"


def _cfg_value(config: Any, key: str, default: Any) -> Any:
    """Read config key from object or dict with fallback."""
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _safe_int(value: Any, default: int | None = None) -> int | None:
    """Safely cast a value to int."""
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely cast a value to float."""
    try:
        return float(value)
    except Exception:
        return default


def _load_json(path: Path) -> dict[str, Any]:
    """Load dict payload from JSON file."""
    if not path.exists():
        return {}
    try:
        import json

        with open(path) as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return {}
    return {}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL rows as a list of dicts."""
    if not path.exists():
        return []

    import json

    rows: list[dict[str, Any]] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
    except Exception:
        return []
    return rows


def _load_tracks(run_dir: Path) -> list[dict[str, Any]]:
    """Load track rows from parquet/jsonl artifacts."""
    parquet_path = run_dir / "tracks.parquet"
    if parquet_path.exists():
        try:
            import pandas as pd

            frame = pd.read_parquet(parquet_path)
            return frame.to_dict(orient="records")
        except Exception:
            pass

    jsonl_path = run_dir / "tracks.jsonl"
    return _load_jsonl(jsonl_path)


def _extract_timestamp(run_dir: Path) -> str:
    """Resolve best-effort run timestamp."""
    summary = _load_json(run_dir / "summary.json")
    manifest = _load_json(run_dir / "run_manifest.json")

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


def _iter_number_candidates(value: Any) -> list[int]:
    """Collect integer values from scalar/list payloads."""
    values = value if isinstance(value, list | tuple | set) else [value]
    result: list[int] = []
    for candidate in values:
        parsed = _safe_int(candidate, default=None)
        if parsed is not None:
            result.append(parsed)
    return result


def _resolve_image_xy(track: dict[str, Any]) -> tuple[float, float] | None:
    """Resolve image-space location from explicit fields or bbox center."""
    image_x = track.get("image_x")
    image_y = track.get("image_y")
    if image_x is not None and image_y is not None:
        return (_safe_float(image_x), _safe_float(image_y))

    image_xy = track.get("image_xy")
    if isinstance(image_xy, list | tuple) and len(image_xy) >= 2:
        return (_safe_float(image_xy[0]), _safe_float(image_xy[1]))

    bbox = track.get("bbox")
    if isinstance(bbox, list | tuple) and len(bbox) >= 4:
        x1 = _safe_float(bbox[0], default=float("nan"))
        y1 = _safe_float(bbox[1], default=float("nan"))
        x2 = _safe_float(bbox[2], default=float("nan"))
        y2 = _safe_float(bbox[3], default=float("nan"))
        if x2 > x1 and y2 > y1:
            return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
    return None


def _normalize_player_name(value: Any) -> str | None:
    """Normalize player name values."""
    if value is None:
        return None
    name = str(value).strip()
    return name or None


def _normalize_team_hint(value: Any) -> str | None:
    """Normalize team hint values."""
    if value is None:
        return None
    label = str(value).strip()
    return label or None


def _build_assignment_indexes(
    assignments: list[dict[str, Any]],
    min_confidence: float,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    """Build track->player and player metadata indexes from assignment rows."""
    track_index: dict[int, dict[str, Any]] = {}
    player_index: dict[int, dict[str, Any]] = {}

    for row in assignments:
        track_id = _safe_int(row.get("track_id"), default=None)
        player_id = _safe_int(row.get("player_id"), default=None)
        if track_id is None or player_id is None:
            continue

        confidence = _safe_float(row.get("confidence"), default=0.0)
        if confidence < min_confidence:
            continue

        player_name = _normalize_player_name(row.get("player_name"))
        team_hint = _normalize_team_hint(row.get("team_hint"))

        track_index[track_id] = {
            "player_id": player_id,
            "player_name": player_name,
            "team_hint": team_hint,
            "confidence": confidence,
        }

        existing_player = player_index.get(player_id)
        if existing_player is None:
            player_index[player_id] = {
                "player_name": player_name,
                "team_hint": team_hint,
                "confidence": confidence,
            }
        else:
            if existing_player.get("player_name") is None and player_name is not None:
                existing_player["player_name"] = player_name
            if existing_player.get("team_hint") is None and team_hint is not None:
                existing_player["team_hint"] = team_hint
            if confidence > float(existing_player.get("confidence", 0.0)):
                existing_player["confidence"] = confidence

    return track_index, player_index


def _walk_nested_values(prefix: str, value: Any) -> list[tuple[str, Any]]:
    """Walk nested dict/list payloads and return (path, value) leaf rows."""
    rows: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            rows.extend(_walk_nested_values(path, item))
        return rows

    if isinstance(value, list | tuple):
        for item in value:
            if isinstance(item, dict | list | tuple):
                rows.extend(_walk_nested_values(prefix, item))
            else:
                rows.append((prefix, item))
        return rows

    rows.append((prefix, value))
    return rows


def _collect_player_ids_from_payload(
    payload: dict[str, Any],
    assignment_by_track_id: dict[int, dict[str, Any]],
) -> set[int]:
    """Collect player IDs from event payload keys/values."""
    player_ids: set[int] = set()

    for key, value in _walk_nested_values("", payload):
        lowered = key.lower()
        if not lowered:
            continue

        if "player_id" in lowered:
            for candidate in _iter_number_candidates(value):
                if candidate >= 0:
                    player_ids.add(candidate)
            continue

        if "track_id" not in lowered:
            continue
        for track_id in _iter_number_candidates(value):
            mapped = assignment_by_track_id.get(track_id)
            if mapped is None:
                continue
            player_id = _safe_int(mapped.get("player_id"), default=None)
            if player_id is not None and player_id >= 0:
                player_ids.add(player_id)

    return player_ids


def _compute_motion_metrics(
    samples: list[dict[str, Any]],
    sprint_speed_threshold_px_per_sec: float,
    sprint_min_duration_seconds: float,
    max_track_gap_frames: int,
    fps: float,
) -> tuple[float, int]:
    """Compute distance and sprint episodes for one track timeline."""
    if len(samples) < 2:
        return 0.0, 0

    total_distance = 0.0
    sprint_count = 0
    sprint_duration = 0.0
    in_sprint = False

    for previous, current in zip(samples[:-1], samples[1:]):
        prev_xy = previous.get("image_xy")
        curr_xy = current.get("image_xy")
        if prev_xy is None or curr_xy is None:
            if in_sprint and sprint_duration >= sprint_min_duration_seconds:
                sprint_count += 1
            sprint_duration = 0.0
            in_sprint = False
            continue

        prev_frame = _safe_int(previous.get("frame_idx"), default=None)
        curr_frame = _safe_int(current.get("frame_idx"), default=None)
        frame_gap = None
        if prev_frame is not None and curr_frame is not None:
            frame_gap = curr_frame - prev_frame
            if frame_gap <= 0:
                continue
            if frame_gap > max_track_gap_frames:
                if in_sprint and sprint_duration >= sprint_min_duration_seconds:
                    sprint_count += 1
                sprint_duration = 0.0
                in_sprint = False
                continue

        prev_ts = _safe_float(previous.get("timestamp"), default=float("nan"))
        curr_ts = _safe_float(current.get("timestamp"), default=float("nan"))
        dt = curr_ts - prev_ts
        if dt <= 0:
            if frame_gap is None:
                continue
            dt = frame_gap / max(1e-6, fps)
        if dt <= 0:
            continue

        distance = hypot(float(curr_xy[0]) - float(prev_xy[0]), float(curr_xy[1]) - float(prev_xy[1]))
        total_distance += max(0.0, distance)

        speed = distance / dt
        if speed >= sprint_speed_threshold_px_per_sec:
            sprint_duration += dt
            in_sprint = True
        else:
            if in_sprint and sprint_duration >= sprint_min_duration_seconds:
                sprint_count += 1
            sprint_duration = 0.0
            in_sprint = False

    if in_sprint and sprint_duration >= sprint_min_duration_seconds:
        sprint_count += 1

    return float(total_distance), int(sprint_count)


def _parse_run_player_metrics(
    run_dir: Path,
    config: Any,
) -> dict[str, Any] | None:
    """Parse one run directory into per-player analytics rows."""
    assignments_payload = _load_json(run_dir / "player_assignments.json")
    assignments = assignments_payload.get("assignments")
    assignments = assignments if isinstance(assignments, list) else []

    tracks = _load_tracks(run_dir)
    events = _load_jsonl(run_dir / "events.jsonl")
    video_metadata = _load_json(run_dir / "video_metadata.json")
    if not assignments and not tracks and not events:
        return None

    fps = max(1.0, _safe_float(video_metadata.get("fps"), default=30.0))
    min_assignment_confidence = float(_cfg_value(config, "min_assignment_confidence", 0.0))
    sprint_speed_threshold = float(_cfg_value(config, "sprint_speed_threshold_px_per_sec", 240.0))
    sprint_min_duration = float(_cfg_value(config, "sprint_min_duration_seconds", 0.6))
    max_track_gap_frames = max(1, int(_cfg_value(config, "max_track_gap_frames", 3)))

    assignment_by_track_id, player_info = _build_assignment_indexes(
        assignments=assignments,
        min_confidence=min_assignment_confidence,
    )

    player_frames: dict[int, set[int]] = defaultdict(set)
    player_timestamps: dict[int, list[float]] = defaultdict(list)
    samples_by_player_track: dict[int, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

    for row in tracks:
        if str(row.get("object_type", "")).strip().lower() != "player":
            continue

        track_id = _safe_int(row.get("track_id"), default=None)
        if track_id is None:
            continue
        assignment = assignment_by_track_id.get(track_id)
        if assignment is None:
            continue

        player_id = int(assignment["player_id"])
        frame_idx = _safe_int(row.get("frame_idx"), default=None)
        timestamp = _safe_float(
            row.get("timestamp"),
            default=(float(frame_idx) / fps) if frame_idx is not None else 0.0,
        )
        image_xy = _resolve_image_xy(row)

        if frame_idx is not None:
            player_frames[player_id].add(frame_idx)
        player_timestamps[player_id].append(timestamp)
        samples_by_player_track[player_id][track_id].append(
            {
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "image_xy": image_xy,
            }
        )

    per_player: dict[int, dict[str, Any]] = {}
    for player_id, tracks_map in samples_by_player_track.items():
        info = player_info.get(player_id, {})
        distance_total = 0.0
        sprint_total = 0
        track_ids: list[int] = []

        for track_id, samples in tracks_map.items():
            samples.sort(
                key=lambda row: (
                    _safe_int(row.get("frame_idx"), default=10**12),
                    _safe_float(row.get("timestamp"), default=0.0),
                )
            )
            distance_pixels, sprint_count = _compute_motion_metrics(
                samples=samples,
                sprint_speed_threshold_px_per_sec=sprint_speed_threshold,
                sprint_min_duration_seconds=sprint_min_duration,
                max_track_gap_frames=max_track_gap_frames,
                fps=fps,
            )
            distance_total += distance_pixels
            sprint_total += sprint_count
            track_ids.append(track_id)

        frame_count = len(player_frames.get(player_id, set()))
        minutes_played = frame_count / fps / 60.0 if frame_count > 0 else 0.0
        if minutes_played <= 0:
            ts_values = sorted(player_timestamps.get(player_id, []))
            if len(ts_values) >= 2:
                minutes_played = max(0.0, (ts_values[-1] - ts_values[0]) / 60.0)

        per_player[player_id] = {
            "player_id": player_id,
            "player_name": info.get("player_name"),
            "team_hint": info.get("team_hint"),
            "minutes_played": float(minutes_played),
            "distance_pixels": float(distance_total),
            "sprints": int(sprint_total),
            "events_total": 0,
            "events_by_type": Counter(),
            "track_ids": sorted(set(track_ids)),
        }

    for event in events:
        event_type = str(event.get("event_type") or "unknown").strip() or "unknown"
        metadata = event.get("metadata")
        event_payload: dict[str, Any] = {
            "metadata": metadata if isinstance(metadata, dict) else {},
            "player_id": event.get("player_id"),
            "track_id": event.get("track_id"),
        }
        player_ids = _collect_player_ids_from_payload(
            payload=event_payload,
            assignment_by_track_id=assignment_by_track_id,
        )
        if not player_ids:
            continue

        for player_id in sorted(player_ids):
            entry = per_player.get(player_id)
            if entry is None:
                info = player_info.get(player_id, {})
                entry = {
                    "player_id": player_id,
                    "player_name": info.get("player_name"),
                    "team_hint": info.get("team_hint"),
                    "minutes_played": 0.0,
                    "distance_pixels": 0.0,
                    "sprints": 0,
                    "events_total": 0,
                    "events_by_type": Counter(),
                    "track_ids": [],
                }
                per_player[player_id] = entry

            entry["events_total"] = int(entry.get("events_total", 0)) + 1
            event_counts: Counter[str] = entry["events_by_type"]
            event_counts[event_type] += 1

    player_rows: list[dict[str, Any]] = []
    events_total = 0
    distance_total = 0.0
    sprints_total = 0
    minutes_total = 0.0
    events_by_type_total: Counter[str] = Counter()

    for player_id in sorted(per_player.keys()):
        row = per_player[player_id]
        event_counter = row.get("events_by_type")
        if not isinstance(event_counter, Counter):
            event_counter = Counter(event_counter or {})

        serialized_row = {
            "player_id": int(player_id),
            "player_name": row.get("player_name"),
            "team_hint": row.get("team_hint"),
            "minutes_played": float(row.get("minutes_played", 0.0)),
            "distance_pixels": float(row.get("distance_pixels", 0.0)),
            "sprints": int(row.get("sprints", 0)),
            "events_total": int(row.get("events_total", 0)),
            "events_by_type": dict(sorted(event_counter.items())),
            "track_ids": sorted({int(track_id) for track_id in row.get("track_ids", [])}),
        }
        player_rows.append(serialized_row)

        minutes_total += float(serialized_row["minutes_played"])
        distance_total += float(serialized_row["distance_pixels"])
        sprints_total += int(serialized_row["sprints"])
        events_total += int(serialized_row["events_total"])
        events_by_type_total.update(serialized_row["events_by_type"])

    return {
        "run_name": run_dir.name,
        "video_id": run_dir.name,
        "timestamp": _extract_timestamp(run_dir),
        "fps": fps,
        "players": player_rows,
        "summary": {
            "players_detected": len(player_rows),
            "minutes_total": float(minutes_total),
            "distance_pixels_total": float(distance_total),
            "sprints_total": int(sprints_total),
            "events_total": int(events_total),
            "events_by_type": dict(sorted(events_by_type_total.items())),
        },
    }


def build_player_analytics_report(
    runs_root: Path,
    current_run: Path,
    config: Any,
) -> dict[str, Any]:
    """
    Build per-player analytics across runs.

    Returns:
        JSON-safe payload with top-level summary, per-run summaries, and
        per-player totals/averages.
    """
    include_current_run = bool(_cfg_value(config, "include_current_run", True))
    max_runs = max(1, int(_cfg_value(config, "max_runs", 80)))

    run_dirs: list[Path] = []
    if runs_root.exists():
        for candidate in sorted(runs_root.iterdir(), key=lambda path: path.name):
            if not candidate.is_dir():
                continue
            if candidate.name.startswith("."):
                continue
            if not include_current_run and candidate.resolve() == current_run.resolve():
                continue
            run_dirs.append(candidate)

    run_rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        parsed = _parse_run_player_metrics(run_dir=run_dir, config=config)
        if parsed is not None:
            run_rows.append(parsed)

    run_rows.sort(key=lambda row: (str(row.get("timestamp", "")), str(row.get("run_name", ""))))
    if len(run_rows) > max_runs:
        run_rows = run_rows[-max_runs:]

    player_totals: dict[int, dict[str, Any]] = {}
    all_events_by_type: Counter[str] = Counter()
    minutes_total = 0.0
    distance_total = 0.0
    sprints_total = 0
    events_total = 0

    run_summaries: list[dict[str, Any]] = []
    for run_row in run_rows:
        summary = run_row.get("summary", {})
        run_summary = {
            "run_name": run_row.get("run_name"),
            "video_id": run_row.get("video_id"),
            "timestamp": run_row.get("timestamp"),
            "players_detected": int(_safe_int(summary.get("players_detected"), default=0) or 0),
            "minutes_total": float(_safe_float(summary.get("minutes_total"), default=0.0)),
            "distance_pixels_total": float(_safe_float(summary.get("distance_pixels_total"), default=0.0)),
            "sprints_total": int(_safe_int(summary.get("sprints_total"), default=0) or 0),
            "events_total": int(_safe_int(summary.get("events_total"), default=0) or 0),
            "events_by_type": dict(summary.get("events_by_type", {})),
        }
        run_summaries.append(run_summary)

        minutes_total += run_summary["minutes_total"]
        distance_total += run_summary["distance_pixels_total"]
        sprints_total += run_summary["sprints_total"]
        events_total += run_summary["events_total"]
        all_events_by_type.update(run_summary["events_by_type"])

        for player_row in run_row.get("players", []):
            player_id = _safe_int(player_row.get("player_id"), default=None)
            if player_id is None:
                continue

            entry = player_totals.setdefault(
                player_id,
                {
                    "player_id": player_id,
                    "player_name": player_row.get("player_name"),
                    "team_hint": player_row.get("team_hint"),
                    "matches_played": 0,
                    "totals": {
                        "minutes_played": 0.0,
                        "distance_pixels": 0.0,
                        "sprints": 0,
                        "events_total": 0,
                        "events_by_type": Counter(),
                    },
                    "runs": [],
                },
            )

            player_name = _normalize_player_name(player_row.get("player_name"))
            team_hint = _normalize_team_hint(player_row.get("team_hint"))
            if entry.get("player_name") is None and player_name is not None:
                entry["player_name"] = player_name
            if entry.get("team_hint") is None and team_hint is not None:
                entry["team_hint"] = team_hint

            minutes_played = float(_safe_float(player_row.get("minutes_played"), default=0.0))
            distance_pixels = float(_safe_float(player_row.get("distance_pixels"), default=0.0))
            sprints = int(_safe_int(player_row.get("sprints"), default=0) or 0)
            player_events_total = int(_safe_int(player_row.get("events_total"), default=0) or 0)
            player_events_by_type = Counter(player_row.get("events_by_type", {}))
            track_ids = sorted({int(track_id) for track_id in player_row.get("track_ids", [])})

            entry["matches_played"] += 1
            totals = entry["totals"]
            totals["minutes_played"] = float(totals["minutes_played"]) + minutes_played
            totals["distance_pixels"] = float(totals["distance_pixels"]) + distance_pixels
            totals["sprints"] = int(totals["sprints"]) + sprints
            totals["events_total"] = int(totals["events_total"]) + player_events_total
            event_counter: Counter[str] = totals["events_by_type"]
            event_counter.update(player_events_by_type)

            entry["runs"].append(
                {
                    "run_name": run_row.get("run_name"),
                    "video_id": run_row.get("video_id"),
                    "timestamp": run_row.get("timestamp"),
                    "minutes_played": minutes_played,
                    "distance_pixels": distance_pixels,
                    "sprints": sprints,
                    "events_total": player_events_total,
                    "events_by_type": dict(sorted(player_events_by_type.items())),
                    "track_ids": track_ids,
                }
            )

    players_payload: list[dict[str, Any]] = []
    for player_id, entry in player_totals.items():
        matches_played = max(1, int(entry["matches_played"]))
        totals = entry["totals"]
        events_counter = totals["events_by_type"]

        player_payload = {
            "player_id": int(player_id),
            "player_name": entry.get("player_name"),
            "team_hint": entry.get("team_hint"),
            "matches_played": int(entry["matches_played"]),
            "totals": {
                "minutes_played": float(totals["minutes_played"]),
                "distance_pixels": float(totals["distance_pixels"]),
                "sprints": int(totals["sprints"]),
                "events_total": int(totals["events_total"]),
                "events_by_type": dict(sorted(events_counter.items())),
            },
            "averages": {
                "minutes_played_per_match": float(totals["minutes_played"]) / matches_played,
                "distance_pixels_per_match": float(totals["distance_pixels"]) / matches_played,
                "sprints_per_match": float(totals["sprints"]) / matches_played,
                "events_per_match": float(totals["events_total"]) / matches_played,
            },
            "runs": sorted(
                entry["runs"],
                key=lambda row: (str(row.get("timestamp", "")), str(row.get("run_name", ""))),
            ),
        }
        players_payload.append(player_payload)

    players_payload.sort(
        key=lambda row: (
            -int(row["totals"]["events_total"]),
            -float(row["totals"]["minutes_played"]),
            -float(row["totals"]["distance_pixels"]),
            int(row["player_id"]),
        )
    )

    return {
        "schema_version": PLAYER_ANALYTICS_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs_root": str(runs_root),
        "summary": {
            "runs_analyzed": len(run_summaries),
            "players_detected": len(players_payload),
            "minutes_total": float(minutes_total),
            "distance_pixels_total": float(distance_total),
            "sprints_total": int(sprints_total),
            "events_total": int(events_total),
            "events_by_type": dict(sorted(all_events_by_type.items())),
        },
        "runs": run_summaries,
        "players": players_payload,
    }
