"""Per-player highlight reel generation from fused assignments and highlight segments."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Coerce a value to float with fallback."""
    try:
        return float(value)
    except Exception:
        return default


def _segment_duration(segment: dict[str, Any]) -> float:
    """Get segment duration from fields or compute from start/end."""
    duration = segment.get("duration")
    if duration is not None:
        return max(0.0, _safe_float(duration, default=0.0))
    start = _safe_float(segment.get("start_time"), default=0.0)
    end = _safe_float(segment.get("end_time"), default=start)
    return max(0.0, end - start)


def build_player_reels(
    segments: list[dict[str, Any]],
    tracks: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
    fps: float,
    max_segments_per_player: int = 8,
    min_presence_seconds: float = 1.5,
    min_player_segment_score: float = 0.2,
    min_assignment_confidence: float = 0.6,
    include_suggested: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Build per-player highlight reel segments.

    Returns:
        (player_reels, summary)
    """
    if fps <= 0:
        fps = 30.0

    # Filter/normalize segments.
    usable_segments: list[dict[str, Any]] = []
    segment_by_id: dict[str, dict[str, Any]] = {}
    for index, segment in enumerate(segments, start=1):
        segment_id = str(segment.get("segment_id") or f"segment_{index:03d}")
        start = _safe_float(segment.get("start_time"), default=-1.0)
        end = _safe_float(segment.get("end_time"), default=-1.0)
        if start < 0 or end <= start:
            continue
        copy = dict(segment)
        copy["segment_id"] = segment_id
        copy["start_time"] = start
        copy["end_time"] = end
        copy["duration"] = _segment_duration(copy)
        copy["score"] = _safe_float(copy.get("score"), default=0.0)
        copy["reasons"] = list(copy.get("reasons") or [])
        copy["sources"] = list(copy.get("sources") or [])
        usable_segments.append(copy)
        segment_by_id[segment_id] = copy

    if not usable_segments:
        return [], {
            "players_with_reels": 0,
            "player_segments_total": 0,
            "segments_considered": 0,
            "assignments_considered": 0,
        }

    # Build track assignment index.
    assignment_index: dict[int, dict[str, Any]] = {}
    player_name_index: dict[int, str | None] = {}
    for assignment in assignments:
        player_id = assignment.get("player_id")
        if player_id is None:
            continue

        track_id = assignment.get("track_id")
        if track_id is None:
            continue

        method = str(assignment.get("match_method", ""))
        if not include_suggested and method != "auto":
            continue

        confidence = _safe_float(assignment.get("confidence"), default=0.0)
        if confidence < min_assignment_confidence:
            continue

        assignment_index[int(track_id)] = {
            "player_id": int(player_id),
            "confidence": confidence,
            "match_method": method,
            "player_name": assignment.get("player_name"),
        }
        if assignment.get("player_name"):
            player_name_index[int(player_id)] = str(assignment.get("player_name"))

    if not assignment_index:
        return [], {
            "players_with_reels": 0,
            "player_segments_total": 0,
            "segments_considered": len(usable_segments),
            "assignments_considered": 0,
        }

    # Aggregate player presence per segment.
    stats: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for track in tracks:
        if track.get("object_type") != "player":
            continue

        track_id_raw = track.get("track_id")
        if track_id_raw is None:
            continue
        track_id = int(track_id_raw)
        assigned = assignment_index.get(track_id)
        if assigned is None:
            continue

        frame_idx = track.get("frame_idx")
        timestamp = track.get("timestamp")
        if timestamp is None:
            if frame_idx is None:
                continue
            timestamp = int(frame_idx) / fps
        timestamp = _safe_float(timestamp, default=-1.0)
        if timestamp < 0:
            continue

        player_id = int(assigned["player_id"])
        for segment in usable_segments:
            if not (segment["start_time"] <= timestamp <= segment["end_time"]):
                continue

            segment_id = segment["segment_id"]
            entry = stats[player_id].setdefault(
                segment_id,
                {
                    "frame_idxs": set(),
                    "track_ids": set(),
                    "point_count": 0,
                    "weighted_points": 0.0,
                    "confidence_sum": 0.0,
                },
            )
            if frame_idx is not None:
                entry["frame_idxs"].add(int(frame_idx))
            entry["track_ids"].add(track_id)
            entry["point_count"] += 1
            entry["weighted_points"] += float(assigned["confidence"])
            entry["confidence_sum"] += float(assigned["confidence"])

    player_reels: list[dict[str, Any]] = []
    player_segments_total = 0

    for player_id, segment_stats in stats.items():
        player_rows: list[dict[str, Any]] = []

        for segment_id, entry in segment_stats.items():
            segment = segment_by_id[segment_id]
            duration = max(1e-6, _safe_float(segment["duration"], default=0.0))
            segment_score = _safe_float(segment.get("score"), default=0.0)
            frame_count = len(entry["frame_idxs"])
            presence_seconds = frame_count / fps if frame_count > 0 else entry["point_count"] / fps

            if presence_seconds < min_presence_seconds:
                continue

            presence_ratio = min(1.0, presence_seconds / duration)
            assignment_conf = entry["confidence_sum"] / max(1, entry["point_count"])
            activity_score = min(1.0, entry["weighted_points"] / max(1.0, duration * fps * 0.6))

            player_score = segment_score * (0.6 + 0.25 * presence_ratio + 0.15 * activity_score)
            player_score *= (0.85 + 0.15 * assignment_conf)

            if player_score < min_player_segment_score:
                continue

            player_rows.append(
                {
                    "segment_id": segment_id,
                    "start_time": segment["start_time"],
                    "end_time": segment["end_time"],
                    "duration": segment["duration"],
                    "base_segment_score": segment_score,
                    "player_segment_score": player_score,
                    "presence_seconds": presence_seconds,
                    "presence_ratio": presence_ratio,
                    "track_ids": sorted(entry["track_ids"]),
                    "track_count": len(entry["track_ids"]),
                    "assignment_confidence_avg": assignment_conf,
                    "activity_score": activity_score,
                    "reasons": segment.get("reasons", []),
                    "sources": segment.get("sources", []),
                }
            )

        if not player_rows:
            continue

        player_rows.sort(key=lambda row: row["player_segment_score"], reverse=True)
        player_rows = player_rows[:max_segments_per_player]
        player_segments_total += len(player_rows)

        player_reels.append(
            {
                "player_id": player_id,
                "player_name": player_name_index.get(player_id),
                "segments": player_rows,
                "segment_count": len(player_rows),
            }
        )

    player_reels.sort(
        key=lambda p: max((row["player_segment_score"] for row in p["segments"]), default=0.0),
        reverse=True,
    )

    summary = {
        "players_with_reels": len(player_reels),
        "player_segments_total": player_segments_total,
        "segments_considered": len(usable_segments),
        "assignments_considered": len(assignment_index),
    }
    return player_reels, summary

