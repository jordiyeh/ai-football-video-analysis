"""Team-level analytics (possession, territory, pass network, pressing)."""

from __future__ import annotations

from collections import Counter, defaultdict
from math import hypot, sqrt
from typing import Any


UNKNOWN_TEAM = "unknown"


def _cfg_value(config: Any, key: str, default: Any) -> Any:
    """Read config key from model/object/dict with fallback."""
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely cast value to float."""
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int | None = None) -> int | None:
    """Safely cast value to int."""
    try:
        return int(value)
    except Exception:
        return default


def _resolve_team_label(track: dict[str, Any]) -> str:
    """Resolve team label from track row."""
    team_name = track.get("team_name")
    if team_name is not None:
        text = str(team_name).strip()
        if text and text.lower() != UNKNOWN_TEAM:
            return text

    team_id = _safe_int(track.get("team_id"), default=None)
    if team_id is not None and team_id >= 0:
        return f"team_{team_id}"
    return UNKNOWN_TEAM


def _center_from_bbox(track: dict[str, Any]) -> tuple[float, float] | None:
    """Return bbox center if bbox is valid."""
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


def _resolve_image_xy(track: dict[str, Any]) -> tuple[float, float] | None:
    """Resolve image-space center point from explicit fields or bbox."""
    image_x = track.get("image_x")
    image_y = track.get("image_y")
    if image_x is not None and image_y is not None:
        return (_safe_float(image_x), _safe_float(image_y))
    image_xy = track.get("image_xy")
    if isinstance(image_xy, list | tuple) and len(image_xy) >= 2:
        return (_safe_float(image_xy[0]), _safe_float(image_xy[1]))
    return _center_from_bbox(track)


def _resolve_norm_xy(
    track: dict[str, Any],
    image_xy: tuple[float, float] | None,
    frame_width: int,
    frame_height: int,
    prefer_norm: bool,
) -> tuple[float, float] | None:
    """Resolve normalized point from explicit fields or image fallback."""
    if prefer_norm:
        norm_x = track.get("norm_x")
        norm_y = track.get("norm_y")
        if norm_x is not None and norm_y is not None:
            return (_safe_float(norm_x), _safe_float(norm_y))
        norm_xy = track.get("norm_xy")
        if isinstance(norm_xy, list | tuple) and len(norm_xy) >= 2:
            return (_safe_float(norm_xy[0]), _safe_float(norm_xy[1]))

    if image_xy is None:
        return None

    width = max(1.0, float(frame_width - 1))
    height = max(1.0, float(frame_height - 1))
    return (
        max(0.0, min(1.0, image_xy[0] / width)),
        max(0.0, min(1.0, image_xy[1] / height)),
    )


def _majority_smooth(values: list[int | None], window: int) -> list[int | None]:
    """Smooth noisy ownership sequence by centered majority vote."""
    if window <= 1 or len(values) <= 2:
        return list(values)

    radius = max(0, window // 2)
    smoothed: list[int | None] = []
    for idx, value in enumerate(values):
        start = max(0, idx - radius)
        end = min(len(values), idx + radius + 1)
        counter = Counter(v for v in values[start:end] if v is not None)
        if not counter:
            smoothed.append(None)
            continue

        best_count = max(counter.values())
        tied = [candidate for candidate, count in counter.items() if count == best_count]
        if value in tied:
            smoothed.append(value)
        else:
            smoothed.append(tied[0])
    return smoothed


def _remove_short_runs(values: list[int | None], min_run: int) -> list[int | None]:
    """Replace short ownership runs with neighboring value or unknown."""
    if min_run <= 1 or not values:
        return list(values)

    output = list(values)
    runs: list[tuple[int, int, int | None]] = []
    start = 0
    current = values[0]

    for idx in range(1, len(values)):
        if values[idx] != current:
            runs.append((start, idx - 1, current))
            start = idx
            current = values[idx]
    runs.append((start, len(values) - 1, current))

    for run_idx, (run_start, run_end, run_value) in enumerate(runs):
        if run_value is None:
            continue
        run_len = run_end - run_start + 1
        if run_len >= min_run:
            continue

        prev_value = runs[run_idx - 1][2] if run_idx > 0 else None
        next_value = runs[run_idx + 1][2] if run_idx + 1 < len(runs) else None
        replacement = prev_value if prev_value == next_value else None
        for idx in range(run_start, run_end + 1):
            output[idx] = replacement
    return output


def _bin_index(value: float, bins: int) -> int:
    """Convert [0, 1] value into 0..bins-1 bucket index."""
    clipped = max(0.0, min(1.0, value))
    if clipped >= 1.0:
        return bins - 1
    return int(clipped * bins)


def _bin_label(axis: str, idx: int, bins: int) -> str:
    """Return semantic bin labels for default 3-bin layouts."""
    if bins == 3 and axis == "x":
        return ["left", "center", "right"][idx]
    if bins == 3 and axis == "y":
        return ["top", "middle", "bottom"][idx]
    return f"{axis}_bin_{idx}"


def _count_press_episodes(rows: list[dict[str, Any]], min_frames: int) -> int:
    """Count consecutive high-press sequences."""
    if not rows:
        return 0
    sorted_rows = sorted(rows, key=lambda row: int(row["frame_idx"]))
    episodes = 0
    run = 0
    prev_frame = None

    for row in sorted_rows:
        frame_idx = int(row["frame_idx"])
        high = bool(row["high_press"])
        contiguous = prev_frame is not None and frame_idx == prev_frame + 1

        if high and (run == 0 or contiguous):
            run += 1
        elif high:
            if run >= min_frames:
                episodes += 1
            run = 1
        else:
            if run >= min_frames:
                episodes += 1
            run = 0
        prev_frame = frame_idx

    if run >= min_frames:
        episodes += 1
    return episodes


def _build_assignment_index(
    assignments: list[dict[str, Any]],
    min_confidence: float,
) -> dict[int, dict[str, Any]]:
    """Map track_id -> player metadata from fused assignment rows."""
    assignment_index: dict[int, dict[str, Any]] = {}
    for row in assignments:
        track_id = _safe_int(row.get("track_id"), default=None)
        player_id = _safe_int(row.get("player_id"), default=None)
        if track_id is None or player_id is None:
            continue

        confidence = _safe_float(row.get("confidence"), default=0.0)
        if confidence < min_confidence:
            continue

        assignment_index[track_id] = {
            "player_id": player_id,
            "player_name": row.get("player_name"),
            "confidence": confidence,
            "match_method": row.get("match_method"),
        }
    return assignment_index


def build_team_analytics(
    tracks: list[dict[str, Any]],
    assignments: list[dict[str, Any]] | None,
    fps: float,
    frame_width: int,
    frame_height: int,
    config: Any,
) -> dict[str, Any]:
    """
    Build team-level tactical analytics from tracks.

    Returns JSON-safe summary plus flat row tables for CSV export.
    """
    if fps <= 0:
        fps = 30.0

    use_norm_coordinates = bool(_cfg_value(config, "use_norm_coordinates", True))
    max_ball_player_distance_px = float(
        _cfg_value(config, "possession_max_ball_distance_px", 140.0)
    )
    possession_smoothing_frames = max(
        1,
        int(_cfg_value(config, "possession_smoothing_frames", 3)),
    )
    possession_min_stable_frames = max(
        1,
        int(_cfg_value(config, "possession_min_stable_frames", 3)),
    )
    possession_min_segment_frames = max(
        1,
        int(_cfg_value(config, "possession_min_segment_frames", 4)),
    )
    pass_min_gap_seconds = float(_cfg_value(config, "pass_min_gap_seconds", 0.15))
    pass_max_gap_seconds = float(_cfg_value(config, "pass_max_gap_seconds", 2.5))
    territory_x_bins = max(2, int(_cfg_value(config, "territory_x_bins", 3)))
    territory_y_bins = max(2, int(_cfg_value(config, "territory_y_bins", 3)))
    pressure_radius_norm = max(1e-6, float(_cfg_value(config, "pressure_radius_norm", 0.10)))
    high_press_threshold = float(_cfg_value(config, "high_press_threshold", 0.65))
    high_press_min_frames = max(1, int(_cfg_value(config, "high_press_min_frames", 8)))
    min_assignment_confidence = float(_cfg_value(config, "min_assignment_confidence", 0.0))

    assignment_index = _build_assignment_index(
        assignments=assignments or [],
        min_confidence=min_assignment_confidence,
    )

    players_by_frame: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    balls_by_frame: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    frame_indices: set[int] = set()

    for track in tracks:
        frame_idx = _safe_int(track.get("frame_idx"), default=None)
        track_id = _safe_int(track.get("track_id"), default=None)
        object_type = str(track.get("object_type", ""))
        if frame_idx is None or track_id is None:
            continue

        frame_indices.add(frame_idx)

        image_xy = _resolve_image_xy(track)
        norm_xy = _resolve_norm_xy(
            track=track,
            image_xy=image_xy,
            frame_width=frame_width,
            frame_height=frame_height,
            prefer_norm=use_norm_coordinates,
        )
        row = {
            "frame_idx": frame_idx,
            "track_id": track_id,
            "timestamp": _safe_float(track.get("timestamp"), default=frame_idx / fps),
            "confidence": _safe_float(track.get("confidence"), default=0.0),
            "image_xy": image_xy,
            "norm_xy": norm_xy,
        }

        if object_type == "player":
            team = _resolve_team_label(track)
            assignment = assignment_index.get(track_id)
            player_row = {
                **row,
                "team": team,
                "player_id": assignment.get("player_id") if assignment else None,
                "player_name": assignment.get("player_name") if assignment else None,
            }
            existing = players_by_frame[frame_idx].get(track_id)
            if existing is None or player_row["confidence"] >= existing["confidence"]:
                players_by_frame[frame_idx][track_id] = player_row
        elif object_type == "ball":
            existing = balls_by_frame[frame_idx].get(track_id)
            if existing is None or row["confidence"] >= existing["confidence"]:
                balls_by_frame[frame_idx][track_id] = row

    team_labels = sorted(
        {
            row["team"]
            for frame_rows in players_by_frame.values()
            for row in frame_rows.values()
            if row["team"] != UNKNOWN_TEAM
        }
    )

    frame_diagonal = sqrt(max(1.0, float(frame_width) ** 2 + float(frame_height) ** 2))

    # Possession timeline from nearest player to ball.
    possession_timeline: list[dict[str, Any]] = []
    for frame_idx in sorted(balls_by_frame.keys()):
        frame_players = list(players_by_frame.get(frame_idx, {}).values())
        frame_balls = list(balls_by_frame.get(frame_idx, {}).values())
        if not frame_balls:
            continue

        ball = max(frame_balls, key=lambda row: row["confidence"])
        ball_image = ball["image_xy"]
        ball_norm = ball["norm_xy"]

        nearest_player = None
        nearest_distance_px = None
        for player in frame_players:
            player_image = player["image_xy"]
            player_norm = player["norm_xy"]

            if ball_image is not None and player_image is not None:
                distance_px = hypot(
                    player_image[0] - ball_image[0],
                    player_image[1] - ball_image[1],
                )
            elif ball_norm is not None and player_norm is not None:
                distance_px = hypot(
                    player_norm[0] - ball_norm[0],
                    player_norm[1] - ball_norm[1],
                ) * frame_diagonal
            else:
                continue

            if nearest_distance_px is None or distance_px < nearest_distance_px:
                nearest_distance_px = distance_px
                nearest_player = player

        raw_owner_track_id = None
        if (
            nearest_player is not None
            and nearest_distance_px is not None
            and nearest_distance_px <= max_ball_player_distance_px
        ):
            raw_owner_track_id = int(nearest_player["track_id"])

        possession_timeline.append(
            {
                "frame_idx": frame_idx,
                "timestamp": ball["timestamp"],
                "ball_track_id": ball["track_id"],
                "raw_owner_track_id": raw_owner_track_id,
                "nearest_distance_px": nearest_distance_px,
                "owner_track_id": None,
                "owner_team": UNKNOWN_TEAM,
                "owner_player_id": None,
                "owner_player_name": None,
                "owner_norm_x": None,
                "owner_norm_y": None,
                "available_players": len(frame_players),
            }
        )

    owner_sequence = [row["raw_owner_track_id"] for row in possession_timeline]
    owner_sequence = _majority_smooth(owner_sequence, possession_smoothing_frames)
    owner_sequence = _remove_short_runs(owner_sequence, possession_min_stable_frames)

    for idx, smoothed_track_id in enumerate(owner_sequence):
        row = possession_timeline[idx]
        row["owner_track_id"] = smoothed_track_id
        if smoothed_track_id is None:
            continue

        player = players_by_frame.get(row["frame_idx"], {}).get(smoothed_track_id)
        if player is None:
            row["owner_track_id"] = None
            continue

        row["owner_team"] = player["team"]
        row["owner_player_id"] = player["player_id"]
        row["owner_player_name"] = player["player_name"]
        norm_xy = player["norm_xy"]
        if norm_xy is not None:
            row["owner_norm_x"] = float(norm_xy[0])
            row["owner_norm_y"] = float(norm_xy[1])

    possession_counter = Counter(
        row["owner_team"] for row in possession_timeline if row["owner_team"] != UNKNOWN_TEAM
    )
    frames_with_ball = len(possession_timeline)
    frames_with_possession = int(sum(possession_counter.values()))
    unknown_possession_frames = max(0, frames_with_ball - frames_with_possession)

    possession_by_team = {}
    for team in team_labels:
        frames = int(possession_counter.get(team, 0))
        possession_by_team[team] = {
            "frames": frames,
            "seconds": frames / fps,
            "share": (frames / frames_with_possession) if frames_with_possession > 0 else 0.0,
        }

    dominant_team = None
    if possession_by_team:
        dominant_team = max(
            possession_by_team.items(),
            key=lambda item: item[1]["share"],
        )[0]

    # Possession segments and carrier rollup.
    possession_segments: list[dict[str, Any]] = []
    current_segment = None

    for row in possession_timeline:
        owner_track_id = row["owner_track_id"]
        owner_team = row["owner_team"]
        frame_idx = int(row["frame_idx"])

        if owner_track_id is None or owner_team == UNKNOWN_TEAM:
            if current_segment is not None:
                segment_frames = current_segment["end_frame"] - current_segment["start_frame"] + 1
                if segment_frames >= possession_min_segment_frames:
                    current_segment["frames"] = segment_frames
                    current_segment["duration_seconds"] = segment_frames / fps
                    possession_segments.append(current_segment)
                current_segment = None
            continue

        if (
            current_segment is not None
            and current_segment["owner_track_id"] == owner_track_id
            and current_segment["owner_team"] == owner_team
            and frame_idx == current_segment["end_frame"] + 1
        ):
            current_segment["end_frame"] = frame_idx
            current_segment["end_time"] = row["timestamp"]
            current_segment["end_norm_x"] = row["owner_norm_x"]
            current_segment["end_norm_y"] = row["owner_norm_y"]
        else:
            if current_segment is not None:
                segment_frames = current_segment["end_frame"] - current_segment["start_frame"] + 1
                if segment_frames >= possession_min_segment_frames:
                    current_segment["frames"] = segment_frames
                    current_segment["duration_seconds"] = segment_frames / fps
                    possession_segments.append(current_segment)

            current_segment = {
                "owner_team": owner_team,
                "owner_track_id": owner_track_id,
                "owner_player_id": row["owner_player_id"],
                "owner_player_name": row["owner_player_name"],
                "start_frame": frame_idx,
                "end_frame": frame_idx,
                "start_time": row["timestamp"],
                "end_time": row["timestamp"],
                "start_norm_x": row["owner_norm_x"],
                "start_norm_y": row["owner_norm_y"],
                "end_norm_x": row["owner_norm_x"],
                "end_norm_y": row["owner_norm_y"],
            }

    if current_segment is not None:
        segment_frames = current_segment["end_frame"] - current_segment["start_frame"] + 1
        if segment_frames >= possession_min_segment_frames:
            current_segment["frames"] = segment_frames
            current_segment["duration_seconds"] = segment_frames / fps
            possession_segments.append(current_segment)

    carrier_frames: dict[tuple[str, int], int] = defaultdict(int)
    carrier_player_ids: dict[tuple[str, int], int | None] = {}
    carrier_player_names: dict[tuple[str, int], str | None] = {}
    for segment in possession_segments:
        key = (segment["owner_team"], int(segment["owner_track_id"]))
        carrier_frames[key] += int(segment["frames"])
        carrier_player_ids[key] = segment.get("owner_player_id")
        carrier_player_names[key] = segment.get("owner_player_name")

    top_carriers = []
    for (team, track_id), frames in sorted(
        carrier_frames.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:12]:
        top_carriers.append(
            {
                "team": team,
                "track_id": track_id,
                "player_id": carrier_player_ids.get((team, track_id)),
                "player_name": carrier_player_names.get((team, track_id)),
                "frames": frames,
                "seconds": frames / fps,
                "share_of_known_possession": (
                    frames / frames_with_possession
                    if frames_with_possession > 0 else 0.0
                ),
            }
        )

    # Pass network from consecutive same-team possession transfers.
    edge_accumulator: dict[tuple[str, int, int], dict[str, Any]] = {}
    for idx in range(1, len(possession_segments)):
        previous = possession_segments[idx - 1]
        current = possession_segments[idx]

        if previous["owner_team"] != current["owner_team"]:
            continue
        if previous["owner_track_id"] == current["owner_track_id"]:
            continue

        gap_seconds = _safe_float(current["start_time"]) - _safe_float(previous["end_time"])
        if gap_seconds < pass_min_gap_seconds or gap_seconds > pass_max_gap_seconds:
            continue

        key = (
            str(current["owner_team"]),
            int(previous["owner_track_id"]),
            int(current["owner_track_id"]),
        )
        edge = edge_accumulator.setdefault(
            key,
            {
                "team": str(current["owner_team"]),
                "from_track_id": int(previous["owner_track_id"]),
                "to_track_id": int(current["owner_track_id"]),
                "from_player_id": previous.get("owner_player_id"),
                "to_player_id": current.get("owner_player_id"),
                "from_player_name": previous.get("owner_player_name"),
                "to_player_name": current.get("owner_player_name"),
                "pass_count": 0,
                "total_gap_seconds": 0.0,
                "distance_norm_sum": 0.0,
                "distance_norm_samples": 0,
            },
        )
        edge["pass_count"] += 1
        edge["total_gap_seconds"] += gap_seconds

        from_xy = (previous.get("end_norm_x"), previous.get("end_norm_y"))
        to_xy = (current.get("start_norm_x"), current.get("start_norm_y"))
        if None not in from_xy and None not in to_xy:
            distance_norm = hypot(float(from_xy[0]) - float(to_xy[0]), float(from_xy[1]) - float(to_xy[1]))
            edge["distance_norm_sum"] += distance_norm
            edge["distance_norm_samples"] += 1

    pass_network_edges: list[dict[str, Any]] = []
    for edge in edge_accumulator.values():
        avg_gap_seconds = edge["total_gap_seconds"] / max(1, edge["pass_count"])
        avg_distance_norm = None
        if edge["distance_norm_samples"] > 0:
            avg_distance_norm = edge["distance_norm_sum"] / edge["distance_norm_samples"]

        pass_network_edges.append(
            {
                "team": edge["team"],
                "from_track_id": edge["from_track_id"],
                "to_track_id": edge["to_track_id"],
                "from_player_id": edge["from_player_id"],
                "to_player_id": edge["to_player_id"],
                "from_player_name": edge["from_player_name"],
                "to_player_name": edge["to_player_name"],
                "pass_count": int(edge["pass_count"]),
                "avg_gap_seconds": avg_gap_seconds,
                "avg_distance_norm": avg_distance_norm,
            }
        )

    pass_network_edges.sort(
        key=lambda row: (str(row["team"]), -int(row["pass_count"]), int(row["from_track_id"]), int(row["to_track_id"]))
    )

    pass_team_counter = Counter()
    pass_team_nodes: dict[str, set[int]] = defaultdict(set)
    for row in pass_network_edges:
        team = str(row["team"])
        count = int(row["pass_count"])
        pass_team_counter[team] += count
        pass_team_nodes[team].add(int(row["from_track_id"]))
        pass_team_nodes[team].add(int(row["to_track_id"]))

    pass_network_summary = {
        "passes_inferred": int(sum(pass_team_counter.values())),
        "edges": len(pass_network_edges),
        "teams": {
            team: {
                "passes": int(pass_team_counter.get(team, 0)),
                "unique_nodes": len(pass_team_nodes.get(team, set())),
                "unique_edges": len([row for row in pass_network_edges if row["team"] == team]),
            }
            for team in sorted(set(team_labels) | set(pass_team_counter.keys()))
        },
        "top_edges": pass_network_edges[:20],
    }

    # Territory by normalized occupancy grid.
    territory_acc: dict[str, dict[str, Any]] = {}
    x_zone_totals = Counter()
    y_zone_totals = Counter()

    for frame_rows in players_by_frame.values():
        for player in frame_rows.values():
            team = player["team"]
            norm_xy = player.get("norm_xy")
            if team == UNKNOWN_TEAM or norm_xy is None:
                continue

            x_value = float(norm_xy[0])
            y_value = float(norm_xy[1])
            x_bin = _bin_index(x_value, territory_x_bins)
            y_bin = _bin_index(y_value, territory_y_bins)
            x_label = _bin_label("x", x_bin, territory_x_bins)
            y_label = _bin_label("y", y_bin, territory_y_bins)

            acc = territory_acc.setdefault(
                team,
                {
                    "count": 0,
                    "sum_x": 0.0,
                    "sum_y": 0.0,
                    "sum_x2": 0.0,
                    "sum_y2": 0.0,
                    "track_ids": set(),
                    "x_counts": Counter(),
                    "y_counts": Counter(),
                },
            )
            acc["count"] += 1
            acc["sum_x"] += x_value
            acc["sum_y"] += y_value
            acc["sum_x2"] += x_value * x_value
            acc["sum_y2"] += y_value * y_value
            acc["track_ids"].add(int(player["track_id"]))
            acc["x_counts"][x_label] += 1
            acc["y_counts"][y_label] += 1
            x_zone_totals[x_label] += 1
            y_zone_totals[y_label] += 1

    x_labels = [_bin_label("x", idx, territory_x_bins) for idx in range(territory_x_bins)]
    y_labels = [_bin_label("y", idx, territory_y_bins) for idx in range(territory_y_bins)]

    territory_teams: dict[str, Any] = {}
    territory_rows: list[dict[str, Any]] = []
    for team in sorted(territory_acc.keys()):
        acc = territory_acc[team]
        count = int(acc["count"])
        if count <= 0:
            continue

        mean_x = acc["sum_x"] / count
        mean_y = acc["sum_y"] / count
        std_x = sqrt(max(0.0, (acc["sum_x2"] / count) - (mean_x * mean_x)))
        std_y = sqrt(max(0.0, (acc["sum_y2"] / count) - (mean_y * mean_y)))

        x_bins = {}
        y_bins = {}
        x_control = {}
        y_control = {}

        for label in x_labels:
            bin_count = int(acc["x_counts"].get(label, 0))
            total = int(x_zone_totals.get(label, 0))
            x_bins[label] = {
                "count": bin_count,
                "ratio": (bin_count / count) if count > 0 else 0.0,
            }
            x_control[label] = (bin_count / total) if total > 0 else 0.0
            territory_rows.append(
                {
                    "team": team,
                    "axis": "x",
                    "zone": label,
                    "count": bin_count,
                    "team_ratio": x_bins[label]["ratio"],
                    "zone_control_share": x_control[label],
                }
            )

        for label in y_labels:
            bin_count = int(acc["y_counts"].get(label, 0))
            total = int(y_zone_totals.get(label, 0))
            y_bins[label] = {
                "count": bin_count,
                "ratio": (bin_count / count) if count > 0 else 0.0,
            }
            y_control[label] = (bin_count / total) if total > 0 else 0.0
            territory_rows.append(
                {
                    "team": team,
                    "axis": "y",
                    "zone": label,
                    "count": bin_count,
                    "team_ratio": y_bins[label]["ratio"],
                    "zone_control_share": y_control[label],
                }
            )

        territory_teams[team] = {
            "samples": count,
            "unique_tracks": len(acc["track_ids"]),
            "centroid_norm": [mean_x, mean_y],
            "spread_norm": [std_x, std_y],
            "x_bins": x_bins,
            "y_bins": y_bins,
            "x_zone_control_share": x_control,
            "y_zone_control_share": y_control,
        }

    territory_summary = {
        "samples": int(sum(team["samples"] for team in territory_teams.values())),
        "x_bins": x_labels,
        "y_bins": y_labels,
        "teams": territory_teams,
    }

    # Pressing metrics from defender proximity to ball carrier.
    pressing_timeline: list[dict[str, Any]] = []
    for row in possession_timeline:
        owner_team = row["owner_team"]
        owner_track_id = row["owner_track_id"]
        if owner_team == UNKNOWN_TEAM or owner_track_id is None:
            continue

        owner_norm_x = row.get("owner_norm_x")
        owner_norm_y = row.get("owner_norm_y")
        if owner_norm_x is None or owner_norm_y is None:
            continue
        owner_point = (float(owner_norm_x), float(owner_norm_y))

        defenders_by_team: dict[str, list[tuple[float, float]]] = defaultdict(list)
        frame_rows = players_by_frame.get(int(row["frame_idx"]), {})
        for player in frame_rows.values():
            team = player["team"]
            norm_xy = player.get("norm_xy")
            if norm_xy is None or team == owner_team or team == UNKNOWN_TEAM:
                continue
            defenders_by_team[team].append((float(norm_xy[0]), float(norm_xy[1])))

        for defending_team, defender_points in defenders_by_team.items():
            if not defender_points:
                continue

            distances = [
                hypot(owner_point[0] - defender[0], owner_point[1] - defender[1])
                for defender in defender_points
            ]
            nearest_distance = min(distances)
            defenders_within_radius = sum(1 for dist in distances if dist <= pressure_radius_norm)

            proximity_score = max(0.0, 1.0 - (nearest_distance / pressure_radius_norm))
            density_score = min(1.0, defenders_within_radius / 4.0)
            pressure_score = (0.65 * proximity_score) + (0.35 * density_score)
            high_press = pressure_score >= high_press_threshold

            pressing_timeline.append(
                {
                    "frame_idx": int(row["frame_idx"]),
                    "timestamp": _safe_float(row["timestamp"]),
                    "attacking_team": owner_team,
                    "defending_team": defending_team,
                    "carrier_track_id": int(owner_track_id),
                    "carrier_player_id": row.get("owner_player_id"),
                    "nearest_distance_norm": nearest_distance,
                    "defenders_within_radius": defenders_within_radius,
                    "pressure_score": pressure_score,
                    "high_press": high_press,
                }
            )

    pressing_by_team_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pressing_timeline:
        pressing_by_team_rows[str(row["defending_team"])].append(row)

    pressing_teams = {}
    for team, rows in sorted(pressing_by_team_rows.items()):
        frames = len(rows)
        high_frames = sum(1 for row in rows if row["high_press"])
        avg_nearest = sum(float(row["nearest_distance_norm"]) for row in rows) / max(1, frames)
        avg_within = sum(int(row["defenders_within_radius"]) for row in rows) / max(1, frames)
        avg_score = sum(float(row["pressure_score"]) for row in rows) / max(1, frames)

        pressing_teams[team] = {
            "frames_defending": frames,
            "avg_pressure_score": avg_score,
            "avg_nearest_distance_norm": avg_nearest,
            "avg_defenders_within_radius": avg_within,
            "high_press_frames": high_frames,
            "high_press_rate": (high_frames / frames) if frames > 0 else 0.0,
            "high_press_episodes": _count_press_episodes(rows, high_press_min_frames),
        }

    pressing_summary = {
        "evaluations": len(pressing_timeline),
        "teams": pressing_teams,
    }

    summary = {
        "frames_total": len(frame_indices),
        "frames_with_players": len(players_by_frame),
        "frames_with_ball": frames_with_ball,
        "frames_with_possession": frames_with_possession,
        "teams_detected": team_labels,
        "passes_inferred": pass_network_summary["passes_inferred"],
        "pressing_evaluations": pressing_summary["evaluations"],
        "territory_samples": territory_summary["samples"],
    }

    possession_summary = {
        "frames_with_ball": frames_with_ball,
        "frames_with_possession": frames_with_possession,
        "unknown_frames": unknown_possession_frames,
        "teams": possession_by_team,
        "segments": len(possession_segments),
        "dominant_team": dominant_team,
        "top_carriers": top_carriers,
    }

    return {
        "summary": summary,
        "possession": possession_summary,
        "territory": territory_summary,
        "pass_network": pass_network_summary,
        "pressing": pressing_summary,
        "possession_timeline": possession_timeline,
        "pass_network_edges": pass_network_edges,
        "pressing_timeline": pressing_timeline,
        "territory_rows": territory_rows,
    }

