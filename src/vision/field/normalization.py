"""Field-view normalization helpers for zoom-aware positional analytics."""

from __future__ import annotations

from typing import Any

import numpy as np


def _cfg_value(config: Any, key: str, default: Any) -> Any:
    """Read config key from object or dict with fallback."""
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _clip_viewport(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame_width: int,
    frame_height: int,
) -> tuple[float, float, float, float]:
    """Clip viewport coordinates to frame boundaries."""
    max_x = max(1.0, float(frame_width - 1))
    max_y = max(1.0, float(frame_height - 1))
    x1 = float(np.clip(x1, 0.0, max_x))
    x2 = float(np.clip(x2, 0.0, max_x))
    y1 = float(np.clip(y1, 0.0, max_y))
    y2 = float(np.clip(y2, 0.0, max_y))

    if x2 <= x1:
        x2 = min(max_x, x1 + 1.0)
    if y2 <= y1:
        y2 = min(max_y, y1 + 1.0)

    return x1, y1, x2, y2


def _enforce_min_viewport_size(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame_width: int,
    frame_height: int,
    min_width_ratio: float,
    min_height_ratio: float,
) -> tuple[float, float, float, float]:
    """Expand viewport if too narrow/tall for stable normalization."""
    width = x2 - x1
    height = y2 - y1
    min_width = max(1.0, float(frame_width) * float(min_width_ratio))
    min_height = max(1.0, float(frame_height) * float(min_height_ratio))

    if width < min_width:
        center_x = (x1 + x2) * 0.5
        x1 = center_x - (min_width * 0.5)
        x2 = center_x + (min_width * 0.5)

    if height < min_height:
        center_y = (y1 + y2) * 0.5
        y1 = center_y - (min_height * 0.5)
        y2 = center_y + (min_height * 0.5)

    return _clip_viewport(x1, y1, x2, y2, frame_width, frame_height)


def _bbox_center(track: dict[str, Any]) -> tuple[float, float] | None:
    """Return center point from track bbox."""
    bbox = track.get("bbox")
    if not isinstance(bbox, list | tuple) or len(bbox) < 4:
        return None

    try:
        x1 = float(bbox[0])
        y1 = float(bbox[1])
        x2 = float(bbox[2])
        y2 = float(bbox[3])
    except (TypeError, ValueError):
        return None

    if x2 <= x1 or y2 <= y1:
        return None

    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def estimate_frame_viewports(
    tracks: list[dict[str, Any]],
    frame_width: int,
    frame_height: int,
    config: Any,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """
    Estimate per-frame dynamic viewports from player spread with temporal smoothing.

    Returns:
        (frame_idx -> viewport row, summary)
    """
    min_players = int(_cfg_value(config, "min_players_per_frame", 6))
    low_q = float(_cfg_value(config, "player_percentile_low", 0.10))
    high_q = float(_cfg_value(config, "player_percentile_high", 0.90))
    margin_ratio = float(_cfg_value(config, "margin_ratio", 0.12))
    smoothing_alpha = float(_cfg_value(config, "smoothing_alpha", 0.25))
    min_width_ratio = float(_cfg_value(config, "min_viewport_width_ratio", 0.35))
    min_height_ratio = float(_cfg_value(config, "min_viewport_height_ratio", 0.35))

    low_q = float(np.clip(low_q, 0.0, 0.49))
    high_q = float(np.clip(high_q, 0.51, 1.0))
    smoothing_alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))

    players_by_frame: dict[int, list[tuple[float, float]]] = {}
    frame_indices: set[int] = set()

    for track in tracks:
        frame_raw = track.get("frame_idx")
        try:
            frame_idx = int(frame_raw)
        except (TypeError, ValueError):
            continue

        frame_indices.add(frame_idx)

        if track.get("object_type") != "player":
            continue

        center = _bbox_center(track)
        if center is None:
            continue

        players_by_frame.setdefault(frame_idx, []).append(center)

    full_view = (0.0, 0.0, max(1.0, frame_width - 1.0), max(1.0, frame_height - 1.0))
    viewport_map: dict[int, dict[str, Any]] = {}
    prev_smoothed: tuple[float, float, float, float] | None = None

    frames_dynamic = 0
    frames_fallback = 0
    frames_carry_forward = 0

    for frame_idx in sorted(frame_indices):
        centers = players_by_frame.get(frame_idx, [])

        if len(centers) >= min_players:
            xs = np.array([c[0] for c in centers], dtype=np.float32)
            ys = np.array([c[1] for c in centers], dtype=np.float32)

            x_low = float(np.quantile(xs, low_q))
            x_high = float(np.quantile(xs, high_q))
            y_low = float(np.quantile(ys, low_q))
            y_high = float(np.quantile(ys, high_q))

            width = max(1.0, x_high - x_low)
            height = max(1.0, y_high - y_low)
            x1 = x_low - (width * margin_ratio)
            x2 = x_high + (width * margin_ratio)
            y1 = y_low - (height * margin_ratio)
            y2 = y_high + (height * margin_ratio)
            method = "dynamic_player_spread"
            frames_dynamic += 1
        elif prev_smoothed is not None:
            x1, y1, x2, y2 = prev_smoothed
            method = "carry_forward"
            frames_carry_forward += 1
        else:
            x1, y1, x2, y2 = full_view
            method = "frame_full"
            frames_fallback += 1

        x1, y1, x2, y2 = _enforce_min_viewport_size(
            x1,
            y1,
            x2,
            y2,
            frame_width=frame_width,
            frame_height=frame_height,
            min_width_ratio=min_width_ratio,
            min_height_ratio=min_height_ratio,
        )

        if prev_smoothed is not None:
            px1, py1, px2, py2 = prev_smoothed
            x1 = (smoothing_alpha * x1) + ((1.0 - smoothing_alpha) * px1)
            y1 = (smoothing_alpha * y1) + ((1.0 - smoothing_alpha) * py1)
            x2 = (smoothing_alpha * x2) + ((1.0 - smoothing_alpha) * px2)
            y2 = (smoothing_alpha * y2) + ((1.0 - smoothing_alpha) * py2)
            x1, y1, x2, y2 = _clip_viewport(x1, y1, x2, y2, frame_width, frame_height)

        prev_smoothed = (x1, y1, x2, y2)
        viewport_width = max(1.0, x2 - x1)
        viewport_height = max(1.0, y2 - y1)
        viewport_map[frame_idx] = {
            "frame_idx": frame_idx,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "width": viewport_width,
            "height": viewport_height,
            "width_ratio": viewport_width / max(1.0, float(frame_width)),
            "height_ratio": viewport_height / max(1.0, float(frame_height)),
            "method": method,
            "player_points": len(centers),
        }

    summary = {
        "frames_total": len(frame_indices),
        "frames_dynamic": frames_dynamic,
        "frames_carry_forward": frames_carry_forward,
        "frames_fallback": frames_fallback,
    }
    return viewport_map, summary


def normalize_tracks_to_field_view(
    tracks: list[dict[str, Any]],
    frame_width: int,
    frame_height: int,
    config: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """
    Add `image_xy` + `norm_xy` fields onto tracks for zoom-aware analytics.

    Returns:
        (normalized_tracks, viewport_rows, summary)
    """
    clip_norm = bool(_cfg_value(config, "clip_norm", True))

    viewport_map, viewport_summary = estimate_frame_viewports(
        tracks=tracks,
        frame_width=frame_width,
        frame_height=frame_height,
        config=config,
    )
    viewport_rows = [viewport_map[idx] for idx in sorted(viewport_map.keys())]

    normalized_tracks: list[dict[str, Any]] = []
    normalized_points = 0
    points_missing_center = 0

    for track in tracks:
        row = dict(track)
        center = _bbox_center(track)
        frame_raw = track.get("frame_idx")
        try:
            frame_idx = int(frame_raw)
        except (TypeError, ValueError):
            frame_idx = None

        if center is None:
            row["image_x"] = None
            row["image_y"] = None
            row["image_xy"] = None
            row["norm_x"] = None
            row["norm_y"] = None
            row["norm_xy"] = None
            row["norm_source"] = None
            points_missing_center += 1
            normalized_tracks.append(row)
            continue

        image_x, image_y = center
        row["image_x"] = image_x
        row["image_y"] = image_y
        row["image_xy"] = [image_x, image_y]

        if frame_idx is None or frame_idx not in viewport_map:
            x1, y1, x2, y2 = (0.0, 0.0, max(1.0, frame_width - 1.0), max(1.0, frame_height - 1.0))
            norm_source = "frame_full"
        else:
            viewport = viewport_map[frame_idx]
            x1 = float(viewport["x1"])
            y1 = float(viewport["y1"])
            x2 = float(viewport["x2"])
            y2 = float(viewport["y2"])
            norm_source = str(viewport["method"])

        width = max(1e-6, x2 - x1)
        height = max(1e-6, y2 - y1)
        norm_x = (image_x - x1) / width
        norm_y = (image_y - y1) / height

        if clip_norm:
            norm_x = float(np.clip(norm_x, 0.0, 1.0))
            norm_y = float(np.clip(norm_y, 0.0, 1.0))
        else:
            norm_x = float(norm_x)
            norm_y = float(norm_y)

        row["norm_x"] = norm_x
        row["norm_y"] = norm_y
        row["norm_xy"] = [norm_x, norm_y]
        row["norm_source"] = norm_source
        normalized_points += 1
        normalized_tracks.append(row)

    avg_width_ratio = float(np.mean([row["width_ratio"] for row in viewport_rows])) if viewport_rows else 1.0
    avg_height_ratio = float(np.mean([row["height_ratio"] for row in viewport_rows])) if viewport_rows else 1.0

    summary = {
        **viewport_summary,
        "track_points_total": len(tracks),
        "track_points_normalized": normalized_points,
        "track_points_missing_center": points_missing_center,
        "avg_viewport_width_ratio": avg_width_ratio,
        "avg_viewport_height_ratio": avg_height_ratio,
    }

    return normalized_tracks, viewport_rows, summary
