"""Shot-map renderer with team/player/time filtering support."""

from __future__ import annotations

import base64
from collections import Counter
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from src.export.visualizations.base import (
    VisualizationArtifact,
    VisualizationQuery,
    VisualizationRenderer,
)
from src.export.visualizations.field_canvas import FieldCanvas, FieldCanvasConfig


SHOT_MAP_SCHEMA_VERSION = "1.0"
UNKNOWN_TEAM = "unknown"

_DEFAULT_TEAM_COLORS_BGR = {
    "ours": (225, 125, 45),
    "opponent": (70, 70, 225),
    UNKNOWN_TEAM: (95, 210, 240),
}


def _safe_float(value: Any, default: float | None = None) -> float | None:
    """Cast to finite float with fallback."""
    try:
        parsed = float(value)
    except Exception:
        return default
    if not np.isfinite(parsed):
        return default
    return parsed


def _safe_int(value: Any, default: int | None = None) -> int | None:
    """Cast to int with fallback."""
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except Exception:
        return default


def _clip01(value: float) -> float:
    """Clamp scalar into [0, 1]."""
    return float(np.clip(value, 0.0, 1.0))


def _normalize_team_id(value: Any) -> str:
    """Normalize arbitrary team labels into stable values."""
    if value is None:
        return UNKNOWN_TEAM
    if isinstance(value, bool):
        return UNKNOWN_TEAM
    if isinstance(value, int):
        return f"team_{value}" if value >= 0 else UNKNOWN_TEAM

    text = str(value).strip()
    if not text:
        return UNKNOWN_TEAM

    lowered = text.lower()
    if lowered in {"none", "null", "unknown", "-1"}:
        return UNKNOWN_TEAM
    if lowered in {"ours", "opponent"}:
        return lowered
    if lowered.isdigit():
        numeric = _safe_int(lowered, default=None)
        if numeric is None or numeric < 0:
            return UNKNOWN_TEAM
        return f"team_{numeric}"
    return lowered


def _event_to_dict(event: Any) -> dict[str, Any]:
    """Normalize dict/dataclass-like events into dictionary payload."""
    if isinstance(event, dict):
        row = dict(event)
    else:
        row = {
            "event_type": getattr(event, "event_type", None),
            "frame_idx": getattr(event, "frame_idx", None),
            "timestamp": getattr(event, "timestamp", None),
            "confidence": getattr(event, "confidence", None),
            "location": getattr(event, "location", None),
            "metadata": getattr(event, "metadata", None),
        }

    metadata = row.get("metadata")
    row["metadata"] = dict(metadata) if isinstance(metadata, dict) else {}
    return row


def _parse_xy(value: Any) -> tuple[float, float] | None:
    """Parse an XY pair from list/tuple values."""
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None

    x = _safe_float(value[0], default=None)
    y = _safe_float(value[1], default=None)
    if x is None or y is None:
        return None
    return (x, y)


def _looks_normalized(xy: tuple[float, float]) -> bool:
    """Return True when an XY pair appears normalized already."""
    return 0.0 <= xy[0] <= 1.0 and 0.0 <= xy[1] <= 1.0


def _resolve_frame_size(context: dict[str, Any]) -> tuple[float, float] | None:
    """Resolve frame width/height from visualization context."""
    for width_key, height_key in (
        ("frame_width", "frame_height"),
        ("video_width", "video_height"),
    ):
        width = _safe_float(context.get(width_key), default=None)
        height = _safe_float(context.get(height_key), default=None)
        if width is not None and height is not None and width > 1 and height > 1:
            return (width, height)

    frame_shape = context.get("frame_shape")
    if isinstance(frame_shape, (list, tuple)) and len(frame_shape) >= 2:
        height = _safe_float(frame_shape[0], default=None)
        width = _safe_float(frame_shape[1], default=None)
        if width is not None and height is not None and width > 1 and height > 1:
            return (width, height)

    return None


def _extract_norm_xy(
    event: dict[str, Any],
    frame_size: tuple[float, float] | None,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None, str]:
    """Extract normalized and image-space coordinates from one event."""
    metadata = event["metadata"]

    for key in ("norm_xy", "normalized_xy", "normalized_location", "norm_location"):
        xy = _parse_xy(event.get(key))
        if xy is not None:
            return (_clip01(xy[0]), _clip01(xy[1])), None, f"event.{key}"
        xy = _parse_xy(metadata.get(key))
        if xy is not None:
            return (_clip01(xy[0]), _clip01(xy[1])), None, f"metadata.{key}"

    image_xy: tuple[float, float] | None = None
    provenance = "missing_location"
    for key in ("location", "image_xy"):
        candidate = _parse_xy(event.get(key))
        source = f"event.{key}"
        if candidate is None:
            candidate = _parse_xy(metadata.get(key))
            source = f"metadata.{key}"
        if candidate is None:
            continue
        image_xy = candidate
        provenance = source
        break

    if image_xy is None:
        return None, None, provenance

    if frame_size is not None:
        frame_w, frame_h = frame_size
        norm_x = _clip01(image_xy[0] / frame_w)
        norm_y = _clip01(image_xy[1] / frame_h)
        return (norm_x, norm_y), image_xy, provenance

    if _looks_normalized(image_xy):
        return (_clip01(image_xy[0]), _clip01(image_xy[1])), None, f"{provenance}_as_norm"

    return None, image_xy, provenance


def _extract_player_id(event: dict[str, Any]) -> int | None:
    """Resolve player identifier from common event fields."""
    metadata = event["metadata"]
    for key in (
        "player_id",
        "shooter_player_id",
        "kick_player_id",
        "track_id",
        "shooter_track_id",
        "from_track_id",
    ):
        value = _safe_int(metadata.get(key), default=None)
        if value is not None:
            return value

    for key in ("player_id", "track_id"):
        value = _safe_int(event.get(key), default=None)
        if value is not None:
            return value

    return None


def _extract_team_id(event: dict[str, Any], track_team_lookup: dict[int, str]) -> str:
    """Resolve team from event payload fields with track fallback."""
    metadata = event["metadata"]

    for key in ("team_id", "team_name", "owner_team", "attacking_team", "scoring_team"):
        team = _normalize_team_id(event.get(key))
        if team != UNKNOWN_TEAM:
            return team
        team = _normalize_team_id(metadata.get(key))
        if team != UNKNOWN_TEAM:
            return team

    for key in ("track_id", "player_id", "shooter_track_id", "kick_player_id", "from_track_id"):
        track_id = _safe_int(metadata.get(key), default=None)
        if track_id is None:
            track_id = _safe_int(event.get(key), default=None)
        if track_id is None:
            continue
        team = _normalize_team_id(track_team_lookup.get(track_id))
        if team != UNKNOWN_TEAM:
            return team

    return UNKNOWN_TEAM


def _build_track_team_lookup(tracks: list[dict[str, Any]]) -> dict[int, str]:
    """Build best-effort track_id -> team map from track rows."""
    votes: dict[int, Counter[str]] = {}

    for track in tracks:
        if not isinstance(track, dict):
            continue
        object_type = str(track.get("object_type", "")).strip().lower()
        if object_type and object_type != "player":
            continue

        track_id = _safe_int(track.get("track_id"), default=None)
        if track_id is None:
            continue

        team = _normalize_team_id(track.get("team_name"))
        if team == UNKNOWN_TEAM:
            team = _normalize_team_id(track.get("team_id"))
        if team == UNKNOWN_TEAM:
            continue

        counter = votes.setdefault(track_id, Counter())
        counter[team] += 1

    lookup: dict[int, str] = {}
    for track_id, counter in votes.items():
        if counter:
            lookup[track_id] = counter.most_common(1)[0][0]
    return lookup


def _collect_goal_linked_shot_frames(events: list[dict[str, Any]]) -> set[int]:
    """Collect shot frame indices linked from goal events."""
    linked: set[int] = set()
    for event in events:
        if str(event.get("event_type", "")).strip().lower() != "goal":
            continue
        metadata = event["metadata"]
        shot_frame = _safe_int(metadata.get("shot_frame"), default=None)
        if shot_frame is not None:
            linked.add(shot_frame)
    return linked


def _parse_rgb_color(value: Any) -> tuple[int, int, int] | None:
    """Parse RGB color from hex string or tuple/list."""
    if isinstance(value, str):
        color = value.strip().lstrip("#")
        if len(color) == 6:
            try:
                red = int(color[0:2], 16)
                green = int(color[2:4], 16)
                blue = int(color[4:6], 16)
            except ValueError:
                return None
            return red, green, blue
        return None

    if isinstance(value, (list, tuple)) and len(value) >= 3:
        red = _safe_int(value[0], default=None)
        green = _safe_int(value[1], default=None)
        blue = _safe_int(value[2], default=None)
        if red is None or green is None or blue is None:
            return None
        return (
            int(np.clip(red, 0, 255)),
            int(np.clip(green, 0, 255)),
            int(np.clip(blue, 0, 255)),
        )

    return None


def _rgb_to_bgr(color: tuple[int, int, int]) -> tuple[int, int, int]:
    """Convert RGB tuple to OpenCV BGR ordering."""
    return (color[2], color[1], color[0])


def _resolve_team_colors(context: dict[str, Any]) -> dict[str, tuple[int, int, int]]:
    """Resolve team colors from context while preserving defaults."""
    colors = dict(_DEFAULT_TEAM_COLORS_BGR)
    raw = context.get("team_colors")
    if not isinstance(raw, dict):
        return colors

    for raw_key, raw_value in raw.items():
        team = _normalize_team_id(raw_key)
        rgb = _parse_rgb_color(raw_value)
        if rgb is None:
            continue
        colors[team] = _rgb_to_bgr(rgb)

    return colors


def _encode_png_base64(image: np.ndarray) -> str:
    """Encode image buffer to base64 PNG."""
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        return ""
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _resolve_canvas_config(
    context: dict[str, Any],
    base_config: FieldCanvasConfig | None,
) -> FieldCanvasConfig:
    """Resolve field canvas config from constructor defaults + context overrides."""
    if base_config is None:
        base_config = FieldCanvasConfig()

    canvas_width = _safe_int(context.get("canvas_width"), default=base_config.width)
    canvas_height = _safe_int(context.get("canvas_height"), default=base_config.height)
    canvas_padding = _safe_int(context.get("canvas_padding"), default=base_config.padding)
    width = max(64, int(canvas_width or base_config.width))
    height = max(64, int(canvas_height or base_config.height))
    max_padding = max(0, (min(width, height) // 2) - 1)
    padding = int(np.clip(int(canvas_padding or base_config.padding), 0, max_padding))

    return FieldCanvasConfig(
        width=width,
        height=height,
        padding=padding,
        background_color=base_config.background_color,
        pitch_color=base_config.pitch_color,
        line_color=base_config.line_color,
        line_thickness=base_config.line_thickness,
    )


def _extract_min_confidence(query: VisualizationQuery) -> float:
    """Resolve optional confidence floor from query extras."""
    min_confidence = 0.0
    if query.extra:
        parsed = _safe_float(query.extra.get("min_confidence"), default=0.0)
        if parsed is not None:
            min_confidence = parsed
    return _clip01(min_confidence)


def _matches_query(point: "ShotPoint", query: VisualizationQuery, min_confidence: float) -> bool:
    """Evaluate whether a shot point satisfies all active filters."""
    if point.confidence < min_confidence:
        return False

    if query.team_id is not None:
        team_filter = _normalize_team_id(query.team_id)
        if _normalize_team_id(point.team_id) != team_filter:
            return False

    if query.player_id is not None:
        if point.player_id is None or int(query.player_id) != int(point.player_id):
            return False

    if query.start_t is not None:
        if point.timestamp is None or point.timestamp < float(query.start_t):
            return False

    if query.end_t is not None:
        if point.timestamp is None or point.timestamp > float(query.end_t):
            return False

    return True


@dataclass(slots=True)
class ShotPoint:
    """Normalized shot event payload used in map rendering + exports."""

    frame_idx: int | None
    timestamp: float | None
    confidence: float
    team_id: str
    player_id: int | None
    norm_xy: tuple[float, float]
    image_xy: tuple[float, float] | None
    is_goal: bool
    source_event_type: str
    provenance: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize shot point to JSON-safe dictionary."""
        row: dict[str, Any] = {
            "confidence": float(self.confidence),
            "team_id": self.team_id,
            "norm_xy": [float(self.norm_xy[0]), float(self.norm_xy[1])],
            "is_goal": bool(self.is_goal),
            "source_event_type": self.source_event_type,
            "provenance": self.provenance,
        }
        if self.frame_idx is not None:
            row["frame_idx"] = int(self.frame_idx)
        if self.timestamp is not None:
            row["timestamp"] = float(self.timestamp)
        if self.player_id is not None:
            row["player_id"] = int(self.player_id)
        if self.image_xy is not None:
            row["image_xy"] = [float(self.image_xy[0]), float(self.image_xy[1])]
        if self.metadata:
            row["metadata"] = dict(self.metadata)
        return row


class ShotMapRenderer(VisualizationRenderer):
    """Render shot maps from event streams with configurable filters."""

    visualization_type = "shot_map"
    schema_version = SHOT_MAP_SCHEMA_VERSION

    def __init__(
        self,
        *,
        canvas_config: FieldCanvasConfig | None = None,
        include_markings: bool = True,
    ):
        self._canvas_config = canvas_config
        self._include_markings = include_markings

    def render(
        self,
        *,
        tracks: list[dict[str, Any]],
        events: list[dict[str, Any]] | None = None,
        query: VisualizationQuery | None = None,
        context: dict[str, Any] | None = None,
    ) -> VisualizationArtifact:
        """Render shot map artifact and return encoded image + point payload."""
        query = query or VisualizationQuery()
        context = dict(context or {})

        canvas_config = _resolve_canvas_config(context, self._canvas_config)
        field_canvas = FieldCanvas(config=canvas_config)
        canvas = field_canvas.blank(include_markings=self._include_markings)

        normalized_events = [_event_to_dict(event) for event in (events or [])]
        goal_linked_shot_frames = _collect_goal_linked_shot_frames(normalized_events)
        track_team_lookup = _build_track_team_lookup(tracks or [])
        frame_size = _resolve_frame_size(context)
        min_confidence = _extract_min_confidence(query)
        team_colors = _resolve_team_colors(context)

        shots_seen = 0
        skipped_missing_coordinates = 0
        shot_points: list[ShotPoint] = []

        for event in normalized_events:
            event_type = str(event.get("event_type", "")).strip().lower()
            if event_type != "shot":
                continue

            shots_seen += 1
            norm_xy, image_xy, coordinate_source = _extract_norm_xy(event, frame_size)
            if norm_xy is None:
                skipped_missing_coordinates += 1
                continue

            metadata = event["metadata"]
            frame_idx = _safe_int(event.get("frame_idx"), default=None)
            timestamp = _safe_float(event.get("timestamp"), default=None)
            confidence = _safe_float(event.get("confidence"), default=0.0)
            player_id = _extract_player_id(event)
            team_id = _extract_team_id(event, track_team_lookup)

            is_goal = bool(metadata.get("is_goal"))
            if not is_goal:
                outcome = str(metadata.get("outcome", "")).strip().lower()
                is_goal = outcome == "goal"
            if not is_goal and frame_idx is not None and frame_idx in goal_linked_shot_frames:
                is_goal = True

            shot = ShotPoint(
                frame_idx=frame_idx,
                timestamp=timestamp,
                confidence=max(0.0, min(1.0, float(confidence or 0.0))),
                team_id=team_id,
                player_id=player_id,
                norm_xy=norm_xy,
                image_xy=image_xy,
                is_goal=is_goal,
                source_event_type=event_type,
                provenance=str(metadata.get("provenance") or coordinate_source),
                metadata={
                    "event_family": metadata.get("event_family"),
                    "target_goal": metadata.get("target_goal"),
                },
            )

            if _matches_query(shot, query, min_confidence=min_confidence):
                shot_points.append(shot)

        shot_points.sort(key=lambda item: item.confidence)
        for shot in shot_points:
            px, py = field_canvas.norm_to_pixel(shot.norm_xy[0], shot.norm_xy[1])
            color = team_colors.get(_normalize_team_id(shot.team_id), team_colors[UNKNOWN_TEAM])
            radius = int(np.clip(4 + (shot.confidence * 6), 4, 12))

            if shot.is_goal:
                cv2.circle(canvas, (px, py), radius + 2, (248, 248, 248), thickness=2)
                cv2.circle(canvas, (px, py), radius, color, thickness=-1)
            else:
                cv2.circle(canvas, (px, py), radius, color, thickness=2)
                cv2.circle(canvas, (px, py), max(2, radius // 3), color, thickness=-1)

        goal_count = sum(1 for item in shot_points if item.is_goal)
        teams = sorted({item.team_id for item in shot_points})
        player_ids = sorted({int(item.player_id) for item in shot_points if item.player_id is not None})

        if query.player_id is not None:
            title = f"Shot Map - Player {int(query.player_id)}"
        elif query.team_id is not None:
            title = f"Shot Map - {str(query.team_id)}"
        else:
            title = "Shot Map - All"

        payload_points = [item.to_dict() for item in shot_points]
        payload = {
            "encoding": "png_base64",
            "image_png_base64": _encode_png_base64(canvas),
            "points": payload_points,
            "totals": {
                "shots": len(payload_points),
                "goals": goal_count,
            },
        }
        metadata = {
            "shots_seen": shots_seen,
            "shots_rendered": len(payload_points),
            "shots_skipped_missing_coordinates": skipped_missing_coordinates,
            "goals_rendered": goal_count,
            "teams": teams,
            "player_ids": player_ids,
        }

        return self.build_artifact(
            title=title,
            width=canvas_config.width,
            height=canvas_config.height,
            query=query,
            metadata=metadata,
            payload=payload,
        )


def build_shot_map(
    *,
    tracks: list[dict[str, Any]],
    events: list[dict[str, Any]] | None = None,
    query: VisualizationQuery | None = None,
    context: dict[str, Any] | None = None,
    renderer: ShotMapRenderer | None = None,
) -> VisualizationArtifact:
    """Functional helper for callers that do not need renderer lifecycle control."""
    map_renderer = renderer or ShotMapRenderer()
    return map_renderer.render(tracks=tracks, events=events, query=query, context=context)
