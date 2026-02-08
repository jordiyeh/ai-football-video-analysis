"""Pass-map renderer with team/player/time filtering and node-edge overlays."""

from __future__ import annotations

import base64
from collections import Counter, defaultdict
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


PASS_MAP_SCHEMA_VERSION = "1.0"
UNKNOWN_TEAM = "unknown"

_DEFAULT_TEAM_COLORS_BGR = {
    "ours": (225, 125, 45),
    "opponent": (70, 70, 225),
    UNKNOWN_TEAM: (95, 210, 240),
}

_NORM_CANDIDATE_KEYS = (
    "norm_xy",
    "normalized_xy",
    "normalized_location",
    "norm_location",
)


def _safe_float(value: Any, default: float | None = None) -> float | None:
    """Safely cast finite value to float."""
    try:
        parsed = float(value)
    except Exception:
        return default
    if not np.isfinite(parsed):
        return default
    return parsed


def _safe_int(value: Any, default: int | None = None) -> int | None:
    """Safely cast value to int."""
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except Exception:
        return default


def _clip01(value: float) -> float:
    """Clamp value to [0, 1]."""
    return float(np.clip(value, 0.0, 1.0))


def _normalize_team_id(value: Any) -> str:
    """Normalize team labels into stable string identifiers."""
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


def _parse_xy(value: Any) -> tuple[float, float] | None:
    """Parse XY coordinates from list/tuple payloads."""
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    x = _safe_float(value[0], default=None)
    y = _safe_float(value[1], default=None)
    if x is None or y is None:
        return None
    return (x, y)


def _looks_normalized(xy: tuple[float, float]) -> bool:
    """Return True when XY appears normalized already."""
    return 0.0 <= xy[0] <= 1.0 and 0.0 <= xy[1] <= 1.0


def _resolve_frame_size(context: dict[str, Any]) -> tuple[float, float] | None:
    """Resolve frame width/height from context fields."""
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


def _parse_rgb_color(value: Any) -> tuple[int, int, int] | None:
    """Parse RGB color from hex string or list/tuple."""
    if isinstance(value, str):
        color = value.strip().lstrip("#")
        if len(color) != 6:
            return None
        try:
            red = int(color[0:2], 16)
            green = int(color[2:4], 16)
            blue = int(color[4:6], 16)
        except ValueError:
            return None
        return red, green, blue

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
    """Resolve team colors from context with defaults."""
    colors = dict(_DEFAULT_TEAM_COLORS_BGR)
    raw_colors = context.get("team_colors")
    if not isinstance(raw_colors, dict):
        return colors

    for raw_team, raw_value in raw_colors.items():
        team = _normalize_team_id(raw_team)
        rgb = _parse_rgb_color(raw_value)
        if rgb is None:
            continue
        colors[team] = _rgb_to_bgr(rgb)
    return colors


def _resolve_canvas_config(
    context: dict[str, Any],
    base_config: FieldCanvasConfig | None,
) -> FieldCanvasConfig:
    """Resolve field canvas dimensions from defaults + context overrides."""
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


def _encode_png_base64(image: np.ndarray) -> str:
    """Encode image array into base64 PNG."""
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        return ""
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _resolve_timestamp(row: dict[str, Any], fps: float | None) -> float | None:
    """Resolve timestamp from explicit fields or frame index and fps."""
    for key in ("timestamp", "t", "time_sec"):
        ts = _safe_float(row.get(key), default=None)
        if ts is not None:
            return ts

    frame_idx = _safe_int(row.get("frame_idx"), default=None)
    if frame_idx is None or fps is None or fps <= 0:
        return None
    return float(frame_idx) / fps


def _bbox_center(track: dict[str, Any]) -> tuple[float, float] | None:
    """Return bbox center when bbox exists and is valid."""
    bbox = track.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None

    x1 = _safe_float(bbox[0], default=None)
    y1 = _safe_float(bbox[1], default=None)
    x2 = _safe_float(bbox[2], default=None)
    y2 = _safe_float(bbox[3], default=None)
    if x1 is None or y1 is None or x2 is None or y2 is None:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _extract_image_xy(track: dict[str, Any]) -> tuple[float, float] | None:
    """Resolve image-space coordinates from explicit fields or bbox center."""
    image_x = _safe_float(track.get("image_x"), default=None)
    image_y = _safe_float(track.get("image_y"), default=None)
    if image_x is not None and image_y is not None:
        return (image_x, image_y)

    image_xy = _parse_xy(track.get("image_xy"))
    if image_xy is not None:
        return image_xy

    return _bbox_center(track)


def _resolve_norm_xy(
    row: dict[str, Any],
    frame_size: tuple[float, float] | None,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None, str]:
    """Resolve normalized coordinates from row fields or image fallback."""
    norm_x = _safe_float(row.get("norm_x"), default=None)
    norm_y = _safe_float(row.get("norm_y"), default=None)
    if norm_x is not None and norm_y is not None:
        return (_clip01(norm_x), _clip01(norm_y)), None, "track.norm_x_norm_y"

    for key in _NORM_CANDIDATE_KEYS:
        xy = _parse_xy(row.get(key))
        if xy is not None:
            return (_clip01(xy[0]), _clip01(xy[1])), None, f"track.{key}"

    image_xy = _extract_image_xy(row)
    if image_xy is None:
        return None, None, "missing_coordinates"

    if frame_size is not None:
        frame_w, frame_h = frame_size
        return (
            _clip01(image_xy[0] / frame_w),
            _clip01(image_xy[1] / frame_h),
        ), image_xy, "track.image_xy"

    if _looks_normalized(image_xy):
        return (_clip01(image_xy[0]), _clip01(image_xy[1])), None, "track.image_xy_as_norm"

    return None, image_xy, "missing_frame_size"


def _extract_track_id(row: dict[str, Any]) -> int | None:
    """Resolve track identifier from row payload."""
    return _safe_int(row.get("track_id"), default=None)


def _extract_player_id(row: dict[str, Any]) -> int | None:
    """Resolve player identifier from row payload."""
    for key in (
        "player_id",
        "assigned_player_id",
        "identity_player_id",
        "owner_player_id",
        "track_id",
    ):
        value = _safe_int(row.get(key), default=None)
        if value is not None:
            return value
    return None


def _extract_team_id(row: dict[str, Any]) -> str:
    """Resolve team label from common row fields."""
    for key in ("team_name", "team", "team_label", "team_id"):
        team = _normalize_team_id(row.get(key))
        if team != UNKNOWN_TEAM:
            return team
    return UNKNOWN_TEAM


def _extract_min_confidence(query: VisualizationQuery) -> float:
    """Resolve confidence threshold from query extras."""
    min_confidence = 0.0
    if query.extra:
        parsed = _safe_float(query.extra.get("min_confidence"), default=0.0)
        if parsed is not None:
            min_confidence = parsed
    return _clip01(min_confidence)


def _extract_min_pass_count(query: VisualizationQuery) -> int:
    """Resolve minimum pass-count threshold from query extras."""
    minimum = 1
    if query.extra:
        parsed = _safe_int(query.extra.get("min_pass_count"), default=1)
        if parsed is not None:
            minimum = parsed
    return max(1, int(minimum))


def _extract_endpoint_from_payload(
    payload: dict[str, Any],
    *,
    prefix: str,
    frame_size: tuple[float, float] | None,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None, str]:
    """Extract endpoint coordinates from event/metadata payload."""
    norm_keys = (
        f"{prefix}_norm_xy",
        f"{prefix}_normalized_xy",
        f"{prefix}_norm_location",
    )
    for key in norm_keys:
        xy = _parse_xy(payload.get(key))
        if xy is not None:
            return (_clip01(xy[0]), _clip01(xy[1])), None, key

    image_keys = (
        f"{prefix}_image_xy",
        f"{prefix}_xy",
        f"{prefix}_location",
    )
    for key in image_keys:
        xy = _parse_xy(payload.get(key))
        if xy is None:
            continue

        if frame_size is not None:
            frame_w, frame_h = frame_size
            return (
                _clip01(xy[0] / frame_w),
                _clip01(xy[1] / frame_h),
            ), xy, key

        if _looks_normalized(xy):
            return (_clip01(xy[0]), _clip01(xy[1])), None, f"{key}_as_norm"

        return None, xy, key

    return None, None, "missing"


@dataclass(slots=True)
class TrackSample:
    """Track sample row used for nearest endpoint lookups."""

    frame_idx: int | None
    timestamp: float | None
    track_id: int
    player_id: int | None
    team_id: str
    norm_xy: tuple[float, float]
    image_xy: tuple[float, float] | None


@dataclass(slots=True)
class PassPoint:
    """One resolved pass edge with endpoint coordinates."""

    frame_idx: int | None
    timestamp: float | None
    confidence: float
    team_id: str
    from_track_id: int | None
    to_track_id: int | None
    from_player_id: int | None
    to_player_id: int | None
    from_norm_xy: tuple[float, float]
    to_norm_xy: tuple[float, float]
    from_image_xy: tuple[float, float] | None
    to_image_xy: tuple[float, float] | None
    provenance: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize pass edge payload."""
        row: dict[str, Any] = {
            "confidence": float(self.confidence),
            "team_id": self.team_id,
            "from_norm_xy": [float(self.from_norm_xy[0]), float(self.from_norm_xy[1])],
            "to_norm_xy": [float(self.to_norm_xy[0]), float(self.to_norm_xy[1])],
            "provenance": self.provenance,
        }
        if self.frame_idx is not None:
            row["frame_idx"] = int(self.frame_idx)
        if self.timestamp is not None:
            row["timestamp"] = float(self.timestamp)
        if self.from_track_id is not None:
            row["from_track_id"] = int(self.from_track_id)
        if self.to_track_id is not None:
            row["to_track_id"] = int(self.to_track_id)
        if self.from_player_id is not None:
            row["from_player_id"] = int(self.from_player_id)
        if self.to_player_id is not None:
            row["to_player_id"] = int(self.to_player_id)
        if self.from_image_xy is not None:
            row["from_image_xy"] = [float(self.from_image_xy[0]), float(self.from_image_xy[1])]
        if self.to_image_xy is not None:
            row["to_image_xy"] = [float(self.to_image_xy[0]), float(self.to_image_xy[1])]
        if self.metadata:
            row["metadata"] = dict(self.metadata)
        return row


@dataclass(slots=True)
class PassEdgeSummary:
    """Aggregated pass edge used for drawing and export."""

    team_id: str
    from_track_id: int | None
    to_track_id: int | None
    from_player_id: int | None
    to_player_id: int | None
    pass_count: int
    avg_confidence: float
    from_norm_xy: tuple[float, float]
    to_norm_xy: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        """Serialize aggregated edge."""
        row: dict[str, Any] = {
            "team_id": self.team_id,
            "pass_count": int(self.pass_count),
            "avg_confidence": float(self.avg_confidence),
            "from_norm_xy": [float(self.from_norm_xy[0]), float(self.from_norm_xy[1])],
            "to_norm_xy": [float(self.to_norm_xy[0]), float(self.to_norm_xy[1])],
        }
        if self.from_track_id is not None:
            row["from_track_id"] = int(self.from_track_id)
        if self.to_track_id is not None:
            row["to_track_id"] = int(self.to_track_id)
        if self.from_player_id is not None:
            row["from_player_id"] = int(self.from_player_id)
        if self.to_player_id is not None:
            row["to_player_id"] = int(self.to_player_id)
        return row


@dataclass(slots=True)
class PassNodeSummary:
    """Aggregated pass-map node payload."""

    node_id: str
    team_id: str
    track_id: int | None
    player_id: int | None
    norm_xy: tuple[float, float]
    samples: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize node payload."""
        row: dict[str, Any] = {
            "node_id": self.node_id,
            "team_id": self.team_id,
            "norm_xy": [float(self.norm_xy[0]), float(self.norm_xy[1])],
            "samples": int(self.samples),
        }
        if self.track_id is not None:
            row["track_id"] = int(self.track_id)
        if self.player_id is not None:
            row["player_id"] = int(self.player_id)
        return row


def _build_track_samples(
    tracks: list[dict[str, Any]],
    *,
    frame_size: tuple[float, float] | None,
    fps: float | None,
) -> dict[int, list[TrackSample]]:
    """Build lookup of track_id -> sorted track samples."""
    lookup: dict[int, list[TrackSample]] = defaultdict(list)

    for row in tracks:
        if not isinstance(row, dict):
            continue

        object_type = str(row.get("object_type", "player")).strip().lower() or "player"
        if object_type != "player":
            continue

        track_id = _extract_track_id(row)
        if track_id is None:
            continue

        norm_xy, image_xy, _ = _resolve_norm_xy(row, frame_size)
        if norm_xy is None:
            continue

        sample = TrackSample(
            frame_idx=_safe_int(row.get("frame_idx"), default=None),
            timestamp=_resolve_timestamp(row, fps),
            track_id=int(track_id),
            player_id=_extract_player_id(row),
            team_id=_extract_team_id(row),
            norm_xy=norm_xy,
            image_xy=image_xy,
        )
        lookup[int(track_id)].append(sample)

    for samples in lookup.values():
        samples.sort(
            key=lambda sample: (
                sample.frame_idx if sample.frame_idx is not None else 10**12,
                sample.timestamp if sample.timestamp is not None else 10**12,
            )
        )

    return lookup


def _nearest_sample(
    samples: list[TrackSample] | None,
    *,
    target_frame: int | None,
    target_timestamp: float | None,
) -> TrackSample | None:
    """Return nearest track sample by frame or timestamp distance."""
    if not samples:
        return None

    def score(sample: TrackSample) -> tuple[float, float]:
        frame_delta = float("inf")
        if target_frame is not None and sample.frame_idx is not None:
            frame_delta = abs(float(sample.frame_idx - target_frame))

        time_delta = float("inf")
        if target_timestamp is not None and sample.timestamp is not None:
            time_delta = abs(float(sample.timestamp - target_timestamp))

        return (frame_delta, time_delta)

    return min(samples, key=score)


def _matches_query(point: PassPoint, query: VisualizationQuery, min_confidence: float) -> bool:
    """Return True when point satisfies active filters."""
    if point.confidence < min_confidence:
        return False

    if query.team_id is not None:
        if _normalize_team_id(query.team_id) != _normalize_team_id(point.team_id):
            return False

    if query.player_id is not None:
        candidate = int(query.player_id)
        matches = {
            point.from_player_id,
            point.to_player_id,
            point.from_track_id,
            point.to_track_id,
        }
        if candidate not in {value for value in matches if value is not None}:
            return False

    if query.start_t is not None:
        if point.timestamp is None or point.timestamp < float(query.start_t):
            return False

    if query.end_t is not None:
        if point.timestamp is None or point.timestamp > float(query.end_t):
            return False

    return True


def _resolve_endpoint_team(
    *,
    explicit_team: str,
    track_sample: TrackSample | None,
) -> str:
    """Resolve endpoint team from explicit event value or track fallback."""
    team = _normalize_team_id(explicit_team)
    if team != UNKNOWN_TEAM:
        return team
    if track_sample is not None:
        team = _normalize_team_id(track_sample.team_id)
        if team != UNKNOWN_TEAM:
            return team
    return UNKNOWN_TEAM


def _average_norm_xy(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Compute centroid of normalized points."""
    if not points:
        return (0.5, 0.5)
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (_clip01(float(np.mean(xs))), _clip01(float(np.mean(ys))))


def _aggregate_edges(
    points: list[PassPoint],
    *,
    min_pass_count: int,
) -> list[PassEdgeSummary]:
    """Aggregate raw pass points into unique edge summaries."""
    grouped: dict[tuple[str, int | None, int | None, int | None, int | None], list[PassPoint]] = defaultdict(list)

    for point in points:
        key = (
            point.team_id,
            point.from_track_id,
            point.to_track_id,
            point.from_player_id,
            point.to_player_id,
        )
        grouped[key].append(point)

    summaries: list[PassEdgeSummary] = []
    for key, rows in grouped.items():
        if len(rows) < min_pass_count:
            continue

        from_norm = _average_norm_xy([row.from_norm_xy for row in rows])
        to_norm = _average_norm_xy([row.to_norm_xy for row in rows])
        avg_conf = float(np.mean([row.confidence for row in rows])) if rows else 0.0

        summaries.append(
            PassEdgeSummary(
                team_id=key[0],
                from_track_id=key[1],
                to_track_id=key[2],
                from_player_id=key[3],
                to_player_id=key[4],
                pass_count=len(rows),
                avg_confidence=float(np.clip(avg_conf, 0.0, 1.0)),
                from_norm_xy=from_norm,
                to_norm_xy=to_norm,
            )
        )

    summaries.sort(
        key=lambda row: (
            row.pass_count,
            row.avg_confidence,
            row.team_id,
            row.from_track_id if row.from_track_id is not None else -1,
            row.to_track_id if row.to_track_id is not None else -1,
        )
    )
    return summaries


def _build_node_summaries(edges: list[PassEdgeSummary]) -> list[PassNodeSummary]:
    """Build node summaries from aggregated edges."""
    node_votes: dict[str, dict[str, Any]] = {}

    def update_node(
        *,
        team_id: str,
        track_id: int | None,
        player_id: int | None,
        norm_xy: tuple[float, float],
    ) -> None:
        if track_id is not None:
            node_id = f"track:{track_id}"
        elif player_id is not None:
            node_id = f"player:{player_id}"
        else:
            node_id = f"anon:{team_id}:{round(norm_xy[0], 3)}:{round(norm_xy[1], 3)}"

        bucket = node_votes.setdefault(
            node_id,
            {
                "team_id": team_id,
                "track_id": track_id,
                "player_id": player_id,
                "points": [],
            },
        )
        bucket["points"].append(norm_xy)
        if bucket.get("team_id") == UNKNOWN_TEAM and team_id != UNKNOWN_TEAM:
            bucket["team_id"] = team_id

    for edge in edges:
        update_node(
            team_id=edge.team_id,
            track_id=edge.from_track_id,
            player_id=edge.from_player_id,
            norm_xy=edge.from_norm_xy,
        )
        update_node(
            team_id=edge.team_id,
            track_id=edge.to_track_id,
            player_id=edge.to_player_id,
            norm_xy=edge.to_norm_xy,
        )

    summaries: list[PassNodeSummary] = []
    for node_id, bucket in node_votes.items():
        points = list(bucket["points"])
        norm_xy = _average_norm_xy(points)
        summaries.append(
            PassNodeSummary(
                node_id=node_id,
                team_id=_normalize_team_id(bucket.get("team_id")),
                track_id=_safe_int(bucket.get("track_id"), default=None),
                player_id=_safe_int(bucket.get("player_id"), default=None),
                norm_xy=norm_xy,
                samples=len(points),
            )
        )

    summaries.sort(
        key=lambda node: (
            node.team_id,
            -(node.samples),
            node.track_id if node.track_id is not None else 10**12,
            node.player_id if node.player_id is not None else 10**12,
        )
    )
    return summaries


class PassMapRenderer(VisualizationRenderer):
    """Render pass maps from inferred pass events and tracking data."""

    visualization_type = "pass_map"
    schema_version = PASS_MAP_SCHEMA_VERSION

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
        """Render pass-map artifact with edge + node payloads."""
        query = query or VisualizationQuery()
        context = dict(context or {})

        canvas_config = _resolve_canvas_config(context, self._canvas_config)
        field_canvas = FieldCanvas(config=canvas_config)
        canvas = field_canvas.blank(include_markings=self._include_markings)

        frame_size = _resolve_frame_size(context)
        fps = _safe_float(context.get("fps"), default=None)
        track_lookup = _build_track_samples(tracks or [], frame_size=frame_size, fps=fps)

        min_confidence = _extract_min_confidence(query)
        min_pass_count = _extract_min_pass_count(query)
        team_colors = _resolve_team_colors(context)

        normalized_events = [_event_to_dict(event) for event in (events or [])]

        passes_seen = 0
        passes_skipped_filters = 0
        passes_skipped_missing_coordinates = 0
        provenance_counter: Counter[str] = Counter()

        pass_points: list[PassPoint] = []

        for event in normalized_events:
            event_type = str(event.get("event_type", "")).strip().lower()
            if event_type != "pass":
                continue

            passes_seen += 1
            metadata = event["metadata"]

            frame_idx = _safe_int(event.get("frame_idx"), default=None)
            timestamp = _safe_float(event.get("timestamp"), default=None)
            confidence = _safe_float(event.get("confidence"), default=0.0)
            confidence = _clip01(float(confidence or 0.0))

            from_track_id = _safe_int(metadata.get("from_track_id"), default=None)
            to_track_id = _safe_int(metadata.get("to_track_id"), default=None)
            from_player_id = _safe_int(metadata.get("from_player_id"), default=None)
            to_player_id = _safe_int(metadata.get("to_player_id"), default=None)

            if from_player_id is None:
                from_player_id = _safe_int(metadata.get("passer_player_id"), default=None)
            if to_player_id is None:
                to_player_id = _safe_int(metadata.get("receiver_player_id"), default=None)

            explicit_team = _normalize_team_id(metadata.get("team_id"))

            event_from_norm, event_from_image, event_from_source = _extract_endpoint_from_payload(
                event,
                prefix="from",
                frame_size=frame_size,
            )
            meta_from_norm, meta_from_image, meta_from_source = _extract_endpoint_from_payload(
                metadata,
                prefix="from",
                frame_size=frame_size,
            )
            from_norm = event_from_norm or meta_from_norm
            from_image = event_from_image or meta_from_image
            from_source = "event." + event_from_source if event_from_norm is not None else "metadata." + meta_from_source

            event_to_norm, event_to_image, event_to_source = _extract_endpoint_from_payload(
                event,
                prefix="to",
                frame_size=frame_size,
            )
            meta_to_norm, meta_to_image, meta_to_source = _extract_endpoint_from_payload(
                metadata,
                prefix="to",
                frame_size=frame_size,
            )
            to_norm = event_to_norm or meta_to_norm
            to_image = event_to_image or meta_to_image
            to_source = "event." + event_to_source if event_to_norm is not None else "metadata." + meta_to_source

            from_sample = _nearest_sample(
                track_lookup.get(from_track_id),
                target_frame=frame_idx,
                target_timestamp=timestamp,
            )
            to_sample = _nearest_sample(
                track_lookup.get(to_track_id),
                target_frame=frame_idx,
                target_timestamp=timestamp,
            )

            if from_norm is None and from_sample is not None:
                from_norm = from_sample.norm_xy
                from_image = from_sample.image_xy
                from_source = "track.from_lookup"
            if to_norm is None and to_sample is not None:
                to_norm = to_sample.norm_xy
                to_image = to_sample.image_xy
                to_source = "track.to_lookup"

            if from_norm is None or to_norm is None:
                passes_skipped_missing_coordinates += 1
                provenance_counter["missing_coordinates"] += 1
                continue

            team_id = _resolve_endpoint_team(
                explicit_team=explicit_team,
                track_sample=from_sample or to_sample,
            )

            point = PassPoint(
                frame_idx=frame_idx,
                timestamp=timestamp,
                confidence=confidence,
                team_id=team_id,
                from_track_id=from_track_id,
                to_track_id=to_track_id,
                from_player_id=from_player_id,
                to_player_id=to_player_id,
                from_norm_xy=from_norm,
                to_norm_xy=to_norm,
                from_image_xy=from_image,
                to_image_xy=to_image,
                provenance=f"{from_source}->{to_source}",
                metadata={
                    "gap_seconds": _safe_float(metadata.get("gap_seconds"), default=None),
                    "gap_frames": _safe_int(metadata.get("gap_frames"), default=None),
                    "event_family": metadata.get("event_family"),
                    "provenance": metadata.get("provenance"),
                },
            )

            if _matches_query(point, query, min_confidence=min_confidence):
                pass_points.append(point)
                provenance_counter[point.provenance] += 1
            else:
                passes_skipped_filters += 1

        edge_summaries = _aggregate_edges(pass_points, min_pass_count=min_pass_count)
        node_summaries = _build_node_summaries(edge_summaries)

        for edge in edge_summaries:
            color = team_colors.get(_normalize_team_id(edge.team_id), team_colors[UNKNOWN_TEAM])
            start = field_canvas.norm_to_pixel(edge.from_norm_xy[0], edge.from_norm_xy[1])
            end = field_canvas.norm_to_pixel(edge.to_norm_xy[0], edge.to_norm_xy[1])
            if start == end:
                continue

            thickness = int(np.clip(1 + edge.pass_count, 1, 8))
            tip_length = float(np.clip(0.06 + (0.01 * edge.pass_count), 0.08, 0.24))
            cv2.arrowedLine(
                canvas,
                start,
                end,
                color,
                thickness=thickness,
                tipLength=tip_length,
                line_type=cv2.LINE_AA,
            )

        for node in node_summaries:
            color = team_colors.get(_normalize_team_id(node.team_id), team_colors[UNKNOWN_TEAM])
            px, py = field_canvas.norm_to_pixel(node.norm_xy[0], node.norm_xy[1])
            radius = int(np.clip(4 + np.log1p(node.samples) * 2.0, 4, 12))
            cv2.circle(canvas, (px, py), radius + 1, (245, 245, 245), thickness=-1)
            cv2.circle(canvas, (px, py), radius, color, thickness=-1)
            cv2.circle(canvas, (px, py), max(2, radius // 3), (28, 28, 28), thickness=-1)

        pass_total = int(sum(edge.pass_count for edge in edge_summaries))
        teams = sorted({edge.team_id for edge in edge_summaries})
        player_ids = sorted(
            {
                player_id
                for edge in edge_summaries
                for player_id in (edge.from_player_id, edge.to_player_id)
                if player_id is not None
            }
        )
        track_ids = sorted(
            {
                track_id
                for edge in edge_summaries
                for track_id in (edge.from_track_id, edge.to_track_id)
                if track_id is not None
            }
        )

        if query.player_id is not None:
            title = f"Pass Map - Player {int(query.player_id)}"
        elif query.team_id is not None:
            title = f"Pass Map - {str(query.team_id)}"
        else:
            title = "Pass Map - All Teams"

        payload = {
            "encoding": "png_base64",
            "image_png_base64": _encode_png_base64(canvas),
            "edges": [edge.to_dict() for edge in edge_summaries],
            "nodes": [node.to_dict() for node in node_summaries],
            "totals": {
                "passes": pass_total,
                "edges": len(edge_summaries),
                "nodes": len(node_summaries),
            },
        }
        metadata = {
            "passes_seen": passes_seen,
            "passes_rendered": len(pass_points),
            "edges_rendered": len(edge_summaries),
            "nodes_rendered": len(node_summaries),
            "passes_skipped_filter": passes_skipped_filters,
            "passes_skipped_missing_coordinates": passes_skipped_missing_coordinates,
            "teams": teams,
            "player_ids": player_ids,
            "track_ids": track_ids,
            "min_pass_count": min_pass_count,
            "coordinate_provenance": dict(sorted(provenance_counter.items())),
        }

        return self.build_artifact(
            title=title,
            width=canvas_config.width,
            height=canvas_config.height,
            query=query,
            metadata=metadata,
            payload=payload,
        )


def build_pass_map(
    *,
    tracks: list[dict[str, Any]],
    events: list[dict[str, Any]] | None = None,
    query: VisualizationQuery | None = None,
    context: dict[str, Any] | None = None,
    renderer: PassMapRenderer | None = None,
) -> VisualizationArtifact:
    """Functional helper to render pass-map artifacts."""
    map_renderer = renderer or PassMapRenderer()
    return map_renderer.render(tracks=tracks, events=events, query=query, context=context)
