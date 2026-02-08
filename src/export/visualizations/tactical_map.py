"""Tactical-map renderer for live team shapes and territory overlays."""

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


TACTICAL_MAP_SCHEMA_VERSION = "1.0"
UNKNOWN_TEAM = "unknown"

_DEFAULT_TEAM_COLORS_BGR = {
    "ours": (225, 125, 45),
    "opponent": (70, 70, 225),
    UNKNOWN_TEAM: (120, 190, 215),
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
    """Clamp value into [0, 1]."""
    return float(np.clip(value, 0.0, 1.0))


def _normalize_team_id(value: Any) -> str:
    """Normalize team labels into stable identifiers."""
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
    """Parse XY tuple from list/tuple payloads."""
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


def _bbox_center(row: dict[str, Any]) -> tuple[float, float] | None:
    """Resolve bbox center coordinates."""
    bbox = row.get("bbox")
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


def _extract_image_xy(row: dict[str, Any]) -> tuple[float, float] | None:
    """Resolve image-space coordinates from explicit fields or bbox center."""
    image_x = _safe_float(row.get("image_x"), default=None)
    image_y = _safe_float(row.get("image_y"), default=None)
    if image_x is not None and image_y is not None:
        return (image_x, image_y)

    image_xy = _parse_xy(row.get("image_xy"))
    if image_xy is not None:
        return image_xy

    return _bbox_center(row)


def _resolve_norm_xy(
    row: dict[str, Any],
    frame_size: tuple[float, float] | None,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None, str]:
    """Resolve normalized coordinates from row payload."""
    norm_x = _safe_float(row.get("norm_x"), default=None)
    norm_y = _safe_float(row.get("norm_y"), default=None)
    if norm_x is not None and norm_y is not None:
        return (_clip01(norm_x), _clip01(norm_y)), None, "track.norm_x_norm_y"

    for key in _NORM_CANDIDATE_KEYS:
        norm_xy = _parse_xy(row.get(key))
        if norm_xy is not None:
            return (_clip01(norm_xy[0]), _clip01(norm_xy[1])), None, f"track.{key}"

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


def _resolve_timestamp(row: dict[str, Any], fps: float | None) -> float | None:
    """Resolve timestamp from row fields or frame/fps."""
    for key in ("timestamp", "t", "time_sec"):
        ts = _safe_float(row.get(key), default=None)
        if ts is not None:
            return ts

    frame_idx = _safe_int(row.get("frame_idx"), default=None)
    if frame_idx is None or fps is None or fps <= 0:
        return None
    return float(frame_idx) / fps


def _extract_team_id(row: dict[str, Any]) -> str:
    """Resolve team identifier from row payload."""
    for key in ("team_name", "team", "team_label", "team_id"):
        team = _normalize_team_id(row.get(key))
        if team != UNKNOWN_TEAM:
            return team
    return UNKNOWN_TEAM


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


def _extract_min_confidence(query: VisualizationQuery) -> float:
    """Resolve optional confidence filter from query extras."""
    min_confidence = 0.0
    if query.extra:
        parsed = _safe_float(query.extra.get("min_confidence"), default=0.0)
        if parsed is not None:
            min_confidence = parsed
    return _clip01(min_confidence)


def _extract_include_points(query: VisualizationQuery) -> bool:
    """Resolve whether to include sample points in payload."""
    if not query.extra:
        return False
    return bool(query.extra.get("include_points"))


def _extract_max_tracks_per_team(query: VisualizationQuery) -> int:
    """Resolve max tracks per team to display."""
    max_tracks = 11
    if query.extra:
        parsed = _safe_int(query.extra.get("max_tracks_per_team"), default=11)
        if parsed is not None:
            max_tracks = parsed
    return max(1, int(max_tracks))


def _extract_min_samples_per_track(query: VisualizationQuery) -> int:
    """Resolve minimum sample count per track."""
    minimum = 3
    if query.extra:
        parsed = _safe_int(query.extra.get("min_samples_per_track"), default=3)
        if parsed is not None:
            minimum = parsed
    return max(1, int(minimum))


def _extract_include_territory(query: VisualizationQuery) -> bool:
    """Resolve territory overlay toggle."""
    if not query.extra:
        return True
    value = query.extra.get("include_territory")
    if value is None:
        return True
    return bool(value)


def _extract_include_pressing(query: VisualizationQuery) -> bool:
    """Resolve pressing legend toggle."""
    if not query.extra:
        return True
    value = query.extra.get("include_pressing")
    if value is None:
        return True
    return bool(value)


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
    """Convert RGB tuple to OpenCV BGR tuple."""
    return (color[2], color[1], color[0])


def _resolve_team_colors(context: dict[str, Any]) -> dict[str, tuple[int, int, int]]:
    """Resolve team colors from context and defaults."""
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
    """Encode image as base64 PNG string."""
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        return ""
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _matches_query(point: "TacticalPoint", query: VisualizationQuery, min_confidence: float) -> bool:
    """Return True when tactical sample satisfies query filters."""
    if point.confidence < min_confidence:
        return False

    if query.team_id is not None:
        if _normalize_team_id(query.team_id) != _normalize_team_id(point.team_id):
            return False

    if query.player_id is not None:
        candidate = int(query.player_id)
        matches = {point.player_id, point.track_id}
        if candidate not in {value for value in matches if value is not None}:
            return False

    if query.start_t is not None:
        if point.timestamp is None or point.timestamp < float(query.start_t):
            return False

    if query.end_t is not None:
        if point.timestamp is None or point.timestamp > float(query.end_t):
            return False

    return True


@dataclass(slots=True)
class TacticalPoint:
    """Track sample used by tactical-map shape inference."""

    frame_idx: int | None
    timestamp: float | None
    track_id: int | None
    player_id: int | None
    team_id: str
    confidence: float
    norm_xy: tuple[float, float]
    image_xy: tuple[float, float] | None
    provenance: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize tactical sample."""
        row: dict[str, Any] = {
            "team_id": self.team_id,
            "confidence": float(self.confidence),
            "norm_xy": [float(self.norm_xy[0]), float(self.norm_xy[1])],
            "provenance": self.provenance,
        }
        if self.frame_idx is not None:
            row["frame_idx"] = int(self.frame_idx)
        if self.timestamp is not None:
            row["timestamp"] = float(self.timestamp)
        if self.track_id is not None:
            row["track_id"] = int(self.track_id)
        if self.player_id is not None:
            row["player_id"] = int(self.player_id)
        if self.image_xy is not None:
            row["image_xy"] = [float(self.image_xy[0]), float(self.image_xy[1])]
        return row


@dataclass(slots=True)
class TacticalTrackNode:
    """Aggregated node for one player track."""

    track_id: int
    player_id: int | None
    team_id: str
    samples: int
    confidence_avg: float
    norm_xy: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        """Serialize track node payload."""
        row: dict[str, Any] = {
            "track_id": int(self.track_id),
            "team_id": self.team_id,
            "samples": int(self.samples),
            "confidence_avg": float(self.confidence_avg),
            "norm_xy": [float(self.norm_xy[0]), float(self.norm_xy[1])],
        }
        if self.player_id is not None:
            row["player_id"] = int(self.player_id)
        return row


@dataclass(slots=True)
class TacticalTeamShape:
    """Aggregated tactical shape per team."""

    team_id: str
    centroid_norm: tuple[float, float]
    spread_norm: tuple[float, float]
    bbox_norm: tuple[float, float, float, float]
    samples: int
    unique_tracks: int
    nodes: list[TacticalTrackNode]
    territory: dict[str, float]
    pressing: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize team shape payload."""
        return {
            "team_id": self.team_id,
            "centroid_norm": [float(self.centroid_norm[0]), float(self.centroid_norm[1])],
            "spread_norm": [float(self.spread_norm[0]), float(self.spread_norm[1])],
            "bbox_norm": [
                float(self.bbox_norm[0]),
                float(self.bbox_norm[1]),
                float(self.bbox_norm[2]),
                float(self.bbox_norm[3]),
            ],
            "samples": int(self.samples),
            "unique_tracks": int(self.unique_tracks),
            "nodes": [node.to_dict() for node in self.nodes],
            "territory": dict(self.territory),
            "pressing": dict(self.pressing),
        }


def _average_norm(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Compute average normalized XY point."""
    if not points:
        return (0.5, 0.5)
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (_clip01(float(np.mean(xs))), _clip01(float(np.mean(ys))))


def _spread_norm(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Compute standard deviation in normalized space."""
    if not points:
        return (0.0, 0.0)
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (float(np.std(xs)), float(np.std(ys)))


def _bbox_norm(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    """Compute normalized bounding box of given points."""
    if not points:
        return (0.0, 0.0, 1.0, 1.0)
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (
        _clip01(min(xs)),
        _clip01(min(ys)),
        _clip01(max(xs)),
        _clip01(max(ys)),
    )


def _blend_rect(
    image: np.ndarray,
    left: int,
    top: int,
    right: int,
    bottom: int,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    """Blend translucent rectangle over image region."""
    if alpha <= 0.0:
        return

    h, w = image.shape[:2]
    left = int(np.clip(left, 0, max(0, w - 1)))
    right = int(np.clip(right, 0, max(0, w - 1)))
    top = int(np.clip(top, 0, max(0, h - 1)))
    bottom = int(np.clip(bottom, 0, max(0, h - 1)))

    if right <= left or bottom <= top:
        return

    region = image[top : bottom + 1, left : right + 1]
    overlay = np.full_like(region, color, dtype=np.uint8)
    blended = cv2.addWeighted(overlay, float(alpha), region, float(1.0 - alpha), 0.0)
    region[:] = blended


def _extract_territory_overlay(
    team_id: str,
    team_analytics: dict[str, Any],
) -> dict[str, float]:
    """Extract x-lane territory control values for one team."""
    territory = team_analytics.get("territory") if isinstance(team_analytics, dict) else {}
    if not isinstance(territory, dict):
        return {}

    teams = territory.get("teams")
    if not isinstance(teams, dict):
        return {}

    team_row = teams.get(team_id)
    if not isinstance(team_row, dict):
        return {}

    lane_share = team_row.get("x_zone_control_share")
    if not isinstance(lane_share, dict):
        return {}

    payload: dict[str, float] = {}
    for lane, raw_value in lane_share.items():
        value = _safe_float(raw_value, default=None)
        if value is None:
            continue
        payload[str(lane)] = _clip01(value)
    return payload


def _extract_pressing_summary(
    team_id: str,
    team_analytics: dict[str, Any],
) -> dict[str, Any]:
    """Extract per-team pressing summary from analytics payload."""
    pressing = team_analytics.get("pressing") if isinstance(team_analytics, dict) else {}
    if not isinstance(pressing, dict):
        return {}

    teams = pressing.get("teams")
    if not isinstance(teams, dict):
        return {}

    row = teams.get(team_id)
    if not isinstance(row, dict):
        return {}

    output: dict[str, Any] = {}
    for key in (
        "frames_defending",
        "avg_pressure_score",
        "avg_nearest_distance_norm",
        "avg_defenders_within_radius",
        "high_press_frames",
        "high_press_rate",
        "high_press_episodes",
    ):
        if key not in row:
            continue
        value = row.get(key)
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            output[key] = float(value) if isinstance(value, float) else int(value)
    return output


class TacticalMapRenderer(VisualizationRenderer):
    """Render tactical team-shape maps from player tracks."""

    visualization_type = "tactical_map"
    schema_version = TACTICAL_MAP_SCHEMA_VERSION

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
        """Render tactical-map artifact and return encoded image + summaries."""
        del events
        query = query or VisualizationQuery()
        context = dict(context or {})

        canvas_config = _resolve_canvas_config(context, self._canvas_config)
        field_canvas = FieldCanvas(config=canvas_config)
        canvas = field_canvas.blank(include_markings=self._include_markings)

        frame_size = _resolve_frame_size(context)
        fps = _safe_float(context.get("fps"), default=None)
        min_confidence = _extract_min_confidence(query)
        max_tracks_per_team = _extract_max_tracks_per_team(query)
        min_samples_per_track = _extract_min_samples_per_track(query)
        include_points = _extract_include_points(query)
        include_territory = _extract_include_territory(query)
        include_pressing = _extract_include_pressing(query)
        team_colors = _resolve_team_colors(context)

        team_analytics = context.get("team_analytics")
        if not isinstance(team_analytics, dict):
            team_analytics = {}

        tracks_seen = 0
        tracks_skipped_filter = 0
        tracks_skipped_missing_coordinates = 0
        provenance_counter: Counter[str] = Counter()

        points: list[TacticalPoint] = []
        for row in tracks or []:
            if not isinstance(row, dict):
                continue

            tracks_seen += 1
            object_type = str(row.get("object_type", "player")).strip().lower() or "player"
            if object_type != "player":
                tracks_skipped_filter += 1
                continue

            team_id = _extract_team_id(row)
            track_id = _extract_track_id(row)
            player_id = _extract_player_id(row)
            frame_idx = _safe_int(row.get("frame_idx"), default=None)
            timestamp = _resolve_timestamp(row, fps)
            confidence = _safe_float(row.get("confidence"), default=1.0)
            confidence = _clip01(float(confidence or 1.0))

            norm_xy, image_xy, provenance = _resolve_norm_xy(row, frame_size)
            if norm_xy is None:
                tracks_skipped_missing_coordinates += 1
                provenance_counter[provenance] += 1
                continue

            point = TacticalPoint(
                frame_idx=frame_idx,
                timestamp=timestamp,
                track_id=track_id,
                player_id=player_id,
                team_id=team_id,
                confidence=confidence,
                norm_xy=norm_xy,
                image_xy=image_xy,
                provenance=provenance,
            )

            if _matches_query(point, query, min_confidence=min_confidence):
                points.append(point)
                provenance_counter[provenance] += 1
            else:
                tracks_skipped_filter += 1

        team_track_points: dict[str, dict[int, list[TacticalPoint]]] = defaultdict(lambda: defaultdict(list))
        for point in points:
            if point.track_id is None:
                continue
            team_track_points[point.team_id][int(point.track_id)].append(point)

        team_shapes: list[TacticalTeamShape] = []
        for team_id, track_map in sorted(team_track_points.items(), key=lambda row: row[0]):
            nodes: list[TacticalTrackNode] = []
            for track_id, samples in track_map.items():
                if len(samples) < min_samples_per_track:
                    continue

                samples_sorted = sorted(
                    samples,
                    key=lambda item: (
                        item.frame_idx if item.frame_idx is not None else 10**12,
                        item.timestamp if item.timestamp is not None else 10**12,
                    ),
                )
                sample_points = [sample.norm_xy for sample in samples_sorted]
                centroid = _average_norm(sample_points)
                confidence_avg = float(np.mean([sample.confidence for sample in samples_sorted]))
                player_candidates = [sample.player_id for sample in samples_sorted if sample.player_id is not None]
                player_id = None
                if player_candidates:
                    player_id = Counter(player_candidates).most_common(1)[0][0]

                nodes.append(
                    TacticalTrackNode(
                        track_id=int(track_id),
                        player_id=player_id,
                        team_id=team_id,
                        samples=len(samples_sorted),
                        confidence_avg=float(np.clip(confidence_avg, 0.0, 1.0)),
                        norm_xy=centroid,
                    )
                )

            if not nodes:
                continue

            nodes.sort(key=lambda node: (-node.samples, node.track_id))
            nodes = nodes[:max_tracks_per_team]

            all_points = [node.norm_xy for node in nodes]
            centroid = _average_norm(all_points)
            spread = _spread_norm(all_points)
            bbox = _bbox_norm(all_points)

            territory = _extract_territory_overlay(team_id, team_analytics)
            pressing = _extract_pressing_summary(team_id, team_analytics)

            team_shapes.append(
                TacticalTeamShape(
                    team_id=team_id,
                    centroid_norm=centroid,
                    spread_norm=spread,
                    bbox_norm=bbox,
                    samples=sum(node.samples for node in nodes),
                    unique_tracks=len(nodes),
                    nodes=nodes,
                    territory=territory,
                    pressing=pressing,
                )
            )

        # Territory x-lane overlays (left/center/right-style bins)
        if include_territory and team_shapes:
            left, top, right, bottom = field_canvas.pitch_bounds
            pitch_w = max(1, right - left + 1)
            pitch_h = max(1, bottom - top + 1)

            all_lanes: list[str] = []
            lane_seen: set[str] = set()
            for shape in team_shapes:
                for lane in shape.territory.keys():
                    lane_key = str(lane)
                    if lane_key in lane_seen:
                        continue
                    lane_seen.add(lane_key)
                    all_lanes.append(lane_key)

            if not all_lanes:
                all_lanes = ["left", "center", "right"]

            lane_to_index = {lane: idx for idx, lane in enumerate(all_lanes)}
            lane_count = max(1, len(all_lanes))

            for shape in team_shapes:
                color = team_colors.get(_normalize_team_id(shape.team_id), team_colors[UNKNOWN_TEAM])
                for lane, share in shape.territory.items():
                    idx = lane_to_index.get(str(lane), None)
                    if idx is None:
                        continue
                    lane_left = left + int(round((idx / lane_count) * pitch_w))
                    lane_right = left + int(round(((idx + 1) / lane_count) * pitch_w))
                    alpha = float(np.clip(0.06 + (0.22 * float(share)), 0.0, 0.35))
                    _blend_rect(
                        canvas,
                        lane_left,
                        top,
                        lane_right,
                        bottom,
                        color,
                        alpha,
                    )

        # Optional raw point cloud layer before shape rendering.
        if include_points:
            for point in points:
                color = team_colors.get(_normalize_team_id(point.team_id), team_colors[UNKNOWN_TEAM])
                px, py = field_canvas.norm_to_pixel(point.norm_xy[0], point.norm_xy[1])
                cv2.circle(canvas, (px, py), 2, color, thickness=-1)

        # Team shape overlays.
        for shape in team_shapes:
            color = team_colors.get(_normalize_team_id(shape.team_id), team_colors[UNKNOWN_TEAM])
            team_nodes = sorted(shape.nodes, key=lambda node: (node.norm_xy[0], node.norm_xy[1]))

            # Draw connective polyline for selected nodes (rough team shape).
            if len(team_nodes) >= 2:
                polyline = [field_canvas.norm_to_pixel(node.norm_xy[0], node.norm_xy[1]) for node in team_nodes]
                cv2.polylines(canvas, [np.array(polyline, dtype=np.int32)], isClosed=False, color=color, thickness=2)

            # Draw centroid spread ellipse.
            cx, cy = field_canvas.norm_to_pixel(shape.centroid_norm[0], shape.centroid_norm[1])
            pitch_w, pitch_h = field_canvas.pitch_size
            axis_x = max(14, int(round(shape.spread_norm[0] * pitch_w * 2.0)))
            axis_y = max(10, int(round(shape.spread_norm[1] * pitch_h * 2.0)))
            cv2.ellipse(
                canvas,
                (cx, cy),
                (axis_x, axis_y),
                0.0,
                0.0,
                360.0,
                color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )
            cv2.circle(canvas, (cx, cy), 8, (245, 245, 245), thickness=-1)
            cv2.circle(canvas, (cx, cy), 5, color, thickness=-1)

            # Draw player-track nodes.
            for node in team_nodes:
                px, py = field_canvas.norm_to_pixel(node.norm_xy[0], node.norm_xy[1])
                radius = int(np.clip(4 + np.log1p(node.samples) * 1.8, 4, 11))
                cv2.circle(canvas, (px, py), radius + 1, (240, 240, 240), thickness=-1)
                cv2.circle(canvas, (px, py), radius, color, thickness=-1)
                cv2.circle(canvas, (px, py), max(2, radius // 3), (24, 24, 24), thickness=-1)

        # Pressing legend
        if include_pressing and team_shapes:
            y_cursor = 16
            for shape in team_shapes:
                pressing = shape.pressing
                if not pressing:
                    continue

                color = team_colors.get(_normalize_team_id(shape.team_id), team_colors[UNKNOWN_TEAM])
                score = _safe_float(pressing.get("avg_pressure_score"), default=0.0) or 0.0
                rate = _safe_float(pressing.get("high_press_rate"), default=0.0) or 0.0
                label = f"{shape.team_id}: pressure {score:.2f}, high press {rate * 100:.1f}%"

                cv2.rectangle(canvas, (12, y_cursor - 9), (22, y_cursor + 1), color, thickness=-1)
                cv2.putText(
                    canvas,
                    label,
                    (28, y_cursor),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.43,
                    (248, 248, 248),
                    1,
                    cv2.LINE_AA,
                )
                y_cursor += 18

        team_shapes.sort(key=lambda row: row.team_id)

        if query.player_id is not None:
            title = f"Tactical Map - Player {int(query.player_id)}"
        elif query.team_id is not None:
            title = f"Tactical Map - {str(query.team_id)}"
        else:
            title = "Tactical Map - Team Shapes"

        payload: dict[str, Any] = {
            "encoding": "png_base64",
            "image_png_base64": _encode_png_base64(canvas),
            "teams": [shape.to_dict() for shape in team_shapes],
            "totals": {
                "samples": len(points),
                "teams": len(team_shapes),
                "tracks": int(sum(shape.unique_tracks for shape in team_shapes)),
            },
        }
        if include_points:
            payload["points"] = [point.to_dict() for point in points]

        metadata = {
            "mode": "player" if query.player_id is not None else ("team" if query.team_id is not None else "all"),
            "tracks_seen": tracks_seen,
            "tracks_rendered": len(points),
            "tracks_skipped_filter": tracks_skipped_filter,
            "tracks_skipped_missing_coordinates": tracks_skipped_missing_coordinates,
            "teams": [shape.team_id for shape in team_shapes],
            "player_ids": sorted({point.player_id for point in points if point.player_id is not None}),
            "max_tracks_per_team": max_tracks_per_team,
            "min_samples_per_track": min_samples_per_track,
            "include_points": include_points,
            "include_territory": include_territory,
            "include_pressing": include_pressing,
            "has_team_analytics": bool(team_analytics),
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


def build_tactical_map(
    *,
    tracks: list[dict[str, Any]],
    query: VisualizationQuery | None = None,
    context: dict[str, Any] | None = None,
    renderer: TacticalMapRenderer | None = None,
) -> VisualizationArtifact:
    """Functional helper to render tactical-map artifacts."""
    map_renderer = renderer or TacticalMapRenderer()
    return map_renderer.render(tracks=tracks, query=query, context=context)
