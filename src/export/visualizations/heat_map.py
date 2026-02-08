"""Heat-map renderer from player tracks and normalized coordinates."""

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
from src.export.visualizations.field_canvas import (
    FieldCanvas,
    FieldCanvasConfig,
    accumulate_heat_grid,
    normalize_heat_grid,
)

HEAT_MAP_SCHEMA_VERSION = "1.0"
UNKNOWN_TEAM = "unknown"

_COLORMAP_BY_NAME = {
    "autumn": cv2.COLORMAP_AUTUMN,
    "bone": cv2.COLORMAP_BONE,
    "hot": cv2.COLORMAP_HOT,
    "inferno": cv2.COLORMAP_INFERNO,
    "jet": cv2.COLORMAP_JET,
    "magma": cv2.COLORMAP_MAGMA,
    "plasma": cv2.COLORMAP_PLASMA,
    "turbo": cv2.COLORMAP_TURBO,
    "viridis": cv2.COLORMAP_VIRIDIS,
}

_NORM_CANDIDATE_KEYS = (
    "norm_xy",
    "normalized_xy",
    "normalized_location",
    "norm_location",
)


def _safe_float(value: Any, default: float | None = None) -> float | None:
    """Safely cast finite values to float."""
    try:
        parsed = float(value)
    except Exception:
        return default
    if not np.isfinite(parsed):
        return default
    return parsed


def _safe_int(value: Any, default: int | None = None) -> int | None:
    """Safely cast values to int."""
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except Exception:
        return default


def _clip01(value: float) -> float:
    """Clamp scalar to [0, 1]."""
    return float(np.clip(value, 0.0, 1.0))


def _normalize_team_id(value: Any) -> str:
    """Normalize team labels to stable string identifiers."""
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
    """Parse coordinate pair from list/tuple payload."""
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    x = _safe_float(value[0], default=None)
    y = _safe_float(value[1], default=None)
    if x is None or y is None:
        return None
    return (x, y)


def _looks_normalized(xy: tuple[float, float]) -> bool:
    """Return True when XY already looks normalized."""
    return 0.0 <= xy[0] <= 1.0 and 0.0 <= xy[1] <= 1.0


def _resolve_frame_size(context: dict[str, Any]) -> tuple[float, float] | None:
    """Resolve frame size from context fields."""
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


def _bbox_center(track: dict[str, Any]) -> tuple[float, float] | None:
    """Resolve bbox center point."""
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
    track: dict[str, Any],
    frame_size: tuple[float, float] | None,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None, str]:
    """Resolve normalized coordinates with image fallback + provenance."""
    norm_x = _safe_float(track.get("norm_x"), default=None)
    norm_y = _safe_float(track.get("norm_y"), default=None)
    if norm_x is not None and norm_y is not None:
        return (_clip01(norm_x), _clip01(norm_y)), None, "track.norm_x_norm_y"

    for key in _NORM_CANDIDATE_KEYS:
        norm_xy = _parse_xy(track.get(key))
        if norm_xy is not None:
            return (_clip01(norm_xy[0]), _clip01(norm_xy[1])), None, f"track.{key}"

    image_xy = _extract_image_xy(track)
    if image_xy is None:
        return None, None, "missing_coordinates"

    if frame_size is not None:
        frame_w, frame_h = frame_size
        norm_x = _clip01(image_xy[0] / frame_w)
        norm_y = _clip01(image_xy[1] / frame_h)
        return (norm_x, norm_y), image_xy, "track.image_xy"

    if _looks_normalized(image_xy):
        return (_clip01(image_xy[0]), _clip01(image_xy[1])), None, "track.image_xy_as_norm"

    return None, image_xy, "missing_frame_size"


def _extract_team_id(track: dict[str, Any]) -> str:
    """Resolve team label from track payload."""
    for key in ("team_name", "team", "team_label", "team_id"):
        team = _normalize_team_id(track.get(key))
        if team != UNKNOWN_TEAM:
            return team
    return UNKNOWN_TEAM


def _extract_player_id(track: dict[str, Any]) -> int | None:
    """Resolve player identifier from track payload."""
    for key in (
        "player_id",
        "assigned_player_id",
        "identity_player_id",
        "owner_player_id",
        "track_id",
    ):
        value = _safe_int(track.get(key), default=None)
        if value is not None:
            return value
    return None


def _resolve_timestamp(track: dict[str, Any], fps: float | None) -> float | None:
    """Resolve timestamp from track rows using explicit fields or frame/fps."""
    for key in ("timestamp", "t", "time_sec"):
        ts = _safe_float(track.get(key), default=None)
        if ts is not None:
            return ts

    frame_idx = _safe_int(track.get("frame_idx"), default=None)
    if frame_idx is None:
        return None
    if fps is None or fps <= 0:
        return None
    return float(frame_idx) / fps


def _extract_min_confidence(query: VisualizationQuery) -> float:
    """Resolve optional confidence floor from query extras."""
    min_confidence = 0.0
    if query.extra:
        parsed = _safe_float(query.extra.get("min_confidence"), default=0.0)
        if parsed is not None:
            min_confidence = parsed
    return _clip01(min_confidence)


def _resolve_weight_mode(query: VisualizationQuery) -> str:
    """Resolve point weighting mode."""
    raw = ""
    if query.extra:
        raw = str(query.extra.get("weight_mode", "")).strip().lower()
    if raw in {"confidence", "uniform"}:
        return raw
    return "uniform"


def _resolve_bins(
    query: VisualizationQuery,
    context: dict[str, Any],
    default_bins_x: int,
    default_bins_y: int,
) -> tuple[int, int]:
    """Resolve heat-map bin dimensions from query/context."""
    bins_x = _safe_int(context.get("heat_bins_x"), default=default_bins_x)
    bins_y = _safe_int(context.get("heat_bins_y"), default=default_bins_y)

    if query.extra:
        bins_x = _safe_int(query.extra.get("bins_x"), default=bins_x)
        bins_y = _safe_int(query.extra.get("bins_y"), default=bins_y)

    bins_x = int(np.clip(int(bins_x or default_bins_x), 4, 256))
    bins_y = int(np.clip(int(bins_y or default_bins_y), 4, 256))
    return bins_x, bins_y


def _resolve_object_filter(query: VisualizationQuery) -> set[str]:
    """Resolve object-type filter, defaulting to players only."""
    object_types: set[str] = {"player"}

    if query.extra:
        raw_types = query.extra.get("object_types")
        if isinstance(raw_types, str):
            object_types = {raw_types.strip().lower()} if raw_types.strip() else {"player"}
        elif isinstance(raw_types, (list, tuple, set)):
            parsed = {
                str(item).strip().lower()
                for item in raw_types
                if str(item).strip()
            }
            if parsed:
                object_types = parsed

        include_ball = bool(query.extra.get("include_ball"))
        if include_ball:
            object_types.add("ball")

    return object_types


def _resolve_canvas_config(
    context: dict[str, Any],
    base_config: FieldCanvasConfig | None,
) -> FieldCanvasConfig:
    """Resolve field canvas config from defaults + context overrides."""
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


def _resolve_colormap(query: VisualizationQuery, context: dict[str, Any]) -> int:
    """Resolve OpenCV colormap constant from context/query."""
    name = str(context.get("heat_colormap", "turbo")).strip().lower()
    if query.extra:
        extra_name = str(query.extra.get("colormap", "")).strip().lower()
        if extra_name:
            name = extra_name
    return _COLORMAP_BY_NAME.get(name, cv2.COLORMAP_TURBO)


def _resolve_blur_kernel(query: VisualizationQuery, default_kernel: int) -> int:
    """Resolve Gaussian blur kernel size for smoothing."""
    blur_kernel = default_kernel
    if query.extra:
        parsed = _safe_int(query.extra.get("blur_kernel"), default=default_kernel)
        if parsed is not None:
            blur_kernel = parsed
    blur_kernel = max(0, int(blur_kernel))
    if blur_kernel > 1 and blur_kernel % 2 == 0:
        blur_kernel += 1
    return blur_kernel


def _resolve_alpha(
    query: VisualizationQuery,
    default_min_alpha: float,
    default_max_alpha: float,
) -> tuple[float, float]:
    """Resolve alpha blending range."""
    min_alpha = default_min_alpha
    max_alpha = default_max_alpha

    if query.extra:
        min_alpha = _safe_float(query.extra.get("min_alpha"), default=default_min_alpha) or default_min_alpha
        max_alpha = _safe_float(query.extra.get("max_alpha"), default=default_max_alpha) or default_max_alpha

    min_alpha = float(np.clip(min_alpha, 0.0, 1.0))
    max_alpha = float(np.clip(max_alpha, 0.0, 1.0))
    if max_alpha < min_alpha:
        max_alpha = min_alpha
    return min_alpha, max_alpha


def _encode_png_base64(image: np.ndarray) -> str:
    """Encode image array as base64 PNG string."""
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        return ""
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _draw_heat_overlay(
    *,
    canvas: np.ndarray,
    field_canvas: FieldCanvas,
    normalized_grid: np.ndarray,
    colormap: int,
    min_alpha: float,
    max_alpha: float,
) -> np.ndarray:
    """Blend normalized heat intensity on top of the soccer pitch."""
    if normalized_grid.size == 0:
        return canvas
    if float(np.max(normalized_grid)) <= 0.0:
        return canvas

    left, top, right, bottom = field_canvas.pitch_bounds
    region = canvas[top : bottom + 1, left : right + 1]
    if region.size == 0:
        return canvas

    heat = cv2.resize(
        normalized_grid.astype(np.float32),
        (region.shape[1], region.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )
    heat = np.clip(heat, 0.0, 1.0)

    heat_u8 = np.clip(np.round(heat * 255.0), 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(heat_u8, colormap)

    alpha = min_alpha + (heat * max(0.0, max_alpha - min_alpha))
    alpha = np.expand_dims(np.clip(alpha, 0.0, 1.0), axis=2)

    blended = (
        (color.astype(np.float32) * alpha)
        + (region.astype(np.float32) * (1.0 - alpha))
    )
    blended = np.clip(np.round(blended), 0, 255).astype(np.uint8)

    mask = heat > 1e-4
    region[mask] = blended[mask]
    return canvas


def _matches_query(point: "HeatPoint", query: VisualizationQuery, min_confidence: float) -> bool:
    """Return True when heat sample satisfies active query filters."""
    if point.confidence < min_confidence:
        return False

    if query.team_id is not None:
        if _normalize_team_id(query.team_id) != _normalize_team_id(point.team_id):
            return False

    if query.player_id is not None:
        if point.player_id is None or int(point.player_id) != int(query.player_id):
            return False

    if query.start_t is not None:
        if point.timestamp is None or point.timestamp < float(query.start_t):
            return False

    if query.end_t is not None:
        if point.timestamp is None or point.timestamp > float(query.end_t):
            return False

    return True


@dataclass(slots=True)
class HeatPoint:
    """Normalized track sample used for heat-map rendering."""

    frame_idx: int | None
    timestamp: float | None
    track_id: int | None
    player_id: int | None
    team_id: str
    confidence: float
    weight: float
    norm_xy: tuple[float, float]
    image_xy: tuple[float, float] | None
    provenance: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize heat sample to JSON-safe dictionary."""
        row: dict[str, Any] = {
            "team_id": self.team_id,
            "confidence": float(self.confidence),
            "weight": float(self.weight),
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


class HeatMapRenderer(VisualizationRenderer):
    """Render team/player heat maps from track trajectories."""

    visualization_type = "heat_map"
    schema_version = HEAT_MAP_SCHEMA_VERSION

    def __init__(
        self,
        *,
        canvas_config: FieldCanvasConfig | None = None,
        include_markings: bool = True,
        bins_x: int = 48,
        bins_y: int = 32,
        blur_kernel: int = 11,
        min_alpha: float = 0.15,
        max_alpha: float = 0.85,
    ):
        self._canvas_config = canvas_config
        self._include_markings = include_markings
        self._bins_x = int(max(4, bins_x))
        self._bins_y = int(max(4, bins_y))
        self._blur_kernel = int(max(0, blur_kernel))
        self._min_alpha = float(np.clip(min_alpha, 0.0, 1.0))
        self._max_alpha = float(np.clip(max_alpha, 0.0, 1.0))

    def render(
        self,
        *,
        tracks: list[dict[str, Any]],
        events: list[dict[str, Any]] | None = None,
        query: VisualizationQuery | None = None,
        context: dict[str, Any] | None = None,
    ) -> VisualizationArtifact:
        """Render heat map artifact from input track rows."""
        del events
        query = query or VisualizationQuery()
        context = dict(context or {})

        canvas_config = _resolve_canvas_config(context, self._canvas_config)
        field_canvas = FieldCanvas(config=canvas_config)
        canvas = field_canvas.blank(include_markings=self._include_markings)

        frame_size = _resolve_frame_size(context)
        fps = _safe_float(context.get("fps"), default=None)
        min_confidence = _extract_min_confidence(query)
        bins_x, bins_y = _resolve_bins(query, context, self._bins_x, self._bins_y)
        blur_kernel = _resolve_blur_kernel(query, self._blur_kernel)
        min_alpha, max_alpha = _resolve_alpha(query, self._min_alpha, self._max_alpha)
        colormap = _resolve_colormap(query, context)
        object_filter = _resolve_object_filter(query)
        weight_mode = _resolve_weight_mode(query)
        include_points = bool(query.extra.get("include_points")) if query.extra else False

        tracks_seen = 0
        tracks_considered = 0
        tracks_skipped_filter = 0
        tracks_skipped_missing_coordinates = 0
        provenance_counter: Counter[str] = Counter()

        points: list[HeatPoint] = []
        for row in tracks or []:
            if not isinstance(row, dict):
                continue
            tracks_seen += 1

            object_type = str(row.get("object_type", "player")).strip().lower() or "player"
            if object_type not in object_filter:
                tracks_skipped_filter += 1
                continue

            team_id = _extract_team_id(row)
            track_id = _safe_int(row.get("track_id"), default=None)
            player_id = _extract_player_id(row)
            frame_idx = _safe_int(row.get("frame_idx"), default=None)
            timestamp = _resolve_timestamp(row, fps)

            confidence = _safe_float(row.get("confidence"), default=1.0) or 1.0
            confidence = _clip01(confidence)

            norm_xy, image_xy, provenance = _resolve_norm_xy(row, frame_size)
            if norm_xy is None:
                tracks_skipped_missing_coordinates += 1
                provenance_counter[provenance] += 1
                continue

            if weight_mode == "confidence":
                weight = max(1e-3, float(confidence))
            else:
                weight = 1.0

            point = HeatPoint(
                frame_idx=frame_idx,
                timestamp=timestamp,
                track_id=track_id,
                player_id=player_id,
                team_id=team_id,
                confidence=confidence,
                weight=weight,
                norm_xy=norm_xy,
                image_xy=image_xy,
                provenance=provenance,
            )

            if _matches_query(point, query, min_confidence=min_confidence):
                points.append(point)
                tracks_considered += 1
                provenance_counter[provenance] += 1
            else:
                tracks_skipped_filter += 1

        points.sort(
            key=lambda item: (
                item.frame_idx if item.frame_idx is not None else -1,
                item.track_id if item.track_id is not None else -1,
            )
        )

        heat_inputs = [
            (point.norm_xy[0], point.norm_xy[1], point.weight)
            for point in points
        ]
        raw_grid = accumulate_heat_grid(heat_inputs, bins_x=bins_x, bins_y=bins_y, clip=True)
        normalized_grid = normalize_heat_grid(raw_grid)
        if blur_kernel > 1 and float(np.max(normalized_grid)) > 0.0:
            normalized_grid = cv2.GaussianBlur(
                normalized_grid.astype(np.float32),
                (blur_kernel, blur_kernel),
                sigmaX=0.0,
                sigmaY=0.0,
                borderType=cv2.BORDER_REPLICATE,
            )
            normalized_grid = normalize_heat_grid(normalized_grid)

        canvas = _draw_heat_overlay(
            canvas=canvas,
            field_canvas=field_canvas,
            normalized_grid=normalized_grid,
            colormap=colormap,
            min_alpha=min_alpha,
            max_alpha=max_alpha,
        )

        teams = sorted({point.team_id for point in points})
        player_ids = sorted({int(point.player_id) for point in points if point.player_id is not None})

        mode = "all"
        if query.player_id is not None:
            mode = "player"
        elif query.team_id is not None:
            mode = "team"

        if mode == "player":
            title = f"Heat Map - Player {int(query.player_id)}"
        elif mode == "team":
            title = f"Heat Map - {str(query.team_id)}"
        else:
            title = "Heat Map - All Players"

        payload: dict[str, Any] = {
            "encoding": "png_base64",
            "image_png_base64": _encode_png_base64(canvas),
            "grid_shape": [int(raw_grid.shape[0]), int(raw_grid.shape[1])],
            "raw_heat_grid": raw_grid.tolist(),
            "heat_grid": normalized_grid.tolist(),
            "totals": {
                "samples": len(points),
                "total_weight": float(np.sum(raw_grid)) if raw_grid.size else 0.0,
                "max_cell_weight": float(np.max(raw_grid)) if raw_grid.size else 0.0,
            },
        }
        if include_points:
            payload["points"] = [point.to_dict() for point in points]

        metadata = {
            "mode": mode,
            "object_filter": sorted(object_filter),
            "weight_mode": weight_mode,
            "tracks_seen": tracks_seen,
            "tracks_considered": tracks_considered,
            "tracks_rendered": len(points),
            "tracks_skipped_filter": tracks_skipped_filter,
            "tracks_skipped_missing_coordinates": tracks_skipped_missing_coordinates,
            "teams": teams,
            "player_ids": player_ids,
            "bins_x": bins_x,
            "bins_y": bins_y,
            "blur_kernel": blur_kernel,
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


def build_heat_map(
    *,
    tracks: list[dict[str, Any]],
    query: VisualizationQuery | None = None,
    context: dict[str, Any] | None = None,
    renderer: HeatMapRenderer | None = None,
) -> VisualizationArtifact:
    """Functional helper for callers that do not manage renderer lifecycle."""
    map_renderer = renderer or HeatMapRenderer()
    return map_renderer.render(tracks=tracks, query=query, context=context)
