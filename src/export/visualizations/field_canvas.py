"""Shared field canvas helpers used by tactical map renderers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np


def _clip01(value: float) -> float:
    """Clamp scalar to normalized [0, 1] range."""
    return float(np.clip(value, 0.0, 1.0))


@dataclass(slots=True, frozen=True)
class FieldCanvasConfig:
    """Styling and geometry options for the soccer field canvas."""

    width: int = 1200
    height: int = 780
    padding: int = 48
    background_color: tuple[int, int, int] = (15, 68, 33)
    pitch_color: tuple[int, int, int] = (32, 122, 62)
    line_color: tuple[int, int, int] = (240, 240, 240)
    line_thickness: int = 2

    def __post_init__(self) -> None:
        """Validate that requested geometry can produce a valid pitch."""
        if self.width < 64 or self.height < 64:
            raise ValueError("Field canvas must be at least 64x64 pixels.")
        if self.padding < 0:
            raise ValueError("Field canvas padding must be non-negative.")
        if (self.padding * 2) >= self.width or (self.padding * 2) >= self.height:
            raise ValueError("Field canvas padding leaves no drawable pitch area.")
        if self.line_thickness <= 0:
            raise ValueError("Field line thickness must be positive.")


def _pitch_bounds(config: FieldCanvasConfig) -> tuple[int, int, int, int]:
    """Return drawable pitch rectangle (left, top, right, bottom)."""
    left = int(config.padding)
    top = int(config.padding)
    right = int(config.width - config.padding - 1)
    bottom = int(config.height - config.padding - 1)
    return left, top, right, bottom


def _draw_pitch_markings(
    canvas: np.ndarray,
    *,
    left: int,
    top: int,
    right: int,
    bottom: int,
    config: FieldCanvasConfig,
) -> None:
    """Draw common soccer pitch markings."""
    line_color = tuple(int(v) for v in config.line_color)
    thickness = int(config.line_thickness)

    cv2.rectangle(canvas, (left, top), (right, bottom), line_color, thickness=thickness)

    mid_x = (left + right) // 2
    mid_y = (top + bottom) // 2
    pitch_w = right - left
    pitch_h = bottom - top

    cv2.line(canvas, (mid_x, top), (mid_x, bottom), line_color, thickness=thickness)

    center_radius = max(4, int(round(min(pitch_w, pitch_h) * 0.11)))
    cv2.circle(canvas, (mid_x, mid_y), center_radius, line_color, thickness=thickness)
    cv2.circle(canvas, (mid_x, mid_y), max(2, thickness + 1), line_color, thickness=-1)

    penalty_depth = max(10, int(round(pitch_w * 0.16)))
    goal_depth = max(8, int(round(pitch_w * 0.06)))
    penalty_half_height = max(16, int(round(pitch_h * 0.22)))
    goal_half_height = max(8, int(round(pitch_h * 0.11)))

    y_penalty_1 = max(top, mid_y - penalty_half_height)
    y_penalty_2 = min(bottom, mid_y + penalty_half_height)
    y_goal_1 = max(top, mid_y - goal_half_height)
    y_goal_2 = min(bottom, mid_y + goal_half_height)

    cv2.rectangle(
        canvas,
        (left, y_penalty_1),
        (min(right, left + penalty_depth), y_penalty_2),
        line_color,
        thickness=thickness,
    )
    cv2.rectangle(
        canvas,
        (max(left, right - penalty_depth), y_penalty_1),
        (right, y_penalty_2),
        line_color,
        thickness=thickness,
    )

    cv2.rectangle(
        canvas,
        (left, y_goal_1),
        (min(right, left + goal_depth), y_goal_2),
        line_color,
        thickness=thickness,
    )
    cv2.rectangle(
        canvas,
        (max(left, right - goal_depth), y_goal_1),
        (right, y_goal_2),
        line_color,
        thickness=thickness,
    )


class FieldCanvas:
    """Coordinate mapper + pitch drawing utility for map visualizations."""

    def __init__(self, config: FieldCanvasConfig | None = None):
        self.config = config or FieldCanvasConfig()
        self.left, self.top, self.right, self.bottom = _pitch_bounds(self.config)

    @property
    def pitch_bounds(self) -> tuple[int, int, int, int]:
        """Pitch rectangle as (left, top, right, bottom)."""
        return self.left, self.top, self.right, self.bottom

    @property
    def pitch_size(self) -> tuple[int, int]:
        """Pitch area size as (width, height)."""
        return self.right - self.left, self.bottom - self.top

    def blank(self, include_markings: bool = True) -> np.ndarray:
        """Create a new RGB canvas with field background and optional markings."""
        canvas = np.full(
            (self.config.height, self.config.width, 3),
            tuple(int(v) for v in self.config.background_color),
            dtype=np.uint8,
        )
        cv2.rectangle(
            canvas,
            (self.left, self.top),
            (self.right, self.bottom),
            tuple(int(v) for v in self.config.pitch_color),
            thickness=-1,
        )

        if include_markings:
            _draw_pitch_markings(
                canvas,
                left=self.left,
                top=self.top,
                right=self.right,
                bottom=self.bottom,
                config=self.config,
            )

        return canvas

    def norm_to_pixel(self, norm_x: float, norm_y: float, *, clip: bool = True) -> tuple[int, int]:
        """Project normalized [0,1] point to pitch pixel coordinates."""
        x = float(norm_x)
        y = float(norm_y)

        if clip:
            x = _clip01(x)
            y = _clip01(y)
        elif x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0:
            raise ValueError("Normalized coordinates must be within [0,1] when clip=False.")

        pitch_w, pitch_h = self.pitch_size
        pixel_x = self.left + int(round(x * pitch_w))
        pixel_y = self.top + int(round(y * pitch_h))

        pixel_x = max(self.left, min(self.right, pixel_x))
        pixel_y = max(self.top, min(self.bottom, pixel_y))
        return pixel_x, pixel_y

    def pixel_to_norm(self, pixel_x: int, pixel_y: int, *, clip: bool = True) -> tuple[float, float]:
        """Project pitch pixel coordinates back to normalized [0,1] space."""
        x = int(pixel_x)
        y = int(pixel_y)

        if clip:
            x = max(self.left, min(self.right, x))
            y = max(self.top, min(self.bottom, y))
        elif x < self.left or x > self.right or y < self.top or y > self.bottom:
            raise ValueError("Pixel coordinates must fall inside the pitch when clip=False.")

        pitch_w, pitch_h = self.pitch_size
        norm_x = (x - self.left) / max(1, pitch_w)
        norm_y = (y - self.top) / max(1, pitch_h)
        return _clip01(norm_x), _clip01(norm_y)

    def project_points(
        self,
        norm_points: Iterable[Sequence[float]],
        *,
        clip: bool = True,
    ) -> list[tuple[int, int]]:
        """Project a sequence of normalized points into pixel coordinates."""
        projected: list[tuple[int, int]] = []
        for point in norm_points:
            if len(point) < 2:
                continue
            try:
                x = float(point[0])
                y = float(point[1])
            except (TypeError, ValueError):
                continue
            projected.append(self.norm_to_pixel(x, y, clip=clip))
        return projected


def build_field_canvas(
    config: FieldCanvasConfig | None = None,
    *,
    include_markings: bool = True,
) -> np.ndarray:
    """Create a fresh soccer-field canvas image."""
    return FieldCanvas(config=config).blank(include_markings=include_markings)


def accumulate_heat_grid(
    norm_points: Iterable[Sequence[float]],
    *,
    bins_x: int = 24,
    bins_y: int = 16,
    clip: bool = True,
) -> np.ndarray:
    """
    Accumulate normalized points into a 2D heat grid.

    Input points may optionally include a weight as third element.
    """
    if bins_x <= 0 or bins_y <= 0:
        raise ValueError("Heat grid bins must be positive.")

    grid = np.zeros((bins_y, bins_x), dtype=np.float32)
    for point in norm_points:
        if len(point) < 2:
            continue

        try:
            norm_x = float(point[0])
            norm_y = float(point[1])
        except (TypeError, ValueError):
            continue

        weight = 1.0
        if len(point) >= 3:
            try:
                weight = float(point[2])
            except (TypeError, ValueError):
                weight = 1.0

        if weight <= 0.0:
            continue

        if clip:
            norm_x = _clip01(norm_x)
            norm_y = _clip01(norm_y)
        elif norm_x < 0.0 or norm_x > 1.0 or norm_y < 0.0 or norm_y > 1.0:
            continue

        x_idx = min(bins_x - 1, int(norm_x * bins_x))
        y_idx = min(bins_y - 1, int(norm_y * bins_y))
        grid[y_idx, x_idx] += float(weight)

    return grid


def normalize_heat_grid(grid: np.ndarray) -> np.ndarray:
    """Scale grid values to [0, 1] by global max."""
    if grid.size == 0:
        return grid.astype(np.float32, copy=True)

    normalized = grid.astype(np.float32, copy=True)
    max_value = float(np.max(normalized))
    if max_value <= 0.0:
        normalized.fill(0.0)
        return normalized
    normalized /= max_value
    return normalized
