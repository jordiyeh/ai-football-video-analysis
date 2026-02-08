"""Visualization scaffolding for shot/heat/pass/tactical map renderers."""

from src.export.visualizations.base import (
    VISUALIZATION_SCHEMA_VERSION,
    VisualizationArtifact,
    VisualizationQuery,
    VisualizationRenderer,
)
from src.export.visualizations.field_canvas import (
    FieldCanvas,
    FieldCanvasConfig,
    accumulate_heat_grid,
    build_field_canvas,
    normalize_heat_grid,
)
from src.export.visualizations.shot_map import (
    SHOT_MAP_SCHEMA_VERSION,
    ShotMapRenderer,
    build_shot_map,
)

__all__ = [
    "VISUALIZATION_SCHEMA_VERSION",
    "VisualizationArtifact",
    "VisualizationQuery",
    "VisualizationRenderer",
    "FieldCanvasConfig",
    "FieldCanvas",
    "build_field_canvas",
    "accumulate_heat_grid",
    "normalize_heat_grid",
    "SHOT_MAP_SCHEMA_VERSION",
    "ShotMapRenderer",
    "build_shot_map",
]
