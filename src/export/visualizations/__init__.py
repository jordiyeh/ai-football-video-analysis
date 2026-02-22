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
from src.export.visualizations.heat_map import (
    HEAT_MAP_SCHEMA_VERSION,
    HeatMapRenderer,
    build_heat_map,
)
from src.export.visualizations.pass_map import (
    PASS_MAP_SCHEMA_VERSION,
    PassMapRenderer,
    build_pass_map,
)
from src.export.visualizations.shot_map import (
    SHOT_MAP_SCHEMA_VERSION,
    ShotMapRenderer,
    build_shot_map,
)
from src.export.visualizations.tactical_map import (
    TACTICAL_MAP_SCHEMA_VERSION,
    TacticalMapRenderer,
    build_tactical_map,
)
from src.export.visualizations.momentum_graph import (
    MOMENTUM_GRAPH_SCHEMA_VERSION,
    MomentumGraphRenderer,
    build_momentum_graph,
)
from src.export.visualizations.pass_strings import (
    PASS_STRINGS_SCHEMA_VERSION,
    PassStringsRenderer,
    build_pass_strings,
)
from src.export.visualizations.radial_chart import (
    RADIAL_CHART_SCHEMA_VERSION,
    RadialChartRenderer,
    build_radial_chart,
)
from src.export.visualizations.progress_chart import (
    PROGRESS_CHART_SCHEMA_VERSION,
    ProgressChartRenderer,
    build_progress_chart,
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
    "HEAT_MAP_SCHEMA_VERSION",
    "HeatMapRenderer",
    "build_heat_map",
    "PASS_MAP_SCHEMA_VERSION",
    "PassMapRenderer",
    "build_pass_map",
    "TACTICAL_MAP_SCHEMA_VERSION",
    "TacticalMapRenderer",
    "build_tactical_map",
    "MOMENTUM_GRAPH_SCHEMA_VERSION",
    "MomentumGraphRenderer",
    "build_momentum_graph",
    "PASS_STRINGS_SCHEMA_VERSION",
    "PassStringsRenderer",
    "build_pass_strings",
    "RADIAL_CHART_SCHEMA_VERSION",
    "RadialChartRenderer",
    "build_radial_chart",
    "PROGRESS_CHART_SCHEMA_VERSION",
    "ProgressChartRenderer",
    "build_progress_chart",
]
