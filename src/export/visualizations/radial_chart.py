"""Radial (radar/spider) chart visualization comparing teams across metrics."""

from __future__ import annotations

import base64
import math
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Any

from src.export.visualizations.base import (
    VISUALIZATION_SCHEMA_VERSION,
    VisualizationArtifact,
    VisualizationQuery,
    VisualizationRenderer,
)

RADIAL_CHART_SCHEMA_VERSION = "1.0"

# Default metrics to compare
DEFAULT_METRICS = [
    "possession",
    "shots",
    "goals",
    "passes",
    "territory",
    "pressing",
]


@dataclass(slots=True)
class RadialMetric:
    """A single metric axis on the radar chart."""
    name: str
    label: str
    values: dict[str, float]  # team_id -> normalized [0, 1]
    raw_values: dict[str, float]  # team_id -> raw value


class RadialChartRenderer(VisualizationRenderer):
    """Render a spider/radar chart comparing teams."""

    visualization_type = "radial_chart"
    schema_version = RADIAL_CHART_SCHEMA_VERSION

    def __init__(
        self,
        *,
        canvas_width: int = 800,
        canvas_height: int = 800,
        metrics: list[str] | None = None,
    ):
        self.canvas_width = max(200, canvas_width)
        self.canvas_height = max(200, canvas_height)
        self.metrics = metrics or list(DEFAULT_METRICS)

    def render(
        self,
        *,
        tracks: list[dict[str, Any]],
        events: list[dict[str, Any]] | None = None,
        query: VisualizationQuery | None = None,
        context: dict[str, Any] | None = None,
    ) -> VisualizationArtifact:
        context = context or {}
        query = query or VisualizationQuery()
        events = events or []

        team_analytics = context.get("team_analytics", {})
        match_stats = context.get("match_stats", {})

        # Extract team data
        teams_stats = match_stats.get("teams", {})
        possession_data = team_analytics.get("possession", {})
        territory_data = team_analytics.get("territory", {})
        pressing_data = team_analytics.get("pressing", {})

        # Determine teams
        team_list = sorted(
            set(teams_stats.keys())
            | set(possession_data.get("teams", {}).keys())
        )
        if not team_list:
            team_list = ["ours", "opponent"]

        if query.team_id:
            team_list = [t for t in team_list if t == query.team_id] or team_list

        # Build metric values
        radial_metrics = self._extract_metrics(
            team_list=team_list,
            teams_stats=teams_stats,
            possession_data=possession_data,
            territory_data=territory_data,
            pressing_data=pressing_data,
        )

        # Render chart
        image = self._render_chart(radial_metrics, team_list, context)

        _, png_buf = cv2.imencode(".png", image)
        image_b64 = base64.b64encode(png_buf.tobytes()).decode("ascii")

        metrics_payload = []
        for m in radial_metrics:
            metrics_payload.append({
                "name": m.name,
                "label": m.label,
                "values": dict(m.values),
                "raw_values": dict(m.raw_values),
            })

        return self.build_artifact(
            title="Team Comparison",
            width=self.canvas_width,
            height=self.canvas_height,
            query=query,
            metadata={
                "teams": team_list,
                "metrics_count": len(radial_metrics),
                "metric_names": [m.name for m in radial_metrics],
            },
            payload={
                "encoding": "png_base64",
                "image_png_base64": image_b64,
                "metrics": metrics_payload,
                "totals": {
                    "metrics": len(radial_metrics),
                    "teams": len(team_list),
                },
            },
        )

    def _extract_metrics(
        self,
        team_list: list[str],
        teams_stats: dict[str, Any],
        possession_data: dict[str, Any],
        territory_data: dict[str, Any],
        pressing_data: dict[str, Any],
    ) -> list[RadialMetric]:
        """Extract and normalize metric values for each team."""
        metrics: list[RadialMetric] = []

        metric_extractors = {
            "possession": ("Possession %", self._get_possession),
            "shots": ("Shots", self._get_stat("shots")),
            "goals": ("Goals", self._get_stat("goals")),
            "passes": ("Passes", self._get_stat("passes")),
            "territory": ("Territory", self._get_territory),
            "pressing": ("Pressing", self._get_pressing),
            "set_pieces": ("Set Pieces", self._get_stat("set_pieces")),
        }

        for metric_name in self.metrics:
            if metric_name not in metric_extractors:
                continue
            label, extractor = metric_extractors[metric_name]
            raw_values: dict[str, float] = {}
            for team in team_list:
                raw_values[team] = extractor(
                    team, teams_stats, possession_data, territory_data, pressing_data,
                )

            # Normalize to [0, 1]
            max_val = max(raw_values.values()) if raw_values else 1.0
            if max_val <= 0:
                max_val = 1.0
            normalized = {t: v / max_val for t, v in raw_values.items()}

            metrics.append(RadialMetric(
                name=metric_name,
                label=label,
                values=normalized,
                raw_values=raw_values,
            ))

        return metrics

    def _get_possession(
        self, team: str, stats: dict, poss: dict, terr: dict, press: dict,
    ) -> float:
        teams = poss.get("teams", {})
        team_data = teams.get(team, {})
        return float(team_data.get("share", 0.0)) * 100

    def _get_stat(self, stat_name: str):
        def _extract(team: str, stats: dict, poss: dict, terr: dict, press: dict) -> float:
            team_stats = stats.get(team, {})
            return float(team_stats.get(stat_name, 0))
        return _extract

    def _get_territory(
        self, team: str, stats: dict, poss: dict, terr: dict, press: dict,
    ) -> float:
        teams = terr.get("teams", {})
        team_data = teams.get(team, {})
        return float(team_data.get("samples", 0))

    def _get_pressing(
        self, team: str, stats: dict, poss: dict, terr: dict, press: dict,
    ) -> float:
        teams = press.get("teams", {})
        team_data = teams.get(team, {})
        return float(team_data.get("avg_pressure_score", 0.0)) * 100

    def _render_chart(
        self,
        metrics: list[RadialMetric],
        team_list: list[str],
        context: dict[str, Any],
    ) -> np.ndarray:
        """Render the radar chart as a BGR image."""
        w = self.canvas_width
        h = self.canvas_height
        canvas = np.full((h, w, 3), (30, 30, 30), dtype=np.uint8)

        if not metrics:
            cv2.putText(canvas, "No metrics available",
                       (w // 4, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       (200, 200, 200), 1, cv2.LINE_AA)
            return canvas

        n = len(metrics)
        cx, cy = w // 2, h // 2
        radius = min(cx, cy) - 80

        # Team colors
        team_colors = context.get("team_colors", {})
        default_colors = [(255, 140, 80), (80, 255, 140)]
        colors: dict[str, tuple[int, int, int]] = {}
        for idx, team in enumerate(team_list[:2]):
            hex_color = team_colors.get(team)
            if hex_color and isinstance(hex_color, str) and hex_color.startswith("#"):
                try:
                    r = int(hex_color[1:3], 16)
                    g = int(hex_color[3:5], 16)
                    b = int(hex_color[5:7], 16)
                    colors[team] = (b, g, r)
                except (ValueError, IndexError):
                    colors[team] = default_colors[idx % 2]
            else:
                colors[team] = default_colors[idx % 2]

        # Draw grid rings
        for ring in range(1, 5):
            ring_radius = int(radius * ring / 4)
            cv2.circle(canvas, (cx, cy), ring_radius, (60, 60, 60), 1, cv2.LINE_AA)

        # Draw axes and labels
        angles = []
        for i in range(n):
            angle = (2 * math.pi * i / n) - (math.pi / 2)
            angles.append(angle)
            end_x = int(cx + radius * math.cos(angle))
            end_y = int(cy + radius * math.sin(angle))
            cv2.line(canvas, (cx, cy), (end_x, end_y), (60, 60, 60), 1, cv2.LINE_AA)

            # Label
            label_x = int(cx + (radius + 30) * math.cos(angle))
            label_y = int(cy + (radius + 30) * math.sin(angle))
            label = metrics[i].label
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            label_x -= text_size[0] // 2
            label_y += text_size[1] // 2
            cv2.putText(canvas, label, (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        # Draw team polygons
        for team in team_list[:2]:
            color = colors.get(team, (200, 200, 200))
            points = []
            for i, metric in enumerate(metrics):
                value = metric.values.get(team, 0.0)
                r = radius * value
                x = int(cx + r * math.cos(angles[i]))
                y = int(cy + r * math.sin(angles[i]))
                points.append((x, y))

            if len(points) >= 3:
                pts_array = np.array(points, dtype=np.int32)
                fill_color = tuple(max(0, c // 4) for c in color)
                cv2.fillPoly(canvas, [pts_array], fill_color)
                cv2.polylines(canvas, [pts_array], True, color, 2, cv2.LINE_AA)

            for pt in points:
                cv2.circle(canvas, pt, 4, color, -1, cv2.LINE_AA)

        # Legend
        legend_y = h - 40
        for idx, team in enumerate(team_list[:2]):
            color = colors.get(team, (200, 200, 200))
            x_offset = 20 + idx * 200
            cv2.rectangle(canvas, (x_offset, legend_y), (x_offset + 12, legend_y + 12), color, -1)
            cv2.putText(canvas, team[:20], (x_offset + 18, legend_y + 11),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        # Title
        cv2.putText(canvas, "Team Comparison", (cx - 60, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

        return canvas


def build_radial_chart(
    *,
    tracks: list[dict[str, Any]],
    events: list[dict[str, Any]] | None = None,
    query: VisualizationQuery | None = None,
    context: dict[str, Any] | None = None,
    renderer: RadialChartRenderer | None = None,
) -> VisualizationArtifact:
    """Convenience entry point for building radial chart."""
    renderer = renderer or RadialChartRenderer()
    return renderer.render(
        tracks=tracks, events=events, query=query, context=context,
    )
