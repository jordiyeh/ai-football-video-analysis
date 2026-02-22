"""Player progress chart — development over time across matches."""

from __future__ import annotations

import base64
import cv2
import numpy as np
from typing import Any

from src.export.visualizations.base import (
    VISUALIZATION_SCHEMA_VERSION,
    VisualizationArtifact,
    VisualizationQuery,
    VisualizationRenderer,
)

PROGRESS_CHART_SCHEMA_VERSION = "1.0"


class ProgressChartRenderer(VisualizationRenderer):
    """Render player progress chart showing metrics across matches."""

    visualization_type = "progress_chart"
    schema_version = PROGRESS_CHART_SCHEMA_VERSION

    def __init__(
        self,
        *,
        canvas_width: int = 1000,
        canvas_height: int = 500,
        metrics: list[str] | None = None,
    ):
        self.canvas_width = max(200, canvas_width)
        self.canvas_height = max(100, canvas_height)
        self.metrics = metrics or ["goals", "shots", "passes", "possession_seconds"]

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

        # Extract cross-match player data from context
        player_progress = context.get("player_progress", [])
        player_id = query.player_id

        if not player_progress:
            # Build empty chart
            image = self._render_empty(player_id)
            _, png_buf = cv2.imencode(".png", image)
            image_b64 = base64.b64encode(png_buf.tobytes()).decode("ascii")
            return self.build_artifact(
                title=f"Player Progress{f' (#{player_id})' if player_id else ''}",
                width=self.canvas_width,
                height=self.canvas_height,
                query=query,
                metadata={"player_id": player_id, "matches": 0},
                payload={
                    "encoding": "png_base64",
                    "image_png_base64": image_b64,
                    "data_points": [],
                    "totals": {"matches": 0, "metrics": 0},
                },
            )

        # Filter to specific player if requested
        if player_id is not None:
            player_progress = [
                p for p in player_progress
                if p.get("player_id") == player_id
            ]

        # Render chart
        image = self._render_chart(player_progress, self.metrics)
        _, png_buf = cv2.imencode(".png", image)
        image_b64 = base64.b64encode(png_buf.tobytes()).decode("ascii")

        return self.build_artifact(
            title=f"Player Progress{f' (#{player_id})' if player_id else ''}",
            width=self.canvas_width,
            height=self.canvas_height,
            query=query,
            metadata={
                "player_id": player_id,
                "matches": len(player_progress),
                "metrics": self.metrics,
            },
            payload={
                "encoding": "png_base64",
                "image_png_base64": image_b64,
                "data_points": player_progress,
                "totals": {
                    "matches": len(player_progress),
                    "metrics": len(self.metrics),
                },
            },
        )

    def _render_empty(self, player_id: int | None) -> np.ndarray:
        """Render empty state."""
        canvas = np.full((self.canvas_height, self.canvas_width, 3), (30, 30, 30), dtype=np.uint8)
        msg = "No cross-match data available"
        if player_id is not None:
            msg = f"No data for player #{player_id}"
        cv2.putText(canvas, msg, (self.canvas_width // 4, self.canvas_height // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        return canvas

    def _render_chart(
        self, data_points: list[dict[str, Any]], metrics: list[str],
    ) -> np.ndarray:
        """Render multi-metric line chart."""
        w = self.canvas_width
        h = self.canvas_height
        pad = 60
        canvas = np.full((h, w, 3), (30, 30, 30), dtype=np.uint8)

        if not data_points or not metrics:
            return self._render_empty(None)

        n_points = len(data_points)
        plot_left = pad
        plot_right = w - pad
        plot_top = pad
        plot_bottom = h - pad
        plot_w = plot_right - plot_left
        plot_h = plot_bottom - plot_top

        # Draw axes
        cv2.line(canvas, (plot_left, plot_bottom), (plot_right, plot_bottom),
                (100, 100, 100), 1)
        cv2.line(canvas, (plot_left, plot_top), (plot_left, plot_bottom),
                (100, 100, 100), 1)

        # Title
        cv2.putText(canvas, "Player Progress", (w // 2 - 60, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

        # Match labels on x-axis
        for i in range(n_points):
            x = plot_left + int((i / max(1, n_points - 1)) * plot_w)
            label = data_points[i].get("match_label", f"M{i+1}")
            cv2.putText(canvas, str(label)[:6], (x - 10, h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1, cv2.LINE_AA)

        # Colors for metrics
        metric_colors = [
            (80, 180, 255), (80, 255, 120), (255, 100, 100),
            (255, 255, 100), (180, 100, 255), (100, 255, 255),
        ]

        for m_idx, metric in enumerate(metrics):
            color = metric_colors[m_idx % len(metric_colors)]
            values = []
            for dp in data_points:
                val = dp.get(metric, dp.get("stats", {}).get(metric, 0))
                try:
                    values.append(float(val))
                except (TypeError, ValueError):
                    values.append(0.0)

            max_val = max(values) if values else 1.0
            if max_val <= 0:
                max_val = 1.0

            points = []
            for i, val in enumerate(values):
                x = plot_left + int((i / max(1, n_points - 1)) * plot_w)
                y = plot_bottom - int((val / max_val) * plot_h)
                y = max(plot_top, min(plot_bottom, y))
                points.append((x, y))

            # Draw line
            for i in range(len(points) - 1):
                cv2.line(canvas, points[i], points[i + 1], color, 2, cv2.LINE_AA)
            for pt in points:
                cv2.circle(canvas, pt, 3, color, -1, cv2.LINE_AA)

            # Legend
            legend_y = 15 + m_idx * 15
            cv2.rectangle(canvas, (w - pad + 5, legend_y - 5),
                         (w - pad + 15, legend_y + 5), color, -1)
            cv2.putText(canvas, metric[:12], (w - pad + 20, legend_y + 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1, cv2.LINE_AA)

        return canvas


def build_progress_chart(
    *,
    tracks: list[dict[str, Any]],
    events: list[dict[str, Any]] | None = None,
    query: VisualizationQuery | None = None,
    context: dict[str, Any] | None = None,
    renderer: ProgressChartRenderer | None = None,
) -> VisualizationArtifact:
    """Convenience entry point for building progress chart."""
    renderer = renderer or ProgressChartRenderer()
    return renderer.render(
        tracks=tracks, events=events, query=query, context=context,
    )
