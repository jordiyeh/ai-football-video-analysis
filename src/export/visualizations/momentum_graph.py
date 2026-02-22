"""Momentum graph visualization — time-series of team momentum."""

from __future__ import annotations

import base64
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Any

from src.export.visualizations.base import (
    VISUALIZATION_SCHEMA_VERSION,
    VisualizationArtifact,
    VisualizationQuery,
    VisualizationRenderer,
)

MOMENTUM_GRAPH_SCHEMA_VERSION = "1.0"


@dataclass(slots=True)
class MomentumWindow:
    """Single time window in the momentum series."""
    start_t: float
    end_t: float
    scores: dict[str, float]  # team_id -> momentum score [-1, 1]
    dominant_team: str | None
    possession_share: dict[str, float]
    avg_speed: dict[str, float]
    territorial_control: dict[str, float]


class MomentumGraphRenderer(VisualizationRenderer):
    visualization_type = "momentum"
    schema_version = MOMENTUM_GRAPH_SCHEMA_VERSION

    def __init__(
        self,
        *,
        window_seconds: float = 60.0,
        canvas_width: int = 1200,
        canvas_height: int = 400,
    ):
        self.window_seconds = max(10.0, window_seconds)
        self.canvas_width = max(200, canvas_width)
        self.canvas_height = max(100, canvas_height)

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

        # Extract team_analytics from context if available
        team_analytics = context.get("team_analytics", {})
        possession_timeline = team_analytics.get("possession_timeline", [])
        possession = team_analytics.get("possession", {})
        territory = team_analytics.get("territory", {})
        pressing = team_analytics.get("pressing", {})

        # Determine time range
        fps = float(context.get("fps", 30.0))

        # Build per-frame data from tracks if no team_analytics
        if not possession_timeline:
            possession_timeline = self._build_possession_from_tracks(tracks, fps)

        # Determine teams
        teams = set()
        for row in possession_timeline:
            owner = row.get("owner_team", "unknown")
            if owner and owner != "unknown":
                teams.add(owner)
        team_list = sorted(teams)

        if not team_list:
            team_list = ["ours", "opponent"]

        # Build momentum windows
        windows = self._compute_momentum_windows(
            possession_timeline=possession_timeline,
            tracks=tracks,
            team_list=team_list,
            fps=fps,
            query=query,
        )

        # Render the graph image
        image = self._render_graph(windows, team_list, context)

        _, png_buf = cv2.imencode(".png", image)
        image_b64 = base64.b64encode(png_buf.tobytes()).decode("ascii")

        # Build serializable windows
        windows_payload = []
        for w in windows:
            windows_payload.append({
                "start_t": w.start_t,
                "end_t": w.end_t,
                "scores": dict(w.scores),
                "dominant_team": w.dominant_team,
                "possession_share": dict(w.possession_share),
            })

        duration = max(row.get("timestamp", 0) for row in possession_timeline) if possession_timeline else 0

        return self.build_artifact(
            title="Momentum Graph",
            width=self.canvas_width,
            height=self.canvas_height,
            query=query,
            metadata={
                "teams": team_list,
                "window_seconds": self.window_seconds,
                "windows_count": len(windows),
            },
            payload={
                "encoding": "png_base64",
                "image_png_base64": image_b64,
                "windows": windows_payload,
                "totals": {
                    "windows": len(windows),
                    "teams": len(team_list),
                    "duration_seconds": duration,
                },
            },
        )

    def _build_possession_from_tracks(
        self, tracks: list[dict[str, Any]], fps: float
    ) -> list[dict[str, Any]]:
        """Fallback: build minimal possession timeline from raw tracks."""
        timeline = []
        for t in tracks:
            obj_type = str(t.get("object_type", ""))
            if obj_type != "ball":
                continue
            frame_idx = t.get("frame_idx")
            if frame_idx is None:
                continue
            timeline.append({
                "frame_idx": int(frame_idx),
                "timestamp": float(t.get("timestamp", int(frame_idx) / fps)),
                "owner_team": "unknown",
            })
        timeline.sort(key=lambda r: r["frame_idx"])
        return timeline

    def _compute_momentum_windows(
        self,
        *,
        possession_timeline: list[dict[str, Any]],
        tracks: list[dict[str, Any]],
        team_list: list[str],
        fps: float,
        query: VisualizationQuery,
    ) -> list[MomentumWindow]:
        """Compute momentum score per time window."""
        if not possession_timeline:
            return []

        timestamps = [float(r.get("timestamp", 0)) for r in possession_timeline]
        min_t = query.start_t if query.start_t is not None else min(timestamps)
        max_t = query.end_t if query.end_t is not None else max(timestamps)

        windows: list[MomentumWindow] = []
        t = min_t
        while t < max_t:
            window_end = min(t + self.window_seconds, max_t)

            # Filter possession rows in this window
            window_rows = [
                r for r in possession_timeline
                if t <= float(r.get("timestamp", 0)) < window_end
            ]

            # Possession share
            team_frames: dict[str, int] = {team: 0 for team in team_list}
            total_known = 0
            for r in window_rows:
                owner = r.get("owner_team", "unknown")
                if owner in team_frames:
                    team_frames[owner] += 1
                    total_known += 1

            possession_share: dict[str, float] = {}
            for team in team_list:
                possession_share[team] = (
                    team_frames[team] / total_known if total_known > 0 else 0.0
                )

            # Territorial control: average norm_x of possessing team
            territorial: dict[str, float] = {}
            for team in team_list:
                team_rows = [
                    r for r in window_rows
                    if r.get("owner_team") == team and r.get("owner_norm_x") is not None
                ]
                if team_rows:
                    avg_x = sum(float(r["owner_norm_x"]) for r in team_rows) / len(team_rows)
                    territorial[team] = avg_x
                else:
                    territorial[team] = 0.5

            # Compute momentum scores: combine possession + territory advancement
            scores: dict[str, float] = {}
            for team in team_list:
                poss_component = possession_share.get(team, 0.0)
                terr_component = abs(territorial.get(team, 0.5) - 0.5) * 2  # 0 at center, 1 at edges
                scores[team] = (0.6 * poss_component) + (0.4 * terr_component)

            # Normalize to relative momentum (-1 to 1 for 2 teams)
            if len(team_list) == 2:
                diff = scores[team_list[0]] - scores[team_list[1]]
                scores[team_list[0]] = max(-1.0, min(1.0, diff))
                scores[team_list[1]] = -scores[team_list[0]]

            dominant = None
            if scores:
                dominant = max(scores, key=lambda k: scores[k])
                if scores[dominant] <= 0:
                    dominant = None

            avg_speed: dict[str, float] = {team: 0.0 for team in team_list}

            windows.append(MomentumWindow(
                start_t=t,
                end_t=window_end,
                scores=scores,
                dominant_team=dominant,
                possession_share=possession_share,
                avg_speed=avg_speed,
                territorial_control=territorial,
            ))
            t = window_end

        return windows

    def _render_graph(
        self,
        windows: list[MomentumWindow],
        team_list: list[str],
        context: dict[str, Any],
    ) -> np.ndarray:
        """Render momentum time series as a PNG image."""
        w = self.canvas_width
        h = self.canvas_height
        pad = 60

        # Create canvas
        canvas = np.full((h, w, 3), (30, 30, 30), dtype=np.uint8)

        if not windows or len(team_list) < 2:
            cv2.putText(canvas, "Insufficient data for momentum graph",
                       (w // 4, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       (200, 200, 200), 1, cv2.LINE_AA)
            return canvas

        # Team colors
        team_colors = context.get("team_colors", {})
        default_colors = [(80, 180, 255), (80, 255, 120)]  # BGR
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

        plot_left = pad
        plot_right = w - pad
        plot_top = pad
        plot_bottom = h - pad
        plot_w = plot_right - plot_left
        plot_h = plot_bottom - plot_top

        # Draw axes
        cv2.line(canvas, (plot_left, plot_top), (plot_left, plot_bottom),
                (100, 100, 100), 1)
        cv2.line(canvas, (plot_left, (plot_top + plot_bottom) // 2),
                (plot_right, (plot_top + plot_bottom) // 2),
                (80, 80, 80), 1, cv2.LINE_AA)

        # Labels
        mid_y = (plot_top + plot_bottom) // 2
        cv2.putText(canvas, team_list[0][:12], (5, plot_top + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[team_list[0]], 1, cv2.LINE_AA)
        cv2.putText(canvas, team_list[1][:12], (5, plot_bottom - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[team_list[1]], 1, cv2.LINE_AA)
        cv2.putText(canvas, "0", (plot_left - 15, mid_y + 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)

        # Title
        cv2.putText(canvas, "Momentum", (w // 2 - 40, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

        # Time labels
        if windows:
            start_min = int(windows[0].start_t // 60)
            end_min = int(windows[-1].end_t // 60)
            cv2.putText(canvas, f"{start_min}'", (plot_left, h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{end_min}'", (plot_right - 20, h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)

        # Plot momentum line for first team (second is mirror)
        team0 = team_list[0]
        points: list[tuple[int, int]] = []
        for idx, win in enumerate(windows):
            x = plot_left + int((idx / max(1, len(windows) - 1)) * plot_w)
            score = win.scores.get(team0, 0.0)
            # score ranges from -1 (bottom) to +1 (top)
            y = mid_y - int(score * (plot_h // 2))
            y = max(plot_top, min(plot_bottom, y))
            points.append((x, y))

        # Fill area above/below center line
        if len(points) >= 2:
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                # Determine color by which side of center
                avg_y = (y1 + y2) / 2
                if avg_y < mid_y:
                    fill_color = tuple(max(0, c // 3) for c in colors[team_list[0]])
                else:
                    fill_color = tuple(max(0, c // 3) for c in colors[team_list[1]])

                polygon = np.array([
                    [x1, mid_y], [x1, y1], [x2, y2], [x2, mid_y]
                ], dtype=np.int32)
                cv2.fillPoly(canvas, [polygon], fill_color)

            # Draw line
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                avg_y = (y1 + y2) / 2
                color = colors[team_list[0]] if avg_y < mid_y else colors[team_list[1]]
                cv2.line(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        return canvas


def build_momentum_graph(
    *,
    tracks: list[dict[str, Any]],
    events: list[dict[str, Any]] | None = None,
    query: VisualizationQuery | None = None,
    context: dict[str, Any] | None = None,
    renderer: MomentumGraphRenderer | None = None,
) -> VisualizationArtifact:
    """Convenience entry point for building momentum graph."""
    renderer = renderer or MomentumGraphRenderer()
    return renderer.render(
        tracks=tracks, events=events, query=query, context=context,
    )
