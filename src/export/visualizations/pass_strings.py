"""Pass strings (sequential chain) visualization."""

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
from src.export.visualizations.field_canvas import FieldCanvas, FieldCanvasConfig

PASS_STRINGS_SCHEMA_VERSION = "1.0"


@dataclass(slots=True)
class PassChain:
    """A consecutive sequence of passes by a single team."""
    team_id: str
    passes: list[dict[str, Any]]
    start_t: float
    end_t: float
    length: int


class PassStringsRenderer(VisualizationRenderer):
    """Render consecutive pass chains as connected arrows on a field."""

    visualization_type = "pass_strings"
    schema_version = PASS_STRINGS_SCHEMA_VERSION

    def __init__(
        self,
        *,
        canvas_config: FieldCanvasConfig | None = None,
        include_markings: bool = True,
        min_chain_length: int = 3,
        max_gap_seconds: float = 5.0,
    ):
        self.canvas_config = canvas_config
        self.include_markings = include_markings
        self.min_chain_length = max(2, min_chain_length)
        self.max_gap_seconds = max(0.5, max_gap_seconds)

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

        min_chain = int(query.extra.get("min_chain_length", self.min_chain_length))
        max_gap = float(query.extra.get("max_gap_seconds", self.max_gap_seconds))

        # Resolve canvas config from context overrides
        cw = int(context.get("canvas_width", self.canvas_config.width if self.canvas_config else 1200))
        ch = int(context.get("canvas_height", self.canvas_config.height if self.canvas_config else 780))
        cp = int(context.get("canvas_padding", self.canvas_config.padding if self.canvas_config else 48))
        cfg = FieldCanvasConfig(width=cw, height=ch, padding=cp)
        field = FieldCanvas(cfg)

        frame_width = int(context.get("frame_width", 1920))
        frame_height = int(context.get("frame_height", 1080))

        # Filter pass events
        pass_events = [
            e for e in events
            if str(e.get("event_type", "")).lower() == "pass"
        ]

        # Apply query filters
        if query.team_id:
            pass_events = [
                e for e in pass_events
                if self._event_team(e) == query.team_id
            ]
        if query.player_id is not None:
            pass_events = [
                e for e in pass_events
                if self._event_player(e) == query.player_id
            ]
        if query.start_t is not None:
            pass_events = [
                e for e in pass_events
                if float(e.get("timestamp", 0)) >= query.start_t
            ]
        if query.end_t is not None:
            pass_events = [
                e for e in pass_events
                if float(e.get("timestamp", 0)) <= query.end_t
            ]

        min_confidence = float(query.extra.get("min_confidence", 0.0))
        if min_confidence > 0:
            pass_events = [
                e for e in pass_events
                if float(e.get("confidence", 0)) >= min_confidence
            ]

        # Sort by timestamp
        pass_events.sort(key=lambda e: float(e.get("timestamp", 0)))

        # Group into chains
        chains = self._build_chains(pass_events, min_chain, max_gap)

        # Render
        canvas = field.blank(include_markings=self.include_markings)

        # Chain colors (cycle through distinct hues)
        chain_colors = [
            (255, 100, 100), (100, 255, 100), (100, 100, 255),
            (255, 255, 100), (255, 100, 255), (100, 255, 255),
            (200, 150, 100), (100, 200, 150),
        ]

        team_colors = context.get("team_colors", {})
        chains_payload = []
        max_chain_length = 0

        for chain_idx, chain in enumerate(chains):
            color = self._resolve_chain_color(
                chain, chain_idx, team_colors, chain_colors,
            )
            points = []
            for p in chain.passes:
                norm_xy = self._resolve_norm_xy(p, frame_width, frame_height)
                if norm_xy:
                    points.append(norm_xy)

            if len(points) < 2:
                continue

            max_chain_length = max(max_chain_length, chain.length)

            # Draw connected arrows
            pixel_points = [field.norm_to_pixel(nx, ny) for nx, ny in points]
            for i in range(len(pixel_points) - 1):
                pt1 = pixel_points[i]
                pt2 = pixel_points[i + 1]
                cv2.arrowedLine(canvas, pt1, pt2, color, 2, cv2.LINE_AA, tipLength=0.15)
                cv2.circle(canvas, pt1, 4, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, pixel_points[-1], 4, color, -1, cv2.LINE_AA)

            # Label chain length
            mid_idx = len(pixel_points) // 2
            label_pt = pixel_points[mid_idx]
            cv2.putText(
                canvas, str(chain.length),
                (label_pt[0] + 6, label_pt[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA,
            )

            chains_payload.append({
                "team_id": chain.team_id,
                "length": chain.length,
                "start_t": chain.start_t,
                "end_t": chain.end_t,
                "points": [{"norm_x": nx, "norm_y": ny} for nx, ny in points],
            })

        _, png_buf = cv2.imencode(".png", canvas)
        image_b64 = base64.b64encode(png_buf.tobytes()).decode("ascii")

        teams_seen = sorted(set(c.team_id for c in chains))

        return self.build_artifact(
            title="Pass Strings",
            width=cfg.width,
            height=cfg.height,
            query=query,
            metadata={
                "teams": teams_seen,
                "chains_count": len(chains),
                "max_chain_length": max_chain_length,
                "min_chain_length_filter": min_chain,
                "pass_events_total": len(pass_events),
            },
            payload={
                "encoding": "png_base64",
                "image_png_base64": image_b64,
                "chains": chains_payload,
                "totals": {
                    "chains": len(chains),
                    "max_chain_length": max_chain_length,
                    "teams": len(teams_seen),
                },
            },
        )

    def _event_team(self, event: dict[str, Any]) -> str | None:
        """Extract team from event."""
        team = event.get("team_id") or event.get("team")
        meta = event.get("metadata", {})
        if team is None and isinstance(meta, dict):
            team = meta.get("team_id") or meta.get("team")
        return str(team) if team is not None else None

    def _event_player(self, event: dict[str, Any]) -> int | None:
        """Extract player_id from event."""
        pid = event.get("player_id")
        meta = event.get("metadata", {})
        if pid is None and isinstance(meta, dict):
            pid = meta.get("player_id") or meta.get("from_player_id")
        try:
            return int(pid) if pid is not None else None
        except (TypeError, ValueError):
            return None

    def _build_chains(
        self,
        pass_events: list[dict[str, Any]],
        min_chain: int,
        max_gap: float,
    ) -> list[PassChain]:
        """Group consecutive same-team passes into chains."""
        if not pass_events:
            return []

        chains: list[PassChain] = []
        current_team = self._event_team(pass_events[0])
        current_passes = [pass_events[0]]
        current_start = float(pass_events[0].get("timestamp", 0))

        for event in pass_events[1:]:
            team = self._event_team(event)
            ts = float(event.get("timestamp", 0))
            prev_ts = float(current_passes[-1].get("timestamp", 0))
            gap = ts - prev_ts

            if team == current_team and gap <= max_gap:
                current_passes.append(event)
            else:
                if len(current_passes) >= min_chain and current_team:
                    chains.append(PassChain(
                        team_id=current_team,
                        passes=list(current_passes),
                        start_t=current_start,
                        end_t=float(current_passes[-1].get("timestamp", 0)),
                        length=len(current_passes),
                    ))
                current_team = team
                current_passes = [event]
                current_start = ts

        if len(current_passes) >= min_chain and current_team:
            chains.append(PassChain(
                team_id=current_team,
                passes=list(current_passes),
                start_t=current_start,
                end_t=float(current_passes[-1].get("timestamp", 0)),
                length=len(current_passes),
            ))

        return chains

    def _resolve_norm_xy(
        self, event: dict[str, Any], frame_w: int, frame_h: int,
    ) -> tuple[float, float] | None:
        """Resolve normalized coordinates from a pass event."""
        # Check direct norm fields
        for key in ("norm_xy", "normalized_xy", "normalized_location"):
            val = event.get(key)
            if isinstance(val, (list, tuple)) and len(val) >= 2:
                try:
                    return (float(val[0]), float(val[1]))
                except (TypeError, ValueError):
                    pass

        # Check location field
        loc = event.get("location")
        if isinstance(loc, (list, tuple)) and len(loc) >= 2:
            try:
                x, y = float(loc[0]), float(loc[1])
                if 0 <= x <= 1 and 0 <= y <= 1:
                    return (x, y)
                # Assume image coords
                return (
                    max(0, min(1, x / max(1, frame_w - 1))),
                    max(0, min(1, y / max(1, frame_h - 1))),
                )
            except (TypeError, ValueError):
                pass

        # Check metadata
        meta = event.get("metadata", {})
        if isinstance(meta, dict):
            for key in ("from_norm_xy", "norm_xy", "location"):
                val = meta.get(key)
                if isinstance(val, (list, tuple)) and len(val) >= 2:
                    try:
                        x, y = float(val[0]), float(val[1])
                        if 0 <= x <= 1 and 0 <= y <= 1:
                            return (x, y)
                        return (
                            max(0, min(1, x / max(1, frame_w - 1))),
                            max(0, min(1, y / max(1, frame_h - 1))),
                        )
                    except (TypeError, ValueError):
                        pass

        return None

    def _resolve_chain_color(
        self,
        chain: PassChain,
        chain_idx: int,
        team_colors: dict[str, Any],
        fallback_colors: list[tuple[int, int, int]],
    ) -> tuple[int, int, int]:
        """Resolve BGR color for a chain."""
        hex_color = team_colors.get(chain.team_id)
        if hex_color and isinstance(hex_color, str) and hex_color.startswith("#"):
            try:
                r = int(hex_color[1:3], 16)
                g = int(hex_color[3:5], 16)
                b = int(hex_color[5:7], 16)
                return (b, g, r)
            except (ValueError, IndexError):
                pass
        return fallback_colors[chain_idx % len(fallback_colors)]


def build_pass_strings(
    *,
    tracks: list[dict[str, Any]],
    events: list[dict[str, Any]] | None = None,
    query: VisualizationQuery | None = None,
    context: dict[str, Any] | None = None,
    renderer: PassStringsRenderer | None = None,
) -> VisualizationArtifact:
    """Convenience entry point for building pass strings visualization."""
    renderer = renderer or PassStringsRenderer()
    return renderer.render(
        tracks=tracks, events=events, query=query, context=context,
    )
