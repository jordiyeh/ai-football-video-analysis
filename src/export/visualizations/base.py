"""Shared visualization renderer interfaces and artifact contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

VISUALIZATION_SCHEMA_VERSION = "1.0"


@dataclass(slots=True)
class VisualizationQuery:
    """Common filtering inputs for map visualizations."""

    team_id: str | None = None
    player_id: int | None = None
    start_t: float | None = None
    end_t: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize query filters into JSON-safe payload."""
        payload: dict[str, Any] = {}
        if self.team_id is not None:
            payload["team_id"] = self.team_id
        if self.player_id is not None:
            payload["player_id"] = self.player_id
        if self.start_t is not None:
            payload["start_t"] = float(self.start_t)
        if self.end_t is not None:
            payload["end_t"] = float(self.end_t)
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload


@dataclass(slots=True)
class VisualizationArtifact:
    """Schema-versioned visualization metadata payload."""

    visualization_type: str
    schema_version: str = VISUALIZATION_SCHEMA_VERSION
    title: str | None = None
    width: int | None = None
    height: int | None = None
    query: VisualizationQuery | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert artifact to plain dictionary for JSON export."""
        row: dict[str, Any] = {
            "schema_version": self.schema_version,
            "visualization_type": self.visualization_type,
            "metadata": dict(self.metadata),
            "payload": dict(self.payload),
        }
        if self.title is not None:
            row["title"] = self.title
        if self.width is not None:
            row["width"] = int(self.width)
        if self.height is not None:
            row["height"] = int(self.height)
        if self.query is not None:
            row["query"] = self.query.to_dict()
        return row


class VisualizationRenderer(ABC):
    """Base interface for visualization renderers (shot/heat/pass/tactical)."""

    visualization_type = "visualization"
    schema_version = VISUALIZATION_SCHEMA_VERSION

    @abstractmethod
    def render(
        self,
        *,
        tracks: list[dict[str, Any]],
        events: list[dict[str, Any]] | None = None,
        query: VisualizationQuery | None = None,
        context: dict[str, Any] | None = None,
    ) -> VisualizationArtifact:
        """Render visualization output and return artifact metadata."""

    def build_artifact(
        self,
        *,
        title: str | None = None,
        width: int | None = None,
        height: int | None = None,
        query: VisualizationQuery | None = None,
        metadata: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> VisualizationArtifact:
        """Construct a schema-versioned artifact payload."""
        return VisualizationArtifact(
            visualization_type=self.visualization_type,
            schema_version=self.schema_version,
            title=title,
            width=width,
            height=height,
            query=query,
            metadata=dict(metadata or {}),
            payload=dict(payload or {}),
        )
