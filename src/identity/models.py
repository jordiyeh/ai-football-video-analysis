"""Pydantic models for player identity data."""

from datetime import datetime
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class Player(BaseModel):
    """A persistent player identity."""

    model_config = ConfigDict(from_attributes=True)

    player_id: int
    name: str | None = None
    jersey_number: int | None = None
    team_hint: Literal["ours", "opponent"] | None = None
    embedding_centroid: list[float] | None = None  # 512 floats
    embedding_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def get_centroid_array(self) -> np.ndarray | None:
        """Get embedding centroid as numpy array."""
        if self.embedding_centroid is None:
            return None
        return np.array(self.embedding_centroid, dtype=np.float32)


class Appearance(BaseModel):
    """A player appearance in a specific video/run."""

    model_config = ConfigDict(from_attributes=True)

    appearance_id: int
    video_id: str
    run_name: str
    track_id: int
    player_id: int | None = None
    match_confidence: float | None = None
    match_method: Literal["auto", "suggested", "manual"] | None = None
    frame_start: int | None = None
    frame_end: int | None = None
    embedding: list[float] | None = None  # 512 floats

    def get_embedding_array(self) -> np.ndarray | None:
        """Get embedding as numpy array."""
        if self.embedding is None:
            return None
        return np.array(self.embedding, dtype=np.float32)


class PlayerWithAppearances(Player):
    """Player with their appearances."""

    appearances: list[Appearance] = []


class AppearanceWithPlayer(Appearance):
    """Appearance with linked player info."""

    player: Player | None = None
