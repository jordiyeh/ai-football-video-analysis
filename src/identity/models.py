"""Pydantic models for player identity data."""

from datetime import datetime
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class Team(BaseModel):
    """A persistent team entity."""

    model_config = ConfigDict(from_attributes=True)

    team_id: int
    name: str
    short_name: str | None = None
    logo_path: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class TeamKit(BaseModel):
    """A kit (jersey set) associated with a team."""

    model_config = ConfigDict(from_attributes=True)

    kit_id: int
    team_id: int
    kit_type: Literal["home", "away", "third"]
    image_path: str | None = None
    dominant_color_hsv: list[float] | None = None
    secondary_color_hsv: list[float] | None = None
    color_hex: str | None = None
    secondary_color_hex: str | None = None


class TeamWithKits(Team):
    """Team with its kits and player count."""

    kits: list[TeamKit] = []
    player_count: int = 0


class RunTeamAssociation(BaseModel):
    """Links a run to a team with a role (home/away)."""

    model_config = ConfigDict(from_attributes=True)

    run_name: str
    role: Literal["home", "away"]
    team_id: int
    active_kit: str = "home"
    cluster_id: int | None = None


class MatchMetadata(BaseModel):
    """Match-level metadata linked to a run."""

    model_config = ConfigDict(from_attributes=True)

    run_name: str
    video_id: str | None = None
    match_date: str | None = None
    competition: str | None = None
    season: str | None = None
    venue: str | None = None
    home_team_id: int | None = None
    away_team_id: int | None = None
    notes: str | None = None
    extra: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class MatchTag(BaseModel):
    """Tag annotation linked to a run and optional entity."""

    model_config = ConfigDict(from_attributes=True)

    tag_id: int
    run_name: str
    label: str
    category: str = "general"
    start_time: float | None = None
    end_time: float | None = None
    frame_idx: int | None = None
    track_id: int | None = None
    player_id: int | None = None
    team_id: int | None = None
    confidence: float | None = None
    source: Literal["manual", "auto", "imported"] = "manual"
    notes: str | None = None
    metadata: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Player(BaseModel):
    """A persistent player identity."""

    model_config = ConfigDict(from_attributes=True)

    player_id: int
    name: str | None = None
    jersey_number: int | None = None
    team_hint: Literal["ours", "opponent"] | None = None
    team_id: int | None = None
    photo_path: str | None = None
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
