"""Configuration schemas for the video analysis pipeline."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class VideoConfig(BaseModel):
    """Video ingestion and processing configuration."""

    sampling_strategy: Literal["every_frame", "every_2nd", "every_nth"] = "every_frame"
    sampling_interval: int = 1  # For "every_nth" strategy
    target_fps: float | None = None  # If set, resample video to this fps


class DetectionConfig(BaseModel):
    """Object detection configuration."""

    model_name: str = "yolov8x.pt"
    device: Literal["mps", "cpu", "cuda"] = "mps"
    confidence_threshold: float = 0.5
    batch_size: int = 8
    # COCO class IDs to detect
    player_class_id: int = 0  # person
    ball_class_id: int = 32  # sports ball


class TrackingConfig(BaseModel):
    """Multi-object tracking configuration."""

    algorithm: Literal["bytetrack", "botsort", "deepsort"] = "bytetrack"
    max_age: int = 30  # Max frames to keep lost track
    min_hits: int = 3  # Min detections before confirming track
    iou_threshold: float = 0.3


class TeamConfig(BaseModel):
    """Team identification configuration."""

    method: Literal["color_clustering", "manual"] = "color_clustering"
    n_clusters: int = 2  # For k-means clustering
    # Optional team color hints (hex format)
    our_team_color: str | None = None
    opponent_color: str | None = None


class EventsConfig(BaseModel):
    """Event detection configuration."""

    detect_shots: bool = True
    detect_goals: bool = True
    detect_passes: bool = False  # Phase 2+
    shot_velocity_threshold: float = 5.0  # pixels/frame
    goal_confidence_threshold: float = 0.7


class OverlayConfig(BaseModel):
    """Video overlay rendering configuration."""

    bbox_thickness: int = 2
    show_confidence: bool = True
    show_track_ids: bool = True
    show_team_colors: bool = True
    player_color: str = "#00FF00"  # Green
    ball_color: str = "#FF0000"  # Red
    trail_length: int = 30  # frames


class ExportConfig(BaseModel):
    """Export format configuration."""

    save_detections: bool = True
    save_tracks: bool = True
    save_events: bool = True
    save_overlay_video: bool = True
    detections_format: Literal["parquet", "jsonl", "csv"] = "parquet"
    video_codec: str = "mp4v"
    video_fps: float | None = None  # If None, use original fps


class PipelineConfig(BaseModel):
    """Main pipeline configuration."""

    video: VideoConfig = Field(default_factory=VideoConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    team: TeamConfig = Field(default_factory=TeamConfig)
    events: EventsConfig = Field(default_factory=EventsConfig)
    overlay: OverlayConfig = Field(default_factory=OverlayConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)

    cache_dir: str = ".cache"
    output_dir: str = "runs"
    enable_cache: bool = True

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "PipelineConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, yaml_path: str | Path) -> None:
        """Save configuration to YAML file."""
        with open(yaml_path, "w") as f:
            yaml.safe_dump(self.model_dump(), f, default_flow_style=False)
