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


class BallDetectionConfig(BaseModel):
    """Ball-specific detection configuration for boosting ball detection rate."""

    # Confidence threshold (lowered from default 0.25)
    confidence_threshold: float = 0.15

    # Multi-scale detection
    enable_multiscale: bool = True
    scales: list[float] = [0.5, 1.0, 1.5]
    merge_iou_threshold: float = 0.5

    # Temporal consistency filter
    enable_temporal_filter: bool = True
    temporal_window_size: int = 5
    max_frame_displacement: float = 100.0  # pixels
    min_temporal_confirmations: int = 2

    # Ball candidate tracking (soft-track before committing)
    enable_candidate_tracking: bool = True
    candidate_min_hits: int = 3
    candidate_max_age: int = 5

    # Size/shape filtering
    max_size_ratio: float = 0.05  # Max ball bbox dimension as fraction of frame
    max_aspect_ratio: float = 3.0  # Reject very elongated boxes


class BallSpecialistConfig(BaseModel):
    """Configuration for specialized ball detection model."""

    enabled: bool = False  # Disabled by default for backward compatibility
    model_source: str = "keremberke/yolov8n-soccer-ball-detection"  # HuggingFace model ID
    confidence_threshold: float = 0.3
    ball_class_id: int = 0  # Class ID for ball in specialized model
    max_size_ratio: float = 0.08  # Max ball bbox dimension as fraction of frame
    max_aspect_ratio: float = 3.0  # Reject very elongated boxes
    cache_dir: str = "models"  # Directory to cache downloaded models


class EnsembleConfig(BaseModel):
    """Configuration for detector ensemble with box fusion."""

    enabled: bool = False  # Disabled by default for backward compatibility
    weights: dict[str, float] = {"yolo": 1.0, "ball_specialist": 1.5}  # Detector weights
    iou_threshold: float = 0.5  # IoU threshold for clustering boxes
    skip_box_threshold: float = 0.01  # Skip boxes below this confidence
    fusion_type: Literal["wbf", "nms", "soft_nms"] = "wbf"  # Fusion algorithm


class DetectionConfig(BaseModel):
    """Object detection configuration."""

    model_name: str = "yolov8x.pt"
    device: Literal["mps", "cpu", "cuda"] = "mps"
    confidence_threshold: float = 0.5
    ball_confidence_threshold: float = 0.25  # Lower threshold for ball (small object) - DEPRECATED, use ball.confidence_threshold
    batch_size: int = 8
    # COCO class IDs to detect
    player_class_id: int = 0  # person
    ball_class_id: int = 32  # sports ball
    # Ball size filtering (relative to frame dimensions) - DEPRECATED, use ball.max_size_ratio
    ball_max_size_ratio: float = 0.05  # Ball bbox max dimension as fraction of frame

    # New ball-specific configuration
    ball: BallDetectionConfig = Field(default_factory=BallDetectionConfig)

    # Ball specialist detector (specialized soccer ball model)
    ball_specialist: BallSpecialistConfig = Field(default_factory=BallSpecialistConfig)

    # Detector ensemble (combines multiple detectors with WBF)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)

    @property
    def effective_ball_confidence(self) -> float:
        """Get effective ball confidence threshold (prefer new nested config)."""
        return self.ball.confidence_threshold

    @property
    def effective_ball_max_size_ratio(self) -> float:
        """Get effective ball max size ratio (prefer new nested config)."""
        return self.ball.max_size_ratio

    @property
    def effective_ball_max_aspect_ratio(self) -> float:
        """Get effective ball max aspect ratio."""
        return self.ball.max_aspect_ratio


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
    # Persistent team pre-selection (set via UI or CLI)
    home_team_id: int | None = None
    away_team_id: int | None = None
    home_kit: str = "home"
    away_kit: str = "home"


class FieldNormalizationConfig(BaseModel):
    """Field-view normalization configuration (zoom-aware norm_xy)."""

    enabled: bool = True
    min_players_per_frame: int = 6
    player_percentile_low: float = 0.10
    player_percentile_high: float = 0.90
    margin_ratio: float = 0.12
    smoothing_alpha: float = 0.25
    min_viewport_width_ratio: float = 0.35
    min_viewport_height_ratio: float = 0.35
    clip_norm: bool = True


class TeamAnalyticsConfig(BaseModel):
    """Team-level analytics configuration (possession/territory/pass/pressing)."""

    enabled: bool = True
    use_norm_coordinates: bool = True

    # Possession / pass inference.
    possession_max_ball_distance_px: float = 140.0
    possession_smoothing_frames: int = 3
    possession_min_stable_frames: int = 3
    possession_min_segment_frames: int = 4
    pass_min_gap_seconds: float = 0.15
    pass_max_gap_seconds: float = 2.5

    # Territory occupancy bins.
    territory_x_bins: int = 3
    territory_y_bins: int = 3

    # Pressing proxy.
    pressure_radius_norm: float = 0.10
    high_press_threshold: float = 0.65
    high_press_min_frames: int = 8

    # Optional identity confidence threshold for track -> player link usage.
    min_assignment_confidence: float = 0.6


class CrossMatchReportingConfig(BaseModel):
    """Cross-match reporting/export configuration."""

    enabled: bool = True
    runs_root: str | None = None  # Defaults to current run parent directory
    include_current_run: bool = True
    max_runs: int = 60
    last_n_window: int = 5
    top_players: int = 15
    min_player_segment_score: float = 0.25
    match_type_filter: list[str] = Field(default_factory=list)
    formation_filter: list[str] = Field(default_factory=list)


class CoachAssistConfig(BaseModel):
    """Coach assist generation configuration."""

    enabled: bool = False
    provider: Literal["heuristic", "cloud"] = "heuristic"
    allow_cloud: bool = False
    max_insights: int = 5
    min_confidence: float = 0.45


class InterpolationConfig(BaseModel):
    """Ball trajectory interpolation configuration."""

    max_gap: int = 300  # Max frames to interpolate across
    physics_threshold: int = 10  # Use physics-based interpolation above this gap
    process_noise_position: float = 1.0  # Kalman process noise for position
    process_noise_velocity: float = 0.5  # Kalman process noise for velocity
    process_noise_acceleration: float = 0.1  # Kalman process noise for acceleration
    measurement_noise: float = 5.0  # Kalman measurement noise
    acceleration_decay: float = 0.98  # Decay factor for acceleration per frame
    confidence_decay_rate: float = 0.97  # Confidence decay per second (at 30fps)
    min_confidence: float = 0.1  # Minimum confidence floor for interpolated points
    use_bidirectional: bool = True  # Blend forward and backward predictions


class GoalRegionDetectionConfig(BaseModel):
    """Goal region detection configuration for visual detection vs heuristic fallback."""

    enabled: bool = True
    detection_method: Literal["visual", "heuristic", "hybrid"] = "hybrid"

    # Heuristic fallback (current behavior)
    heuristic_edge_margin: float = 0.15  # Goal regions within 15% of top/bottom edges
    heuristic_goal_width_fraction: float = 0.30  # Goal width as fraction of frame width

    # Hough line detection parameters
    canny_low_threshold: int = 50
    canny_high_threshold: int = 150
    hough_threshold: int = 100
    hough_min_line_length: int = 100
    hough_max_line_gap: int = 10
    line_angle_tolerance: float = 15.0  # Degrees from horizontal/vertical

    # Goalpost detection (HSV white threshold)
    white_hsv_low: tuple[int, int, int] = (0, 0, 200)
    white_hsv_high: tuple[int, int, int] = (180, 30, 255)
    min_goalpost_height: float = 0.05  # Min height as fraction of frame height
    goalpost_aspect_ratio_min: float = 3.0  # Min aspect ratio (height/width)

    # Temporal smoothing
    enable_temporal_smoothing: bool = True
    smoothing_window_frames: int = 30
    max_frame_displacement: float = 50.0  # Max pixels jump between frames

    # Fallback strategy
    fallback_confidence_threshold: float = 0.3  # Use heuristic below this
    blend_threshold: float = 0.6  # Blend visual+heuristic between 0.3 and 0.6
    interpolation_max_gap: int = 60  # Max frames to interpolate across


class CelebrationConfig(BaseModel):
    """Celebration detection configuration for goal confirmation."""

    enabled: bool = True

    # Arms-up detection
    arms_up_aspect_ratio_threshold: float = 0.5  # Aspect ratio indicating raised arms
    arms_up_height_change_threshold: float = 1.2  # Min height change ratio for arms up
    arms_up_min_duration_frames: int = 5  # Min frames to hold pose

    # Group huddle detection
    huddle_max_player_distance: float = 100.0  # Max distance between players in huddle
    huddle_min_players: int = 3  # Min players for huddle detection
    huddle_convergence_threshold: float = 0.5  # Convergence ratio threshold

    # Temporal constraints
    post_shot_window_frames: int = 150  # 5 sec at 30fps - window to look for celebrations
    celebration_cooldown_frames: int = 300  # 10 sec cooldown between celebrations

    # Fusion
    signal_weight: float = 0.15  # Weight in shot fusion
    min_confidence: float = 0.4  # Minimum confidence for celebration event


class AlternativeShotDetectionConfig(BaseModel):
    """Alternative shot detection using player behavior signals (for sparse ball data)."""

    enabled: bool = True  # Enable alternative detection when ball data is sparse

    # Kick detection parameters
    foot_region_fraction: float = 0.2  # Bottom 20% of player bbox = foot area
    kick_proximity_threshold: float = 50.0  # Max distance (pixels) ball-to-foot for kick
    kick_lookback_frames: int = 5  # Frames to check before kick
    kick_lookahead_frames: int = 10  # Frames to check after kick

    # Goal entry detection parameters
    goal_entry_margin: float = 0.05  # Extra margin around goal region
    max_kick_association_frames: int = 90  # Max frames (3 sec @ 30fps) to associate kick with goal entry

    # Goalkeeper dive detection parameters
    gk_region_margin: float = 0.20  # Players within this margin of goal considered GK candidates
    gk_dive_displacement: float = 30.0  # Min horizontal displacement (pixels) for dive
    gk_aspect_change_threshold: float = 1.5  # Min aspect ratio change (standing -> horizontal)

    # Player clustering parameters
    attack_spread_threshold: float = 0.25  # Max spread (fraction of frame) for attack formation
    min_players_per_team: int = 5  # Min players to compute clustering metrics

    # Signal fusion parameters
    kick_weight: float = 0.35  # Weight for kick event signal
    goal_entry_weight: float = 0.30  # Weight for goal entry signal
    gk_dive_weight: float = 0.25  # Weight for goalkeeper dive signal
    attack_context_weight: float = 0.10  # Weight for attacking formation context
    fusion_min_confidence: float = 0.3  # Min confidence to report shot
    fusion_temporal_window: int = 60  # Frames (2 sec @ 30fps) to group events

    # Ball coverage threshold to activate alternative detection
    ball_coverage_threshold: float = 0.5  # Use alternative if coverage < 50%

    # Celebration detection
    celebration: CelebrationConfig = Field(default_factory=CelebrationConfig)


class EventsConfig(BaseModel):
    """Event detection configuration."""

    detect_shots: bool = True
    detect_goals: bool = True
    detect_passes: bool = False  # Phase 2+
    detect_set_pieces: bool = True
    detect_tactical: bool = True
    shot_velocity_threshold: float = 8.0  # pixels/frame (lower = more sensitive)
    goal_confidence_threshold: float = 0.5  # Lower threshold for more goal candidates
    min_shot_duration_frames: int = 2  # Minimum frames for high-speed segment
    interpolate_ball: bool = True  # Fill gaps in ball trajectory
    max_interpolation_gap: int = 300  # Max frames to interpolate across (legacy, use interpolation.max_gap)
    interpolation: InterpolationConfig = Field(default_factory=InterpolationConfig)
    alternative_shot: AlternativeShotDetectionConfig = Field(default_factory=AlternativeShotDetectionConfig)
    goal_region: GoalRegionDetectionConfig = Field(default_factory=GoalRegionDetectionConfig)


class HighlightEventSignalConfig(BaseModel):
    """Event-driven highlight scoring configuration."""

    include_goals: bool = True
    include_shots: bool = True
    goal_weight: float = 1.0
    shot_weight: float = 0.7
    min_confidence: float = 0.2


class HighlightAudioSignalConfig(BaseModel):
    """Audio-driven highlight scoring configuration."""

    enabled: bool = True
    sample_rate: int = 2000
    window_seconds: float = 1.0
    hop_seconds: float = 0.25
    min_z_score: float = 2.0
    min_abs_rms: float = 0.01
    min_gap_seconds: float = 2.0
    max_spikes: int = 120
    weight: float = 0.45


class HighlightActionSignalConfig(BaseModel):
    """Action-driven highlight scoring configuration."""

    enabled: bool = True
    min_speed_pixels_per_sec: float = 220.0
    player_pressure_radius: float = 120.0
    score_quantile: float = 0.9
    min_candidate_score: float = 0.45
    max_candidates: int = 120
    weight: float = 0.55


class HighlightSegmentConfig(BaseModel):
    """Highlight segment selection configuration."""

    pre_roll_seconds: float = 8.0
    post_roll_seconds: float = 12.0
    merge_gap_seconds: float = 4.0
    top_n: int = 20
    min_segment_score: float = 0.4


class HighlightExportConfig(BaseModel):
    """Highlight export configuration."""

    save_clips: bool = False
    clips_dir: str = "clips"
    clip_video_codec: str = "libx264"
    clip_audio_codec: str = "aac"


class PlayerReelsConfig(BaseModel):
    """Per-player reel generation configuration."""

    enabled: bool = True
    max_segments_per_player: int = 8
    min_presence_seconds: float = 1.5
    min_player_segment_score: float = 0.2
    min_assignment_confidence: float = 0.6
    include_suggested_assignments: bool = True
    save_clips: bool = False
    clips_dir: str = "player_clips"


class HighlightsConfig(BaseModel):
    """Automatic highlight generation configuration."""

    enabled: bool = True
    event: HighlightEventSignalConfig = Field(default_factory=HighlightEventSignalConfig)
    audio: HighlightAudioSignalConfig = Field(default_factory=HighlightAudioSignalConfig)
    action: HighlightActionSignalConfig = Field(default_factory=HighlightActionSignalConfig)
    segment: HighlightSegmentConfig = Field(default_factory=HighlightSegmentConfig)
    export: HighlightExportConfig = Field(default_factory=HighlightExportConfig)
    player_reels: PlayerReelsConfig = Field(default_factory=PlayerReelsConfig)


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


class ReIDConfig(BaseModel):
    """Re-identification (ReID) configuration for player embeddings."""

    model_name: str = "osnet_x0_25"  # OSNet variant (~2MB, fast)
    device: Literal["mps", "cpu", "cuda"] = "mps"
    embedding_dim: int = 512  # Standard for OSNet
    crop_size: tuple[int, int] = (256, 128)  # height x width
    batch_size: int = 32
    cache_dir: str = "models"


class ProfileIngestionConfig(BaseModel):
    """Profile ingestion configuration for external player profile bundles."""

    enabled: bool = False
    profiles_root: str | None = None
    recursive_image_scan: bool = False
    image_extensions: list[str] = Field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".webp"]
    )
    # Build body-embedding gallery from profile photos (same OSNet space as track embeddings)
    enable_body_embedding_seed: bool = True
    max_images_per_profile_for_reid: int = 5
    min_profile_crops_for_seed: int = 2
    detector_confidence_threshold: float = 0.3
    fallback_full_image: bool = True
    # Fusion thresholds for profile-evidence override/boost
    profile_match_enabled: bool = True
    profile_match_auto_threshold: float = 0.82
    profile_match_suggest_threshold: float = 0.68
    profile_override_margin: float = 0.05
    profile_agreement_bonus: float = 0.05


class IdentityFaceEvidenceConfig(BaseModel):
    """Face-evidence configuration for multimodal identity fusion."""

    enabled: bool = True
    max_images_per_profile: int = 5
    min_profile_face_images: int = 1
    min_track_support_frames: int = 2
    suggest_threshold: float = 0.68
    override_margin: float = 0.08
    agreement_bonus: float = 0.04


class IdentityJerseyOCRConfig(BaseModel):
    """Jersey OCR configuration for multimodal identity fusion."""

    enabled: bool = True
    min_ocr_confidence: float = 0.45
    min_track_support_frames: int = 2
    override_margin: float = 0.12
    agreement_bonus: float = 0.03


class IdentityLockingConfig(BaseModel):
    """Substitution-aware lock/unlock configuration."""

    enabled: bool = True
    lock_confidence_threshold: float = 0.82
    overlap_conflict_frames: int = 45
    substitution_gap_frames: int = 150
    demote_conflicting_auto: bool = True


class IdentityMultimodalConfig(BaseModel):
    """Multimodal dynamic tagging configuration."""

    enabled: bool = True
    face: IdentityFaceEvidenceConfig = Field(default_factory=IdentityFaceEvidenceConfig)
    jersey_ocr: IdentityJerseyOCRConfig = Field(default_factory=IdentityJerseyOCRConfig)
    locking: IdentityLockingConfig = Field(default_factory=IdentityLockingConfig)


class IdentityConfig(BaseModel):
    """Player identity persistence configuration."""

    enabled: bool = True
    database_path: str = "players.db"  # Project-level, shared across runs
    samples_per_track: int = 10  # Number of crop samples per track
    min_crop_height: int = 50  # Minimum crop height in pixels
    min_crop_width: int = 25  # Minimum crop width in pixels
    auto_match_threshold: float = 0.85  # Auto-assign player above this similarity
    suggest_threshold: float = 0.70  # Suggest match above this similarity
    new_player_threshold: float = 0.60  # Create new player below this similarity
    profile_ingestion: ProfileIngestionConfig = Field(default_factory=ProfileIngestionConfig)
    multimodal: IdentityMultimodalConfig = Field(default_factory=IdentityMultimodalConfig)


class PipelineConfig(BaseModel):
    """Main pipeline configuration."""

    video: VideoConfig = Field(default_factory=VideoConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    team: TeamConfig = Field(default_factory=TeamConfig)
    field: FieldNormalizationConfig = Field(default_factory=FieldNormalizationConfig)
    team_analytics: TeamAnalyticsConfig = Field(default_factory=TeamAnalyticsConfig)
    cross_match: CrossMatchReportingConfig = Field(default_factory=CrossMatchReportingConfig)
    coach_assist: CoachAssistConfig = Field(default_factory=CoachAssistConfig)
    events: EventsConfig = Field(default_factory=EventsConfig)
    highlights: HighlightsConfig = Field(default_factory=HighlightsConfig)
    overlay: OverlayConfig = Field(default_factory=OverlayConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    reid: ReIDConfig = Field(default_factory=ReIDConfig)
    identity: IdentityConfig = Field(default_factory=IdentityConfig)

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
