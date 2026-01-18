"""Tests for configuration system."""

from src.config.schemas import (
    PipelineConfig,
    VideoConfig,
    DetectionConfig,
    OverlayConfig,
)


def test_default_config():
    """Test default configuration creation."""
    config = PipelineConfig()

    assert config.video.sampling_strategy == "every_frame"
    assert config.detection.device == "mps"
    assert config.detection.confidence_threshold == 0.5
    assert config.overlay.bbox_thickness == 2
    assert config.cache_dir == ".cache"
    assert config.output_dir == "runs"


def test_video_config():
    """Test video configuration."""
    config = VideoConfig(
        sampling_strategy="every_2nd",
        target_fps=30.0,
    )

    assert config.sampling_strategy == "every_2nd"
    assert config.target_fps == 30.0


def test_detection_config():
    """Test detection configuration."""
    config = DetectionConfig(
        model_name="yolov8n.pt",
        device="cpu",
        confidence_threshold=0.7,
    )

    assert config.model_name == "yolov8n.pt"
    assert config.device == "cpu"
    assert config.confidence_threshold == 0.7


def test_overlay_config():
    """Test overlay configuration."""
    config = OverlayConfig(
        bbox_thickness=3,
        show_confidence=False,
        player_color="#0000FF",
    )

    assert config.bbox_thickness == 3
    assert config.show_confidence is False
    assert config.player_color == "#0000FF"
