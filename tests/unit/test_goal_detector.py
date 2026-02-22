"""Unit tests for goal region detection."""

import numpy as np
import pytest

from src.config.schemas import GoalRegionDetectionConfig
from src.vision.field.goal_detector import (
    GoalRegion,
    GoalRegionDetector,
    GoalRegionTracker,
    HeuristicGoalRegionProvider,
)


class TestGoalRegion:
    """Tests for GoalRegion dataclass."""

    def test_goal_region_creation(self):
        """Test creating a GoalRegion."""
        region = GoalRegion(
            name="top",
            bounds={"x_min": 100, "x_max": 300, "y_min": 0, "y_max": 50},
            confidence=0.8,
            detection_method="visual",
        )
        assert region.name == "top"
        assert region.bounds["x_min"] == 100
        assert region.confidence == 0.8
        assert region.detection_method == "visual"


class TestHeuristicGoalRegionProvider:
    """Tests for HeuristicGoalRegionProvider."""

    def test_default_goal_regions(self):
        """Test that heuristic produces correct default regions."""
        provider = HeuristicGoalRegionProvider(
            frame_width=1920,
            frame_height=1080,
        )

        regions = provider.get_goal_regions(0)
        assert len(regions) == 2

        # Find top and bottom regions
        top = next(r for r in regions if r.name == "top")
        bottom = next(r for r in regions if r.name == "bottom")

        # Default edge margin is 15%
        assert top.bounds["y_min"] == 0
        assert top.bounds["y_max"] == pytest.approx(1080 * 0.15, rel=1e-3)

        assert bottom.bounds["y_min"] == pytest.approx(1080 * 0.85, rel=1e-3)
        assert bottom.bounds["y_max"] == 1080

    def test_custom_margins(self):
        """Test heuristic with custom margins."""
        provider = HeuristicGoalRegionProvider(
            frame_width=1920,
            frame_height=1080,
            edge_margin=0.20,
            goal_width_fraction=0.40,
        )

        regions = provider.get_goal_regions(0)
        top = next(r for r in regions if r.name == "top")

        # 20% edge margin
        assert top.bounds["y_max"] == pytest.approx(1080 * 0.20, rel=1e-3)

        # 40% width centered
        x_center = 1920 / 2
        half_width = 1920 * 0.40 / 2
        assert top.bounds["x_min"] == pytest.approx(x_center - half_width, rel=1e-3)
        assert top.bounds["x_max"] == pytest.approx(x_center + half_width, rel=1e-3)

    def test_is_in_goal_region_inside(self):
        """Test is_in_goal_region for point inside goal."""
        provider = HeuristicGoalRegionProvider(
            frame_width=1920,
            frame_height=1080,
        )

        # Point in top center (should be in top goal)
        is_in, name = provider.is_in_goal_region((960, 50), frame_idx=0)
        assert is_in is True
        assert name == "top"

    def test_is_in_goal_region_outside(self):
        """Test is_in_goal_region for point outside goal."""
        provider = HeuristicGoalRegionProvider(
            frame_width=1920,
            frame_height=1080,
        )

        # Point in center of frame (not in any goal)
        is_in, name = provider.is_in_goal_region((960, 540), frame_idx=0)
        assert is_in is False
        assert name is None

    def test_is_in_goal_region_edge_case(self):
        """Test is_in_goal_region at exact boundary."""
        provider = HeuristicGoalRegionProvider(
            frame_width=1920,
            frame_height=1080,
            edge_margin=0.15,
            goal_width_fraction=0.30,
        )

        # Point at bottom goal y boundary
        bottom_y = 1080 * 0.85
        is_in, name = provider.is_in_goal_region((960, bottom_y), frame_idx=0)
        assert is_in is True
        assert name == "bottom"

    def test_confidence_is_one(self):
        """Test that heuristic regions have confidence 1.0."""
        provider = HeuristicGoalRegionProvider(
            frame_width=1920,
            frame_height=1080,
        )

        regions = provider.get_goal_regions(0)
        for r in regions:
            assert r.confidence == 1.0
            assert r.detection_method == "heuristic"


class TestGoalRegionTracker:
    """Tests for GoalRegionTracker temporal smoothing."""

    def test_update_adds_to_history(self):
        """Test that update adds regions to history."""
        tracker = GoalRegionTracker(
            frame_width=1920,
            frame_height=1080,
            smoothing_window=5,
        )

        region = GoalRegion(
            name="top",
            bounds={"x_min": 100, "x_max": 300, "y_min": 0, "y_max": 50},
            confidence=0.8,
            detection_method="visual",
        )

        smoothed = tracker.update([region], frame_idx=0)
        assert len(smoothed) == 1
        assert smoothed[0].name == "top"

    def test_smoothing_averages_bounds(self):
        """Test that smoothing averages bounds over time."""
        tracker = GoalRegionTracker(
            frame_width=1920,
            frame_height=1080,
            smoothing_window=10,
            max_displacement=100.0,
        )

        # Add several regions with slightly different bounds
        for i in range(5):
            region = GoalRegion(
                name="top",
                bounds={
                    "x_min": 100 + i * 2,
                    "x_max": 300 + i * 2,
                    "y_min": 0,
                    "y_max": 50 + i,
                },
                confidence=0.8,
                detection_method="visual",
            )
            smoothed = tracker.update([region], frame_idx=i)

        # Final smoothed bounds should be weighted average
        final = smoothed[0]
        assert final.detection_method == "smoothed"
        # With recency weighting, should be closer to later values
        assert final.bounds["x_min"] > 100

    def test_outlier_rejection(self):
        """Test that outliers (large jumps) are rejected."""
        tracker = GoalRegionTracker(
            frame_width=1920,
            frame_height=1080,
            smoothing_window=5,
            max_displacement=50.0,
        )

        # Add initial region
        region1 = GoalRegion(
            name="top",
            bounds={"x_min": 100, "x_max": 300, "y_min": 0, "y_max": 50},
            confidence=0.8,
            detection_method="visual",
        )
        tracker.update([region1], frame_idx=0)

        # Add outlier with large displacement (>50 pixels)
        outlier = GoalRegion(
            name="top",
            bounds={"x_min": 300, "x_max": 500, "y_min": 0, "y_max": 50},  # 200px jump
            confidence=0.8,
            detection_method="visual",
        )
        smoothed = tracker.update([outlier], frame_idx=1)

        # Should return interpolated, not the outlier
        assert len(smoothed) == 1
        assert smoothed[0].detection_method == "interpolated"

    def test_interpolation_for_gaps(self):
        """Test interpolation when no detection available."""
        tracker = GoalRegionTracker(
            frame_width=1920,
            frame_height=1080,
            smoothing_window=5,
            interpolation_max_gap=10,
        )

        # Add initial region
        region = GoalRegion(
            name="top",
            bounds={"x_min": 100, "x_max": 300, "y_min": 0, "y_max": 50},
            confidence=0.8,
            detection_method="visual",
        )
        tracker.update([region], frame_idx=0)

        # Interpolate for frame 5 (within gap)
        interpolated = tracker.interpolate(frame_idx=5)
        assert len(interpolated) == 1
        assert interpolated[0].name == "top"
        assert interpolated[0].detection_method == "interpolated"
        # Confidence should decay
        assert interpolated[0].confidence < 0.8

    def test_interpolation_beyond_max_gap(self):
        """Test that interpolation returns empty beyond max gap."""
        tracker = GoalRegionTracker(
            frame_width=1920,
            frame_height=1080,
            smoothing_window=5,
            interpolation_max_gap=10,
        )

        # Add initial region
        region = GoalRegion(
            name="top",
            bounds={"x_min": 100, "x_max": 300, "y_min": 0, "y_max": 50},
            confidence=0.8,
            detection_method="visual",
        )
        tracker.update([region], frame_idx=0)

        # Try to interpolate beyond max gap
        interpolated = tracker.interpolate(frame_idx=100)
        assert len(interpolated) == 0


class TestGoalRegionDetector:
    """Tests for GoalRegionDetector visual detection."""

    @pytest.fixture
    def config(self):
        """Create default config for tests."""
        return GoalRegionDetectionConfig()

    @pytest.fixture
    def detector(self, config):
        """Create detector with default config."""
        return GoalRegionDetector(
            frame_width=1920,
            frame_height=1080,
            config=config,
        )

    def test_fallback_to_heuristic(self, detector):
        """Test that detector falls back to heuristic for unprocessed frames."""
        regions = detector.get_goal_regions(frame_idx=999)

        # Should return heuristic regions
        assert len(regions) == 2
        for r in regions:
            assert r.detection_method == "heuristic"

    def test_is_in_goal_region_delegates(self, detector):
        """Test that is_in_goal_region works correctly."""
        # Point in center (not in goal)
        is_in, name = detector.is_in_goal_region((960, 540), frame_idx=0)
        assert is_in is False

        # Point near top (in top goal)
        is_in, name = detector.is_in_goal_region((960, 50), frame_idx=0)
        assert is_in is True
        assert name == "top"

    def test_detect_goals_black_frame(self, detector):
        """Test detection on black frame (no features)."""
        black_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        regions = detector.detect_goals(black_frame, frame_idx=0)

        # Should fall back to heuristic due to low confidence
        # (no crossbars or posts detected)
        for r in regions:
            assert r.detection_method in ["heuristic", "blended", "visual"]

    def test_detect_goals_synthetic_goalpost(self, detector):
        """Test detection with synthetic white goalpost."""
        # Create frame with white vertical line (simulating goalpost)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # Draw white vertical lines (goalposts) near top
        # Left post
        frame[10:150, 700:710] = (255, 255, 255)
        # Right post
        frame[10:150, 1210:1220] = (255, 255, 255)
        # Crossbar
        frame[10:15, 700:1220] = (255, 255, 255)

        regions = detector.detect_goals(frame, frame_idx=0)

        # Should detect something (though may blend with heuristic)
        assert len(regions) >= 1
        # At least top goal should be detected
        top_regions = [r for r in regions if r.name == "top"]
        assert len(top_regions) == 1

    def test_caching(self, detector):
        """Test that results are cached."""
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # First call processes the frame
        regions1 = detector.detect_goals(frame, frame_idx=10)

        # Second call should return cached result
        regions2 = detector.detect_goals(frame, frame_idx=10)

        assert regions1 == regions2

    def test_config_heuristic_only(self):
        """Test with detection_method set to heuristic."""
        config = GoalRegionDetectionConfig(detection_method="heuristic")
        detector = GoalRegionDetector(
            frame_width=1920,
            frame_height=1080,
            config=config,
        )

        regions = detector.get_goal_regions(0)

        # Should always use heuristic
        for r in regions:
            assert r.detection_method == "heuristic"


class TestGoalRegionDetectorLines:
    """Tests for line detection in GoalRegionDetector."""

    @pytest.fixture
    def detector(self):
        """Create detector with default config."""
        config = GoalRegionDetectionConfig()
        return GoalRegionDetector(
            frame_width=640,
            frame_height=480,
            config=config,
        )

    def test_detect_horizontal_lines(self, detector):
        """Test detection of horizontal lines."""
        # Create frame with horizontal line
        gray = np.zeros((480, 640), dtype=np.uint8)
        gray[100, 100:500] = 255  # Horizontal line

        lines = detector._detect_pitch_lines(gray)

        # Should detect at least one line
        # Note: detection depends on Hough parameters
        # This is a smoke test to ensure the method runs
        assert isinstance(lines, list)

    def test_detect_vertical_lines(self, detector):
        """Test detection of vertical lines."""
        gray = np.zeros((480, 640), dtype=np.uint8)
        gray[50:200, 100] = 255  # Vertical line

        lines = detector._detect_pitch_lines(gray)
        assert isinstance(lines, list)


class TestConfidenceScoring:
    """Tests for confidence scoring logic."""

    def test_confidence_components(self):
        """Test that confidence scoring weights components correctly."""
        from src.vision.field.goal_detector import _DetectionResult

        # Full detection: crossbar + both posts + in zone
        result = _DetectionResult(
            name="top",
            bounds={"x_min": 100, "x_max": 300, "y_min": 0, "y_max": 50},
            crossbar_detected=True,
            left_post_detected=True,
            right_post_detected=True,
            in_expected_zone=True,
        )

        config = GoalRegionDetectionConfig()
        detector = GoalRegionDetector(1920, 1080, config)
        confidence = detector._compute_confidence(result)

        # Should be high (all components detected)
        assert confidence == pytest.approx(1.0, rel=0.01)

    def test_confidence_partial_detection(self):
        """Test confidence with partial detection."""
        from src.vision.field.goal_detector import _DetectionResult

        # Only crossbar detected
        result = _DetectionResult(
            name="top",
            bounds={"x_min": 100, "x_max": 300, "y_min": 0, "y_max": 50},
            crossbar_detected=True,
            left_post_detected=False,
            right_post_detected=False,
            in_expected_zone=True,  # Zone is inferred from crossbar
        )

        config = GoalRegionDetectionConfig()
        detector = GoalRegionDetector(1920, 1080, config)
        confidence = detector._compute_confidence(result)

        # Should be moderate (crossbar + zone = 0.35 + 0.35 = 0.70)
        assert confidence == pytest.approx(0.70, rel=0.01)

    def test_confidence_no_detection(self):
        """Test confidence with no features detected."""
        from src.vision.field.goal_detector import _DetectionResult

        result = _DetectionResult(
            name="top",
            bounds={"x_min": 100, "x_max": 300, "y_min": 0, "y_max": 50},
            crossbar_detected=False,
            left_post_detected=False,
            right_post_detected=False,
            in_expected_zone=False,
        )

        config = GoalRegionDetectionConfig()
        detector = GoalRegionDetector(1920, 1080, config)
        confidence = detector._compute_confidence(result)

        assert confidence == 0.0


class TestFallbackStrategy:
    """Tests for fallback blending strategy."""

    def test_high_confidence_uses_visual(self):
        """Test that high confidence uses visual detection."""
        config = GoalRegionDetectionConfig(
            fallback_confidence_threshold=0.3,
            blend_threshold=0.6,
        )
        detector = GoalRegionDetector(1920, 1080, config)

        visual = GoalRegion(
            name="top",
            bounds={"x_min": 100, "x_max": 300, "y_min": 0, "y_max": 60},
            confidence=0.9,
            detection_method="visual",
        )

        result = detector._apply_fallback([visual], frame_idx=0)
        # Should have 2 regions (top from visual, bottom from heuristic fallback)
        assert len(result) == 2

        # Find the top region - should use visual since confidence is high
        top = next((r for r in result if r.name == "top"), None)
        assert top is not None
        assert top.detection_method == "visual"

    def test_medium_confidence_blends(self):
        """Test that medium confidence blends visual and heuristic."""
        config = GoalRegionDetectionConfig(
            fallback_confidence_threshold=0.3,
            blend_threshold=0.6,
        )
        detector = GoalRegionDetector(1920, 1080, config)

        visual = GoalRegion(
            name="top",
            bounds={"x_min": 100, "x_max": 300, "y_min": 0, "y_max": 60},
            confidence=0.45,  # Between 0.3 and 0.6
            detection_method="visual",
        )

        result = detector._apply_fallback([visual], frame_idx=0)
        assert len(result) >= 1

        # Find the top region
        top = next((r for r in result if r.name == "top"), None)
        assert top is not None
        assert top.detection_method == "blended"

    def test_low_confidence_uses_heuristic(self):
        """Test that low confidence uses heuristic."""
        config = GoalRegionDetectionConfig(
            fallback_confidence_threshold=0.3,
            blend_threshold=0.6,
        )
        detector = GoalRegionDetector(1920, 1080, config)

        visual = GoalRegion(
            name="top",
            bounds={"x_min": 100, "x_max": 300, "y_min": 0, "y_max": 60},
            confidence=0.1,  # Below 0.3
            detection_method="visual",
        )

        result = detector._apply_fallback([visual], frame_idx=0)

        # Should use heuristic
        top = next((r for r in result if r.name == "top"), None)
        assert top is not None
        assert top.detection_method == "heuristic"


class TestIntegration:
    """Integration tests for goal detection pipeline."""

    def test_full_pipeline_heuristic_mode(self):
        """Test full pipeline in heuristic-only mode."""
        config = GoalRegionDetectionConfig(detection_method="heuristic")
        detector = GoalRegionDetector(1920, 1080, config)

        # Check multiple points
        test_points = [
            ((960, 50), True, "top"),
            ((960, 1030), True, "bottom"),
            ((960, 540), False, None),
            ((100, 50), False, None),  # Outside goal width
        ]

        for point, expected_in, expected_name in test_points:
            is_in, name = detector.is_in_goal_region(point, frame_idx=0)
            assert is_in == expected_in, f"Point {point}: expected in_goal={expected_in}"
            assert name == expected_name, f"Point {point}: expected name={expected_name}"

    def test_multiple_frames_with_smoothing(self):
        """Test processing multiple frames with temporal smoothing."""
        config = GoalRegionDetectionConfig(
            detection_method="hybrid",
            enable_temporal_smoothing=True,
            smoothing_window_frames=5,
        )
        detector = GoalRegionDetector(1920, 1080, config)

        # Create simple frames
        frames = [np.zeros((1080, 1920, 3), dtype=np.uint8) for _ in range(10)]

        # Process frames
        for i, frame in enumerate(frames):
            detector.detect_goals(frame, frame_idx=i)

        # Check that we can query any frame
        for i in range(10):
            regions = detector.get_goal_regions(i)
            assert len(regions) >= 2  # At least top and bottom
