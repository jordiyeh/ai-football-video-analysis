"""Tests for jersey color extraction and analysis."""

import pytest
import numpy as np

from src.vision.team.colors import (
    extract_jersey_color,
    extract_dominant_color_kmeans,
    bgr_to_hsv,
    color_distance,
    is_similar_color,
)


# -----------------------------------------------------------------------------
# extract_jersey_color Tests
# -----------------------------------------------------------------------------

class TestExtractJerseyColor:
    """Tests for extract_jersey_color function."""

    @pytest.fixture
    def solid_red_frame(self):
        """Create a frame with solid red color."""
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[:, :] = [0, 0, 255]  # Red in BGR
        return frame

    @pytest.fixture
    def solid_blue_frame(self):
        """Create a frame with solid blue color."""
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[:, :] = [255, 0, 0]  # Blue in BGR
        return frame

    @pytest.fixture
    def frame_with_regions(self):
        """Create a frame with different colored regions."""
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # Red region at (100, 100) to (200, 300)
        frame[100:300, 100:200] = [0, 0, 255]
        # Blue region at (500, 100) to (600, 300)
        frame[100:300, 500:600] = [255, 0, 0]
        return frame

    def test_valid_region_uniform_color(self, solid_red_frame):
        """Test extraction from valid region with uniform color."""
        bbox = (100.0, 100.0, 200.0, 300.0)

        color = extract_jersey_color(solid_red_frame, bbox)

        # Should be red [0, 0, 255] in BGR
        assert color[2] == pytest.approx(255.0)  # Red channel
        assert color[0] == pytest.approx(0.0)    # Blue channel
        assert color[1] == pytest.approx(0.0)    # Green channel

    def test_nan_bbox_returns_black(self, solid_red_frame):
        """Test that NaN bbox returns black."""
        bbox = (float('nan'), 100.0, 200.0, 300.0)

        color = extract_jersey_color(solid_red_frame, bbox)

        assert np.allclose(color, [0, 0, 0])

    def test_inf_bbox_returns_black(self, solid_red_frame):
        """Test that inf bbox returns black."""
        bbox = (100.0, float('inf'), 200.0, 300.0)

        color = extract_jersey_color(solid_red_frame, bbox)

        assert np.allclose(color, [0, 0, 0])

    def test_zero_size_bbox_returns_black(self, solid_red_frame):
        """Test that zero-size bbox returns black."""
        bbox = (100.0, 100.0, 100.0, 100.0)  # Zero width and height

        color = extract_jersey_color(solid_red_frame, bbox)

        assert np.allclose(color, [0, 0, 0])

    def test_negative_size_bbox_returns_black(self, solid_red_frame):
        """Test that negative-size bbox returns black."""
        bbox = (200.0, 300.0, 100.0, 100.0)  # Inverted coordinates

        color = extract_jersey_color(solid_red_frame, bbox)

        assert np.allclose(color, [0, 0, 0])

    def test_clipping_to_frame_bounds(self, solid_blue_frame):
        """Test that bbox is clipped to frame bounds."""
        # Bbox extends beyond frame
        bbox = (1800.0, 900.0, 2000.0, 1200.0)

        color = extract_jersey_color(solid_blue_frame, bbox)

        # Should still extract blue from the portion within frame
        # (May return black if clipping results in empty region)
        assert color.shape == (3,)

    def test_upper_sample_region(self, frame_with_regions):
        """Test upper sample region (top 40%)."""
        # Create frame where top and bottom have different colors
        frame = np.zeros((200, 100, 3), dtype=np.uint8)
        frame[0:80, :] = [0, 0, 255]    # Top 40% red
        frame[80:200, :] = [255, 0, 0]  # Bottom 60% blue

        bbox = (0.0, 0.0, 100.0, 200.0)

        color = extract_jersey_color(frame, bbox, sample_region="upper")

        # Should be mostly red (sampling top 40%)
        assert color[2] > 200  # High red
        assert color[0] < 50   # Low blue

    def test_center_sample_region(self):
        """Test center sample region (middle 40%)."""
        frame = np.zeros((200, 100, 3), dtype=np.uint8)
        frame[0:60, :] = [0, 0, 255]     # Top 30% red
        frame[60:140, :] = [0, 255, 0]   # Middle 40% green
        frame[140:200, :] = [255, 0, 0]  # Bottom 30% blue

        bbox = (0.0, 0.0, 100.0, 200.0)

        color = extract_jersey_color(frame, bbox, sample_region="center")

        # Should be mostly green
        assert color[1] > 200  # High green
        assert color[0] < 50   # Low blue
        assert color[2] < 50   # Low red

    def test_full_sample_region(self, solid_red_frame):
        """Test full sample region."""
        bbox = (100.0, 100.0, 200.0, 300.0)

        color = extract_jersey_color(solid_red_frame, bbox, sample_region="full")

        # Should get the full region color
        assert color[2] == pytest.approx(255.0)  # Red

    def test_returns_float32_array(self, solid_red_frame):
        """Test that result is float32 numpy array."""
        bbox = (100.0, 100.0, 200.0, 300.0)

        color = extract_jersey_color(solid_red_frame, bbox)

        assert isinstance(color, np.ndarray)
        assert color.dtype == np.float32
        assert color.shape == (3,)


# -----------------------------------------------------------------------------
# extract_dominant_color_kmeans Tests
# -----------------------------------------------------------------------------

class TestExtractDominantColorKmeans:
    """Tests for extract_dominant_color_kmeans function."""

    @pytest.fixture
    def uniform_frame(self):
        """Create a frame with uniform color."""
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[:, :] = [100, 150, 200]  # Some color
        return frame

    @pytest.fixture
    def mixed_frame(self):
        """Create a frame with mixed colors (dominant red)."""
        frame = np.zeros((200, 100, 3), dtype=np.uint8)
        # 70% red
        frame[0:140, :] = [0, 0, 255]
        # 30% blue
        frame[140:200, :] = [255, 0, 0]
        return frame

    def test_uniform_color_detection(self, uniform_frame):
        """Test detection of uniform color."""
        bbox = (100.0, 100.0, 200.0, 300.0)

        color = extract_dominant_color_kmeans(uniform_frame, bbox, n_colors=3)

        # Should match the uniform color
        assert color[0] == pytest.approx(100.0, abs=5)
        assert color[1] == pytest.approx(150.0, abs=5)
        assert color[2] == pytest.approx(200.0, abs=5)

    def test_dominant_color_from_mixed(self, mixed_frame):
        """Test extraction of dominant color from mixed region."""
        bbox = (0.0, 0.0, 100.0, 200.0)

        color = extract_dominant_color_kmeans(mixed_frame, bbox, n_colors=2)

        # Should be red (70% of region)
        assert color[2] > 200  # High red
        assert color[0] < 50   # Low blue

    def test_nan_bbox_returns_black(self, uniform_frame):
        """Test that NaN bbox returns black."""
        bbox = (float('nan'), 100.0, 200.0, 300.0)

        color = extract_dominant_color_kmeans(uniform_frame, bbox)

        assert np.allclose(color, [0, 0, 0])

    def test_inf_bbox_returns_black(self, uniform_frame):
        """Test that inf bbox returns black."""
        bbox = (100.0, float('inf'), 200.0, 300.0)

        color = extract_dominant_color_kmeans(uniform_frame, bbox)

        assert np.allclose(color, [0, 0, 0])

    def test_zero_size_bbox_returns_black(self, uniform_frame):
        """Test that zero-size bbox returns black."""
        bbox = (100.0, 100.0, 100.0, 100.0)

        color = extract_dominant_color_kmeans(uniform_frame, bbox)

        assert np.allclose(color, [0, 0, 0])

    def test_returns_float32_array(self, uniform_frame):
        """Test that result is float32 numpy array."""
        bbox = (100.0, 100.0, 200.0, 300.0)

        color = extract_dominant_color_kmeans(uniform_frame, bbox)

        assert isinstance(color, np.ndarray)
        assert color.dtype == np.float32
        assert color.shape == (3,)

    def test_n_colors_parameter(self, mixed_frame):
        """Test n_colors parameter affects clustering."""
        bbox = (0.0, 0.0, 100.0, 200.0)

        # With more colors, should still find the dominant one
        color1 = extract_dominant_color_kmeans(mixed_frame, bbox, n_colors=2)
        color3 = extract_dominant_color_kmeans(mixed_frame, bbox, n_colors=5)

        # Both should identify red as dominant
        assert color1[2] > 200
        assert color3[2] > 200


# -----------------------------------------------------------------------------
# bgr_to_hsv Tests
# -----------------------------------------------------------------------------

class TestBgrToHsv:
    """Tests for bgr_to_hsv conversion."""

    def test_red_conversion(self):
        """Test conversion of pure red."""
        bgr = np.array([0, 0, 255], dtype=np.float32)  # Pure red in BGR

        hsv = bgr_to_hsv(bgr)

        # Red in HSV: H=0 or H=180 (hue), S=255 (saturated), V=255 (bright)
        assert hsv[0] == pytest.approx(0.0, abs=5) or hsv[0] == pytest.approx(180.0, abs=5)
        assert hsv[1] == pytest.approx(255.0, abs=5)  # Saturation
        assert hsv[2] == pytest.approx(255.0, abs=5)  # Value

    def test_green_conversion(self):
        """Test conversion of pure green."""
        bgr = np.array([0, 255, 0], dtype=np.float32)  # Pure green in BGR

        hsv = bgr_to_hsv(bgr)

        # Green in HSV: H=60 (in 0-180 scale)
        assert hsv[0] == pytest.approx(60.0, abs=5)
        assert hsv[1] == pytest.approx(255.0, abs=5)
        assert hsv[2] == pytest.approx(255.0, abs=5)

    def test_blue_conversion(self):
        """Test conversion of pure blue."""
        bgr = np.array([255, 0, 0], dtype=np.float32)  # Pure blue in BGR

        hsv = bgr_to_hsv(bgr)

        # Blue in HSV: H=120 (in 0-180 scale)
        assert hsv[0] == pytest.approx(120.0, abs=5)
        assert hsv[1] == pytest.approx(255.0, abs=5)
        assert hsv[2] == pytest.approx(255.0, abs=5)

    def test_white_conversion(self):
        """Test conversion of white."""
        bgr = np.array([255, 255, 255], dtype=np.float32)  # White

        hsv = bgr_to_hsv(bgr)

        # White: S=0 (no saturation), V=255 (bright)
        assert hsv[1] == pytest.approx(0.0, abs=5)  # No saturation
        assert hsv[2] == pytest.approx(255.0, abs=5)  # Full value

    def test_black_conversion(self):
        """Test conversion of black."""
        bgr = np.array([0, 0, 0], dtype=np.float32)  # Black

        hsv = bgr_to_hsv(bgr)

        # Black: V=0 (no brightness)
        assert hsv[2] == pytest.approx(0.0, abs=5)  # No value

    def test_returns_float32_array(self):
        """Test that result is float32 numpy array."""
        bgr = np.array([100, 150, 200])

        hsv = bgr_to_hsv(bgr)

        assert isinstance(hsv, np.ndarray)
        assert hsv.dtype == np.float32
        assert hsv.shape == (3,)


# -----------------------------------------------------------------------------
# color_distance Tests
# -----------------------------------------------------------------------------

class TestColorDistance:
    """Tests for color_distance function."""

    def test_identical_colors_bgr(self):
        """Test distance between identical BGR colors is 0."""
        color = np.array([100, 150, 200])

        distance = color_distance(color, color, space="bgr")

        assert distance == pytest.approx(0.0)

    def test_identical_colors_hsv(self):
        """Test distance between identical HSV colors is 0."""
        color = np.array([60, 200, 200])

        distance = color_distance(color, color, space="hsv")

        assert distance == pytest.approx(0.0)

    def test_different_colors_bgr(self):
        """Test distance between different BGR colors."""
        color1 = np.array([0, 0, 0])
        color2 = np.array([100, 0, 0])

        distance = color_distance(color1, color2, space="bgr")

        # Euclidean distance = 100
        assert distance == pytest.approx(100.0)

    def test_hsv_hue_wraparound(self):
        """Test HSV hue wraparound (0 and 180 are close)."""
        # Hue 0 (red) and Hue 170 (also reddish)
        color1 = np.array([0, 255, 255])
        color2 = np.array([170, 255, 255])

        distance = color_distance(color1, color2, space="hsv")

        # Hue difference should be wrapped: min(170, 180-170) = 10
        # With hue weight of 2: sqrt((10*2)^2) = 20
        # Distance should be around 20 (just from hue)
        assert distance < 50  # Should be relatively small due to wraparound

    def test_hsv_opposite_hues(self):
        """Test HSV with opposite hues."""
        # Hue 0 (red) and Hue 90 (somewhere between green and cyan)
        color1 = np.array([0, 255, 255])
        color2 = np.array([90, 255, 255])

        distance = color_distance(color1, color2, space="hsv")

        # Hue difference = min(90, 180-90) = 90
        assert distance > 100  # Significant difference

    def test_returns_float(self):
        """Test that result is a float."""
        color1 = np.array([100, 150, 200])
        color2 = np.array([110, 160, 210])

        distance = color_distance(color1, color2)

        assert isinstance(distance, float)


# -----------------------------------------------------------------------------
# is_similar_color Tests
# -----------------------------------------------------------------------------

class TestIsSimilarColor:
    """Tests for is_similar_color function."""

    def test_identical_colors_similar(self):
        """Test that identical colors are similar."""
        color = np.array([100, 150, 200])

        result = is_similar_color(color, color, threshold=50.0)

        assert result is True

    def test_very_different_colors_not_similar(self):
        """Test that very different colors are not similar."""
        color1 = np.array([0, 0, 0])
        color2 = np.array([255, 255, 255])

        result = is_similar_color(color1, color2, threshold=50.0)

        assert result is False

    def test_threshold_boundary_below(self):
        """Test colors just below threshold."""
        color1 = np.array([0, 0, 0])
        color2 = np.array([40, 0, 0])  # Distance = 40

        result = is_similar_color(color1, color2, threshold=50.0)

        assert result is True

    def test_threshold_boundary_above(self):
        """Test colors just above threshold."""
        color1 = np.array([0, 0, 0])
        color2 = np.array([60, 0, 0])  # Distance = 60

        result = is_similar_color(color1, color2, threshold=50.0)

        assert result is False

    def test_threshold_exact_boundary(self):
        """Test colors exactly at threshold."""
        color1 = np.array([0, 0, 0])
        color2 = np.array([50, 0, 0])  # Distance = 50

        # Function uses < (strict), so equal is not similar
        result = is_similar_color(color1, color2, threshold=50.0)

        assert result is False

    def test_custom_threshold(self):
        """Test with custom threshold."""
        color1 = np.array([100, 100, 100])
        color2 = np.array([150, 100, 100])  # Distance = 50

        # With threshold 100, should be similar
        assert is_similar_color(color1, color2, threshold=100.0) is True

        # With threshold 30, should not be similar
        assert is_similar_color(color1, color2, threshold=30.0) is False

    def test_hsv_space(self):
        """Test similarity in HSV space."""
        color1 = np.array([0, 200, 200])  # Red-ish in HSV
        color2 = np.array([5, 200, 200])  # Slightly different red

        # Should be similar in HSV (small hue difference)
        result = is_similar_color(color1, color2, threshold=50.0, space="hsv")

        assert result is True

    def test_returns_bool(self):
        """Test that result is boolean."""
        color1 = np.array([100, 150, 200])
        color2 = np.array([110, 160, 210])

        result = is_similar_color(color1, color2)

        assert isinstance(result, bool)
