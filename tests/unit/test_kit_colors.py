"""Unit tests for kit color extraction."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.vision.team.kit_colors import extract_kit_colors, hsv_to_hex


def _create_test_image(path: Path, color_bgr: tuple[int, int, int], width: int = 100, height: int = 100):
    """Create a solid-color test image."""
    img = np.full((height, width, 3), color_bgr, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _create_two_color_image(path: Path, color1_bgr: tuple, color2_bgr: tuple, width: int = 100, height: int = 100):
    """Create an image with two horizontal bands."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    mid = height // 2
    img[:mid, :] = color1_bgr
    img[mid:, :] = color2_bgr
    cv2.imwrite(str(path), img)


class TestExtractKitColors:
    """Tests for extract_kit_colors function."""

    def test_solid_red_image(self, tmp_path):
        img_path = tmp_path / "red.png"
        _create_test_image(img_path, (0, 0, 200))  # BGR red
        primary, secondary = extract_kit_colors(img_path)

        assert primary is not None
        assert len(primary) == 3
        # Red in HSV should have H near 0 or 180
        # The V channel should be high
        assert primary[2] > 100

    def test_solid_blue_image(self, tmp_path):
        img_path = tmp_path / "blue.png"
        _create_test_image(img_path, (200, 0, 0))  # BGR blue
        primary, secondary = extract_kit_colors(img_path)

        assert primary is not None
        # Blue in HSV should have H around 120 (OpenCV convention)
        assert 90 < primary[0] < 140

    def test_two_color_image_returns_both(self, tmp_path):
        img_path = tmp_path / "two_color.png"
        _create_two_color_image(img_path, (200, 0, 0), (0, 200, 0))  # Blue + Green
        primary, secondary = extract_kit_colors(img_path, n_dominant=2)

        assert primary is not None
        assert secondary is not None
        # The two colors should be different
        assert not np.allclose(primary, secondary, atol=10)

    def test_invalid_image_raises(self, tmp_path):
        bad_path = tmp_path / "notanimage.txt"
        bad_path.write_text("not an image")
        with pytest.raises(ValueError, match="Could not read"):
            extract_kit_colors(bad_path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Could not read"):
            extract_kit_colors(tmp_path / "missing.png")

    def test_n_dominant_1(self, tmp_path):
        img_path = tmp_path / "test.png"
        _create_test_image(img_path, (0, 180, 0))
        primary, secondary = extract_kit_colors(img_path, n_dominant=1)
        assert primary is not None
        assert secondary is None

    def test_near_white_filtered(self, tmp_path):
        """Near-white pixels should be filtered, falling back to all pixels."""
        img_path = tmp_path / "white.png"
        _create_test_image(img_path, (250, 250, 250))
        # Should not crash, will fall back to using all pixels
        primary, secondary = extract_kit_colors(img_path)
        assert primary is not None


class TestHsvToHex:
    """Tests for hsv_to_hex conversion."""

    def test_red_hsv_to_hex(self):
        # Red in OpenCV HSV: H=0, S=255, V=255
        result = hsv_to_hex([0, 255, 255])
        assert result.startswith('#')
        assert len(result) == 7
        # Should be near pure red
        r = int(result[1:3], 16)
        assert r > 200

    def test_blue_hsv_to_hex(self):
        # Blue in OpenCV HSV: H=120, S=255, V=255
        result = hsv_to_hex([120, 255, 255])
        b = int(result[5:7], 16)
        assert b > 200

    def test_green_hsv_to_hex(self):
        # Green in OpenCV HSV: H=60, S=255, V=255
        result = hsv_to_hex([60, 255, 255])
        g = int(result[3:5], 16)
        assert g > 200

    def test_accepts_numpy_array(self):
        hsv = np.array([60, 255, 255], dtype=np.float32)
        result = hsv_to_hex(hsv)
        assert result.startswith('#')

    def test_black(self):
        result = hsv_to_hex([0, 0, 0])
        assert result == '#000000'
