"""Tests for overlay rendering."""

from src.export.overlay import hex_to_bgr
from src.config.schemas import OverlayConfig


def test_hex_to_bgr_conversion():
    """Test hex color to BGR conversion."""
    # Red
    assert hex_to_bgr("#FF0000") == (0, 0, 255)

    # Green
    assert hex_to_bgr("#00FF00") == (0, 255, 0)

    # Blue
    assert hex_to_bgr("#0000FF") == (255, 0, 0)

    # White
    assert hex_to_bgr("#FFFFFF") == (255, 255, 255)

    # Black
    assert hex_to_bgr("#000000") == (0, 0, 0)


def test_hex_to_bgr_without_hash():
    """Test hex conversion without # prefix."""
    assert hex_to_bgr("FF0000") == (0, 0, 255)
    assert hex_to_bgr("00FF00") == (0, 255, 0)


def test_overlay_config_colors():
    """Test overlay config color conversions."""
    config = OverlayConfig(
        player_color="#00FF00",
        ball_color="#FF0000",
    )

    player_bgr = hex_to_bgr(config.player_color)
    ball_bgr = hex_to_bgr(config.ball_color)

    assert player_bgr == (0, 255, 0)  # Green in BGR
    assert ball_bgr == (0, 0, 255)  # Red in BGR
