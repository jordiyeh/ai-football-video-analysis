"""Regression tests for VideoReader cleanup edge cases."""

from src.video.reader import VideoReader


def test_cleanup_is_safe_on_partially_initialized_instance() -> None:
    """close/__del__ should be no-ops when cap was never created."""
    reader = VideoReader.__new__(VideoReader)

    reader.close()
    reader.__del__()
