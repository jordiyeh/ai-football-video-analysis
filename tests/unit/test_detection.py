"""Tests for detection system."""

import numpy as np
from src.vision.detect.yolo import Detection


def test_detection_creation():
    """Test Detection object creation."""
    det = Detection(
        object_type="player",
        bbox=(100, 100, 200, 200),
        confidence=0.85,
        class_id=0,
    )

    assert det.object_type == "player"
    assert det.bbox == (100, 100, 200, 200)
    assert det.confidence == 0.85
    assert det.class_id == 0


def test_detection_center():
    """Test detection center calculation."""
    det = Detection(
        object_type="player",
        bbox=(100, 100, 200, 200),
        confidence=0.85,
        class_id=0,
    )

    center = det.center
    assert center == (150, 150)


def test_detection_dimensions():
    """Test detection width and height."""
    det = Detection(
        object_type="ball",
        bbox=(50, 50, 100, 150),
        confidence=0.9,
        class_id=32,
    )

    assert det.width == 50
    assert det.height == 100
    assert det.area == 5000


def test_detection_to_dict():
    """Test detection serialization."""
    det = Detection(
        object_type="player",
        bbox=(100, 100, 200, 200),
        confidence=0.85,
        class_id=0,
    )

    det_dict = det.to_dict()

    assert det_dict["object_type"] == "player"
    assert det_dict["bbox"] == [100, 100, 200, 200]
    assert det_dict["center"] == [150, 150]
    assert det_dict["confidence"] == 0.85
    assert det_dict["width"] == 100
    assert det_dict["height"] == 100
