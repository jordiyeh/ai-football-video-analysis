"""Unit tests for detector ensemble and ball specialist."""

import numpy as np
import pytest

from src.vision.detect.yolo import Detection
from src.vision.detect.ensemble import DetectorEnsemble, compute_iou
from src.vision.detect.base import ObjectDetector
from src.config.schemas import (
    BallSpecialistConfig,
    EnsembleConfig,
    DetectionConfig,
    PipelineConfig,
)


class MockDetector(ObjectDetector):
    """Mock detector for testing."""

    def __init__(
        self,
        name: str,
        supported_types: list,
        detections: list[Detection] | None = None,
    ):
        self._name = name
        self._supported_types = supported_types
        self._detections = detections or []

    @property
    def name(self) -> str:
        return self._name

    @property
    def supported_types(self) -> list:
        return self._supported_types

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return self._detections

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        return [self._detections for _ in frames]


class TestComputeIoU:
    """Tests for IoU computation."""

    def test_identical_boxes(self):
        """Identical boxes should have IoU = 1.0."""
        box = (10, 10, 50, 50)
        assert compute_iou(box, box) == 1.0

    def test_no_overlap(self):
        """Non-overlapping boxes should have IoU = 0.0."""
        box1 = (0, 0, 10, 10)
        box2 = (20, 20, 30, 30)
        assert compute_iou(box1, box2) == 0.0

    def test_partial_overlap(self):
        """Partially overlapping boxes should have IoU between 0 and 1."""
        box1 = (0, 0, 20, 20)
        box2 = (10, 10, 30, 30)
        iou = compute_iou(box1, box2)
        assert 0 < iou < 1
        # Overlap area = 10*10 = 100
        # Union = 400 + 400 - 100 = 700
        assert abs(iou - 100 / 700) < 0.001

    def test_one_inside_other(self):
        """Small box inside large box."""
        large = (0, 0, 100, 100)
        small = (25, 25, 75, 75)
        iou = compute_iou(large, small)
        # Intersection = 50*50 = 2500
        # Union = 10000 + 2500 - 2500 = 10000
        assert abs(iou - 0.25) < 0.001


class TestDetectorEnsemble:
    """Tests for DetectorEnsemble."""

    def test_single_detector(self):
        """Ensemble with single detector should return its detections."""
        detection = Detection(
            object_type="ball",
            bbox=(100, 100, 120, 120),
            confidence=0.8,
            class_id=0,
        )
        mock = MockDetector("test", ["ball"], [detection])
        ensemble = DetectorEnsemble({"test": mock})

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = ensemble.detect(frame)

        assert len(results) == 1
        assert results[0].object_type == "ball"
        assert results[0].confidence == 0.8

    def test_identical_detections_fused(self):
        """Identical detections from two detectors should be fused."""
        detection1 = Detection(
            object_type="ball",
            bbox=(100, 100, 120, 120),
            confidence=0.7,
            class_id=0,
        )
        detection2 = Detection(
            object_type="ball",
            bbox=(100, 100, 120, 120),
            confidence=0.8,
            class_id=0,
        )
        mock1 = MockDetector("det1", ["ball"], [detection1])
        mock2 = MockDetector("det2", ["ball"], [detection2])

        ensemble = DetectorEnsemble(
            {"det1": mock1, "det2": mock2},
            weights={"det1": 1.0, "det2": 1.0},
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = ensemble.detect(frame)

        # Should fuse into single detection
        assert len(results) == 1
        # Confidence should be boosted (max * boost factor)
        # Boost = 1.0 + 0.1 * (2 - 1) = 1.1
        assert results[0].confidence == pytest.approx(0.8 * 1.1, rel=0.01)

    def test_weighted_average_coordinates(self):
        """WBF should compute weighted average of coordinates."""
        # Two detectors with slightly different boxes
        detection1 = Detection(
            object_type="ball",
            bbox=(100, 100, 120, 120),  # Center at (110, 110)
            confidence=0.8,
            class_id=0,
        )
        detection2 = Detection(
            object_type="ball",
            bbox=(104, 104, 124, 124),  # Center at (114, 114), overlaps with above
            confidence=0.6,
            class_id=0,
        )
        mock1 = MockDetector("det1", ["ball"], [detection1])
        mock2 = MockDetector("det2", ["ball"], [detection2])

        # Equal weights
        ensemble = DetectorEnsemble(
            {"det1": mock1, "det2": mock2},
            weights={"det1": 1.0, "det2": 1.0},
            iou_threshold=0.3,  # Ensure they overlap enough to fuse
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = ensemble.detect(frame)

        # Should fuse into single detection
        assert len(results) == 1

        # Weighted average: w1=0.8*1.0=0.8, w2=0.6*1.0=0.6, total=1.4
        # x1 = (100*0.8 + 104*0.6) / 1.4 = 142.4 / 1.4 ≈ 101.7
        fused_bbox = results[0].bbox
        expected_x1 = (100 * 0.8 + 104 * 0.6) / 1.4
        assert abs(fused_bbox[0] - expected_x1) < 0.1

    def test_non_overlapping_boxes_preserved(self):
        """Non-overlapping boxes should be kept separate."""
        detection1 = Detection(
            object_type="ball",
            bbox=(10, 10, 30, 30),
            confidence=0.8,
            class_id=0,
        )
        detection2 = Detection(
            object_type="ball",
            bbox=(200, 200, 220, 220),  # Far away
            confidence=0.7,
            class_id=0,
        )
        mock = MockDetector("test", ["ball"], [detection1, detection2])
        ensemble = DetectorEnsemble({"test": mock})

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = ensemble.detect(frame)

        assert len(results) == 2

    def test_detector_weights_respected(self):
        """Higher weighted detector should have more influence."""
        detection1 = Detection(
            object_type="ball",
            bbox=(100, 100, 120, 120),
            confidence=0.5,
            class_id=0,
        )
        detection2 = Detection(
            object_type="ball",
            bbox=(105, 105, 125, 125),
            confidence=0.5,
            class_id=0,
        )
        mock1 = MockDetector("det1", ["ball"], [detection1])
        mock2 = MockDetector("det2", ["ball"], [detection2])

        # det2 has much higher weight
        ensemble = DetectorEnsemble(
            {"det1": mock1, "det2": mock2},
            weights={"det1": 1.0, "det2": 10.0},
            iou_threshold=0.3,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = ensemble.detect(frame)

        # Fused box should be closer to detection2
        fused_bbox = results[0].bbox
        # x1 should be closer to 105 than 100
        assert fused_bbox[0] > 103  # Closer to 105

    def test_object_type_separation(self):
        """Ball and player detections should be fused separately."""
        ball = Detection(
            object_type="ball",
            bbox=(100, 100, 120, 120),
            confidence=0.8,
            class_id=32,
        )
        player = Detection(
            object_type="player",
            bbox=(100, 100, 200, 300),  # Same top-left corner, different size
            confidence=0.9,
            class_id=0,
        )
        mock = MockDetector("test", ["ball", "player"], [ball, player])
        ensemble = DetectorEnsemble({"test": mock})

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = ensemble.detect(frame)

        # Both should be preserved (not fused together)
        assert len(results) == 2
        types = {r.object_type for r in results}
        assert types == {"ball", "player"}

    def test_batch_detection(self):
        """Batch detection should process all frames."""
        detection = Detection(
            object_type="ball",
            bbox=(100, 100, 120, 120),
            confidence=0.8,
            class_id=0,
        )
        mock = MockDetector("test", ["ball"], [detection])
        ensemble = DetectorEnsemble({"test": mock})

        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
        results = ensemble.detect_batch(frames)

        assert len(results) == 5
        for frame_results in results:
            assert len(frame_results) == 1

    def test_empty_detections(self):
        """Empty detections should return empty list."""
        mock = MockDetector("test", ["ball"], [])
        ensemble = DetectorEnsemble({"test": mock})

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = ensemble.detect(frame)

        assert results == []

    def test_skip_box_threshold(self):
        """Low confidence boxes should be skipped."""
        detection = Detection(
            object_type="ball",
            bbox=(100, 100, 120, 120),
            confidence=0.005,  # Below default threshold
            class_id=0,
        )
        mock = MockDetector("test", ["ball"], [detection])
        ensemble = DetectorEnsemble(
            {"test": mock},
            skip_box_threshold=0.01,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = ensemble.detect(frame)

        assert results == []


class TestConfigBackwardCompatibility:
    """Tests for backward compatibility of configuration."""

    def test_default_configs_disabled(self):
        """New configs should be disabled by default."""
        config = DetectionConfig()

        assert config.ball_specialist.enabled is False
        assert config.ensemble.enabled is False

    def test_pipeline_config_with_defaults(self):
        """PipelineConfig should work with all defaults."""
        config = PipelineConfig()

        # Should not raise
        assert config.detection.ball_specialist.enabled is False
        assert config.detection.ensemble.enabled is False

    def test_ball_specialist_config_values(self):
        """BallSpecialistConfig should have expected defaults."""
        config = BallSpecialistConfig()

        assert config.model_source == "keremberke/yolov8n-soccer-ball-detection"
        assert config.confidence_threshold == 0.3
        assert config.ball_class_id == 0
        assert config.max_size_ratio == 0.08
        assert config.max_aspect_ratio == 3.0
        assert config.cache_dir == "models"

    def test_ensemble_config_values(self):
        """EnsembleConfig should have expected defaults."""
        config = EnsembleConfig()

        assert config.weights == {"yolo": 1.0, "ball_specialist": 1.5}
        assert config.iou_threshold == 0.5
        assert config.skip_box_threshold == 0.01
        assert config.fusion_type == "wbf"


class TestNMSFusion:
    """Tests for NMS fusion mode."""

    def test_nms_keeps_highest_confidence(self):
        """NMS should keep highest confidence box."""
        detection1 = Detection(
            object_type="ball",
            bbox=(100, 100, 120, 120),
            confidence=0.9,
            class_id=0,
        )
        detection2 = Detection(
            object_type="ball",
            bbox=(102, 102, 122, 122),  # Overlapping
            confidence=0.7,
            class_id=0,
        )
        mock = MockDetector("test", ["ball"], [detection1, detection2])
        ensemble = DetectorEnsemble(
            {"test": mock},
            fusion_type="nms",
            iou_threshold=0.3,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = ensemble.detect(frame)

        # NMS should keep only highest confidence
        assert len(results) == 1
        assert results[0].confidence == 0.9


class TestSoftNMSFusion:
    """Tests for Soft-NMS fusion mode."""

    def test_soft_nms_reduces_overlapping_confidence(self):
        """Soft-NMS should reduce confidence of overlapping boxes."""
        detection1 = Detection(
            object_type="ball",
            bbox=(100, 100, 120, 120),
            confidence=0.9,
            class_id=0,
        )
        detection2 = Detection(
            object_type="ball",
            bbox=(105, 105, 125, 125),  # Overlapping
            confidence=0.8,
            class_id=0,
        )
        mock = MockDetector("test", ["ball"], [detection1, detection2])
        ensemble = DetectorEnsemble(
            {"test": mock},
            fusion_type="soft_nms",
            iou_threshold=0.3,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = ensemble.detect(frame)

        # Soft-NMS may keep both but with reduced confidence on second
        assert len(results) >= 1
        # First box should keep original confidence
        assert results[0].confidence == 0.9
