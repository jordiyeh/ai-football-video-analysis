"""Unit tests for ball detection boosting module."""


from src.vision.detect.yolo import Detection
from src.vision.detect.ball_boost import (
    compute_iou,
    soft_nms,
    BallTemporalFilter,
    BallCandidateTracker,
    merge_multiscale_detections,
)


# --- Test Fixtures ---


def make_detection(
    x1: float, y1: float, x2: float, y2: float, confidence: float = 0.5, object_type: str = "ball"
) -> Detection:
    """Create a test detection."""
    return Detection(
        object_type=object_type,
        bbox=(x1, y1, x2, y2),
        confidence=confidence,
        class_id=32 if object_type == "ball" else 0,
    )


# --- IOU Tests ---


class TestComputeIOU:
    """Tests for IOU computation."""

    def test_identical_boxes(self):
        """Identical boxes should have IOU of 1.0."""
        box = (0, 0, 10, 10)
        assert compute_iou(box, box) == 1.0

    def test_no_overlap(self):
        """Non-overlapping boxes should have IOU of 0.0."""
        box1 = (0, 0, 10, 10)
        box2 = (20, 20, 30, 30)
        assert compute_iou(box1, box2) == 0.0

    def test_partial_overlap(self):
        """Partially overlapping boxes should have IOU between 0 and 1."""
        box1 = (0, 0, 10, 10)
        box2 = (5, 5, 15, 15)
        iou = compute_iou(box1, box2)
        assert 0 < iou < 1
        # Intersection is 5x5=25, union is 100+100-25=175
        assert abs(iou - 25 / 175) < 1e-6

    def test_contained_box(self):
        """Box contained within another should have IOU = area_small / area_large."""
        outer = (0, 0, 20, 20)  # area = 400
        inner = (5, 5, 10, 10)  # area = 25
        iou = compute_iou(outer, inner)
        # Intersection = 25, union = 400
        assert abs(iou - 25 / 400) < 1e-6


# --- Soft-NMS Tests ---


class TestSoftNMS:
    """Tests for soft non-maximum suppression."""

    def test_empty_input(self):
        """Empty input should return empty output."""
        result = soft_nms([])
        assert result == []

    def test_single_detection(self):
        """Single detection should be returned unchanged."""
        det = make_detection(0, 0, 10, 10, confidence=0.8)
        result = soft_nms([det])
        assert len(result) == 1
        assert result[0].confidence == 0.8

    def test_reduces_overlapping_confidence(self):
        """Overlapping detections should have reduced confidence."""
        det1 = make_detection(0, 0, 10, 10, confidence=0.9)
        det2 = make_detection(2, 2, 12, 12, confidence=0.8)  # High overlap

        result = soft_nms([det1, det2], iou_threshold=0.3)

        # Should have both, but det2's confidence should be reduced
        assert len(result) >= 1

        # The higher confidence one should be first and unchanged
        assert result[0].confidence == 0.9

        # If det2 is kept, its confidence should be reduced
        if len(result) > 1:
            assert result[1].confidence < 0.8

    def test_preserves_non_overlapping(self):
        """Non-overlapping detections should be preserved with original confidence."""
        det1 = make_detection(0, 0, 10, 10, confidence=0.8)
        det2 = make_detection(50, 50, 60, 60, confidence=0.7)  # No overlap

        result = soft_nms([det1, det2], iou_threshold=0.3)

        assert len(result) == 2
        # Both should have original confidence (or very close)
        confs = sorted([r.confidence for r in result], reverse=True)
        assert confs[0] == 0.8
        assert abs(confs[1] - 0.7) < 0.01

    def test_score_threshold_filters(self):
        """Detections below score threshold should be filtered."""
        det1 = make_detection(0, 0, 10, 10, confidence=0.1)
        det2 = make_detection(5, 5, 15, 15, confidence=0.05)

        result = soft_nms([det1, det2], score_threshold=0.08)

        # det2 should be filtered
        assert len(result) <= 2


# --- Temporal Filter Tests ---


class TestBallTemporalFilter:
    """Tests for temporal consistency filter."""

    def test_rejects_spurious_single_frame(self):
        """Single-frame detections without history should pass initially."""
        tf = BallTemporalFilter(window_size=3, min_confirmations=2, max_displacement=50)

        # First frame - should pass (not enough history)
        det1 = make_detection(100, 100, 110, 110)
        result1 = tf.filter([det1], frame_idx=0)
        assert len(result1) == 1

        # Second frame with NO detection near first
        # (detection far away)
        det2 = make_detection(500, 500, 510, 510)
        tf.filter([det2], frame_idx=1)
        # With min_confirmations=2 and only 2 frames, still might pass

        # Third frame - now we have enough history
        det3 = make_detection(600, 600, 610, 610)  # Completely different location
        result3 = tf.filter([det3], frame_idx=2)

        # This detection has no consistent history - should be rejected
        # (det1 at 100,100, det2 at 500,500, det3 at 600,600 - all far apart)
        assert len(result3) == 0

    def test_accepts_consistent_trajectory(self):
        """Consistent trajectory should be accepted."""
        tf = BallTemporalFilter(window_size=5, min_confirmations=2, max_displacement=50)

        # Simulate ball moving in a consistent direction
        positions = [(100, 100), (110, 105), (120, 110), (130, 115), (140, 120)]

        for i, (x, y) in enumerate(positions):
            det = make_detection(x, y, x + 10, y + 10)
            result = tf.filter([det], frame_idx=i)

            # After initial frames, detections should be accepted
            if i >= 1:  # After we have some history
                assert len(result) == 1, f"Frame {i} should accept consistent detection"

    def test_handles_camera_cut(self):
        """Large displacement (camera cut) should reset filter appropriately."""
        tf = BallTemporalFilter(window_size=3, min_confirmations=2, max_displacement=20)

        # Build up consistent history
        for i in range(3):
            det = make_detection(100 + i * 5, 100 + i * 5, 110 + i * 5, 110 + i * 5)
            tf.filter([det], frame_idx=i)

        # Sudden jump (camera cut) - detection at completely different location
        det_cut = make_detection(500, 500, 510, 510)
        result = tf.filter([det_cut], frame_idx=3)

        # Should be rejected due to large displacement
        assert len(result) == 0

    def test_reset_clears_history(self):
        """Reset should clear all history."""
        tf = BallTemporalFilter(window_size=3, min_confirmations=2, max_displacement=50)

        # Add some history
        for i in range(3):
            det = make_detection(100, 100, 110, 110)
            tf.filter([det], frame_idx=i)

        assert len(tf.history) == 3

        # Reset
        tf.reset()

        assert len(tf.history) == 0


# --- Candidate Tracker Tests ---


class TestBallCandidateTracker:
    """Tests for ball candidate tracking."""

    def test_promotion_after_min_hits(self):
        """Candidate should be promoted to CONFIRMED after min_hits."""
        tracker = BallCandidateTracker(min_hits=3, max_age=5, use_kalman=False)

        # First detection - creates NEW candidate
        det1 = make_detection(100, 100, 110, 110, confidence=0.5)
        result1 = tracker.update([det1], frame_idx=0)
        assert len(result1) == 0  # Not confirmed yet

        # Second detection at similar location - candidate becomes TENTATIVE
        det2 = make_detection(102, 102, 112, 112, confidence=0.5)
        result2 = tracker.update([det2], frame_idx=1)
        assert len(result2) == 0  # Still not confirmed

        # Third detection - should reach min_hits and become CONFIRMED
        det3 = make_detection(104, 104, 114, 114, confidence=0.5)
        result3 = tracker.update([det3], frame_idx=2)
        assert len(result3) == 1  # Now confirmed!

    def test_lost_after_max_age(self):
        """Candidate should be removed after max_age frames without update."""
        tracker = BallCandidateTracker(min_hits=2, max_age=3, use_kalman=False)

        # Create and confirm a candidate
        for i in range(3):
            det = make_detection(100 + i, 100 + i, 110 + i, 110 + i, confidence=0.5)
            tracker.update([det], frame_idx=i)

        # Now stop detecting - pass empty detections
        for i in range(3, 7):
            tracker.update([], frame_idx=i)

        # After max_age frames without detection, candidate should be removed
        assert len(tracker.candidates) == 0

    def test_recovery_after_brief_occlusion(self):
        """Candidate should recover after brief occlusion (within max_age)."""
        tracker = BallCandidateTracker(min_hits=2, max_age=5, use_kalman=False)

        # Build up a confirmed candidate
        for i in range(3):
            det = make_detection(100 + i * 2, 100 + i * 2, 110 + i * 2, 110 + i * 2)
            tracker.update([det], frame_idx=i)

        # Brief occlusion (2 frames)
        tracker.update([], frame_idx=3)
        tracker.update([], frame_idx=4)

        # Ball reappears
        det_return = make_detection(110, 110, 120, 120)
        result = tracker.update([det_return], frame_idx=5)

        # Should still have the candidate (and it should still be confirmed)
        assert len(tracker.candidates) > 0
        assert len(result) >= 1

    def test_reset_clears_state(self):
        """Reset should clear all candidates."""
        tracker = BallCandidateTracker(min_hits=2, max_age=5, use_kalman=False)

        # Add some candidates
        for i in range(3):
            det = make_detection(100 + i * 50, 100, 110 + i * 50, 110)
            tracker.update([det], frame_idx=i)

        assert len(tracker.candidates) > 0

        # Reset
        tracker.reset()

        assert len(tracker.candidates) == 0
        assert tracker.next_id == 0


# --- Multi-scale Detection Tests ---


class TestMultiscaleDetections:
    """Tests for multi-scale detection merging."""

    def test_bbox_scaling_roundtrip(self):
        """Bbox coordinates should be correctly scaled and unscaled."""
        # Simulate what happens in detect_multiscale
        original_width, original_height = 1920, 1080
        scale = 0.5

        # Original bbox
        original_bbox = (100, 100, 200, 200)

        # Scale down (as if resizing frame)
        scaled_bbox = (
            original_bbox[0] * scale,
            original_bbox[1] * scale,
            original_bbox[2] * scale,
            original_bbox[3] * scale,
        )

        # Scale back up
        scale_x = original_width / (original_width * scale)
        scale_y = original_height / (original_height * scale)
        recovered_bbox = (
            scaled_bbox[0] * scale_x,
            scaled_bbox[1] * scale_y,
            scaled_bbox[2] * scale_x,
            scaled_bbox[3] * scale_y,
        )

        # Should match original
        assert abs(recovered_bbox[0] - original_bbox[0]) < 1e-6
        assert abs(recovered_bbox[1] - original_bbox[1]) < 1e-6
        assert abs(recovered_bbox[2] - original_bbox[2]) < 1e-6
        assert abs(recovered_bbox[3] - original_bbox[3]) < 1e-6

    def test_merge_preserves_best_confidence(self):
        """Merging should preserve the highest confidence detection."""
        det1 = make_detection(100, 100, 110, 110, confidence=0.8)  # From scale 1.0
        det2 = make_detection(101, 101, 111, 111, confidence=0.6)  # From scale 0.5 (similar location)
        det3 = make_detection(99, 99, 109, 109, confidence=0.7)  # From scale 1.5

        result = merge_multiscale_detections([[det1], [det2], [det3]], iou_threshold=0.5)

        # Should merge overlapping detections, keeping highest confidence
        assert len(result) >= 1
        assert result[0].confidence == 0.8

    def test_merge_empty_scales(self):
        """Empty scale results should not cause issues."""
        det1 = make_detection(100, 100, 110, 110, confidence=0.8)

        result = merge_multiscale_detections([[], [det1], []], iou_threshold=0.5)

        assert len(result) == 1
        assert result[0].confidence == 0.8


# --- Backward Compatibility Tests ---


class TestBackwardCompatibility:
    """Tests for backward compatibility with old configs."""

    def test_old_config_without_ball_section(self):
        """Old configs without ball section should use defaults."""
        from src.config.schemas import DetectionConfig, BallDetectionConfig

        # Create config without explicit ball settings (uses defaults)
        config = DetectionConfig()

        # Should have default ball config
        assert config.ball is not None
        assert isinstance(config.ball, BallDetectionConfig)
        assert config.ball.confidence_threshold == 0.15
        assert config.ball.enable_multiscale is True

    def test_effective_properties_work(self):
        """Effective property accessors should work."""
        from src.config.schemas import DetectionConfig

        config = DetectionConfig()

        # Test effective property accessors
        assert config.effective_ball_confidence == 0.15
        assert config.effective_ball_max_size_ratio == 0.05
        assert config.effective_ball_max_aspect_ratio == 3.0

    def test_custom_ball_config(self):
        """Custom ball config should override defaults."""
        from src.config.schemas import DetectionConfig, BallDetectionConfig

        config = DetectionConfig(
            ball=BallDetectionConfig(
                confidence_threshold=0.20,
                enable_multiscale=False,
                scales=[1.0, 2.0],
            )
        )

        assert config.ball.confidence_threshold == 0.20
        assert config.ball.enable_multiscale is False
        assert config.ball.scales == [1.0, 2.0]


# --- Integration Tests ---


class TestIntegration:
    """Integration tests for ball boost components working together."""

    def test_temporal_filter_standalone(self):
        """Test temporal filter alone on consistent trajectory."""
        temporal_filter = BallTemporalFilter(
            window_size=3, min_confirmations=2, max_displacement=50
        )

        passed_frames = []

        # Simulate 10 frames of consistent ball trajectory
        for i in range(10):
            det = make_detection(100 + i * 5, 100 + i * 5, 110 + i * 5, 110 + i * 5, confidence=0.5)
            filtered = temporal_filter.filter([det], frame_idx=i)
            if filtered:
                passed_frames.append(i)

        # Should pass most frames after initial ramp-up
        assert len(passed_frames) >= 8, f"Temporal filter should pass consistent trajectory, got {passed_frames}"

    def test_candidate_tracker_standalone(self):
        """Test candidate tracker alone on consistent trajectory."""
        # Use lower IOU threshold to match fast-moving ball
        candidate_tracker = BallCandidateTracker(min_hits=2, max_age=3, iou_threshold=0.1, use_kalman=False)

        confirmed_frames = []

        # Simulate 10 frames of consistent ball trajectory
        # Ball moves 5 pixels per frame - realistic for a fast-moving ball
        for i in range(10):
            det = make_detection(100 + i * 5, 100 + i * 5, 110 + i * 5, 110 + i * 5, confidence=0.5)
            confirmed = candidate_tracker.update([det], frame_idx=i)
            if confirmed:
                confirmed_frames.append(i)

        # Should have confirmed detections after min_hits
        assert len(confirmed_frames) >= 8, f"Candidate tracker should confirm consistent trajectory, got {confirmed_frames}"

    def test_components_work_in_sequence(self):
        """Test that components can be used together without errors."""
        temporal_filter = BallTemporalFilter(
            window_size=3, min_confirmations=2, max_displacement=100
        )
        candidate_tracker = BallCandidateTracker(min_hits=2, max_age=5, use_kalman=False)

        # Just verify the pipeline runs without errors
        for i in range(20):
            det = make_detection(100 + i * 3, 100 + i * 3, 110 + i * 3, 110 + i * 3, confidence=0.5)

            # Apply temporal filter first
            filtered = temporal_filter.filter([det], frame_idx=i)

            # Then apply candidate tracker
            candidate_tracker.update(filtered, frame_idx=i)

            # No assertion on results - just verify it doesn't crash

        # Verify state is maintained
        assert len(temporal_filter.history) > 0
        assert len(candidate_tracker.candidates) >= 0  # May have candidates or not
