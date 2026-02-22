"""Tests for ByteTrack multi-object tracking."""

import pytest

from src.vision.track.bytetrack import ByteTracker, Track, iou
from src.vision.track.kalman import BBoxKalmanFilter, bbox_to_measurement


# -----------------------------------------------------------------------------
# IoU Tests
# -----------------------------------------------------------------------------

class TestIoU:
    """Tests for IoU (Intersection over Union) calculation."""

    def test_identical_boxes(self):
        """Test IoU of identical boxes is 1.0."""
        bbox = (100.0, 100.0, 200.0, 200.0)
        assert iou(bbox, bbox) == pytest.approx(1.0)

    def test_no_overlap(self):
        """Test IoU of non-overlapping boxes is 0.0."""
        bbox1 = (0.0, 0.0, 50.0, 50.0)
        bbox2 = (100.0, 100.0, 150.0, 150.0)
        assert iou(bbox1, bbox2) == pytest.approx(0.0)

    def test_partial_overlap(self):
        """Test IoU of partially overlapping boxes."""
        bbox1 = (0.0, 0.0, 100.0, 100.0)
        bbox2 = (50.0, 50.0, 150.0, 150.0)
        # Intersection: 50*50 = 2500
        # Union: 10000 + 10000 - 2500 = 17500
        # IoU: 2500/17500 = 1/7
        assert iou(bbox1, bbox2) == pytest.approx(1.0 / 7.0, rel=0.01)

    def test_containment(self):
        """Test IoU when one box contains another."""
        outer = (0.0, 0.0, 100.0, 100.0)
        inner = (25.0, 25.0, 75.0, 75.0)
        # Intersection: 50*50 = 2500
        # Union: 10000 (outer area)
        # IoU: 2500/10000 = 0.25
        assert iou(outer, inner) == pytest.approx(0.25)

    def test_adjacent_boxes(self):
        """Test IoU of adjacent (touching) boxes is 0.0."""
        bbox1 = (0.0, 0.0, 50.0, 50.0)
        bbox2 = (50.0, 0.0, 100.0, 50.0)
        assert iou(bbox1, bbox2) == pytest.approx(0.0)

    def test_very_small_overlap(self):
        """Test IoU with very small overlap."""
        bbox1 = (0.0, 0.0, 100.0, 100.0)
        bbox2 = (99.0, 99.0, 199.0, 199.0)
        # Intersection: 1*1 = 1
        # Union: 10000 + 10000 - 1 = 19999
        # IoU: 1/19999
        assert iou(bbox1, bbox2) == pytest.approx(1.0 / 19999.0, rel=0.01)

    def test_zero_area_box(self):
        """Test IoU with zero-area box."""
        bbox1 = (0.0, 0.0, 0.0, 0.0)
        bbox2 = (0.0, 0.0, 100.0, 100.0)
        # Zero union should return 0
        assert iou(bbox1, bbox2) == pytest.approx(0.0)

    def test_half_overlap(self):
        """Test IoU with 50% overlap."""
        bbox1 = (0.0, 0.0, 100.0, 100.0)
        bbox2 = (0.0, 50.0, 100.0, 150.0)
        # Intersection: 100*50 = 5000
        # Union: 10000 + 10000 - 5000 = 15000
        # IoU: 5000/15000 = 1/3
        assert iou(bbox1, bbox2) == pytest.approx(1.0 / 3.0, rel=0.01)


# -----------------------------------------------------------------------------
# Track Tests
# -----------------------------------------------------------------------------

class TestTrack:
    """Tests for the Track dataclass."""

    @pytest.fixture
    def sample_track(self):
        """Create a sample track for testing."""
        kf = BBoxKalmanFilter()
        bbox = (100.0, 100.0, 150.0, 200.0)
        kf.initiate(bbox_to_measurement(bbox))

        return Track(
            track_id=1,
            bbox=bbox,
            confidence=0.9,
            object_type="player",
            kf=kf,
            hits=1,
        )

    def test_track_creation(self, sample_track):
        """Test track is created with correct initial values."""
        assert sample_track.track_id == 1
        assert sample_track.bbox == (100.0, 100.0, 150.0, 200.0)
        assert sample_track.confidence == 0.9
        assert sample_track.object_type == "player"
        assert sample_track.age == 0
        assert sample_track.hits == 1
        assert sample_track.time_since_update == 0
        assert sample_track.state == "tentative"

    def test_track_predict(self, sample_track):
        """Test track prediction updates state correctly."""
        original_age = sample_track.age
        original_tsu = sample_track.time_since_update

        sample_track.predict()

        assert sample_track.age == original_age + 1
        assert sample_track.time_since_update == original_tsu + 1
        # Bbox should be updated (may be slightly different due to KF prediction)
        assert sample_track.bbox is not None

    def test_track_update(self, sample_track):
        """Test track update with new detection."""
        new_bbox = (105.0, 102.0, 155.0, 202.0)
        new_conf = 0.95

        sample_track.update(new_bbox, new_conf)

        assert sample_track.bbox == new_bbox
        assert sample_track.confidence == new_conf
        assert sample_track.hits == 2
        assert sample_track.time_since_update == 0

    def test_track_state_promotion(self, sample_track):
        """Test that track is promoted to confirmed after enough hits."""
        assert sample_track.state == "tentative"

        # Need 3 hits to be promoted (already has 1)
        for i in range(2):
            sample_track.update(sample_track.bbox, 0.9)

        assert sample_track.hits >= 3
        assert sample_track.state == "confirmed"

    def test_mark_missed_tentative(self, sample_track):
        """Test that tentative track is deleted when missed."""
        assert sample_track.state == "tentative"

        sample_track.mark_missed()

        assert sample_track.state == "deleted"

    def test_mark_missed_confirmed(self, sample_track):
        """Test that confirmed track is not deleted when missed."""
        # Promote to confirmed
        for _ in range(2):
            sample_track.update(sample_track.bbox, 0.9)
        assert sample_track.state == "confirmed"

        sample_track.mark_missed()

        # Confirmed tracks are NOT deleted on miss (they just increment time_since_update)
        assert sample_track.state == "confirmed"

    def test_multiple_predictions(self, sample_track):
        """Test multiple predictions accumulate age and time_since_update."""
        for i in range(5):
            sample_track.predict()

        assert sample_track.age == 5
        assert sample_track.time_since_update == 5


# -----------------------------------------------------------------------------
# ByteTracker Tests
# -----------------------------------------------------------------------------

class TestByteTracker:
    """Tests for the ByteTracker class."""

    @pytest.fixture
    def tracker(self):
        """Create a fresh tracker for testing."""
        return ByteTracker(
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.5,
            min_hits=3,
        )

    def test_tracker_initialization(self, tracker):
        """Test tracker is initialized correctly."""
        assert tracker.track_thresh == 0.5
        assert tracker.track_buffer == 30
        assert tracker.match_thresh == 0.5
        assert tracker.min_hits == 3
        assert len(tracker.tracks) == 0
        assert tracker.next_id == 1
        assert tracker.frame_count == 0

    def test_update_empty_input(self, tracker):
        """Test update with no detections."""
        result = tracker.update([])

        assert len(result) == 0
        assert tracker.frame_count == 1

    def test_update_single_detection(self, tracker):
        """Test update with a single high-confidence detection."""
        detection = {
            "bbox": (100.0, 100.0, 150.0, 200.0),
            "confidence": 0.9,
            "object_type": "player",
        }

        # First frame: track is created but not confirmed
        result = tracker.update([detection])
        assert len(result) == 0  # Not confirmed yet
        assert len(tracker.tracks) == 1

        # Next frames: track gets confirmed
        for _ in range(3):
            result = tracker.update([detection])

        # After enough hits, track should be confirmed
        assert len(result) == 1
        assert result[0].object_type == "player"

    def test_track_confirmation_threshold(self, tracker):
        """Test that tracks need min_hits to be confirmed."""
        detection = {
            "bbox": (100.0, 100.0, 150.0, 200.0),
            "confidence": 0.9,
            "object_type": "player",
        }

        # Track should not be confirmed until min_hits
        for i in range(tracker.min_hits - 1):
            result = tracker.update([detection])
            assert len(result) == 0, f"Track confirmed too early at frame {i}"

        # After min_hits, track should be confirmed
        result = tracker.update([detection])
        assert len(result) == 1

    def test_high_low_confidence_matching(self, tracker):
        """Test that high confidence detections are matched first."""
        # Create initial track with high confidence
        high_conf_det = {
            "bbox": (100.0, 100.0, 150.0, 200.0),
            "confidence": 0.9,
            "object_type": "player",
        }

        # Build up confirmed track
        for _ in range(4):
            tracker.update([high_conf_det])

        # Now send a low confidence detection at same location
        low_conf_det = {
            "bbox": (102.0, 102.0, 152.0, 202.0),
            "confidence": 0.3,  # Below track_thresh
            "object_type": "player",
        }

        result = tracker.update([low_conf_det])

        # Track should still be maintained (matched via low confidence path)
        assert len(result) == 1

    def test_object_type_separation(self, tracker):
        """Test that players and balls are tracked separately."""
        player_det = {
            "bbox": (100.0, 100.0, 150.0, 200.0),
            "confidence": 0.9,
            "object_type": "player",
        }
        ball_det = {
            "bbox": (100.0, 100.0, 120.0, 120.0),  # Same area as player
            "confidence": 0.9,
            "object_type": "ball",
        }

        # Create both tracks
        for _ in range(4):
            tracker.update([player_det, ball_det])

        # Should have separate tracks for player and ball
        result = tracker.update([player_det, ball_det])

        assert len(result) == 2
        object_types = {t.object_type for t in result}
        assert "player" in object_types
        assert "ball" in object_types

    def test_track_deletion_buffer(self, tracker):
        """Test that tracks are deleted after track_buffer frames without update."""
        detection = {
            "bbox": (100.0, 100.0, 150.0, 200.0),
            "confidence": 0.9,
            "object_type": "player",
        }

        # Build confirmed track
        for _ in range(4):
            tracker.update([detection])

        assert len(tracker.tracks) == 1

        # Send empty detections for track_buffer frames
        for _ in range(tracker.track_buffer):
            tracker.update([])

        # Track should be deleted
        assert len(tracker.tracks) == 0

    def test_below_threshold_detection_no_new_track(self, tracker):
        """Test that detections below track_thresh don't create new tracks."""
        low_conf_det = {
            "bbox": (100.0, 100.0, 150.0, 200.0),
            "confidence": 0.3,  # Below track_thresh of 0.5
            "object_type": "player",
        }

        tracker.update([low_conf_det])

        # Should not create a new track
        assert len(tracker.tracks) == 0

    def test_multiple_players_tracking(self, tracker):
        """Test tracking multiple players simultaneously."""
        detections = [
            {"bbox": (100.0, 100.0, 150.0, 200.0), "confidence": 0.9, "object_type": "player"},
            {"bbox": (300.0, 100.0, 350.0, 200.0), "confidence": 0.85, "object_type": "player"},
            {"bbox": (500.0, 100.0, 550.0, 200.0), "confidence": 0.8, "object_type": "player"},
        ]

        # Build up tracks
        for _ in range(4):
            tracker.update(detections)

        result = tracker.update(detections)

        assert len(result) == 3
        track_ids = {t.track_id for t in result}
        assert len(track_ids) == 3  # All unique IDs

    def test_track_id_increment(self, tracker):
        """Test that track IDs increment correctly."""
        detection = {
            "bbox": (100.0, 100.0, 150.0, 200.0),
            "confidence": 0.9,
            "object_type": "player",
        }

        tracker.update([detection])
        assert tracker.next_id == 2

        # Create another track at different location
        detection2 = {
            "bbox": (500.0, 500.0, 550.0, 600.0),
            "confidence": 0.9,
            "object_type": "player",
        }
        tracker.update([detection, detection2])
        assert tracker.next_id == 3

    def test_match_thresh_effect(self):
        """Test that match_thresh affects tracking behavior."""
        # Strict tracker
        strict_tracker = ByteTracker(match_thresh=0.9)

        # Lenient tracker
        lenient_tracker = ByteTracker(match_thresh=0.3)

        det1 = {"bbox": (100.0, 100.0, 200.0, 200.0), "confidence": 0.9, "object_type": "player"}
        det2 = {"bbox": (120.0, 120.0, 220.0, 220.0), "confidence": 0.9, "object_type": "player"}  # Shifted

        # Build tracks
        for _ in range(4):
            strict_tracker.update([det1])
            lenient_tracker.update([det1])

        # Send shifted detection
        strict_tracker.update([det2])
        lenient_result = lenient_tracker.update([det2])

        # Lenient should match, strict might not (depends on IoU)
        # The shifted box has ~44% IoU, so strict (0.9) won't match, lenient (0.3) will
        assert len(lenient_result) >= 1

    def test_frame_count_increment(self, tracker):
        """Test that frame count increments correctly."""
        assert tracker.frame_count == 0

        tracker.update([])
        assert tracker.frame_count == 1

        tracker.update([])
        assert tracker.frame_count == 2


class TestByteTrackerMatching:
    """Tests for the internal matching logic of ByteTracker."""

    def test_empty_tracks_empty_detections(self):
        """Test matching with both empty."""
        tracker = ByteTracker()

        unmatched_tracks, unmatched_dets = tracker._match_tracks_detections([], [])

        assert unmatched_tracks == []
        assert unmatched_dets == []

    def test_empty_tracks_with_detections(self):
        """Test matching with no tracks but some detections."""
        tracker = ByteTracker()

        detections = [
            {"bbox": (100.0, 100.0, 150.0, 200.0), "confidence": 0.9, "object_type": "player"},
        ]

        unmatched_tracks, unmatched_dets = tracker._match_tracks_detections([], detections)

        assert unmatched_tracks == []
        assert len(unmatched_dets) == 1

    def test_tracks_with_no_detections(self):
        """Test matching with tracks but no detections."""
        tracker = ByteTracker()

        # Create a track
        kf = BBoxKalmanFilter()
        bbox = (100.0, 100.0, 150.0, 200.0)
        kf.initiate(bbox_to_measurement(bbox))
        track = Track(track_id=1, bbox=bbox, confidence=0.9, object_type="player", kf=kf)

        unmatched_tracks, unmatched_dets = tracker._match_tracks_detections([track], [])

        assert len(unmatched_tracks) == 1
        assert unmatched_dets == []

    def test_perfect_match(self):
        """Test matching identical track and detection."""
        tracker = ByteTracker(match_thresh=0.5)

        bbox = (100.0, 100.0, 150.0, 200.0)
        kf = BBoxKalmanFilter()
        kf.initiate(bbox_to_measurement(bbox))
        track = Track(track_id=1, bbox=bbox, confidence=0.9, object_type="player", kf=kf)

        detection = {"bbox": bbox, "confidence": 0.9, "object_type": "player"}

        unmatched_tracks, unmatched_dets = tracker._match_tracks_detections([track], [detection])

        # Should match
        assert len(unmatched_tracks) == 0
        assert len(unmatched_dets) == 0
        # Track should be updated
        assert track.hits == 1  # Updated by match

    def test_type_mismatch_no_match(self):
        """Test that different object types don't match."""
        tracker = ByteTracker(match_thresh=0.5)

        bbox = (100.0, 100.0, 150.0, 200.0)
        kf = BBoxKalmanFilter()
        kf.initiate(bbox_to_measurement(bbox))
        track = Track(track_id=1, bbox=bbox, confidence=0.9, object_type="player", kf=kf)

        # Same bbox but different type
        detection = {"bbox": bbox, "confidence": 0.9, "object_type": "ball"}

        unmatched_tracks, unmatched_dets = tracker._match_tracks_detections([track], [detection])

        # Should not match due to type difference
        assert len(unmatched_tracks) == 1
        assert len(unmatched_dets) == 1
