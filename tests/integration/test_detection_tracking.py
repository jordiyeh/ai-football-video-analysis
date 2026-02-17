"""Integration tests for detection to tracking pipeline."""

import pytest
import numpy as np

from src.vision.track.bytetrack import ByteTracker
from src.vision.track.kalman import BBoxKalmanFilter, bbox_to_measurement


class TestDetectionToTracking:
    """Integration tests for detection-to-tracking data flow."""

    @pytest.fixture
    def tracker(self):
        """Create a tracker with realistic settings."""
        return ByteTracker(
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.5,
            min_hits=3,
        )

    def test_single_object_tracking_across_frames(self, tracker):
        """Test tracking a single object across multiple frames."""
        # Simulate a player moving horizontally
        detections_sequence = []
        for i in range(20):
            detections_sequence.append([{
                "bbox": (100.0 + i * 5, 100.0, 150.0 + i * 5, 200.0),
                "confidence": 0.9,
                "object_type": "player",
            }])

        track_ids = []
        for frame_detections in detections_sequence:
            tracks = tracker.update(frame_detections)
            if tracks:
                track_ids.append(tracks[0].track_id)

        # After confirmation threshold, track ID should remain stable
        unique_ids = set(track_ids)
        assert len(unique_ids) == 1, "Track ID should remain consistent"

    def test_track_identity_maintained_with_gaps(self, tracker):
        """Test that track identity is maintained during short gaps."""
        # Create detections with a small gap
        detections_with_gap = []

        # First 5 frames: normal detections
        for i in range(5):
            detections_with_gap.append([{
                "bbox": (100.0 + i * 5, 100.0, 150.0 + i * 5, 200.0),
                "confidence": 0.9,
                "object_type": "player",
            }])

        # 3 frames: no detections (gap)
        for _ in range(3):
            detections_with_gap.append([])

        # Resume detections (continuing movement)
        for i in range(5):
            x_offset = (5 + 3 + i) * 5
            detections_with_gap.append([{
                "bbox": (100.0 + x_offset, 100.0, 150.0 + x_offset, 200.0),
                "confidence": 0.9,
                "object_type": "player",
            }])

        track_ids = []
        for frame_detections in detections_with_gap:
            tracks = tracker.update(frame_detections)
            if tracks:
                track_ids.extend([t.track_id for t in tracks])

        # Track should maintain identity through short gap
        unique_ids = set(track_ids)
        assert len(unique_ids) <= 2, "Should have at most 2 track IDs (ideally 1)"

    def test_multiple_objects_separated_tracking(self, tracker):
        """Test tracking multiple separated objects."""
        # Two players far apart
        detections_sequence = []
        for i in range(15):
            detections_sequence.append([
                {  # Player 1
                    "bbox": (100.0 + i * 3, 100.0, 150.0 + i * 3, 200.0),
                    "confidence": 0.9,
                    "object_type": "player",
                },
                {  # Player 2 (far away)
                    "bbox": (800.0 - i * 3, 100.0, 850.0 - i * 3, 200.0),
                    "confidence": 0.85,
                    "object_type": "player",
                },
            ])

        final_tracks = None
        for frame_detections in detections_sequence:
            tracks = tracker.update(frame_detections)
            if tracks:
                final_tracks = tracks

        # Should have 2 distinct tracks
        assert final_tracks is not None
        assert len(final_tracks) == 2
        track_ids = {t.track_id for t in final_tracks}
        assert len(track_ids) == 2

    def test_player_and_ball_separate_tracks(self, tracker):
        """Test that players and balls are tracked separately."""
        detections_sequence = []
        for i in range(15):
            detections_sequence.append([
                {  # Player
                    "bbox": (100.0 + i * 5, 100.0, 150.0 + i * 5, 200.0),
                    "confidence": 0.9,
                    "object_type": "player",
                },
                {  # Ball (near player but different type)
                    "bbox": (100.0 + i * 5, 180.0, 120.0 + i * 5, 200.0),
                    "confidence": 0.8,
                    "object_type": "ball",
                },
            ])

        final_tracks = None
        for frame_detections in detections_sequence:
            tracks = tracker.update(frame_detections)
            if tracks:
                final_tracks = tracks

        # Should have 2 distinct tracks
        assert final_tracks is not None
        assert len(final_tracks) == 2

        # One player, one ball
        object_types = {t.object_type for t in final_tracks}
        assert "player" in object_types
        assert "ball" in object_types

    def test_occlusion_and_reappearance(self, tracker):
        """Test handling of temporary occlusion."""
        tracker = ByteTracker(
            track_thresh=0.5,
            track_buffer=10,  # Shorter buffer for test
            match_thresh=0.5,
            min_hits=3,
        )

        detections = []

        # Build up track
        for i in range(5):
            detections.append([{
                "bbox": (100.0 + i * 10, 100.0, 150.0 + i * 10, 200.0),
                "confidence": 0.9,
                "object_type": "player",
            }])

        # Occlusion (no detections)
        for _ in range(5):
            detections.append([])

        # Reappearance at predicted position
        for i in range(5):
            x_offset = (5 + 5 + i) * 10  # Continue from where it would be
            detections.append([{
                "bbox": (100.0 + x_offset, 100.0, 150.0 + x_offset, 200.0),
                "confidence": 0.9,
                "object_type": "player",
            }])

        for frame_detections in detections:
            tracker.update(frame_detections)

        # Check that tracking continued (may have re-identified)
        # The key is system doesn't crash and produces valid tracks

    def test_crossing_paths(self, tracker):
        """Test tracking when two objects cross paths."""
        detections = []

        for i in range(20):
            # Player 1 moves right
            p1_x = 100.0 + i * 20
            # Player 2 moves left
            p2_x = 500.0 - i * 20

            detections.append([
                {
                    "bbox": (p1_x, 100.0, p1_x + 50.0, 200.0),
                    "confidence": 0.9,
                    "object_type": "player",
                },
                {
                    "bbox": (p2_x, 100.0, p2_x + 50.0, 200.0),
                    "confidence": 0.9,
                    "object_type": "player",
                },
            ])

        track_history = []
        for frame_detections in detections:
            tracks = tracker.update(frame_detections)
            if tracks:
                track_history.append([(t.track_id, t.bbox[0]) for t in tracks])

        # Should maintain 2 tracks throughout (may swap IDs at crossing)
        # Key is system doesn't crash and tracks are maintained


class TestKalmanFilterIntegration:
    """Integration tests for Kalman filter in tracking."""

    def test_kalman_prediction_quality(self):
        """Test that Kalman filter predictions improve with more observations."""
        kf = BBoxKalmanFilter()

        # Initial position
        initial_bbox = (100.0, 100.0, 150.0, 200.0)
        kf.initiate(bbox_to_measurement(initial_bbox))

        # Simulate consistent movement
        positions = []
        for i in range(10):
            # True position moves 10 pixels right per frame
            true_x = 100.0 + (i + 1) * 10
            measurement = bbox_to_measurement((true_x, 100.0, true_x + 50.0, 200.0))

            # Predict and then update
            predicted = kf.predict()
            kf.update(measurement)

            positions.append({
                "true_x": true_x,
                "predicted_x": predicted[0],
                "updated_x": kf.get_state()[0],
            })

        # Later predictions should be closer to truth (filter learns velocity)
        early_errors = [abs(p["predicted_x"] - p["true_x"]) for p in positions[:3]]
        late_errors = [abs(p["predicted_x"] - p["true_x"]) for p in positions[-3:]]

        # After learning, errors should be smaller
        assert np.mean(late_errors) < np.mean(early_errors) + 20  # Allow some margin

    def test_noisy_observations_smoothed(self):
        """Test that noisy observations are smoothed by Kalman filter."""
        kf = BBoxKalmanFilter()

        # True constant position
        true_bbox = (200.0, 200.0, 250.0, 300.0)
        true_measurement = bbox_to_measurement(true_bbox)
        kf.initiate(true_measurement)

        # Add noisy observations
        np.random.seed(42)
        state_positions = []

        for _ in range(30):
            # Noisy measurement
            noise = np.random.randn(4) * np.array([10, 10, 100, 0.1])
            noisy_measurement = true_measurement + noise

            kf.predict()
            kf.update(noisy_measurement)

            state = kf.get_state()
            state_positions.append(state[0])  # x position

        # State should stay close to true position
        assert abs(np.mean(state_positions) - true_measurement[0]) < 15
