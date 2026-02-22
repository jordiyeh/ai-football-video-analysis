"""Tests for ball trajectory analysis and interpolation."""

import pytest

from src.events.ball_trajectory import BallTrajectory, BallTrajectoryPoint
from src.config.schemas import InterpolationConfig


class TestBallTrajectoryPoint:
    """Tests for BallTrajectoryPoint dataclass."""

    def test_default_interpolation_fields(self):
        """Test that default interpolation fields are set correctly."""
        point = BallTrajectoryPoint(
            frame_idx=0,
            timestamp=0.0,
            position=(100.0, 100.0),
            velocity=None,
            speed=None,
            confidence=0.9,
        )
        assert point.interpolated is False
        assert point.interpolation_source is None

    def test_interpolated_point(self):
        """Test creating an interpolated point."""
        point = BallTrajectoryPoint(
            frame_idx=5,
            timestamp=5 / 30.0,
            position=(150.0, 150.0),
            velocity=None,
            speed=None,
            confidence=0.5,
            interpolated=True,
            interpolation_source="linear",
        )
        assert point.interpolated is True
        assert point.interpolation_source == "linear"


class TestLinearInterpolation:
    """Tests for linear interpolation (short gaps)."""

    def test_linear_interpolation_short_gap(self):
        """Test linear interpolation for gaps <= physics_threshold."""
        trajectory = BallTrajectory()
        trajectory.points = [
            BallTrajectoryPoint(
                frame_idx=0, timestamp=0.0, position=(100.0, 100.0),
                velocity=(10.0, 0.0), speed=10.0, confidence=0.9,
            ),
            BallTrajectoryPoint(
                frame_idx=5, timestamp=5 / 30.0, position=(150.0, 100.0),
                velocity=(10.0, 0.0), speed=10.0, confidence=0.9,
            ),
        ]

        config = InterpolationConfig(physics_threshold=10)
        result = trajectory.interpolate_gaps(fps=30.0, config=config)

        # Should have 6 points (2 original + 4 interpolated)
        assert len(result.points) == 6

        # Check interpolated points
        for i, point in enumerate(result.points):
            if i == 0 or i == 5:
                assert point.interpolated is False
            else:
                assert point.interpolated is True
                assert point.interpolation_source == "linear"

        # Check positions are linearly interpolated
        assert result.points[1].position[0] == pytest.approx(110.0, rel=1e-3)
        assert result.points[2].position[0] == pytest.approx(120.0, rel=1e-3)
        assert result.points[3].position[0] == pytest.approx(130.0, rel=1e-3)
        assert result.points[4].position[0] == pytest.approx(140.0, rel=1e-3)

    def test_linear_interpolation_preserves_y(self):
        """Test that linear interpolation works for both x and y."""
        trajectory = BallTrajectory()
        trajectory.points = [
            BallTrajectoryPoint(
                frame_idx=0, timestamp=0.0, position=(0.0, 0.0),
                velocity=(10.0, 20.0), speed=22.36, confidence=0.9,
            ),
            BallTrajectoryPoint(
                frame_idx=4, timestamp=4 / 30.0, position=(40.0, 80.0),
                velocity=(10.0, 20.0), speed=22.36, confidence=0.9,
            ),
        ]

        config = InterpolationConfig(physics_threshold=10)
        result = trajectory.interpolate_gaps(fps=30.0, config=config)

        # Check y positions
        assert result.points[1].position[1] == pytest.approx(20.0, rel=1e-3)
        assert result.points[2].position[1] == pytest.approx(40.0, rel=1e-3)
        assert result.points[3].position[1] == pytest.approx(60.0, rel=1e-3)


class TestPhysicsInterpolation:
    """Tests for physics-based interpolation (longer gaps)."""

    def test_physics_interpolation_medium_gap(self):
        """Test physics interpolation for gaps > physics_threshold."""
        trajectory = BallTrajectory()
        trajectory.points = [
            BallTrajectoryPoint(
                frame_idx=0, timestamp=0.0, position=(100.0, 100.0),
                velocity=(10.0, 0.0), speed=10.0, confidence=0.9,
            ),
            BallTrajectoryPoint(
                frame_idx=30, timestamp=1.0, position=(400.0, 100.0),
                velocity=(10.0, 0.0), speed=10.0, confidence=0.9,
            ),
        ]

        config = InterpolationConfig(
            physics_threshold=10,
            use_bidirectional=True,
        )
        result = trajectory.interpolate_gaps(fps=30.0, config=config)

        # Should have 31 points
        assert len(result.points) == 31

        # Check that middle points use physics blending
        middle_point = result.points[15]
        assert middle_point.interpolated is True
        assert middle_point.interpolation_source == "physics_blended"

    def test_physics_forward_only(self):
        """Test physics interpolation without bidirectional blending."""
        trajectory = BallTrajectory()
        trajectory.points = [
            BallTrajectoryPoint(
                frame_idx=0, timestamp=0.0, position=(100.0, 100.0),
                velocity=(5.0, 0.0), speed=5.0, confidence=0.9,
            ),
            BallTrajectoryPoint(
                frame_idx=20, timestamp=20 / 30.0, position=(200.0, 100.0),
                velocity=(5.0, 0.0), speed=5.0, confidence=0.9,
            ),
        ]

        config = InterpolationConfig(
            physics_threshold=10,
            use_bidirectional=False,
        )
        result = trajectory.interpolate_gaps(fps=30.0, config=config)

        # Check interpolation source
        middle_point = result.points[10]
        assert middle_point.interpolation_source == "physics_forward"

    def test_physics_with_no_velocity(self):
        """Test physics interpolation when velocity is None."""
        trajectory = BallTrajectory()
        trajectory.points = [
            BallTrajectoryPoint(
                frame_idx=0, timestamp=0.0, position=(100.0, 100.0),
                velocity=None, speed=None, confidence=0.9,
            ),
            BallTrajectoryPoint(
                frame_idx=20, timestamp=20 / 30.0, position=(200.0, 100.0),
                velocity=None, speed=None, confidence=0.9,
            ),
        ]

        config = InterpolationConfig(physics_threshold=10)
        result = trajectory.interpolate_gaps(fps=30.0, config=config)

        # Should still interpolate (Kalman filter handles None velocity)
        assert len(result.points) == 21
        assert result.points[10].interpolated is True


class TestBidirectionalBlending:
    """Tests for bidirectional blending with smoothstep."""

    def test_smoothstep_weights(self):
        """Test that smoothstep blending produces expected weights."""
        # Smoothstep: 3t² - 2t³
        # At t=0: weight=0 (all forward)
        # At t=0.5: weight=0.5 (equal blend)
        # At t=1: weight=1 (all backward)
        def smoothstep(t):
            return 3 * t * t - 2 * t * t * t

        assert smoothstep(0) == pytest.approx(0.0)
        assert smoothstep(0.5) == pytest.approx(0.5)
        assert smoothstep(1.0) == pytest.approx(1.0)

        # Smoothstep is symmetric around 0.5
        assert smoothstep(0.25) == pytest.approx(1 - smoothstep(0.75), abs=1e-6)

    def test_blending_respects_endpoints(self):
        """Test that blended trajectory is reasonable near endpoints."""
        trajectory = BallTrajectory()
        trajectory.points = [
            BallTrajectoryPoint(
                frame_idx=0, timestamp=0.0, position=(0.0, 0.0),
                velocity=(10.0, 0.0), speed=10.0, confidence=0.9,
            ),
            BallTrajectoryPoint(
                frame_idx=100, timestamp=100 / 30.0, position=(1000.0, 0.0),
                velocity=(10.0, 0.0), speed=10.0, confidence=0.9,
            ),
        ]

        config = InterpolationConfig(
            physics_threshold=10,
            use_bidirectional=True,
        )
        result = trajectory.interpolate_gaps(fps=30.0, config=config)

        # Near start, should be close to forward prediction starting from (0,0)
        early_point = result.points[5]
        # Should be moving right from origin
        assert early_point.position[0] > 0

        # Near end, should be approaching target
        late_point = result.points[95]
        # Should be close to end position
        assert late_point.position[0] < 1000


class TestConfidenceDecay:
    """Tests for confidence decay during interpolation."""

    def test_confidence_decays_with_distance(self):
        """Test that confidence decreases as distance from endpoints increases."""
        trajectory = BallTrajectory()
        trajectory.points = [
            BallTrajectoryPoint(
                frame_idx=0, timestamp=0.0, position=(0.0, 0.0),
                velocity=(10.0, 0.0), speed=10.0, confidence=0.9,
            ),
            BallTrajectoryPoint(
                frame_idx=60, timestamp=2.0, position=(600.0, 0.0),
                velocity=(10.0, 0.0), speed=10.0, confidence=0.9,
            ),
        ]

        config = InterpolationConfig(
            physics_threshold=10,
            confidence_decay_rate=0.97,
            min_confidence=0.1,
        )
        result = trajectory.interpolate_gaps(fps=30.0, config=config)

        # Confidence should be highest near endpoints, lowest in middle
        edge_conf = result.points[1].confidence
        middle_conf = result.points[30].confidence

        assert middle_conf < edge_conf
        assert middle_conf >= config.min_confidence

    def test_confidence_respects_floor(self):
        """Test that confidence never goes below min_confidence."""
        trajectory = BallTrajectory()
        trajectory.points = [
            BallTrajectoryPoint(
                frame_idx=0, timestamp=0.0, position=(0.0, 0.0),
                velocity=(10.0, 0.0), speed=10.0, confidence=0.9,
            ),
            BallTrajectoryPoint(
                frame_idx=300, timestamp=10.0, position=(3000.0, 0.0),
                velocity=(10.0, 0.0), speed=10.0, confidence=0.9,
            ),
        ]

        config = InterpolationConfig(
            physics_threshold=10,
            confidence_decay_rate=0.9,  # Aggressive decay
            min_confidence=0.15,
        )
        result = trajectory.interpolate_gaps(fps=30.0, config=config)

        # All points should have confidence >= min_confidence
        for point in result.points:
            assert point.confidence >= config.min_confidence

    def test_confidence_decay_formula(self):
        """Test the specific confidence decay formula."""
        config = InterpolationConfig(
            confidence_decay_rate=0.97,
            min_confidence=0.1,
        )

        # At 1 second distance (30 frames at 30fps), decay should be 0.97^30
        expected_decay_1s = 0.97 ** 30
        assert expected_decay_1s == pytest.approx(0.401, rel=0.01)

        # Create trajectory with known values
        trajectory = BallTrajectory()
        trajectory.points = [
            BallTrajectoryPoint(
                frame_idx=0, timestamp=0.0, position=(0.0, 0.0),
                velocity=(10.0, 0.0), speed=10.0, confidence=1.0,
            ),
            BallTrajectoryPoint(
                frame_idx=60, timestamp=2.0, position=(600.0, 0.0),
                velocity=(10.0, 0.0), speed=10.0, confidence=1.0,
            ),
        ]

        result = trajectory.interpolate_gaps(fps=30.0, config=config)

        # Point at frame 30 is 1 second from both endpoints
        middle_point = result.points[30]
        # Confidence = 1.0 * 0.97^30 ≈ 0.401
        assert middle_point.confidence == pytest.approx(expected_decay_1s, rel=0.01)


class TestEdgeCases:
    """Tests for edge cases in interpolation."""

    def test_no_gap(self):
        """Test that consecutive frames don't create interpolated points."""
        trajectory = BallTrajectory()
        trajectory.points = [
            BallTrajectoryPoint(
                frame_idx=0, timestamp=0.0, position=(0.0, 0.0),
                velocity=(10.0, 0.0), speed=10.0, confidence=0.9,
            ),
            BallTrajectoryPoint(
                frame_idx=1, timestamp=1 / 30.0, position=(10.0, 0.0),
                velocity=(10.0, 0.0), speed=10.0, confidence=0.9,
            ),
        ]

        result = trajectory.interpolate_gaps(fps=30.0)

        assert len(result.points) == 2
        assert result.points[0].interpolated is False
        assert result.points[1].interpolated is False

    def test_gap_exceeds_max(self):
        """Test that gaps exceeding max_gap are not interpolated."""
        trajectory = BallTrajectory()
        trajectory.points = [
            BallTrajectoryPoint(
                frame_idx=0, timestamp=0.0, position=(0.0, 0.0),
                velocity=(10.0, 0.0), speed=10.0, confidence=0.9,
            ),
            BallTrajectoryPoint(
                frame_idx=500, timestamp=500 / 30.0, position=(5000.0, 0.0),
                velocity=(10.0, 0.0), speed=10.0, confidence=0.9,
            ),
        ]

        config = InterpolationConfig(max_gap=300)
        result = trajectory.interpolate_gaps(fps=30.0, config=config)

        # Only the two original points
        assert len(result.points) == 2

    def test_single_point(self):
        """Test interpolation with single point returns unchanged."""
        trajectory = BallTrajectory()
        trajectory.points = [
            BallTrajectoryPoint(
                frame_idx=0, timestamp=0.0, position=(0.0, 0.0),
                velocity=None, speed=None, confidence=0.9,
            ),
        ]

        result = trajectory.interpolate_gaps(fps=30.0)

        assert len(result.points) == 1
        assert result.points[0].interpolated is False

    def test_empty_trajectory(self):
        """Test interpolation with no points."""
        trajectory = BallTrajectory()

        result = trajectory.interpolate_gaps(fps=30.0)

        assert len(result.points) == 0

    def test_unsorted_input(self):
        """Test that unsorted input is handled correctly."""
        trajectory = BallTrajectory()
        trajectory.points = [
            BallTrajectoryPoint(
                frame_idx=10, timestamp=10 / 30.0, position=(100.0, 0.0),
                velocity=(10.0, 0.0), speed=10.0, confidence=0.9,
            ),
            BallTrajectoryPoint(
                frame_idx=0, timestamp=0.0, position=(0.0, 0.0),
                velocity=(10.0, 0.0), speed=10.0, confidence=0.9,
            ),
            BallTrajectoryPoint(
                frame_idx=5, timestamp=5 / 30.0, position=(50.0, 0.0),
                velocity=(10.0, 0.0), speed=10.0, confidence=0.9,
            ),
        ]

        config = InterpolationConfig(physics_threshold=10)
        result = trajectory.interpolate_gaps(fps=30.0, config=config)

        # Should be sorted by frame_idx
        frame_indices = [p.frame_idx for p in result.points]
        assert frame_indices == sorted(frame_indices)


class TestKalmanFilter:
    """Tests for the Kalman filter implementation."""

    def test_kalman_filter_initialization(self):
        """Test Kalman filter initializes correctly."""
        from src.events.kalman_filter import BallKalmanFilter

        kf = BallKalmanFilter()
        kf.initialize(position=(100.0, 200.0), velocity=(5.0, -3.0))

        state = kf.get_state()
        assert state["position"] == pytest.approx((100.0, 200.0))
        assert state["velocity"] == pytest.approx((5.0, -3.0))
        assert state["acceleration"] == pytest.approx((0.0, 0.0))

    def test_kalman_filter_prediction(self):
        """Test Kalman filter prediction step."""
        from src.events.kalman_filter import BallKalmanFilter

        kf = BallKalmanFilter(acceleration_decay=1.0)  # No decay for simple test
        kf.initialize(position=(0.0, 0.0), velocity=(10.0, 5.0))

        # After 1 frame: x = 0 + 10*1 = 10, y = 0 + 5*1 = 5
        pos = kf.predict(dt=1.0)
        assert pos[0] == pytest.approx(10.0, abs=0.5)
        assert pos[1] == pytest.approx(5.0, abs=0.5)

    def test_kalman_filter_acceleration_decay(self):
        """Test that acceleration decays over time."""
        from src.events.kalman_filter import BallKalmanFilter

        kf = BallKalmanFilter(acceleration_decay=0.9)
        kf.initialize(
            position=(0.0, 0.0),
            velocity=(0.0, 0.0),
            acceleration=(10.0, 10.0),
        )

        state_before = kf.get_state()
        kf.predict(dt=1.0)
        state_after = kf.get_state()

        # Acceleration should have decayed
        assert state_after["acceleration"][0] < state_before["acceleration"][0]
        assert state_after["acceleration"][0] == pytest.approx(9.0, rel=0.01)

    def test_kalman_filter_copy(self):
        """Test that filter copy creates independent copy."""
        from src.events.kalman_filter import BallKalmanFilter

        kf1 = BallKalmanFilter()
        kf1.initialize(position=(100.0, 100.0), velocity=(5.0, 5.0))

        kf2 = kf1.copy()

        # Modify kf1
        kf1.predict(dt=1.0)

        # kf2 should be unchanged
        state2 = kf2.get_state()
        assert state2["position"] == pytest.approx((100.0, 100.0))


class TestInterpolationConfig:
    """Tests for InterpolationConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = InterpolationConfig()

        assert config.max_gap == 300
        assert config.physics_threshold == 10
        assert config.confidence_decay_rate == 0.97
        assert config.min_confidence == 0.1
        assert config.use_bidirectional is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = InterpolationConfig(
            max_gap=500,
            physics_threshold=20,
            confidence_decay_rate=0.95,
        )

        assert config.max_gap == 500
        assert config.physics_threshold == 20
        assert config.confidence_decay_rate == 0.95


class TestBackwardCompatibility:
    """Tests for backward compatibility with old API."""

    def test_legacy_max_gap_parameter(self):
        """Test that legacy max_gap_frames parameter still works."""
        trajectory = BallTrajectory()
        trajectory.points = [
            BallTrajectoryPoint(
                frame_idx=0, timestamp=0.0, position=(0.0, 0.0),
                velocity=(10.0, 0.0), speed=10.0, confidence=0.9,
            ),
            BallTrajectoryPoint(
                frame_idx=20, timestamp=20 / 30.0, position=(200.0, 0.0),
                velocity=(10.0, 0.0), speed=10.0, confidence=0.9,
            ),
        ]

        # Call with legacy parameter only
        result = trajectory.interpolate_gaps(max_gap_frames=30, fps=30.0)

        assert len(result.points) == 21

    def test_config_overrides_legacy(self):
        """Test that config.max_gap takes precedence when larger."""
        trajectory = BallTrajectory()
        trajectory.points = [
            BallTrajectoryPoint(
                frame_idx=0, timestamp=0.0, position=(0.0, 0.0),
                velocity=(10.0, 0.0), speed=10.0, confidence=0.9,
            ),
            BallTrajectoryPoint(
                frame_idx=50, timestamp=50 / 30.0, position=(500.0, 0.0),
                velocity=(10.0, 0.0), speed=10.0, confidence=0.9,
            ),
        ]

        # Legacy says 30, but config says 100
        config = InterpolationConfig(max_gap=100)
        result = trajectory.interpolate_gaps(max_gap_frames=30, fps=30.0, config=config)

        # Should interpolate because config.max_gap (100) > gap (50)
        assert len(result.points) == 51
