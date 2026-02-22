"""Tests for Kalman filter for bounding box tracking."""

import pytest
import numpy as np

from src.vision.track.kalman import (
    BBoxKalmanFilter,
    bbox_to_measurement,
    measurement_to_bbox,
)


# -----------------------------------------------------------------------------
# bbox_to_measurement Tests
# -----------------------------------------------------------------------------

class TestBboxToMeasurement:
    """Tests for bbox_to_measurement conversion."""

    def test_standard_bbox(self):
        """Test conversion of a standard bounding box."""
        bbox = (100.0, 100.0, 200.0, 200.0)
        measurement = bbox_to_measurement(bbox)

        # Center should be (150, 150)
        assert measurement[0] == pytest.approx(150.0)
        assert measurement[1] == pytest.approx(150.0)
        # Area should be 100*100 = 10000
        assert measurement[2] == pytest.approx(10000.0)
        # Aspect ratio (width/height) should be 1.0
        assert measurement[3] == pytest.approx(1.0)

    def test_square_bbox(self):
        """Test conversion of a square bounding box."""
        bbox = (0.0, 0.0, 50.0, 50.0)
        measurement = bbox_to_measurement(bbox)

        assert measurement[0] == pytest.approx(25.0)  # x_center
        assert measurement[1] == pytest.approx(25.0)  # y_center
        assert measurement[2] == pytest.approx(2500.0)  # area
        assert measurement[3] == pytest.approx(1.0)  # aspect ratio

    def test_wide_bbox(self):
        """Test conversion of a wide (landscape) bounding box."""
        bbox = (0.0, 0.0, 200.0, 100.0)
        measurement = bbox_to_measurement(bbox)

        assert measurement[0] == pytest.approx(100.0)  # x_center
        assert measurement[1] == pytest.approx(50.0)  # y_center
        assert measurement[2] == pytest.approx(20000.0)  # area (200*100)
        assert measurement[3] == pytest.approx(2.0)  # width/height = 200/100

    def test_tall_bbox(self):
        """Test conversion of a tall (portrait) bounding box."""
        bbox = (0.0, 0.0, 100.0, 200.0)
        measurement = bbox_to_measurement(bbox)

        assert measurement[0] == pytest.approx(50.0)  # x_center
        assert measurement[1] == pytest.approx(100.0)  # y_center
        assert measurement[2] == pytest.approx(20000.0)  # area (100*200)
        assert measurement[3] == pytest.approx(0.5)  # width/height = 100/200

    def test_offset_bbox(self):
        """Test conversion of a bbox not at origin."""
        bbox = (500.0, 300.0, 600.0, 400.0)
        measurement = bbox_to_measurement(bbox)

        assert measurement[0] == pytest.approx(550.0)  # x_center
        assert measurement[1] == pytest.approx(350.0)  # y_center
        assert measurement[2] == pytest.approx(10000.0)  # area (100*100)
        assert measurement[3] == pytest.approx(1.0)  # aspect ratio

    def test_zero_height_bbox(self):
        """Test conversion with zero height (edge case)."""
        bbox = (0.0, 50.0, 100.0, 50.0)  # height = 0
        measurement = bbox_to_measurement(bbox)

        # Should default to aspect_ratio of 1.0 to avoid division by zero
        assert measurement[3] == pytest.approx(1.0)

    def test_returns_numpy_array(self):
        """Test that result is a numpy array."""
        bbox = (0.0, 0.0, 100.0, 100.0)
        measurement = bbox_to_measurement(bbox)

        assert isinstance(measurement, np.ndarray)
        assert measurement.shape == (4,)


# -----------------------------------------------------------------------------
# measurement_to_bbox Tests
# -----------------------------------------------------------------------------

class TestMeasurementToBbox:
    """Tests for measurement_to_bbox conversion."""

    def test_standard_measurement(self):
        """Test conversion of a standard measurement."""
        measurement = np.array([150.0, 150.0, 10000.0, 1.0])
        bbox = measurement_to_bbox(measurement)

        # Width = sqrt(10000*1) = 100, Height = 100/1 = 100
        # x1 = 150 - 50 = 100, y1 = 150 - 50 = 100
        # x2 = 150 + 50 = 200, y2 = 150 + 50 = 200
        assert bbox[0] == pytest.approx(100.0)
        assert bbox[1] == pytest.approx(100.0)
        assert bbox[2] == pytest.approx(200.0)
        assert bbox[3] == pytest.approx(200.0)

    def test_wide_measurement(self):
        """Test conversion with wide aspect ratio."""
        # 200x100 box centered at (100, 50)
        # area = 20000, aspect = 2.0
        measurement = np.array([100.0, 50.0, 20000.0, 2.0])
        bbox = measurement_to_bbox(measurement)

        # Height = sqrt(20000/2) = 100, Width = 2*100 = 200
        assert bbox[0] == pytest.approx(0.0)  # 100 - 200/2
        assert bbox[1] == pytest.approx(0.0)  # 50 - 100/2
        assert bbox[2] == pytest.approx(200.0)  # 100 + 200/2
        assert bbox[3] == pytest.approx(100.0)  # 50 + 100/2

    def test_round_trip_preservation(self):
        """Test that bbox -> measurement -> bbox preserves values."""
        original_bbox = (123.0, 456.0, 223.0, 656.0)

        measurement = bbox_to_measurement(original_bbox)
        recovered_bbox = measurement_to_bbox(measurement)

        assert recovered_bbox[0] == pytest.approx(original_bbox[0], rel=0.01)
        assert recovered_bbox[1] == pytest.approx(original_bbox[1], rel=0.01)
        assert recovered_bbox[2] == pytest.approx(original_bbox[2], rel=0.01)
        assert recovered_bbox[3] == pytest.approx(original_bbox[3], rel=0.01)

    def test_round_trip_various_sizes(self):
        """Test round-trip for various bbox sizes."""
        test_bboxes = [
            (0.0, 0.0, 50.0, 50.0),
            (100.0, 100.0, 300.0, 200.0),
            (500.0, 400.0, 550.0, 600.0),
            (0.0, 0.0, 1920.0, 1080.0),
        ]

        for original in test_bboxes:
            measurement = bbox_to_measurement(original)
            recovered = measurement_to_bbox(measurement)

            assert recovered[0] == pytest.approx(original[0], rel=0.01)
            assert recovered[1] == pytest.approx(original[1], rel=0.01)
            assert recovered[2] == pytest.approx(original[2], rel=0.01)
            assert recovered[3] == pytest.approx(original[3], rel=0.01)

    def test_negative_area_clamped(self):
        """Test that negative area is clamped to zero."""
        measurement = np.array([100.0, 100.0, -100.0, 1.0])
        bbox = measurement_to_bbox(measurement)

        # Area clamped to 0, so dimensions are 0
        assert bbox[0] == pytest.approx(100.0)
        assert bbox[1] == pytest.approx(100.0)
        assert bbox[2] == pytest.approx(100.0)
        assert bbox[3] == pytest.approx(100.0)

    def test_very_small_aspect_ratio_clamped(self):
        """Test that very small aspect ratio is clamped."""
        measurement = np.array([100.0, 100.0, 1000.0, 0.001])  # Very small aspect
        bbox = measurement_to_bbox(measurement)

        # Should not crash, aspect ratio clamped to 0.1 minimum
        assert all(np.isfinite(bbox))

    def test_returns_tuple_of_floats(self):
        """Test that result is a tuple of floats."""
        measurement = np.array([100.0, 100.0, 10000.0, 1.0])
        bbox = measurement_to_bbox(measurement)

        assert isinstance(bbox, tuple)
        assert len(bbox) == 4
        assert all(isinstance(v, float) for v in bbox)


# -----------------------------------------------------------------------------
# BBoxKalmanFilter Tests
# -----------------------------------------------------------------------------

class TestBBoxKalmanFilter:
    """Tests for the BBoxKalmanFilter class."""

    @pytest.fixture
    def filter(self):
        """Create a fresh Kalman filter for testing."""
        return BBoxKalmanFilter()

    def test_initialization(self, filter):
        """Test filter is initialized correctly."""
        assert filter.kf.dim_x == 8  # State dimension
        assert filter.kf.dim_z == 4  # Measurement dimension

    def test_initiate(self, filter):
        """Test initiate sets state correctly."""
        measurement = np.array([100.0, 100.0, 5000.0, 1.0])
        filter.initiate(measurement)

        state = filter.get_state()
        assert state[0] == pytest.approx(100.0)  # x_center
        assert state[1] == pytest.approx(100.0)  # y_center
        assert state[2] == pytest.approx(5000.0)  # area
        assert state[3] == pytest.approx(1.0)  # aspect_ratio

    def test_initiate_zero_velocity(self, filter):
        """Test that initiate sets velocity to zero."""
        measurement = np.array([100.0, 100.0, 5000.0, 1.0])
        filter.initiate(measurement)

        # Velocities (last 4 elements of state) should be zero
        state = filter.kf.x
        assert state[4, 0] == pytest.approx(0.0)  # dx
        assert state[5, 0] == pytest.approx(0.0)  # dy
        assert state[6, 0] == pytest.approx(0.0)  # da
        assert state[7, 0] == pytest.approx(0.0)  # dr

    def test_predict_returns_measurement(self, filter):
        """Test predict returns measurement vector."""
        measurement = np.array([100.0, 100.0, 5000.0, 1.0])
        filter.initiate(measurement)

        predicted = filter.predict()

        assert isinstance(predicted, np.ndarray)
        assert predicted.shape == (4,)

    def test_predict_constant_position(self, filter):
        """Test prediction with zero velocity maintains position."""
        measurement = np.array([100.0, 100.0, 5000.0, 1.0])
        filter.initiate(measurement)

        predicted = filter.predict()

        # With zero velocity, prediction should be close to initial
        assert predicted[0] == pytest.approx(100.0, rel=0.1)
        assert predicted[1] == pytest.approx(100.0, rel=0.1)

    def test_update(self, filter):
        """Test update adjusts state towards measurement."""
        initial = np.array([100.0, 100.0, 5000.0, 1.0])
        filter.initiate(initial)
        filter.predict()

        # Update with a different measurement
        new_measurement = np.array([110.0, 105.0, 5000.0, 1.0])
        filter.update(new_measurement)

        state = filter.get_state()
        # State should move towards new measurement
        assert state[0] > 100.0  # x moved towards 110
        assert state[1] > 100.0  # y moved towards 105

    def test_velocity_estimation(self, filter):
        """Test that velocity is estimated from consecutive updates."""
        filter.initiate(np.array([100.0, 100.0, 5000.0, 1.0]))

        # Simulate movement
        for i in range(5):
            filter.predict()
            filter.update(np.array([100.0 + i * 10.0, 100.0, 5000.0, 1.0]))

        # Velocity should be positive in x direction
        state = filter.kf.x
        assert state[4, 0] > 0  # dx should be positive

    def test_get_state(self, filter):
        """Test get_state returns current measurement estimate."""
        measurement = np.array([150.0, 200.0, 8000.0, 0.8])
        filter.initiate(measurement)

        state = filter.get_state()

        assert isinstance(state, np.ndarray)
        assert state.shape == (4,)
        assert state[0] == pytest.approx(150.0)
        assert state[1] == pytest.approx(200.0)

    def test_multiple_predict_update_cycles(self, filter):
        """Test filter stability over multiple cycles."""
        filter.initiate(np.array([100.0, 100.0, 5000.0, 1.0]))

        # Run many prediction/update cycles
        for _ in range(100):
            filter.predict()
            filter.update(np.array([100.0, 100.0, 5000.0, 1.0]))

        state = filter.get_state()
        # Should still be close to the consistent measurement
        assert state[0] == pytest.approx(100.0, rel=0.01)
        assert state[1] == pytest.approx(100.0, rel=0.01)

    def test_prediction_extrapolation(self, filter):
        """Test prediction extrapolates with velocity."""
        filter.initiate(np.array([100.0, 100.0, 5000.0, 1.0]))

        # Build up velocity estimate with consistent movement
        for i in range(10):
            filter.predict()
            filter.update(np.array([100.0 + (i + 1) * 10.0, 100.0, 5000.0, 1.0]))

        # Now predict without update
        predicted1 = filter.predict()
        predicted2 = filter.predict()

        # Each prediction should continue the motion
        assert predicted2[0] > predicted1[0]


class TestKalmanFilterEdgeCases:
    """Tests for edge cases in Kalman filter."""

    def test_large_measurement_noise(self):
        """Test filter handles noisy measurements."""
        filter = BBoxKalmanFilter()
        filter.initiate(np.array([100.0, 100.0, 5000.0, 1.0]))

        # Feed noisy measurements
        np.random.seed(42)
        for _ in range(20):
            filter.predict()
            noisy = np.array([
                100.0 + np.random.randn() * 10,
                100.0 + np.random.randn() * 10,
                5000.0 + np.random.randn() * 500,
                1.0 + np.random.randn() * 0.1,
            ])
            filter.update(noisy)

        state = filter.get_state()
        # Should be smoothed to approximately the mean
        assert abs(state[0] - 100.0) < 20
        assert abs(state[1] - 100.0) < 20

    def test_missing_updates(self):
        """Test filter continues predicting when updates are missing."""
        filter = BBoxKalmanFilter()
        filter.initiate(np.array([100.0, 100.0, 5000.0, 1.0]))

        # Build up velocity
        for i in range(5):
            filter.predict()
            filter.update(np.array([100.0 + (i + 1) * 5.0, 100.0, 5000.0, 1.0]))

        # Now only predict (no updates)
        predictions = []
        for _ in range(5):
            pred = filter.predict()
            predictions.append(pred.copy())

        # Should continue extrapolating
        x_positions = [p[0] for p in predictions]
        # Each prediction should be further along
        for i in range(1, len(x_positions)):
            assert x_positions[i] >= x_positions[i - 1]
