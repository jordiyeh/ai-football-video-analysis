"""Kalman filter for bounding box tracking."""

import numpy as np
from filterpy.kalman import KalmanFilter


class BBoxKalmanFilter:
    """
    Kalman filter for tracking bounding boxes in image space.

    State: [x_center, y_center, area, aspect_ratio, dx, dy, da, dr]
    - x_center, y_center: center coordinates
    - area: bbox area
    - aspect_ratio: width / height
    - dx, dy, da, dr: velocities
    """

    def __init__(self):
        """Initialize Kalman filter for bbox tracking."""
        # 8 state variables, 4 measurement variables
        self.kf = KalmanFilter(dim_x=8, dim_z=4)

        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],  # x = x + dx
            [0, 1, 0, 0, 0, 1, 0, 0],  # y = y + dy
            [0, 0, 1, 0, 0, 0, 1, 0],  # a = a + da
            [0, 0, 0, 1, 0, 0, 0, 1],  # r = r + dr
            [0, 0, 0, 0, 1, 0, 0, 0],  # dx = dx
            [0, 0, 0, 0, 0, 1, 0, 0],  # dy = dy
            [0, 0, 0, 0, 0, 0, 1, 0],  # da = da
            [0, 0, 0, 0, 0, 0, 0, 1],  # dr = dr
        ])

        # Measurement matrix (we measure x, y, a, r)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ])

        # Measurement noise covariance
        self.kf.R *= 1.0

        # Process noise covariance (higher = trust model less)
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # Initial state covariance
        self.kf.P[4:, 4:] *= 1000.0  # High uncertainty on velocities
        self.kf.P *= 10.0

    def initiate(self, measurement: np.ndarray) -> None:
        """
        Initialize state from first measurement.

        Args:
            measurement: [x_center, y_center, area, aspect_ratio]
        """
        self.kf.x[:4, 0] = measurement
        self.kf.x[4:, 0] = 0  # Zero velocity initially

    def predict(self) -> np.ndarray:
        """
        Predict next state.

        Returns:
            Predicted measurement [x_center, y_center, area, aspect_ratio]
        """
        self.kf.predict()
        return self.kf.x[:4, 0]

    def update(self, measurement: np.ndarray) -> None:
        """
        Update state with new measurement.

        Args:
            measurement: [x_center, y_center, area, aspect_ratio]
        """
        self.kf.update(measurement)

    def get_state(self) -> np.ndarray:
        """
        Get current state estimate.

        Returns:
            Current measurement estimate [x_center, y_center, area, aspect_ratio]
        """
        return self.kf.x[:4, 0]


def bbox_to_measurement(bbox: tuple[float, float, float, float]) -> np.ndarray:
    """
    Convert bounding box to measurement vector.

    Args:
        bbox: (x1, y1, x2, y2)

    Returns:
        Measurement: [x_center, y_center, area, aspect_ratio]
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    x_center = x1 + width / 2
    y_center = y1 + height / 2
    area = width * height
    aspect_ratio = width / height if height > 0 else 1.0

    return np.array([x_center, y_center, area, aspect_ratio])


def measurement_to_bbox(measurement: np.ndarray) -> tuple[float, float, float, float]:
    """
    Convert measurement vector to bounding box.

    Args:
        measurement: [x_center, y_center, area, aspect_ratio]

    Returns:
        bbox: (x1, y1, x2, y2)
    """
    x_center, y_center, area, aspect_ratio = measurement

    # Ensure valid values (avoid sqrt of negative numbers)
    area = max(0.0, area)
    aspect_ratio = max(0.1, aspect_ratio)  # Avoid division by zero

    # Calculate width and height from area and aspect ratio
    height = np.sqrt(area / aspect_ratio)
    width = aspect_ratio * height

    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    return (float(x1), float(y1), float(x2), float(y2))
