"""Kalman filter for ball trajectory prediction."""

import numpy as np


class BallKalmanFilter:
    """
    Kalman filter for ball trajectory prediction.

    Uses a 6-state model: [x, y, vx, vy, ax, ay]
    - Position (x, y)
    - Velocity (vx, vy) in pixels/frame
    - Acceleration (ax, ay) in pixels/frame^2

    Acceleration decays over time to model deceleration due to friction/air resistance.
    """

    def __init__(
        self,
        process_noise_position: float = 1.0,
        process_noise_velocity: float = 0.5,
        process_noise_acceleration: float = 0.1,
        measurement_noise: float = 5.0,
        acceleration_decay: float = 0.98,
    ):
        """
        Initialize Kalman filter.

        Args:
            process_noise_position: Process noise for position states
            process_noise_velocity: Process noise for velocity states
            process_noise_acceleration: Process noise for acceleration states
            measurement_noise: Measurement noise (observation uncertainty)
            acceleration_decay: Decay factor applied to acceleration each frame
        """
        self.process_noise_position = process_noise_position
        self.process_noise_velocity = process_noise_velocity
        self.process_noise_acceleration = process_noise_acceleration
        self.measurement_noise = measurement_noise
        self.acceleration_decay = acceleration_decay

        # State vector: [x, y, vx, vy, ax, ay]
        self.state = np.zeros(6)

        # State covariance matrix
        self.P = np.eye(6) * 100  # High initial uncertainty

        # Process noise covariance
        self.Q = np.diag([
            process_noise_position,
            process_noise_position,
            process_noise_velocity,
            process_noise_velocity,
            process_noise_acceleration,
            process_noise_acceleration,
        ])

        # Measurement noise covariance (we only observe position)
        self.R = np.eye(2) * measurement_noise

        # Measurement matrix (we observe x, y)
        self.H = np.zeros((2, 6))
        self.H[0, 0] = 1  # observe x
        self.H[1, 1] = 1  # observe y

        self._initialized = False

    def _get_transition_matrix(self, dt: float = 1.0) -> np.ndarray:
        """
        Get state transition matrix for time step dt.

        With acceleration decay, the model is:
        x(t+1) = x(t) + vx(t)*dt + 0.5*ax(t)*dt^2
        vx(t+1) = vx(t) + ax(t)*dt
        ax(t+1) = ax(t) * acceleration_decay

        Args:
            dt: Time step (in frames)

        Returns:
            6x6 state transition matrix
        """
        F = np.eye(6)

        # Position update: x += vx*dt + 0.5*ax*dt^2
        F[0, 2] = dt  # x += vx*dt
        F[0, 4] = 0.5 * dt * dt  # x += 0.5*ax*dt^2
        F[1, 3] = dt  # y += vy*dt
        F[1, 5] = 0.5 * dt * dt  # y += 0.5*ay*dt^2

        # Velocity update: vx += ax*dt
        F[2, 4] = dt  # vx += ax*dt
        F[3, 5] = dt  # vy += ay*dt

        # Acceleration decay
        F[4, 4] = self.acceleration_decay ** dt
        F[5, 5] = self.acceleration_decay ** dt

        return F

    def initialize(
        self,
        position: tuple[float, float],
        velocity: tuple[float, float] | None = None,
        acceleration: tuple[float, float] | None = None,
    ) -> None:
        """
        Initialize filter with initial state.

        Args:
            position: Initial (x, y) position
            velocity: Initial (vx, vy) velocity, defaults to (0, 0)
            acceleration: Initial (ax, ay) acceleration, defaults to (0, 0)
        """
        self.state[0] = position[0]
        self.state[1] = position[1]

        if velocity is not None:
            self.state[2] = velocity[0]
            self.state[3] = velocity[1]
        else:
            self.state[2] = 0.0
            self.state[3] = 0.0

        if acceleration is not None:
            self.state[4] = acceleration[0]
            self.state[5] = acceleration[1]
        else:
            self.state[4] = 0.0
            self.state[5] = 0.0

        # Reset covariance - lower uncertainty for position if we have a good initial value
        self.P = np.diag([
            self.measurement_noise,  # x position uncertainty
            self.measurement_noise,  # y position uncertainty
            self.process_noise_velocity * 10,  # vx uncertainty (higher if not provided)
            self.process_noise_velocity * 10,  # vy uncertainty
            self.process_noise_acceleration * 10,  # ax uncertainty
            self.process_noise_acceleration * 10,  # ay uncertainty
        ])

        self._initialized = True

    def predict(self, dt: float = 1.0) -> tuple[float, float]:
        """
        Predict next state.

        Args:
            dt: Time step in frames

        Returns:
            Predicted (x, y) position
        """
        if not self._initialized:
            raise RuntimeError("Kalman filter not initialized. Call initialize() first.")

        F = self._get_transition_matrix(dt)

        # State prediction
        self.state = F @ self.state

        # Covariance prediction
        self.P = F @ self.P @ F.T + self.Q * dt

        return (self.state[0], self.state[1])

    def update(self, measurement: tuple[float, float]) -> tuple[float, float]:
        """
        Update state with measurement.

        Args:
            measurement: Observed (x, y) position

        Returns:
            Updated (x, y) position
        """
        if not self._initialized:
            raise RuntimeError("Kalman filter not initialized. Call initialize() first.")

        z = np.array([measurement[0], measurement[1]])

        # Innovation (measurement residual)
        y = z - self.H @ self.state

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.state = self.state + K @ y

        # Covariance update
        identity_matrix = np.eye(6)
        self.P = (identity_matrix - K @ self.H) @ self.P

        return (self.state[0], self.state[1])

    def get_state(self) -> dict:
        """
        Get current state.

        Returns:
            Dict with position, velocity, acceleration
        """
        return {
            "position": (self.state[0], self.state[1]),
            "velocity": (self.state[2], self.state[3]),
            "acceleration": (self.state[4], self.state[5]),
        }

    def get_position_uncertainty(self) -> float:
        """
        Get position uncertainty (sqrt of position variance).

        Returns:
            Position uncertainty in pixels
        """
        return float(np.sqrt(self.P[0, 0] + self.P[1, 1]))

    def copy(self) -> "BallKalmanFilter":
        """
        Create a copy of this filter with the same state.

        Returns:
            New BallKalmanFilter with copied state
        """
        new_filter = BallKalmanFilter(
            process_noise_position=self.process_noise_position,
            process_noise_velocity=self.process_noise_velocity,
            process_noise_acceleration=self.process_noise_acceleration,
            measurement_noise=self.measurement_noise,
            acceleration_decay=self.acceleration_decay,
        )
        new_filter.state = self.state.copy()
        new_filter.P = self.P.copy()
        new_filter._initialized = self._initialized
        return new_filter
