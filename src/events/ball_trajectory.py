"""Ball trajectory analysis for event detection."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.config.schemas import InterpolationConfig


@dataclass
class BallTrajectoryPoint:
    """Single point in ball trajectory."""

    frame_idx: int
    timestamp: float
    position: tuple[float, float]  # (x, y) center
    velocity: tuple[float, float] | None  # (vx, vy) pixels/frame
    speed: float | None  # pixels/frame
    confidence: float
    interpolated: bool = False
    interpolation_source: str | None = None  # "linear", "physics_forward", "physics_blended"


class BallTrajectory:
    """Analyze ball movement from tracking data."""

    def __init__(self, smoothing_window: int = 3):
        """
        Initialize ball trajectory analyzer.

        Args:
            smoothing_window: Window size for velocity smoothing
        """
        self.smoothing_window = smoothing_window
        self.points: list[BallTrajectoryPoint] = []

    def add_from_tracks(self, ball_tracks: list[dict]) -> None:
        """
        Build trajectory from ball tracking data.

        Args:
            ball_tracks: List of ball track dicts sorted by frame_idx
        """
        self.points = []

        for track in ball_tracks:
            bbox = track["bbox"]
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2

            point = BallTrajectoryPoint(
                frame_idx=track["frame_idx"],
                timestamp=track["timestamp"],
                position=(center_x, center_y),
                velocity=None,
                speed=None,
                confidence=track["confidence"],
            )
            self.points.append(point)

        # Compute velocities
        self._compute_velocities()

    def _compute_velocities(self) -> None:
        """Compute velocity and speed for each point."""
        if len(self.points) < 2:
            return

        for i in range(len(self.points)):
            # Use forward/backward difference at edges, central difference in middle
            if i == 0:
                # Forward difference
                next_point = self.points[i + 1]
                dt = next_point.frame_idx - self.points[i].frame_idx
                if dt > 0:
                    dx = next_point.position[0] - self.points[i].position[0]
                    dy = next_point.position[1] - self.points[i].position[1]
                    vx, vy = dx / dt, dy / dt
                else:
                    vx, vy = 0.0, 0.0
            elif i == len(self.points) - 1:
                # Backward difference
                prev_point = self.points[i - 1]
                dt = self.points[i].frame_idx - prev_point.frame_idx
                if dt > 0:
                    dx = self.points[i].position[0] - prev_point.position[0]
                    dy = self.points[i].position[1] - prev_point.position[1]
                    vx, vy = dx / dt, dy / dt
                else:
                    vx, vy = 0.0, 0.0
            else:
                # Central difference (more accurate)
                prev_point = self.points[i - 1]
                next_point = self.points[i + 1]
                dt = next_point.frame_idx - prev_point.frame_idx
                if dt > 0:
                    dx = next_point.position[0] - prev_point.position[0]
                    dy = next_point.position[1] - prev_point.position[1]
                    vx, vy = dx / dt, dy / dt
                else:
                    vx, vy = 0.0, 0.0

            self.points[i].velocity = (vx, vy)
            self.points[i].speed = np.sqrt(vx**2 + vy**2)

        # Smooth velocities
        self._smooth_velocities()

    def _smooth_velocities(self) -> None:
        """Apply moving average smoothing to velocities."""
        if len(self.points) < self.smoothing_window:
            return

        window = self.smoothing_window
        half_window = window // 2

        # Store original velocities
        orig_velocities = [p.velocity for p in self.points]

        for i in range(len(self.points)):
            start = max(0, i - half_window)
            end = min(len(self.points), i + half_window + 1)

            # Average velocities in window
            velocities = [orig_velocities[j] for j in range(start, end) if orig_velocities[j] is not None]

            if velocities:
                avg_vx = np.mean([v[0] for v in velocities])
                avg_vy = np.mean([v[1] for v in velocities])
                self.points[i].velocity = (float(avg_vx), float(avg_vy))
                self.points[i].speed = float(np.sqrt(avg_vx**2 + avg_vy**2))

    def get_high_speed_segments(
        self, speed_threshold: float, min_duration_frames: int = 3
    ) -> list[tuple[int, int]]:
        """
        Find segments where ball is moving fast.

        Args:
            speed_threshold: Minimum speed in pixels/frame
            min_duration_frames: Minimum segment length

        Returns:
            List of (start_idx, end_idx) tuples into self.points
        """
        if not self.points:
            return []

        segments = []
        in_segment = False
        segment_start = 0

        for i, point in enumerate(self.points):
            if point.speed is not None and point.speed >= speed_threshold:
                if not in_segment:
                    in_segment = True
                    segment_start = i
            else:
                if in_segment:
                    segment_length = i - segment_start
                    if segment_length >= min_duration_frames:
                        segments.append((segment_start, i - 1))
                    in_segment = False

        # Handle segment at end
        if in_segment:
            segment_length = len(self.points) - segment_start
            if segment_length >= min_duration_frames:
                segments.append((segment_start, len(self.points) - 1))

        return segments

    def get_direction_changes(self, angle_threshold: float = 45.0) -> list[int]:
        """
        Find indices where ball changes direction significantly.

        Args:
            angle_threshold: Minimum angle change in degrees

        Returns:
            List of indices where direction changes
        """
        if len(self.points) < 3:
            return []

        changes = []

        for i in range(1, len(self.points) - 1):
            prev_vel = self.points[i - 1].velocity
            curr_vel = self.points[i].velocity
            next_vel = self.points[i + 1].velocity

            if prev_vel is None or curr_vel is None or next_vel is None:
                continue

            # Compute angle between consecutive velocity vectors
            v1 = np.array(curr_vel)
            v2 = np.array(next_vel)

            # Skip if either velocity is near zero
            if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
                continue

            # Angle between vectors
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))

            if angle >= angle_threshold:
                changes.append(i)

        return changes

    def is_near_edge(
        self,
        point_idx: int,
        frame_width: int,
        frame_height: int,
        edge_margin: float = 0.1,
    ) -> bool:
        """
        Check if ball position is near frame edges.

        Args:
            point_idx: Index into self.points
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            edge_margin: Edge margin as fraction of frame size

        Returns:
            True if near any edge
        """
        if point_idx >= len(self.points):
            return False

        x, y = self.points[point_idx].position

        margin_x = frame_width * edge_margin
        margin_y = frame_height * edge_margin

        return (
            x < margin_x
            or x > frame_width - margin_x
            or y < margin_y
            or y > frame_height - margin_y
        )

    def interpolate_gaps(
        self,
        max_gap_frames: int = 300,
        fps: float = 30.0,
        config: "InterpolationConfig | None" = None,
    ) -> "BallTrajectory":
        """
        Fill gaps in trajectory using linear or physics-based interpolation.

        For short gaps (<= physics_threshold), uses linear interpolation.
        For longer gaps, uses Kalman filter prediction with bidirectional blending.

        Args:
            max_gap_frames: Maximum frame gap to interpolate across (legacy, use config.max_gap)
            fps: Video FPS for timestamp calculation
            config: Interpolation configuration (optional, uses defaults if not provided)

        Returns:
            New BallTrajectory with interpolated points
        """
        # Use config or defaults
        if config is None:
            from src.config.schemas import InterpolationConfig
            config = InterpolationConfig()

        # Use config.max_gap if larger than legacy parameter
        effective_max_gap = max(max_gap_frames, config.max_gap)

        if len(self.points) < 2:
            return self

        # Sort points by frame index
        sorted_points = sorted(self.points, key=lambda p: p.frame_idx)

        interpolated_points = []

        for i in range(len(sorted_points) - 1):
            current = sorted_points[i]
            next_point = sorted_points[i + 1]

            # Add current point (not interpolated)
            interpolated_points.append(current)

            # Check gap size
            gap = next_point.frame_idx - current.frame_idx

            if gap > 1 and gap <= effective_max_gap:
                if gap <= config.physics_threshold:
                    # Short gaps: use linear interpolation
                    new_points = self._interpolate_linear(
                        current, next_point, gap, fps, config
                    )
                else:
                    # Longer gaps: use physics-based interpolation
                    new_points = self._interpolate_physics(
                        current, next_point, gap, fps, config
                    )
                interpolated_points.extend(new_points)

        # Add last point (not interpolated)
        interpolated_points.append(sorted_points[-1])

        # Create new trajectory with interpolated points
        new_trajectory = BallTrajectory(smoothing_window=self.smoothing_window)
        new_trajectory.points = interpolated_points
        new_trajectory._compute_velocities()

        return new_trajectory

    def _interpolate_linear(
        self,
        start: BallTrajectoryPoint,
        end: BallTrajectoryPoint,
        gap: int,
        fps: float,
        config: "InterpolationConfig",
    ) -> list[BallTrajectoryPoint]:
        """
        Linear interpolation for short gaps.

        Args:
            start: Starting point
            end: Ending point
            gap: Number of frames between start and end
            fps: Video FPS
            config: Interpolation configuration

        Returns:
            List of interpolated points (not including start/end)
        """
        points = []

        for frame_offset in range(1, gap):
            t = frame_offset / gap  # 0 to 1

            interp_x = start.position[0] + t * (end.position[0] - start.position[0])
            interp_y = start.position[1] + t * (end.position[1] - start.position[1])

            interp_frame = start.frame_idx + frame_offset
            interp_timestamp = interp_frame / fps

            # Confidence calculation for linear interpolation
            interp_confidence = self._compute_interpolation_confidence(
                start, end, frame_offset, gap, fps, config
            )

            interp_point = BallTrajectoryPoint(
                frame_idx=interp_frame,
                timestamp=interp_timestamp,
                position=(interp_x, interp_y),
                velocity=None,
                speed=None,
                confidence=interp_confidence,
                interpolated=True,
                interpolation_source="linear",
            )
            points.append(interp_point)

        return points

    def _interpolate_physics(
        self,
        start: BallTrajectoryPoint,
        end: BallTrajectoryPoint,
        gap: int,
        fps: float,
        config: "InterpolationConfig",
    ) -> list[BallTrajectoryPoint]:
        """
        Physics-based interpolation using Kalman filter with bidirectional blending.

        Args:
            start: Starting point
            end: Ending point
            gap: Number of frames between start and end
            fps: Video FPS
            config: Interpolation configuration

        Returns:
            List of interpolated points (not including start/end)
        """
        from src.events.kalman_filter import BallKalmanFilter

        # Create forward Kalman filter from start point
        forward_filter = BallKalmanFilter(
            process_noise_position=config.process_noise_position,
            process_noise_velocity=config.process_noise_velocity,
            process_noise_acceleration=config.process_noise_acceleration,
            measurement_noise=config.measurement_noise,
            acceleration_decay=config.acceleration_decay,
        )
        forward_filter.initialize(
            position=start.position,
            velocity=start.velocity,
        )

        # Forward predictions
        forward_positions = []
        for _ in range(1, gap):
            pos = forward_filter.predict(dt=1.0)
            forward_positions.append(pos)

        if config.use_bidirectional:
            # Create backward Kalman filter from end point
            backward_filter = BallKalmanFilter(
                process_noise_position=config.process_noise_position,
                process_noise_velocity=config.process_noise_velocity,
                process_noise_acceleration=config.process_noise_acceleration,
                measurement_noise=config.measurement_noise,
                acceleration_decay=config.acceleration_decay,
            )

            # Reverse velocity for backward prediction
            backward_velocity = None
            if end.velocity is not None:
                backward_velocity = (-end.velocity[0], -end.velocity[1])

            backward_filter.initialize(
                position=end.position,
                velocity=backward_velocity,
            )

            # Backward predictions (in reverse order)
            backward_positions_reversed = []
            for _ in range(1, gap):
                pos = backward_filter.predict(dt=1.0)
                backward_positions_reversed.append(pos)

            # Reverse to get forward order
            backward_positions = list(reversed(backward_positions_reversed))

            # Blend forward and backward predictions using smoothstep
            blended_positions = []
            for i in range(len(forward_positions)):
                t = (i + 1) / gap  # 0 to 1 (exclusive of endpoints)
                # Smoothstep: 3t² - 2t³ (gives more weight to endpoints)
                blend_weight = 3 * t * t - 2 * t * t * t

                fx, fy = forward_positions[i]
                bx, by = backward_positions[i]

                # Blend: start with forward (weight 1-blend), end with backward (weight blend)
                blended_x = fx * (1 - blend_weight) + bx * blend_weight
                blended_y = fy * (1 - blend_weight) + by * blend_weight

                blended_positions.append((blended_x, blended_y))

            final_positions = blended_positions
            interpolation_source = "physics_blended"
        else:
            final_positions = forward_positions
            interpolation_source = "physics_forward"

        # Create trajectory points
        points = []
        for i, (x, y) in enumerate(final_positions):
            frame_offset = i + 1
            interp_frame = start.frame_idx + frame_offset
            interp_timestamp = interp_frame / fps

            interp_confidence = self._compute_interpolation_confidence(
                start, end, frame_offset, gap, fps, config
            )

            interp_point = BallTrajectoryPoint(
                frame_idx=interp_frame,
                timestamp=interp_timestamp,
                position=(x, y),
                velocity=None,
                speed=None,
                confidence=interp_confidence,
                interpolated=True,
                interpolation_source=interpolation_source,
            )
            points.append(interp_point)

        return points

    def _compute_interpolation_confidence(
        self,
        start: BallTrajectoryPoint,
        end: BallTrajectoryPoint,
        frame_offset: int,
        gap: int,
        fps: float,
        config: "InterpolationConfig",
    ) -> float:
        """
        Compute confidence for an interpolated point.

        Confidence decays exponentially with distance from nearest known point.
        Formula: base_conf * decay_rate^(distance_in_seconds * 30) with floor at min_confidence

        Args:
            start: Starting known point
            end: Ending known point
            frame_offset: Frames from start
            gap: Total gap size
            fps: Video FPS
            config: Interpolation configuration

        Returns:
            Confidence value between min_confidence and base confidence
        """
        # Base confidence is minimum of endpoints
        base_confidence = min(start.confidence, end.confidence)

        # Distance from nearest endpoint (in frames)
        distance_from_start = frame_offset
        distance_from_end = gap - frame_offset
        min_distance = min(distance_from_start, distance_from_end)

        # Convert to seconds for decay calculation
        distance_seconds = min_distance / fps

        # Per-second decay rate raised to distance in seconds
        # The decay_rate is configured as "per second at 30fps"
        decay = config.confidence_decay_rate ** (distance_seconds * 30)

        # Apply decay with floor
        confidence = base_confidence * decay
        confidence = max(confidence, config.min_confidence)

        return confidence
