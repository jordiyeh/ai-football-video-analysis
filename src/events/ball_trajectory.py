"""Ball trajectory analysis for event detection."""

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class BallTrajectoryPoint:
    """Single point in ball trajectory."""

    frame_idx: int
    timestamp: float
    position: tuple[float, float]  # (x, y) center
    velocity: tuple[float, float] | None  # (vx, vy) pixels/frame
    speed: float | None  # pixels/frame
    confidence: float


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
