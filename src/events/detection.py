"""Event detection for shots, goals, and other match events."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.events.ball_trajectory import BallTrajectory


@dataclass
class Event:
    """Single match event."""

    event_type: Literal["shot", "goal", "pass", "tackle", "other"]
    frame_idx: int
    timestamp: float
    confidence: float
    location: tuple[float, float] | None = None  # (x, y) in pixels
    metadata: dict | None = None  # Event-specific data


class EventDetector:
    """Detect match events from tracking data."""

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        shot_velocity_threshold: float = 15.0,
        goal_confidence_threshold: float = 0.6,
        fps: float = 30.0,
    ):
        """
        Initialize event detector.

        Args:
            frame_width: Video frame width
            frame_height: Video frame height
            shot_velocity_threshold: Minimum ball speed for shot (pixels/frame)
            goal_confidence_threshold: Minimum confidence for goal detection
            fps: Video frames per second
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.shot_velocity_threshold = shot_velocity_threshold
        self.goal_confidence_threshold = goal_confidence_threshold
        self.fps = fps

        # Goal regions (estimated as edges of frame where goals typically are)
        # This is simplified - in practice would use field detection
        self.goal_regions = self._estimate_goal_regions()

    def _estimate_goal_regions(self) -> list[dict]:
        """
        Estimate goal regions in pixel space.

        For now, assumes goals are near top and bottom edges (typical broadcast view).
        Returns list of region dicts with bounds.
        """
        edge_margin = 0.15  # 15% from edge
        goal_width_fraction = 0.3  # Goals span central 30% of frame width

        x_center = self.frame_width / 2
        goal_half_width = (self.frame_width * goal_width_fraction) / 2

        # Top goal region
        top_goal = {
            "name": "top",
            "bounds": {
                "x_min": x_center - goal_half_width,
                "x_max": x_center + goal_half_width,
                "y_min": 0,
                "y_max": self.frame_height * edge_margin,
            },
        }

        # Bottom goal region
        bottom_goal = {
            "name": "bottom",
            "bounds": {
                "x_min": x_center - goal_half_width,
                "x_max": x_center + goal_half_width,
                "y_min": self.frame_height * (1 - edge_margin),
                "y_max": self.frame_height,
            },
        }

        return [top_goal, bottom_goal]

    def is_in_goal_region(self, position: tuple[float, float]) -> tuple[bool, str | None]:
        """
        Check if position is in a goal region.

        Args:
            position: (x, y) position

        Returns:
            (is_in_goal, goal_name)
        """
        x, y = position

        for goal in self.goal_regions:
            bounds = goal["bounds"]
            if (
                bounds["x_min"] <= x <= bounds["x_max"]
                and bounds["y_min"] <= y <= bounds["y_max"]
            ):
                return True, goal["name"]

        return False, None

    def detect_shots(self, ball_trajectory: BallTrajectory) -> list[Event]:
        """
        Detect shot events from ball trajectory.

        A shot is detected when:
        1. Ball moves at high speed
        2. Ball is moving towards a goal region
        3. Direction is relatively straight

        Args:
            ball_trajectory: Ball trajectory data

        Returns:
            List of shot events
        """
        events = []

        # Find high-speed segments
        high_speed_segments = ball_trajectory.get_high_speed_segments(
            speed_threshold=self.shot_velocity_threshold,
            min_duration_frames=2,
        )

        for start_idx, end_idx in high_speed_segments:
            # Check if ball is moving towards goal
            start_point = ball_trajectory.points[start_idx]
            end_point = ball_trajectory.points[min(end_idx + 5, len(ball_trajectory.points) - 1)]

            # Direction vector
            dx = end_point.position[0] - start_point.position[0]
            dy = end_point.position[1] - start_point.position[1]

            # Check if moving towards goal regions
            is_towards_goal = False
            target_goal = None

            for goal in self.goal_regions:
                bounds = goal["bounds"]
                goal_center_y = (bounds["y_min"] + bounds["y_max"]) / 2

                # Check if movement is towards this goal
                if goal["name"] == "top" and dy < -5:  # Moving up
                    is_towards_goal = True
                    target_goal = goal["name"]
                    break
                elif goal["name"] == "bottom" and dy > 5:  # Moving down
                    is_towards_goal = True
                    target_goal = goal["name"]
                    break

            if is_towards_goal:
                # Compute confidence based on speed and straightness
                avg_speed = np.mean([
                    p.speed for p in ball_trajectory.points[start_idx:end_idx + 1]
                    if p.speed is not None
                ])

                # Higher speed = higher confidence
                speed_confidence = min(1.0, avg_speed / (self.shot_velocity_threshold * 2))

                # Straightness: check if trajectory is relatively straight
                # (low variance in velocity direction)
                velocities = [
                    p.velocity for p in ball_trajectory.points[start_idx:end_idx + 1]
                    if p.velocity is not None
                ]

                if len(velocities) > 1:
                    angles = []
                    for v in velocities:
                        angle = np.arctan2(v[1], v[0])
                        angles.append(angle)

                    angle_std = np.std(angles)
                    straightness_confidence = max(0.0, 1.0 - angle_std / np.pi)
                else:
                    straightness_confidence = 0.5

                confidence = (speed_confidence + straightness_confidence) / 2

                event = Event(
                    event_type="shot",
                    frame_idx=start_point.frame_idx,
                    timestamp=start_point.timestamp,
                    confidence=confidence,
                    location=start_point.position,
                    metadata={
                        "speed": float(avg_speed),
                        "target_goal": target_goal,
                        "duration_frames": end_idx - start_idx + 1,
                    },
                )
                events.append(event)

        return events

    def detect_goals(
        self,
        ball_trajectory: BallTrajectory,
        shot_events: list[Event],
    ) -> list[Event]:
        """
        Detect goal events from ball trajectory and shot events.

        A goal is detected when:
        1. Ball enters goal region
        2. Ball was moving fast (from a shot)
        3. Ball stays in/near goal region (not a rebound)

        Args:
            ball_trajectory: Ball trajectory data
            shot_events: Previously detected shot events

        Returns:
            List of goal events
        """
        events = []

        # Track when ball enters goal regions
        for i, point in enumerate(ball_trajectory.points):
            in_goal, goal_name = self.is_in_goal_region(point.position)

            if not in_goal:
                continue

            # Check if there was a recent shot
            recent_shot = None
            for shot in shot_events:
                frame_diff = point.frame_idx - shot.frame_idx
                time_diff = frame_diff / self.fps

                # Shot within last 3 seconds
                if 0 <= time_diff <= 3.0:
                    recent_shot = shot
                    break

            if recent_shot is None:
                continue

            # Check if ball stays in/near goal (not immediate rebound)
            stays_in_goal = self._check_ball_stays_in_goal(ball_trajectory, i, duration_frames=10)

            if stays_in_goal:
                # High confidence goal
                confidence = min(1.0, recent_shot.confidence * 1.2)

                # Check if targets match
                if recent_shot.metadata and recent_shot.metadata.get("target_goal") == goal_name:
                    confidence = min(1.0, confidence * 1.1)

                if confidence >= self.goal_confidence_threshold:
                    event = Event(
                        event_type="goal",
                        frame_idx=point.frame_idx,
                        timestamp=point.timestamp,
                        confidence=confidence,
                        location=point.position,
                        metadata={
                            "goal_region": goal_name,
                            "shot_frame": recent_shot.frame_idx,
                            "shot_timestamp": recent_shot.timestamp,
                        },
                    )
                    events.append(event)

        # Deduplicate nearby goals (keep highest confidence)
        events = self._deduplicate_events(events, time_window=5.0)

        return events

    def _check_ball_stays_in_goal(
        self,
        ball_trajectory: BallTrajectory,
        start_idx: int,
        duration_frames: int = 10,
    ) -> bool:
        """Check if ball stays in/near goal region after entering."""
        if start_idx + duration_frames >= len(ball_trajectory.points):
            # Not enough data, assume it stays
            return True

        # Check next few points
        in_goal_count = 0
        for i in range(start_idx, min(start_idx + duration_frames, len(ball_trajectory.points))):
            point = ball_trajectory.points[i]
            in_goal, _ = self.is_in_goal_region(point.position)

            # Also count as "in goal" if near goal (within 10% of frame height)
            if not in_goal:
                # Check if near goal edges
                near_top = point.position[1] < self.frame_height * 0.2
                near_bottom = point.position[1] > self.frame_height * 0.8
                if near_top or near_bottom:
                    in_goal = True

            if in_goal:
                in_goal_count += 1

        # Ball "stays" if it's in goal for at least 50% of duration
        return in_goal_count >= duration_frames * 0.5

    def _deduplicate_events(self, events: list[Event], time_window: float = 3.0) -> list[Event]:
        """Remove duplicate events within time window, keeping highest confidence."""
        if not events:
            return []

        # Sort by timestamp
        events = sorted(events, key=lambda e: e.timestamp)

        deduplicated = []
        i = 0

        while i < len(events):
            current = events[i]
            best_event = current

            # Find all events within time window
            j = i + 1
            while j < len(events):
                if events[j].timestamp - current.timestamp <= time_window:
                    # Keep highest confidence
                    if events[j].confidence > best_event.confidence:
                        best_event = events[j]
                    j += 1
                else:
                    break

            deduplicated.append(best_event)
            i = j

        return deduplicated
