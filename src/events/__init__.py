"""Event detection module for shots, goals, and match events."""

from src.events.ball_trajectory import BallTrajectory, BallTrajectoryPoint
from src.events.detection import Event, EventDetector

__all__ = [
    "BallTrajectory",
    "BallTrajectoryPoint",
    "Event",
    "EventDetector",
]
