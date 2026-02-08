"""Event detection for shots, goals, and other match events."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from src.events.ball_trajectory import BallTrajectory
from src.vision.field.goal_detector import (
    GoalRegionProvider,
    HeuristicGoalRegionProvider,
)

if TYPE_CHECKING:
    from src.config.schemas import AlternativeShotDetectionConfig


EventType = Literal[
    "shot",
    "goal",
    "pass",
    "set_piece",
    "kickoff",
    "throw_in",
    "corner_kick",
    "free_kick",
    "goal_kick",
    "tackle",
    "other",
]

EventFamily = Literal["shot", "goal", "pass", "set_piece", "defensive", "other"]

# Schema for per-event metadata payload, including pass/set-piece families.
EVENT_METADATA_SCHEMA_VERSION = "1.0"

SET_PIECE_EVENT_TYPES = frozenset(
    {"set_piece", "kickoff", "throw_in", "corner_kick", "free_kick", "goal_kick"}
)

EVENT_TYPE_TO_FAMILY: dict[str, EventFamily] = {
    "shot": "shot",
    "goal": "goal",
    "pass": "pass",
    "set_piece": "set_piece",
    "kickoff": "set_piece",
    "throw_in": "set_piece",
    "corner_kick": "set_piece",
    "free_kick": "set_piece",
    "goal_kick": "set_piece",
    "tackle": "defensive",
    "other": "other",
}


def infer_event_family(event_type: EventType | str) -> EventFamily:
    """Map an event type to its event family."""
    return EVENT_TYPE_TO_FAMILY.get(event_type, "other")


def normalize_event_metadata(
    event_type: EventType | str,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Normalize metadata into schema-versioned structure.

    Pass and set-piece events use this canonical payload so downstream
    detectors can add fields without breaking consumers.
    """
    normalized: dict[str, Any] = dict(metadata or {})
    event_family = infer_event_family(event_type)

    normalized.setdefault("schema_version", EVENT_METADATA_SCHEMA_VERSION)
    normalized.setdefault("event_family", event_family)
    normalized.setdefault("event_type", event_type)

    if event_family == "set_piece":
        set_piece_type = normalized.get("set_piece_type")
        if event_type != "set_piece" and not set_piece_type:
            set_piece_type = event_type
        if set_piece_type:
            normalized["set_piece_type"] = str(set_piece_type)

    return normalized


@dataclass
class Event:
    """Single match event."""

    event_type: EventType
    frame_idx: int
    timestamp: float
    confidence: float
    location: tuple[float, float] | None = None  # (x, y) in pixels
    metadata: dict[str, Any] | None = None  # Event-specific data

    @property
    def event_family(self) -> EventFamily:
        """Return event family derived from event type."""
        return infer_event_family(self.event_type)

    def __post_init__(self) -> None:
        """Ensure pass/set-piece events carry schema-versioned metadata."""
        if self.event_family in {"pass", "set_piece"}:
            self.metadata = normalize_event_metadata(self.event_type, self.metadata)


class EventDetector:
    """Detect match events from tracking data."""

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        shot_velocity_threshold: float = 15.0,
        goal_confidence_threshold: float = 0.6,
        fps: float = 30.0,
        alternative_config: "AlternativeShotDetectionConfig | None" = None,
        goal_region_provider: GoalRegionProvider | None = None,
    ):
        """
        Initialize event detector.

        Args:
            frame_width: Video frame width
            frame_height: Video frame height
            shot_velocity_threshold: Minimum ball speed for shot (pixels/frame)
            goal_confidence_threshold: Minimum confidence for goal detection
            fps: Video frames per second
            alternative_config: Config for alternative shot detection (sparse ball data)
            goal_region_provider: Optional provider for goal regions (defaults to heuristic)
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.shot_velocity_threshold = shot_velocity_threshold
        self.goal_confidence_threshold = goal_confidence_threshold
        self.fps = fps
        self.alternative_config = alternative_config

        # Goal region provider (use heuristic fallback if not provided)
        if goal_region_provider is not None:
            self._goal_region_provider = goal_region_provider
        else:
            self._goal_region_provider = HeuristicGoalRegionProvider(
                frame_width, frame_height
            )

        # Legacy compatibility: expose goal_regions as list of dicts
        self.goal_regions = self._get_goal_regions_as_dicts()

        # Initialize alternative detection components if enabled
        self._kick_detector = None
        self._goal_entry_detector = None
        self._clustering_analyzer = None
        self._gk_analyzer = None
        self._fusion_engine = None
        self._celebration_detector = None

        if alternative_config and alternative_config.enabled:
            self._init_alternative_detectors()

    def _get_goal_regions_as_dicts(self, frame_idx: int = 0) -> list[dict]:
        """
        Get goal regions as list of dicts for legacy compatibility.

        Args:
            frame_idx: Frame index (defaults to 0 for static heuristic)

        Returns:
            List of region dicts with 'name' and 'bounds' keys
        """
        regions = self._goal_region_provider.get_goal_regions(frame_idx)
        return [
            {"name": r.name, "bounds": dict(r.bounds)}
            for r in regions
        ]

    def _estimate_goal_regions(self) -> list[dict]:
        """
        Estimate goal regions in pixel space.

        DEPRECATED: Use _goal_region_provider instead.
        Kept for backward compatibility.
        """
        return self._get_goal_regions_as_dicts()

    def is_in_goal_region(
        self, position: tuple[float, float], frame_idx: int = 0
    ) -> tuple[bool, str | None]:
        """
        Check if position is in a goal region.

        Args:
            position: (x, y) position
            frame_idx: Frame index (for dynamic goal detection)

        Returns:
            (is_in_goal, goal_name)
        """
        return self._goal_region_provider.is_in_goal_region(position, frame_idx)

    def get_goal_regions(self, frame_idx: int = 0) -> list[dict]:
        """
        Get goal regions for a specific frame.

        Args:
            frame_idx: Frame index

        Returns:
            List of region dicts with 'name' and 'bounds' keys
        """
        return self._get_goal_regions_as_dicts(frame_idx)

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
            dy = end_point.position[1] - start_point.position[1]

            # Check if moving towards goal regions
            is_towards_goal = False
            target_goal = None

            for goal in self.goal_regions:
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
            in_goal, goal_name = self.is_in_goal_region(point.position, point.frame_idx)

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
            in_goal, _ = self.is_in_goal_region(point.position, point.frame_idx)

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

    def _init_alternative_detectors(self) -> None:
        """Initialize alternative detection components."""
        from src.events.kick_detection import (
            GoalAreaEntryDetector,
            KickEventDetector,
            ShotFusionEngine,
        )
        from src.events.player_analysis import GoalkeeperAnalyzer, PlayerClusteringAnalyzer

        self._kick_detector = KickEventDetector(self.alternative_config)
        self._goal_entry_detector = GoalAreaEntryDetector(
            self.frame_width,
            self.frame_height,
            self.alternative_config,
            goal_region_provider=self._goal_region_provider,
        )
        self._clustering_analyzer = PlayerClusteringAnalyzer(
            self.frame_width, self.frame_height, self.alternative_config
        )
        self._gk_analyzer = GoalkeeperAnalyzer(
            self.frame_width,
            self.frame_height,
            self.alternative_config,
            goal_region_provider=self._goal_region_provider,
        )
        self._fusion_engine = ShotFusionEngine(self.alternative_config, self.fps)

        # Initialize celebration detector if enabled
        if (
            hasattr(self.alternative_config, 'celebration')
            and self.alternative_config.celebration.enabled
        ):
            from src.events.celebration_detection import CelebrationDetector
            self._celebration_detector = CelebrationDetector(
                self.frame_width,
                self.frame_height,
                self.alternative_config.celebration,
            )

    def _compute_ball_coverage(
        self,
        ball_tracks: list[dict],
        total_frames: int | None = None,
    ) -> float:
        """
        Compute ball detection coverage as fraction of frames with ball.

        Args:
            ball_tracks: List of ball track dicts
            total_frames: Total frames in video (if known)

        Returns:
            Coverage ratio (0.0 to 1.0)
        """
        if not ball_tracks:
            return 0.0

        # Get unique frames with ball detections
        ball_frames = set(t["frame_idx"] for t in ball_tracks)

        if total_frames:
            return len(ball_frames) / total_frames

        # Estimate from min/max frame
        min_frame = min(ball_frames)
        max_frame = max(ball_frames)
        frame_span = max_frame - min_frame + 1

        if frame_span <= 0:
            return 0.0

        return len(ball_frames) / frame_span

    def detect_shots_all(
        self,
        ball_trajectory: BallTrajectory,
        player_tracks: list[dict],
        ball_tracks: list[dict],
        total_frames: int | None = None,
    ) -> list[Event]:
        """
        Combined velocity-based and alternative shot detection.

        Uses velocity-based detection when ball coverage is good,
        falls back to alternative detection when ball data is sparse.

        Args:
            ball_trajectory: Computed ball trajectory
            player_tracks: List of player track dicts
            ball_tracks: List of ball track dicts
            total_frames: Total frames in video (optional)

        Returns:
            List of shot events
        """
        # 1. Try velocity-based detection (existing method)
        velocity_shots = self.detect_shots(ball_trajectory)

        # 2. Try alternative detection if enabled and ball data is sparse
        alternative_shots = []
        if self.alternative_config and self.alternative_config.enabled:
            coverage = self._compute_ball_coverage(ball_tracks, total_frames)

            if coverage < self.alternative_config.ball_coverage_threshold:
                alternative_shots = self.detect_shots_alternative(
                    player_tracks, ball_tracks
                )

        # 3. Merge and deduplicate
        return self._merge_shot_detections(velocity_shots, alternative_shots)

    def detect_shots_alternative(
        self,
        player_tracks: list[dict],
        ball_tracks: list[dict],
    ) -> list[Event]:
        """
        Detect shots using player behavior signals when ball data is sparse.

        Uses multiple signals:
        - Kick events (ball near player foot)
        - Goal area entries
        - Goalkeeper dives
        - Attacking formations
        - Celebrations (for goal confirmation)

        Args:
            player_tracks: List of player track dicts
            ball_tracks: List of ball track dicts

        Returns:
            List of shot events detected via alternative method
        """
        if not self._kick_detector:
            return []

        # 1. Detect kick events
        kick_events = self._kick_detector.detect_kicks(
            player_tracks, ball_tracks, self.frame_width, self.frame_height
        )

        # 2. Detect goal area entries
        goal_entries = self._goal_entry_detector.detect_goal_entries(
            ball_tracks, kick_events, self.fps
        )

        # 3. Analyze player clustering for attack windows
        clustering_states = self._clustering_analyzer.analyze_clustering(player_tracks)
        attack_windows = self._clustering_analyzer.detect_attack_windows(clustering_states)

        # 4. Detect goalkeeper dives
        gk_dives = self._gk_analyzer.detect_goalkeeper_dives(player_tracks, self.fps)
        gk_dive_frames = [d.frame_idx for d in gk_dives]

        # 5. First pass fusion to get initial shot candidates (for celebration detection)
        initial_candidates = self._fusion_engine.fuse_signals(
            kick_events, goal_entries, gk_dive_frames, attack_windows
        )

        # 6. Detect celebrations after initial shot candidates
        celebration_events = []
        if self._celebration_detector and initial_candidates:
            celebration_events = self._celebration_detector.detect_celebrations(
                player_tracks, initial_candidates, self.fps
            )

        # 7. Final fusion with celebration events
        shot_candidates = self._fusion_engine.fuse_signals(
            kick_events, goal_entries, gk_dive_frames, attack_windows,
            celebration_events=celebration_events if celebration_events else None,
        )

        # 8. Convert candidates to Event objects
        events = []
        ball_coverage = self._compute_ball_coverage(ball_tracks)

        for candidate in shot_candidates:
            # Apply sparse data penalty
            confidence = candidate.confidence
            if ball_coverage < 0.1:
                confidence *= 0.7  # 30% penalty for very sparse data

            # Build metadata
            metadata = {
                "detection_method": "alternative",
                "signals_present": candidate.signals_present,
                "attack_score": candidate.attack_score,
                "ball_coverage": ball_coverage,
            }

            if candidate.kick_event:
                metadata["kick_player_id"] = candidate.kick_event.player_track_id
                metadata["kick_confidence"] = candidate.kick_event.confidence

            if candidate.goal_entry:
                metadata["target_goal"] = candidate.goal_entry.goal_region
                metadata["goal_entry_confidence"] = candidate.goal_entry.confidence

            if candidate.gk_dive_frame:
                metadata["gk_dive_frame"] = candidate.gk_dive_frame

            if candidate.celebration_event:
                metadata["celebration_type"] = candidate.celebration_event.celebration_type
                metadata["celebration_confidence"] = candidate.celebration_event.confidence
                metadata["celebration_players"] = candidate.celebration_event.participating_track_ids

            # Get location from kick or goal entry
            location = None
            if candidate.kick_event:
                location = candidate.kick_event.ball_position
            elif candidate.goal_entry:
                location = candidate.goal_entry.entry_position

            event = Event(
                event_type="shot",
                frame_idx=candidate.frame_idx,
                timestamp=candidate.timestamp,
                confidence=confidence,
                location=location,
                metadata=metadata,
            )
            events.append(event)

        return events

    def _merge_shot_detections(
        self,
        velocity_shots: list[Event],
        alternative_shots: list[Event],
        time_window: float = 3.0,
    ) -> list[Event]:
        """
        Merge velocity-based and alternative shot detections.

        Removes duplicates, preferring velocity-based when both detect same shot.

        Args:
            velocity_shots: Shots from velocity-based detection
            alternative_shots: Shots from alternative detection
            time_window: Time window (seconds) to consider events as duplicates

        Returns:
            Merged and deduplicated shot events
        """
        if not velocity_shots:
            return alternative_shots
        if not alternative_shots:
            return velocity_shots

        # Combine all shots
        all_shots = velocity_shots + alternative_shots
        all_shots.sort(key=lambda e: e.timestamp)

        merged = []
        used_indices: set[int] = set()

        for i, shot in enumerate(all_shots):
            if i in used_indices:
                continue

            # Find overlapping shots
            best_shot = shot
            for j in range(i + 1, len(all_shots)):
                if j in used_indices:
                    continue

                other = all_shots[j]
                if other.timestamp - shot.timestamp > time_window:
                    break

                # Prefer velocity-based detection (more reliable when available)
                shot_method = shot.metadata.get("detection_method", "velocity") if shot.metadata else "velocity"
                other_method = other.metadata.get("detection_method", "velocity") if other.metadata else "velocity"

                if shot_method == "velocity" and other_method == "alternative":
                    # Keep velocity-based, mark alternative as used
                    used_indices.add(j)
                elif shot_method == "alternative" and other_method == "velocity":
                    # Switch to velocity-based
                    best_shot = other
                    used_indices.add(i)
                    used_indices.add(j)
                else:
                    # Same method, keep higher confidence
                    if other.confidence > best_shot.confidence:
                        best_shot = other
                    used_indices.add(j)

            used_indices.add(i)
            merged.append(best_shot)

        return merged
