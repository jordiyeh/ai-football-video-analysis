"""Kick event detection for alternative shot detection with sparse ball data."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.config.schemas import AlternativeShotDetectionConfig
    from src.events.celebration_detection import CelebrationEvent
    from src.vision.field.goal_detector import GoalRegionProvider


@dataclass
class KickEvent:
    """A detected kick event where ball was near a player's foot."""

    frame_idx: int
    timestamp: float
    player_track_id: int
    player_bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    ball_position: tuple[float, float]  # x, y center
    confidence: float
    evidence: dict = field(default_factory=dict)


@dataclass
class GoalAreaEntryEvent:
    """Ball entering a goal area."""

    frame_idx: int
    timestamp: float
    goal_region: str  # "top" or "bottom"
    entry_position: tuple[float, float]
    confidence: float
    associated_kick: KickEvent | None = None


@dataclass
class ShotCandidate:
    """Composite shot candidate from multiple signals."""

    frame_idx: int
    timestamp: float
    confidence: float
    kick_event: KickEvent | None = None
    goal_entry: GoalAreaEntryEvent | None = None
    gk_dive_frame: int | None = None
    attack_score: float = 0.0
    signals_present: list[str] = field(default_factory=list)
    detection_method: str = "alternative"
    celebration_event: "CelebrationEvent | None" = None


class KickEventDetector:
    """Detect kicks by finding ball detections near player foot regions."""

    def __init__(self, config: "AlternativeShotDetectionConfig"):
        """
        Initialize kick event detector.

        Args:
            config: Alternative shot detection configuration
        """
        self.config = config

    def detect_kicks(
        self,
        player_tracks: list[dict],
        ball_tracks: list[dict],
        frame_width: int,
        frame_height: int,
    ) -> list[KickEvent]:
        """
        Detect kick events from player and ball tracks.

        A kick is detected when:
        1. Ball appears near a player's foot region (bottom of bbox)
        2. Ball wasn't there in previous frames (lookback)
        3. Ball moves away in subsequent frames (lookahead)

        Args:
            player_tracks: List of player track dicts with bbox, frame_idx, track_id
            ball_tracks: List of ball track dicts with bbox, frame_idx
            frame_width: Video frame width
            frame_height: Video frame height

        Returns:
            List of detected kick events
        """
        if not ball_tracks or not player_tracks:
            return []

        # Index player tracks by frame
        players_by_frame: dict[int, list[dict]] = {}
        for track in player_tracks:
            frame_idx = track["frame_idx"]
            if frame_idx not in players_by_frame:
                players_by_frame[frame_idx] = []
            players_by_frame[frame_idx].append(track)

        # Index ball tracks by frame
        ball_by_frame: dict[int, dict] = {}
        for track in ball_tracks:
            frame_idx = track["frame_idx"]
            # Keep highest confidence ball if multiple
            if frame_idx not in ball_by_frame or track["confidence"] > ball_by_frame[frame_idx]["confidence"]:
                ball_by_frame[frame_idx] = track

        kick_events = []
        processed_frames: set[int] = set()

        # Check each frame with a ball detection
        for frame_idx in sorted(ball_by_frame.keys()):
            if frame_idx in processed_frames:
                continue

            ball = ball_by_frame[frame_idx]
            ball_center = self._get_ball_center(ball)

            # Get players in this frame
            frame_players = players_by_frame.get(frame_idx, [])
            if not frame_players:
                continue

            # Find nearest player with ball in foot region
            for player in frame_players:
                if self._is_ball_in_foot_region(ball_center, player):
                    # Check lookback: ball shouldn't have been near this player
                    was_nearby_before = self._check_ball_nearby_before(
                        player["track_id"],
                        frame_idx,
                        players_by_frame,
                        ball_by_frame,
                    )

                    # Check lookahead: ball should move away
                    moves_away_after = self._check_ball_moves_away(
                        ball_center,
                        frame_idx,
                        ball_by_frame,
                    )

                    # Compute confidence
                    base_confidence = ball["confidence"]
                    if not was_nearby_before:
                        base_confidence *= 1.2  # Boost for "new" ball contact
                    if moves_away_after:
                        base_confidence *= 1.2  # Boost for ball leaving

                    confidence = min(1.0, base_confidence)

                    if confidence >= 0.2:  # Minimum threshold for kick event
                        kick = KickEvent(
                            frame_idx=frame_idx,
                            timestamp=ball.get("timestamp", frame_idx / 30.0),
                            player_track_id=player["track_id"],
                            player_bbox=tuple(player["bbox"]),
                            ball_position=ball_center,
                            confidence=confidence,
                            evidence={
                                "ball_confidence": ball["confidence"],
                                "was_nearby_before": was_nearby_before,
                                "moves_away_after": moves_away_after,
                            },
                        )
                        kick_events.append(kick)

                        # Mark nearby frames as processed to avoid duplicates
                        for df in range(-2, 3):
                            processed_frames.add(frame_idx + df)

                        break  # One kick per frame

        return kick_events

    def _get_ball_center(self, ball_track: dict) -> tuple[float, float]:
        """Get center position of ball from track dict."""
        bbox = ball_track["bbox"]
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        return (cx, cy)

    def _is_ball_in_foot_region(
        self,
        ball_center: tuple[float, float],
        player: dict,
    ) -> bool:
        """
        Check if ball is in player's foot region.

        Foot region is the bottom portion of player bbox.
        """
        bbox = player["bbox"]
        x1, y1, x2, y2 = bbox

        # Foot region: bottom fraction of bbox
        foot_top = y2 - (y2 - y1) * self.config.foot_region_fraction
        foot_region = (x1, foot_top, x2, y2)

        # Expand by proximity threshold
        threshold = self.config.kick_proximity_threshold
        expanded = (
            foot_region[0] - threshold,
            foot_region[1] - threshold,
            foot_region[2] + threshold,
            foot_region[3] + threshold,
        )

        bx, by = ball_center
        return (
            expanded[0] <= bx <= expanded[2]
            and expanded[1] <= by <= expanded[3]
        )

    def _check_ball_nearby_before(
        self,
        player_track_id: int,
        frame_idx: int,
        players_by_frame: dict[int, list[dict]],
        ball_by_frame: dict[int, dict],
    ) -> bool:
        """Check if ball was near this player in previous frames."""
        for df in range(1, self.config.kick_lookback_frames + 1):
            prev_frame = frame_idx - df
            if prev_frame not in ball_by_frame:
                continue

            ball = ball_by_frame[prev_frame]
            ball_center = self._get_ball_center(ball)

            # Find this player in previous frame
            for player in players_by_frame.get(prev_frame, []):
                if player["track_id"] == player_track_id:
                    if self._is_ball_in_foot_region(ball_center, player):
                        return True

        return False

    def _check_ball_moves_away(
        self,
        kick_position: tuple[float, float],
        frame_idx: int,
        ball_by_frame: dict[int, dict],
    ) -> bool:
        """Check if ball moves away from kick position in subsequent frames."""
        distances = []

        for df in range(1, self.config.kick_lookahead_frames + 1):
            next_frame = frame_idx + df
            if next_frame not in ball_by_frame:
                continue

            ball = ball_by_frame[next_frame]
            ball_center = self._get_ball_center(ball)

            dist = np.sqrt(
                (ball_center[0] - kick_position[0]) ** 2
                + (ball_center[1] - kick_position[1]) ** 2
            )
            distances.append(dist)

        if len(distances) < 2:
            return True  # Assume it moved (sparse data)

        # Check if distances are increasing (ball moving away)
        return distances[-1] > distances[0] + 20  # At least 20px increase


class GoalAreaEntryDetector:
    """Detect ball entering goal areas."""

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        config: "AlternativeShotDetectionConfig",
        goal_region_provider: "GoalRegionProvider | None" = None,
    ):
        """
        Initialize goal area entry detector.

        Args:
            frame_width: Video frame width
            frame_height: Video frame height
            config: Alternative shot detection configuration
            goal_region_provider: Optional provider for goal regions
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.config = config
        self._goal_region_provider = goal_region_provider

        # Only compute static regions if no provider given
        if goal_region_provider is None:
            self._static_goal_regions = self._compute_goal_regions()
        else:
            self._static_goal_regions = None

    def _compute_goal_regions(self) -> dict[str, dict]:
        """Compute goal region bounds (fallback when no provider)."""
        edge_margin = 0.15 + self.config.goal_entry_margin
        goal_width_fraction = 0.3

        x_center = self.frame_width / 2
        goal_half_width = (self.frame_width * goal_width_fraction) / 2

        return {
            "top": {
                "x_min": x_center - goal_half_width,
                "x_max": x_center + goal_half_width,
                "y_min": 0,
                "y_max": self.frame_height * edge_margin,
            },
            "bottom": {
                "x_min": x_center - goal_half_width,
                "x_max": x_center + goal_half_width,
                "y_min": self.frame_height * (1 - edge_margin),
                "y_max": self.frame_height,
            },
        }

    def _get_goal_regions(self, frame_idx: int) -> dict[str, dict]:
        """Get goal regions for a frame, using provider if available."""
        if self._goal_region_provider is not None:
            regions = self._goal_region_provider.get_goal_regions(frame_idx)
            return {r.name: dict(r.bounds) for r in regions}
        return self._static_goal_regions

    def detect_goal_entries(
        self,
        ball_tracks: list[dict],
        kick_events: list[KickEvent],
        fps: float = 30.0,
    ) -> list[GoalAreaEntryEvent]:
        """
        Detect ball entries into goal areas.

        Args:
            ball_tracks: List of ball track dicts
            kick_events: Detected kick events to associate
            fps: Video frames per second

        Returns:
            List of goal area entry events
        """
        if not ball_tracks:
            return []

        # Sort ball tracks by frame
        sorted_tracks = sorted(ball_tracks, key=lambda t: t["frame_idx"])

        entries = []
        in_goal_region: dict[str, int | None] = {"top": None, "bottom": None}

        for track in sorted_tracks:
            bbox = track["bbox"]
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            frame_idx = track["frame_idx"]

            # Get goal regions for this frame (may vary with visual detection)
            goal_regions = self._get_goal_regions(frame_idx)
            for goal_name, bounds in goal_regions.items():
                is_in_goal = (
                    bounds["x_min"] <= cx <= bounds["x_max"]
                    and bounds["y_min"] <= cy <= bounds["y_max"]
                )

                if is_in_goal and in_goal_region[goal_name] is None:
                    # New entry into goal region
                    in_goal_region[goal_name] = frame_idx

                    # Find associated kick
                    associated_kick = self._find_associated_kick(
                        frame_idx, kick_events, fps
                    )

                    confidence = track["confidence"]
                    if associated_kick:
                        # Boost confidence if we have a kick
                        confidence = min(1.0, confidence * 1.3)

                    entry = GoalAreaEntryEvent(
                        frame_idx=frame_idx,
                        timestamp=track.get("timestamp", frame_idx / fps),
                        goal_region=goal_name,
                        entry_position=(cx, cy),
                        confidence=confidence,
                        associated_kick=associated_kick,
                    )
                    entries.append(entry)

                elif not is_in_goal and in_goal_region[goal_name] is not None:
                    # Exited goal region
                    in_goal_region[goal_name] = None

        return entries

    def _find_associated_kick(
        self,
        entry_frame: int,
        kick_events: list[KickEvent],
        fps: float,
    ) -> KickEvent | None:
        """Find the most recent kick that could have caused this goal entry."""
        max_frames = self.config.max_kick_association_frames
        best_kick = None
        best_frame_diff = float("inf")

        for kick in kick_events:
            frame_diff = entry_frame - kick.frame_idx
            if 0 < frame_diff <= max_frames and frame_diff < best_frame_diff:
                best_kick = kick
                best_frame_diff = frame_diff

        return best_kick


class ShotFusionEngine:
    """Fuse multiple signals to detect shots."""

    def __init__(self, config: "AlternativeShotDetectionConfig", fps: float = 30.0):
        """
        Initialize shot fusion engine.

        Args:
            config: Alternative shot detection configuration
            fps: Video frames per second
        """
        self.config = config
        self.fps = fps

    def fuse_signals(
        self,
        kick_events: list[KickEvent],
        goal_entries: list[GoalAreaEntryEvent],
        gk_dive_frames: list[int],
        attack_windows: list[tuple[int, int, float]],  # (start, end, avg_score)
        celebration_events: "list[CelebrationEvent] | None" = None,
    ) -> list[ShotCandidate]:
        """
        Fuse multiple signals to detect shot candidates.

        Args:
            kick_events: Detected kick events
            goal_entries: Detected goal area entries
            gk_dive_frames: Frames where goalkeeper dives detected
            attack_windows: Windows of attacking formation (start, end, score)
            celebration_events: Detected celebration events (optional)

        Returns:
            List of shot candidates above confidence threshold
        """
        candidates: list[ShotCandidate] = []
        temporal_window = self.config.fusion_temporal_window

        # Create time-indexed events
        all_events: list[tuple[int, str, object]] = []

        for kick in kick_events:
            all_events.append((kick.frame_idx, "kick", kick))

        for entry in goal_entries:
            all_events.append((entry.frame_idx, "goal_entry", entry))

        for gk_frame in gk_dive_frames:
            all_events.append((gk_frame, "gk_dive", gk_frame))

        if not all_events:
            return []

        # Sort by frame
        all_events.sort(key=lambda x: x[0])

        # Index celebration events by frame for quick lookup
        celebrations_by_frame: dict[int, list] = {}
        if celebration_events:
            for cel in celebration_events:
                if cel.frame_idx not in celebrations_by_frame:
                    celebrations_by_frame[cel.frame_idx] = []
                celebrations_by_frame[cel.frame_idx].append(cel)

        # Group events within temporal window
        processed: set[int] = set()

        for i, (frame_idx, event_type, event) in enumerate(all_events):
            if i in processed:
                continue

            # Collect all events within window of this event
            window_start = frame_idx
            window_end = frame_idx + temporal_window

            window_events = {"kicks": [], "goal_entries": [], "gk_dives": [], "celebrations": []}

            for j, (other_frame, other_type, other_event) in enumerate(all_events):
                if window_start <= other_frame <= window_end:
                    processed.add(j)
                    if other_type == "kick":
                        window_events["kicks"].append(other_event)
                    elif other_type == "goal_entry":
                        window_events["goal_entries"].append(other_event)
                    elif other_type == "gk_dive":
                        window_events["gk_dives"].append(other_event)

            # Find celebrations within post-shot window (celebrations happen AFTER shot)
            celebration_window_end = frame_idx + self.config.celebration.post_shot_window_frames if hasattr(self.config, 'celebration') else window_end
            for cel_frame in celebrations_by_frame:
                if window_start <= cel_frame <= celebration_window_end:
                    window_events["celebrations"].extend(celebrations_by_frame[cel_frame])

            # Compute attack score for this window
            attack_score = self._get_attack_score(frame_idx, attack_windows)

            # Fuse signals
            candidate = self._create_candidate(
                frame_idx,
                window_events,
                attack_score,
            )

            if candidate.confidence >= self.config.fusion_min_confidence:
                candidates.append(candidate)

        return candidates

    def _get_attack_score(
        self,
        frame_idx: int,
        attack_windows: list[tuple[int, int, float]],
    ) -> float:
        """Get attack formation score for a given frame."""
        for start, end, score in attack_windows:
            if start <= frame_idx <= end:
                return score
        return 0.0

    def _create_candidate(
        self,
        anchor_frame: int,
        window_events: dict,
        attack_score: float,
    ) -> ShotCandidate:
        """Create a shot candidate from collected signals."""
        signals_present = []
        weighted_sum = 0.0
        present_weights = 0.0  # Sum of weights for present signals only

        # Check if celebration is enabled
        celebration_enabled = (
            hasattr(self.config, 'celebration')
            and self.config.celebration.enabled
        )

        # Calculate adjusted weights when celebration is enabled
        # Rebalance: kick 0.30, goal_entry 0.25, gk_dive 0.20, attack 0.10, celebration 0.15
        if celebration_enabled:
            kick_weight = 0.30
            goal_entry_weight = 0.25
            gk_dive_weight = 0.20
            attack_weight = self.config.attack_context_weight  # Keep at 0.10
            celebration_weight = self.config.celebration.signal_weight
        else:
            kick_weight = self.config.kick_weight
            goal_entry_weight = self.config.goal_entry_weight
            gk_dive_weight = self.config.gk_dive_weight
            attack_weight = self.config.attack_context_weight
            celebration_weight = 0.0

        # Best kick in window
        kick_event = None
        if window_events["kicks"]:
            kick_event = max(window_events["kicks"], key=lambda k: k.confidence)
            signals_present.append("kick")
            weighted_sum += kick_weight * kick_event.confidence
            present_weights += kick_weight

        # Best goal entry
        goal_entry = None
        if window_events["goal_entries"]:
            goal_entry = max(window_events["goal_entries"], key=lambda e: e.confidence)
            signals_present.append("goal_entry")
            weighted_sum += goal_entry_weight * goal_entry.confidence
            present_weights += goal_entry_weight

        # GK dive
        gk_dive_frame = None
        if window_events["gk_dives"]:
            gk_dive_frame = window_events["gk_dives"][0]
            signals_present.append("gk_dive")
            weighted_sum += gk_dive_weight * 0.8  # GK dive confidence fixed at 0.8
            present_weights += gk_dive_weight

        # Attack context
        if attack_score > 0:
            signals_present.append("attack_context")
            weighted_sum += attack_weight * attack_score
            present_weights += attack_weight

        # Celebration signal (new)
        celebration_event = None
        if celebration_enabled and window_events.get("celebrations"):
            celebration_event = max(
                window_events["celebrations"],
                key=lambda c: c.confidence
            )
            signals_present.append("celebration")
            weighted_sum += celebration_weight * celebration_event.confidence
            present_weights += celebration_weight

        # Normalize confidence by weights of present signals only
        # This ensures a single high-confidence signal can produce a high-confidence candidate
        if present_weights > 0:
            confidence = weighted_sum / present_weights
        else:
            confidence = 0.0

        # Bonus for multiple corroborating signals
        if len(signals_present) >= 3:
            confidence = min(1.0, confidence * 1.2)
        elif len(signals_present) >= 2:
            confidence = min(1.0, confidence * 1.1)

        # Penalty for single weak signal (less certainty without corroboration)
        if len(signals_present) == 1:
            confidence *= 0.9  # 10% penalty for single signal

        # Use kick frame as anchor if available, else goal entry
        if kick_event:
            anchor_frame = kick_event.frame_idx
            timestamp = kick_event.timestamp
        elif goal_entry:
            anchor_frame = goal_entry.frame_idx
            timestamp = goal_entry.timestamp
        else:
            timestamp = anchor_frame / self.fps

        return ShotCandidate(
            frame_idx=anchor_frame,
            timestamp=timestamp,
            confidence=confidence,
            kick_event=kick_event,
            goal_entry=goal_entry,
            gk_dive_frame=gk_dive_frame,
            attack_score=attack_score,
            signals_present=signals_present,
            detection_method="alternative",
            celebration_event=celebration_event,
        )
