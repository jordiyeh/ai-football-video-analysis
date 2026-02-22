"""Celebration detection for goal confirmation."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.config.schemas import CelebrationConfig
    from src.events.kick_detection import ShotCandidate


@dataclass
class CelebrationEvent:
    """A detected celebration event."""

    frame_idx: int
    timestamp: float
    confidence: float
    celebration_type: str  # "individual_arms_up", "group_huddle", "running"
    participating_track_ids: list[int]
    team_id: int | None
    center_position: tuple[float, float]
    evidence: dict = field(default_factory=dict)


class CelebrationDetector:
    """Detect player celebrations as goal confirmation signal."""

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        config: "CelebrationConfig",
    ):
        """
        Initialize celebration detector.

        Args:
            frame_width: Video frame width
            frame_height: Video frame height
            config: Celebration detection configuration
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.config = config

        # Track baselines for aspect ratio comparison
        self._track_baselines: dict[int, dict] = {}

    def detect_celebrations(
        self,
        player_tracks: list[dict],
        shot_candidates: list["ShotCandidate"],
        fps: float = 30.0,
    ) -> list[CelebrationEvent]:
        """
        Detect celebrations following shot candidates.

        Args:
            player_tracks: List of player track dicts with bbox, frame_idx, track_id
            shot_candidates: List of shot candidates to check for celebrations after
            fps: Video frames per second

        Returns:
            List of detected celebration events
        """
        if not player_tracks or not shot_candidates:
            return []

        # Index player tracks by track_id and frame
        tracks_by_id: dict[int, list[dict]] = {}
        for track in player_tracks:
            track_id = track["track_id"]
            if track_id not in tracks_by_id:
                tracks_by_id[track_id] = []
            tracks_by_id[track_id].append(track)

        # Sort each track by frame
        for track_id in tracks_by_id:
            tracks_by_id[track_id].sort(key=lambda t: t["frame_idx"])

        # Compute baselines for all tracks
        self._compute_track_baselines(tracks_by_id)

        # Index tracks by frame for huddle detection
        tracks_by_frame: dict[int, list[dict]] = {}
        for track in player_tracks:
            frame_idx = track["frame_idx"]
            if frame_idx not in tracks_by_frame:
                tracks_by_frame[frame_idx] = []
            tracks_by_frame[frame_idx].append(track)

        all_celebrations: list[CelebrationEvent] = []

        # Look for celebrations after each shot candidate
        for candidate in shot_candidates:
            window_start = candidate.frame_idx
            window_end = candidate.frame_idx + self.config.post_shot_window_frames

            # Detect arms-up celebrations
            arms_up = self._detect_arms_up(
                tracks_by_id, window_start, window_end, fps
            )
            all_celebrations.extend(arms_up)

            # Detect group huddles
            huddles = self._detect_group_huddle(
                tracks_by_frame, window_start, window_end, fps
            )
            all_celebrations.extend(huddles)

        # Deduplicate and apply cooldown
        celebrations = self._deduplicate_celebrations(all_celebrations, fps)

        # Filter by minimum confidence
        celebrations = [
            c for c in celebrations
            if c.confidence >= self.config.min_confidence
        ]

        return celebrations

    def _compute_track_baselines(self, tracks_by_id: dict[int, list[dict]]) -> None:
        """
        Compute baseline aspect ratios for each track.

        Uses early frames to establish normal standing pose.
        """
        self._track_baselines = {}

        for track_id, track_history in tracks_by_id.items():
            if len(track_history) < 10:
                continue

            # Use first 10-30 frames to compute baseline
            baseline_tracks = track_history[:min(30, len(track_history))]

            aspect_ratios = []
            heights = []
            for track in baseline_tracks:
                bbox = track["bbox"]
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                if height > 0 and width > 0:
                    aspect_ratios.append(width / height)
                    heights.append(height)

            if aspect_ratios:
                self._track_baselines[track_id] = {
                    "aspect_ratio": float(np.median(aspect_ratios)),
                    "height": float(np.median(heights)),
                }

    def _detect_arms_up(
        self,
        tracks_by_id: dict[int, list[dict]],
        window_start: int,
        window_end: int,
        fps: float,
    ) -> list[CelebrationEvent]:
        """
        Detect arms-up celebration pose.

        Arms-up is detected when:
        1. Bbox becomes wider (aspect ratio increases)
        2. Height may decrease slightly (arms spread)
        3. Pose is held for min_duration_frames
        """
        celebrations = []

        for track_id, track_history in tracks_by_id.items():
            if track_id not in self._track_baselines:
                continue

            baseline = self._track_baselines[track_id]
            baseline_aspect = baseline["aspect_ratio"]

            # Filter tracks within window
            window_tracks = [
                t for t in track_history
                if window_start <= t["frame_idx"] <= window_end
            ]

            if len(window_tracks) < self.config.arms_up_min_duration_frames:
                continue

            # Look for sustained aspect ratio change
            consecutive_frames = 0
            celebration_start_frame = None
            aspect_changes = []

            for track in window_tracks:
                bbox = track["bbox"]
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]

                if height <= 0 or width <= 0:
                    consecutive_frames = 0
                    continue

                current_aspect = width / height
                aspect_change = current_aspect / baseline_aspect if baseline_aspect > 0 else 1.0

                # Arms up detection: aspect ratio increases significantly
                if aspect_change > self.config.arms_up_aspect_ratio_threshold:
                    if consecutive_frames == 0:
                        celebration_start_frame = track["frame_idx"]
                    consecutive_frames += 1
                    aspect_changes.append(aspect_change)
                else:
                    if consecutive_frames >= self.config.arms_up_min_duration_frames:
                        # Valid celebration detected
                        avg_aspect_change = float(np.mean(aspect_changes))
                        confidence = self._compute_arms_up_confidence(
                            avg_aspect_change, consecutive_frames
                        )

                        # Get position
                        mid_idx = len(window_tracks) // 2
                        if mid_idx < len(window_tracks):
                            mid_track = window_tracks[mid_idx]
                            bbox = mid_track["bbox"]
                            center_pos = (
                                (bbox[0] + bbox[2]) / 2,
                                (bbox[1] + bbox[3]) / 2,
                            )
                        else:
                            center_pos = (0.0, 0.0)

                        team_id = window_tracks[0].get("team_id")

                        event = CelebrationEvent(
                            frame_idx=celebration_start_frame,
                            timestamp=celebration_start_frame / fps,
                            confidence=confidence,
                            celebration_type="individual_arms_up",
                            participating_track_ids=[track_id],
                            team_id=team_id,
                            center_position=center_pos,
                            evidence={
                                "avg_aspect_change": avg_aspect_change,
                                "duration_frames": consecutive_frames,
                                "baseline_aspect": baseline_aspect,
                            },
                        )
                        celebrations.append(event)

                    consecutive_frames = 0
                    aspect_changes = []

            # Check if celebration extends to end of window
            if consecutive_frames >= self.config.arms_up_min_duration_frames:
                avg_aspect_change = float(np.mean(aspect_changes))
                confidence = self._compute_arms_up_confidence(
                    avg_aspect_change, consecutive_frames
                )

                mid_idx = len(window_tracks) // 2
                if mid_idx < len(window_tracks):
                    mid_track = window_tracks[mid_idx]
                    bbox = mid_track["bbox"]
                    center_pos = (
                        (bbox[0] + bbox[2]) / 2,
                        (bbox[1] + bbox[3]) / 2,
                    )
                else:
                    center_pos = (0.0, 0.0)

                team_id = window_tracks[0].get("team_id") if window_tracks else None

                event = CelebrationEvent(
                    frame_idx=celebration_start_frame,
                    timestamp=celebration_start_frame / fps,
                    confidence=confidence,
                    celebration_type="individual_arms_up",
                    participating_track_ids=[track_id],
                    team_id=team_id,
                    center_position=center_pos,
                    evidence={
                        "avg_aspect_change": avg_aspect_change,
                        "duration_frames": consecutive_frames,
                        "baseline_aspect": baseline_aspect,
                    },
                )
                celebrations.append(event)

        return celebrations

    def _compute_arms_up_confidence(
        self,
        aspect_change: float,
        duration_frames: int,
    ) -> float:
        """Compute confidence for arms-up detection."""
        # Aspect change contribution (normalized)
        aspect_score = min(1.0, (aspect_change - 1.0) / 0.5)

        # Duration contribution (longer = more confident)
        duration_score = min(1.0, duration_frames / 15)

        # Combined confidence
        confidence = 0.6 * aspect_score + 0.4 * duration_score

        return min(1.0, max(0.0, confidence))

    def _detect_group_huddle(
        self,
        tracks_by_frame: dict[int, list[dict]],
        window_start: int,
        window_end: int,
        fps: float,
    ) -> list[CelebrationEvent]:
        """
        Detect group huddle celebration.

        A huddle is detected when:
        1. Multiple players converge to a small area
        2. They weren't clustered before the shot
        3. At least huddle_min_players are involved
        """
        celebrations = []

        # Compute pre-shot spread (for convergence check)
        pre_shot_frames = range(
            max(0, window_start - 30),
            window_start
        )
        pre_shot_positions_by_team: dict[int | None, list[tuple[float, float]]] = {}

        for frame_idx in pre_shot_frames:
            if frame_idx not in tracks_by_frame:
                continue
            for track in tracks_by_frame[frame_idx]:
                team_id = track.get("team_id")
                bbox = track["bbox"]
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                if team_id not in pre_shot_positions_by_team:
                    pre_shot_positions_by_team[team_id] = []
                pre_shot_positions_by_team[team_id].append((cx, cy))

        pre_shot_spread: dict[int | None, float] = {}
        for team_id, positions in pre_shot_positions_by_team.items():
            if len(positions) >= 3:
                positions_arr = np.array(positions)
                spread = float(np.mean(np.std(positions_arr, axis=0)))
                pre_shot_spread[team_id] = spread

        # Look for huddles in post-shot window
        huddle_detected: dict[int | None, bool] = {}

        for frame_idx in range(window_start, window_end + 1):
            if frame_idx not in tracks_by_frame:
                continue

            frame_tracks = tracks_by_frame[frame_idx]

            # Group by team
            teams: dict[int | None, list[dict]] = {}
            for track in frame_tracks:
                team_id = track.get("team_id")
                if team_id not in teams:
                    teams[team_id] = []
                teams[team_id].append(track)

            # Check each team for huddle
            for team_id, team_tracks in teams.items():
                if len(team_tracks) < self.config.huddle_min_players:
                    continue

                if team_id in huddle_detected and huddle_detected[team_id]:
                    continue  # Already detected huddle for this team

                # Compute pairwise distances
                positions = []
                track_ids = []
                for track in team_tracks:
                    bbox = track["bbox"]
                    cx = (bbox[0] + bbox[2]) / 2
                    cy = (bbox[1] + bbox[3]) / 2
                    positions.append((cx, cy))
                    track_ids.append(track["track_id"])

                # Find cluster of nearby players
                clustered_players = self._find_cluster(
                    positions, track_ids
                )

                if len(clustered_players) >= self.config.huddle_min_players:
                    # Check convergence
                    current_positions = [p for p, _ in clustered_players]
                    current_spread = float(np.mean(np.std(
                        np.array(current_positions), axis=0
                    )))

                    pre_spread = pre_shot_spread.get(team_id, current_spread * 2)

                    if pre_spread > 0:
                        convergence_ratio = pre_spread / max(current_spread, 1.0)
                    else:
                        convergence_ratio = 0.0

                    if convergence_ratio > self.config.huddle_convergence_threshold:
                        # Valid huddle detected
                        cluster_track_ids = [tid for _, tid in clustered_players]
                        cluster_positions = [pos for pos, _ in clustered_players]
                        center = (
                            float(np.mean([p[0] for p in cluster_positions])),
                            float(np.mean([p[1] for p in cluster_positions])),
                        )

                        confidence = self._compute_huddle_confidence(
                            len(clustered_players), convergence_ratio
                        )

                        event = CelebrationEvent(
                            frame_idx=frame_idx,
                            timestamp=frame_idx / fps,
                            confidence=confidence,
                            celebration_type="group_huddle",
                            participating_track_ids=cluster_track_ids,
                            team_id=team_id,
                            center_position=center,
                            evidence={
                                "num_players": len(clustered_players),
                                "convergence_ratio": convergence_ratio,
                                "current_spread": current_spread,
                                "pre_shot_spread": pre_spread,
                            },
                        )
                        celebrations.append(event)
                        huddle_detected[team_id] = True

        return celebrations

    def _find_cluster(
        self,
        positions: list[tuple[float, float]],
        track_ids: list[int],
    ) -> list[tuple[tuple[float, float], int]]:
        """
        Find largest cluster of nearby players.

        Returns list of (position, track_id) tuples for clustered players.
        """
        if len(positions) < 2:
            return []

        max_dist = self.config.huddle_max_player_distance

        # Build adjacency based on distance threshold
        n = len(positions)
        adjacency: list[set[int]] = [set() for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(
                    (positions[i][0] - positions[j][0]) ** 2 +
                    (positions[i][1] - positions[j][1]) ** 2
                )
                if dist < max_dist:
                    adjacency[i].add(j)
                    adjacency[j].add(i)

        # Find connected components (simple BFS)
        visited = [False] * n
        clusters: list[list[int]] = []

        for start in range(n):
            if visited[start]:
                continue

            cluster = []
            queue = [start]
            while queue:
                node = queue.pop(0)
                if visited[node]:
                    continue
                visited[node] = True
                cluster.append(node)
                for neighbor in adjacency[node]:
                    if not visited[neighbor]:
                        queue.append(neighbor)

            if cluster:
                clusters.append(cluster)

        # Return largest cluster
        if not clusters:
            return []

        largest = max(clusters, key=len)
        return [(positions[i], track_ids[i]) for i in largest]

    def _compute_huddle_confidence(
        self,
        num_players: int,
        convergence_ratio: float,
    ) -> float:
        """Compute confidence for huddle detection."""
        # More players = higher confidence
        player_score = min(1.0, num_players / 5)

        # Higher convergence = higher confidence
        convergence_score = min(1.0, convergence_ratio / 2.0)

        # Combined confidence
        confidence = 0.5 * player_score + 0.5 * convergence_score

        return min(1.0, max(0.0, confidence))

    def _deduplicate_celebrations(
        self,
        celebrations: list[CelebrationEvent],
        fps: float,
    ) -> list[CelebrationEvent]:
        """
        Deduplicate celebrations using cooldown.

        Keeps highest confidence celebration within cooldown window.
        """
        if not celebrations:
            return []

        # Sort by frame
        celebrations = sorted(celebrations, key=lambda c: c.frame_idx)

        deduplicated = []
        last_frame: dict[str, int] = {}  # type -> last frame

        for celebration in celebrations:
            ctype = celebration.celebration_type

            # Check cooldown
            if ctype in last_frame:
                frames_since = celebration.frame_idx - last_frame[ctype]
                if frames_since < self.config.celebration_cooldown_frames:
                    # Within cooldown, check if higher confidence
                    for i, existing in enumerate(deduplicated):
                        if existing.celebration_type == ctype:
                            if celebration.confidence > existing.confidence:
                                deduplicated[i] = celebration
                            break
                    continue

            deduplicated.append(celebration)
            last_frame[ctype] = celebration.frame_idx

        return deduplicated
