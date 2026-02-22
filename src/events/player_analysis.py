"""Player analysis for alternative shot detection."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.config.schemas import AlternativeShotDetectionConfig
    from src.vision.field.goal_detector import GoalRegionProvider


@dataclass
class ClusteringState:
    """Per-frame clustering metrics for players."""

    frame_idx: int
    team_centroids: dict[str, tuple[float, float]]  # team_id -> (x, y) centroid
    team_spreads: dict[str, float]  # team_id -> spread (std dev of positions)
    attack_score: float  # How "attacking" the formation is (0-1)


@dataclass
class GoalkeeperDiveEvent:
    """A detected goalkeeper dive."""

    frame_idx: int
    timestamp: float
    track_id: int
    dive_direction: str  # "left" or "right"
    displacement: float  # Horizontal displacement in pixels
    aspect_ratio_change: float  # Change in bbox aspect ratio
    confidence: float


class PlayerClusteringAnalyzer:
    """Analyze player clustering to identify attacking formations."""

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        config: "AlternativeShotDetectionConfig",
    ):
        """
        Initialize player clustering analyzer.

        Args:
            frame_width: Video frame width
            frame_height: Video frame height
            config: Alternative shot detection configuration
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.config = config

    def analyze_clustering(
        self,
        player_tracks: list[dict],
    ) -> dict[int, ClusteringState]:
        """
        Analyze player clustering for each frame.

        Args:
            player_tracks: List of player track dicts with bbox, frame_idx, team_id

        Returns:
            Dict mapping frame_idx to ClusteringState
        """
        # Group tracks by frame
        tracks_by_frame: dict[int, list[dict]] = {}
        for track in player_tracks:
            frame_idx = track["frame_idx"]
            if frame_idx not in tracks_by_frame:
                tracks_by_frame[frame_idx] = []
            tracks_by_frame[frame_idx].append(track)

        states: dict[int, ClusteringState] = {}

        for frame_idx, frame_tracks in tracks_by_frame.items():
            state = self._analyze_frame(frame_idx, frame_tracks)
            if state is not None:
                states[frame_idx] = state

        return states

    def _analyze_frame(
        self,
        frame_idx: int,
        frame_tracks: list[dict],
    ) -> ClusteringState | None:
        """Analyze clustering for a single frame."""
        # Group by team
        teams: dict[str, list[tuple[float, float]]] = {}
        for track in frame_tracks:
            team_id = track.get("team_id", -1)
            if team_id == -1:
                continue

            team_key = str(team_id)
            if team_key not in teams:
                teams[team_key] = []

            bbox = track["bbox"]
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            teams[team_key].append((cx, cy))

        # Need at least some players from at least one team
        if not teams:
            return None

        # Compute centroids and spreads
        team_centroids = {}
        team_spreads = {}

        for team_id, positions in teams.items():
            if len(positions) < self.config.min_players_per_team:
                continue

            positions_arr = np.array(positions)
            centroid = (float(np.mean(positions_arr[:, 0])), float(np.mean(positions_arr[:, 1])))
            spread = float(np.mean(np.std(positions_arr, axis=0)))

            team_centroids[team_id] = centroid
            team_spreads[team_id] = spread

        if not team_centroids:
            return None

        # Compute attack score
        attack_score = self._compute_attack_score(team_centroids, team_spreads)

        return ClusteringState(
            frame_idx=frame_idx,
            team_centroids=team_centroids,
            team_spreads=team_spreads,
            attack_score=attack_score,
        )

    def _compute_attack_score(
        self,
        team_centroids: dict[str, tuple[float, float]],
        team_spreads: dict[str, float],
    ) -> float:
        """
        Compute how "attacking" the formation is.

        High attack score when:
        - One team's centroid is near goal (top or bottom edge)
        - That team has low spread (compact formation)
        - Other team is pushed back
        """
        if len(team_centroids) < 2:
            # Single team, check if near goal
            for team_id, (cx, cy) in team_centroids.items():
                spread = team_spreads.get(team_id, self.frame_height)
                norm_spread = spread / self.frame_height

                # Near top or bottom
                near_goal = cy < self.frame_height * 0.25 or cy > self.frame_height * 0.75
                compact = norm_spread < self.config.attack_spread_threshold

                if near_goal and compact:
                    return 0.6
                elif near_goal:
                    return 0.3

            return 0.0

        # Multiple teams - look for attacking vs defending pattern
        max_attack_score = 0.0

        for attack_team, (ax, ay) in team_centroids.items():
            attack_spread = team_spreads.get(attack_team, self.frame_height)
            norm_attack_spread = attack_spread / self.frame_height

            # Check if this team is in attacking position
            near_top = ay < self.frame_height * 0.30
            near_bottom = ay > self.frame_height * 0.70

            if not (near_top or near_bottom):
                continue

            # Check if spread is compact
            compact = norm_attack_spread < self.config.attack_spread_threshold

            # Check if opponent is pushed back
            opponent_pushed = False
            for defend_team, (dx, dy) in team_centroids.items():
                if defend_team == attack_team:
                    continue
                # Opponent should be in opposite half
                if near_top and dy < self.frame_height * 0.5:
                    opponent_pushed = True
                elif near_bottom and dy > self.frame_height * 0.5:
                    opponent_pushed = True

            # Score components
            score = 0.0
            if near_top or near_bottom:
                score += 0.4
            if compact:
                score += 0.3
            if opponent_pushed:
                score += 0.3

            max_attack_score = max(max_attack_score, score)

        return min(1.0, max_attack_score)

    def detect_attack_windows(
        self,
        clustering_states: dict[int, ClusteringState],
        min_window_frames: int = 15,
        score_threshold: float = 0.4,
    ) -> list[tuple[int, int, float]]:
        """
        Detect windows of attacking formations.

        Args:
            clustering_states: Per-frame clustering states
            min_window_frames: Minimum frames for a valid window
            score_threshold: Minimum attack score

        Returns:
            List of (start_frame, end_frame, avg_score) tuples
        """
        if not clustering_states:
            return []

        sorted_frames = sorted(clustering_states.keys())
        windows = []

        in_window = False
        window_start = 0
        window_scores = []

        for frame_idx in sorted_frames:
            state = clustering_states[frame_idx]

            if state.attack_score >= score_threshold:
                if not in_window:
                    in_window = True
                    window_start = frame_idx
                    window_scores = []
                window_scores.append(state.attack_score)
            else:
                if in_window:
                    # End of window
                    if len(window_scores) >= min_window_frames:
                        avg_score = float(np.mean(window_scores))
                        windows.append((window_start, frame_idx - 1, avg_score))
                    in_window = False

        # Handle window at end
        if in_window and len(window_scores) >= min_window_frames:
            avg_score = float(np.mean(window_scores))
            windows.append((window_start, sorted_frames[-1], avg_score))

        return windows


class GoalkeeperAnalyzer:
    """Detect goalkeeper dives from player tracks."""

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        config: "AlternativeShotDetectionConfig",
        goal_region_provider: "GoalRegionProvider | None" = None,
    ):
        """
        Initialize goalkeeper analyzer.

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
        """Compute regions where goalkeepers are expected (fallback when no provider)."""
        margin = self.config.gk_region_margin

        return {
            "top": {
                "x_min": 0,
                "x_max": self.frame_width,
                "y_min": 0,
                "y_max": self.frame_height * margin,
            },
            "bottom": {
                "x_min": 0,
                "x_max": self.frame_width,
                "y_min": self.frame_height * (1 - margin),
                "y_max": self.frame_height,
            },
        }

    def _get_goal_regions(self, frame_idx: int = 0) -> dict[str, dict]:
        """Get goal regions for a frame, using provider if available."""
        if self._goal_region_provider is not None:
            regions = self._goal_region_provider.get_goal_regions(frame_idx)
            # For GK analysis, expand the regions horizontally (GK can be anywhere in goal area)
            result = {}
            for r in regions:
                result[r.name] = {
                    "x_min": 0,  # GK can be anywhere horizontally
                    "x_max": self.frame_width,
                    "y_min": r.bounds["y_min"],
                    "y_max": r.bounds["y_max"],
                }
            return result
        return self._static_goal_regions

    def detect_goalkeeper_dives(
        self,
        player_tracks: list[dict],
        fps: float = 30.0,
    ) -> list[GoalkeeperDiveEvent]:
        """
        Detect goalkeeper dives from player tracks.

        A dive is detected when:
        1. Player is near goal region (likely goalkeeper)
        2. Significant horizontal displacement
        3. Aspect ratio change (standing -> horizontal)

        Args:
            player_tracks: List of player track dicts
            fps: Video frames per second

        Returns:
            List of goalkeeper dive events
        """
        if not player_tracks:
            return []

        # Group tracks by track_id
        tracks_by_id: dict[int, list[dict]] = {}
        for track in player_tracks:
            track_id = track["track_id"]
            if track_id not in tracks_by_id:
                tracks_by_id[track_id] = []
            tracks_by_id[track_id].append(track)

        # Sort each track by frame
        for track_id in tracks_by_id:
            tracks_by_id[track_id].sort(key=lambda t: t["frame_idx"])

        dive_events = []

        for track_id, track_history in tracks_by_id.items():
            # Check if this track is in goalkeeper region
            gk_region = self._get_goalkeeper_region(track_history)
            if gk_region is None:
                continue

            # Look for dive patterns
            dives = self._detect_dives_in_track(track_id, track_history, gk_region, fps)
            dive_events.extend(dives)

        return dive_events

    def _get_goalkeeper_region(self, track_history: list[dict]) -> str | None:
        """
        Determine if this track is a goalkeeper based on position.

        Returns "top" or "bottom" if goalkeeper, None otherwise.
        """
        if len(track_history) < 10:
            return None

        # Check where this player spends most time
        top_count = 0
        bottom_count = 0

        for track in track_history:
            bbox = track["bbox"]
            cy = (bbox[1] + bbox[3]) / 2
            frame_idx = track.get("frame_idx", 0)

            # Get goal regions for this frame
            goal_regions = self._get_goal_regions(frame_idx)

            if "top" in goal_regions:
                if goal_regions["top"]["y_min"] <= cy <= goal_regions["top"]["y_max"]:
                    top_count += 1
            if "bottom" in goal_regions:
                if goal_regions["bottom"]["y_min"] <= cy <= goal_regions["bottom"]["y_max"]:
                    bottom_count += 1

        total = len(track_history)
        if top_count > total * 0.5:
            return "top"
        elif bottom_count > total * 0.5:
            return "bottom"

        return None

    def _detect_dives_in_track(
        self,
        track_id: int,
        track_history: list[dict],
        gk_region: str,
        fps: float,
    ) -> list[GoalkeeperDiveEvent]:
        """Detect dive events within a single goalkeeper track."""
        dives = []
        window_size = 10  # Frames to compare for dive detection

        if len(track_history) < window_size:
            return []

        for i in range(len(track_history) - window_size):
            start_track = track_history[i]
            end_track = track_history[i + window_size]

            # Compute displacement
            start_bbox = start_track["bbox"]
            end_bbox = end_track["bbox"]

            start_cx = (start_bbox[0] + start_bbox[2]) / 2
            end_cx = (end_bbox[0] + end_bbox[2]) / 2
            displacement = abs(end_cx - start_cx)

            # Compute aspect ratio change
            start_width = start_bbox[2] - start_bbox[0]
            start_height = start_bbox[3] - start_bbox[1]
            end_width = end_bbox[2] - end_bbox[0]
            end_height = end_bbox[3] - end_bbox[1]

            # Avoid division by zero
            if start_height < 1 or end_height < 1 or start_width < 1 or end_width < 1:
                continue

            start_aspect = start_width / start_height
            end_aspect = end_width / end_height

            # Aspect ratio change (standing is tall/narrow, diving is short/wide)
            aspect_change = end_aspect / start_aspect if start_aspect > 0 else 1.0

            # Check for dive
            if (
                displacement >= self.config.gk_dive_displacement
                and aspect_change >= self.config.gk_aspect_change_threshold
            ):
                # Determine dive direction
                direction = "right" if end_cx > start_cx else "left"

                # Compute confidence based on how strong the dive indicators are
                disp_score = min(1.0, displacement / (self.config.gk_dive_displacement * 2))
                aspect_score = min(1.0, (aspect_change - 1) / (self.config.gk_aspect_change_threshold - 1))
                confidence = (disp_score + aspect_score) / 2

                dive = GoalkeeperDiveEvent(
                    frame_idx=end_track["frame_idx"],
                    timestamp=end_track.get("timestamp", end_track["frame_idx"] / fps),
                    track_id=track_id,
                    dive_direction=direction,
                    displacement=displacement,
                    aspect_ratio_change=aspect_change,
                    confidence=confidence,
                )
                dives.append(dive)

                # Skip ahead to avoid detecting same dive multiple times
                i += window_size

        # Deduplicate nearby dives
        return self._deduplicate_dives(dives, fps)

    def _deduplicate_dives(
        self,
        dives: list[GoalkeeperDiveEvent],
        fps: float,
        time_window: float = 1.0,
    ) -> list[GoalkeeperDiveEvent]:
        """Remove duplicate dive detections within time window."""
        if not dives:
            return []

        dives = sorted(dives, key=lambda d: d.frame_idx)
        deduplicated = []

        i = 0
        while i < len(dives):
            best_dive = dives[i]
            j = i + 1

            while j < len(dives):
                time_diff = (dives[j].frame_idx - dives[i].frame_idx) / fps
                if time_diff <= time_window:
                    if dives[j].confidence > best_dive.confidence:
                        best_dive = dives[j]
                    j += 1
                else:
                    break

            deduplicated.append(best_dive)
            i = j

        return deduplicated
