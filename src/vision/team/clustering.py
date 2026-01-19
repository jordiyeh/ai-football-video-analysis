"""Team clustering and assignment based on jersey colors."""

from collections import defaultdict
from typing import Literal

import numpy as np
from sklearn.cluster import KMeans

from src.vision.team.colors import bgr_to_hsv, color_distance


class TeamAssigner:
    """Assign players to teams based on jersey colors."""

    def __init__(
        self,
        n_teams: int = 2,
        color_space: Literal["bgr", "hsv"] = "hsv",
        min_samples_per_track: int = 5,
    ):
        """
        Initialize team assigner.

        Args:
            n_teams: Number of teams to identify (usually 2, can be 3 with referee)
            color_space: Color space for clustering ("bgr" or "hsv")
            min_samples_per_track: Minimum color samples needed per track
        """
        self.n_teams = n_teams
        self.color_space = color_space
        self.min_samples_per_track = min_samples_per_track

        self.team_colors = None  # Will store cluster centers
        self.team_labels = {}  # track_id -> team_id
        self.track_colors = {}  # track_id -> mean_color

    def fit(self, track_colors: dict[int, list[np.ndarray]]) -> None:
        """
        Fit team clusters from collected track colors.

        Args:
            track_colors: Mapping of track_id to list of color samples [B, G, R]
        """
        # Compute mean color for each track
        track_mean_colors = {}
        for track_id, colors in track_colors.items():
            if len(colors) >= self.min_samples_per_track:
                mean_color = np.mean(colors, axis=0)
                track_mean_colors[track_id] = mean_color

        if len(track_mean_colors) < self.n_teams:
            raise ValueError(
                f"Not enough tracks ({len(track_mean_colors)}) for {self.n_teams} teams"
            )

        # Convert to HSV if needed
        if self.color_space == "hsv":
            colors_for_clustering = np.array([
                bgr_to_hsv(color) for color in track_mean_colors.values()
            ])
        else:
            colors_for_clustering = np.array(list(track_mean_colors.values()))

        # K-means clustering
        kmeans = KMeans(n_clusters=self.n_teams, random_state=42, n_init=10)
        labels = kmeans.fit_predict(colors_for_clustering)

        # Store results
        self.team_colors = kmeans.cluster_centers_
        self.track_colors = track_mean_colors

        # Assign team labels
        for (track_id, _), team_id in zip(track_mean_colors.items(), labels):
            self.team_labels[track_id] = int(team_id)

    def predict(self, color: np.ndarray) -> int:
        """
        Predict team ID for a given color.

        Args:
            color: BGR color [B, G, R]

        Returns:
            Team ID (0, 1, ...)
        """
        if self.team_colors is None:
            raise ValueError("TeamAssigner not fitted. Call fit() first.")

        # Convert to appropriate color space
        if self.color_space == "hsv":
            color_transformed = bgr_to_hsv(color)
        else:
            color_transformed = color

        # Find closest team color
        distances = [
            color_distance(color_transformed, team_color, self.color_space)
            for team_color in self.team_colors
        ]

        return int(np.argmin(distances))

    def get_team_label(self, track_id: int) -> int | None:
        """
        Get team label for a track.

        Args:
            track_id: Track ID

        Returns:
            Team ID (0, 1, ...) or None if unknown
        """
        return self.team_labels.get(track_id)

    def get_team_colors_bgr(self) -> dict[int, np.ndarray]:
        """
        Get team colors in BGR format.

        Returns:
            Mapping of team_id to BGR color
        """
        if self.team_colors is None:
            return {}

        colors = {}
        for team_id, color in enumerate(self.team_colors):
            if self.color_space == "hsv":
                # Convert HSV back to BGR
                hsv_pixel = color.reshape(1, 1, 3).astype(np.uint8)
                import cv2
                bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)
                colors[team_id] = bgr_pixel[0, 0].astype(np.float32)
            else:
                colors[team_id] = color

        return colors

    def assign_team_names(
        self,
        team_names: dict[int, str] | None = None,
        our_team_id: int | None = None,
    ) -> dict[int, str]:
        """
        Assign human-readable names to teams.

        Args:
            team_names: Optional mapping of team_id to name
            our_team_id: Optional ID of "our" team (rest become "opponent")

        Returns:
            Mapping of team_id to name
        """
        if team_names is not None:
            return team_names

        names = {}
        if our_team_id is not None:
            # Binary classification: our team vs opponent(s)
            for team_id in range(self.n_teams):
                if team_id == our_team_id:
                    names[team_id] = "ours"
                else:
                    names[team_id] = "opponent"
        else:
            # Generic names
            team_labels = ["A", "B", "C", "D"]
            for team_id in range(self.n_teams):
                names[team_id] = f"team_{team_labels[team_id]}"

        return names


def collect_track_colors(
    tracks_by_frame: dict[int, list[dict]],
    frames: dict[int, np.ndarray],
    extract_fn,
) -> dict[int, list[np.ndarray]]:
    """
    Collect color samples for each track.

    Args:
        tracks_by_frame: Mapping of frame_idx to list of track dicts
        frames: Mapping of frame_idx to frame image
        extract_fn: Function to extract color from (frame, bbox)

    Returns:
        Mapping of track_id to list of color samples
    """
    track_colors = defaultdict(list)

    for frame_idx, tracks in tracks_by_frame.items():
        if frame_idx not in frames:
            continue

        frame = frames[frame_idx]

        for track in tracks:
            # Only collect colors for players (not ball)
            if track.get("object_type") != "player":
                continue

            track_id = track["track_id"]
            bbox = track["bbox"]

            # Skip tracks with invalid bounding boxes (NaN or inf)
            if any(np.isnan(v) or np.isinf(v) for v in bbox):
                continue

            color = extract_fn(frame, bbox)

            # Only add valid colors (not black/zero which indicates failure)
            if not np.allclose(color, [0, 0, 0]):
                track_colors[track_id].append(color)

    return dict(track_colors)
