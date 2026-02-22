"""Tests for team clustering and assignment."""

import pytest
import numpy as np

from src.vision.team.clustering import TeamAssigner, collect_track_colors


# -----------------------------------------------------------------------------
# TeamAssigner Tests
# -----------------------------------------------------------------------------

class TestTeamAssignerInit:
    """Tests for TeamAssigner initialization."""

    def test_default_init(self):
        """Test default initialization."""
        assigner = TeamAssigner()

        assert assigner.n_teams == 2
        assert assigner.color_space == "hsv"
        assert assigner.min_samples_per_track == 5
        assert assigner.team_colors is None
        assert assigner.team_labels == {}

    def test_custom_init(self):
        """Test initialization with custom parameters."""
        assigner = TeamAssigner(
            n_teams=3,
            color_space="bgr",
            min_samples_per_track=10,
        )

        assert assigner.n_teams == 3
        assert assigner.color_space == "bgr"
        assert assigner.min_samples_per_track == 10


class TestTeamAssignerFit:
    """Tests for TeamAssigner.fit()."""

    @pytest.fixture
    def two_team_colors(self):
        """Create color samples for two distinct teams."""
        # Red team (high red in BGR: [0, 0, 200+])
        red_colors = [
            np.array([30, 30, 200 + i * 5]) for i in range(10)
        ]
        # Blue team (high blue in BGR: [200+, 0, 0])
        blue_colors = [
            np.array([200 + i * 5, 30, 30]) for i in range(10)
        ]

        return {
            1: red_colors,  # Track 1 - red team
            2: red_colors,  # Track 2 - red team
            3: blue_colors,  # Track 3 - blue team
            4: blue_colors,  # Track 4 - blue team
        }

    def test_fit_with_two_teams(self, two_team_colors):
        """Test fitting with two distinct team colors."""
        assigner = TeamAssigner(n_teams=2, color_space="bgr")

        assigner.fit(two_team_colors)

        # Should have team colors
        assert assigner.team_colors is not None
        assert len(assigner.team_colors) == 2

        # Should have labels for all tracks with enough samples
        assert len(assigner.team_labels) == 4

    def test_fit_assigns_same_team_to_similar_colors(self, two_team_colors):
        """Test that similar colors get assigned to same team."""
        assigner = TeamAssigner(n_teams=2, color_space="bgr")

        assigner.fit(two_team_colors)

        # Tracks 1 and 2 (red) should have same team
        assert assigner.team_labels[1] == assigner.team_labels[2]

        # Tracks 3 and 4 (blue) should have same team
        assert assigner.team_labels[3] == assigner.team_labels[4]

        # Red and blue should be different teams
        assert assigner.team_labels[1] != assigner.team_labels[3]

    def test_fit_insufficient_samples(self):
        """Test fit with tracks that have insufficient samples."""
        assigner = TeamAssigner(n_teams=2, min_samples_per_track=5)

        track_colors = {
            1: [np.array([200, 30, 30]) for _ in range(10)],  # Enough samples
            2: [np.array([200, 30, 30]) for _ in range(3)],   # Not enough
            3: [np.array([30, 30, 200]) for _ in range(10)],  # Enough samples
        }

        assigner.fit(track_colors)

        # Track 2 should not have a label (insufficient samples)
        assert 1 in assigner.team_labels
        assert 2 not in assigner.team_labels
        assert 3 in assigner.team_labels

    def test_fit_not_enough_tracks(self):
        """Test fit with fewer tracks than teams."""
        assigner = TeamAssigner(n_teams=3)

        track_colors = {
            1: [np.array([200, 30, 30]) for _ in range(10)],
            2: [np.array([30, 30, 200]) for _ in range(10)],
        }

        with pytest.raises(ValueError, match="Not enough tracks"):
            assigner.fit(track_colors)

    def test_fit_hsv_color_space(self):
        """Test fitting in HSV color space."""
        assigner = TeamAssigner(n_teams=2, color_space="hsv")

        # Create distinct colors in BGR
        track_colors = {
            1: [np.array([200, 30, 30]) for _ in range(10)],  # Blue
            2: [np.array([200, 30, 30]) for _ in range(10)],
            3: [np.array([30, 200, 30]) for _ in range(10)],  # Green
            4: [np.array([30, 200, 30]) for _ in range(10)],
        }

        assigner.fit(track_colors)

        # Should work with HSV conversion
        assert assigner.team_colors is not None


class TestTeamAssignerPredict:
    """Tests for TeamAssigner.predict()."""

    @pytest.fixture
    def fitted_assigner(self):
        """Create a fitted assigner."""
        assigner = TeamAssigner(n_teams=2, color_space="bgr")

        track_colors = {
            1: [np.array([200, 30, 30]) for _ in range(10)],
            2: [np.array([200, 30, 30]) for _ in range(10)],
            3: [np.array([30, 30, 200]) for _ in range(10)],
            4: [np.array([30, 30, 200]) for _ in range(10)],
        }
        assigner.fit(track_colors)
        return assigner

    def test_predict_not_fitted(self):
        """Test predict raises error when not fitted."""
        assigner = TeamAssigner()

        with pytest.raises(ValueError, match="not fitted"):
            assigner.predict(np.array([100, 100, 100]))

    def test_predict_similar_to_team1(self, fitted_assigner):
        """Test predicting color similar to first team."""
        # Color similar to blue team (high blue in BGR)
        color = np.array([190, 40, 40])

        team_id = fitted_assigner.predict(color)

        # Should match the blue team
        assert team_id == fitted_assigner.team_labels[1]

    def test_predict_similar_to_team2(self, fitted_assigner):
        """Test predicting color similar to second team."""
        # Color similar to red team (high red in BGR)
        color = np.array([40, 40, 190])

        team_id = fitted_assigner.predict(color)

        # Should match the red team
        assert team_id == fitted_assigner.team_labels[3]

    def test_predict_returns_int(self, fitted_assigner):
        """Test that predict returns an integer."""
        color = np.array([100, 100, 100])

        team_id = fitted_assigner.predict(color)

        assert isinstance(team_id, int)


class TestTeamAssignerGetTeamLabel:
    """Tests for TeamAssigner.get_team_label()."""

    @pytest.fixture
    def fitted_assigner(self):
        """Create a fitted assigner."""
        assigner = TeamAssigner(n_teams=2, color_space="bgr")

        track_colors = {
            1: [np.array([200, 30, 30]) for _ in range(10)],
            3: [np.array([30, 30, 200]) for _ in range(10)],
        }
        assigner.fit(track_colors)
        return assigner

    def test_get_existing_label(self, fitted_assigner):
        """Test getting label for existing track."""
        label = fitted_assigner.get_team_label(1)

        assert label is not None
        assert isinstance(label, int)

    def test_get_nonexistent_label(self, fitted_assigner):
        """Test getting label for non-existent track."""
        label = fitted_assigner.get_team_label(999)

        assert label is None


class TestTeamAssignerGetTeamColorsBgr:
    """Tests for TeamAssigner.get_team_colors_bgr()."""

    def test_not_fitted_returns_empty(self):
        """Test that not fitted returns empty dict."""
        assigner = TeamAssigner()

        colors = assigner.get_team_colors_bgr()

        assert colors == {}

    def test_fitted_returns_colors(self):
        """Test that fitted assigner returns team colors."""
        assigner = TeamAssigner(n_teams=2, color_space="bgr")

        track_colors = {
            1: [np.array([200, 30, 30]) for _ in range(10)],
            2: [np.array([30, 30, 200]) for _ in range(10)],
        }
        assigner.fit(track_colors)

        colors = assigner.get_team_colors_bgr()

        assert len(colors) == 2
        assert 0 in colors
        assert 1 in colors
        assert isinstance(colors[0], np.ndarray)

    def test_hsv_to_bgr_conversion(self):
        """Test that HSV colors are converted to BGR."""
        assigner = TeamAssigner(n_teams=2, color_space="hsv")

        track_colors = {
            1: [np.array([200, 30, 30]) for _ in range(10)],
            2: [np.array([30, 200, 30]) for _ in range(10)],
        }
        assigner.fit(track_colors)

        colors = assigner.get_team_colors_bgr()

        # Should return valid BGR colors (not HSV)
        for color in colors.values():
            assert color.shape == (3,)


class TestTeamAssignerAssignTeamNames:
    """Tests for TeamAssigner.assign_team_names()."""

    @pytest.fixture
    def assigner(self):
        """Create an assigner."""
        return TeamAssigner(n_teams=2)

    def test_explicit_team_names(self, assigner):
        """Test with explicit team name mapping."""
        team_names = {0: "Home", 1: "Away"}

        result = assigner.assign_team_names(team_names=team_names)

        assert result == {0: "Home", 1: "Away"}

    def test_our_team_id(self, assigner):
        """Test with our_team_id specified."""
        result = assigner.assign_team_names(our_team_id=0)

        assert result[0] == "ours"
        assert result[1] == "opponent"

    def test_default_names(self, assigner):
        """Test default generic team names."""
        result = assigner.assign_team_names()

        assert result[0] == "team_A"
        assert result[1] == "team_B"

    def test_three_teams_default(self):
        """Test default names with three teams."""
        assigner = TeamAssigner(n_teams=3)

        result = assigner.assign_team_names()

        assert result[0] == "team_A"
        assert result[1] == "team_B"
        assert result[2] == "team_C"


# -----------------------------------------------------------------------------
# collect_track_colors Tests
# -----------------------------------------------------------------------------

class TestCollectTrackColors:
    """Tests for collect_track_colors function."""

    @pytest.fixture
    def mock_extract_fn(self):
        """Create a mock color extraction function."""
        def extract_fn(frame, bbox):
            # Return different color based on bbox x position
            x1 = bbox[0]
            if x1 < 500:
                return np.array([200, 30, 30])  # Blue
            else:
                return np.array([30, 30, 200])  # Red
        return extract_fn

    @pytest.fixture
    def sample_frames(self):
        """Create sample frames."""
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        return {i: frame.copy() for i in range(10)}

    def test_collect_player_colors_only(self, mock_extract_fn, sample_frames):
        """Test that only player colors are collected (not ball)."""
        tracks_by_frame = {
            0: [
                {"track_id": 1, "bbox": (100, 100, 150, 200), "object_type": "player"},
                {"track_id": 2, "bbox": (500, 100, 520, 120), "object_type": "ball"},
            ],
        }

        result = collect_track_colors(tracks_by_frame, sample_frames, mock_extract_fn)

        # Should only have player track
        assert 1 in result
        assert 2 not in result

    def test_collect_multiple_frames(self, mock_extract_fn, sample_frames):
        """Test collecting colors across multiple frames."""
        tracks_by_frame = {
            i: [{"track_id": 1, "bbox": (100, 100, 150, 200), "object_type": "player"}]
            for i in range(5)
        }

        result = collect_track_colors(tracks_by_frame, sample_frames, mock_extract_fn)

        # Should have 5 color samples for track 1
        assert len(result[1]) == 5

    def test_skip_invalid_bbox(self, mock_extract_fn, sample_frames):
        """Test that invalid bboxes are skipped."""
        tracks_by_frame = {
            0: [
                {"track_id": 1, "bbox": (100, 100, 150, 200), "object_type": "player"},
                {"track_id": 2, "bbox": (float('nan'), 100, 150, 200), "object_type": "player"},
                {"track_id": 3, "bbox": (100, float('inf'), 150, 200), "object_type": "player"},
            ],
        }

        result = collect_track_colors(tracks_by_frame, sample_frames, mock_extract_fn)

        # Only track 1 should have colors
        assert 1 in result
        assert 2 not in result
        assert 3 not in result

    def test_skip_missing_frames(self, mock_extract_fn):
        """Test that missing frames are skipped."""
        frames = {0: np.zeros((1080, 1920, 3), dtype=np.uint8)}

        tracks_by_frame = {
            0: [{"track_id": 1, "bbox": (100, 100, 150, 200), "object_type": "player"}],
            1: [{"track_id": 1, "bbox": (100, 100, 150, 200), "object_type": "player"}],  # Frame 1 not in frames
        }

        result = collect_track_colors(tracks_by_frame, frames, mock_extract_fn)

        # Should only have 1 color sample (from frame 0)
        assert len(result[1]) == 1

    def test_skip_black_colors(self, sample_frames):
        """Test that black colors (failures) are skipped."""
        def extract_fn_with_failures(frame, bbox):
            x1 = bbox[0]
            if x1 < 200:
                return np.array([0, 0, 0])  # Black (failure)
            return np.array([200, 30, 30])

        tracks_by_frame = {
            0: [
                {"track_id": 1, "bbox": (100, 100, 150, 200), "object_type": "player"},  # Will return black
                {"track_id": 2, "bbox": (300, 100, 350, 200), "object_type": "player"},  # Valid color
            ],
        }

        result = collect_track_colors(tracks_by_frame, sample_frames, extract_fn_with_failures)

        # Track 1 should not have colors (black filtered out)
        assert 1 not in result or len(result.get(1, [])) == 0
        # Track 2 should have colors
        assert 2 in result
        assert len(result[2]) == 1

    def test_empty_tracks_by_frame(self, mock_extract_fn, sample_frames):
        """Test with empty tracks_by_frame."""
        result = collect_track_colors({}, sample_frames, mock_extract_fn)

        assert result == {}

    def test_multiple_tracks(self, mock_extract_fn, sample_frames):
        """Test collecting colors for multiple tracks."""
        tracks_by_frame = {
            0: [
                {"track_id": 1, "bbox": (100, 100, 150, 200), "object_type": "player"},
                {"track_id": 2, "bbox": (600, 100, 650, 200), "object_type": "player"},
            ],
            1: [
                {"track_id": 1, "bbox": (110, 102, 160, 202), "object_type": "player"},
                {"track_id": 2, "bbox": (610, 102, 660, 202), "object_type": "player"},
            ],
        }

        result = collect_track_colors(tracks_by_frame, sample_frames, mock_extract_fn)

        assert 1 in result
        assert 2 in result
        assert len(result[1]) == 2
        assert len(result[2]) == 2
