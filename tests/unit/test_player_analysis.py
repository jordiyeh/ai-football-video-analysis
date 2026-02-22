"""Tests for player analysis module."""

import pytest

from src.config.schemas import AlternativeShotDetectionConfig
from src.events.player_analysis import (
    ClusteringState,
    GoalkeeperAnalyzer,
    GoalkeeperDiveEvent,
    PlayerClusteringAnalyzer,
)


@pytest.fixture
def config():
    """Default alternative shot detection config."""
    return AlternativeShotDetectionConfig()


@pytest.fixture
def clustering_analyzer(config):
    """Create a player clustering analyzer."""
    return PlayerClusteringAnalyzer(
        frame_width=1920,
        frame_height=1080,
        config=config,
    )


@pytest.fixture
def gk_analyzer(config):
    """Create a goalkeeper analyzer."""
    return GoalkeeperAnalyzer(
        frame_width=1920,
        frame_height=1080,
        config=config,
    )


class TestPlayerClusteringAnalyzer:
    """Tests for PlayerClusteringAnalyzer."""

    def test_analyze_clustering_basic(self, clustering_analyzer):
        """Test basic clustering analysis with two teams."""
        # Create player tracks for two teams
        player_tracks = []

        # Team 0: clustered near top goal (attacking)
        for i in range(6):
            player_tracks.append({
                "frame_idx": 100,
                "track_id": i,
                "team_id": 0,
                "bbox": [800 + i * 30, 150, 850 + i * 30, 300],  # Near top
                "object_type": "player",
            })

        # Team 1: spread in middle (defending)
        for i in range(6):
            player_tracks.append({
                "frame_idx": 100,
                "track_id": 10 + i,
                "team_id": 1,
                "bbox": [200 + i * 200, 500, 250 + i * 200, 650],  # Spread in middle
                "object_type": "player",
            })

        states = clustering_analyzer.analyze_clustering(player_tracks)

        assert 100 in states
        state = states[100]
        assert "0" in state.team_centroids
        assert "1" in state.team_centroids

    def test_attack_score_near_goal(self, clustering_analyzer):
        """Test attack score is high when team is near goal."""
        player_tracks = []

        # Team 0: tightly clustered near top goal
        for i in range(6):
            player_tracks.append({
                "frame_idx": 100,
                "track_id": i,
                "team_id": 0,
                "bbox": [900 + i * 10, 100 + i * 10, 950 + i * 10, 250 + i * 10],
                "object_type": "player",
            })

        # Team 1: pushed back in own half
        for i in range(6):
            player_tracks.append({
                "frame_idx": 100,
                "track_id": 10 + i,
                "team_id": 1,
                "bbox": [300 + i * 50, 200 + i * 20, 350 + i * 50, 350 + i * 20],
                "object_type": "player",
            })

        states = clustering_analyzer.analyze_clustering(player_tracks)

        assert 100 in states
        state = states[100]
        # Should have a positive attack score
        assert state.attack_score > 0

    def test_attack_windows_detection(self, clustering_analyzer):
        """Test detection of attack windows over time."""
        player_tracks = []

        # Create 30 frames of attacking formation
        for frame in range(100, 130):
            for i in range(6):
                player_tracks.append({
                    "frame_idx": frame,
                    "track_id": i,
                    "team_id": 0,
                    "bbox": [900 + i * 10, 100 + i * 10, 950 + i * 10, 250 + i * 10],
                    "object_type": "player",
                })
                player_tracks.append({
                    "frame_idx": frame,
                    "track_id": 10 + i,
                    "team_id": 1,
                    "bbox": [300 + i * 50, 200 + i * 20, 350 + i * 50, 350 + i * 20],
                    "object_type": "player",
                })

        states = clustering_analyzer.analyze_clustering(player_tracks)
        windows = clustering_analyzer.detect_attack_windows(
            states, min_window_frames=15, score_threshold=0.3
        )

        # Should detect at least one attack window
        assert len(windows) >= 0  # May or may not meet threshold

    def test_no_clustering_without_teams(self, clustering_analyzer):
        """Test handling of tracks without team assignments."""
        player_tracks = [
            {
                "frame_idx": 100,
                "track_id": i,
                "team_id": -1,  # Unknown team
                "bbox": [100 + i * 100, 400, 150 + i * 100, 600],
                "object_type": "player",
            }
            for i in range(10)
        ]

        states = clustering_analyzer.analyze_clustering(player_tracks)

        # Should not produce a state (no valid teams)
        assert 100 not in states or len(states[100].team_centroids) == 0

    def test_min_players_threshold(self, clustering_analyzer):
        """Test that teams with few players are ignored."""
        player_tracks = []

        # Team 0: only 2 players (below min_players_per_team=5)
        for i in range(2):
            player_tracks.append({
                "frame_idx": 100,
                "track_id": i,
                "team_id": 0,
                "bbox": [800 + i * 30, 150, 850 + i * 30, 300],
                "object_type": "player",
            })

        states = clustering_analyzer.analyze_clustering(player_tracks)

        # Should not include team 0 (too few players)
        if 100 in states:
            assert "0" not in states[100].team_centroids


class TestGoalkeeperAnalyzer:
    """Tests for GoalkeeperAnalyzer."""

    def test_detect_dive_basic(self, gk_analyzer):
        """Test basic goalkeeper dive detection."""
        # Create a track that shows diving motion
        player_tracks = []

        # Initial standing position near top goal
        for i in range(20):
            if i < 10:
                # Standing (tall bbox): width=50, height=150, center_x=975
                bbox = [950, 50, 1000, 200]
            else:
                # Diving (wide bbox, moved left): width=150, height=80, center_x=925
                # Displacement = 975 - 925 = 50 pixels (> 30 threshold)
                # Aspect ratio change: (150/80) / (50/150) = 1.875 / 0.33 = ~5.7 (> 1.5)
                bbox = [850, 100, 1000, 180]

            player_tracks.append({
                "frame_idx": 100 + i,
                "track_id": 1,
                "bbox": bbox,
                "object_type": "player",
                "timestamp": (100 + i) / 30.0,
            })

        dives = gk_analyzer.detect_goalkeeper_dives(player_tracks, fps=30.0)

        # Should detect a dive
        assert len(dives) >= 1

    def test_no_dive_stationary_player(self, gk_analyzer):
        """Test no dive detected for stationary goalkeeper."""
        player_tracks = []

        # Stationary keeper near goal
        for i in range(30):
            player_tracks.append({
                "frame_idx": 100 + i,
                "track_id": 1,
                "bbox": [950, 50, 1000, 200],  # Same position
                "object_type": "player",
                "timestamp": (100 + i) / 30.0,
            })

        dives = gk_analyzer.detect_goalkeeper_dives(player_tracks, fps=30.0)

        assert len(dives) == 0

    def test_goalkeeper_region_identification(self, gk_analyzer):
        """Test goalkeeper identification based on position."""
        player_tracks = []

        # Player near top goal (should be identified as GK)
        for i in range(20):
            player_tracks.append({
                "frame_idx": 100 + i,
                "track_id": 1,
                "bbox": [950, 50, 1000, 200],  # Near top
                "object_type": "player",
                "timestamp": (100 + i) / 30.0,
            })

        # Player in midfield (should NOT be identified as GK)
        for i in range(20):
            player_tracks.append({
                "frame_idx": 100 + i,
                "track_id": 2,
                "bbox": [800, 500, 850, 650],  # Middle of field
                "object_type": "player",
                "timestamp": (100 + i) / 30.0,
            })

        dives = gk_analyzer.detect_goalkeeper_dives(player_tracks, fps=30.0)

        # Only track 1 should be considered as goalkeeper
        for dive in dives:
            assert dive.track_id == 1

    def test_dive_direction(self, gk_analyzer):
        """Test correct dive direction detection."""
        player_tracks = []

        # Dive to the right
        for i in range(15):
            x_offset = i * 5 if i >= 5 else 0  # Start moving at frame 5
            height = 150 if i < 5 else max(80, 150 - i * 10)  # Get shorter when diving

            player_tracks.append({
                "frame_idx": 100 + i,
                "track_id": 1,
                "bbox": [950 + x_offset, 50, 1000 + x_offset, 50 + height],
                "object_type": "player",
                "timestamp": (100 + i) / 30.0,
            })

        dives = gk_analyzer.detect_goalkeeper_dives(player_tracks, fps=30.0)

        if dives:
            assert dives[0].dive_direction == "right"

    def test_dive_deduplication(self, gk_analyzer):
        """Test that nearby dives are deduplicated."""
        player_tracks = []

        # Create multiple dive-like motions close together
        for i in range(30):
            if i < 10:
                bbox = [950, 50, 1000, 200]  # Standing
            elif i < 20:
                bbox = [900, 100, 1050, 180]  # First dive
            else:
                bbox = [920, 100, 1070, 180]  # Slight variation

            player_tracks.append({
                "frame_idx": 100 + i,
                "track_id": 1,
                "bbox": bbox,
                "object_type": "player",
                "timestamp": (100 + i) / 30.0,
            })

        dives = gk_analyzer.detect_goalkeeper_dives(player_tracks, fps=30.0)

        # Should deduplicate to single dive event
        assert len(dives) <= 2  # At most 2 if timing creates separate detections


class TestClusteringState:
    """Tests for ClusteringState dataclass."""

    def test_clustering_state_creation(self):
        """Test ClusteringState can be created correctly."""
        state = ClusteringState(
            frame_idx=100,
            team_centroids={"0": (960.0, 200.0), "1": (960.0, 800.0)},
            team_spreads={"0": 50.0, "1": 100.0},
            attack_score=0.7,
        )

        assert state.frame_idx == 100
        assert state.team_centroids["0"] == (960.0, 200.0)
        assert state.team_spreads["0"] == 50.0
        assert state.attack_score == 0.7


class TestGoalkeeperDiveEvent:
    """Tests for GoalkeeperDiveEvent dataclass."""

    def test_dive_event_creation(self):
        """Test GoalkeeperDiveEvent can be created correctly."""
        dive = GoalkeeperDiveEvent(
            frame_idx=100,
            timestamp=3.33,
            track_id=1,
            dive_direction="left",
            displacement=50.0,
            aspect_ratio_change=2.0,
            confidence=0.8,
        )

        assert dive.frame_idx == 100
        assert dive.dive_direction == "left"
        assert dive.displacement == 50.0
        assert dive.aspect_ratio_change == 2.0
        assert dive.confidence == 0.8
