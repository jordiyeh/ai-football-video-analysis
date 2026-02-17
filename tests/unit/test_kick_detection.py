"""Tests for kick detection module."""

import pytest

from src.config.schemas import AlternativeShotDetectionConfig
from src.events.kick_detection import (
    GoalAreaEntryDetector,
    GoalAreaEntryEvent,
    KickEvent,
    KickEventDetector,
    ShotFusionEngine,
)


@pytest.fixture
def config():
    """Default alternative shot detection config."""
    return AlternativeShotDetectionConfig()


@pytest.fixture
def kick_detector(config):
    """Create a kick event detector."""
    return KickEventDetector(config)


@pytest.fixture
def goal_entry_detector(config):
    """Create a goal area entry detector."""
    return GoalAreaEntryDetector(
        frame_width=1920,
        frame_height=1080,
        config=config,
    )


@pytest.fixture
def fusion_engine(config):
    """Create a shot fusion engine."""
    return ShotFusionEngine(config, fps=30.0)


class TestKickEventDetector:
    """Tests for KickEventDetector."""

    def test_detect_kicks_basic(self, kick_detector):
        """Test basic kick detection with ball near player foot."""
        player_tracks = [
            {
                "frame_idx": 100,
                "track_id": 1,
                "bbox": [500, 400, 550, 600],  # Player bbox
                "object_type": "player",
            }
        ]
        ball_tracks = [
            {
                "frame_idx": 100,
                "bbox": [510, 580, 530, 600],  # Ball near bottom of player
                "confidence": 0.8,
                "object_type": "ball",
                "timestamp": 100 / 30.0,
            }
        ]

        kicks = kick_detector.detect_kicks(
            player_tracks, ball_tracks, 1920, 1080
        )

        assert len(kicks) == 1
        assert kicks[0].frame_idx == 100
        assert kicks[0].player_track_id == 1
        assert kicks[0].confidence > 0

    def test_detect_kicks_no_ball(self, kick_detector):
        """Test no kicks detected when no ball present."""
        player_tracks = [
            {
                "frame_idx": 100,
                "track_id": 1,
                "bbox": [500, 400, 550, 600],
                "object_type": "player",
            }
        ]
        ball_tracks = []

        kicks = kick_detector.detect_kicks(
            player_tracks, ball_tracks, 1920, 1080
        )

        assert len(kicks) == 0

    def test_detect_kicks_ball_not_near_player(self, kick_detector):
        """Test no kicks when ball is far from player."""
        player_tracks = [
            {
                "frame_idx": 100,
                "track_id": 1,
                "bbox": [500, 400, 550, 600],
                "object_type": "player",
            }
        ]
        ball_tracks = [
            {
                "frame_idx": 100,
                "bbox": [1000, 100, 1020, 120],  # Ball far from player
                "confidence": 0.8,
                "object_type": "ball",
                "timestamp": 100 / 30.0,
            }
        ]

        kicks = kick_detector.detect_kicks(
            player_tracks, ball_tracks, 1920, 1080
        )

        assert len(kicks) == 0

    def test_detect_kicks_multiple_players(self, kick_detector):
        """Test kick detection finds the correct player."""
        player_tracks = [
            {
                "frame_idx": 100,
                "track_id": 1,
                "bbox": [100, 400, 150, 600],  # Player 1 (far)
                "object_type": "player",
            },
            {
                "frame_idx": 100,
                "track_id": 2,
                "bbox": [500, 400, 550, 600],  # Player 2 (near ball)
                "object_type": "player",
            },
        ]
        ball_tracks = [
            {
                "frame_idx": 100,
                "bbox": [510, 580, 530, 600],  # Ball near player 2
                "confidence": 0.8,
                "object_type": "ball",
                "timestamp": 100 / 30.0,
            }
        ]

        kicks = kick_detector.detect_kicks(
            player_tracks, ball_tracks, 1920, 1080
        )

        assert len(kicks) == 1
        assert kicks[0].player_track_id == 2

    def test_foot_region_calculation(self, kick_detector):
        """Test that foot region is correctly calculated."""
        # Ball at player's head region should not trigger kick
        player_tracks = [
            {
                "frame_idx": 100,
                "track_id": 1,
                "bbox": [500, 400, 550, 600],
                "object_type": "player",
            }
        ]
        ball_tracks = [
            {
                "frame_idx": 100,
                "bbox": [510, 410, 530, 430],  # Ball at head height
                "confidence": 0.8,
                "object_type": "ball",
                "timestamp": 100 / 30.0,
            }
        ]

        kicks = kick_detector.detect_kicks(
            player_tracks, ball_tracks, 1920, 1080
        )

        # Should not detect kick - ball is at head, not feet
        assert len(kicks) == 0


class TestGoalAreaEntryDetector:
    """Tests for GoalAreaEntryDetector."""

    def test_detect_goal_entry_top(self, goal_entry_detector):
        """Test detecting ball entering top goal area."""
        ball_tracks = [
            {
                "frame_idx": 100,
                "bbox": [950, 50, 970, 70],  # Ball in top goal area
                "confidence": 0.8,
                "object_type": "ball",
                "timestamp": 100 / 30.0,
            }
        ]
        kick_events = []

        entries = goal_entry_detector.detect_goal_entries(
            ball_tracks, kick_events, fps=30.0
        )

        assert len(entries) == 1
        assert entries[0].goal_region == "top"
        assert entries[0].frame_idx == 100

    def test_detect_goal_entry_bottom(self, goal_entry_detector):
        """Test detecting ball entering bottom goal area."""
        ball_tracks = [
            {
                "frame_idx": 100,
                "bbox": [950, 1000, 970, 1020],  # Ball in bottom goal area
                "confidence": 0.8,
                "object_type": "ball",
                "timestamp": 100 / 30.0,
            }
        ]
        kick_events = []

        entries = goal_entry_detector.detect_goal_entries(
            ball_tracks, kick_events, fps=30.0
        )

        assert len(entries) == 1
        assert entries[0].goal_region == "bottom"

    def test_goal_entry_with_kick_association(self, goal_entry_detector, config):
        """Test goal entry is associated with preceding kick."""
        kick_events = [
            KickEvent(
                frame_idx=50,
                timestamp=50 / 30.0,
                player_track_id=1,
                player_bbox=(500, 400, 550, 600),
                ball_position=(520, 590),
                confidence=0.7,
            )
        ]
        ball_tracks = [
            {
                "frame_idx": 100,
                "bbox": [950, 50, 970, 70],  # Ball enters goal 50 frames after kick
                "confidence": 0.8,
                "object_type": "ball",
                "timestamp": 100 / 30.0,
            }
        ]

        entries = goal_entry_detector.detect_goal_entries(
            ball_tracks, kick_events, fps=30.0
        )

        assert len(entries) == 1
        assert entries[0].associated_kick is not None
        assert entries[0].associated_kick.frame_idx == 50

    def test_no_goal_entry_outside_region(self, goal_entry_detector):
        """Test no entry detected when ball is in middle of field."""
        ball_tracks = [
            {
                "frame_idx": 100,
                "bbox": [960, 540, 980, 560],  # Ball in center of frame
                "confidence": 0.8,
                "object_type": "ball",
                "timestamp": 100 / 30.0,
            }
        ]
        kick_events = []

        entries = goal_entry_detector.detect_goal_entries(
            ball_tracks, kick_events, fps=30.0
        )

        assert len(entries) == 0


class TestShotFusionEngine:
    """Tests for ShotFusionEngine."""

    def test_single_kick_signal(self, fusion_engine):
        """Test shot candidate from single kick signal."""
        kick_events = [
            KickEvent(
                frame_idx=100,
                timestamp=100 / 30.0,
                player_track_id=1,
                player_bbox=(500, 400, 550, 600),
                ball_position=(520, 590),
                confidence=0.8,
            )
        ]

        candidates = fusion_engine.fuse_signals(
            kick_events=kick_events,
            goal_entries=[],
            gk_dive_frames=[],
            attack_windows=[],
        )

        assert len(candidates) == 1
        assert "kick" in candidates[0].signals_present
        assert candidates[0].kick_event is not None

    def test_multiple_signals_boost_confidence(self, fusion_engine):
        """Test that multiple signals increase confidence."""
        kick_events = [
            KickEvent(
                frame_idx=100,
                timestamp=100 / 30.0,
                player_track_id=1,
                player_bbox=(500, 400, 550, 600),
                ball_position=(520, 590),
                confidence=0.6,
            )
        ]
        goal_entries = [
            GoalAreaEntryEvent(
                frame_idx=110,  # Within temporal window
                timestamp=110 / 30.0,
                goal_region="top",
                entry_position=(960, 50),
                confidence=0.7,
            )
        ]

        candidates = fusion_engine.fuse_signals(
            kick_events=kick_events,
            goal_entries=goal_entries,
            gk_dive_frames=[115],  # GK dive also in window
            attack_windows=[],
        )

        assert len(candidates) == 1
        assert len(candidates[0].signals_present) >= 2
        # Multiple signals should boost confidence
        assert candidates[0].confidence > 0.3

    def test_below_threshold_filtered(self, fusion_engine):
        """Test that candidates below threshold are filtered."""
        # Single weak signal
        kick_events = [
            KickEvent(
                frame_idx=100,
                timestamp=100 / 30.0,
                player_track_id=1,
                player_bbox=(500, 400, 550, 600),
                ball_position=(520, 590),
                confidence=0.1,  # Very low confidence kick
            )
        ]

        candidates = fusion_engine.fuse_signals(
            kick_events=kick_events,
            goal_entries=[],
            gk_dive_frames=[],
            attack_windows=[],
        )

        # May or may not produce candidate depending on threshold
        # With 0.1 kick conf and 0.35 weight, weighted sum is 0.035
        # Normalized by max possible (all weights) ~ 0.035
        # This is below fusion_min_confidence of 0.3
        assert len(candidates) == 0

    def test_temporal_grouping(self, fusion_engine):
        """Test that events within temporal window are grouped."""
        kick_events = [
            KickEvent(
                frame_idx=100,
                timestamp=100 / 30.0,
                player_track_id=1,
                player_bbox=(500, 400, 550, 600),
                ball_position=(520, 590),
                confidence=0.8,
            ),
            KickEvent(
                frame_idx=200,  # Outside temporal window (60 frames default)
                timestamp=200 / 30.0,
                player_track_id=2,
                player_bbox=(600, 400, 650, 600),
                ball_position=(620, 590),
                confidence=0.8,
            ),
        ]

        candidates = fusion_engine.fuse_signals(
            kick_events=kick_events,
            goal_entries=[],
            gk_dive_frames=[],
            attack_windows=[],
        )

        # Should produce two separate candidates (different temporal windows)
        assert len(candidates) == 2

    def test_empty_signals(self, fusion_engine):
        """Test handling of no signals."""
        candidates = fusion_engine.fuse_signals(
            kick_events=[],
            goal_entries=[],
            gk_dive_frames=[],
            attack_windows=[],
        )

        assert len(candidates) == 0
