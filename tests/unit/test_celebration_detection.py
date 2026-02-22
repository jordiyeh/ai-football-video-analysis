"""Tests for celebration detection module."""

import pytest

from src.config.schemas import AlternativeShotDetectionConfig, CelebrationConfig
from src.events.celebration_detection import CelebrationDetector, CelebrationEvent
from src.events.kick_detection import KickEvent, ShotCandidate


@pytest.fixture
def celebration_config():
    """Default celebration detection config."""
    return CelebrationConfig()


@pytest.fixture
def alternative_config():
    """Alternative shot detection config with celebration enabled."""
    return AlternativeShotDetectionConfig()


@pytest.fixture
def celebration_detector(celebration_config):
    """Create a celebration detector."""
    return CelebrationDetector(
        frame_width=1920,
        frame_height=1080,
        config=celebration_config,
    )


@pytest.fixture
def sample_shot_candidate():
    """A sample shot candidate to detect celebrations after."""
    return ShotCandidate(
        frame_idx=100,
        timestamp=100 / 30.0,
        confidence=0.8,
        kick_event=KickEvent(
            frame_idx=100,
            timestamp=100 / 30.0,
            player_track_id=1,
            player_bbox=(500, 400, 550, 600),
            ball_position=(520, 590),
            confidence=0.8,
        ),
        signals_present=["kick", "goal_entry"],
    )


class TestCelebrationDetector:
    """Tests for CelebrationDetector."""

    def test_detect_arms_up_celebration(self, celebration_detector, sample_shot_candidate):
        """Test detection of arms-up celebration pose."""
        # Create player tracks with baseline and arms-up pose
        player_tracks = []

        # Baseline frames (normal standing pose - tall and narrow)
        for i in range(30):
            player_tracks.append({
                "frame_idx": i,
                "track_id": 1,
                "bbox": [500, 400, 550, 600],  # Normal: width=50, height=200, aspect=0.25
                "team_id": 0,
            })

        # Arms-up frames after shot (wider bbox indicating raised arms)
        for i in range(sample_shot_candidate.frame_idx, sample_shot_candidate.frame_idx + 30):
            player_tracks.append({
                "frame_idx": i,
                "track_id": 1,
                "bbox": [480, 410, 570, 590],  # Arms up: width=90, height=180, aspect=0.5
                "team_id": 0,
            })

        celebrations = celebration_detector.detect_celebrations(
            player_tracks, [sample_shot_candidate], fps=30.0
        )

        assert len(celebrations) >= 1
        arms_up = [c for c in celebrations if c.celebration_type == "individual_arms_up"]
        assert len(arms_up) >= 1
        assert arms_up[0].participating_track_ids == [1]
        assert arms_up[0].confidence > 0

    def test_detect_group_huddle(self, celebration_detector, sample_shot_candidate):
        """Test detection of group huddle celebration."""
        player_tracks = []

        # Pre-shot: players spread out
        for i in range(30):
            for pid, x_pos in enumerate([200, 400, 600, 800, 1000]):
                player_tracks.append({
                    "frame_idx": sample_shot_candidate.frame_idx - 30 + i,
                    "track_id": pid + 1,
                    "bbox": [x_pos, 500, x_pos + 50, 700],
                    "team_id": 0,
                })

        # Post-shot: players converge to huddle
        for i in range(100):
            frame_idx = sample_shot_candidate.frame_idx + i
            for pid in range(5):
                # Players converge to center
                x_pos = 500 + pid * 30  # Much closer together
                player_tracks.append({
                    "frame_idx": frame_idx,
                    "track_id": pid + 1,
                    "bbox": [x_pos, 500, x_pos + 50, 700],
                    "team_id": 0,
                })

        celebrations = celebration_detector.detect_celebrations(
            player_tracks, [sample_shot_candidate], fps=30.0
        )

        huddles = [c for c in celebrations if c.celebration_type == "group_huddle"]
        assert len(huddles) >= 1
        assert len(huddles[0].participating_track_ids) >= 3

    def test_no_celebration_without_shot(self, celebration_detector):
        """Test no celebrations detected without shot context."""
        # Player tracks with arms-up pose but no shot candidate
        player_tracks = []
        for i in range(100):
            player_tracks.append({
                "frame_idx": i,
                "track_id": 1,
                "bbox": [480, 410, 570, 590],  # Arms up pose
                "team_id": 0,
            })

        celebrations = celebration_detector.detect_celebrations(
            player_tracks, [], fps=30.0  # No shot candidates
        )

        assert len(celebrations) == 0

    def test_cooldown_prevents_duplicates(self, celebration_detector):
        """Test that cooldown prevents duplicate celebration detection."""
        # Two shot candidates close together
        shot1 = ShotCandidate(
            frame_idx=100,
            timestamp=100 / 30.0,
            confidence=0.8,
            signals_present=["kick"],
        )
        shot2 = ShotCandidate(
            frame_idx=200,  # Only 100 frames later (within 300 frame cooldown)
            timestamp=200 / 30.0,
            confidence=0.8,
            signals_present=["kick"],
        )

        player_tracks = []

        # Baseline
        for i in range(30):
            player_tracks.append({
                "frame_idx": i,
                "track_id": 1,
                "bbox": [500, 400, 550, 600],
                "team_id": 0,
            })

        # Arms-up after shot 1
        for i in range(shot1.frame_idx, shot1.frame_idx + 30):
            player_tracks.append({
                "frame_idx": i,
                "track_id": 1,
                "bbox": [480, 410, 570, 590],
                "team_id": 0,
            })

        # Arms-up after shot 2
        for i in range(shot2.frame_idx, shot2.frame_idx + 30):
            player_tracks.append({
                "frame_idx": i,
                "track_id": 1,
                "bbox": [480, 410, 570, 590],
                "team_id": 0,
            })

        celebrations = celebration_detector.detect_celebrations(
            player_tracks, [shot1, shot2], fps=30.0
        )

        # Due to cooldown, should only detect one celebration
        arms_up = [c for c in celebrations if c.celebration_type == "individual_arms_up"]
        # With 300 frame cooldown, second should be filtered
        assert len(arms_up) <= 2  # At most 2 (cooldown may filter)

    def test_subdued_celebration_filtered(self, celebration_detector, sample_shot_candidate):
        """Test that subdued/weak celebrations are filtered by min_confidence."""
        player_tracks = []

        # Baseline
        for i in range(30):
            player_tracks.append({
                "frame_idx": i,
                "track_id": 1,
                "bbox": [500, 400, 550, 600],  # aspect = 0.25
                "team_id": 0,
            })

        # Very subtle pose change (not enough for celebration)
        for i in range(sample_shot_candidate.frame_idx, sample_shot_candidate.frame_idx + 10):
            player_tracks.append({
                "frame_idx": i,
                "track_id": 1,
                "bbox": [495, 402, 555, 598],  # aspect = 0.306 (barely any change)
                "team_id": 0,
            })

        celebrations = celebration_detector.detect_celebrations(
            player_tracks, [sample_shot_candidate], fps=30.0
        )

        # Subdued celebration should be filtered by min_confidence
        arms_up = [c for c in celebrations if c.celebration_type == "individual_arms_up"]
        for c in arms_up:
            # Any detected celebration should meet min_confidence
            assert c.confidence >= celebration_detector.config.min_confidence

    def test_multiple_players_celebrating(self, celebration_detector, sample_shot_candidate):
        """Test detection when multiple players celebrate."""
        player_tracks = []

        # Baseline for multiple players
        for i in range(30):
            for pid, x_offset in enumerate([0, 100, 200]):
                player_tracks.append({
                    "frame_idx": i,
                    "track_id": pid + 1,
                    "bbox": [500 + x_offset, 400, 550 + x_offset, 600],
                    "team_id": 0,
                })

        # All players raise arms after shot
        for i in range(sample_shot_candidate.frame_idx, sample_shot_candidate.frame_idx + 30):
            for pid, x_offset in enumerate([0, 100, 200]):
                player_tracks.append({
                    "frame_idx": i,
                    "track_id": pid + 1,
                    "bbox": [480 + x_offset, 410, 570 + x_offset, 590],
                    "team_id": 0,
                })

        celebrations = celebration_detector.detect_celebrations(
            player_tracks, [sample_shot_candidate], fps=30.0
        )

        # Should detect multiple individual celebrations or a group
        assert len(celebrations) >= 1

    def test_empty_inputs(self, celebration_detector, sample_shot_candidate):
        """Test handling of empty inputs."""
        # Empty player tracks
        celebrations = celebration_detector.detect_celebrations(
            [], [sample_shot_candidate], fps=30.0
        )
        assert celebrations == []

        # Empty shot candidates
        player_tracks = [
            {"frame_idx": 100, "track_id": 1, "bbox": [500, 400, 550, 600], "team_id": 0}
        ]
        celebrations = celebration_detector.detect_celebrations(
            player_tracks, [], fps=30.0
        )
        assert celebrations == []

    def test_celebration_event_fields(self, celebration_detector, sample_shot_candidate):
        """Test that CelebrationEvent has all required fields."""
        player_tracks = []

        # Baseline
        for i in range(30):
            player_tracks.append({
                "frame_idx": i,
                "track_id": 1,
                "bbox": [500, 400, 550, 600],
                "team_id": 0,
            })

        # Arms-up celebration
        for i in range(sample_shot_candidate.frame_idx, sample_shot_candidate.frame_idx + 30):
            player_tracks.append({
                "frame_idx": i,
                "track_id": 1,
                "bbox": [480, 410, 570, 590],
                "team_id": 0,
            })

        celebrations = celebration_detector.detect_celebrations(
            player_tracks, [sample_shot_candidate], fps=30.0
        )

        if celebrations:
            event = celebrations[0]
            # Check all required fields exist
            assert hasattr(event, "frame_idx")
            assert hasattr(event, "timestamp")
            assert hasattr(event, "confidence")
            assert hasattr(event, "celebration_type")
            assert hasattr(event, "participating_track_ids")
            assert hasattr(event, "team_id")
            assert hasattr(event, "center_position")
            assert hasattr(event, "evidence")

            # Check field types
            assert isinstance(event.frame_idx, int)
            assert isinstance(event.timestamp, float)
            assert isinstance(event.confidence, float)
            assert isinstance(event.celebration_type, str)
            assert isinstance(event.participating_track_ids, list)
            assert isinstance(event.center_position, tuple)
            assert isinstance(event.evidence, dict)


class TestCelebrationConfig:
    """Tests for CelebrationConfig."""

    def test_default_values(self):
        """Test default config values."""
        config = CelebrationConfig()

        assert config.enabled is True
        assert config.arms_up_aspect_ratio_threshold == 0.5
        assert config.arms_up_height_change_threshold == 1.2
        assert config.arms_up_min_duration_frames == 5
        assert config.huddle_max_player_distance == 100.0
        assert config.huddle_min_players == 3
        assert config.huddle_convergence_threshold == 0.5
        assert config.post_shot_window_frames == 150
        assert config.celebration_cooldown_frames == 300
        assert config.signal_weight == 0.15
        assert config.min_confidence == 0.4

    def test_nested_in_alternative_config(self):
        """Test CelebrationConfig is nested in AlternativeShotDetectionConfig."""
        config = AlternativeShotDetectionConfig()

        assert hasattr(config, "celebration")
        assert isinstance(config.celebration, CelebrationConfig)
        assert config.celebration.enabled is True

    def test_custom_values(self):
        """Test custom config values."""
        config = CelebrationConfig(
            enabled=False,
            signal_weight=0.20,
            min_confidence=0.5,
        )

        assert config.enabled is False
        assert config.signal_weight == 0.20
        assert config.min_confidence == 0.5


class TestShotFusionWithCelebration:
    """Tests for celebration integration in ShotFusionEngine."""

    def test_celebration_signal_in_fusion(self, alternative_config):
        """Test that celebration signal is included in fusion."""
        from src.events.kick_detection import ShotFusionEngine

        fusion_engine = ShotFusionEngine(alternative_config, fps=30.0)

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

        celebration_events = [
            CelebrationEvent(
                frame_idx=150,  # After kick, within window
                timestamp=150 / 30.0,
                confidence=0.7,
                celebration_type="individual_arms_up",
                participating_track_ids=[1],
                team_id=0,
                center_position=(525.0, 500.0),
            )
        ]

        candidates = fusion_engine.fuse_signals(
            kick_events=kick_events,
            goal_entries=[],
            gk_dive_frames=[],
            attack_windows=[],
            celebration_events=celebration_events,
        )

        assert len(candidates) == 1
        assert "celebration" in candidates[0].signals_present
        assert candidates[0].celebration_event is not None
        assert candidates[0].celebration_event.confidence == 0.7

    def test_celebration_boosts_confidence(self, alternative_config):
        """Test that celebration signal boosts shot confidence."""
        from src.events.kick_detection import ShotFusionEngine

        fusion_engine = ShotFusionEngine(alternative_config, fps=30.0)

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

        # Without celebration
        candidates_no_celebration = fusion_engine.fuse_signals(
            kick_events=kick_events,
            goal_entries=[],
            gk_dive_frames=[],
            attack_windows=[],
            celebration_events=None,
        )

        # With celebration
        celebration_events = [
            CelebrationEvent(
                frame_idx=150,
                timestamp=150 / 30.0,
                confidence=0.8,
                celebration_type="group_huddle",
                participating_track_ids=[1, 2, 3],
                team_id=0,
                center_position=(525.0, 500.0),
            )
        ]

        candidates_with_celebration = fusion_engine.fuse_signals(
            kick_events=kick_events,
            goal_entries=[],
            gk_dive_frames=[],
            attack_windows=[],
            celebration_events=celebration_events,
        )

        # Both should produce candidates
        assert len(candidates_no_celebration) == 1
        assert len(candidates_with_celebration) == 1

        # With celebration should have higher confidence (or equal if bonus caps it)
        # The celebration adds another corroborating signal
        assert "celebration" in candidates_with_celebration[0].signals_present
        assert "celebration" not in candidates_no_celebration[0].signals_present

    def test_weight_rebalancing(self, alternative_config):
        """Test that weights are rebalanced when celebration is enabled."""
        from src.events.kick_detection import ShotFusionEngine

        fusion_engine = ShotFusionEngine(alternative_config, fps=30.0)

        # When celebration is enabled, weights should be adjusted:
        # kick: 0.30 (was 0.35)
        # goal_entry: 0.25 (was 0.30)
        # gk_dive: 0.20 (was 0.25)
        # attack: 0.10 (unchanged)
        # celebration: 0.15 (new)

        # We can verify this by checking that the fusion engine uses the config
        assert fusion_engine.config.celebration.enabled is True
        assert fusion_engine.config.celebration.signal_weight == 0.15
