"""Unit tests for highlight generation helpers."""

import numpy as np

from src.events.highlights import (
    HighlightCandidate,
    build_action_candidates,
    build_event_candidates,
    build_segments_from_candidates,
    select_highlight_segments,
)


class TestBuildEventCandidates:
    """Tests for event candidate creation."""

    def test_builds_goal_and_shot_candidates(self):
        """Goals and shots should become event candidates."""
        events = [
            {"event_type": "goal", "timestamp": 100.0, "frame_idx": 3000, "confidence": 0.9},
            {"event_type": "shot", "timestamp": 90.0, "frame_idx": 2700, "confidence": 0.8},
            {"event_type": "other", "timestamp": 95.0, "frame_idx": 2850, "confidence": 0.99},
        ]

        candidates = build_event_candidates(
            events=events,
            include_goals=True,
            include_shots=True,
            goal_weight=1.0,
            shot_weight=0.5,
            min_confidence=0.2,
        )

        assert len(candidates) == 2

        goal = next(c for c in candidates if c.reason == "goal")
        shot = next(c for c in candidates if c.reason == "shot")

        assert goal.must_include is True
        assert np.isclose(goal.score, 0.9)
        assert shot.must_include is False
        assert np.isclose(shot.score, 0.4)

    def test_respects_include_flags(self):
        """Include flags should suppress disabled event types."""
        events = [
            {"event_type": "goal", "timestamp": 100.0, "frame_idx": 3000, "confidence": 0.9},
            {"event_type": "shot", "timestamp": 90.0, "frame_idx": 2700, "confidence": 0.8},
        ]

        candidates = build_event_candidates(
            events=events,
            include_goals=True,
            include_shots=False,
            goal_weight=1.0,
            shot_weight=0.7,
            min_confidence=0.2,
        )

        assert len(candidates) == 1
        assert candidates[0].reason == "goal"


class TestBuildActionCandidates:
    """Tests for action-based candidate creation."""

    def test_detects_high_action_from_ball_speed_and_pressure(self):
        """Fast ball motion with nearby players should produce candidates."""
        tracks = []

        # Ball moves quickly over 6 frames.
        for i in range(6):
            x = 100 + i * 100
            tracks.append(
                {
                    "frame_idx": i,
                    "object_type": "ball",
                    "bbox": [x, 100, x + 10, 110],
                }
            )

            # Add player close to the ball each frame.
            tracks.append(
                {
                    "frame_idx": i,
                    "object_type": "player",
                    "bbox": [x + 5, 95, x + 35, 180],
                }
            )

        candidates = build_action_candidates(
            tracks=tracks,
            fps=10.0,
            min_speed_pixels_per_sec=150.0,
            player_pressure_radius=150.0,
            score_quantile=0.5,
            min_candidate_score=0.2,
            max_candidates=20,
        )

        assert len(candidates) > 0
        assert all(c.source == "action" for c in candidates)
        assert all(c.reason == "high_action" for c in candidates)
        assert all(c.score >= 0.2 for c in candidates)


class TestSegmentSelection:
    """Tests for merging and selecting highlight segments."""

    def test_merges_overlapping_candidates_and_keeps_must_include(self):
        """Nearby candidates should merge and must-include segments should survive top-N."""
        candidates = [
            HighlightCandidate(
                timestamp=10.0,
                score=0.9,
                source="event",
                reason="goal",
                frame_idx=300,
                must_include=True,
            ),
            HighlightCandidate(
                timestamp=11.0,
                score=0.6,
                source="audio",
                reason="crowd_spike",
                frame_idx=None,
                must_include=False,
            ),
            HighlightCandidate(
                timestamp=45.0,
                score=0.7,
                source="action",
                reason="high_action",
                frame_idx=1300,
                must_include=False,
            ),
        ]

        segments = build_segments_from_candidates(
            candidates=candidates,
            duration_seconds=90.0,
            pre_roll_seconds=5.0,
            post_roll_seconds=5.0,
            merge_gap_seconds=2.0,
        )

        assert len(segments) == 2
        assert segments[0].must_include is True
        assert "goal" in segments[0].reasons
        assert "crowd_spike" in segments[0].reasons

        selected = select_highlight_segments(
            segments=segments,
            top_n=1,
            min_segment_score=0.8,
        )

        assert len(selected) == 1
        assert selected[0].must_include is True
        assert "goal" in selected[0].reasons

