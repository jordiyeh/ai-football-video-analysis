"""Unit tests for per-player highlight reel generation."""

from src.events.player_reels import build_player_reels


class TestBuildPlayerReels:
    """Tests for player reel scoring and filtering."""

    def test_builds_reels_from_fused_assignments_and_segments(self):
        """Player with assigned tracks inside highlights should receive reel segments."""
        segments = [
            {
                "segment_id": "highlight_001",
                "start_time": 10.0,
                "end_time": 20.0,
                "duration": 10.0,
                "score": 0.9,
                "reasons": ["goal"],
                "sources": ["event"],
            },
            {
                "segment_id": "highlight_002",
                "start_time": 30.0,
                "end_time": 40.0,
                "duration": 10.0,
                "score": 0.6,
                "reasons": ["high_action"],
                "sources": ["action"],
            },
        ]

        assignments = [
            {"track_id": 10, "player_id": 101, "player_name": "Nick", "match_method": "auto", "confidence": 0.92},
            {"track_id": 11, "player_id": 101, "player_name": "Nick", "match_method": "suggested", "confidence": 0.85},
            {"track_id": 20, "player_id": 102, "player_name": "Sam", "match_method": "auto", "confidence": 0.95},
        ]

        tracks = []
        # Player 101 appears in both segments.
        for frame_idx in range(300, 451):  # 10.0s to 15.0s @30fps
            tracks.append(
                {
                    "frame_idx": frame_idx,
                    "timestamp": frame_idx / 30.0,
                    "object_type": "player",
                    "track_id": 10,
                }
            )
        for frame_idx in range(930, 1051):  # 31.0s to 35.0s
            tracks.append(
                {
                    "frame_idx": frame_idx,
                    "timestamp": frame_idx / 30.0,
                    "object_type": "player",
                    "track_id": 11,
                }
            )
        # Player 102 appears only briefly (below presence threshold).
        for frame_idx in range(315, 330):
            tracks.append(
                {
                    "frame_idx": frame_idx,
                    "timestamp": frame_idx / 30.0,
                    "object_type": "player",
                    "track_id": 20,
                }
            )

        reels, summary = build_player_reels(
            segments=segments,
            tracks=tracks,
            assignments=assignments,
            fps=30.0,
            max_segments_per_player=5,
            min_presence_seconds=1.0,
            min_player_segment_score=0.2,
            min_assignment_confidence=0.6,
            include_suggested=True,
        )

        assert summary["players_with_reels"] == 1
        assert summary["player_segments_total"] == 2
        assert len(reels) == 1
        assert reels[0]["player_id"] == 101
        assert reels[0]["player_name"] == "Nick"
        assert reels[0]["segment_count"] == 2
        assert reels[0]["segments"][0]["player_segment_score"] >= reels[0]["segments"][1]["player_segment_score"]

    def test_excludes_suggested_when_disabled(self):
        """Suggested assignments should be dropped when include_suggested is false."""
        segments = [
            {
                "segment_id": "highlight_001",
                "start_time": 5.0,
                "end_time": 10.0,
                "duration": 5.0,
                "score": 0.8,
            }
        ]
        assignments = [
            {"track_id": 10, "player_id": 101, "player_name": "Nick", "match_method": "suggested", "confidence": 0.9},
        ]
        tracks = [
            {
                "frame_idx": 180,
                "timestamp": 6.0,
                "object_type": "player",
                "track_id": 10,
            }
            for _ in range(40)
        ]

        reels, summary = build_player_reels(
            segments=segments,
            tracks=tracks,
            assignments=assignments,
            fps=30.0,
            include_suggested=False,
            min_presence_seconds=0.5,
            min_player_segment_score=0.1,
        )

        assert reels == []
        assert summary["players_with_reels"] == 0

