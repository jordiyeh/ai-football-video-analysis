"""Integration tests for team assignment pipeline."""

import pytest
import numpy as np

from src.vision.team.clustering import TeamAssigner, collect_track_colors
from src.vision.team.colors import extract_jersey_color, bgr_to_hsv


class TestTeamAssignmentPipeline:
    """Integration tests for team assignment from colors to team IDs."""

    @pytest.fixture
    def red_team_frame(self):
        """Create a frame with red player regions."""
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # Red players at specific locations
        frame[100:200, 100:150] = [0, 0, 255]  # Player 1
        frame[100:200, 200:250] = [0, 0, 240]  # Player 2 (slightly different red)
        frame[100:200, 300:350] = [20, 20, 250]  # Player 3 (red with slight variation)
        return frame

    @pytest.fixture
    def blue_team_frame(self):
        """Create a frame with blue player regions."""
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # Blue players at specific locations
        frame[100:200, 500:550] = [255, 0, 0]  # Player 4
        frame[100:200, 600:650] = [240, 0, 20]  # Player 5 (slightly different blue)
        frame[100:200, 700:750] = [250, 20, 20]  # Player 6 (blue with slight variation)
        return frame

    def test_full_team_assignment_pipeline(self, red_team_frame, blue_team_frame):
        """Test complete pipeline from frames to team assignments."""
        # Combined frame with both teams
        frame = red_team_frame.copy()
        frame[100:200, 500:550] = [255, 0, 0]
        frame[100:200, 600:650] = [240, 0, 20]
        frame[100:200, 700:750] = [250, 20, 20]

        # Simulate tracks
        tracks_by_frame = {
            0: [
                {"track_id": 1, "bbox": (100.0, 100.0, 150.0, 200.0), "object_type": "player"},
                {"track_id": 2, "bbox": (200.0, 100.0, 250.0, 200.0), "object_type": "player"},
                {"track_id": 3, "bbox": (300.0, 100.0, 350.0, 200.0), "object_type": "player"},
                {"track_id": 4, "bbox": (500.0, 100.0, 550.0, 200.0), "object_type": "player"},
                {"track_id": 5, "bbox": (600.0, 100.0, 650.0, 200.0), "object_type": "player"},
                {"track_id": 6, "bbox": (700.0, 100.0, 750.0, 200.0), "object_type": "player"},
            ],
        }

        # Add more frames for sufficient samples
        frames = {i: frame.copy() for i in range(10)}
        tracks_by_frame = {i: tracks_by_frame[0] for i in range(10)}

        # Collect colors
        track_colors = collect_track_colors(
            tracks_by_frame,
            frames,
            extract_jersey_color,
        )

        # Fit team assigner
        assigner = TeamAssigner(n_teams=2, color_space="bgr")
        assigner.fit(track_colors)

        # Check team assignments
        # Red team (tracks 1, 2, 3) should be same team
        team_1 = assigner.get_team_label(1)
        team_2 = assigner.get_team_label(2)
        team_3 = assigner.get_team_label(3)

        assert team_1 == team_2 == team_3, "Red team players should be same team"

        # Blue team (tracks 4, 5, 6) should be same team
        team_4 = assigner.get_team_label(4)
        team_5 = assigner.get_team_label(5)
        team_6 = assigner.get_team_label(6)

        assert team_4 == team_5 == team_6, "Blue team players should be same team"

        # Red and blue should be different teams
        assert team_1 != team_4, "Red and blue teams should be different"

    def test_team_color_consistency_across_frames(self):
        """Test that team colors remain consistent across multiple frames."""
        # Create frames with slightly varying colors (realistic scenario)
        np.random.seed(42)

        frames = {}
        base_red = np.array([0, 0, 220])
        base_blue = np.array([220, 0, 0])

        for i in range(20):
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            # Red player with slight variation
            variation = np.random.randint(-20, 20, 3)
            frame[100:200, 100:150] = np.clip(base_red + variation, 0, 255)
            # Blue player with slight variation
            variation = np.random.randint(-20, 20, 3)
            frame[100:200, 500:550] = np.clip(base_blue + variation, 0, 255)
            frames[i] = frame

        tracks_by_frame = {
            i: [
                {"track_id": 1, "bbox": (100.0, 100.0, 150.0, 200.0), "object_type": "player"},
                {"track_id": 2, "bbox": (500.0, 100.0, 550.0, 200.0), "object_type": "player"},
            ]
            for i in range(20)
        }

        # Collect colors
        track_colors = collect_track_colors(
            tracks_by_frame,
            frames,
            extract_jersey_color,
        )

        # Fit team assigner
        assigner = TeamAssigner(n_teams=2, color_space="bgr")
        assigner.fit(track_colors)

        # Teams should still be correctly separated
        assert assigner.get_team_label(1) != assigner.get_team_label(2)

    def test_predict_new_player_color(self):
        """Test predicting team for a new player based on their color."""
        # Setup known team colors
        track_colors = {
            1: [np.array([0, 0, 220])] * 10,  # Red team
            2: [np.array([0, 0, 210])] * 10,
            3: [np.array([220, 0, 0])] * 10,  # Blue team
            4: [np.array([210, 0, 0])] * 10,
        }

        assigner = TeamAssigner(n_teams=2, color_space="bgr")
        assigner.fit(track_colors)

        # New red player
        new_red = np.array([10, 10, 200])
        predicted_team = assigner.predict(new_red)

        # Should match red team
        assert predicted_team == assigner.get_team_label(1)

        # New blue player
        new_blue = np.array([200, 10, 10])
        predicted_team = assigner.predict(new_blue)

        # Should match blue team
        assert predicted_team == assigner.get_team_label(3)


class TestColorExtractionIntegration:
    """Integration tests for color extraction pipeline."""

    def test_jersey_color_extraction_from_varied_frame(self):
        """Test jersey color extraction from a frame with multiple elements."""
        frame = np.zeros((500, 500, 3), dtype=np.uint8)

        # Background (green grass)
        frame[:, :] = [0, 150, 0]

        # Player 1 (red jersey, blue shorts)
        frame[50:120, 100:140] = [0, 0, 200]  # Upper body (red)
        frame[120:180, 100:140] = [200, 0, 0]  # Lower body (blue)

        # Player 2 (white jersey)
        frame[50:180, 300:340] = [255, 255, 255]

        # Extract colors using "upper" region (jersey area)
        player1_bbox = (100.0, 50.0, 140.0, 180.0)
        player2_bbox = (300.0, 50.0, 340.0, 180.0)

        color1 = extract_jersey_color(frame, player1_bbox, sample_region="upper")
        color2 = extract_jersey_color(frame, player2_bbox, sample_region="upper")

        # Player 1 should be red (upper region)
        assert color1[2] > 150, "Player 1 should have high red"

        # Player 2 should be white
        assert all(c > 200 for c in color2), "Player 2 should be white"

    def test_hsv_conversion_pipeline(self):
        """Test full color conversion pipeline BGR -> HSV."""
        # Pure red in BGR
        red_bgr = np.array([0, 0, 255])
        red_hsv = bgr_to_hsv(red_bgr)

        # Check HSV values for red
        assert red_hsv[0] < 10 or red_hsv[0] > 170  # Hue near 0 or 180
        assert red_hsv[1] > 200  # High saturation
        assert red_hsv[2] > 200  # High value

        # Pure green in BGR
        green_bgr = np.array([0, 255, 0])
        green_hsv = bgr_to_hsv(green_bgr)

        # Check HSV values for green
        assert 50 < green_hsv[0] < 70  # Hue around 60
        assert green_hsv[1] > 200
        assert green_hsv[2] > 200
