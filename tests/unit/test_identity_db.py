"""Unit tests for player identity database."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.identity.database import PlayerDatabase
from src.identity.models import MatchMetadata, MatchTag


class TestPlayerDatabase:
    """Tests for PlayerDatabase."""

    @pytest.fixture
    def db(self):
        """Create a temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_players.db"
            db = PlayerDatabase(db_path)
            yield db
            db.close()

    def test_create_player_basic(self, db):
        """Test creating a player without embedding."""
        player = db.create_player(
            name="John Doe",
            jersey_number=10,
            team_hint="ours",
        )

        assert player.player_id is not None
        assert player.name == "John Doe"
        assert player.jersey_number == 10
        assert player.team_hint == "ours"
        assert player.embedding_centroid is None
        assert player.embedding_count == 0

    def test_create_player_with_embedding(self, db):
        """Test creating a player with initial embedding."""
        embedding = np.random.randn(512).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        player = db.create_player(
            name="Jane Doe",
            embedding=embedding,
        )

        assert player.embedding_centroid is not None
        assert len(player.embedding_centroid) == 512
        assert player.embedding_count == 1

    def test_get_player(self, db):
        """Test retrieving a player."""
        created = db.create_player(name="Test Player")

        retrieved = db.get_player(created.player_id)

        assert retrieved is not None
        assert retrieved.player_id == created.player_id
        assert retrieved.name == "Test Player"

    def test_get_nonexistent_player(self, db):
        """Test retrieving a non-existent player."""
        player = db.get_player(9999)

        assert player is None

    def test_list_players(self, db):
        """Test listing all players."""
        db.create_player(name="Player 1")
        db.create_player(name="Player 2")
        db.create_player(name="Player 3")

        players = db.list_players()

        assert len(players) == 3
        names = [p.name for p in players]
        assert "Player 1" in names
        assert "Player 2" in names
        assert "Player 3" in names

    def test_find_player_by_name_and_number(self, db):
        """Test lookup by normalized name and jersey number."""
        p1 = db.create_player(name="Nicholas Oestringer", jersey_number=10)
        db.create_player(name="Another Player", jersey_number=10)

        found = db.find_player_by_name_and_number(name="  nicholas oestringer  ", jersey_number=10)

        assert found is not None
        assert found.player_id == p1.player_id

    def test_update_player(self, db):
        """Test updating player metadata."""
        player = db.create_player(name="Original Name")

        updated = db.update_player(
            player_id=player.player_id,
            name="New Name",
            jersey_number=7,
        )

        assert updated is not None
        assert updated.name == "New Name"
        assert updated.jersey_number == 7

    def test_update_player_centroid(self, db):
        """Test updating player embedding centroid."""
        # Create with initial embedding
        initial_emb = np.ones(512, dtype=np.float32)
        initial_emb = initial_emb / np.linalg.norm(initial_emb)

        player = db.create_player(embedding=initial_emb)
        assert player.embedding_count == 1

        # Update with new embedding
        new_emb = np.zeros(512, dtype=np.float32)
        new_emb[0] = 1.0  # Unit vector in first dimension

        updated = db.update_player_centroid(player.player_id, new_emb)

        assert updated is not None
        assert updated.embedding_count == 2
        # Centroid should be average of the two (normalized)
        assert updated.embedding_centroid is not None

    def test_set_player_centroid_direct(self, db):
        """Test directly setting player centroid and count."""
        player = db.create_player(name="Direct Centroid")
        assert player.embedding_count == 0

        centroid = np.random.randn(512).astype(np.float32)
        centroid = centroid / np.linalg.norm(centroid)

        updated = db.set_player_centroid_direct(player.player_id, centroid, 5)

        assert updated is not None
        assert updated.embedding_count == 5
        assert updated.embedding_centroid is not None
        assert len(updated.embedding_centroid) == 512
        # Verify it's normalized
        norm = np.linalg.norm(np.array(updated.embedding_centroid))
        assert abs(norm - 1.0) < 1e-4

    def test_set_player_centroid_direct_replaces(self, db):
        """Test that direct centroid set fully replaces previous values."""
        emb1 = np.ones(512, dtype=np.float32) / np.sqrt(512)
        player = db.create_player(name="Replace Test", embedding=emb1)
        assert player.embedding_count == 1

        emb2 = np.zeros(512, dtype=np.float32)
        emb2[0] = 1.0
        updated = db.set_player_centroid_direct(player.player_id, emb2, 10)

        assert updated is not None
        assert updated.embedding_count == 10
        # First element should dominate
        arr = np.array(updated.embedding_centroid)
        assert abs(arr[0] - 1.0) < 1e-4

    def test_set_player_centroid_direct_not_found(self, db):
        """Test set_player_centroid_direct returns None for missing player."""
        result = db.set_player_centroid_direct(9999, np.ones(512), 1)
        assert result is None

    def test_delete_player(self, db):
        """Test deleting a player."""
        player = db.create_player(name="To Delete")

        deleted = db.delete_player(player.player_id)

        assert deleted is True
        assert db.get_player(player.player_id) is None

    def test_delete_nonexistent_player(self, db):
        """Test deleting a non-existent player."""
        deleted = db.delete_player(9999)

        assert deleted is False

    def test_merge_players(self, db):
        """Test merging two players."""
        # Create two players with embeddings
        emb1 = np.random.randn(512).astype(np.float32)
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = np.random.randn(512).astype(np.float32)
        emb2 = emb2 / np.linalg.norm(emb2)

        player1 = db.create_player(name="Keep", embedding=emb1)
        player2 = db.create_player(name="Merge", embedding=emb2)

        # Create appearances for both
        db.create_appearance(
            video_id="vid1", run_name="run1", track_id=1,
            player_id=player1.player_id,
        )
        db.create_appearance(
            video_id="vid1", run_name="run1", track_id=2,
            player_id=player2.player_id,
        )

        # Merge
        kept = db.merge_players(player1.player_id, player2.player_id)

        assert kept is not None
        assert kept.player_id == player1.player_id
        assert kept.embedding_count == 2  # Combined embeddings

        # Merged player should be deleted
        assert db.get_player(player2.player_id) is None

        # Appearances should be transferred
        appearances = db.get_appearances_for_player(player1.player_id)
        assert len(appearances) == 2


class TestAppearance:
    """Tests for appearance operations."""

    @pytest.fixture
    def db(self):
        """Create a temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_players.db"
            db = PlayerDatabase(db_path)
            yield db
            db.close()

    def test_create_appearance(self, db):
        """Test creating an appearance."""
        appearance = db.create_appearance(
            video_id="test_video",
            run_name="test_run",
            track_id=1,
            frame_start=0,
            frame_end=100,
        )

        assert appearance.appearance_id is not None
        assert appearance.video_id == "test_video"
        assert appearance.track_id == 1
        assert appearance.player_id is None

    def test_create_appearance_with_player(self, db):
        """Test creating an appearance linked to a player."""
        player = db.create_player(name="Test Player")

        appearance = db.create_appearance(
            video_id="test_video",
            run_name="test_run",
            track_id=1,
            player_id=player.player_id,
            match_confidence=0.9,
            match_method="auto",
        )

        assert appearance.player_id == player.player_id
        assert appearance.match_confidence == 0.9
        assert appearance.match_method == "auto"

    def test_get_appearance(self, db):
        """Test retrieving an appearance."""
        created = db.create_appearance(
            video_id="test_video",
            run_name="test_run",
            track_id=1,
        )

        retrieved = db.get_appearance("test_video", 1)

        assert retrieved is not None
        assert retrieved.appearance_id == created.appearance_id

    def test_get_appearances_for_video(self, db):
        """Test getting all appearances for a video."""
        db.create_appearance(video_id="vid1", run_name="run1", track_id=1)
        db.create_appearance(video_id="vid1", run_name="run1", track_id=2)
        db.create_appearance(video_id="vid2", run_name="run1", track_id=1)

        appearances = db.get_appearances_for_video("vid1")

        assert len(appearances) == 2
        assert all(a.video_id == "vid1" for a in appearances)

    def test_get_appearances_for_player(self, db):
        """Test getting all appearances for a player."""
        player = db.create_player(name="Test")

        db.create_appearance(
            video_id="vid1", run_name="run1", track_id=1,
            player_id=player.player_id,
        )
        db.create_appearance(
            video_id="vid2", run_name="run1", track_id=1,
            player_id=player.player_id,
        )
        db.create_appearance(
            video_id="vid3", run_name="run1", track_id=1,
            player_id=None,  # Unassigned
        )

        appearances = db.get_appearances_for_player(player.player_id)

        assert len(appearances) == 2
        assert all(a.player_id == player.player_id for a in appearances)

    def test_assign_appearance_to_player(self, db):
        """Test manually assigning an appearance to a player."""
        player = db.create_player(name="Test")
        db.create_appearance(
            video_id="test_video", run_name="run1", track_id=1,
        )

        assigned = db.assign_appearance_to_player(
            video_id="test_video",
            track_id=1,
            player_id=player.player_id,
            confidence=1.0,
            method="manual",
        )

        assert assigned is not None
        assert assigned.player_id == player.player_id
        assert assigned.match_method == "manual"

    def test_delete_appearance(self, db):
        """Test deleting an appearance by video_id and track_id."""
        db.create_appearance(video_id="test_video", run_name="run1", track_id=1)

        deleted = db.delete_appearance(video_id="test_video", track_id=1)
        missing_after_delete = db.get_appearance("test_video", 1)

        assert deleted is True
        assert missing_after_delete is None
        assert db.delete_appearance(video_id="test_video", track_id=1) is False

    def test_upsert_appearance(self, db):
        """Test that creating appearance with same video_id/track_id updates."""
        db.create_appearance(
            video_id="test_video", run_name="run1", track_id=1,
            match_confidence=0.5,
        )

        # Create again with same video_id/track_id
        db.create_appearance(
            video_id="test_video", run_name="run1", track_id=1,
            match_confidence=0.9,
        )

        # Should have updated, not created new
        all_appearances = db.get_appearances_for_video("test_video")
        assert len(all_appearances) == 1
        assert all_appearances[0].match_confidence == 0.9


class TestPlayerWithAppearances:
    """Tests for getting player with appearances."""

    @pytest.fixture
    def db(self):
        """Create a temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_players.db"
            db = PlayerDatabase(db_path)
            yield db
            db.close()

    def test_get_player_with_appearances(self, db):
        """Test getting a player with all their appearances."""
        player = db.create_player(name="Test Player")

        # Create appearances
        db.create_appearance(
            video_id="vid1", run_name="run1", track_id=1,
            player_id=player.player_id,
        )
        db.create_appearance(
            video_id="vid2", run_name="run2", track_id=5,
            player_id=player.player_id,
        )

        player_with_apps = db.get_player_with_appearances(player.player_id)

        assert player_with_apps is not None
        assert player_with_apps.player_id == player.player_id
        assert len(player_with_apps.appearances) == 2

    def test_get_all_player_centroids(self, db):
        """Test getting all player centroids for matching."""
        # Create players with embeddings
        emb1 = np.random.randn(512).astype(np.float32)
        emb2 = np.random.randn(512).astype(np.float32)

        p1 = db.create_player(embedding=emb1)
        p2 = db.create_player(embedding=emb2)
        p3 = db.create_player()  # No embedding

        centroids = db.get_all_player_centroids()

        assert len(centroids) == 2  # Only players with embeddings
        assert p1.player_id in centroids
        assert p2.player_id in centroids
        assert p3.player_id not in centroids

    def test_photo_path_migration(self, db):
        """Test that photo_path column exists after schema migration."""
        player = db.create_player(name="Photo Test")
        retrieved = db.get_player(player.player_id)
        assert retrieved is not None
        assert retrieved.photo_path is None

    def test_set_player_photo(self, db):
        """Test setting and clearing a player's photo_path."""
        player = db.create_player(name="Photo Player")
        assert player.photo_path is None

        # Set photo
        updated = db.set_player_photo(player.player_id, "data/player_photos/1/photo.jpg")
        assert updated is not None
        assert updated.photo_path == "data/player_photos/1/photo.jpg"

        # Verify via get_player
        retrieved = db.get_player(player.player_id)
        assert retrieved.photo_path == "data/player_photos/1/photo.jpg"

        # Clear photo
        cleared = db.set_player_photo(player.player_id, None)
        assert cleared is not None
        assert cleared.photo_path is None

    def test_set_player_photo_nonexistent(self, db):
        """Test set_player_photo returns None for nonexistent player."""
        result = db.set_player_photo(9999, "some/path.jpg")
        assert result is None

    def test_photo_path_in_list_players(self, db):
        """Test that photo_path is included in list_players results."""
        p = db.create_player(name="Listed")
        db.set_player_photo(p.player_id, "data/player_photos/1/photo.png")
        players = db.list_players()
        assert len(players) == 1
        assert players[0].photo_path == "data/player_photos/1/photo.png"

    def test_photo_path_in_find_player(self, db):
        """Test that photo_path is included in find_player_by_name_and_number."""
        p = db.create_player(name="Findable", jersey_number=7)
        db.set_player_photo(p.player_id, "data/player_photos/1/photo.jpg")
        found = db.find_player_by_name_and_number(name="Findable")
        assert found is not None
        assert found.photo_path == "data/player_photos/1/photo.jpg"


class TestMatchMetadataAndTags:
    """Tests for match metadata and tags schema/migrations."""

    @pytest.fixture
    def db(self):
        """Create a temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_players.db"
            db = PlayerDatabase(db_path)
            yield db
            db.close()

    def test_schema_has_match_metadata_and_tags_tables(self, db):
        """Both new tables should exist after schema initialization."""
        cursor = db.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        table_names = {row["name"] for row in cursor.fetchall()}
        assert "match_metadata" in table_names
        assert "tags" in table_names

    def test_match_metadata_upsert_and_read(self, db):
        """Upsert should persist and then update a run's metadata."""
        created = db.upsert_match_metadata(
            run_name="run-001",
            video_id="video-1",
            competition="League",
            match_date="2026-02-08",
            season="2025-26",
            venue="Home Field",
            notes="First half heavy rain",
            extra={"weather": "rain", "referee": "A. Smith"},
        )

        assert isinstance(created, MatchMetadata)
        assert created.run_name == "run-001"
        assert created.extra == {"weather": "rain", "referee": "A. Smith"}

        updated = db.upsert_match_metadata(
            run_name="run-001",
            competition="Cup",
            notes="Updated notes",
            extra={"weather": "clear"},
        )
        assert updated.competition == "Cup"
        assert updated.notes == "Updated notes"
        assert updated.extra == {"weather": "clear"}

        fetched = db.get_match_metadata("run-001")
        assert fetched is not None
        assert fetched.competition == "Cup"
        assert fetched.extra == {"weather": "clear"}

    def test_delete_match_metadata(self, db):
        """Match metadata delete should be idempotent."""
        db.upsert_match_metadata(run_name="run-del")
        assert db.delete_match_metadata("run-del") is True
        assert db.get_match_metadata("run-del") is None
        assert db.delete_match_metadata("run-del") is False

    def test_create_list_update_delete_tags(self, db):
        """Tag CRUD and filters should work with metadata payloads."""
        player = db.create_player(name="Tagged Player")
        team = db.create_team(name="Tagged FC")
        tag = db.create_tag(
            run_name="run-tags",
            label="shot",
            category="event",
            start_time=10.0,
            end_time=11.5,
            frame_idx=250,
            track_id=7,
            player_id=player.player_id,
            team_id=team.team_id,
            confidence=0.87,
            source="manual",
            notes="Left foot",
            metadata={"xg": 0.31},
        )

        assert isinstance(tag, MatchTag)
        assert tag.label == "shot"
        assert tag.metadata == {"xg": 0.31}

        tags_for_run = db.list_tags(run_name="run-tags")
        assert len(tags_for_run) == 1
        assert tags_for_run[0].tag_id == tag.tag_id

        by_category = db.list_tags(run_name="run-tags", category="event")
        assert len(by_category) == 1
        assert by_category[0].tag_id == tag.tag_id

        by_time = db.list_tags(run_name="run-tags", min_time=11.0, max_time=12.0)
        assert len(by_time) == 1

        updated = db.update_tag(tag.tag_id, notes="Updated", metadata={"xg": 0.44})
        assert updated is not None
        assert updated.notes == "Updated"
        assert updated.metadata == {"xg": 0.44}

        assert db.delete_tag(tag.tag_id) is True
        assert db.get_tag(tag.tag_id) is None
        assert db.delete_tag(tag.tag_id) is False

    def test_tag_validation_rejects_invalid_time_window(self, db):
        """Invalid temporal windows should raise ValueError."""
        with pytest.raises(ValueError, match="end_time"):
            db.create_tag(
                run_name="run-invalid",
                label="bad",
                start_time=8.0,
                end_time=7.0,
            )

    def test_schema_migration_is_idempotent(self):
        """Re-opening the same DB should keep schema healthy and reusable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_players.db"

            with PlayerDatabase(db_path) as db:
                db.upsert_match_metadata(run_name="run-1", competition="League")
                db.create_tag(run_name="run-1", label="goal", start_time=42.0)

            with PlayerDatabase(db_path) as db:
                metadata_rows = db.list_match_metadata()
                tag_rows = db.list_tags(run_name="run-1")
                assert len(metadata_rows) == 1
                assert len(tag_rows) == 1
                assert metadata_rows[0].run_name == "run-1"
                assert tag_rows[0].label == "goal"
