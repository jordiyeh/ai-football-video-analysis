"""Unit tests for team/kit/run-team CRUD in PlayerDatabase."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.identity.database import PlayerDatabase
from src.identity.models import TeamWithKits


class TestTeamCRUD:
    """Tests for team create/read/update/delete."""

    @pytest.fixture
    def db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = PlayerDatabase(db_path)
            yield db
            db.close()

    def test_create_team(self, db):
        team = db.create_team(name="FC Test", short_name="TST")
        assert team.team_id is not None
        assert team.name == "FC Test"
        assert team.short_name == "TST"

    def test_create_team_duplicate_name_raises(self, db):
        db.create_team(name="FC Test")
        with pytest.raises(Exception, match="UNIQUE"):
            db.create_team(name="FC Test")

    def test_get_team(self, db):
        created = db.create_team(name="FC Test")
        fetched = db.get_team(created.team_id)
        assert fetched is not None
        assert fetched.name == "FC Test"

    def test_get_team_not_found(self, db):
        assert db.get_team(9999) is None

    def test_list_teams_empty(self, db):
        result = db.list_teams()
        assert result == []

    def test_list_teams_with_kits_and_player_count(self, db):
        team = db.create_team(name="FC Test")
        db.upsert_kit(team.team_id, "home", color_hex="#FF0000")
        player = db.create_player(name="Player 1")
        db.set_player_team(player.player_id, team.team_id)

        teams = db.list_teams()
        assert len(teams) == 1
        assert isinstance(teams[0], TeamWithKits)
        assert teams[0].player_count == 1
        assert len(teams[0].kits) == 1
        assert teams[0].kits[0].color_hex == "#FF0000"

    def test_update_team(self, db):
        team = db.create_team(name="FC Old")
        updated = db.update_team(team.team_id, name="FC New", short_name="NEW")
        assert updated is not None
        assert updated.name == "FC New"
        assert updated.short_name == "NEW"

    def test_update_team_not_found(self, db):
        assert db.update_team(9999, name="X") is None

    def test_update_team_no_change(self, db):
        team = db.create_team(name="FC Test")
        result = db.update_team(team.team_id)
        assert result.name == "FC Test"

    def test_delete_team(self, db):
        team = db.create_team(name="FC Test")
        assert db.delete_team(team.team_id) is True
        assert db.get_team(team.team_id) is None

    def test_delete_team_not_found(self, db):
        assert db.delete_team(9999) is False

    def test_delete_team_cascades_kits(self, db):
        team = db.create_team(name="FC Test")
        db.upsert_kit(team.team_id, "home")
        db.delete_team(team.team_id)
        assert db.get_kits_for_team(team.team_id) == []

    def test_delete_team_nulls_player_team_id(self, db):
        team = db.create_team(name="FC Test")
        player = db.create_player(name="Player")
        db.set_player_team(player.player_id, team.team_id)
        db.delete_team(team.team_id)
        p = db.get_player(player.player_id)
        assert p.team_id is None

    def test_delete_team_cascades_run_teams(self, db):
        t1 = db.create_team(name="Home")
        t2 = db.create_team(name="Away")
        db.set_run_teams("run1", t1.team_id, t2.team_id)
        db.delete_team(t1.team_id)
        assocs = db.get_run_teams("run1")
        assert all(a.team_id != t1.team_id for a in assocs)

    def test_create_team_with_logo_path(self, db):
        team = db.create_team(name="FC Logo", logo_path="data/team_logos/1/logo.png")
        assert team.logo_path == "data/team_logos/1/logo.png"

    def test_set_team_logo(self, db):
        team = db.create_team(name="FC Test")
        assert team.logo_path is None
        updated = db.set_team_logo(team.team_id, "data/team_logos/1/logo.png")
        assert updated is not None
        assert updated.logo_path == "data/team_logos/1/logo.png"

    def test_set_team_logo_clear(self, db):
        team = db.create_team(name="FC Test", logo_path="data/team_logos/1/logo.png")
        updated = db.set_team_logo(team.team_id, None)
        assert updated.logo_path is None

    def test_set_team_logo_not_found(self, db):
        assert db.set_team_logo(9999, "path.png") is None

    def test_logo_path_persists_in_get_team(self, db):
        team = db.create_team(name="FC Test")
        db.set_team_logo(team.team_id, "logos/test.png")
        fetched = db.get_team(team.team_id)
        assert fetched.logo_path == "logos/test.png"

    def test_logo_path_in_list_teams(self, db):
        db.create_team(name="FC Logo", logo_path="logos/1.png")
        teams = db.list_teams()
        assert len(teams) == 1
        assert teams[0].logo_path == "logos/1.png"


class TestKitCRUD:
    """Tests for kit create/read/delete."""

    @pytest.fixture
    def db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = PlayerDatabase(db_path)
            yield db
            db.close()

    def test_upsert_kit_basic(self, db):
        team = db.create_team(name="FC Test")
        kit = db.upsert_kit(team.team_id, "home", color_hex="#FF0000")
        assert kit.kit_id is not None
        assert kit.team_id == team.team_id
        assert kit.kit_type == "home"
        assert kit.color_hex == "#FF0000"

    def test_upsert_kit_with_hsv(self, db):
        team = db.create_team(name="FC Test")
        hsv = np.array([120.0, 200.0, 180.0], dtype=np.float32)
        kit = db.upsert_kit(team.team_id, "away", dominant_color_hsv=hsv, color_hex="#00AA00")
        assert kit.dominant_color_hsv is not None
        assert len(kit.dominant_color_hsv) == 3
        assert abs(kit.dominant_color_hsv[0] - 120.0) < 0.01

    def test_upsert_kit_replaces_existing(self, db):
        team = db.create_team(name="FC Test")
        db.upsert_kit(team.team_id, "home", color_hex="#FF0000")
        db.upsert_kit(team.team_id, "home", color_hex="#00FF00")
        kits = db.get_kits_for_team(team.team_id)
        assert len(kits) == 1
        assert kits[0].color_hex == "#00FF00"

    def test_get_kits_for_team(self, db):
        team = db.create_team(name="FC Test")
        db.upsert_kit(team.team_id, "home", color_hex="#FF0000")
        db.upsert_kit(team.team_id, "away", color_hex="#0000FF")
        kits = db.get_kits_for_team(team.team_id)
        assert len(kits) == 2

    def test_get_kit_specific(self, db):
        team = db.create_team(name="FC Test")
        db.upsert_kit(team.team_id, "home", color_hex="#FF0000")
        kit = db.get_kit(team.team_id, "home")
        assert kit is not None
        assert kit.color_hex == "#FF0000"
        assert db.get_kit(team.team_id, "third") is None

    def test_delete_kit(self, db):
        team = db.create_team(name="FC Test")
        db.upsert_kit(team.team_id, "home")
        assert db.delete_kit(team.team_id, "home") is True
        assert db.get_kit(team.team_id, "home") is None

    def test_delete_kit_not_found(self, db):
        team = db.create_team(name="FC Test")
        assert db.delete_kit(team.team_id, "third") is False


class TestRunTeamAssociation:
    """Tests for run-team association CRUD."""

    @pytest.fixture
    def db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = PlayerDatabase(db_path)
            yield db
            db.close()

    def test_set_run_teams(self, db):
        t1 = db.create_team(name="Home FC")
        t2 = db.create_team(name="Away FC")
        assocs = db.set_run_teams("match_01", t1.team_id, t2.team_id, "home", "away")
        assert len(assocs) == 2
        roles = {a.role: a for a in assocs}
        assert roles["home"].team_id == t1.team_id
        assert roles["away"].team_id == t2.team_id
        assert roles["home"].active_kit == "home"
        assert roles["away"].active_kit == "away"

    def test_get_run_teams(self, db):
        t1 = db.create_team(name="Home")
        t2 = db.create_team(name="Away")
        db.set_run_teams("run1", t1.team_id, t2.team_id)
        assocs = db.get_run_teams("run1")
        assert len(assocs) == 2

    def test_get_run_teams_empty(self, db):
        assert db.get_run_teams("nonexistent") == []

    def test_set_run_teams_replaces(self, db):
        t1 = db.create_team(name="Home")
        t2 = db.create_team(name="Away")
        t3 = db.create_team(name="New Home")
        db.set_run_teams("run1", t1.team_id, t2.team_id)
        db.set_run_teams("run1", t3.team_id, t2.team_id)
        assocs = db.get_run_teams("run1")
        home = [a for a in assocs if a.role == "home"][0]
        assert home.team_id == t3.team_id

    def test_update_run_team_cluster(self, db):
        t1 = db.create_team(name="Home")
        t2 = db.create_team(name="Away")
        db.set_run_teams("run1", t1.team_id, t2.team_id)
        assert db.update_run_team_cluster("run1", "home", 0) is True
        assert db.update_run_team_cluster("run1", "away", 1) is True
        assocs = db.get_run_teams("run1")
        cluster_map = {a.role: a.cluster_id for a in assocs}
        assert cluster_map["home"] == 0
        assert cluster_map["away"] == 1


class TestPlayerTeamLinking:
    """Tests for player-team linking."""

    @pytest.fixture
    def db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = PlayerDatabase(db_path)
            yield db
            db.close()

    def test_set_player_team(self, db):
        team = db.create_team(name="FC Test")
        player = db.create_player(name="Player 1")
        result = db.set_player_team(player.player_id, team.team_id)
        assert result is not None
        assert result.team_id == team.team_id

    def test_set_player_team_to_none(self, db):
        team = db.create_team(name="FC Test")
        player = db.create_player(name="Player 1")
        db.set_player_team(player.player_id, team.team_id)
        result = db.set_player_team(player.player_id, None)
        assert result.team_id is None

    def test_set_player_team_not_found(self, db):
        assert db.set_player_team(9999, 1) is None

    def test_player_team_id_in_list(self, db):
        team = db.create_team(name="FC Test")
        player = db.create_player(name="Player 1")
        db.set_player_team(player.player_id, team.team_id)
        players = db.list_players()
        assert players[0].team_id == team.team_id


class TestSchemaMigration:
    """Test that schema v2.1 migration works on fresh and existing databases."""

    def test_fresh_database_has_teams_table(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            with PlayerDatabase(db_path) as db:
                # Should be able to create a team
                team = db.create_team(name="Test")
                assert team.team_id is not None

    def test_schema_version_is_2_1(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            with PlayerDatabase(db_path) as db:
                cursor = db.conn.cursor()
                cursor.execute("SELECT version FROM schema_version")
                row = cursor.fetchone()
                assert row["version"] == "2.1"

    def test_players_have_team_id_column(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            with PlayerDatabase(db_path) as db:
                player = db.create_player(name="Test")
                assert player.team_id is None
                # The column exists
                cursor = db.conn.cursor()
                cursor.execute("PRAGMA table_info(players)")
                columns = [row["name"] for row in cursor.fetchall()]
                assert "team_id" in columns
