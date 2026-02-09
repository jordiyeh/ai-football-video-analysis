"""SQLite database for player identity persistence."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from src.identity.models import (
    Appearance,
    MatchMetadata,
    MatchTag,
    Player,
    PlayerWithAppearances,
    Team,
    TeamKit,
    TeamWithKits,
    RunTeamAssociation,
)


def _serialize_embedding(embedding: np.ndarray | list[float] | None) -> bytes | None:
    """Serialize embedding to bytes for SQLite BLOB storage."""
    if embedding is None:
        return None
    if isinstance(embedding, list):
        embedding = np.array(embedding, dtype=np.float32)
    return embedding.astype(np.float32).tobytes()


def _deserialize_embedding(blob: bytes | None) -> list[float] | None:
    """Deserialize embedding from SQLite BLOB."""
    if blob is None:
        return None
    arr = np.frombuffer(blob, dtype=np.float32)
    return arr.tolist()


def _serialize_json(value: dict[str, Any] | None) -> str | None:
    """Serialize dictionary payload for SQLite TEXT storage."""
    if value is None:
        return None
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _deserialize_json(value: str | None) -> dict[str, Any] | None:
    """Deserialize dictionary payload from SQLite TEXT storage."""
    if value is None:
        return None
    parsed = json.loads(value)
    return parsed if isinstance(parsed, dict) else None


_UNSET = object()


class PlayerDatabase:
    """
    SQLite database for persistent player identities.

    Manages players and their appearances across videos.
    """

    SCHEMA_VERSION = "2.1"

    def __init__(self, db_path: str | Path):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._ensure_schema()

    @property
    def conn(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self):
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _column_exists(self, table_name: str, column_name: str) -> bool:
        """Check whether a column exists for a table."""
        cursor = self.conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        for row in cursor.fetchall():
            if row["name"] == column_name:
                return True
        return False

    def _ensure_column(self, table_name: str, column_name: str, column_definition: str) -> None:
        """Add column to an existing table if missing."""
        if not self._column_exists(table_name, column_name):
            cursor = self.conn.cursor()
            cursor.execute(
                f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}"
            )

    def _ensure_schema(self):
        """Create database schema if not exists."""
        cursor = self.conn.cursor()

        # Create players table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS players (
                player_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                jersey_number INTEGER,
                team_hint TEXT,
                embedding_centroid BLOB,
                embedding_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create appearances table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS appearances (
                appearance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL,
                run_name TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                player_id INTEGER,
                match_confidence REAL,
                match_method TEXT,
                frame_start INTEGER,
                frame_end INTEGER,
                embedding BLOB,
                UNIQUE(video_id, track_id),
                FOREIGN KEY (player_id) REFERENCES players(player_id)
            )
        """)

        # Create indices
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_appearances_video
            ON appearances(video_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_appearances_player
            ON appearances(player_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_appearances_run
            ON appearances(run_name)
        """)

        # Teams table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS teams (
                team_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                short_name TEXT,
                logo_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Team kits table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS team_kits (
                kit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_id INTEGER NOT NULL,
                kit_type TEXT NOT NULL CHECK(kit_type IN ('home', 'away', 'third')),
                image_path TEXT,
                dominant_color_hsv BLOB,
                secondary_color_hsv BLOB,
                color_hex TEXT,
                secondary_color_hex TEXT,
                UNIQUE(team_id, kit_type),
                FOREIGN KEY (team_id) REFERENCES teams(team_id) ON DELETE CASCADE
            )
        """)

        # Run-team associations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS run_teams (
                run_name TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('home', 'away')),
                team_id INTEGER NOT NULL,
                active_kit TEXT DEFAULT 'home',
                cluster_id INTEGER,
                UNIQUE(run_name, role),
                FOREIGN KEY (team_id) REFERENCES teams(team_id) ON DELETE CASCADE
            )
        """)

        # Match metadata table (per run)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS match_metadata (
                run_name TEXT PRIMARY KEY,
                video_id TEXT,
                match_date TEXT,
                competition TEXT,
                season TEXT,
                venue TEXT,
                home_team_id INTEGER,
                away_team_id INTEGER,
                notes TEXT,
                extra_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (home_team_id) REFERENCES teams(team_id) ON DELETE SET NULL,
                FOREIGN KEY (away_team_id) REFERENCES teams(team_id) ON DELETE SET NULL
            )
        """)

        # Match tags table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tags (
                tag_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_name TEXT NOT NULL,
                label TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT 'general',
                start_time REAL,
                end_time REAL,
                frame_idx INTEGER,
                track_id INTEGER,
                player_id INTEGER,
                team_id INTEGER,
                confidence REAL,
                source TEXT NOT NULL DEFAULT 'manual' CHECK(source IN ('manual', 'auto', 'imported')),
                notes TEXT,
                metadata_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (player_id) REFERENCES players(player_id) ON DELETE SET NULL,
                FOREIGN KEY (team_id) REFERENCES teams(team_id) ON DELETE SET NULL
            )
        """)

        # Column migrations kept idempotent for existing databases.
        # Run these before dependent index creation so legacy schemas upgrade safely.
        self._ensure_column("players", "team_id", "INTEGER")
        self._ensure_column("players", "photo_path", "TEXT")
        self._ensure_column("teams", "logo_path", "TEXT")
        self._ensure_column("match_metadata", "extra_json", "TEXT")
        self._ensure_column("tags", "category", "TEXT NOT NULL DEFAULT 'general'")
        self._ensure_column("tags", "metadata_json", "TEXT")
        self._ensure_column("tags", "source", "TEXT NOT NULL DEFAULT 'manual'")
        self._ensure_column("tags", "updated_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        self._ensure_column("tags", "notes", "TEXT")

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tags_run_name
            ON tags(run_name)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tags_category
            ON tags(category)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tags_label
            ON tags(label)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tags_time
            ON tags(start_time, end_time)
        """)

        # Create schema_version table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version TEXT PRIMARY KEY
            )
        """)
        cursor.execute(
            "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
            (self.SCHEMA_VERSION,),
        )

        self.conn.commit()

    # Player CRUD operations

    def create_player(
        self,
        name: str | None = None,
        jersey_number: int | None = None,
        team_hint: Literal["ours", "opponent"] | None = None,
        embedding: np.ndarray | list[float] | None = None,
    ) -> Player:
        """
        Create a new player.

        Args:
            name: Player name (optional).
            jersey_number: Jersey number (optional).
            team_hint: Team hint ('ours' or 'opponent').
            embedding: Initial embedding (optional).

        Returns:
            Created Player object.
        """
        cursor = self.conn.cursor()

        embedding_blob = _serialize_embedding(embedding)
        embedding_count = 1 if embedding is not None else 0
        now = datetime.utcnow()

        cursor.execute(
            """
            INSERT INTO players (name, jersey_number, team_hint, embedding_centroid,
                                 embedding_count, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (name, jersey_number, team_hint, embedding_blob, embedding_count, now, now),
        )
        self.conn.commit()

        player_id = cursor.lastrowid
        return Player(
            player_id=player_id,
            name=name,
            jersey_number=jersey_number,
            team_hint=team_hint,
            team_id=None,
            embedding_centroid=_deserialize_embedding(embedding_blob),
            embedding_count=embedding_count,
            created_at=now,
            updated_at=now,
        )

    def get_player(self, player_id: int) -> Player | None:
        """Get a player by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM players WHERE player_id = ?", (player_id,))
        row = cursor.fetchone()

        if row is None:
            return None

        return Player(
            player_id=row["player_id"],
            name=row["name"],
            jersey_number=row["jersey_number"],
            team_hint=row["team_hint"],
            team_id=row["team_id"] if "team_id" in row.keys() else None,
            photo_path=row["photo_path"] if "photo_path" in row.keys() else None,
            embedding_centroid=_deserialize_embedding(row["embedding_centroid"]),
            embedding_count=row["embedding_count"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def get_player_with_appearances(self, player_id: int) -> PlayerWithAppearances | None:
        """Get a player with all their appearances."""
        player = self.get_player(player_id)
        if player is None:
            return None

        appearances = self.get_appearances_for_player(player_id)

        return PlayerWithAppearances(
            **player.model_dump(),
            appearances=appearances,
        )

    def list_players(self) -> list[Player]:
        """List all players."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM players ORDER BY player_id")
        rows = cursor.fetchall()

        return [
            Player(
                player_id=row["player_id"],
                name=row["name"],
                jersey_number=row["jersey_number"],
                team_hint=row["team_hint"],
                team_id=row["team_id"] if "team_id" in row.keys() else None,
                photo_path=row["photo_path"] if "photo_path" in row.keys() else None,
                embedding_centroid=_deserialize_embedding(row["embedding_centroid"]),
                embedding_count=row["embedding_count"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    def find_player_by_name_and_number(
        self,
        name: str | None = None,
        jersey_number: int | None = None,
    ) -> Player | None:
        """
        Find a player by normalized name and/or jersey number.

        Args:
            name: Player name hint.
            jersey_number: Jersey number hint.

        Returns:
            Matching player or None.
        """
        normalized_name = name.strip() if isinstance(name, str) and name.strip() else None
        if normalized_name is None and jersey_number is None:
            return None

        query = "SELECT * FROM players WHERE 1=1"
        values: list[object] = []
        if normalized_name is not None:
            query += " AND lower(name) = lower(?)"
            values.append(normalized_name)
        if jersey_number is not None:
            query += " AND jersey_number = ?"
            values.append(jersey_number)

        query += " ORDER BY player_id LIMIT 1"

        cursor = self.conn.cursor()
        cursor.execute(query, values)
        row = cursor.fetchone()
        if row is None:
            return None

        return Player(
            player_id=row["player_id"],
            name=row["name"],
            jersey_number=row["jersey_number"],
            team_hint=row["team_hint"],
            team_id=row["team_id"] if "team_id" in row.keys() else None,
            photo_path=row["photo_path"] if "photo_path" in row.keys() else None,
            embedding_centroid=_deserialize_embedding(row["embedding_centroid"]),
            embedding_count=row["embedding_count"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def update_player(
        self,
        player_id: int,
        name: str | None = None,
        jersey_number: int | None = None,
        team_hint: Literal["ours", "opponent"] | None = None,
    ) -> Player | None:
        """
        Update player metadata (not embedding).

        Args:
            player_id: Player ID to update.
            name: New name (None to keep existing).
            jersey_number: New jersey number (None to keep existing).
            team_hint: New team hint (None to keep existing).

        Returns:
            Updated Player object or None if not found.
        """
        player = self.get_player(player_id)
        if player is None:
            return None

        # Build update query dynamically
        updates = []
        values = []

        if name is not None:
            updates.append("name = ?")
            values.append(name)
        if jersey_number is not None:
            updates.append("jersey_number = ?")
            values.append(jersey_number)
        if team_hint is not None:
            updates.append("team_hint = ?")
            values.append(team_hint)

        if not updates:
            return player

        updates.append("updated_at = ?")
        values.append(datetime.utcnow())
        values.append(player_id)

        cursor = self.conn.cursor()
        cursor.execute(
            f"UPDATE players SET {', '.join(updates)} WHERE player_id = ?",
            values,
        )
        self.conn.commit()

        return self.get_player(player_id)

    def update_player_centroid(
        self,
        player_id: int,
        new_embedding: np.ndarray | list[float],
    ) -> Player | None:
        """
        Update player embedding centroid using running average.

        Args:
            player_id: Player ID to update.
            new_embedding: New embedding to incorporate.

        Returns:
            Updated Player object or None if not found.
        """
        player = self.get_player(player_id)
        if player is None:
            return None

        if isinstance(new_embedding, list):
            new_embedding = np.array(new_embedding, dtype=np.float32)

        # Compute running average
        if player.embedding_centroid is None or player.embedding_count == 0:
            new_centroid = new_embedding
            new_count = 1
        else:
            current_centroid = np.array(player.embedding_centroid, dtype=np.float32)
            n = player.embedding_count
            new_centroid = (current_centroid * n + new_embedding) / (n + 1)
            new_count = n + 1

        # Normalize centroid
        new_centroid = new_centroid / np.linalg.norm(new_centroid)

        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE players
            SET embedding_centroid = ?, embedding_count = ?, updated_at = ?
            WHERE player_id = ?
            """,
            (
                _serialize_embedding(new_centroid),
                new_count,
                datetime.utcnow(),
                player_id,
            ),
        )
        self.conn.commit()

        return self.get_player(player_id)

    def set_player_centroid_direct(
        self,
        player_id: int,
        centroid: np.ndarray | list[float],
        embedding_count: int,
    ) -> Player | None:
        """Directly set player embedding centroid and count (full recompute, not running avg).

        Args:
            player_id: Player ID to update.
            centroid: Pre-computed centroid (should be L2-normalized).
            embedding_count: Number of embeddings used to compute the centroid.

        Returns:
            Updated Player object or None if not found.
        """
        player = self.get_player(player_id)
        if player is None:
            return None

        if isinstance(centroid, list):
            centroid = np.array(centroid, dtype=np.float32)

        # Normalize centroid
        norm = np.linalg.norm(centroid)
        if norm > 1e-8:
            centroid = centroid / norm

        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE players
            SET embedding_centroid = ?, embedding_count = ?, updated_at = ?
            WHERE player_id = ?
            """,
            (
                _serialize_embedding(centroid),
                embedding_count,
                datetime.utcnow(),
                player_id,
            ),
        )
        self.conn.commit()

        return self.get_player(player_id)

    def delete_player(self, player_id: int) -> bool:
        """
        Delete a player and unlink their appearances.

        Args:
            player_id: Player ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        cursor = self.conn.cursor()

        # Unlink appearances
        cursor.execute(
            "UPDATE appearances SET player_id = NULL WHERE player_id = ?",
            (player_id,),
        )

        # Delete player
        cursor.execute("DELETE FROM players WHERE player_id = ?", (player_id,))
        self.conn.commit()

        return cursor.rowcount > 0

    def merge_players(self, keep_id: int, merge_id: int) -> Player | None:
        """
        Merge two players, keeping one and transferring appearances.

        Args:
            keep_id: Player ID to keep.
            merge_id: Player ID to merge (will be deleted).

        Returns:
            The kept player with updated centroid, or None if either not found.
        """
        keep_player = self.get_player(keep_id)
        merge_player = self.get_player(merge_id)

        if keep_player is None or merge_player is None:
            return None

        cursor = self.conn.cursor()

        # Transfer appearances
        cursor.execute(
            "UPDATE appearances SET player_id = ? WHERE player_id = ?",
            (keep_id, merge_id),
        )

        # Merge centroids (weighted average)
        if (
            keep_player.embedding_centroid is not None
            and merge_player.embedding_centroid is not None
        ):
            keep_emb = np.array(keep_player.embedding_centroid, dtype=np.float32)
            merge_emb = np.array(merge_player.embedding_centroid, dtype=np.float32)
            keep_n = keep_player.embedding_count
            merge_n = merge_player.embedding_count

            total = keep_n + merge_n
            if total > 0:
                new_centroid = (keep_emb * keep_n + merge_emb * merge_n) / total
                new_centroid = new_centroid / np.linalg.norm(new_centroid)

                cursor.execute(
                    """
                    UPDATE players
                    SET embedding_centroid = ?, embedding_count = ?, updated_at = ?
                    WHERE player_id = ?
                    """,
                    (
                        _serialize_embedding(new_centroid),
                        total,
                        datetime.utcnow(),
                        keep_id,
                    ),
                )

        # Delete merged player
        cursor.execute("DELETE FROM players WHERE player_id = ?", (merge_id,))
        self.conn.commit()

        return self.get_player(keep_id)

    # Appearance CRUD operations

    def create_appearance(
        self,
        video_id: str,
        run_name: str,
        track_id: int,
        player_id: int | None = None,
        match_confidence: float | None = None,
        match_method: Literal["auto", "suggested", "manual"] | None = None,
        frame_start: int | None = None,
        frame_end: int | None = None,
        embedding: np.ndarray | list[float] | None = None,
    ) -> Appearance:
        """
        Create or update an appearance record.

        Uses INSERT OR REPLACE to handle unique constraint on (video_id, track_id).
        """
        cursor = self.conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO appearances
            (video_id, run_name, track_id, player_id, match_confidence,
             match_method, frame_start, frame_end, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                video_id,
                run_name,
                track_id,
                player_id,
                match_confidence,
                match_method,
                frame_start,
                frame_end,
                _serialize_embedding(embedding),
            ),
        )
        self.conn.commit()

        return Appearance(
            appearance_id=cursor.lastrowid,
            video_id=video_id,
            run_name=run_name,
            track_id=track_id,
            player_id=player_id,
            match_confidence=match_confidence,
            match_method=match_method,
            frame_start=frame_start,
            frame_end=frame_end,
            embedding=embedding if isinstance(embedding, list) else (
                embedding.tolist() if embedding is not None else None
            ),
        )

    def get_appearance(self, video_id: str, track_id: int) -> Appearance | None:
        """Get an appearance by video_id and track_id."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM appearances WHERE video_id = ? AND track_id = ?",
            (video_id, track_id),
        )
        row = cursor.fetchone()

        if row is None:
            return None

        return Appearance(
            appearance_id=row["appearance_id"],
            video_id=row["video_id"],
            run_name=row["run_name"],
            track_id=row["track_id"],
            player_id=row["player_id"],
            match_confidence=row["match_confidence"],
            match_method=row["match_method"],
            frame_start=row["frame_start"],
            frame_end=row["frame_end"],
            embedding=_deserialize_embedding(row["embedding"]),
        )

    def get_appearances_for_video(self, video_id: str) -> list[Appearance]:
        """Get all appearances for a video."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM appearances WHERE video_id = ? ORDER BY track_id",
            (video_id,),
        )
        rows = cursor.fetchall()

        return [
            Appearance(
                appearance_id=row["appearance_id"],
                video_id=row["video_id"],
                run_name=row["run_name"],
                track_id=row["track_id"],
                player_id=row["player_id"],
                match_confidence=row["match_confidence"],
                match_method=row["match_method"],
                frame_start=row["frame_start"],
                frame_end=row["frame_end"],
                embedding=_deserialize_embedding(row["embedding"]),
            )
            for row in rows
        ]

    def get_appearances_for_player(self, player_id: int) -> list[Appearance]:
        """Get all appearances for a player."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM appearances WHERE player_id = ? ORDER BY video_id, track_id",
            (player_id,),
        )
        rows = cursor.fetchall()

        return [
            Appearance(
                appearance_id=row["appearance_id"],
                video_id=row["video_id"],
                run_name=row["run_name"],
                track_id=row["track_id"],
                player_id=row["player_id"],
                match_confidence=row["match_confidence"],
                match_method=row["match_method"],
                frame_start=row["frame_start"],
                frame_end=row["frame_end"],
                embedding=_deserialize_embedding(row["embedding"]),
            )
            for row in rows
        ]

    def assign_appearance_to_player(
        self,
        video_id: str,
        track_id: int,
        player_id: int | None,
        confidence: float = 1.0,
        method: Literal["auto", "suggested", "manual"] = "manual",
    ) -> Appearance | None:
        """
        Assign an appearance to a player.

        Args:
            video_id: Video identifier.
            track_id: Track identifier.
            player_id: Player to assign to, or None to unassign.
            confidence: Match confidence.
            method: Assignment method.

        Returns:
            Updated Appearance or None if not found.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE appearances
            SET player_id = ?, match_confidence = ?, match_method = ?
            WHERE video_id = ? AND track_id = ?
            """,
            (player_id, confidence, method, video_id, track_id),
        )
        self.conn.commit()

        if cursor.rowcount == 0:
            return None

        return self.get_appearance(video_id, track_id)

    def delete_appearance(self, video_id: str, track_id: int) -> bool:
        """Delete an appearance by video_id and track_id."""
        cursor = self.conn.cursor()
        cursor.execute(
            "DELETE FROM appearances WHERE video_id = ? AND track_id = ?",
            (video_id, track_id),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def get_all_player_centroids(self) -> dict[int, np.ndarray]:
        """
        Get all player centroids for matching.

        Returns:
            Dict mapping player_id to embedding centroid array.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT player_id, embedding_centroid
            FROM players
            WHERE embedding_centroid IS NOT NULL
            """
        )
        rows = cursor.fetchall()

        return {
            row["player_id"]: np.frombuffer(row["embedding_centroid"], dtype=np.float32)
            for row in rows
        }

    # ── Team CRUD ──────────────────────────────────────────────────────────

    def create_team(self, name: str, short_name: str | None = None, logo_path: str | None = None) -> Team:
        """Create a new team."""
        cursor = self.conn.cursor()
        now = datetime.utcnow()
        cursor.execute(
            "INSERT INTO teams (name, short_name, logo_path, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (name, short_name, logo_path, now, now),
        )
        self.conn.commit()
        return Team(team_id=cursor.lastrowid, name=name, short_name=short_name, logo_path=logo_path, created_at=now, updated_at=now)

    def get_team(self, team_id: int) -> Team | None:
        """Get a team by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM teams WHERE team_id = ?", (team_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return Team(
            team_id=row["team_id"],
            name=row["name"],
            short_name=row["short_name"],
            logo_path=row["logo_path"] if "logo_path" in row.keys() else None,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def list_teams(self) -> list[TeamWithKits]:
        """List all teams with kits and player counts."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM teams ORDER BY team_id")
        team_rows = cursor.fetchall()

        results: list[TeamWithKits] = []
        for row in team_rows:
            tid = row["team_id"]
            kits = self.get_kits_for_team(tid)
            cursor.execute("SELECT COUNT(*) as cnt FROM players WHERE team_id = ?", (tid,))
            count_row = cursor.fetchone()
            player_count = count_row["cnt"] if count_row else 0
            results.append(
                TeamWithKits(
                    team_id=tid,
                    name=row["name"],
                    short_name=row["short_name"],
                    logo_path=row["logo_path"] if "logo_path" in row.keys() else None,
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    kits=kits,
                    player_count=player_count,
                )
            )
        return results

    def update_team(self, team_id: int, name: str | None = None, short_name: str | None = None) -> Team | None:
        """Update team metadata."""
        team = self.get_team(team_id)
        if team is None:
            return None
        updates: list[str] = []
        values: list[object] = []
        if name is not None:
            updates.append("name = ?")
            values.append(name)
        if short_name is not None:
            updates.append("short_name = ?")
            values.append(short_name)
        if not updates:
            return team
        updates.append("updated_at = ?")
        values.append(datetime.utcnow())
        values.append(team_id)
        cursor = self.conn.cursor()
        cursor.execute(f"UPDATE teams SET {', '.join(updates)} WHERE team_id = ?", values)
        self.conn.commit()
        return self.get_team(team_id)

    def delete_team(self, team_id: int) -> bool:
        """Delete a team, its kits, its run-team rows, and null out players.team_id."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM team_kits WHERE team_id = ?", (team_id,))
        cursor.execute("DELETE FROM run_teams WHERE team_id = ?", (team_id,))
        cursor.execute("UPDATE players SET team_id = NULL WHERE team_id = ?", (team_id,))
        cursor.execute("DELETE FROM teams WHERE team_id = ?", (team_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def set_team_logo(self, team_id: int, logo_path: str | None) -> Team | None:
        """Set or clear a team's logo_path."""
        team = self.get_team(team_id)
        if team is None:
            return None
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE teams SET logo_path = ?, updated_at = ? WHERE team_id = ?",
            (logo_path, datetime.utcnow(), team_id),
        )
        self.conn.commit()
        return self.get_team(team_id)

    # ── Kit CRUD ───────────────────────────────────────────────────────────

    def upsert_kit(
        self,
        team_id: int,
        kit_type: str,
        image_path: str | None = None,
        dominant_color_hsv: np.ndarray | list[float] | None = None,
        secondary_color_hsv: np.ndarray | list[float] | None = None,
        color_hex: str | None = None,
        secondary_color_hex: str | None = None,
    ) -> TeamKit:
        """Insert or replace a kit for a team."""
        dom_blob = _serialize_embedding(dominant_color_hsv) if dominant_color_hsv is not None else None
        sec_blob = _serialize_embedding(secondary_color_hsv) if secondary_color_hsv is not None else None
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO team_kits
            (team_id, kit_type, image_path, dominant_color_hsv, secondary_color_hsv, color_hex, secondary_color_hex)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (team_id, kit_type, image_path, dom_blob, sec_blob, color_hex, secondary_color_hex),
        )
        self.conn.commit()
        kit_id = cursor.lastrowid
        return TeamKit(
            kit_id=kit_id,
            team_id=team_id,
            kit_type=kit_type,
            image_path=image_path,
            dominant_color_hsv=list(dominant_color_hsv) if dominant_color_hsv is not None else None,
            secondary_color_hsv=list(secondary_color_hsv) if secondary_color_hsv is not None else None,
            color_hex=color_hex,
            secondary_color_hex=secondary_color_hex,
        )

    def get_kits_for_team(self, team_id: int) -> list[TeamKit]:
        """Get all kits for a team."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM team_kits WHERE team_id = ? ORDER BY kit_type", (team_id,))
        rows = cursor.fetchall()
        return [
            TeamKit(
                kit_id=row["kit_id"],
                team_id=row["team_id"],
                kit_type=row["kit_type"],
                image_path=row["image_path"],
                dominant_color_hsv=_deserialize_embedding(row["dominant_color_hsv"]),
                secondary_color_hsv=_deserialize_embedding(row["secondary_color_hsv"]),
                color_hex=row["color_hex"],
                secondary_color_hex=row["secondary_color_hex"],
            )
            for row in rows
        ]

    def get_kit(self, team_id: int, kit_type: str) -> TeamKit | None:
        """Get a specific kit."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM team_kits WHERE team_id = ? AND kit_type = ?",
            (team_id, kit_type),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return TeamKit(
            kit_id=row["kit_id"],
            team_id=row["team_id"],
            kit_type=row["kit_type"],
            image_path=row["image_path"],
            dominant_color_hsv=_deserialize_embedding(row["dominant_color_hsv"]),
            secondary_color_hsv=_deserialize_embedding(row["secondary_color_hsv"]),
            color_hex=row["color_hex"],
            secondary_color_hex=row["secondary_color_hex"],
        )

    def delete_kit(self, team_id: int, kit_type: str) -> bool:
        """Delete a kit."""
        cursor = self.conn.cursor()
        cursor.execute(
            "DELETE FROM team_kits WHERE team_id = ? AND kit_type = ?",
            (team_id, kit_type),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    # ── Run-team association ───────────────────────────────────────────────

    def set_run_teams(
        self,
        run_name: str,
        home_team_id: int,
        away_team_id: int,
        home_kit: str = "home",
        away_kit: str = "home",
    ) -> list[RunTeamAssociation]:
        """Set home+away team associations for a run."""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO run_teams (run_name, role, team_id, active_kit) VALUES (?, 'home', ?, ?)",
            (run_name, home_team_id, home_kit),
        )
        cursor.execute(
            "INSERT OR REPLACE INTO run_teams (run_name, role, team_id, active_kit) VALUES (?, 'away', ?, ?)",
            (run_name, away_team_id, away_kit),
        )
        self.conn.commit()
        return self.get_run_teams(run_name)

    def get_run_teams(self, run_name: str) -> list[RunTeamAssociation]:
        """Get team associations for a run."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM run_teams WHERE run_name = ? ORDER BY role", (run_name,))
        rows = cursor.fetchall()
        return [
            RunTeamAssociation(
                run_name=row["run_name"],
                role=row["role"],
                team_id=row["team_id"],
                active_kit=row["active_kit"],
                cluster_id=row["cluster_id"],
            )
            for row in rows
        ]

    def update_run_team_cluster(self, run_name: str, role: str, cluster_id: int) -> bool:
        """Update cluster_id for a run-team association."""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE run_teams SET cluster_id = ? WHERE run_name = ? AND role = ?",
            (cluster_id, run_name, role),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    # ── Player-team linking ────────────────────────────────────────────────

    def set_player_team(self, player_id: int, team_id: int | None) -> Player | None:
        """Set a player's team_id."""
        player = self.get_player(player_id)
        if player is None:
            return None
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE players SET team_id = ?, updated_at = ? WHERE player_id = ?",
            (team_id, datetime.utcnow(), player_id),
        )
        self.conn.commit()
        return self.get_player(player_id)

    def set_player_photo(self, player_id: int, photo_path: str | None) -> Player | None:
        """Set or clear a player's photo_path."""
        player = self.get_player(player_id)
        if player is None:
            return None
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE players SET photo_path = ?, updated_at = ? WHERE player_id = ?",
            (photo_path, datetime.utcnow(), player_id),
        )
        self.conn.commit()
        return self.get_player(player_id)

    # ── Match metadata ─────────────────────────────────────────────────────

    def _match_metadata_from_row(self, row: sqlite3.Row) -> MatchMetadata:
        """Convert SQLite row to MatchMetadata model."""
        return MatchMetadata(
            run_name=row["run_name"],
            video_id=row["video_id"],
            match_date=row["match_date"],
            competition=row["competition"],
            season=row["season"],
            venue=row["venue"],
            home_team_id=row["home_team_id"],
            away_team_id=row["away_team_id"],
            notes=row["notes"],
            extra=_deserialize_json(row["extra_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def upsert_match_metadata(
        self,
        run_name: str,
        video_id: str | None = None,
        match_date: str | None = None,
        competition: str | None = None,
        season: str | None = None,
        venue: str | None = None,
        home_team_id: int | None = None,
        away_team_id: int | None = None,
        notes: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> MatchMetadata:
        """Create or update match metadata for a run."""
        cursor = self.conn.cursor()
        now = datetime.utcnow()
        cursor.execute(
            """
            INSERT INTO match_metadata (
                run_name, video_id, match_date, competition, season, venue,
                home_team_id, away_team_id, notes, extra_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_name) DO UPDATE SET
                video_id = excluded.video_id,
                match_date = excluded.match_date,
                competition = excluded.competition,
                season = excluded.season,
                venue = excluded.venue,
                home_team_id = excluded.home_team_id,
                away_team_id = excluded.away_team_id,
                notes = excluded.notes,
                extra_json = excluded.extra_json,
                updated_at = excluded.updated_at
            """,
            (
                run_name,
                video_id,
                match_date,
                competition,
                season,
                venue,
                home_team_id,
                away_team_id,
                notes,
                _serialize_json(extra),
                now,
                now,
            ),
        )
        self.conn.commit()
        metadata = self.get_match_metadata(run_name)
        if metadata is None:
            raise RuntimeError("Failed to persist match metadata")
        return metadata

    def get_match_metadata(self, run_name: str) -> MatchMetadata | None:
        """Get match metadata for a run."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM match_metadata WHERE run_name = ?", (run_name,))
        row = cursor.fetchone()
        if row is None:
            return None
        return self._match_metadata_from_row(row)

    def list_match_metadata(self, limit: int | None = None, offset: int = 0) -> list[MatchMetadata]:
        """List match metadata rows ordered by run name."""
        cursor = self.conn.cursor()
        query = "SELECT * FROM match_metadata ORDER BY run_name"
        values: list[object] = []
        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            values.extend([limit, offset])
        cursor.execute(query, values)
        rows = cursor.fetchall()
        return [self._match_metadata_from_row(row) for row in rows]

    def delete_match_metadata(self, run_name: str) -> bool:
        """Delete match metadata for a run."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM match_metadata WHERE run_name = ?", (run_name,))
        self.conn.commit()
        return cursor.rowcount > 0

    # ── Tags ───────────────────────────────────────────────────────────────

    def _tag_from_row(self, row: sqlite3.Row) -> MatchTag:
        """Convert SQLite row to MatchTag model."""
        return MatchTag(
            tag_id=row["tag_id"],
            run_name=row["run_name"],
            label=row["label"],
            category=row["category"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            frame_idx=row["frame_idx"],
            track_id=row["track_id"],
            player_id=row["player_id"],
            team_id=row["team_id"],
            confidence=row["confidence"],
            source=row["source"],
            notes=row["notes"],
            metadata=_deserialize_json(row["metadata_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def create_tag(
        self,
        run_name: str,
        label: str,
        category: str = "general",
        start_time: float | None = None,
        end_time: float | None = None,
        frame_idx: int | None = None,
        track_id: int | None = None,
        player_id: int | None = None,
        team_id: int | None = None,
        confidence: float | None = None,
        source: Literal["manual", "auto", "imported"] = "manual",
        notes: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MatchTag:
        """Create a tag row."""
        if start_time is not None and end_time is not None and end_time < start_time:
            raise ValueError("end_time must be >= start_time")
        now = datetime.utcnow()
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO tags (
                run_name, label, category, start_time, end_time, frame_idx, track_id,
                player_id, team_id, confidence, source, notes, metadata_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_name,
                label,
                category,
                start_time,
                end_time,
                frame_idx,
                track_id,
                player_id,
                team_id,
                confidence,
                source,
                notes,
                _serialize_json(metadata),
                now,
                now,
            ),
        )
        self.conn.commit()
        tag = self.get_tag(cursor.lastrowid)
        if tag is None:
            raise RuntimeError("Failed to persist tag")
        return tag

    def get_tag(self, tag_id: int) -> MatchTag | None:
        """Get tag by id."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM tags WHERE tag_id = ?", (tag_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return self._tag_from_row(row)

    def list_tags(
        self,
        run_name: str | None = None,
        label: str | None = None,
        category: str | None = None,
        source: Literal["manual", "auto", "imported"] | None = None,
        player_id: int | None = None,
        team_id: int | None = None,
        min_time: float | None = None,
        max_time: float | None = None,
    ) -> list[MatchTag]:
        """List tags with optional filters."""
        query = "SELECT * FROM tags WHERE 1=1"
        values: list[object] = []
        if run_name is not None:
            query += " AND run_name = ?"
            values.append(run_name)
        if label is not None:
            query += " AND label = ?"
            values.append(label)
        if category is not None:
            query += " AND category = ?"
            values.append(category)
        if source is not None:
            query += " AND source = ?"
            values.append(source)
        if player_id is not None:
            query += " AND player_id = ?"
            values.append(player_id)
        if team_id is not None:
            query += " AND team_id = ?"
            values.append(team_id)
        if min_time is not None:
            query += " AND (end_time IS NULL OR end_time >= ?)"
            values.append(min_time)
        if max_time is not None:
            query += " AND (start_time IS NULL OR start_time <= ?)"
            values.append(max_time)
        query += " ORDER BY run_name, COALESCE(start_time, -1), tag_id"

        cursor = self.conn.cursor()
        cursor.execute(query, values)
        rows = cursor.fetchall()
        return [self._tag_from_row(row) for row in rows]

    def update_tag(
        self,
        tag_id: int,
        label: str | None | object = _UNSET,
        category: str | None | object = _UNSET,
        start_time: float | None | object = _UNSET,
        end_time: float | None | object = _UNSET,
        frame_idx: int | None | object = _UNSET,
        track_id: int | None | object = _UNSET,
        player_id: int | None | object = _UNSET,
        team_id: int | None | object = _UNSET,
        confidence: float | None | object = _UNSET,
        source: Literal["manual", "auto", "imported"] | None | object = _UNSET,
        notes: str | None | object = _UNSET,
        metadata: dict[str, Any] | None | object = _UNSET,
    ) -> MatchTag | None:
        """Update tag fields. Pass None explicitly to clear nullable fields."""
        tag = self.get_tag(tag_id)
        if tag is None:
            return None

        updates: list[str] = []
        values: list[object] = []

        def _append(name: str, value: object, serialize_json: bool = False) -> None:
            if value is _UNSET:
                return
            updates.append(f"{name} = ?")
            if serialize_json:
                values.append(_serialize_json(value if isinstance(value, dict) else None))
            else:
                values.append(value)

        _append("label", label)
        _append("category", category)
        _append("start_time", start_time)
        _append("end_time", end_time)
        _append("frame_idx", frame_idx)
        _append("track_id", track_id)
        _append("player_id", player_id)
        _append("team_id", team_id)
        _append("confidence", confidence)
        _append("source", source)
        _append("notes", notes)
        _append("metadata_json", metadata, serialize_json=True)

        if not updates:
            return tag

        effective_start = tag.start_time if start_time is _UNSET else start_time
        effective_end = tag.end_time if end_time is _UNSET else end_time
        if (
            effective_start is not None
            and effective_end is not None
            and float(effective_end) < float(effective_start)
        ):
            raise ValueError("end_time must be >= start_time")

        updates.append("updated_at = ?")
        values.append(datetime.utcnow())
        values.append(tag_id)
        cursor = self.conn.cursor()
        cursor.execute(f"UPDATE tags SET {', '.join(updates)} WHERE tag_id = ?", values)
        self.conn.commit()
        return self.get_tag(tag_id)

    def delete_tag(self, tag_id: int) -> bool:
        """Delete a tag by id."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM tags WHERE tag_id = ?", (tag_id,))
        self.conn.commit()
        return cursor.rowcount > 0
