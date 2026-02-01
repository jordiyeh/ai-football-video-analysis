"""SQLite database for player identity persistence."""

import sqlite3
import struct
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np

from src.identity.models import Appearance, Player, PlayerWithAppearances


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


class PlayerDatabase:
    """
    SQLite database for persistent player identities.

    Manages players and their appearances across videos.
    """

    SCHEMA_VERSION = "1.0"

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

        # Create schema_version table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version TEXT PRIMARY KEY
            )
        """)
        cursor.execute(
            "INSERT OR IGNORE INTO schema_version (version) VALUES (?)",
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
                embedding_centroid=_deserialize_embedding(row["embedding_centroid"]),
                embedding_count=row["embedding_count"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

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
        player_id: int,
        confidence: float = 1.0,
        method: Literal["auto", "suggested", "manual"] = "manual",
    ) -> Appearance | None:
        """
        Assign an appearance to a player.

        Args:
            video_id: Video identifier.
            track_id: Track identifier.
            player_id: Player to assign to.
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
