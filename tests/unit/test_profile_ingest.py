"""Unit tests for profile ingestion from external player folders."""

import pickle
from pathlib import Path

import numpy as np

from src.identity.profile_ingest import ingest_profiles, parse_profile_folder_name


class TestParseProfileFolderName:
    """Tests for parsing profile folder names."""

    def test_parse_with_jersey_prefix(self):
        """Folder with jersey prefix should extract number and name."""
        parsed = parse_profile_folder_name("10_Nicholas_Oestringer")

        assert parsed.profile_id == "10_Nicholas_Oestringer"
        assert parsed.jersey_number == 10
        assert parsed.display_name == "Nicholas Oestringer"

    def test_parse_without_jersey_prefix(self):
        """Folder without jersey prefix should keep name and no number."""
        parsed = parse_profile_folder_name("Nicholas_Oestringer")

        assert parsed.profile_id == "Nicholas_Oestringer"
        assert parsed.jersey_number is None
        assert parsed.display_name == "Nicholas Oestringer"


class TestIngestProfiles:
    """Tests for profile ingestion end-to-end."""

    def test_ingest_profiles_with_pkl_and_images(self, tmp_path: Path):
        """Ingestion should parse images and embeddings from .pkl payloads."""
        profile_dir = tmp_path / "10_Nicholas_Oestringer"
        profile_dir.mkdir(parents=True)
        (profile_dir / "headshot.jpg").write_bytes(b"fake-jpg-bytes")
        (profile_dir / "action.png").write_bytes(b"fake-png-bytes")

        payload = {
            "player_id": "10_Nicholas_Oestringer",
            "model": "Facenet512",
            "encodings": [
                {
                    "encoding": np.array([1.0, 2.0, 3.0], dtype=np.float64),
                    "image_path": "headshot.jpg",
                    "image_name": "headshot.jpg",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "facial_area": {},
                    "model": "Facenet512",
                }
            ],
            "averaged_encoding": np.array([2.0, 1.0, 0.5], dtype=np.float64),
        }
        with open(profile_dir / "10_Nicholas_Oestringer.pkl", "wb") as f:
            pickle.dump(payload, f)

        registry, rows = ingest_profiles(tmp_path)

        assert registry["summary"]["profiles_found"] == 1
        assert registry["summary"]["profiles_with_embeddings"] == 1
        assert registry["summary"]["embeddings_total"] == 2
        assert registry["summary"]["errors"] == 0

        assert len(rows) == 2
        for row in rows:
            assert row["profile_id"] == "10_Nicholas_Oestringer"
            assert row["player_name"] == "Nicholas Oestringer"
            assert row["jersey_number"] == 10
            assert row["embedding_model"] == "Facenet512"
            assert row["embedding_dim"] == 3
            assert row["modality"] == "face"

            embedding = np.asarray(row["embedding"], dtype=np.float32)
            assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-6)

    def test_ingest_profiles_handles_missing_root(self, tmp_path: Path):
        """Missing profile root should return empty artifacts without crashing."""
        missing_root = tmp_path / "does_not_exist"
        registry, rows = ingest_profiles(missing_root)

        assert registry["summary"]["profiles_found"] == 0
        assert registry["summary"]["embeddings_total"] == 0
        assert rows == []

    def test_ingest_profiles_captures_pickle_errors(self, tmp_path: Path):
        """Invalid pickle files should produce warnings/errors and continue."""
        profile_dir = tmp_path / "7_Test_Player"
        profile_dir.mkdir(parents=True)
        (profile_dir / "7_Test_Player.pkl").write_bytes(b"not-a-pickle")

        registry, rows = ingest_profiles(tmp_path)

        assert registry["summary"]["profiles_found"] == 1
        assert registry["summary"]["profiles_with_embeddings"] == 0
        assert registry["summary"]["embeddings_total"] == 0
        assert registry["summary"]["errors"] == 1
        assert rows == []

