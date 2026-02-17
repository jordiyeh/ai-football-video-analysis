"""Unit tests for embedding generator."""

import pickle
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.identity.embedding_generator import (
    add_embeddings_from_images,
    generate_player_embeddings,
    load_embeddings_pkl,
    rebuild_embeddings_from_training_dir,
    write_embeddings_pkl,
)


def _make_test_image(path: Path, w: int = 100, h: int = 100) -> Path:
    """Create a synthetic BGR image on disk."""
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path


def _fake_face_region(image, detector):
    """Return the full image as face region (bypasses Haar cascade)."""
    return image


def _fake_facenet_embedding(region):
    """Return a deterministic 512-d embedding for any region."""
    vec = np.random.randn(512).astype(np.float32)
    return vec / np.linalg.norm(vec)


class TestGeneratePlayerEmbeddings:
    """Tests for generate_player_embeddings."""

    @patch("src.identity.embedding_generator._compute_face_descriptor_facenet", side_effect=_fake_facenet_embedding)
    @patch("src.identity.embedding_generator._extract_face_region", side_effect=_fake_face_region)
    @patch("src.identity.embedding_generator._load_face_detector", return_value=None)
    def test_basic(self, _mock_det, _mock_region, _mock_facenet, tmp_path):
        """Process synthetic images and verify pkl schema."""
        imgs = [_make_test_image(tmp_path / f"img_{i}.jpg") for i in range(3)]

        result = generate_player_embeddings(42, imgs, player_name="Test Player")

        assert result["player_id"] == "42"
        assert result["player_name"] == "Test Player"
        assert result["model"] == "Facenet512"
        assert result["num_encodings"] == 3
        assert len(result["encodings"]) == 3
        assert result["averaged_encoding"] is not None
        assert result["averaged_encoding"].shape == (512,)
        # Verify L2 normalized
        norm = np.linalg.norm(result["averaged_encoding"])
        assert abs(norm - 1.0) < 1e-5
        assert result["stats"]["total_images_processed"] == 3
        assert result["stats"]["successful_extractions"] == 3
        assert result["stats"]["failed_extractions"] == 0

        # Verify each encoding entry
        for enc in result["encodings"]:
            assert enc["encoding"].shape == (512,)
            assert enc["model"] == "Facenet512"
            assert "image_path" in enc
            assert "image_name" in enc

    @patch("src.identity.embedding_generator._compute_face_descriptor_facenet", return_value=None)
    @patch("src.identity.embedding_generator._extract_face_region", return_value=None)
    @patch("src.identity.embedding_generator._load_face_detector", return_value=None)
    def test_all_fail(self, _mock_det, _mock_region, _mock_facenet, tmp_path):
        """Images with no detectable face produce graceful empty result."""
        imgs = [_make_test_image(tmp_path / f"img_{i}.jpg") for i in range(2)]

        result = generate_player_embeddings(1, imgs)

        assert result["num_encodings"] == 0
        assert result["averaged_encoding"] is None
        assert result["stats"]["total_images_processed"] == 2
        assert result["stats"]["successful_extractions"] == 0
        assert result["stats"]["failed_extractions"] == 2

    @patch("src.identity.embedding_generator._compute_face_descriptor_facenet", side_effect=_fake_facenet_embedding)
    @patch("src.identity.embedding_generator._extract_face_region", side_effect=_fake_face_region)
    @patch("src.identity.embedding_generator._load_face_detector", return_value=None)
    def test_nonexistent_image(self, _mock_det, _mock_region, _mock_facenet, tmp_path):
        """Non-existent image path is counted as failure."""
        good = _make_test_image(tmp_path / "good.jpg")
        bad = tmp_path / "nonexistent.jpg"

        result = generate_player_embeddings(1, [good, bad])

        assert result["num_encodings"] == 1
        assert result["stats"]["failed_extractions"] == 1


class TestWriteAndLoadPkl:
    """Tests for pkl serialization roundtrip."""

    @patch("src.identity.embedding_generator._compute_face_descriptor_facenet", side_effect=_fake_facenet_embedding)
    @patch("src.identity.embedding_generator._extract_face_region", side_effect=_fake_face_region)
    @patch("src.identity.embedding_generator._load_face_detector", return_value=None)
    def test_roundtrip(self, _mock_det, _mock_region, _mock_facenet, tmp_path):
        """Serialize and deserialize preserves data integrity."""
        imgs = [_make_test_image(tmp_path / f"img_{i}.jpg") for i in range(2)]
        payload = generate_player_embeddings(99, imgs)

        pkl_path = tmp_path / "test.pkl"
        write_embeddings_pkl(payload, pkl_path)

        assert pkl_path.exists()

        with open(pkl_path, "rb") as f:
            loaded = pickle.load(f)

        assert loaded["player_id"] == "99"
        assert loaded["num_encodings"] == 2
        assert len(loaded["encodings"]) == 2
        np.testing.assert_allclose(
            loaded["averaged_encoding"], payload["averaged_encoding"], atol=1e-6,
        )


class TestRebuildFromTrainingDir:
    """Tests for rebuild_embeddings_from_training_dir."""

    @patch("src.identity.embedding_generator._compute_face_descriptor_facenet", side_effect=_fake_facenet_embedding)
    @patch("src.identity.embedding_generator._extract_face_region", side_effect=_fake_face_region)
    @patch("src.identity.embedding_generator._load_face_detector", return_value=None)
    def test_rebuild(self, _mock_det, _mock_region, _mock_facenet, tmp_path):
        """End-to-end: create training dir, rebuild, verify pkl + payload."""
        player_dir = tmp_path / "player_photos" / "7"
        training_dir = player_dir / "training"
        training_dir.mkdir(parents=True)

        for i in range(3):
            _make_test_image(training_dir / f"img_{i:03d}.jpg")

        payload = rebuild_embeddings_from_training_dir(7, training_dir, player_name="Jane")

        assert payload["num_encodings"] == 3
        assert payload["player_name"] == "Jane"
        assert payload["averaged_encoding"] is not None

        pkl_path = player_dir / "embeddings.pkl"
        assert pkl_path.exists()

        with open(pkl_path, "rb") as f:
            loaded = pickle.load(f)
        assert loaded["num_encodings"] == 3

    def test_rebuild_empty_dir(self, tmp_path):
        """Rebuild with empty training dir produces empty payload."""
        training_dir = tmp_path / "player_photos" / "99" / "training"
        training_dir.mkdir(parents=True)

        payload = rebuild_embeddings_from_training_dir(99, training_dir)

        assert payload["num_encodings"] == 0
        assert payload["averaged_encoding"] is None

    def test_rebuild_nonexistent_dir(self, tmp_path):
        """Rebuild with missing dir produces empty payload."""
        training_dir = tmp_path / "does_not_exist" / "training"

        payload = rebuild_embeddings_from_training_dir(1, training_dir)

        assert payload["num_encodings"] == 0


class TestAddEmbeddingsFromImages:
    """Tests for incremental add_embeddings_from_images."""

    @patch("src.identity.embedding_generator._compute_face_descriptor_facenet", side_effect=_fake_facenet_embedding)
    @patch("src.identity.embedding_generator._extract_face_region", side_effect=_fake_face_region)
    @patch("src.identity.embedding_generator._load_face_detector", return_value=None)
    def test_add_to_empty(self, _mock_det, _mock_region, _mock_facenet, tmp_path):
        """Adding images when no pkl exists creates one."""
        imgs = [_make_test_image(tmp_path / f"img_{i}.jpg") for i in range(2)]
        pkl_path = tmp_path / "embeddings.pkl"

        payload = add_embeddings_from_images(1, imgs, pkl_path, player_name="New")

        assert payload["num_encodings"] == 2
        assert payload["averaged_encoding"] is not None
        assert pkl_path.exists()
        assert payload["stats"]["existing_encodings_kept"] == 0

    @patch("src.identity.embedding_generator._compute_face_descriptor_facenet", side_effect=_fake_facenet_embedding)
    @patch("src.identity.embedding_generator._extract_face_region", side_effect=_fake_face_region)
    @patch("src.identity.embedding_generator._load_face_detector", return_value=None)
    def test_add_incremental(self, _mock_det, _mock_region, _mock_facenet, tmp_path):
        """Adding images merges with existing pkl embeddings."""
        # Create initial pkl with 2 embeddings
        initial_imgs = [_make_test_image(tmp_path / f"init_{i}.jpg") for i in range(2)]
        pkl_path = tmp_path / "embeddings.pkl"
        add_embeddings_from_images(1, initial_imgs, pkl_path, player_name="Player")

        # Add 3 more images
        new_imgs = [_make_test_image(tmp_path / f"new_{i}.jpg") for i in range(3)]
        payload = add_embeddings_from_images(1, new_imgs, pkl_path, player_name="Player")

        assert payload["num_encodings"] == 5
        assert payload["stats"]["existing_encodings_kept"] == 2
        assert payload["stats"]["successful_extractions"] == 3

    def test_load_missing_pkl(self, tmp_path):
        """load_embeddings_pkl returns None for missing file."""
        assert load_embeddings_pkl(tmp_path / "nope.pkl") is None
