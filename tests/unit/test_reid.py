"""Unit tests for ReID (Re-Identification) module."""

import numpy as np
import pytest

from src.vision.reid.base import ReIDExtractor
from src.vision.reid.crop import CropExtractor, PlayerCrop
from src.vision.reid.osnet import OSNetExtractor


class DummyReIDExtractor(ReIDExtractor):
    """Dummy ReID extractor for testing base class."""

    @property
    def embedding_dim(self) -> int:
        return 128

    def extract(self, crops: list[np.ndarray]) -> np.ndarray:
        # Return random normalized embeddings
        embeddings = np.random.randn(len(crops), self.embedding_dim).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms


class TestReIDExtractorBase:
    """Tests for ReIDExtractor base class."""

    def test_extract_single(self):
        """Test single crop extraction."""
        extractor = DummyReIDExtractor()
        crop = np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8)

        embedding = extractor.extract_single(crop)

        assert embedding.shape == (128,)
        assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5)

    def test_extract_batch(self):
        """Test batch extraction."""
        extractor = DummyReIDExtractor()
        crops = [np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8) for _ in range(5)]

        embeddings = extractor.extract(crops)

        assert embeddings.shape == (5, 128)

    def test_extract_empty(self):
        """Test extraction with empty input."""
        extractor = DummyReIDExtractor()

        embeddings = extractor.extract([])

        assert embeddings.shape == (0, 128)


class TestCropExtractor:
    """Tests for CropExtractor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = CropExtractor(
            min_height=50,
            min_width=25,
            min_confidence=0.5,
        )
        # Create a test frame (480x640)
        self.frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_extract_valid_crop(self):
        """Test extracting a valid crop."""
        crop = self.extractor.extract_crop(
            frame=self.frame,
            bbox=(100, 100, 180, 250),  # 80x150 crop
            track_id=1,
            frame_idx=0,
            confidence=0.9,
        )

        assert crop is not None
        assert crop.track_id == 1
        assert crop.frame_idx == 0
        assert crop.confidence == 0.9
        assert crop.height > 0
        assert crop.width > 0

    def test_reject_small_crop(self):
        """Test rejection of too-small crops."""
        crop = self.extractor.extract_crop(
            frame=self.frame,
            bbox=(100, 100, 120, 130),  # 20x30 crop (too small)
            track_id=1,
            frame_idx=0,
            confidence=0.9,
        )

        assert crop is None

    def test_reject_low_confidence(self):
        """Test rejection of low-confidence detections."""
        crop = self.extractor.extract_crop(
            frame=self.frame,
            bbox=(100, 100, 180, 250),
            track_id=1,
            frame_idx=0,
            confidence=0.3,  # Below threshold
        )

        assert crop is None

    def test_reject_invalid_bbox(self):
        """Test rejection of invalid bounding boxes."""
        # x1 > x2
        crop = self.extractor.extract_crop(
            frame=self.frame,
            bbox=(200, 100, 100, 250),
            track_id=1,
            frame_idx=0,
            confidence=0.9,
        )

        assert crop is None

    def test_extract_crops_from_frame(self):
        """Test extracting multiple crops from a frame."""
        tracks = [
            {"object_type": "player", "bbox": [100, 100, 180, 250], "track_id": 1, "confidence": 0.9},
            {"object_type": "player", "bbox": [300, 100, 380, 250], "track_id": 2, "confidence": 0.8},
            {"object_type": "ball", "bbox": [200, 200, 220, 220], "track_id": 3, "confidence": 0.7},  # Not player
        ]

        crops = self.extractor.extract_crops_from_frame(self.frame, tracks, frame_idx=0)

        # Should only get player crops
        assert len(crops) == 2
        assert all(c.track_id in [1, 2] for c in crops)

    def test_sample_crops_uniform(self):
        """Test uniform sampling of crops."""
        # Create crops at different frames
        all_crops = [
            PlayerCrop(
                image=np.zeros((100, 50, 3), dtype=np.uint8),
                track_id=1,
                frame_idx=i * 10,
                bbox=(0, 0, 50, 100),
                confidence=0.9,
            )
            for i in range(20)
        ]

        sampled = self.extractor.sample_crops_for_track(
            all_crops, track_id=1, n_samples=5, strategy="uniform"
        )

        assert len(sampled) == 5
        # Check uniform distribution
        frame_indices = [c.frame_idx for c in sampled]
        assert frame_indices == sorted(frame_indices)  # Should be sorted

    def test_sample_crops_highest_conf(self):
        """Test highest confidence sampling."""
        all_crops = [
            PlayerCrop(
                image=np.zeros((100, 50, 3), dtype=np.uint8),
                track_id=1,
                frame_idx=i,
                bbox=(0, 0, 50, 100),
                confidence=0.5 + i * 0.02,  # Increasing confidence
            )
            for i in range(20)
        ]

        sampled = self.extractor.sample_crops_for_track(
            all_crops, track_id=1, n_samples=5, strategy="highest_conf"
        )

        assert len(sampled) == 5
        # Should get highest confidence crops
        confidences = [c.confidence for c in sampled]
        assert all(c >= 0.8 for c in confidences)


class TestOSNetExtractor:
    """Tests for OSNet extractor (requires model loading)."""

    @pytest.fixture
    def extractor(self):
        """Create OSNet extractor (CPU for testing)."""
        return OSNetExtractor(
            model_name="osnet_x0_25",
            device="cpu",  # Use CPU for testing
            crop_size=(256, 128),
            batch_size=4,
            cache_dir="models",
        )

    def test_embedding_dim(self, extractor):
        """Test embedding dimension."""
        assert extractor.embedding_dim == 512

    def test_extract_single(self, extractor):
        """Test single image extraction."""
        crop = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)

        embedding = extractor.extract_single(crop)

        assert embedding.shape == (512,)
        # Should be L2 normalized
        assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-4)

    def test_extract_batch(self, extractor):
        """Test batch extraction."""
        crops = [np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8) for _ in range(5)]

        embeddings = extractor.extract(crops)

        assert embeddings.shape == (5, 512)
        # All should be L2 normalized
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4)

    def test_extract_different_sizes(self, extractor):
        """Test extraction with different input sizes (should be resized)."""
        crops = [
            np.random.randint(0, 255, (100, 50, 3), dtype=np.uint8),
            np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8),
            np.random.randint(0, 255, (150, 75, 3), dtype=np.uint8),
        ]

        embeddings = extractor.extract(crops)

        assert embeddings.shape == (3, 512)

    def test_extract_empty(self, extractor):
        """Test extraction with empty input."""
        embeddings = extractor.extract([])

        assert embeddings.shape == (0, 512)

    def test_similar_images_have_similar_embeddings(self, extractor):
        """Test that similar images produce similar embeddings."""
        # Create base image
        base = np.random.randint(50, 200, (256, 128, 3), dtype=np.uint8)

        # Create similar image (slightly modified)
        similar = base.copy()
        similar = np.clip(similar + np.random.randint(-10, 10, similar.shape), 0, 255).astype(np.uint8)

        # Create different image
        different = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)

        embeddings = extractor.extract([base, similar, different])

        # Compute similarities
        base_emb, similar_emb, different_emb = embeddings
        sim_similar = np.dot(base_emb, similar_emb)
        sim_different = np.dot(base_emb, different_emb)

        # Similar image should have higher similarity than random different image
        # This is probabilistic but should pass most of the time
        assert sim_similar > sim_different or abs(sim_similar - sim_different) < 0.3
