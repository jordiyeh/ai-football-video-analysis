"""Unit tests for player identity matching algorithms."""

import numpy as np
import pytest

from src.identity.matching import (
    cosine_similarity,
    cosine_similarity_batch,
    match_embedding_to_players,
    aggregate_embeddings,
    compute_embedding_quality,
    MatchResult,
)


class TestCosineSimilarity:
    """Tests for cosine similarity functions."""

    def test_identical_vectors(self):
        """Test similarity of identical vectors."""
        v = np.array([1, 2, 3], dtype=np.float32)

        sim = cosine_similarity(v, v)

        assert np.isclose(sim, 1.0)

    def test_orthogonal_vectors(self):
        """Test similarity of orthogonal vectors."""
        v1 = np.array([1, 0, 0], dtype=np.float32)
        v2 = np.array([0, 1, 0], dtype=np.float32)

        sim = cosine_similarity(v1, v2)

        assert np.isclose(sim, 0.0)

    def test_opposite_vectors(self):
        """Test similarity of opposite vectors."""
        v1 = np.array([1, 2, 3], dtype=np.float32)
        v2 = np.array([-1, -2, -3], dtype=np.float32)

        sim = cosine_similarity(v1, v2)

        assert np.isclose(sim, -1.0)

    def test_zero_vector(self):
        """Test handling of zero vectors."""
        v1 = np.array([1, 2, 3], dtype=np.float32)
        v2 = np.array([0, 0, 0], dtype=np.float32)

        sim = cosine_similarity(v1, v2)

        assert sim == 0.0

    def test_batch_similarity(self):
        """Test batch cosine similarity."""
        query = np.array([1, 0, 0], dtype=np.float32)
        gallery = np.array([
            [1, 0, 0],  # Identical
            [0, 1, 0],  # Orthogonal
            [-1, 0, 0],  # Opposite
        ], dtype=np.float32)

        similarities = cosine_similarity_batch(query, gallery)

        assert len(similarities) == 3
        assert np.isclose(similarities[0], 1.0)
        assert np.isclose(similarities[1], 0.0)
        assert np.isclose(similarities[2], -1.0)

    def test_batch_empty_gallery(self):
        """Test batch similarity with empty gallery."""
        query = np.array([1, 0, 0], dtype=np.float32)
        gallery = np.array([]).reshape(0, 3)

        similarities = cosine_similarity_batch(query, gallery)

        assert len(similarities) == 0


class TestMatchEmbeddingToPlayers:
    """Tests for player matching logic."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create some player centroids
        np.random.seed(42)
        self.player_centroids = {
            1: np.array([1, 0, 0], dtype=np.float32),
            2: np.array([0, 1, 0], dtype=np.float32),
            3: np.array([0, 0, 1], dtype=np.float32),
        }

    def test_auto_match(self):
        """Test automatic matching above threshold."""
        # Query very similar to player 1
        query = np.array([0.99, 0.01, 0], dtype=np.float32)
        query = query / np.linalg.norm(query)

        result = match_embedding_to_players(
            query,
            self.player_centroids,
            auto_threshold=0.90,
            suggest_threshold=0.70,
            new_player_threshold=0.50,
        )

        assert result.method == "auto"
        assert result.player_id == 1
        assert result.confidence > 0.90

    def test_suggested_match(self):
        """Test suggested matching between thresholds."""
        # Query somewhat similar to player 1
        query = np.array([0.8, 0.5, 0.1], dtype=np.float32)
        query = query / np.linalg.norm(query)

        result = match_embedding_to_players(
            query,
            self.player_centroids,
            auto_threshold=0.95,
            suggest_threshold=0.70,
            new_player_threshold=0.50,
        )

        assert result.method == "suggested"
        assert result.player_id == 1

    def test_new_player(self):
        """Test new player creation below threshold."""
        # Query not similar to any existing player
        query = np.array([0.4, 0.4, 0.4], dtype=np.float32)
        query = query / np.linalg.norm(query)

        result = match_embedding_to_players(
            query,
            self.player_centroids,
            auto_threshold=0.95,
            suggest_threshold=0.80,
            new_player_threshold=0.70,
        )

        assert result.method == "new_player"
        assert result.player_id is None

    def test_empty_centroids(self):
        """Test matching with no existing players."""
        query = np.array([1, 0, 0], dtype=np.float32)

        result = match_embedding_to_players(query, {})

        assert result.method == "new_player"
        assert result.player_id is None
        assert len(result.all_scores) == 0

    def test_all_scores_returned(self):
        """Test that all similarity scores are returned."""
        query = np.array([1, 0, 0], dtype=np.float32)

        result = match_embedding_to_players(query, self.player_centroids)

        assert len(result.all_scores) == 3
        assert 1 in result.all_scores
        assert 2 in result.all_scores
        assert 3 in result.all_scores


class TestAggregateEmbeddings:
    """Tests for embedding aggregation."""

    def test_single_embedding(self):
        """Test aggregating single embedding."""
        emb = np.array([1, 2, 3], dtype=np.float32)

        result = aggregate_embeddings([emb])

        # Should be normalized version of input
        expected = emb / np.linalg.norm(emb)
        assert np.allclose(result, expected)

    def test_multiple_embeddings(self):
        """Test aggregating multiple embeddings."""
        embeddings = [
            np.array([1, 0, 0], dtype=np.float32),
            np.array([0, 1, 0], dtype=np.float32),
        ]

        result = aggregate_embeddings(embeddings)

        # Mean should be [0.5, 0.5, 0], normalized
        expected = np.array([0.5, 0.5, 0], dtype=np.float32)
        expected = expected / np.linalg.norm(expected)
        assert np.allclose(result, expected)

    def test_normalized_output(self):
        """Test that output is always L2 normalized."""
        embeddings = [
            np.random.randn(512).astype(np.float32)
            for _ in range(10)
        ]

        result = aggregate_embeddings(embeddings)

        assert np.isclose(np.linalg.norm(result), 1.0, atol=1e-6)

    def test_empty_raises_error(self):
        """Test that empty list raises error."""
        with pytest.raises(ValueError):
            aggregate_embeddings([])


class TestComputeEmbeddingQuality:
    """Tests for embedding quality computation."""

    def test_consistent_embeddings(self):
        """Test quality of consistent embeddings."""
        # All similar embeddings
        emb = np.array([1, 0, 0], dtype=np.float32)
        refs = [
            np.array([0.99, 0.1, 0], dtype=np.float32),
            np.array([0.98, 0.15, 0], dtype=np.float32),
        ]
        # Normalize references
        refs = [r / np.linalg.norm(r) for r in refs]

        quality = compute_embedding_quality(emb, refs)

        assert quality > 0.9

    def test_inconsistent_embeddings(self):
        """Test quality of inconsistent embeddings."""
        emb = np.array([1, 0, 0], dtype=np.float32)
        refs = [
            np.array([0, 1, 0], dtype=np.float32),  # Orthogonal
            np.array([0, 0, 1], dtype=np.float32),  # Orthogonal
        ]

        quality = compute_embedding_quality(emb, refs)

        assert quality < 0.1

    def test_empty_references(self):
        """Test quality with no references."""
        emb = np.array([1, 0, 0], dtype=np.float32)

        quality = compute_embedding_quality(emb, [])

        assert quality == 1.0  # Default to perfect quality


class TestMatchResultDataclass:
    """Tests for MatchResult dataclass."""

    def test_match_result_creation(self):
        """Test creating a MatchResult."""
        result = MatchResult(
            player_id=1,
            confidence=0.95,
            method="auto",
            all_scores={1: 0.95, 2: 0.3},
        )

        assert result.player_id == 1
        assert result.confidence == 0.95
        assert result.method == "auto"
        assert len(result.all_scores) == 2

    def test_match_result_new_player(self):
        """Test MatchResult for new player."""
        result = MatchResult(
            player_id=None,
            confidence=0.4,
            method="new_player",
            all_scores={},
        )

        assert result.player_id is None
        assert result.method == "new_player"
