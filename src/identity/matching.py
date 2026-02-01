"""Player identity matching algorithms using ReID embeddings."""

from dataclasses import dataclass
from typing import Literal

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: First vector (1D array).
        b: Second vector (1D array).

    Returns:
        Cosine similarity in range [-1, 1].
    """
    # Handle zero vectors
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_similarity_batch(
    query: np.ndarray,
    gallery: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarities between a query and gallery embeddings.

    Args:
        query: Query embedding (1D array of dim D).
        gallery: Gallery embeddings (2D array of shape NxD).

    Returns:
        Similarity scores (1D array of shape N).
    """
    if gallery.shape[0] == 0:
        return np.array([])

    # Normalize
    query_norm = query / (np.linalg.norm(query) + 1e-8)
    gallery_norms = gallery / (np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-8)

    # Dot product for cosine similarity
    return np.dot(gallery_norms, query_norm)


@dataclass
class MatchResult:
    """Result of matching an embedding to players."""

    player_id: int | None
    confidence: float
    method: Literal["auto", "suggested", "new_player"]
    all_scores: dict[int, float]  # player_id -> similarity score


def match_embedding_to_players(
    embedding: np.ndarray,
    player_centroids: dict[int, np.ndarray],
    auto_threshold: float = 0.85,
    suggest_threshold: float = 0.70,
    new_player_threshold: float = 0.60,
) -> MatchResult:
    """
    Match an embedding to existing players.

    Logic:
    - similarity >= auto_threshold: Auto-assign to best match
    - suggest_threshold <= similarity < auto_threshold: Suggest best match
    - similarity < new_player_threshold: Create new player
    - new_player_threshold <= similarity < suggest_threshold: Suggest, but uncertain

    Args:
        embedding: Query embedding (512D).
        player_centroids: Dict mapping player_id to centroid embedding.
        auto_threshold: Threshold for automatic assignment.
        suggest_threshold: Threshold for suggested assignment.
        new_player_threshold: Threshold below which to create new player.

    Returns:
        MatchResult with player_id, confidence, method, and all scores.
    """
    if not player_centroids:
        # No existing players, create new
        return MatchResult(
            player_id=None,
            confidence=0.0,
            method="new_player",
            all_scores={},
        )

    # Compute similarities to all players
    player_ids = list(player_centroids.keys())
    centroids = np.array([player_centroids[pid] for pid in player_ids])

    similarities = cosine_similarity_batch(embedding, centroids)

    # Build scores dict
    all_scores = {pid: float(sim) for pid, sim in zip(player_ids, similarities)}

    # Find best match
    best_idx = np.argmax(similarities)
    best_player_id = player_ids[best_idx]
    best_similarity = float(similarities[best_idx])

    # Determine method based on thresholds
    if best_similarity >= auto_threshold:
        method = "auto"
        player_id = best_player_id
    elif best_similarity >= suggest_threshold:
        method = "suggested"
        player_id = best_player_id
    elif best_similarity < new_player_threshold:
        method = "new_player"
        player_id = None
    else:
        # In the uncertain zone between new_player and suggest
        # Default to suggested but with lower confidence
        method = "suggested"
        player_id = best_player_id

    return MatchResult(
        player_id=player_id,
        confidence=best_similarity,
        method=method,
        all_scores=all_scores,
    )


def aggregate_embeddings(embeddings: list[np.ndarray]) -> np.ndarray:
    """
    Aggregate multiple embeddings into a single representative embedding.

    Uses mean pooling followed by L2 normalization.

    Args:
        embeddings: List of embedding vectors.

    Returns:
        Aggregated and normalized embedding.
    """
    if not embeddings:
        raise ValueError("Cannot aggregate empty list of embeddings")

    stacked = np.stack(embeddings, axis=0)
    mean_emb = np.mean(stacked, axis=0)

    # L2 normalize
    norm = np.linalg.norm(mean_emb)
    if norm > 0:
        mean_emb = mean_emb / norm

    return mean_emb


def compute_embedding_quality(
    embedding: np.ndarray,
    reference_embeddings: list[np.ndarray],
) -> float:
    """
    Compute quality score for an embedding based on consistency with others.

    Args:
        embedding: Query embedding.
        reference_embeddings: List of reference embeddings from same track.

    Returns:
        Quality score in [0, 1] based on mean similarity to references.
    """
    if not reference_embeddings:
        return 1.0

    similarities = [
        cosine_similarity(embedding, ref)
        for ref in reference_embeddings
    ]

    return float(np.mean(similarities))
