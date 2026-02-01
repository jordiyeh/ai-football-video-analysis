"""Player identity persistence module."""

from src.identity.database import PlayerDatabase
from src.identity.models import Player, Appearance
from src.identity.matching import (
    cosine_similarity,
    match_embedding_to_players,
    MatchResult,
)

__all__ = [
    "PlayerDatabase",
    "Player",
    "Appearance",
    "cosine_similarity",
    "match_embedding_to_players",
    "MatchResult",
]
