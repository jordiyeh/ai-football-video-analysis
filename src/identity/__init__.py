"""Player identity persistence module."""

from importlib import import_module
from typing import Any

__all__ = [
    "PlayerDatabase",
    "Player",
    "Appearance",
    "cosine_similarity",
    "cosine_similarity_batch",
    "match_embedding_to_players",
    "MatchResult",
    "aggregate_embeddings",
    "compute_embedding_quality",
    "ProfileEvidence",
    "fuse_identity_evidence",
    "ingest_profiles",
    "parse_profile_folder_name",
    "PROFILE_REGISTRY_SCHEMA_VERSION",
    "PROFILE_EMBEDDINGS_SCHEMA_VERSION",
    "ProfileSignature",
    "build_profile_signatures",
    "seed_players_from_signatures",
    "match_embedding_to_profile_links",
    "FaceSignature",
    "FaceEvidence",
    "JerseyOCREvidence",
    "build_profile_face_signatures",
    "match_track_face_evidence",
    "build_jersey_player_index",
    "extract_jersey_ocr_evidence",
    "apply_multimodal_evidence",
    "apply_substitution_locks",
    "generate_player_embeddings",
    "rebuild_embeddings_from_training_dir",
]

_SYMBOL_TO_MODULE = {
    "PlayerDatabase": "src.identity.database",
    "Player": "src.identity.models",
    "Appearance": "src.identity.models",
    "cosine_similarity": "src.identity.matching",
    "cosine_similarity_batch": "src.identity.matching",
    "match_embedding_to_players": "src.identity.matching",
    "MatchResult": "src.identity.matching",
    "aggregate_embeddings": "src.identity.matching",
    "compute_embedding_quality": "src.identity.matching",
    "ProfileEvidence": "src.identity.fusion",
    "fuse_identity_evidence": "src.identity.fusion",
    "ingest_profiles": "src.identity.profile_ingest",
    "parse_profile_folder_name": "src.identity.profile_ingest",
    "PROFILE_REGISTRY_SCHEMA_VERSION": "src.identity.profile_ingest",
    "PROFILE_EMBEDDINGS_SCHEMA_VERSION": "src.identity.profile_ingest",
    "ProfileSignature": "src.identity.profile_seed",
    "build_profile_signatures": "src.identity.profile_seed",
    "seed_players_from_signatures": "src.identity.profile_seed",
    "match_embedding_to_profile_links": "src.identity.profile_seed",
    "FaceSignature": "src.identity.multimodal",
    "FaceEvidence": "src.identity.multimodal",
    "JerseyOCREvidence": "src.identity.multimodal",
    "build_profile_face_signatures": "src.identity.multimodal",
    "match_track_face_evidence": "src.identity.multimodal",
    "build_jersey_player_index": "src.identity.multimodal",
    "extract_jersey_ocr_evidence": "src.identity.multimodal",
    "apply_multimodal_evidence": "src.identity.multimodal",
    "apply_substitution_locks": "src.identity.multimodal",
    "generate_player_embeddings": "src.identity.embedding_generator",
    "rebuild_embeddings_from_training_dir": "src.identity.embedding_generator",
}


def __getattr__(name: str) -> Any:
    """Lazily import identity modules with optional CV/ML dependencies."""
    module_name = _SYMBOL_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
