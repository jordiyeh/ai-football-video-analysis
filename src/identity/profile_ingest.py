"""Ingest external player profile folders with photos and optional pickle embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import pickle
import re
from typing import Any

import numpy as np


PROFILE_REGISTRY_SCHEMA_VERSION = "1.0"
PROFILE_EMBEDDINGS_SCHEMA_VERSION = "1.0"


@dataclass
class ParsedProfileName:
    """Parsed fields from a profile folder name."""

    profile_id: str
    display_name: str
    jersey_number: int | None


@dataclass
class ParsedEmbedding:
    """Single parsed and normalized embedding from a profile source."""

    vector: np.ndarray
    original_norm: float
    source_type: str
    source_image_path: str | None
    model_name: str


def parse_profile_folder_name(folder_name: str) -> ParsedProfileName:
    """
    Parse a profile folder name.

    Expected pattern examples:
    - "10_Nicholas_Oestringer"
    - "10-Nicholas-Oestringer"
    """
    profile_id = folder_name.strip()
    match = re.match(r"^(?P<number>\d+)[_-](?P<name>.+)$", profile_id)

    if match:
        jersey_number = int(match.group("number"))
        raw_name = match.group("name")
        display_name = raw_name.replace("_", " ").replace("-", " ").strip()
    else:
        jersey_number = None
        display_name = profile_id.replace("_", " ").replace("-", " ").strip()

    return ParsedProfileName(
        profile_id=profile_id,
        display_name=display_name,
        jersey_number=jersey_number,
    )


def _normalize_embedding(raw_embedding: Any) -> tuple[np.ndarray, float] | None:
    """Convert an embedding to float32 and L2-normalize it."""
    try:
        vector = np.asarray(raw_embedding, dtype=np.float32).reshape(-1)
    except Exception:
        return None

    if vector.size == 0:
        return None
    if not np.isfinite(vector).all():
        return None

    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return None

    return vector / norm, norm


def _parse_encoding_entry(
    entry: Any,
    default_model: str,
) -> ParsedEmbedding | None:
    """Parse one encoding entry from pickle payload."""
    source_image_path: str | None = None
    source_type = "encoding"
    model_name = default_model

    if isinstance(entry, dict):
        raw_embedding = entry.get("encoding")
        source_image_path = entry.get("image_path")
        model_name = str(entry.get("model", default_model))
    else:
        raw_embedding = entry

    normalized = _normalize_embedding(raw_embedding)
    if normalized is None:
        return None

    vector, original_norm = normalized
    return ParsedEmbedding(
        vector=vector,
        original_norm=original_norm,
        source_type=source_type,
        source_image_path=source_image_path,
        model_name=model_name,
    )


def _extract_embeddings_from_payload(payload: dict[str, Any]) -> list[ParsedEmbedding]:
    """Extract and normalize all embeddings from a profile pickle payload."""
    parsed_embeddings: list[ParsedEmbedding] = []
    default_model = str(payload.get("model", "unknown"))

    raw_encodings = payload.get("encodings", [])
    if isinstance(raw_encodings, list):
        for entry in raw_encodings:
            parsed = _parse_encoding_entry(entry, default_model=default_model)
            if parsed is not None:
                parsed_embeddings.append(parsed)

    if "averaged_encoding" in payload:
        averaged = _normalize_embedding(payload["averaged_encoding"])
        if averaged is not None:
            vector, original_norm = averaged
            parsed_embeddings.append(
                ParsedEmbedding(
                    vector=vector,
                    original_norm=original_norm,
                    source_type="averaged_encoding",
                    source_image_path=None,
                    model_name=default_model,
                )
            )

    return parsed_embeddings


def _collect_image_paths(
    profile_dir: Path,
    recursive_image_scan: bool,
    image_extensions: set[str],
) -> list[Path]:
    """Collect image files for a profile directory."""
    pattern = "**/*" if recursive_image_scan else "*"
    paths: list[Path] = []
    for candidate in sorted(profile_dir.glob(pattern)):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in image_extensions:
            continue
        paths.append(candidate)
    return paths


def ingest_profiles(
    profile_root: str | Path,
    recursive_image_scan: bool = False,
    image_extensions: list[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Ingest profile folders and return registry + embedding records.

    Returns:
        (profile_registry, embedding_rows)
    """
    root = Path(profile_root).expanduser()
    extension_set = {ext.lower() for ext in (image_extensions or [".jpg", ".jpeg", ".png", ".webp"])}
    generated_at = datetime.now(timezone.utc).isoformat()

    profiles: list[dict[str, Any]] = []
    embedding_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    if not root.exists() or not root.is_dir():
        registry = {
            "schema_version": PROFILE_REGISTRY_SCHEMA_VERSION,
            "profile_embeddings_schema_version": PROFILE_EMBEDDINGS_SCHEMA_VERSION,
            "profile_root": str(root),
            "generated_at": generated_at,
            "summary": {
                "profiles_found": 0,
                "profiles_with_embeddings": 0,
                "embeddings_total": 0,
                "errors": 0,
            },
            "profiles": [],
            "errors": [],
        }
        return registry, embedding_rows

    for profile_dir in sorted(root.iterdir()):
        if not profile_dir.is_dir():
            continue

        parsed_name = parse_profile_folder_name(profile_dir.name)
        profile_images = _collect_image_paths(
            profile_dir,
            recursive_image_scan=recursive_image_scan,
            image_extensions=extension_set,
        )
        pkl_files = sorted(profile_dir.glob("*.pkl"))

        profile_embedding_count = 0
        profile_models: set[str] = set()
        profile_warnings: list[str] = []

        for pkl_path in pkl_files:
            try:
                with open(pkl_path, "rb") as f:
                    payload = pickle.load(f)
            except Exception as exc:
                errors.append(
                    {
                        "profile_id": parsed_name.profile_id,
                        "source_file": str(pkl_path),
                        "error": f"pickle_load_failed: {exc}",
                    }
                )
                profile_warnings.append(f"Failed to load {pkl_path.name}")
                continue

            if not isinstance(payload, dict):
                errors.append(
                    {
                        "profile_id": parsed_name.profile_id,
                        "source_file": str(pkl_path),
                        "error": "unexpected_pickle_payload_type",
                    }
                )
                profile_warnings.append(f"Invalid payload type in {pkl_path.name}")
                continue

            parsed_embeddings = _extract_embeddings_from_payload(payload)
            if not parsed_embeddings:
                profile_warnings.append(f"No valid embeddings in {pkl_path.name}")
                continue

            for idx, parsed in enumerate(parsed_embeddings):
                embedding_rows.append(
                    {
                        "schema_version": PROFILE_EMBEDDINGS_SCHEMA_VERSION,
                        "profile_id": parsed_name.profile_id,
                        "player_name": parsed_name.display_name,
                        "jersey_number": parsed_name.jersey_number,
                        "modality": "face",
                        "embedding_model": parsed.model_name,
                        "embedding_source": parsed.source_type,
                        "embedding_index": idx,
                        "embedding_dim": int(parsed.vector.size),
                        "embedding_norm": parsed.original_norm,
                        "embedding": parsed.vector.tolist(),
                        "source_file": str(pkl_path),
                        "source_image_path": parsed.source_image_path,
                    }
                )
                profile_embedding_count += 1
                profile_models.add(parsed.model_name)

        profiles.append(
            {
                "profile_id": parsed_name.profile_id,
                "display_name": parsed_name.display_name,
                "jersey_number": parsed_name.jersey_number,
                "profile_dir": str(profile_dir),
                "image_count": len(profile_images),
                "image_paths": [str(path) for path in profile_images],
                "pkl_files": [str(path) for path in pkl_files],
                "embedding_count": profile_embedding_count,
                "embedding_models": sorted(profile_models),
                "warnings": profile_warnings,
            }
        )

    profiles_with_embeddings = sum(1 for p in profiles if p["embedding_count"] > 0)
    registry = {
        "schema_version": PROFILE_REGISTRY_SCHEMA_VERSION,
        "profile_embeddings_schema_version": PROFILE_EMBEDDINGS_SCHEMA_VERSION,
        "profile_root": str(root),
        "generated_at": generated_at,
        "summary": {
            "profiles_found": len(profiles),
            "profiles_with_embeddings": profiles_with_embeddings,
            "embeddings_total": len(embedding_rows),
            "errors": len(errors),
        },
        "profiles": profiles,
        "errors": errors,
    }
    return registry, embedding_rows

