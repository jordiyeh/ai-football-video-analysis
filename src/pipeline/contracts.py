"""Artifact contract validation and schema-version helpers."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


RUN_MANIFEST_SCHEMA_VERSION = "1.1"
SUMMARY_SCHEMA_VERSION = "1.0"
UI_INDEX_SCHEMA_VERSION = "1.0"
DETECTIONS_SCHEMA_VERSION = "1.0"
TRACKS_SCHEMA_VERSION = "1.0"
EVENTS_SCHEMA_VERSION = "1.0"
SCORE_TIMELINE_SCHEMA_VERSION = "1.0"


ARTIFACT_SCHEMA_VERSIONS: dict[str, str] = {
    "run_manifest": RUN_MANIFEST_SCHEMA_VERSION,
    "detections": DETECTIONS_SCHEMA_VERSION,
    "tracks": TRACKS_SCHEMA_VERSION,
    "events": EVENTS_SCHEMA_VERSION,
    "score_timeline": SCORE_TIMELINE_SCHEMA_VERSION,
    "summary": SUMMARY_SCHEMA_VERSION,
    "ui_index": UI_INDEX_SCHEMA_VERSION,
}


class ArtifactContractError(RuntimeError):
    """Raised when required run artifacts violate the run contract."""


@dataclass(frozen=True)
class RequiredArtifact:
    """Artifact requirement for contract validation."""

    key: str
    candidates: tuple[str, ...]
    kind: str = "path"  # path | json | jsonl | tabular
    expected_schema_version: str | None = None


def collect_runtime_environment(device: str | None = None) -> dict[str, Any]:
    """Collect lightweight runtime environment metadata for run manifests."""
    environment = {
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    if device is not None:
        environment["device"] = device
    return environment


def _run_git(args: list[str], cwd: Path | None = None) -> str | None:
    """Run a git command and return stripped stdout, or None on failure."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    output = proc.stdout.strip()
    return output or None


def collect_git_metadata(cwd: Path | None = None) -> dict[str, Any]:
    """Collect current git metadata for run manifests."""
    commit = _run_git(["rev-parse", "HEAD"], cwd=cwd)
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)
    status = _run_git(["status", "--porcelain"], cwd=cwd)

    return {
        "commit": commit,
        "branch": branch,
        "is_dirty": bool(status),
    }


def resolve_artifact_path(output_dir: Path, candidates: Iterable[str]) -> Path | None:
    """Resolve the first existing artifact path among candidate names."""
    for candidate in candidates:
        path = output_dir / candidate
        if path.exists():
            return path
    return None


def has_json_schema_version(path: Path, expected_schema_version: str) -> bool:
    """Return True when JSON artifact has expected top-level schema_version."""
    try:
        with open(path) as f:
            payload = json.load(f)
    except Exception:
        return False

    if not isinstance(payload, dict):
        return False
    return str(payload.get("schema_version")) == expected_schema_version


def has_jsonl_schema_version(path: Path, expected_schema_version: str) -> bool:
    """Return True when JSONL rows carry expected schema_version.

    Empty JSONL artifacts are treated as schema-compatible.
    """
    saw_row = False
    try:
        with open(path) as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    return False
                saw_row = True
                if str(row.get("schema_version")) != expected_schema_version:
                    return False
    except Exception:
        return False

    if not saw_row:
        return True
    return True


def has_tabular_schema_version(path: Path, expected_schema_version: str) -> bool:
    """Return True when tabular artifact carries expected schema_version."""
    if path.suffix == ".parquet":
        try:
            frame = pd.read_parquet(path, columns=["schema_version"])
        except Exception:
            return False
    elif path.suffix == ".csv":
        try:
            frame = pd.read_csv(path, usecols=["schema_version"])
        except Exception:
            return False
    elif path.suffix == ".jsonl":
        return has_jsonl_schema_version(path, expected_schema_version)
    else:
        return False

    if "schema_version" not in frame.columns:
        return False
    if frame.empty:
        return True

    versions = {str(value) for value in frame["schema_version"].dropna().unique()}
    if not versions:
        return False
    return versions == {expected_schema_version}


def expected_required_artifacts(
    stage_names: Iterable[str],
    *,
    save_detections: bool,
    save_tracks: bool,
    save_events: bool,
    save_overlay_video: bool,
) -> list[RequiredArtifact]:
    """Build required artifact list from executed stages and export settings."""
    stage_set = set(stage_names)
    required: list[RequiredArtifact] = [
        RequiredArtifact(
            key="run_manifest",
            candidates=("run_manifest.json",),
            kind="json",
            expected_schema_version=RUN_MANIFEST_SCHEMA_VERSION,
        ),
        RequiredArtifact(
            key="summary",
            candidates=("summary.json",),
            kind="json",
            expected_schema_version=SUMMARY_SCHEMA_VERSION,
        ),
        RequiredArtifact(
            key="ui_index",
            candidates=("ui_index.json",),
            kind="json",
            expected_schema_version=UI_INDEX_SCHEMA_VERSION,
        ),
    ]

    if "detection" in stage_set and save_detections:
        required.append(
            RequiredArtifact(
                key="detections",
                candidates=("detections.parquet", "detections.jsonl", "detections.csv"),
                kind="tabular",
                expected_schema_version=DETECTIONS_SCHEMA_VERSION,
            )
        )

    if "tracking" in stage_set and save_tracks:
        required.append(
            RequiredArtifact(
                key="tracks",
                candidates=("tracks.parquet", "tracks.jsonl", "tracks.csv"),
                kind="tabular",
                expected_schema_version=TRACKS_SCHEMA_VERSION,
            )
        )

    if "event_detection" in stage_set and save_events:
        required.append(
            RequiredArtifact(
                key="events",
                candidates=("events.jsonl",),
                kind="jsonl",
                expected_schema_version=EVENTS_SCHEMA_VERSION,
            )
        )
        required.append(
            RequiredArtifact(
                key="score_timeline",
                candidates=("score_timeline.json",),
                kind="json",
                expected_schema_version=SCORE_TIMELINE_SCHEMA_VERSION,
            )
        )

    if "overlay" in stage_set and save_overlay_video:
        required.append(
            RequiredArtifact(
                key="overlay",
                candidates=("overlay.mp4",),
                kind="path",
                expected_schema_version=None,
            )
        )

    return required


def validate_run_artifact_contract(
    output_dir: Path,
    stage_names: Iterable[str],
    *,
    save_detections: bool,
    save_tracks: bool,
    save_events: bool,
    save_overlay_video: bool,
) -> dict[str, str]:
    """Validate required outputs and schema-version contract for a run."""
    required = expected_required_artifacts(
        stage_names=stage_names,
        save_detections=save_detections,
        save_tracks=save_tracks,
        save_events=save_events,
        save_overlay_video=save_overlay_video,
    )

    resolved: dict[str, str] = {}
    for requirement in required:
        artifact_path = resolve_artifact_path(output_dir, requirement.candidates)
        if artifact_path is None:
            candidates = ", ".join(requirement.candidates)
            raise ArtifactContractError(
                f"Missing required artifact '{requirement.key}' (expected one of: {candidates})"
            )

        resolved[requirement.key] = str(artifact_path.relative_to(output_dir))

        expected_version = requirement.expected_schema_version
        if requirement.kind == "json":
            assert expected_version is not None
            if not has_json_schema_version(artifact_path, expected_version):
                raise ArtifactContractError(
                    f"Artifact '{requirement.key}' has invalid schema_version "
                    f"(expected {expected_version})"
                )
        elif requirement.kind == "jsonl":
            assert expected_version is not None
            if not has_jsonl_schema_version(artifact_path, expected_version):
                raise ArtifactContractError(
                    f"Artifact '{requirement.key}' has invalid row schema_version "
                    f"(expected {expected_version})"
                )
        elif requirement.kind == "tabular":
            assert expected_version is not None
            if not has_tabular_schema_version(artifact_path, expected_version):
                raise ArtifactContractError(
                    f"Artifact '{requirement.key}' is missing schema_version column "
                    f"or contains unexpected schema values (expected {expected_version})"
                )

    manifest_path = output_dir / "run_manifest.json"
    summary_path = output_dir / "summary.json"
    ui_index_path = output_dir / "ui_index.json"

    with open(manifest_path) as f:
        manifest = json.load(f)
    with open(summary_path) as f:
        summary = json.load(f)
    with open(ui_index_path) as f:
        ui_index = json.load(f)

    manifest_schemas = manifest.get("artifact_schemas", {})
    if not isinstance(manifest_schemas, dict):
        raise ArtifactContractError("run_manifest.json must include object field 'artifact_schemas'")

    for key, required_path in resolved.items():
        summary_artifacts = summary.get("artifacts", {})
        ui_artifacts = ui_index.get("artifacts", {})
        if not isinstance(summary_artifacts, dict):
            raise ArtifactContractError("summary.json field 'artifacts' must be an object")
        if not isinstance(ui_artifacts, dict):
            raise ArtifactContractError("ui_index.json field 'artifacts' must be an object")

        if key not in summary_artifacts:
            raise ArtifactContractError(
                f"summary.json artifact index missing required key '{key}'"
            )
        if key not in ui_artifacts:
            raise ArtifactContractError(
                f"ui_index.json artifact index missing required key '{key}'"
            )
        if str(summary_artifacts[key]) != required_path:
            raise ArtifactContractError(
                f"summary.json artifact '{key}' path mismatch "
                f"(expected {required_path}, got {summary_artifacts[key]})"
            )
        if str(ui_artifacts[key]) != required_path:
            raise ArtifactContractError(
                f"ui_index.json artifact '{key}' path mismatch "
                f"(expected {required_path}, got {ui_artifacts[key]})"
            )

        if key in ARTIFACT_SCHEMA_VERSIONS:
            expected_schema = ARTIFACT_SCHEMA_VERSIONS[key]
            if str(manifest_schemas.get(key)) != expected_schema:
                raise ArtifactContractError(
                    f"run_manifest.json artifact_schemas['{key}'] mismatch "
                    f"(expected {expected_schema}, got {manifest_schemas.get(key)})"
                )

    return resolved
