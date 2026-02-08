"""Unit tests for run artifact contract validation helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.pipeline.contracts import (
    ARTIFACT_SCHEMA_VERSIONS,
    DETECTIONS_SCHEMA_VERSION,
    EVENTS_SCHEMA_VERSION,
    SCORE_TIMELINE_SCHEMA_VERSION,
    TRACKS_SCHEMA_VERSION,
    ArtifactContractError,
    has_jsonl_schema_version,
    has_tabular_schema_version,
    validate_run_artifact_contract,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _write_core_index_files(run_dir: Path, artifacts: dict[str, str]) -> None:
    _write_json(
        run_dir / "run_manifest.json",
        {
            "schema_version": "1.1",
            "artifact_schemas": ARTIFACT_SCHEMA_VERSIONS,
        },
    )
    _write_json(
        run_dir / "summary.json",
        {
            "schema_version": "1.0",
            "artifacts": artifacts,
        },
    )
    _write_json(
        run_dir / "ui_index.json",
        {
            "schema_version": "1.0",
            "artifacts": artifacts,
        },
    )


def test_tabular_schema_helpers_support_parquet_and_jsonl(tmp_path: Path) -> None:
    parquet_path = tmp_path / "detections.parquet"
    pd.DataFrame(
        [
            {"schema_version": DETECTIONS_SCHEMA_VERSION, "frame_idx": 1},
            {"schema_version": DETECTIONS_SCHEMA_VERSION, "frame_idx": 2},
        ]
    ).to_parquet(parquet_path, index=False)

    jsonl_path = tmp_path / "events.jsonl"
    with open(jsonl_path, "w") as f:
        f.write(json.dumps({"schema_version": EVENTS_SCHEMA_VERSION, "event_type": "shot"}) + "\n")

    assert has_tabular_schema_version(parquet_path, DETECTIONS_SCHEMA_VERSION)
    assert has_jsonl_schema_version(jsonl_path, EVENTS_SCHEMA_VERSION)


def test_validate_run_artifact_contract_passes_for_required_stage_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    pd.DataFrame(
        [{"schema_version": DETECTIONS_SCHEMA_VERSION, "frame_idx": 0, "object_type": "player"}]
    ).to_parquet(run_dir / "detections.parquet", index=False)
    pd.DataFrame(
        [{"schema_version": TRACKS_SCHEMA_VERSION, "frame_idx": 0, "track_id": 1}]
    ).to_parquet(run_dir / "tracks.parquet", index=False)

    with open(run_dir / "events.jsonl", "w") as f:
        f.write(
            json.dumps(
                {
                    "schema_version": EVENTS_SCHEMA_VERSION,
                    "event_type": "shot",
                    "frame_idx": 3,
                    "timestamp": 0.1,
                    "confidence": 0.9,
                    "location": None,
                    "metadata": {},
                }
            )
            + "\n"
        )

    _write_json(
        run_dir / "score_timeline.json",
        {
            "schema_version": SCORE_TIMELINE_SCHEMA_VERSION,
            "goals": 0,
            "final_score": {"team_a": 0, "team_b": 0},
            "timeline": [],
        },
    )
    (run_dir / "overlay.mp4").write_bytes(b"")

    artifacts = {
        "run_manifest": "run_manifest.json",
        "summary": "summary.json",
        "ui_index": "ui_index.json",
        "detections": "detections.parquet",
        "tracks": "tracks.parquet",
        "events": "events.jsonl",
        "score_timeline": "score_timeline.json",
        "overlay": "overlay.mp4",
    }
    _write_core_index_files(run_dir, artifacts)

    resolved = validate_run_artifact_contract(
        output_dir=run_dir,
        stage_names=("detection", "tracking", "event_detection", "overlay"),
        save_detections=True,
        save_tracks=True,
        save_events=True,
        save_overlay_video=True,
    )

    assert resolved["detections"] == "detections.parquet"
    assert resolved["tracks"] == "tracks.parquet"
    assert resolved["events"] == "events.jsonl"
    assert resolved["overlay"] == "overlay.mp4"


def test_validate_run_artifact_contract_fails_on_missing_event_schema(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with open(run_dir / "events.jsonl", "w") as f:
        f.write(json.dumps({"event_type": "shot", "frame_idx": 2}) + "\n")

    _write_json(
        run_dir / "score_timeline.json",
        {
            "schema_version": SCORE_TIMELINE_SCHEMA_VERSION,
            "goals": 0,
            "final_score": {"team_a": 0, "team_b": 0},
            "timeline": [],
        },
    )

    artifacts = {
        "run_manifest": "run_manifest.json",
        "summary": "summary.json",
        "ui_index": "ui_index.json",
        "events": "events.jsonl",
        "score_timeline": "score_timeline.json",
    }
    _write_core_index_files(run_dir, artifacts)

    with pytest.raises(ArtifactContractError, match="events"):
        validate_run_artifact_contract(
            output_dir=run_dir,
            stage_names=("event_detection",),
            save_detections=True,
            save_tracks=True,
            save_events=True,
            save_overlay_video=True,
        )
