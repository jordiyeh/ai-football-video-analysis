"""FastAPI server for local web UI."""

import json
import hashlib
import csv
import io
import zipfile
import os
import traceback
import threading
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn


class ConfirmRejectBody(BaseModel):
    """Request body for confirm/reject endpoints."""
    notes: str = ""


class AddEventBody(BaseModel):
    """Request body for adding a manual event."""
    event_type: str
    timestamp: float
    frame_idx: int
    notes: str = ""
    metadata: dict = {}


class UpdatePlayerBody(BaseModel):
    """Request body for updating a player."""
    name: str | None = None
    jersey_number: int | None = None
    team_hint: str | None = None
    team_id: int | None = None


class AssignAppearanceBody(BaseModel):
    """Request body for assigning an appearance to a player."""
    confidence: float = 1.0


class CreatePlayerBody(BaseModel):
    """Request body for creating a player."""
    name: str | None = None
    jersey_number: int | None = None
    team_hint: str | None = None


class RecomputePlayerReelsBody(BaseModel):
    """Request body for recomputing per-player reels."""
    preserve_existing_clips: bool = True


class BulkAssignAppearancesBody(BaseModel):
    """Request body for bulk assignment of tracks to one player."""
    track_ids: list[int]
    player_id: int | None
    confidence: float = 1.0
    method: str = "manual"


class AssignTrackBody(BaseModel):
    """Request body for single-track assignment in a run."""
    track_id: int
    player_id: int | None
    confidence: float = 1.0
    method: str = "manual"


class ApplyIdentitySuggestionsBody(BaseModel):
    """Request body for applying identity suggestions."""
    track_ids: list[int] | None = None
    min_confidence: float = 0.7
    suggested_only: bool = True


class ApplyIdentitySuggestionsAndRecomputeBody(ApplyIdentitySuggestionsBody):
    """Request body for one-click apply suggestions + recompute reels."""
    preserve_existing_clips: bool = True


class ApprovePlayerReelsPreviewBody(BaseModel):
    """Request body for approving/persisting a stored player reel preview."""
    preview_id: str | None = None


class ExportPlayerReelsPackageBody(BaseModel):
    """Request body for exporting filtered player reels as a ZIP package."""
    team_filter: str = "all"
    min_score: float = 0.0
    top_n: int = 8
    sort_by: str = "best_score_desc"
    include_clips: bool = True
    player_ids: list[int] | None = None


class ExportCrossMatchPackageBody(BaseModel):
    """Request body for exporting cross-match report artifacts as ZIP package."""
    include_templates: bool = True


class QueuePipelineJobsBody(BaseModel):
    """Request body for queueing one or more pipeline analysis jobs."""
    video_paths: list[str]
    run_name: str | None = None
    run_name_prefix: str | None = None
    config_path: str | None = None
    resume: bool = False
    no_overlay: bool = False
    home_team_id: int | None = None
    away_team_id: int | None = None
    home_kit: str = "home"
    away_kit: str = "home"


class CreateTeamBody(BaseModel):
    """Request body for creating a team."""
    name: str
    short_name: str | None = None


class UpdateTeamBody(BaseModel):
    """Request body for updating a team."""
    name: str | None = None
    short_name: str | None = None


class SetRunTeamsBody(BaseModel):
    """Request body for associating teams with a run."""
    home_team_id: int
    away_team_id: int
    home_kit: str = "home"
    away_kit: str = "home"


class RemapRunTeamsBody(BaseModel):
    """Request body for swapping cluster-to-team mapping."""
    pass


def _utc_now_iso() -> str:
    """Return timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _sanitize_run_component(value: str) -> str:
    """Sanitize user-provided run name components."""
    safe_chars = []
    for ch in value.strip():
        if ch.isalnum() or ch in {"-", "_", "."}:
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    safe = "".join(safe_chars).strip("._-")
    return safe[:96]


class PipelineJobCancelledError(RuntimeError):
    """Internal exception for graceful pipeline job cancellation."""


def generate_event_id(event: dict) -> str:
    """Generate a stable ID for an auto-detected event."""
    event_type = event.get("event_type", "unknown")
    frame_idx = event.get("frame_idx", 0)
    return f"auto_{event_type}_{frame_idx}"


def load_confirmations(run_path: Path) -> dict:
    """
    Load confirmations from events_confirmed.jsonl.

    Returns a dict mapping event_id to the latest action record.
    For manual events, includes the full event data.
    """
    confirmations_path = run_path / "events_confirmed.jsonl"
    confirmations = {}

    if not confirmations_path.exists():
        return confirmations

    with open(confirmations_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            event_id = record.get("event_id")
            if event_id:
                # Last action wins
                confirmations[event_id] = record

    return confirmations


def save_confirmation(run_path: Path, action_record: dict) -> None:
    """Append an action record to events_confirmed.jsonl."""
    confirmations_path = run_path / "events_confirmed.jsonl"

    # Add timestamp
    action_record["recorded_at"] = datetime.now(timezone.utc).isoformat()

    with open(confirmations_path, "a") as f:
        f.write(json.dumps(action_record) + "\n")


def merge_events_with_confirmations(events: list, confirmations: dict) -> list:
    """
    Merge auto-detected events with user confirmations.

    Returns events with added fields:
    - id: unique event identifier
    - status: pending/confirmed/rejected
    - source: auto/manual
    - user_notes: notes from confirmation (if any)
    """
    merged = []

    # Process auto-detected events
    for event in events:
        event_id = generate_event_id(event)
        event_copy = event.copy()
        event_copy["id"] = event_id
        event_copy["source"] = "auto"

        # Check for confirmation status
        if event_id in confirmations:
            action = confirmations[event_id].get("action")
            if action == "confirm":
                event_copy["status"] = "confirmed"
            elif action == "reject":
                event_copy["status"] = "rejected"
            elif action == "delete":
                continue  # Skip deleted events
            else:
                event_copy["status"] = "pending"
            event_copy["user_notes"] = confirmations[event_id].get("notes", "")
        else:
            event_copy["status"] = "pending"
            event_copy["user_notes"] = ""

        merged.append(event_copy)

    # Add manual events
    for event_id, record in confirmations.items():
        if record.get("action") == "add" and event_id.startswith("manual_"):
            # Check if later deleted
            if confirmations.get(event_id, {}).get("action") == "delete":
                continue

            event_data = record.get("event", {})
            event_data["id"] = event_id
            event_data["source"] = "manual"
            event_data["status"] = "confirmed"  # Manual events are auto-confirmed
            event_data["user_notes"] = record.get("notes", "")
            merged.append(event_data)

    # Sort by timestamp
    merged.sort(key=lambda e: e.get("timestamp", 0))

    return merged


def create_app(runs_dir: Path = Path("runs")) -> FastAPI:
    """Create FastAPI application."""
    max_parallel_jobs = max(1, min(int(os.getenv("VEO_UI_MAX_PARALLEL_JOBS", "1")), 8))
    pipeline_executor = ThreadPoolExecutor(
        max_workers=max_parallel_jobs,
        thread_name_prefix="veo-ui-pipeline",
    )

    @asynccontextmanager
    async def app_lifespan(_: FastAPI):
        try:
            yield
        finally:
            pipeline_executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(
        title="Veo Soccer Analysis UI",
        description="Local web interface for soccer match analysis",
        version="0.4.0",
        lifespan=app_lifespan,
    )

    # Serve static files (HTML, JS, CSS)
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    project_root = Path(__file__).resolve().parents[2]
    runs_dir = runs_dir.expanduser().resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)

    pipeline_jobs_lock = threading.Lock()
    pipeline_jobs: dict[str, dict[str, Any]] = {}
    pipeline_job_order: list[str] = []
    pipeline_job_futures: dict[str, Any] = {}
    max_pipeline_job_history = 200
    pipeline_jobs_path = runs_dir / ".ui_pipeline_jobs.json"

    def resolve_user_path(raw_value: str) -> Path:
        """Resolve user path relative to workspace root when needed."""
        candidate = Path(raw_value).expanduser()
        if not candidate.is_absolute():
            candidate = project_root / candidate
        return candidate.resolve()

    def _serialize_pipeline_job_for_storage(job: dict[str, Any]) -> dict[str, Any]:
        """Normalize one pipeline job for JSON storage."""
        return {
            "job_id": str(job.get("job_id") or ""),
            "status": str(job.get("status") or "queued"),
            "video_path": str(job.get("video_path") or ""),
            "run_name": str(job.get("run_name") or ""),
            "output_dir": str(job.get("output_dir") or ""),
            "config_path": job.get("config_path"),
            "resume": bool(job.get("resume", False)),
            "no_overlay": bool(job.get("no_overlay", False)),
            "created_at": job.get("created_at"),
            "started_at": job.get("started_at"),
            "finished_at": job.get("finished_at"),
            "stage_name": job.get("stage_name"),
            "stage_index": int(job.get("stage_index", 0) or 0),
            "stage_total": int(job.get("stage_total", 0) or 0),
            "message": job.get("message"),
            "error": job.get("error"),
            "logs": list(job.get("logs", [])),
            "cancel_requested": bool(job.get("cancel_requested", False)),
            "source_job_id": job.get("source_job_id"),
        }

    def persist_pipeline_jobs_locked() -> None:
        """Persist job state to disk (call while holding pipeline_jobs_lock)."""
        payload = {
            "schema_version": "1.0",
            "updated_at": _utc_now_iso(),
            "jobs": [
                _serialize_pipeline_job_for_storage(pipeline_jobs[job_id])
                for job_id in pipeline_job_order
                if job_id in pipeline_jobs
            ],
        }
        temp_path = Path(f"{pipeline_jobs_path}.tmp")
        try:
            with open(temp_path, "w") as f:
                json.dump(payload, f, indent=2)
            temp_path.replace(pipeline_jobs_path)
        except Exception:
            # Keep runtime UX responsive even if persistence write fails.
            pass

    def append_pipeline_job_log(job_id: str, message: str, persist: bool = True) -> None:
        """Append one timestamped log line to a pipeline job."""
        with pipeline_jobs_lock:
            job = pipeline_jobs.get(job_id)
            if job is None:
                return
            logs = job.setdefault("logs", [])
            logs.append(
                {
                    "timestamp": _utc_now_iso(),
                    "message": str(message),
                }
            )
            if len(logs) > 400:
                del logs[:-400]
            if persist:
                persist_pipeline_jobs_locked()

    def serialize_pipeline_job(job: dict[str, Any], include_logs: bool = False) -> dict[str, Any]:
        """Return API-safe pipeline job payload."""
        payload = {
            "job_id": job.get("job_id"),
            "status": job.get("status"),
            "video_path": job.get("video_path"),
            "video_name": Path(str(job.get("video_path", ""))).name,
            "run_name": job.get("run_name"),
            "output_dir": job.get("output_dir"),
            "config_path": job.get("config_path"),
            "resume": bool(job.get("resume", False)),
            "no_overlay": bool(job.get("no_overlay", False)),
            "created_at": job.get("created_at"),
            "started_at": job.get("started_at"),
            "finished_at": job.get("finished_at"),
            "stage_name": job.get("stage_name"),
            "stage_index": int(job.get("stage_index", 0) or 0),
            "stage_total": int(job.get("stage_total", 0) or 0),
            "message": job.get("message"),
            "error": job.get("error"),
            "cancel_requested": bool(job.get("cancel_requested", False)),
            "source_job_id": job.get("source_job_id"),
        }
        if include_logs:
            payload["logs"] = list(job.get("logs", []))
        return payload

    def prune_pipeline_job_history() -> None:
        """Drop oldest completed jobs from in-memory history."""
        if len(pipeline_job_order) <= max_pipeline_job_history:
            return

        removable_ids: list[str] = []
        for job_id in pipeline_job_order:
            if len(pipeline_job_order) - len(removable_ids) <= max_pipeline_job_history:
                break
            job = pipeline_jobs.get(job_id)
            if job is None:
                removable_ids.append(job_id)
                continue
            if job.get("status") in {"succeeded", "failed", "cancelled"}:
                removable_ids.append(job_id)

        for job_id in removable_ids:
            pipeline_jobs.pop(job_id, None)
            pipeline_job_futures.pop(job_id, None)
            if job_id in pipeline_job_order:
                pipeline_job_order.remove(job_id)

    def ensure_unique_run_name(base_name: str, reserved_names: set[str]) -> str:
        """Generate a run name that is unique across disk and queued/running jobs."""
        normalized = _sanitize_run_component(base_name)
        if not normalized:
            normalized = f"analysis_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        candidate = normalized
        suffix = 2
        while candidate in reserved_names or (runs_dir / candidate).exists():
            candidate = f"{normalized}_{suffix:02d}"
            suffix += 1
        reserved_names.add(candidate)
        return candidate

    def load_pipeline_jobs_from_disk() -> None:
        """Load persisted job state."""
        if not pipeline_jobs_path.exists():
            return

        try:
            with open(pipeline_jobs_path) as f:
                payload = json.load(f)
        except Exception:
            return

        raw_jobs = payload.get("jobs")
        if not isinstance(raw_jobs, list):
            return

        with pipeline_jobs_lock:
            for row in raw_jobs:
                if not isinstance(row, dict):
                    continue
                job_id = str(row.get("job_id") or "").strip()
                if not job_id:
                    continue

                job = _serialize_pipeline_job_for_storage(row)
                if job["status"] in {"queued", "running"}:
                    job["status"] = "failed"
                    job["error"] = "Pipeline server restarted before completion."
                    job["message"] = "Interrupted by server restart"
                    job["finished_at"] = _utc_now_iso()
                    logs = list(job.get("logs", []))
                    logs.append(
                        {
                            "timestamp": _utc_now_iso(),
                            "message": "Marked as failed after server restart",
                        }
                    )
                    job["logs"] = logs[-400:]
                    job["cancel_requested"] = False

                pipeline_jobs[job_id] = job
                pipeline_job_order.append(job_id)

            prune_pipeline_job_history()
            persist_pipeline_jobs_locked()

    def build_pipeline_job_record(
        *,
        job_id: str,
        video_path: str,
        run_name: str,
        output_dir: str,
        config_path: str | None,
        resume: bool,
        no_overlay: bool,
        source_job_id: str | None = None,
        home_team_id: int | None = None,
        away_team_id: int | None = None,
        home_kit: str = "home",
        away_kit: str = "home",
    ) -> dict[str, Any]:
        """Create a normalized job record."""
        return {
            "job_id": job_id,
            "status": "queued",
            "video_path": str(video_path),
            "run_name": run_name,
            "output_dir": str(output_dir),
            "config_path": config_path,
            "resume": bool(resume),
            "no_overlay": bool(no_overlay),
            "created_at": _utc_now_iso(),
            "started_at": None,
            "finished_at": None,
            "stage_name": None,
            "stage_index": 0,
            "stage_total": 0,
            "message": "Queued",
            "error": None,
            "logs": [],
            "cancel_requested": False,
            "source_job_id": source_job_id,
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "home_kit": home_kit,
            "away_kit": away_kit,
        }

    def submit_pipeline_job(job_id: str) -> None:
        """Submit a queued job to the background executor."""
        future = pipeline_executor.submit(execute_pipeline_job, job_id)
        with pipeline_jobs_lock:
            pipeline_job_futures[job_id] = future
            persist_pipeline_jobs_locked()

    def queue_one_pipeline_job(
        *,
        video_path: Path,
        run_name: str,
        config_path: str | None,
        resume: bool,
        no_overlay: bool,
        source_job_id: str | None = None,
        home_team_id: int | None = None,
        away_team_id: int | None = None,
        home_kit: str = "home",
        away_kit: str = "home",
    ) -> dict[str, Any]:
        """Create + queue one job and return serialized payload."""
        output_dir = (runs_dir / run_name).resolve()
        try:
            output_dir.relative_to(runs_dir)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid run name: {run_name}")

        if output_dir.exists() and not resume:
            raise HTTPException(
                status_code=409,
                detail=f"Output already exists for run '{run_name}'. Use resume=true or choose another name.",
            )

        job_id = f"job_{int(datetime.now(timezone.utc).timestamp() * 1000)}_{uuid4().hex[:8]}"
        job_payload = build_pipeline_job_record(
            job_id=job_id,
            video_path=str(video_path),
            run_name=run_name,
            output_dir=str(output_dir),
            config_path=config_path,
            resume=resume,
            no_overlay=no_overlay,
            source_job_id=source_job_id,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_kit=home_kit,
            away_kit=away_kit,
        )

        # Store run-team association in DB when teams are pre-selected
        if home_team_id is not None and away_team_id is not None:
            try:
                from src.identity import PlayerDatabase
                db_path = get_player_db_path()
                with PlayerDatabase(db_path) as db:
                    db.set_run_teams(run_name, home_team_id, away_team_id, home_kit, away_kit)
            except Exception:
                pass  # Non-fatal: DB association is best-effort

        with pipeline_jobs_lock:
            for existing in pipeline_jobs.values():
                if (
                    str(existing.get("run_name") or "") == run_name
                    and str(existing.get("status") or "") in {"queued", "running"}
                ):
                    raise HTTPException(
                        status_code=409,
                        detail=f"Run name '{run_name}' is already queued/running. Choose a different name.",
                    )
            pipeline_jobs[job_id] = job_payload
            pipeline_job_order.append(job_id)
            prune_pipeline_job_history()
            persist_pipeline_jobs_locked()

        submit_pipeline_job(job_id)
        append_pipeline_job_log(job_id, "Job queued")
        with pipeline_jobs_lock:
            return serialize_pipeline_job(pipeline_jobs[job_id], include_logs=False)

    def build_pipeline_for_job(
        pipeline_config: Any,
        no_overlay: bool,
    ) -> Any:
        """Build pipeline using the same stage ordering as CLI."""
        from src.cli import (
            IngestStage,
            DetectionStage,
            TrackingStage,
            TeamAssignmentStage,
            FieldNormalizationStage,
            ProfileIngestionStage,
            PlayerIdentityStage,
            TeamAnalyticsStage,
            EventDetectionStage,
            HighlightGenerationStage,
            PlayerHighlightReelsStage,
            CrossMatchReportingStage,
            OverlayStage,
        )
        from src.pipeline.base import Pipeline

        pipeline = Pipeline(pipeline_config)
        pipeline.add_stage(IngestStage(pipeline_config))
        pipeline.add_stage(DetectionStage(pipeline_config))
        pipeline.add_stage(TrackingStage(pipeline_config))
        pipeline.add_stage(TeamAssignmentStage(pipeline_config))
        pipeline.add_stage(FieldNormalizationStage(pipeline_config))
        pipeline.add_stage(ProfileIngestionStage(pipeline_config))
        pipeline.add_stage(PlayerIdentityStage(pipeline_config))
        pipeline.add_stage(TeamAnalyticsStage(pipeline_config))
        pipeline.add_stage(EventDetectionStage(pipeline_config))
        pipeline.add_stage(HighlightGenerationStage(pipeline_config))
        pipeline.add_stage(PlayerHighlightReelsStage(pipeline_config))
        pipeline.add_stage(CrossMatchReportingStage(pipeline_config))
        if not no_overlay:
            pipeline.add_stage(OverlayStage(pipeline_config))
        return pipeline

    def execute_pipeline_job(job_id: str) -> None:
        """Background worker for one queued pipeline job."""
        from src.config.schemas import PipelineConfig
        from src.pipeline.base import PipelineCancelledError

        stage_total = 0
        with pipeline_jobs_lock:
            job = pipeline_jobs.get(job_id)
            if job is None:
                return
            if job.get("status") != "queued":
                return
            if bool(job.get("cancel_requested", False)):
                job["status"] = "cancelled"
                job["message"] = "Cancelled before start"
                job["finished_at"] = _utc_now_iso()
                persist_pipeline_jobs_locked()
                return
            job["status"] = "running"
            job["started_at"] = _utc_now_iso()
            job["message"] = "Preparing pipeline"
            job["error"] = None
            persist_pipeline_jobs_locked()

        append_pipeline_job_log(job_id, "Preparing pipeline job")

        try:
            with pipeline_jobs_lock:
                job = pipeline_jobs.get(job_id)
                if job is None:
                    return
                config_path_value = job.get("config_path")
                video_path_value = str(job.get("video_path"))
                output_dir_value = str(job.get("output_dir"))
                resume_value = bool(job.get("resume", False))
                no_overlay_value = bool(job.get("no_overlay", False))
                home_team_id_value = job.get("home_team_id")
                away_team_id_value = job.get("away_team_id")
                home_kit_value = str(job.get("home_kit") or "home")
                away_kit_value = str(job.get("away_kit") or "home")

            if config_path_value:
                pipeline_config = PipelineConfig.from_yaml(config_path_value)
                append_pipeline_job_log(job_id, f"Loaded config: {config_path_value}")
            else:
                pipeline_config = PipelineConfig()
                append_pipeline_job_log(job_id, "Using built-in default config")

            pipeline_config.output_dir = output_dir_value

            # Inject pre-selected team IDs into config
            if home_team_id_value is not None:
                pipeline_config.team.home_team_id = int(home_team_id_value)
            if away_team_id_value is not None:
                pipeline_config.team.away_team_id = int(away_team_id_value)
            pipeline_config.team.home_kit = home_kit_value
            pipeline_config.team.away_kit = away_kit_value
            pipeline = build_pipeline_for_job(
                pipeline_config=pipeline_config,
                no_overlay=no_overlay_value,
            )

            stage_total = len(pipeline.stages)
            with pipeline_jobs_lock:
                job = pipeline_jobs.get(job_id)
                if job is not None:
                    job["stage_total"] = stage_total
                    persist_pipeline_jobs_locked()

            for stage_index, stage in enumerate(pipeline.stages, start=1):
                original_run = stage.run
                stage_name = stage.name

                def wrapped_stage_run(
                    context: dict[str, Any],
                    _original_run=original_run,
                    _stage_name=stage_name,
                    _stage_index=stage_index,
                ) -> dict[str, Any]:
                    with pipeline_jobs_lock:
                        target_job = pipeline_jobs.get(job_id)
                        if target_job is not None:
                            if bool(target_job.get("cancel_requested", False)):
                                raise PipelineJobCancelledError("Cancellation requested")
                            target_job["stage_name"] = _stage_name
                            target_job["stage_index"] = _stage_index
                            target_job["message"] = (
                                f"Running stage {_stage_index}/{stage_total}: {_stage_name}"
                            )
                            persist_pipeline_jobs_locked()
                    append_pipeline_job_log(
                        job_id,
                        f"Stage {_stage_index}/{stage_total}: {_stage_name}",
                    )

                    def _check_cancel() -> bool:
                        with pipeline_jobs_lock:
                            j = pipeline_jobs.get(job_id)
                            return bool(j.get("cancel_requested", False)) if j else False

                    context["_check_cancel"] = _check_cancel
                    return _original_run(context)

                stage.run = wrapped_stage_run  # type: ignore[assignment]

            pipeline.run(
                video_path=video_path_value,
                output_dir=output_dir_value,
                resume=resume_value,
            )

            with pipeline_jobs_lock:
                completed = pipeline_jobs.get(job_id)
                if completed is not None:
                    completed["status"] = "succeeded"
                    completed["stage_name"] = "completed"
                    completed["stage_index"] = stage_total
                    completed["message"] = "Pipeline completed"
                    completed["finished_at"] = _utc_now_iso()
                    completed["cancel_requested"] = False
                    persist_pipeline_jobs_locked()

            append_pipeline_job_log(job_id, "Pipeline completed successfully")
        except (PipelineJobCancelledError, PipelineCancelledError):
            with pipeline_jobs_lock:
                cancelled = pipeline_jobs.get(job_id)
                if cancelled is not None:
                    cancelled["status"] = "cancelled"
                    cancelled["error"] = None
                    cancelled["message"] = "Pipeline cancelled by user"
                    cancelled["finished_at"] = _utc_now_iso()
                    cancelled["cancel_requested"] = False
                    persist_pipeline_jobs_locked()
            append_pipeline_job_log(job_id, "Pipeline cancelled by user")
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            traceback_text = traceback.format_exc(limit=30)
            with pipeline_jobs_lock:
                failed = pipeline_jobs.get(job_id)
                if failed is not None:
                    failed["status"] = "failed"
                    failed["error"] = error_message
                    failed["finished_at"] = _utc_now_iso()
                    failed["message"] = "Pipeline failed"
                    failed["cancel_requested"] = False
                    persist_pipeline_jobs_locked()
            append_pipeline_job_log(job_id, error_message)
            append_pipeline_job_log(job_id, traceback_text)
        finally:
            with pipeline_jobs_lock:
                pipeline_job_futures.pop(job_id, None)
                persist_pipeline_jobs_locked()

    load_pipeline_jobs_from_disk()

    def get_player_db_path() -> Path:
        """Get path to player database."""
        db_path = runs_dir / "players.db"
        if db_path.exists():
            return db_path

        # Optional backward-compatible fallback for legacy layouts.
        # Disabled by default to keep tests and temporary run directories isolated.
        allow_legacy = os.getenv("VEO_UI_ALLOW_LEGACY_PLAYER_DB", "0").lower() in {"1", "true", "yes"}
        legacy_db_path = project_root / "players.db"
        if allow_legacy and legacy_db_path.exists():
            return legacy_db_path

        return db_path

    def resolve_run_artifact_path(run_path: Path, value: Any) -> Path | None:
        """Resolve an artifact path and ensure it remains inside the run directory."""
        if value is None:
            return None

        try:
            raw_path = Path(str(value))
        except Exception:
            return None

        candidate = raw_path if raw_path.is_absolute() else (run_path / raw_path)
        try:
            candidate = candidate.resolve()
            run_root = run_path.resolve()
        except Exception:
            return None

        if not candidate.exists() or not candidate.is_file():
            return None

        try:
            candidate.relative_to(run_root)
        except ValueError:
            return None

        return candidate

    def infer_run_video_id(run_path: Path) -> str | None:
        """Infer run video_id from available artifacts."""
        assignments_path = run_path / "player_assignments.json"
        if assignments_path.exists():
            with open(assignments_path) as f:
                assignments_data = json.load(f)
            video_id = assignments_data.get("video_id")
            if isinstance(video_id, str) and video_id:
                return video_id

        player_highlights_path = run_path / "player_highlights.json"
        if player_highlights_path.exists():
            with open(player_highlights_path) as f:
                reels_data = json.load(f)
            video_id = reels_data.get("video_id")
            if isinstance(video_id, str) and video_id:
                return video_id

        manifest_path = run_path / "run_manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            original_video_path = manifest.get("original_video_path")
            if isinstance(original_video_path, str) and original_video_path:
                return Path(original_video_path).stem

        return None

    def load_identity_review_for_run(run_name: str) -> dict[str, Any]:
        """Load run assignments from DB/file for identity review UI."""
        run_path = runs_dir / run_name
        if not run_path.exists():
            raise HTTPException(status_code=404, detail="Run not found")

        video_id = infer_run_video_id(run_path)

        assignments_file_rows: list[dict[str, Any]] = []
        assignments_file_by_track: dict[int, dict[str, Any]] = {}
        assignments_path = run_path / "player_assignments.json"
        if assignments_path.exists():
            with open(assignments_path) as f:
                assignments_data = json.load(f)
            for assignment in assignments_data.get("assignments", []):
                track_id = assignment.get("track_id")
                if track_id is None:
                    continue
                try:
                    track_id_int = int(track_id)
                except (TypeError, ValueError):
                    continue

                fusion_raw = assignment.get("fusion")
                fusion = fusion_raw if isinstance(fusion_raw, dict) else {}
                strategy = fusion.get("strategy")

                multimodal_raw = fusion.get("multimodal")
                multimodal = multimodal_raw if isinstance(multimodal_raw, dict) else {}

                face_raw = multimodal.get("face")
                face = face_raw if isinstance(face_raw, dict) else {}

                jersey_raw = multimodal.get("jersey_ocr")
                jersey = jersey_raw if isinstance(jersey_raw, dict) else {}

                applied_raw = multimodal.get("applied")
                applied = applied_raw if isinstance(applied_raw, list) else []

                lock_conflict_track = assignment.get("lock_conflict_with_track_id")
                try:
                    lock_conflict_track = (
                        int(lock_conflict_track)
                        if lock_conflict_track is not None
                        else None
                    )
                except (TypeError, ValueError):
                    lock_conflict_track = None

                row = {
                    "track_id": track_id_int,
                    "player_id": assignment.get("player_id"),
                    "player_name": assignment.get("player_name"),
                    "jersey_number": None,
                    "team_hint": None,
                    "match_confidence": assignment.get("confidence"),
                    "match_method": assignment.get("match_method"),
                    "frame_start": assignment.get("frame_start"),
                    "frame_end": assignment.get("frame_end"),
                    "fusion_strategy": strategy,
                    "multimodal_applied": applied,
                    "face_player_id": face.get("player_id"),
                    "face_confidence": face.get("confidence"),
                    "face_support_frames": face.get("support_frames"),
                    "face_backend": face.get("backend"),
                    "jersey_number_detected": jersey.get("jersey_number"),
                    "jersey_ocr_player_id": jersey.get("player_id"),
                    "jersey_ocr_confidence": jersey.get("confidence"),
                    "jersey_ocr_support_frames": jersey.get("support_frames"),
                    "jersey_ocr_ambiguous": jersey.get("ambiguous"),
                    "lock_state": assignment.get("lock_state"),
                    "lock_reason": assignment.get("lock_reason"),
                    "lock_conflict_with_track_id": lock_conflict_track,
                }
                assignments_file_rows.append(
                    row
                )
                assignments_file_by_track[track_id_int] = row

        db_path = get_player_db_path()
        players_payload: list[dict[str, Any]] = []
        assignment_rows: list[dict[str, Any]] = []

        if db_path.exists():
            try:
                from src.identity import PlayerDatabase

                with PlayerDatabase(db_path) as db:
                    players = db.list_players()
                    players_payload = [p.model_dump() for p in players]
                    player_lookup = {int(p.player_id): p for p in players}

                    if video_id:
                        appearances = [
                            a
                            for a in db.get_appearances_for_video(video_id)
                            if a.run_name == run_name
                        ]

                        if appearances:
                            for appearance in appearances:
                                player_id = appearance.player_id
                                player = player_lookup.get(int(player_id)) if player_id is not None else None
                                row = {
                                    "track_id": appearance.track_id,
                                    "player_id": player_id,
                                    "player_name": player.name if player else None,
                                    "jersey_number": player.jersey_number if player else None,
                                    "team_hint": player.team_hint if player else None,
                                    "match_confidence": appearance.match_confidence,
                                    "match_method": appearance.match_method,
                                    "frame_start": appearance.frame_start,
                                    "frame_end": appearance.frame_end,
                                    "fusion_strategy": None,
                                    "multimodal_applied": [],
                                    "face_player_id": None,
                                    "face_confidence": None,
                                    "face_support_frames": None,
                                    "face_backend": None,
                                    "jersey_number_detected": None,
                                    "jersey_ocr_player_id": None,
                                    "jersey_ocr_confidence": None,
                                    "jersey_ocr_support_frames": None,
                                    "jersey_ocr_ambiguous": None,
                                    "lock_state": None,
                                    "lock_reason": None,
                                    "lock_conflict_with_track_id": None,
                                }

                                file_row = assignments_file_by_track.get(int(appearance.track_id))
                                if file_row is not None:
                                    for key in (
                                        "fusion_strategy",
                                        "multimodal_applied",
                                        "face_player_id",
                                        "face_confidence",
                                        "face_support_frames",
                                        "face_backend",
                                        "jersey_number_detected",
                                        "jersey_ocr_player_id",
                                        "jersey_ocr_confidence",
                                        "jersey_ocr_support_frames",
                                        "jersey_ocr_ambiguous",
                                        "lock_state",
                                        "lock_reason",
                                        "lock_conflict_with_track_id",
                                    ):
                                        row[key] = file_row.get(key)

                                assignment_rows.append(row)

                    if not assignment_rows:
                        for row in assignments_file_rows:
                            player_id = row.get("player_id")
                            if player_id is None:
                                assignment_rows.append(row)
                                continue
                            try:
                                player = player_lookup.get(int(player_id))
                            except (TypeError, ValueError):
                                player = None
                            enriched = dict(row)
                            if player is not None:
                                if not enriched.get("player_name"):
                                    enriched["player_name"] = player.name
                                enriched["jersey_number"] = player.jersey_number
                                enriched["team_hint"] = player.team_hint
                            assignment_rows.append(enriched)
            except ImportError:
                assignment_rows = assignments_file_rows
        else:
            assignment_rows = assignments_file_rows

        if not assignment_rows:
            assignment_rows = assignments_file_rows

        assignment_rows.sort(key=lambda row: int(row.get("track_id", 0)))

        summary = {
            "total_assignments": len(assignment_rows),
            "assigned": sum(1 for row in assignment_rows if row.get("player_id") is not None),
            "unassigned": sum(1 for row in assignment_rows if row.get("player_id") is None),
            "manual": sum(1 for row in assignment_rows if row.get("match_method") == "manual"),
            "auto": sum(1 for row in assignment_rows if row.get("match_method") == "auto"),
            "suggested": sum(1 for row in assignment_rows if row.get("match_method") == "suggested"),
            "locked": sum(1 for row in assignment_rows if row.get("lock_state") == "locked"),
            "unlocked": sum(1 for row in assignment_rows if row.get("lock_state") == "unlocked"),
            "candidate": sum(1 for row in assignment_rows if row.get("lock_state") == "candidate"),
            "lock_conflicts": sum(
                1 for row in assignment_rows if row.get("lock_reason") == "overlap_conflict"
            ),
        }

        return {
            "run_name": run_name,
            "video_id": video_id,
            "players": players_payload,
            "assignments": assignment_rows,
            "summary": summary,
            "count": len(assignment_rows),
        }

    def load_player_reels_for_run(run_name: str) -> dict[str, Any]:
        """Load per-player reels for a run with optional player metadata enrichment."""
        run_path = runs_dir / run_name
        if not run_path.exists():
            raise HTTPException(status_code=404, detail="Run not found")

        reels_path = run_path / "player_highlights.json"
        if not reels_path.exists():
            raise HTTPException(status_code=404, detail="Player reels not found")

        with open(reels_path) as f:
            reels_data = json.load(f)

        players: list[dict[str, Any]] = []
        for player in reels_data.get("players", []):
            player_copy = dict(player)
            segments = []
            for segment in player.get("segments", []):
                segment_copy = dict(segment)
                clip_file = resolve_run_artifact_path(run_path, segment_copy.get("clip_path"))
                segment_copy["has_clip"] = clip_file is not None
                segments.append(segment_copy)
            player_copy["segments"] = segments
            players.append(player_copy)

        # Enrich with player database metadata when available (name, jersey, team hint).
        db_path = get_player_db_path()
        if db_path.exists():
            try:
                from src.identity import PlayerDatabase

                player_metadata: dict[int, dict[str, Any]] = {}
                with PlayerDatabase(db_path) as db:
                    for player in db.list_players():
                        player_metadata[int(player.player_id)] = {
                            "player_name": player.name,
                            "jersey_number": player.jersey_number,
                            "team_hint": player.team_hint,
                        }

                for player in players:
                    player_id_raw = player.get("player_id")
                    if player_id_raw is None:
                        continue
                    try:
                        player_id = int(player_id_raw)
                    except (TypeError, ValueError):
                        continue

                    metadata = player_metadata.get(player_id)
                    if metadata is None:
                        continue

                    if not player.get("player_name") and metadata.get("player_name"):
                        player["player_name"] = metadata["player_name"]
                    if player.get("jersey_number") is None and metadata.get("jersey_number") is not None:
                        player["jersey_number"] = metadata["jersey_number"]
                    if player.get("team_hint") is None and metadata.get("team_hint") is not None:
                        player["team_hint"] = metadata["team_hint"]

            except ImportError:
                # Identity module is optional for UI reads; return reels without enrichment.
                pass

        return {
            "schema_version": reels_data.get("schema_version", "1.0"),
            "video_id": reels_data.get("video_id"),
            "run_name": run_name,
            "players": players,
            "summary": reels_data.get("summary", {}),
            "count": len(players),
        }

    def get_identity_edits_path(run_path: Path) -> Path:
        """Path to identity edit audit log for a run."""
        return run_path / "identity_edits.jsonl"

    def append_identity_edit_record(run_path: Path, record: dict[str, Any]) -> None:
        """Append a structured identity edit record."""
        payload = {
            **record,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        edits_path = get_identity_edits_path(run_path)
        with open(edits_path, "a") as f:
            f.write(json.dumps(payload) + "\n")

    def load_identity_edit_records(run_path: Path) -> list[dict[str, Any]]:
        """Load identity edit records for a run."""
        edits_path = get_identity_edits_path(run_path)
        if not edits_path.exists():
            return []

        records: list[dict[str, Any]] = []
        with open(edits_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
        return records

    def find_last_undoable_identity_edit(run_path: Path) -> dict[str, Any] | None:
        """Return last edit operation that has not been undone."""
        records = load_identity_edit_records(run_path)
        undone_op_ids: set[str] = set()
        edit_ops: list[dict[str, Any]] = []

        for record in records:
            action = record.get("action")
            if action == "undo":
                target_op_id = record.get("target_op_id")
                if isinstance(target_op_id, str) and target_op_id:
                    undone_op_ids.add(target_op_id)
            elif action in {"assign", "bulk_assign"}:
                op_id = record.get("op_id")
                if isinstance(op_id, str) and op_id:
                    edit_ops.append(record)

        for record in reversed(edit_ops):
            op_id = str(record.get("op_id", ""))
            if op_id and op_id not in undone_op_ids:
                return record
        return None

    def find_undoable_identity_edit_by_op_id(
        run_path: Path, target_op_id: str
    ) -> dict[str, Any] | None:
        """Return a specific undoable identity edit by operation id."""
        records = load_identity_edit_records(run_path)
        undone_op_ids: set[str] = set()
        edit_records_by_op_id: dict[str, dict[str, Any]] = {}

        for record in records:
            action = record.get("action")
            if action == "undo":
                undone_target_op_id = record.get("target_op_id")
                if isinstance(undone_target_op_id, str) and undone_target_op_id:
                    undone_op_ids.add(undone_target_op_id)
            elif action in {"assign", "bulk_assign"}:
                op_id = record.get("op_id")
                if isinstance(op_id, str) and op_id:
                    edit_records_by_op_id[op_id] = record

        record = edit_records_by_op_id.get(target_op_id)
        if record is None or target_op_id in undone_op_ids:
            return None
        return record

    def summarize_identity_edit_record(
        record: dict[str, Any], undone_op_ids: set[str]
    ) -> dict[str, Any]:
        """Convert raw edit record into UI-friendly history payload."""
        action = str(record.get("action", "unknown"))
        op_id = record.get("op_id")
        op_id_value = str(op_id) if isinstance(op_id, str) else None

        track_ids_raw = record.get("track_ids")
        track_ids: list[int] = []
        if isinstance(track_ids_raw, list):
            for track_id in track_ids_raw:
                try:
                    track_ids.append(int(track_id))
                except (TypeError, ValueError):
                    continue

        summary_raw = record.get("summary")
        summary = summary_raw if isinstance(summary_raw, dict) else {}
        player_id_raw = record.get("player_id")
        player_id: int | None = None
        if player_id_raw is not None:
            try:
                player_id = int(player_id_raw)
            except (TypeError, ValueError):
                player_id = None

        target_op_id = record.get("target_op_id")
        target_op_id_value = str(target_op_id) if isinstance(target_op_id, str) else None

        return {
            "action": action,
            "op_id": op_id_value,
            "run_name": record.get("run_name"),
            "video_id": record.get("video_id"),
            "recorded_at": record.get("recorded_at"),
            "track_ids": track_ids,
            "track_count": len(track_ids),
            "player_id": player_id,
            "summary": summary,
            "method": record.get("method"),
            "target_op_id": target_op_id_value,
            "undoable": (
                action in {"assign", "bulk_assign"}
                and op_id_value is not None
                and op_id_value not in undone_op_ids
            ),
        }

    def undo_identity_edit_operation(
        run_name: str,
        run_path: Path,
        target_edit: dict[str, Any],
    ) -> dict[str, Any]:
        """Undo one assignment operation and write an undo audit record."""
        video_id = target_edit.get("video_id")
        if not isinstance(video_id, str) or not video_id:
            raise HTTPException(status_code=400, detail="Undo target is missing video_id")

        target_op_id = target_edit.get("op_id")
        if not isinstance(target_op_id, str) or not target_op_id:
            raise HTTPException(status_code=400, detail="Undo target is missing op_id")

        changes_raw = target_edit.get("changes", [])
        if not isinstance(changes_raw, list) or not changes_raw:
            raise HTTPException(status_code=400, detail="Undo target has no reversible changes")

        db_path = get_player_db_path()
        if not db_path.exists():
            raise HTTPException(status_code=404, detail="Player database not found")

        reverted_count = 0
        deleted_count = 0
        failed_track_ids: list[int] = []

        try:
            from src.identity import PlayerDatabase

            with PlayerDatabase(db_path) as db:
                for change in reversed(changes_raw):
                    if not isinstance(change, dict):
                        continue

                    track_id_raw = change.get("track_id")
                    try:
                        track_id = int(track_id_raw)
                    except (TypeError, ValueError):
                        continue

                    previous = change.get("previous")
                    previous_state = previous if isinstance(previous, dict) else {}
                    existed_before = bool(previous_state.get("exists"))

                    try:
                        if existed_before:
                            previous_player_id_raw = previous_state.get("player_id")
                            previous_player_id: int | None = None
                            if previous_player_id_raw is not None:
                                try:
                                    previous_player_id = int(previous_player_id_raw)
                                except (TypeError, ValueError):
                                    previous_player_id = None

                            previous_confidence_raw = previous_state.get("match_confidence")
                            previous_confidence: float | None = None
                            if previous_confidence_raw is not None:
                                try:
                                    previous_confidence = float(previous_confidence_raw)
                                except (TypeError, ValueError):
                                    previous_confidence = None

                            previous_method_raw = previous_state.get("match_method")
                            previous_method = (
                                str(previous_method_raw)
                                if previous_method_raw in {"manual", "auto", "suggested"}
                                else None
                            )

                            previous_frame_start_raw = previous_state.get("frame_start")
                            previous_frame_end_raw = previous_state.get("frame_end")
                            previous_frame_start = (
                                int(previous_frame_start_raw)
                                if previous_frame_start_raw is not None
                                else None
                            )
                            previous_frame_end = (
                                int(previous_frame_end_raw)
                                if previous_frame_end_raw is not None
                                else None
                            )

                            previous_run_name_raw = previous_state.get("run_name")
                            previous_run_name = (
                                str(previous_run_name_raw)
                                if isinstance(previous_run_name_raw, str) and previous_run_name_raw
                                else run_name
                            )

                            db.create_appearance(
                                video_id=video_id,
                                run_name=previous_run_name,
                                track_id=track_id,
                                player_id=previous_player_id,
                                match_confidence=previous_confidence,
                                match_method=previous_method,  # type: ignore[arg-type]
                                frame_start=previous_frame_start,
                                frame_end=previous_frame_end,
                            )
                            reverted_count += 1
                        else:
                            if db.delete_appearance(video_id=video_id, track_id=track_id):
                                deleted_count += 1
                    except Exception:
                        failed_track_ids.append(track_id)

        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="Identity module not available",
            )

        undo_track_ids: list[int] = []
        for change in changes_raw:
            if not isinstance(change, dict):
                continue
            track_id_raw = change.get("track_id")
            try:
                undo_track_ids.append(int(track_id_raw))
            except (TypeError, ValueError):
                continue

        undo_operation_id = f"edit_{int(datetime.now(timezone.utc).timestamp() * 1000)}_{uuid4().hex[:8]}"
        append_identity_edit_record(
            run_path,
            {
                "action": "undo",
                "op_id": undo_operation_id,
                "target_op_id": target_op_id,
                "target_action": target_edit.get("action"),
                "run_name": run_name,
                "video_id": video_id,
                "track_ids": undo_track_ids,
                "summary": {
                    "reverted_count": reverted_count,
                    "deleted_count": deleted_count,
                    "failed_count": len(failed_track_ids),
                },
            },
        )

        return {
            "success": True,
            "run_name": run_name,
            "video_id": video_id,
            "target_operation_id": target_op_id,
            "undo_operation_id": undo_operation_id,
            "reverted_count": reverted_count,
            "deleted_count": deleted_count,
            "failed_count": len(failed_track_ids),
            "failed_track_ids": failed_track_ids,
        }

    def apply_assignment_edits(
        run_name: str,
        track_ids: list[int],
        player_id: int | None,
        confidence: float,
        method: str,
        action_name: str,
    ) -> dict[str, Any]:
        """Apply assignment/unassignment updates and persist audit record."""
        normalized_track_ids: list[int] = []
        for track_id in track_ids:
            try:
                normalized_track_ids.append(int(track_id))
            except (TypeError, ValueError):
                continue
        normalized_track_ids = sorted(set(normalized_track_ids))
        if not normalized_track_ids:
            raise HTTPException(
                status_code=400,
                detail="track_ids must include at least one valid track id",
            )

        if method not in {"manual", "auto", "suggested"}:
            raise HTTPException(
                status_code=400,
                detail="method must be manual, auto, or suggested",
            )

        run_path = runs_dir / run_name
        if not run_path.exists():
            raise HTTPException(status_code=404, detail="Run not found")

        identity_data = load_identity_review_for_run(run_name)
        video_id = identity_data.get("video_id")
        if not video_id:
            raise HTTPException(status_code=404, detail="Run video_id not found")

        assignment_lookup: dict[int, dict[str, Any]] = {}
        for row in identity_data.get("assignments", []):
            track_id = row.get("track_id")
            if track_id is None:
                continue
            try:
                assignment_lookup[int(track_id)] = row
            except (TypeError, ValueError):
                continue

        db_path = get_player_db_path()
        if not db_path.exists():
            raise HTTPException(status_code=404, detail="Player database not found")

        updated_count = 0
        created_count = 0
        missing_track_ids: list[int] = []
        failed_track_ids: list[int] = []
        warnings: list[str] = []
        changes: list[dict[str, Any]] = []

        try:
            from src.identity import PlayerDatabase

            with PlayerDatabase(db_path) as db:
                if player_id is not None:
                    player = db.get_player(player_id)
                    if player is None:
                        raise HTTPException(status_code=404, detail="Target player not found")

                for track_id in normalized_track_ids:
                    existing = db.get_appearance(video_id, track_id)
                    previous = {
                        "exists": existing is not None,
                        "player_id": existing.player_id if existing else None,
                        "match_confidence": existing.match_confidence if existing else None,
                        "match_method": existing.match_method if existing else None,
                        "frame_start": existing.frame_start if existing else None,
                        "frame_end": existing.frame_end if existing else None,
                        "run_name": existing.run_name if existing else run_name,
                    }

                    try:
                        if existing is not None:
                            db.assign_appearance_to_player(
                                video_id=video_id,
                                track_id=track_id,
                                player_id=player_id,
                                confidence=confidence,
                                method=method,  # type: ignore[arg-type]
                            )
                            updated_count += 1
                        else:
                            fallback = assignment_lookup.get(track_id)
                            if fallback is None:
                                missing_track_ids.append(track_id)
                                continue

                            frame_start = fallback.get("frame_start")
                            frame_end = fallback.get("frame_end")
                            db.create_appearance(
                                video_id=video_id,
                                run_name=run_name,
                                track_id=track_id,
                                player_id=player_id,
                                match_confidence=confidence,
                                match_method=method,  # type: ignore[arg-type]
                                frame_start=int(frame_start) if frame_start is not None else None,
                                frame_end=int(frame_end) if frame_end is not None else None,
                            )
                            created_count += 1

                        changes.append(
                            {
                                "track_id": track_id,
                                "previous": previous,
                                "next": {
                                    "exists": True,
                                    "player_id": player_id,
                                    "match_confidence": confidence,
                                    "match_method": method,
                                    "run_name": run_name,
                                },
                            }
                        )
                    except Exception:
                        failed_track_ids.append(track_id)

        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="Identity module not available",
            )

        if missing_track_ids:
            preview = ",".join(str(track_id) for track_id in missing_track_ids[:5])
            warnings.append(f"Tracks not found in run assignments: {preview}")
        if failed_track_ids:
            preview = ",".join(str(track_id) for track_id in failed_track_ids[:5])
            warnings.append(f"Failed to assign tracks: {preview}")

        operation_id = None
        if changes:
            operation_id = f"edit_{int(datetime.now(timezone.utc).timestamp() * 1000)}_{uuid4().hex[:8]}"
            append_identity_edit_record(
                run_path,
                {
                    "action": action_name,
                    "op_id": operation_id,
                    "run_name": run_name,
                    "video_id": video_id,
                    "track_ids": normalized_track_ids,
                    "player_id": player_id,
                    "confidence": confidence,
                    "method": method,
                    "changes": changes,
                    "summary": {
                        "updated_count": updated_count,
                        "created_count": created_count,
                        "missing_count": len(missing_track_ids),
                        "failed_count": len(failed_track_ids),
                    },
                },
            )

        return {
            "success": True,
            "run_name": run_name,
            "video_id": video_id,
            "player_id": player_id,
            "requested_count": len(normalized_track_ids),
            "updated_count": updated_count,
            "created_count": created_count,
            "missing_count": len(missing_track_ids),
            "failed_count": len(failed_track_ids),
            "missing_track_ids": missing_track_ids,
            "failed_track_ids": failed_track_ids,
            "warnings": warnings,
            "operation_id": operation_id,
        }

    def compute_player_reels_for_run(
        run_name: str,
        preserve_existing_clips: bool = True,
    ) -> dict[str, Any]:
        """Compute per-player reels from highlights + current identity assignments."""
        import pandas as pd
        from src.config.schemas import PlayerReelsConfig
        from src.events.player_reels import build_player_reels

        run_path = runs_dir / run_name
        if not run_path.exists():
            raise HTTPException(status_code=404, detail="Run not found")

        highlights_path = run_path / "highlights.json"
        if not highlights_path.exists():
            raise HTTPException(status_code=404, detail="highlights.json not found")

        tracks_path = run_path / "tracks.parquet"
        if not tracks_path.exists():
            raise HTTPException(status_code=404, detail="tracks.parquet not found")

        with open(highlights_path) as f:
            highlights_data = json.load(f)
        segments = list(highlights_data.get("segments", []))

        identity_data = load_identity_review_for_run(run_name)
        assignment_rows = list(identity_data.get("assignments", []))
        assignments = []
        for row in assignment_rows:
            track_id_value = row.get("track_id")
            if track_id_value is None:
                continue
            try:
                track_id = int(track_id_value)
            except (TypeError, ValueError):
                continue

            player_id_value = row.get("player_id")
            player_id: int | None = None
            if player_id_value is not None:
                try:
                    player_id = int(player_id_value)
                except (TypeError, ValueError):
                    player_id = None

            assignments.append(
                {
                    "track_id": track_id,
                    "player_id": player_id,
                    "player_name": row.get("player_name"),
                    "match_method": row.get("match_method") or "manual",
                    "confidence": row.get("match_confidence", 1.0),
                }
            )

        video_metadata_path = run_path / "video_metadata.json"
        fps = 30.0
        if video_metadata_path.exists():
            with open(video_metadata_path) as f:
                metadata = json.load(f)
            fps = float(metadata.get("fps", 30.0))

        reel_cfg = PlayerReelsConfig()
        manifest_path = run_path / "player_highlights_manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            manifest_cfg = manifest.get("config")
            if isinstance(manifest_cfg, dict):
                try:
                    reel_cfg = PlayerReelsConfig(**manifest_cfg)
                except Exception:
                    pass

        tracks_df = pd.read_parquet(tracks_path)
        tracks = tracks_df.to_dict(orient="records")

        player_reels, summary = build_player_reels(
            segments=segments,
            tracks=tracks,
            assignments=assignments,
            fps=fps,
            max_segments_per_player=reel_cfg.max_segments_per_player,
            min_presence_seconds=reel_cfg.min_presence_seconds,
            min_player_segment_score=reel_cfg.min_player_segment_score,
            min_assignment_confidence=reel_cfg.min_assignment_confidence,
            include_suggested=reel_cfg.include_suggested_assignments,
        )

        preserved_clips = 0
        if preserve_existing_clips:
            existing_reels_path = run_path / "player_highlights.json"
            existing_clip_map: dict[tuple[int, str], str] = {}
            if existing_reels_path.exists():
                with open(existing_reels_path) as f:
                    existing_data = json.load(f)
                for player in existing_data.get("players", []):
                    player_id_raw = player.get("player_id")
                    if player_id_raw is None:
                        continue
                    try:
                        player_id = int(player_id_raw)
                    except (TypeError, ValueError):
                        continue

                    for segment in player.get("segments", []):
                        segment_id = str(segment.get("segment_id", ""))
                        if not segment_id:
                            continue
                        clip_path_value = segment.get("clip_path")
                        clip_path = resolve_run_artifact_path(run_path, clip_path_value)
                        if clip_path is not None:
                            existing_clip_map[(player_id, segment_id)] = str(clip_path)

            for player in player_reels:
                player_id = int(player.get("player_id", -1))
                for segment in player.get("segments", []):
                    key = (player_id, str(segment.get("segment_id", "")))
                    clip_path = existing_clip_map.get(key)
                    if clip_path:
                        segment["clip_path"] = clip_path
                        preserved_clips += 1

        reels_data = {
            "schema_version": "1.0",
            "video_id": identity_data.get("video_id") or run_name,
            "players": player_reels,
            "summary": {
                **summary,
                "recomputed_at": datetime.now(timezone.utc).isoformat(),
                "preserved_clips": preserved_clips,
            },
        }

        return {
            "run_path": run_path,
            "reels_data": reels_data,
            "reel_cfg": reel_cfg,
            "segments_count": len(segments),
            "assignments_count": len(assignments),
            "tracks_count": len(tracks),
            "fps": fps,
        }

    def persist_player_reels_artifacts(
        compute_result: dict[str, Any],
        source: str = "ui_recompute",
    ) -> dict[str, Any]:
        """Persist recomputed player reel artifacts and return output paths/summary."""
        import pandas as pd

        run_path = compute_result["run_path"]
        reels_data = compute_result["reels_data"]
        reel_cfg = compute_result["reel_cfg"]
        segments_count = int(compute_result["segments_count"])
        assignments_count = int(compute_result["assignments_count"])
        tracks_count = int(compute_result["tracks_count"])
        fps = float(compute_result["fps"])
        player_reels = list(reels_data.get("players", []))

        reels_json_path = run_path / "player_highlights.json"
        reels_csv_path = run_path / "player_highlights.csv"
        reels_manifest_path = run_path / "player_highlights_manifest.json"

        with open(reels_json_path, "w") as f:
            json.dump(reels_data, f, indent=2)

        flat_rows = []
        for player in player_reels:
            for segment in player.get("segments", []):
                flat_rows.append(
                    {
                        "player_id": player.get("player_id"),
                        "player_name": player.get("player_name"),
                        **segment,
                    }
                )
        pd.DataFrame(flat_rows).to_csv(reels_csv_path, index=False)

        manifest_payload = {
            "schema_version": "1.0",
            "config": reel_cfg.model_dump(),
            "inputs": {
                "highlights_segments": segments_count,
                "assignments": assignments_count,
                "tracks": tracks_count,
                "fps": fps,
                "source": source,
            },
            "outputs": {
                "player_highlights_json": str(reels_json_path),
                "player_highlights_csv": str(reels_csv_path),
            },
            "summary": reels_data["summary"],
        }
        with open(reels_manifest_path, "w") as f:
            json.dump(manifest_payload, f, indent=2)

        return {
            "summary": reels_data["summary"],
            "inputs": manifest_payload["inputs"],
            "artifacts": {
                "player_highlights_json": str(reels_json_path),
                "player_highlights_csv": str(reels_csv_path),
                "player_highlights_manifest": str(reels_manifest_path),
            },
        }

    def build_recompute_diff(run_path: Path, preview_reels_data: dict[str, Any]) -> dict[str, Any]:
        """Compute diff between current persisted reels and preview reels."""
        current_path = run_path / "player_highlights.json"
        current_data: dict[str, Any] = {}
        if current_path.exists():
            with open(current_path) as f:
                current_data = json.load(f)

        current_players = list(current_data.get("players", []))
        preview_players = list(preview_reels_data.get("players", []))

        def to_player_map(players: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
            result: dict[int, dict[str, Any]] = {}
            for player in players:
                player_id = player.get("player_id")
                if player_id is None:
                    continue
                try:
                    result[int(player_id)] = player
                except (TypeError, ValueError):
                    continue
            return result

        def to_segment_keys(players: list[dict[str, Any]]) -> set[tuple[int, str]]:
            keys: set[tuple[int, str]] = set()
            for player in players:
                player_id = player.get("player_id")
                if player_id is None:
                    continue
                try:
                    pid = int(player_id)
                except (TypeError, ValueError):
                    continue

                for segment in player.get("segments", []):
                    segment_id = str(segment.get("segment_id", ""))
                    if segment_id:
                        keys.add((pid, segment_id))
            return keys

        current_summary = current_data.get("summary", {})
        preview_summary = preview_reels_data.get("summary", {})

        current_player_map = to_player_map(current_players)
        preview_player_map = to_player_map(preview_players)
        current_segment_keys = to_segment_keys(current_players)
        preview_segment_keys = to_segment_keys(preview_players)

        gained = sorted(preview_segment_keys - current_segment_keys)
        lost = sorted(current_segment_keys - preview_segment_keys)

        all_player_ids = sorted(set(current_player_map.keys()) | set(preview_player_map.keys()))
        player_changes: list[dict[str, Any]] = []
        for player_id in all_player_ids:
            current_player = current_player_map.get(player_id, {})
            preview_player = preview_player_map.get(player_id, {})

            current_segments = list(current_player.get("segments", []))
            preview_segments = list(preview_player.get("segments", []))
            current_count = len(current_segments)
            preview_count = len(preview_segments)

            def best_score(rows: list[dict[str, Any]]) -> float:
                return max((float(row.get("player_segment_score", 0.0)) for row in rows), default=0.0)

            current_best = best_score(current_segments)
            preview_best = best_score(preview_segments)

            if current_count != preview_count or abs(current_best - preview_best) > 1e-9:
                player_name = (
                    preview_player.get("player_name")
                    or current_player.get("player_name")
                    or None
                )
                player_changes.append(
                    {
                        "player_id": player_id,
                        "player_name": player_name,
                        "current_segment_count": current_count,
                        "preview_segment_count": preview_count,
                        "delta_segment_count": preview_count - current_count,
                        "current_best_score": current_best,
                        "preview_best_score": preview_best,
                    }
                )

        player_changes.sort(
            key=lambda row: (
                abs(int(row["delta_segment_count"])),
                abs(float(row["preview_best_score"]) - float(row["current_best_score"])),
            ),
            reverse=True,
        )

        return {
            "current": {
                "players_with_reels": current_summary.get("players_with_reels", len(current_players)),
                "player_segments_total": current_summary.get(
                    "player_segments_total",
                    sum(len(player.get("segments", [])) for player in current_players),
                ),
            },
            "preview": {
                "players_with_reels": preview_summary.get("players_with_reels", len(preview_players)),
                "player_segments_total": preview_summary.get(
                    "player_segments_total",
                    sum(len(player.get("segments", [])) for player in preview_players),
                ),
            },
            "delta": {
                "players_with_reels": (
                    preview_summary.get("players_with_reels", len(preview_players))
                    - current_summary.get("players_with_reels", len(current_players))
                ),
                "player_segments_total": (
                    preview_summary.get("player_segments_total", len(preview_segment_keys))
                    - current_summary.get("player_segments_total", len(current_segment_keys))
                ),
                "gained_segments_total": len(gained),
                "lost_segments_total": len(lost),
            },
            "gained_segments_sample": [
                {"player_id": player_id, "segment_id": segment_id}
                for player_id, segment_id in gained[:20]
            ],
            "lost_segments_sample": [
                {"player_id": player_id, "segment_id": segment_id}
                for player_id, segment_id in lost[:20]
            ],
            "player_changes": player_changes[:30],
        }

    def _coerce_int(value: Any) -> int | None:
        """Best-effort conversion of an arbitrary value to int."""
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _coerce_float(value: Any, default: float = 0.0) -> float:
        """Best-effort conversion of an arbitrary value to float."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def build_identity_suggestions_artifact(
        run_name: str,
        refresh: bool = False,
    ) -> dict[str, Any]:
        """Build or load per-run identity suggestions artifact."""
        run_path = runs_dir / run_name
        if not run_path.exists():
            raise HTTPException(status_code=404, detail="Run not found")

        suggestions_path = run_path / "profile_match_suggestions.json"
        if suggestions_path.exists() and not refresh:
            with open(suggestions_path) as f:
                existing_data = json.load(f)
            return existing_data

        assignments_path = run_path / "player_assignments.json"
        if not assignments_path.exists():
            raise HTTPException(status_code=404, detail="player_assignments.json not found")

        with open(assignments_path) as f:
            assignments_data = json.load(f)

        assignment_rows = assignments_data.get("assignments", [])
        if not isinstance(assignment_rows, list):
            assignment_rows = []

        suggestions: list[dict[str, Any]] = []
        pending_count = 0
        with_profile_evidence = 0

        for row in assignment_rows:
            if not isinstance(row, dict):
                continue

            track_id = _coerce_int(row.get("track_id"))
            if track_id is None:
                continue

            recommended_player_id = _coerce_int(row.get("player_id"))
            recommended_confidence = _coerce_float(row.get("confidence"), 0.0)
            recommended_method = str(row.get("match_method") or "unknown")

            fusion = row.get("fusion")
            fusion_data = fusion if isinstance(fusion, dict) else {}
            strategy = str(fusion_data.get("strategy", "body_only"))

            body_match_raw = fusion_data.get("body_match")
            body_match = body_match_raw if isinstance(body_match_raw, dict) else {}
            body_player_id = _coerce_int(body_match.get("player_id"))
            body_confidence = _coerce_float(body_match.get("confidence"), 0.0)

            profile_match_raw = fusion_data.get("profile_match")
            profile_match = profile_match_raw if isinstance(profile_match_raw, dict) else {}
            profile_player_id = _coerce_int(profile_match.get("player_id"))
            profile_confidence = _coerce_float(profile_match.get("confidence"), 0.0)
            profile_id = profile_match.get("profile_id")
            if profile_player_id is not None:
                with_profile_evidence += 1

            candidate_map: dict[int, dict[str, Any]] = {}
            if body_player_id is not None:
                candidate_map[body_player_id] = {
                    "player_id": body_player_id,
                    "reason_breakdown": {
                        "body_reid": body_confidence,
                        "profile_match": 0.0,
                        "agreement_bonus": 0.0,
                    },
                }

            if profile_player_id is not None:
                existing = candidate_map.get(profile_player_id)
                if existing is None:
                    candidate_map[profile_player_id] = {
                        "player_id": profile_player_id,
                        "reason_breakdown": {
                            "body_reid": 0.0,
                            "profile_match": profile_confidence,
                            "agreement_bonus": 0.0,
                        },
                    }
                else:
                    existing["reason_breakdown"]["profile_match"] = profile_confidence

            # Fallback when fusion details are unavailable.
            if not candidate_map and recommended_player_id is not None:
                candidate_map[recommended_player_id] = {
                    "player_id": recommended_player_id,
                    "reason_breakdown": {
                        "body_reid": recommended_confidence,
                        "profile_match": 0.0,
                        "agreement_bonus": 0.0,
                    },
                }

            candidate_rows: list[dict[str, Any]] = []
            for candidate in candidate_map.values():
                candidate_player_id = int(candidate["player_id"])
                breakdown = candidate["reason_breakdown"]
                base_score = max(
                    _coerce_float(breakdown.get("body_reid"), 0.0),
                    _coerce_float(breakdown.get("profile_match"), 0.0),
                )
                agreement_bonus = 0.0
                if (
                    candidate_player_id == body_player_id
                    and candidate_player_id == profile_player_id
                    and strategy == "agreement_boost"
                ):
                    agreement_bonus = max(0.0, recommended_confidence - base_score)
                    breakdown["agreement_bonus"] = agreement_bonus
                score = min(1.0, base_score + agreement_bonus)
                sources = [
                    source
                    for source, key in (
                        ("body_reid", "body_reid"),
                        ("profile_match", "profile_match"),
                    )
                    if _coerce_float(breakdown.get(key), 0.0) > 0.0
                ]
                candidate_rows.append(
                    {
                        "player_id": candidate_player_id,
                        "score": score,
                        "reason_breakdown": {
                            "body_reid": _coerce_float(breakdown.get("body_reid"), 0.0),
                            "profile_match": _coerce_float(breakdown.get("profile_match"), 0.0),
                            "agreement_bonus": _coerce_float(breakdown.get("agreement_bonus"), 0.0),
                        },
                        "sources": sources,
                    }
                )

            candidate_rows.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
            candidate_rows = candidate_rows[:5]
            for rank, candidate in enumerate(candidate_rows, start=1):
                candidate["rank"] = rank

            needs_review = recommended_method == "suggested"
            if needs_review:
                pending_count += 1

            if recommended_method == "suggested":
                status = "pending"
            elif recommended_method == "auto":
                status = "auto_assigned"
            elif recommended_method == "skipped":
                status = "skipped"
            else:
                status = "accepted"

            suggestions.append(
                {
                    "track_id": track_id,
                    "recommended": {
                        "player_id": recommended_player_id,
                        "confidence": recommended_confidence,
                        "method": recommended_method,
                    },
                    "needs_review": needs_review,
                    "status": status,
                    "fusion_strategy": strategy,
                    "profile_id": str(profile_id) if profile_id is not None else None,
                    "candidates": candidate_rows,
                }
            )

        suggestions.sort(
            key=lambda row: (
                not bool(row.get("needs_review")),
                -_coerce_float((row.get("recommended") or {}).get("confidence"), 0.0),
                int(row.get("track_id", 0)),
            )
        )

        payload = {
            "schema_version": "1.0",
            "run_name": run_name,
            "video_id": assignments_data.get("video_id"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_artifact": "player_assignments.json",
            "suggestions": suggestions,
            "summary": {
                "total_tracks": len(suggestions),
                "pending": pending_count,
                "with_profile_evidence": with_profile_evidence,
            },
        }

        with open(suggestions_path, "w") as f:
            json.dump(payload, f, indent=2)

        return payload

    def enrich_identity_suggestions_with_players(
        run_name: str,
        suggestions_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Attach player names/jersey/team metadata onto suggestion payload."""
        identity_data = load_identity_review_for_run(run_name)
        players = identity_data.get("players", [])
        player_lookup: dict[int, dict[str, Any]] = {}
        for row in players:
            player_id = _coerce_int(row.get("player_id"))
            if player_id is None:
                continue
            player_lookup[player_id] = row

        suggestions = suggestions_data.get("suggestions", [])
        if not isinstance(suggestions, list):
            suggestions = []

        enriched_rows: list[dict[str, Any]] = []
        for row in suggestions:
            if not isinstance(row, dict):
                continue
            row_copy = dict(row)
            recommended = row_copy.get("recommended")
            if isinstance(recommended, dict):
                recommended_copy = dict(recommended)
                recommended_player_id = _coerce_int(recommended_copy.get("player_id"))
                if recommended_player_id is not None:
                    player = player_lookup.get(recommended_player_id)
                    if player is not None:
                        recommended_copy["player_name"] = player.get("name")
                        recommended_copy["jersey_number"] = player.get("jersey_number")
                        recommended_copy["team_hint"] = player.get("team_hint")
                row_copy["recommended"] = recommended_copy

            candidates = row_copy.get("candidates", [])
            enriched_candidates: list[dict[str, Any]] = []
            if isinstance(candidates, list):
                for candidate in candidates:
                    if not isinstance(candidate, dict):
                        continue
                    candidate_copy = dict(candidate)
                    candidate_player_id = _coerce_int(candidate_copy.get("player_id"))
                    if candidate_player_id is not None:
                        player = player_lookup.get(candidate_player_id)
                        if player is not None:
                            candidate_copy["player_name"] = player.get("name")
                            candidate_copy["jersey_number"] = player.get("jersey_number")
                            candidate_copy["team_hint"] = player.get("team_hint")
                    enriched_candidates.append(candidate_copy)
            row_copy["candidates"] = enriched_candidates

            enriched_rows.append(row_copy)

        summary = suggestions_data.get("summary")
        summary_payload = summary if isinstance(summary, dict) else {}
        return {
            **suggestions_data,
            "run_name": run_name,
            "suggestions": enriched_rows,
            "summary": summary_payload,
            "count": len(enriched_rows),
        }

    def _apply_identity_suggestions(
        run_name: str,
        body: ApplyIdentitySuggestionsBody,
    ) -> dict[str, Any]:
        """Apply recommended suggestions to selected tracks and persist status."""
        run_path = runs_dir / run_name
        if not run_path.exists():
            raise HTTPException(status_code=404, detail="Run not found")
        min_confidence = max(0.0, min(1.0, _coerce_float(body.min_confidence, 0.7)))

        suggestions_data = build_identity_suggestions_artifact(run_name, refresh=False)
        suggestions = suggestions_data.get("suggestions", [])
        if not isinstance(suggestions, list):
            raise HTTPException(status_code=400, detail="Invalid suggestions payload")

        selected_track_ids: set[int] | None = None
        if body.track_ids is not None:
            selected_track_ids = set()
            for track_id in body.track_ids:
                normalized_track_id = _coerce_int(track_id)
                if normalized_track_id is not None:
                    selected_track_ids.add(normalized_track_id)
            if not selected_track_ids:
                raise HTTPException(status_code=400, detail="track_ids must include at least one valid id")

        applied_count = 0
        skipped_count = 0
        failed_count = 0
        skipped_track_ids: list[int] = []
        failed_track_ids: list[int] = []
        operation_ids: list[str] = []

        for row in suggestions:
            if not isinstance(row, dict):
                continue

            track_id = _coerce_int(row.get("track_id"))
            if track_id is None:
                continue
            if selected_track_ids is not None and track_id not in selected_track_ids:
                continue

            if body.suggested_only and not bool(row.get("needs_review")):
                skipped_count += 1
                skipped_track_ids.append(track_id)
                continue

            status = str(row.get("status", ""))
            if status == "applied":
                skipped_count += 1
                skipped_track_ids.append(track_id)
                continue

            recommended_raw = row.get("recommended")
            recommended = recommended_raw if isinstance(recommended_raw, dict) else {}
            player_id = _coerce_int(recommended.get("player_id"))
            confidence = _coerce_float(recommended.get("confidence"), 0.0)
            method_raw = str(recommended.get("method", "suggested"))
            method = method_raw if method_raw in {"manual", "auto", "suggested"} else "suggested"

            if player_id is None or confidence < min_confidence:
                skipped_count += 1
                skipped_track_ids.append(track_id)
                continue

            try:
                result = apply_assignment_edits(
                    run_name=run_name,
                    track_ids=[track_id],
                    player_id=player_id,
                    confidence=confidence,
                    method=method,
                    action_name="assign",
                )
                if result.get("failed_count", 0) > 0:
                    failed_count += 1
                    failed_track_ids.append(track_id)
                    continue

                applied_count += 1
                operation_id = result.get("operation_id")
                if isinstance(operation_id, str) and operation_id:
                    operation_ids.append(operation_id)
                    row["applied_operation_id"] = operation_id
                row["status"] = "applied"
                row["applied_at"] = datetime.now(timezone.utc).isoformat()
                row["applied_method"] = method
                row["applied_confidence"] = confidence
            except HTTPException:
                failed_count += 1
                failed_track_ids.append(track_id)

        suggestions_data["suggestions"] = suggestions
        suggestions_data["summary"] = {
            **(suggestions_data.get("summary") if isinstance(suggestions_data.get("summary"), dict) else {}),
            "applied_count": sum(
                1 for row in suggestions if isinstance(row, dict) and row.get("status") == "applied"
            ),
            "pending": sum(
                1
                for row in suggestions
                if isinstance(row, dict)
                and bool(row.get("needs_review"))
                and row.get("status") != "applied"
            ),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        suggestions_data["updated_at"] = datetime.now(timezone.utc).isoformat()

        suggestions_path = run_path / "profile_match_suggestions.json"
        with open(suggestions_path, "w") as f:
            json.dump(suggestions_data, f, indent=2)

        return {
            "success": True,
            "run_name": run_name,
            "applied_count": applied_count,
            "skipped_count": skipped_count,
            "failed_count": failed_count,
            "applied_operation_ids": operation_ids,
            "skipped_track_ids": skipped_track_ids,
            "failed_track_ids": failed_track_ids,
        }

    def _recompute_player_reels_payload(
        run_name: str,
        preserve_existing_clips: bool = True,
        source: str = "ui_recompute",
    ) -> dict[str, Any]:
        """Rebuild and persist per-player reels and return summary/artifact metadata."""
        compute_result = compute_player_reels_for_run(
            run_name=run_name,
            preserve_existing_clips=preserve_existing_clips,
        )
        persisted = persist_player_reels_artifacts(compute_result, source=source)
        return {
            "success": True,
            "run_name": run_name,
            "summary": persisted["summary"],
            "inputs": persisted["inputs"],
            "artifacts": persisted["artifacts"],
        }

    def _compute_identity_assignments_signature(run_name: str) -> str:
        """Hash current assignment state to detect stale previews."""
        identity_data = load_identity_review_for_run(run_name)
        assignment_rows = identity_data.get("assignments", [])
        if not isinstance(assignment_rows, list):
            assignment_rows = []

        normalized_rows: list[dict[str, Any]] = []
        for row in assignment_rows:
            if not isinstance(row, dict):
                continue
            track_id = _coerce_int(row.get("track_id"))
            if track_id is None:
                continue
            normalized_rows.append(
                {
                    "track_id": track_id,
                    "player_id": _coerce_int(row.get("player_id")),
                    "match_method": str(row.get("match_method") or ""),
                    "match_confidence": round(_coerce_float(row.get("match_confidence"), 0.0), 6),
                    "frame_start": _coerce_int(row.get("frame_start")),
                    "frame_end": _coerce_int(row.get("frame_end")),
                }
            )

        normalized_rows.sort(key=lambda item: int(item["track_id"]))
        signature_payload = {
            "video_id": identity_data.get("video_id"),
            "assignments": normalized_rows,
        }
        encoded = json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _store_player_reels_preview(
        run_name: str,
        compute_result: dict[str, Any],
        preserve_existing_clips: bool,
        source: str,
    ) -> dict[str, Any]:
        """Persist a preview snapshot so it can be explicitly approved later."""
        run_path = compute_result["run_path"]
        preview_id = f"preview_{int(datetime.now(timezone.utc).timestamp() * 1000)}_{uuid4().hex[:8]}"
        reels_data = compute_result["reels_data"]
        reel_cfg = compute_result["reel_cfg"]
        reel_cfg_payload = reel_cfg.model_dump() if hasattr(reel_cfg, "model_dump") else {}
        inputs_payload = {
            "highlights_segments": int(compute_result["segments_count"]),
            "assignments": int(compute_result["assignments_count"]),
            "tracks": int(compute_result["tracks_count"]),
            "fps": float(compute_result["fps"]),
            "preserve_existing_clips": bool(preserve_existing_clips),
        }
        preview_artifact = {
            "schema_version": "1.0",
            "run_name": run_name,
            "preview_id": preview_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "guard": {
                "assignments_signature": _compute_identity_assignments_signature(run_name),
            },
            "inputs": inputs_payload,
            "reel_cfg": reel_cfg_payload,
            "reels_data": reels_data,
        }

        preview_path = run_path / "player_reels_preview.json"
        with open(preview_path, "w") as f:
            json.dump(preview_artifact, f, indent=2)

        return {
            "success": True,
            "run_name": run_name,
            "preview_id": preview_id,
            "summary": reels_data.get("summary", {}),
            "diff": build_recompute_diff(run_path, reels_data),
            "inputs": inputs_payload,
        }

    def _load_player_reels_preview(run_name: str) -> dict[str, Any]:
        """Load stored preview snapshot."""
        run_path = runs_dir / run_name
        if not run_path.exists():
            raise HTTPException(status_code=404, detail="Run not found")
        preview_path = run_path / "player_reels_preview.json"
        if not preview_path.exists():
            raise HTTPException(status_code=404, detail="No saved preview found. Generate a preview first.")
        with open(preview_path) as f:
            preview_data = json.load(f)
        if not isinstance(preview_data, dict):
            raise HTTPException(status_code=400, detail="Invalid preview payload")
        return preview_data

    def _approve_player_reels_preview(run_name: str, preview_id: str | None = None) -> dict[str, Any]:
        """Persist the last previewed reels if assignment state has not changed."""
        from src.config.schemas import PlayerReelsConfig

        preview_data = _load_player_reels_preview(run_name)
        stored_preview_id = str(preview_data.get("preview_id") or "")
        if preview_id and stored_preview_id and preview_id != stored_preview_id:
            raise HTTPException(
                status_code=409,
                detail="Preview id mismatch. Refresh preview before approving.",
            )

        guard = preview_data.get("guard") if isinstance(preview_data.get("guard"), dict) else {}
        expected_signature = str(guard.get("assignments_signature") or "")
        current_signature = _compute_identity_assignments_signature(run_name)
        if expected_signature and expected_signature != current_signature:
            raise HTTPException(
                status_code=409,
                detail="Preview is stale because identity assignments changed. Refresh preview.",
            )

        run_path = runs_dir / run_name
        reel_cfg_raw = preview_data.get("reel_cfg")
        reel_cfg_payload = reel_cfg_raw if isinstance(reel_cfg_raw, dict) else {}
        try:
            reel_cfg = PlayerReelsConfig(**reel_cfg_payload)
        except Exception:
            reel_cfg = PlayerReelsConfig()

        inputs = preview_data.get("inputs") if isinstance(preview_data.get("inputs"), dict) else {}
        reels_data_raw = preview_data.get("reels_data")
        reels_data = reels_data_raw if isinstance(reels_data_raw, dict) else {}
        compute_result = {
            "run_path": run_path,
            "reels_data": reels_data,
            "reel_cfg": reel_cfg,
            "segments_count": _coerce_int(inputs.get("highlights_segments")) or 0,
            "assignments_count": _coerce_int(inputs.get("assignments")) or 0,
            "tracks_count": _coerce_int(inputs.get("tracks")) or 0,
            "fps": _coerce_float(inputs.get("fps"), 30.0),
        }
        persisted = persist_player_reels_artifacts(compute_result, source="ui_approve_preview")
        return {
            "success": True,
            "run_name": run_name,
            "preview_id": stored_preview_id or preview_id,
            "summary": persisted["summary"],
            "inputs": persisted["inputs"],
            "artifacts": persisted["artifacts"],
        }

    def _normalize_player_reels_team_filter(value: str | None) -> str:
        team_filter = str(value or "all")
        if team_filter not in {"all", "ours", "opponent", "unknown"}:
            return "all"
        return team_filter

    def _normalize_player_reels_sort(value: str | None) -> str:
        sort_by = str(value or "best_score_desc")
        if sort_by not in {"best_score_desc", "segment_count_desc", "name_asc", "player_id_asc"}:
            return "best_score_desc"
        return sort_by

    def _player_display_name(player: dict[str, Any]) -> str:
        jersey_number = player.get("jersey_number")
        name = player.get("player_name") or player.get("name")
        if jersey_number is not None and name:
            return f"#{jersey_number} {name}"
        if name:
            return str(name)
        if jersey_number is not None:
            return f"#{jersey_number}"
        player_id = _coerce_int(player.get("player_id"))
        return f"Player {player_id if player_id is not None else 'unknown'}"

    def _filter_player_reels_for_export(
        reels_data: dict[str, Any],
        team_filter: str,
        min_score: float,
        top_n: int,
        sort_by: str,
        player_ids: set[int] | None = None,
    ) -> dict[str, Any]:
        """Apply player reel export filters consistent with UI behavior."""
        all_players = reels_data.get("players", [])
        if not isinstance(all_players, list):
            all_players = []

        filtered_players: list[dict[str, Any]] = []
        for player in all_players:
            if not isinstance(player, dict):
                continue

            player_id = _coerce_int(player.get("player_id"))
            if player_ids is not None:
                if player_id is None or player_id not in player_ids:
                    continue

            team_hint_raw = player.get("team_hint")
            team_hint = team_hint_raw if team_hint_raw in {"ours", "opponent"} else "unknown"
            if team_filter != "all" and team_hint != team_filter:
                continue

            segments_raw = player.get("segments", [])
            if not isinstance(segments_raw, list):
                segments_raw = []
            segments = [dict(segment) for segment in segments_raw if isinstance(segment, dict)]
            segments = [
                segment
                for segment in segments
                if _coerce_float(segment.get("player_segment_score"), 0.0) >= min_score
            ]
            segments.sort(
                key=lambda row: _coerce_float(row.get("player_segment_score"), 0.0),
                reverse=True,
            )
            segments = segments[:top_n]
            if not segments:
                continue

            max_score = max(
                (_coerce_float(segment.get("player_segment_score"), 0.0) for segment in segments),
                default=0.0,
            )

            player_copy = dict(player)
            player_copy["segments"] = segments
            player_copy["segment_count"] = len(segments)
            player_copy["_team_hint_norm"] = team_hint
            player_copy["_max_score"] = max_score
            filtered_players.append(player_copy)

        if sort_by == "name_asc":
            filtered_players.sort(key=lambda row: _player_display_name(row))
        elif sort_by == "segment_count_desc":
            filtered_players.sort(
                key=lambda row: (
                    int(row.get("segment_count", 0)),
                    _coerce_float(row.get("_max_score"), 0.0),
                ),
                reverse=True,
            )
        elif sort_by == "player_id_asc":
            filtered_players.sort(key=lambda row: (_coerce_int(row.get("player_id")) or 0))
        else:
            filtered_players.sort(key=lambda row: _coerce_float(row.get("_max_score"), 0.0), reverse=True)

        for player in filtered_players:
            player.pop("_team_hint_norm", None)
            player.pop("_max_score", None)

        source_summary = reels_data.get("summary")
        source_summary_payload = source_summary if isinstance(source_summary, dict) else {}
        source_players_count = sum(
            1
            for player in all_players
            if isinstance(player, dict) and isinstance(player.get("segments"), list) and player.get("segments")
        )
        source_segments_count = sum(
            len(player.get("segments", []))
            for player in all_players
            if isinstance(player, dict) and isinstance(player.get("segments"), list)
        )
        filtered_segments_count = sum(int(player.get("segment_count", 0)) for player in filtered_players)

        return {
            "players": filtered_players,
            "summary": {
                "players_with_reels": len(filtered_players),
                "player_segments_total": filtered_segments_count,
                "source_players_with_reels": source_summary_payload.get(
                    "players_with_reels",
                    source_players_count,
                ),
                "source_player_segments_total": source_summary_payload.get(
                    "player_segments_total",
                    source_segments_count,
                ),
                "team_filter": team_filter,
                "min_score": min_score,
                "top_n": top_n,
                "sort_by": sort_by,
            },
        }

    def _export_player_reels_package(
        run_name: str,
        team_filter: str,
        min_score: float,
        top_n: int,
        sort_by: str,
        include_clips: bool,
        player_ids: set[int] | None,
    ) -> dict[str, Any]:
        """Create filtered player reel ZIP package and return metadata."""
        run_path = runs_dir / run_name
        if not run_path.exists():
            raise HTTPException(status_code=404, detail="Run not found")

        reels_data = load_player_reels_for_run(run_name)
        filtered = _filter_player_reels_for_export(
            reels_data=reels_data,
            team_filter=team_filter,
            min_score=min_score,
            top_n=top_n,
            sort_by=sort_by,
            player_ids=player_ids,
        )
        filtered_players = filtered["players"]
        filtered_summary = filtered["summary"]

        filtered_payload = {
            "schema_version": reels_data.get("schema_version", "1.0"),
            "video_id": reels_data.get("video_id"),
            "run_name": run_name,
            "players": filtered_players,
            "summary": filtered_summary,
            "count": len(filtered_players),
        }

        flat_rows: list[dict[str, Any]] = []
        all_keys: set[str] = set()
        for player in filtered_players:
            for segment in player.get("segments", []):
                if not isinstance(segment, dict):
                    continue
                row: dict[str, Any] = {
                    "player_id": player.get("player_id"),
                    "player_name": player.get("player_name"),
                    "jersey_number": player.get("jersey_number"),
                    "team_hint": player.get("team_hint"),
                }
                for key, value in segment.items():
                    if isinstance(value, (dict, list)):
                        row[key] = json.dumps(value)
                    else:
                        row[key] = value
                flat_rows.append(row)
                all_keys.update(row.keys())

        preferred_keys = [
            "player_id",
            "player_name",
            "jersey_number",
            "team_hint",
            "segment_id",
            "start_time",
            "end_time",
            "duration",
            "player_segment_score",
            "score",
            "reasons",
            "sources",
            "clip_path",
            "has_clip",
        ]
        extra_keys = sorted(key for key in all_keys if key not in preferred_keys)
        fieldnames = [key for key in preferred_keys if key in all_keys or key in {"player_id", "player_name"}] + extra_keys

        csv_buffer = io.StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
        writer.writeheader()
        for row in flat_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

        clip_entries: list[dict[str, Any]] = []
        clip_path_map: dict[str, tuple[Path, str]] = {}
        if include_clips:
            for player in filtered_players:
                player_id = _coerce_int(player.get("player_id"))
                for segment in player.get("segments", []):
                    if not isinstance(segment, dict):
                        continue
                    clip_path = resolve_run_artifact_path(run_path, segment.get("clip_path"))
                    if clip_path is None:
                        continue
                    clip_key = str(clip_path)
                    if clip_key in clip_path_map:
                        continue
                    try:
                        relative_clip = clip_path.relative_to(run_path).as_posix()
                        zip_clip_path = f"clips/{relative_clip}"
                    except ValueError:
                        owner = player_id if player_id is not None else "unknown"
                        zip_clip_path = f"clips/player_{owner}/{clip_path.name}"
                    clip_path_map[clip_key] = (clip_path, zip_clip_path)
                    clip_entries.append(
                        {
                            "player_id": player_id,
                            "segment_id": str(segment.get("segment_id", "")),
                            "source_path": str(clip_path),
                            "zip_path": zip_clip_path,
                        }
                    )

        exports_dir = run_path / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        export_name = f"player_reels_export_{int(datetime.now(timezone.utc).timestamp() * 1000)}_{uuid4().hex[:8]}.zip"
        export_path = exports_dir / export_name

        manifest_payload = {
            "schema_version": "1.0",
            "run_name": run_name,
            "video_id": reels_data.get("video_id"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_artifact": "player_highlights.json",
            "filters": {
                "team_filter": team_filter,
                "min_score": min_score,
                "top_n": top_n,
                "sort_by": sort_by,
                "include_clips": include_clips,
                "player_ids": sorted(player_ids) if player_ids is not None else None,
            },
            "summary": {
                **filtered_summary,
                "clip_files_included": len(clip_path_map),
            },
            "outputs": {
                "reels_json": "player_reels/player_highlights_filtered.json",
                "reels_csv": "player_reels/player_highlights_filtered.csv",
                "manifest": "player_reels/export_manifest.json",
                "clips_prefix": "clips/",
            },
            "clip_entries": clip_entries,
        }

        with zipfile.ZipFile(export_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr(
                "player_reels/player_highlights_filtered.json",
                json.dumps(filtered_payload, indent=2),
            )
            zip_file.writestr("player_reels/player_highlights_filtered.csv", csv_buffer.getvalue())
            zip_file.writestr(
                "player_reels/export_manifest.json",
                json.dumps(manifest_payload, indent=2),
            )
            for clip_path, zip_clip_path in clip_path_map.values():
                zip_file.write(clip_path, arcname=zip_clip_path)

        return {
            "success": True,
            "run_name": run_name,
            "export_name": export_name,
            "export_path": str(export_path),
            "download_url": f"/api/runs/{run_name}/player_reels/exports/{export_name}",
            "summary": manifest_payload["summary"],
            "filters": manifest_payload["filters"],
        }

    _cross_match_artifact_map: dict[str, tuple[str, str]] = {
        "report_json": ("cross_match_report.json", "application/json"),
        "match_trends_csv": ("cross_match_match_trends.csv", "text/csv"),
        "player_trends_csv": ("cross_match_player_trends.csv", "text/csv"),
        "coach_template_md": ("coach_report_template.md", "text/markdown"),
        "player_templates_md": ("player_report_templates.md", "text/markdown"),
    }

    def _load_cross_match_payload(run_name: str) -> dict[str, Any]:
        """Load cross-match report payload for a run."""
        run_path = runs_dir / run_name
        if not run_path.exists():
            raise HTTPException(status_code=404, detail="Run not found")

        report_path = run_path / "cross_match_report.json"
        if not report_path.exists():
            raise HTTPException(status_code=404, detail="Cross-match report not found")

        with open(report_path) as f:
            report_data = json.load(f)
        if not isinstance(report_data, dict):
            raise HTTPException(status_code=400, detail="Invalid cross-match report payload")

        available_artifacts: dict[str, dict[str, Any]] = {}
        for artifact_id, (filename, media_type) in _cross_match_artifact_map.items():
            artifact_path = run_path / filename
            if not artifact_path.exists() or not artifact_path.is_file():
                continue
            available_artifacts[artifact_id] = {
                "artifact_id": artifact_id,
                "file_name": filename,
                "media_type": media_type,
                "download_url": f"/api/runs/{run_name}/cross_match/artifacts/{artifact_id}",
                "size_bytes": artifact_path.stat().st_size,
            }

        summary = report_data.get("summary")
        summary_payload = summary if isinstance(summary, dict) else {}
        return {
            "run_name": run_name,
            "report": report_data,
            "summary": summary_payload,
            "available_artifacts": available_artifacts,
        }

    def _resolve_cross_match_artifact_path(run_name: str, artifact_id: str) -> tuple[Path, str, str]:
        """Resolve cross-match artifact file path from artifact id."""
        run_path = runs_dir / run_name
        if not run_path.exists():
            raise HTTPException(status_code=404, detail="Run not found")

        normalized_artifact = str(artifact_id or "").strip()
        mapping = _cross_match_artifact_map.get(normalized_artifact)
        if mapping is None:
            raise HTTPException(status_code=400, detail="Invalid cross-match artifact id")

        filename, media_type = mapping
        artifact_path = run_path / filename
        if not artifact_path.exists() or not artifact_path.is_file():
            raise HTTPException(status_code=404, detail=f"Cross-match artifact not found: {filename}")

        return artifact_path, filename, media_type

    def _export_cross_match_package(run_name: str, include_templates: bool) -> dict[str, Any]:
        """Create ZIP package of cross-match report artifacts."""
        payload = _load_cross_match_payload(run_name)
        run_path = runs_dir / run_name
        report_data = payload["report"]

        selected_artifacts: list[dict[str, Any]] = []
        for artifact_id, (filename, media_type) in _cross_match_artifact_map.items():
            is_template = artifact_id in {"coach_template_md", "player_templates_md"}
            if is_template and not include_templates:
                continue

            artifact_path = run_path / filename
            if not artifact_path.exists() or not artifact_path.is_file():
                continue

            selected_artifacts.append(
                {
                    "artifact_id": artifact_id,
                    "file_name": filename,
                    "media_type": media_type,
                    "path": artifact_path,
                }
            )

        if not selected_artifacts:
            raise HTTPException(status_code=404, detail="No cross-match artifacts available to export")

        exports_dir = run_path / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        export_name = f"cross_match_export_{int(datetime.now(timezone.utc).timestamp() * 1000)}_{uuid4().hex[:8]}.zip"
        export_path = exports_dir / export_name

        manifest_payload = {
            "schema_version": "1.0",
            "run_name": run_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "filters": {
                "include_templates": bool(include_templates),
            },
            "summary": {
                "matches_analyzed": _coerce_int(payload.get("summary", {}).get("matches_analyzed")) or 0,
                "unique_players": _coerce_int(payload.get("summary", {}).get("unique_players")) or 0,
                "artifact_files": len(selected_artifacts),
            },
            "artifacts": [
                {
                    "artifact_id": row["artifact_id"],
                    "file_name": row["file_name"],
                    "media_type": row["media_type"],
                    "zip_path": f"cross_match/{row['file_name']}",
                }
                for row in selected_artifacts
            ],
            "source_schema_version": report_data.get("schema_version"),
        }

        with zipfile.ZipFile(export_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
            for row in selected_artifacts:
                zip_file.write(row["path"], arcname=f"cross_match/{row['file_name']}")
            zip_file.writestr(
                "cross_match/export_manifest.json",
                json.dumps(manifest_payload, indent=2),
            )

        return {
            "success": True,
            "run_name": run_name,
            "export_name": export_name,
            "export_path": str(export_path),
            "download_url": f"/api/runs/{run_name}/cross_match/exports/{export_name}",
            "summary": manifest_payload["summary"],
            "filters": manifest_payload["filters"],
        }

    @app.get("/", response_class=HTMLResponse)
    async def root():
        """Serve main UI page."""
        index_path = static_dir / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return HTMLResponse(
            content="<h1>Veo Soccer Analysis UI</h1><p>Frontend not found. Check src/ui/static/</p>"
        )

    @app.post("/api/browse-videos")
    async def browse_videos():
        """Open a native macOS file picker dialog and return selected video paths."""
        import asyncio
        import subprocess

        def _pick_files() -> list[str]:
            try:
                script = (
                    'set theFiles to choose file of type '
                    '{"public.movie", "public.mpeg-4", "com.apple.quicktime-movie"} '
                    'with prompt "Select video files" '
                    'with multiple selections allowed\n'
                    'set output to ""\n'
                    'repeat with f in theFiles\n'
                    '    set output to output & POSIX path of f & linefeed\n'
                    'end repeat\n'
                    'return output'
                )
                result = subprocess.run(
                    ["osascript", "-e", script],
                    capture_output=True, text=True, timeout=120,
                )
                if result.returncode != 0:
                    return []
                paths = [
                    line.strip() for line in result.stdout.strip().split("\n")
                    if line.strip()
                ]
                return paths
            except Exception:
                return []

        loop = asyncio.get_event_loop()
        paths = await loop.run_in_executor(None, _pick_files)
        return {"paths": paths}

    @app.get("/api/pipeline/configs")
    async def list_pipeline_configs():
        """List available YAML configs for browser-triggered pipeline runs."""
        configs_dir = project_root / "configs"
        config_items: list[dict[str, Any]] = [
            {
                "label": "Built-in default",
                "path": None,
                "file_name": None,
            }
        ]

        if configs_dir.exists():
            for config_path in sorted(configs_dir.glob("*.yaml")):
                try:
                    relative_path = config_path.relative_to(project_root).as_posix()
                except ValueError:
                    relative_path = str(config_path)
                config_items.append(
                    {
                        "label": config_path.stem.replace("_", " "),
                        "path": relative_path,
                        "file_name": config_path.name,
                    }
                )

        return {
            "configs": config_items,
            "count": len(config_items),
        }

    @app.get("/api/pipeline/jobs")
    async def list_pipeline_jobs(limit: int = 60, include_logs: bool = False):
        """List background pipeline jobs."""
        safe_limit = max(1, min(int(limit), 200))
        with pipeline_jobs_lock:
            job_ids = list(pipeline_job_order)[-safe_limit:]
            jobs = [
                serialize_pipeline_job(pipeline_jobs[job_id], include_logs=include_logs)
                for job_id in reversed(job_ids)
                if job_id in pipeline_jobs
            ]

        status_counts: dict[str, int] = {
            "queued": 0,
            "running": 0,
            "succeeded": 0,
            "failed": 0,
            "cancelled": 0,
        }
        for job in jobs:
            status = str(job.get("status") or "")
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "jobs": jobs,
            "count": len(jobs),
            "status_counts": status_counts,
            "max_parallel_jobs": max_parallel_jobs,
        }

    @app.get("/api/pipeline/jobs/{job_id}")
    async def get_pipeline_job(job_id: str, include_logs: bool = True):
        """Get one background pipeline job by id."""
        with pipeline_jobs_lock:
            job = pipeline_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Pipeline job not found")
            return serialize_pipeline_job(job, include_logs=include_logs)

    @app.post("/api/pipeline/jobs/{job_id}/cancel")
    async def cancel_pipeline_job(job_id: str):
        """Cancel queued/running pipeline job."""
        queued_future = None
        queued_cancel = False

        with pipeline_jobs_lock:
            job = pipeline_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Pipeline job not found")

            status = str(job.get("status") or "")
            if status in {"succeeded", "failed", "cancelled"}:
                return {
                    "success": True,
                    "job": serialize_pipeline_job(job, include_logs=False),
                }

            job["cancel_requested"] = True
            if status == "queued":
                job["status"] = "cancelled"
                job["message"] = "Cancelled before start"
                job["finished_at"] = _utc_now_iso()
                queued_future = pipeline_job_futures.get(job_id)
                queued_cancel = True
            else:
                job["message"] = "Cancellation requested"

            persist_pipeline_jobs_locked()
            payload = serialize_pipeline_job(job, include_logs=False)

        if queued_cancel and queued_future is not None:
            queued_future.cancel()
            with pipeline_jobs_lock:
                pipeline_job_futures.pop(job_id, None)
                persist_pipeline_jobs_locked()

        append_pipeline_job_log(
            job_id,
            "Cancellation requested by user",
        )
        return {
            "success": True,
            "job": payload,
        }

    @app.delete("/api/pipeline/jobs/{job_id}")
    async def delete_pipeline_job(job_id: str, clean_files: bool = True):
        """Delete a finished or queued pipeline job and optionally remove its run files."""
        output_dir_to_remove: Path | None = None

        with pipeline_jobs_lock:
            job = pipeline_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Pipeline job not found")

            status = str(job.get("status") or "")
            if status == "running":
                raise HTTPException(
                    status_code=409,
                    detail="Cannot delete a running job. Cancel it first.",
                )

            if clean_files:
                raw_output = str(job.get("output_dir") or "")
                if raw_output:
                    candidate = Path(raw_output).resolve()
                    if candidate != runs_dir and str(candidate).startswith(str(runs_dir)):
                        output_dir_to_remove = candidate

            # Remove from in-memory state
            pipeline_jobs.pop(job_id, None)
            pipeline_job_futures.pop(job_id, None)
            if job_id in pipeline_job_order:
                pipeline_job_order.remove(job_id)
            persist_pipeline_jobs_locked()

        # Remove run directory outside the lock
        files_removed = False
        if output_dir_to_remove is not None and output_dir_to_remove.exists():
            import shutil
            shutil.rmtree(output_dir_to_remove, ignore_errors=True)
            files_removed = True

        return {
            "success": True,
            "job_id": job_id,
            "files_removed": files_removed,
        }

    @app.post("/api/pipeline/jobs/{job_id}/retry")
    async def retry_pipeline_job(job_id: str):
        """Retry a finished job using same inputs/config and a new run name."""
        with pipeline_jobs_lock:
            source = pipeline_jobs.get(job_id)
            if source is None:
                raise HTTPException(status_code=404, detail="Pipeline job not found")
            source_status = str(source.get("status") or "")
            if source_status in {"queued", "running"}:
                raise HTTPException(
                    status_code=409,
                    detail="Cannot retry a queued/running job. Cancel it first.",
                )

            source_video_path = str(source.get("video_path") or "")
            source_run_name = str(source.get("run_name") or "")
            source_config = source.get("config_path")
            source_resume = bool(source.get("resume", False))
            source_no_overlay = bool(source.get("no_overlay", False))

            reserved_names = {
                str(job.get("run_name"))
                for job in pipeline_jobs.values()
                if job.get("status") in {"queued", "running"} and job.get("run_name")
            }

        video_path = Path(source_video_path)
        if not video_path.exists() or not video_path.is_file():
            raise HTTPException(status_code=400, detail=f"Source video not found: {source_video_path}")

        run_name = ensure_unique_run_name(f"{source_run_name}_retry", reserved_names)
        queued = queue_one_pipeline_job(
            video_path=video_path,
            run_name=run_name,
            config_path=str(source_config) if isinstance(source_config, str) else None,
            resume=source_resume,
            no_overlay=source_no_overlay,
            source_job_id=job_id,
        )
        return {
            "success": True,
            "source_job_id": job_id,
            "job": queued,
        }

    @app.post("/api/pipeline/jobs/{job_id}/duplicate")
    async def duplicate_pipeline_job(job_id: str):
        """Duplicate a job with same inputs/settings and queue it."""
        with pipeline_jobs_lock:
            source = pipeline_jobs.get(job_id)
            if source is None:
                raise HTTPException(status_code=404, detail="Pipeline job not found")

            source_video_path = str(source.get("video_path") or "")
            source_run_name = str(source.get("run_name") or "")
            source_config = source.get("config_path")
            source_resume = bool(source.get("resume", False))
            source_no_overlay = bool(source.get("no_overlay", False))

            reserved_names = {
                str(job.get("run_name"))
                for job in pipeline_jobs.values()
                if job.get("status") in {"queued", "running"} and job.get("run_name")
            }

        video_path = Path(source_video_path)
        if not video_path.exists() or not video_path.is_file():
            raise HTTPException(status_code=400, detail=f"Source video not found: {source_video_path}")

        run_name = ensure_unique_run_name(f"{source_run_name}_copy", reserved_names)
        queued = queue_one_pipeline_job(
            video_path=video_path,
            run_name=run_name,
            config_path=str(source_config) if isinstance(source_config, str) else None,
            resume=source_resume,
            no_overlay=source_no_overlay,
            source_job_id=job_id,
        )
        return {
            "success": True,
            "source_job_id": job_id,
            "job": queued,
        }

    @app.post("/api/pipeline/jobs")
    async def queue_pipeline_jobs(body: QueuePipelineJobsBody):
        """Queue one or many full pipeline runs for background execution."""
        raw_video_paths = [
            str(path).strip()
            for path in body.video_paths
            if isinstance(path, str) and str(path).strip()
        ]
        if not raw_video_paths:
            raise HTTPException(status_code=400, detail="video_paths must include at least one path")

        if body.run_name and len(raw_video_paths) != 1:
            raise HTTPException(
                status_code=400,
                detail="run_name can only be used when queueing exactly one video",
            )

        resolved_videos: list[Path] = []
        for raw_path in raw_video_paths:
            resolved = resolve_user_path(raw_path)
            if not resolved.exists() or not resolved.is_file():
                raise HTTPException(
                    status_code=400,
                    detail=f"Video file not found: {raw_path}",
                )
            resolved_videos.append(resolved)

        resolved_config_path: str | None = None
        if body.config_path:
            config_candidate = resolve_user_path(body.config_path)
            if not config_candidate.exists() or not config_candidate.is_file():
                raise HTTPException(status_code=400, detail=f"Config file not found: {body.config_path}")
            resolved_config_path = str(config_candidate)

        run_plan: list[tuple[Path, str]] = []
        with pipeline_jobs_lock:
            reserved_names = {
                str(job.get("run_name"))
                for job in pipeline_jobs.values()
                if job.get("status") in {"queued", "running"} and job.get("run_name")
            }

            for idx, video_path in enumerate(resolved_videos):
                if body.run_name:
                    base_name = body.run_name
                else:
                    prefix = _sanitize_run_component(body.run_name_prefix or "")
                    if prefix:
                        base_name = f"{prefix}_{video_path.stem}"
                    elif len(resolved_videos) > 1:
                        base_name = f"batch_{idx + 1:03d}_{video_path.stem}"
                    else:
                        base_name = video_path.stem

                run_name = ensure_unique_run_name(base_name, reserved_names=reserved_names)
                run_plan.append((video_path, run_name))

        if not body.resume:
            for _, run_name in run_plan:
                output_dir = (runs_dir / run_name).resolve()
                if output_dir.exists():
                    raise HTTPException(
                        status_code=409,
                        detail=f"Output already exists for run '{run_name}'. Use resume=true or choose another name.",
                    )

        created_jobs = [
            queue_one_pipeline_job(
                video_path=video_path,
                run_name=run_name,
                config_path=resolved_config_path,
                resume=bool(body.resume),
                no_overlay=bool(body.no_overlay),
                home_team_id=body.home_team_id,
                away_team_id=body.away_team_id,
                home_kit=body.home_kit,
                away_kit=body.away_kit,
            )
            for video_path, run_name in run_plan
        ]

        return {
            "accepted_count": len(created_jobs),
            "jobs": created_jobs,
            "max_parallel_jobs": max_parallel_jobs,
        }

    @app.get("/api/runs")
    async def list_runs():
        """List all available analysis runs."""
        if not runs_dir.exists():
            return {"runs": []}

        runs = []
        for run_path in sorted(runs_dir.iterdir()):
            if not run_path.is_dir():
                continue

            manifest_path = run_path / "run_manifest.json"
            video_metadata_path = run_path / "video_metadata.json"
            events_path = run_path / "events.jsonl"
            timeline_path = run_path / "score_timeline.json"
            match_stats_path = run_path / "match_stats.json"
            player_reels_path = run_path / "player_highlights.json"
            cross_match_report_path = run_path / "cross_match_report.json"

            # Load basic info
            run_info = {
                "name": run_path.name,
                "path": str(run_path),
                "has_manifest": manifest_path.exists(),
                "has_events": events_path.exists(),
                "has_timeline": timeline_path.exists(),
                "has_match_stats": match_stats_path.exists(),
                "has_player_reels": player_reels_path.exists(),
                "has_cross_match_report": cross_match_report_path.exists(),
            }

            # Add video metadata if available
            if video_metadata_path.exists():
                with open(video_metadata_path) as f:
                    metadata = json.load(f)
                    run_info["duration"] = metadata.get("duration")
                    run_info["fps"] = metadata.get("fps")
                    run_info["resolution"] = f"{metadata.get('width')}x{metadata.get('height')}"

            # Add event counts if available
            if events_path.exists():
                event_counts = {"shot": 0, "goal": 0, "other": 0}
                with open(events_path) as f:
                    for line in f:
                        event = json.loads(line)
                        event_type = event.get("event_type", "other")
                        event_counts[event_type] = event_counts.get(event_type, 0) + 1
                run_info["event_counts"] = event_counts

            if player_reels_path.exists():
                with open(player_reels_path) as f:
                    reels_data = json.load(f)
                summary = reels_data.get("summary", {})
                run_info["player_reel_summary"] = {
                    "players_with_reels": summary.get(
                        "players_with_reels", len(reels_data.get("players", []))
                    ),
                    "player_segments_total": summary.get("player_segments_total", 0),
                }

            if cross_match_report_path.exists():
                with open(cross_match_report_path) as f:
                    cross_match_report = json.load(f)
                summary = cross_match_report.get("summary", {})
                if isinstance(summary, dict):
                    run_info["cross_match_summary"] = {
                        "matches_analyzed": summary.get("matches_analyzed", 0),
                        "unique_players": summary.get("unique_players", 0),
                    }

            runs.append(run_info)

        return {"runs": runs}

    @app.delete("/api/runs/{run_name}")
    async def delete_run(run_name: str):
        """Delete an analysis run — removes the run directory and any associated pipeline job metadata."""
        import shutil

        run_path = (runs_dir / run_name).resolve()
        if run_path == runs_dir or not str(run_path).startswith(str(runs_dir)):
            raise HTTPException(status_code=400, detail="Invalid run name")
        if not run_path.exists():
            raise HTTPException(status_code=404, detail="Run not found")

        # Block deletion if a pipeline job is currently running for this run
        with pipeline_jobs_lock:
            for job in pipeline_jobs.values():
                if (
                    str(job.get("run_name") or "") == run_name
                    and job.get("status") == "running"
                ):
                    raise HTTPException(
                        status_code=409,
                        detail="Cannot delete a run while its pipeline job is running.",
                    )

            # Remove any associated pipeline job records
            jobs_to_remove = [
                jid
                for jid, job in pipeline_jobs.items()
                if str(job.get("run_name") or "") == run_name
            ]
            for jid in jobs_to_remove:
                pipeline_jobs.pop(jid, None)
                pipeline_job_futures.pop(jid, None)
                if jid in pipeline_job_order:
                    pipeline_job_order.remove(jid)
            if jobs_to_remove:
                persist_pipeline_jobs_locked()

        # Remove the run directory
        shutil.rmtree(run_path, ignore_errors=True)

        return {
            "success": True,
            "run_name": run_name,
            "jobs_removed": len(jobs_to_remove),
        }

    @app.get("/api/runs/{run_name}/metadata")
    async def get_run_metadata(run_name: str):
        """Get metadata for a specific run."""
        run_path = runs_dir / run_name
        if not run_path.exists():
            raise HTTPException(status_code=404, detail="Run not found")

        result: dict[str, Any] = {"name": run_name}

        # Load video metadata
        video_metadata_path = run_path / "video_metadata.json"
        if video_metadata_path.exists():
            with open(video_metadata_path) as f:
                result["video_metadata"] = json.load(f)

        # Load run manifest
        manifest_path = run_path / "run_manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                result["manifest"] = json.load(f)

        # Load team info
        teams_path = run_path / "teams.json"
        if teams_path.exists():
            with open(teams_path) as f:
                result["teams"] = json.load(f)

        # Load summary.json
        summary_path = run_path / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                result["summary"] = json.load(f)

        return result

    @app.get("/api/runs/{run_name}/team_analytics")
    async def get_run_team_analytics(run_name: str):
        """Get team analytics data for a specific run."""
        run_path = runs_dir / run_name
        if not run_path.exists():
            raise HTTPException(status_code=404, detail="Run not found")

        analytics_path = run_path / "team_analytics.json"
        if not analytics_path.exists():
            raise HTTPException(status_code=404, detail="Team analytics not available for this run")

        with open(analytics_path) as f:
            return json.load(f)

    @app.get("/api/runs/{run_name}/match_stats")
    async def get_run_match_stats(run_name: str):
        """Get match_stats.json for a specific run."""
        run_path = runs_dir / run_name
        if not run_path.exists():
            raise HTTPException(status_code=404, detail="Run not found")

        match_stats_path = run_path / "match_stats.json"
        if not match_stats_path.exists():
            raise HTTPException(status_code=404, detail="Match stats not available for this run")

        with open(match_stats_path) as f:
            return json.load(f)

    @app.get("/api/runs/{run_name}/events")
    async def get_run_events(run_name: str):
        """Get all events for a specific run, merged with user confirmations."""
        run_path = runs_dir / run_name
        events_path = run_path / "events.jsonl"

        if not run_path.exists():
            raise HTTPException(status_code=404, detail="Run not found")

        # Load auto-detected events
        events = []
        if events_path.exists():
            with open(events_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        events.append(json.loads(line))

        # Load and merge confirmations
        confirmations = load_confirmations(run_path)
        merged_events = merge_events_with_confirmations(events, confirmations)

        return {"events": merged_events, "count": len(merged_events)}

    @app.get("/api/runs/{run_name}/timeline")
    async def get_score_timeline(run_name: str):
        """Get score timeline for a specific run."""
        run_path = runs_dir / run_name
        timeline_path = run_path / "score_timeline.json"

        if not timeline_path.exists():
            raise HTTPException(status_code=404, detail="Timeline not found")

        with open(timeline_path) as f:
            timeline = json.load(f)

        return timeline

    @app.get("/api/runs/{run_name}/video")
    async def get_video(run_name: str, original: bool = False):
        """
        Stream video for playback.

        Args:
            run_name: Run directory name
            original: If True, return original video; if False, return overlay (if exists)
        """
        run_path = runs_dir / run_name

        # Try to get original video path from manifest
        manifest_path = run_path / "run_manifest.json"
        original_video_path = None

        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
                original_video_path = manifest.get("original_video_path")

        # Decide which video to serve
        if original and original_video_path:
            # Serve original video
            video_path = Path(original_video_path)
            if not video_path.exists():
                raise HTTPException(status_code=404, detail="Original video not found")
        else:
            # Serve overlay video (fallback to original if overlay doesn't exist)
            overlay_path = run_path / "overlay.mp4"
            if overlay_path.exists():
                video_path = overlay_path
            elif original_video_path:
                video_path = Path(original_video_path)
            else:
                raise HTTPException(status_code=404, detail="No video found")

        return FileResponse(
            video_path,
            media_type="video/mp4",
            headers={"Accept-Ranges": "bytes"},
        )

    @app.get("/api/runs/{run_name}/tracks")
    async def get_tracks(run_name: str, frame_start: int = None, frame_end: int = None):
        """
        Get track data for a specific run.

        Args:
            run_name: Run directory name
            frame_start: Optional start frame (for windowed loading)
            frame_end: Optional end frame (for windowed loading)
        """
        import pandas as pd
        import numpy as np

        run_path = runs_dir / run_name
        tracks_path = run_path / "tracks.parquet"

        if not tracks_path.exists():
            raise HTTPException(status_code=404, detail="Tracks not found")

        # Load tracks
        df = pd.read_parquet(tracks_path)

        # Filter by frame window if specified
        if frame_start is not None and frame_end is not None:
            df = df[(df["frame_idx"] >= frame_start) & (df["frame_idx"] <= frame_end)].copy()

        # Expand bbox into x1, y1, x2, y2 columns for easier frontend use
        if "bbox" in df.columns:
            bbox_expanded = pd.DataFrame(df["bbox"].tolist(), index=df.index, columns=["x1", "y1", "x2", "y2"])
            df = pd.concat([df.drop("bbox", axis=1), bbox_expanded], axis=1)

        # Coerce coordinate columns to numeric (bbox expansion may produce object dtype)
        coord_cols = ["x1", "y1", "x2", "y2"]
        for col in coord_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Filter out rows with NaN or Inf in coordinates
        for col in coord_cols:
            if col in df.columns:
                df = df[df[col].notna() & ~np.isinf(df[col])]

        # Convert to records and sanitize for JSON serialization
        import math

        def sanitize(val):
            if val is None:
                return None
            if isinstance(val, float):
                return None if (math.isnan(val) or math.isinf(val)) else val
            if isinstance(val, (np.integer,)):
                return int(val)
            if isinstance(val, (np.floating,)):
                v = float(val)
                return None if (math.isnan(v) or math.isinf(v)) else v
            if isinstance(val, np.ndarray):
                return val.tolist()
            if isinstance(val, (np.bool_,)):
                return bool(val)
            return val

        tracks = df.to_dict(orient="records")
        tracks = [{k: sanitize(v) for k, v in row.items()} for row in tracks]

        return {
            "tracks": tracks,
            "count": len(tracks),
            "frame_start": frame_start,
            "frame_end": frame_end,
        }

    @app.post("/api/runs/{run_name}/events/{event_id}/confirm")
    async def confirm_event(run_name: str, event_id: str, body: ConfirmRejectBody):
        """Confirm an auto-detected event."""
        run_path = runs_dir / run_name
        if not run_path.exists():
            raise HTTPException(status_code=404, detail="Run not found")

        # Save confirmation
        action_record = {
            "action": "confirm",
            "event_id": event_id,
            "notes": body.notes,
        }
        save_confirmation(run_path, action_record)

        return {"success": True, "event_id": event_id, "status": "confirmed"}

    @app.post("/api/runs/{run_name}/events/{event_id}/reject")
    async def reject_event(run_name: str, event_id: str, body: ConfirmRejectBody):
        """Reject an auto-detected event (mark as false positive)."""
        run_path = runs_dir / run_name
        if not run_path.exists():
            raise HTTPException(status_code=404, detail="Run not found")

        # Save rejection
        action_record = {
            "action": "reject",
            "event_id": event_id,
            "notes": body.notes,
        }
        save_confirmation(run_path, action_record)

        return {"success": True, "event_id": event_id, "status": "rejected"}

    @app.post("/api/runs/{run_name}/events")
    async def add_manual_event(run_name: str, body: AddEventBody):
        """Add a manually created event."""
        run_path = runs_dir / run_name
        if not run_path.exists():
            raise HTTPException(status_code=404, detail="Run not found")

        # Validate event type
        if body.event_type not in ["shot", "goal"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid event_type. Must be 'shot' or 'goal'."
            )

        # Generate manual event ID
        event_id = f"manual_{int(datetime.now(timezone.utc).timestamp() * 1000)}"

        # Create event data
        event_data = {
            "event_type": body.event_type,
            "timestamp": body.timestamp,
            "frame_idx": body.frame_idx,
            "confidence": 1.0,  # Manual events have full confidence
            "metadata": body.metadata,
        }

        # Save as add action
        action_record = {
            "action": "add",
            "event_id": event_id,
            "event": event_data,
            "notes": body.notes,
        }
        save_confirmation(run_path, action_record)

        return {"success": True, "event_id": event_id, "event": event_data}

    @app.delete("/api/runs/{run_name}/events/{event_id}")
    async def delete_event(run_name: str, event_id: str):
        """Delete a manual event. Auto events cannot be deleted (reject them instead)."""
        run_path = runs_dir / run_name
        if not run_path.exists():
            raise HTTPException(status_code=404, detail="Run not found")

        # Only allow deleting manual events
        if not event_id.startswith("manual_"):
            raise HTTPException(
                status_code=400,
                detail="Cannot delete auto-detected events. Use reject instead."
            )

        # Save delete action
        action_record = {
            "action": "delete",
            "event_id": event_id,
        }
        save_confirmation(run_path, action_record)

        return {"success": True, "event_id": event_id, "status": "deleted"}

    @app.get("/api/runs/{run_name}/player_reels")
    async def get_player_reels(run_name: str):
        """Get all per-player reel segments for a run."""
        return load_player_reels_for_run(run_name)

    @app.get("/api/runs/{run_name}/player_highlights")
    async def get_player_highlights(run_name: str):
        """Alias for player reels data (matches artifact naming)."""
        return load_player_reels_for_run(run_name)

    @app.get("/api/runs/{run_name}/player_reels/{player_id}")
    async def get_player_reel(run_name: str, player_id: int):
        """Get a single player's reel segments for a run."""
        reels_data = load_player_reels_for_run(run_name)
        for player in reels_data.get("players", []):
            player_id_raw = player.get("player_id")
            if player_id_raw is None:
                continue
            try:
                candidate_id = int(player_id_raw)
            except (TypeError, ValueError):
                continue
            if candidate_id == player_id:
                return player
        raise HTTPException(status_code=404, detail="Player reel not found")

    @app.get("/api/runs/{run_name}/player_reels/{player_id}/segments/{segment_id}/clip")
    async def get_player_reel_segment_clip(run_name: str, player_id: int, segment_id: str):
        """Stream a rendered clip for a player reel segment when available."""
        run_path = runs_dir / run_name
        if not run_path.exists():
            raise HTTPException(status_code=404, detail="Run not found")

        reels_data = load_player_reels_for_run(run_name)

        target_player: dict[str, Any] | None = None
        for player in reels_data.get("players", []):
            player_id_raw = player.get("player_id")
            if player_id_raw is None:
                continue
            try:
                candidate_id = int(player_id_raw)
            except (TypeError, ValueError):
                continue
            if candidate_id == player_id:
                target_player = player
                break

        if target_player is None:
            raise HTTPException(status_code=404, detail="Player reel not found")

        target_segment: dict[str, Any] | None = None
        for segment in target_player.get("segments", []):
            if str(segment.get("segment_id", "")) == str(segment_id):
                target_segment = segment
                break

        if target_segment is None:
            raise HTTPException(status_code=404, detail="Player segment not found")

        clip_path = resolve_run_artifact_path(run_path, target_segment.get("clip_path"))
        if clip_path is None:
            raise HTTPException(status_code=404, detail="Clip not available for this segment")

        return FileResponse(
            clip_path,
            media_type="video/mp4",
            headers={"Accept-Ranges": "bytes"},
        )

    @app.get("/api/runs/{run_name}/player_reels/exports/{export_name}")
    async def download_player_reels_export_package(run_name: str, export_name: str):
        """Download a previously generated player reels export package."""
        run_path = runs_dir / run_name
        if not run_path.exists():
            raise HTTPException(status_code=404, detail="Run not found")

        safe_name = Path(export_name).name
        if safe_name != export_name or not safe_name.endswith(".zip"):
            raise HTTPException(status_code=400, detail="Invalid export file name")

        export_path = run_path / "exports" / safe_name
        if not export_path.exists() or not export_path.is_file():
            raise HTTPException(status_code=404, detail="Export package not found")

        return FileResponse(
            export_path,
            media_type="application/zip",
            filename=safe_name,
            headers={"Accept-Ranges": "bytes"},
        )

    @app.get("/api/runs/{run_name}/cross_match")
    async def get_cross_match_report(run_name: str):
        """Get cross-match season report payload and artifact availability."""
        return _load_cross_match_payload(run_name)

    @app.get("/api/runs/{run_name}/cross_match/artifacts/{artifact_id}")
    async def download_cross_match_artifact(run_name: str, artifact_id: str):
        """Download a specific cross-match artifact for a run."""
        artifact_path, filename, media_type = _resolve_cross_match_artifact_path(
            run_name=run_name,
            artifact_id=artifact_id,
        )
        return FileResponse(
            artifact_path,
            media_type=media_type,
            filename=filename,
            headers={"Accept-Ranges": "bytes"},
        )

    @app.post("/api/runs/{run_name}/cross_match/actions/export_package")
    async def export_cross_match_package(run_name: str, body: ExportCrossMatchPackageBody):
        """Create ZIP package for cross-match report artifacts."""
        return _export_cross_match_package(
            run_name=run_name,
            include_templates=bool(body.include_templates),
        )

    @app.get("/api/runs/{run_name}/cross_match/exports/{export_name}")
    async def download_cross_match_export_package(run_name: str, export_name: str):
        """Download a previously generated cross-match export package."""
        run_path = runs_dir / run_name
        if not run_path.exists():
            raise HTTPException(status_code=404, detail="Run not found")

        safe_name = Path(export_name).name
        if safe_name != export_name or not safe_name.endswith(".zip"):
            raise HTTPException(status_code=400, detail="Invalid export file name")

        export_path = run_path / "exports" / safe_name
        if not export_path.exists() or not export_path.is_file():
            raise HTTPException(status_code=404, detail="Export package not found")

        return FileResponse(
            export_path,
            media_type="application/zip",
            filename=safe_name,
            headers={"Accept-Ranges": "bytes"},
        )

    @app.get("/api/runs/{run_name}/identity_review")
    async def get_run_identity_review(run_name: str):
        """Get editable player + assignment state for a run."""
        return load_identity_review_for_run(run_name)

    @app.get("/api/runs/{run_name}/identity_suggestions")
    async def get_run_identity_suggestions(run_name: str, refresh: bool = False):
        """Get per-track identity suggestions derived from fused assignments."""
        suggestions_data = build_identity_suggestions_artifact(run_name, refresh=refresh)
        return enrich_identity_suggestions_with_players(run_name, suggestions_data)

    @app.post("/api/runs/{run_name}/identity_suggestions/actions/apply")
    async def apply_run_identity_suggestions(run_name: str, body: ApplyIdentitySuggestionsBody):
        """Apply recommended suggestions to selected tracks."""
        return _apply_identity_suggestions(run_name, body)

    @app.post("/api/runs/{run_name}/identity_suggestions/actions/apply_and_recompute")
    async def apply_and_recompute_identity_suggestions(
        run_name: str,
        body: ApplyIdentitySuggestionsAndRecomputeBody,
    ):
        """Apply selected suggestions and immediately recompute player reels."""
        apply_payload = _apply_identity_suggestions(
            run_name,
            ApplyIdentitySuggestionsBody(
                track_ids=body.track_ids,
                min_confidence=body.min_confidence,
                suggested_only=body.suggested_only,
            ),
        )
        recompute_payload = _recompute_player_reels_payload(
            run_name=run_name,
            preserve_existing_clips=body.preserve_existing_clips,
            source="ui_apply_suggestions_recompute",
        )
        return {
            "success": True,
            "run_name": run_name,
            "apply": apply_payload,
            "recompute": recompute_payload,
        }

    @app.post("/api/runs/{run_name}/identity_suggestions/actions/apply_and_preview")
    async def apply_and_preview_identity_suggestions(
        run_name: str,
        body: ApplyIdentitySuggestionsAndRecomputeBody,
    ):
        """Apply selected suggestions and return recompute preview diff without writing reel artifacts."""
        apply_payload = _apply_identity_suggestions(
            run_name,
            ApplyIdentitySuggestionsBody(
                track_ids=body.track_ids,
                min_confidence=body.min_confidence,
                suggested_only=body.suggested_only,
            ),
        )
        compute_result = compute_player_reels_for_run(
            run_name=run_name,
            preserve_existing_clips=body.preserve_existing_clips,
        )
        preview_payload = _store_player_reels_preview(
            run_name=run_name,
            compute_result=compute_result,
            preserve_existing_clips=body.preserve_existing_clips,
            source="ui_apply_suggestions_preview",
        )
        return {
            "success": True,
            "run_name": run_name,
            "apply": apply_payload,
            "preview": preview_payload,
        }

    @app.post("/api/runs/{run_name}/player_reels/actions/approve_preview")
    async def approve_recompute_preview(run_name: str, body: ApprovePlayerReelsPreviewBody):
        """Persist the latest stored preview after validating it is still current."""
        return _approve_player_reels_preview(run_name=run_name, preview_id=body.preview_id)

    @app.post("/api/runs/{run_name}/player_reels/actions/export_package")
    async def export_player_reels_package(run_name: str, body: ExportPlayerReelsPackageBody):
        """Create a downloadable ZIP package from filtered player reel segments."""
        team_filter = _normalize_player_reels_team_filter(body.team_filter)
        sort_by = _normalize_player_reels_sort(body.sort_by)
        min_score = max(0.0, min(1.0, _coerce_float(body.min_score, 0.0)))
        top_n = max(1, min(50, _coerce_int(body.top_n) or 8))
        include_clips = bool(body.include_clips)

        selected_player_ids: set[int] | None = None
        if body.player_ids is not None:
            selected_player_ids = set()
            for player_id in body.player_ids:
                normalized_player_id = _coerce_int(player_id)
                if normalized_player_id is not None:
                    selected_player_ids.add(normalized_player_id)
            if not selected_player_ids:
                raise HTTPException(status_code=400, detail="player_ids must include at least one valid id")

        return _export_player_reels_package(
            run_name=run_name,
            team_filter=team_filter,
            min_score=min_score,
            top_n=top_n,
            sort_by=sort_by,
            include_clips=include_clips,
            player_ids=selected_player_ids,
        )

    @app.get("/api/runs/{run_name}/identity_review/edits")
    async def get_run_identity_edits(run_name: str, limit: int = 50):
        """Get recent identity edit history with undoability state."""
        run_path = runs_dir / run_name
        if not run_path.exists():
            raise HTTPException(status_code=404, detail="Run not found")

        safe_limit = max(1, min(int(limit), 200))
        records = load_identity_edit_records(run_path)

        undone_op_ids: set[str] = set()
        for record in records:
            if record.get("action") != "undo":
                continue
            target_op_id = record.get("target_op_id")
            if isinstance(target_op_id, str) and target_op_id:
                undone_op_ids.add(target_op_id)

        edits: list[dict[str, Any]] = []
        for record in reversed(records):
            action = record.get("action")
            if action not in {"assign", "bulk_assign", "undo"}:
                continue
            edits.append(summarize_identity_edit_record(record, undone_op_ids))
            if len(edits) >= safe_limit:
                break

        undoable_count = sum(1 for edit in edits if bool(edit.get("undoable")))

        return {
            "run_name": run_name,
            "edits": edits,
            "count": len(edits),
            "undoable_count": undoable_count,
        }

    @app.post("/api/runs/{run_name}/identity_review/actions/assign")
    async def assign_run_appearance(run_name: str, body: AssignTrackBody):
        """Assign/unassign a single run track in identity review."""
        return apply_assignment_edits(
            run_name=run_name,
            track_ids=[body.track_id],
            player_id=body.player_id,
            confidence=body.confidence,
            method=body.method,
            action_name="assign",
        )

    @app.post("/api/runs/{run_name}/identity_review/actions/bulk_assign")
    async def bulk_assign_run_appearances(run_name: str, body: BulkAssignAppearancesBody):
        """Assign multiple run track IDs to one player in a single request."""
        return apply_assignment_edits(
            run_name=run_name,
            track_ids=body.track_ids,
            player_id=body.player_id,
            confidence=body.confidence,
            method=body.method,
            action_name="bulk_assign",
        )

    @app.post("/api/runs/{run_name}/identity_review/actions/undo")
    async def undo_last_identity_edit(run_name: str):
        """Undo the most recent non-undone identity assignment edit for this run."""
        run_path = runs_dir / run_name
        if not run_path.exists():
            raise HTTPException(status_code=404, detail="Run not found")

        target_edit = find_last_undoable_identity_edit(run_path)
        if target_edit is None:
            raise HTTPException(status_code=404, detail="No undoable identity edit found")

        return undo_identity_edit_operation(
            run_name=run_name,
            run_path=run_path,
            target_edit=target_edit,
        )

    @app.post("/api/runs/{run_name}/identity_review/actions/undo/{op_id}")
    async def undo_identity_edit_by_operation(run_name: str, op_id: str):
        """Undo a specific assignment operation by operation id."""
        run_path = runs_dir / run_name
        if not run_path.exists():
            raise HTTPException(status_code=404, detail="Run not found")

        target_edit = find_undoable_identity_edit_by_op_id(run_path, op_id)
        if target_edit is None:
            raise HTTPException(status_code=404, detail="Undoable operation not found")

        return undo_identity_edit_operation(
            run_name=run_name,
            run_path=run_path,
            target_edit=target_edit,
        )

    @app.post("/api/runs/{run_name}/player_reels/actions/recompute")
    async def recompute_player_reels(run_name: str, body: RecomputePlayerReelsBody):
        """Rebuild per-player reels from highlights + current identity assignments."""
        return _recompute_player_reels_payload(
            run_name=run_name,
            preserve_existing_clips=body.preserve_existing_clips,
            source="ui_recompute",
        )

    @app.post("/api/runs/{run_name}/player_reels/actions/recompute_preview")
    async def preview_recompute_player_reels(run_name: str, body: RecomputePlayerReelsBody):
        """Preview recompute output and return diff against persisted reels without writing artifacts."""
        compute_result = compute_player_reels_for_run(
            run_name=run_name,
            preserve_existing_clips=body.preserve_existing_clips,
        )
        return _store_player_reels_preview(
            run_name=run_name,
            compute_result=compute_result,
            preserve_existing_clips=body.preserve_existing_clips,
            source="ui_recompute_preview",
        )

    # Player identity endpoints

    @app.post("/api/players")
    async def create_player(body: CreatePlayerBody):
        """Create a player identity record."""
        team_hint = body.team_hint
        if team_hint is not None and team_hint not in {"ours", "opponent"}:
            raise HTTPException(status_code=400, detail="team_hint must be 'ours' or 'opponent'")

        db_path = get_player_db_path()
        try:
            from src.identity import PlayerDatabase
            with PlayerDatabase(db_path) as db:
                player = db.create_player(
                    name=body.name,
                    jersey_number=body.jersey_number,
                    team_hint=team_hint,
                )
                return {"success": True, "player": player.model_dump()}
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="Identity module not available"
            )

    @app.get("/api/players")
    async def list_players():
        """List all players in the identity database."""
        db_path = get_player_db_path()
        if not db_path.exists():
            return {"players": [], "count": 0}

        try:
            from src.identity import PlayerDatabase
            with PlayerDatabase(db_path) as db:
                players = db.list_players()
                # Build team_id -> name lookup
                teams = db.list_teams()
                team_names = {t.team_id: t.name for t in teams}
                enriched = []
                for p in players:
                    d = p.model_dump()
                    d["team_name"] = team_names.get(p.team_id) if p.team_id else None
                    enriched.append(d)
                return {
                    "players": enriched,
                    "count": len(enriched),
                }
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="Identity module not available"
            )

    @app.get("/api/players/{player_id}")
    async def get_player(player_id: int):
        """Get a player with all their appearances."""
        db_path = get_player_db_path()
        if not db_path.exists():
            raise HTTPException(status_code=404, detail="Player database not found")

        try:
            from src.identity import PlayerDatabase
            with PlayerDatabase(db_path) as db:
                player = db.get_player_with_appearances(player_id)
                if player is None:
                    raise HTTPException(status_code=404, detail="Player not found")
                return player.model_dump()
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="Identity module not available"
            )

    @app.patch("/api/players/{player_id}")
    async def update_player(player_id: int, body: UpdatePlayerBody):
        """Update player metadata (name, jersey number, team hint)."""
        db_path = get_player_db_path()
        if not db_path.exists():
            raise HTTPException(status_code=404, detail="Player database not found")

        try:
            from src.identity import PlayerDatabase
            with PlayerDatabase(db_path) as db:
                player = db.update_player(
                    player_id=player_id,
                    name=body.name,
                    jersey_number=body.jersey_number,
                    team_hint=body.team_hint,
                )
                if player is None:
                    raise HTTPException(status_code=404, detail="Player not found")
                # Handle team_id update separately
                if body.team_id is not None:
                    player = db.set_player_team(player_id, body.team_id)
                return {"success": True, "player": player.model_dump()}
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="Identity module not available"
            )

    @app.post("/api/appearances/{video_id}/{track_id}/assign/{player_id}")
    async def assign_appearance(
        video_id: str,
        track_id: int,
        player_id: int,
        body: AssignAppearanceBody,
    ):
        """Manually assign an appearance to a player."""
        db_path = get_player_db_path()
        if not db_path.exists():
            raise HTTPException(status_code=404, detail="Player database not found")

        try:
            from src.identity import PlayerDatabase
            with PlayerDatabase(db_path) as db:
                appearance = db.assign_appearance_to_player(
                    video_id=video_id,
                    track_id=track_id,
                    player_id=player_id,
                    confidence=body.confidence,
                    method="manual",
                )
                if appearance is None:
                    raise HTTPException(status_code=404, detail="Appearance not found")
                return {"success": True, "appearance": appearance.model_dump()}
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="Identity module not available"
            )

    @app.post("/api/players/merge/{keep_id}/{merge_id}")
    async def merge_players(keep_id: int, merge_id: int):
        """Merge two players, keeping one and transferring appearances from the other."""
        db_path = get_player_db_path()
        if not db_path.exists():
            raise HTTPException(status_code=404, detail="Player database not found")

        try:
            from src.identity import PlayerDatabase
            with PlayerDatabase(db_path) as db:
                player = db.merge_players(keep_id, merge_id)
                if player is None:
                    raise HTTPException(
                        status_code=404,
                        detail="One or both players not found"
                    )
                return {
                    "success": True,
                    "kept_player": player.model_dump(),
                    "merged_player_id": merge_id,
                }
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="Identity module not available"
            )

    @app.delete("/api/players/{player_id}")
    async def delete_player(player_id: int):
        """Delete a player and unlink their appearances."""
        db_path = get_player_db_path()
        if not db_path.exists():
            raise HTTPException(status_code=404, detail="Player database not found")

        try:
            from src.identity import PlayerDatabase
            with PlayerDatabase(db_path) as db:
                deleted = db.delete_player(player_id)
                if not deleted:
                    raise HTTPException(status_code=404, detail="Player not found")
                return {"success": True, "player_id": player_id}
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="Identity module not available"
            )

    @app.get("/api/runs/{run_name}/player_assignments")
    async def get_player_assignments(run_name: str):
        """Get player assignments for a specific run."""
        run_path = runs_dir / run_name
        assignments_path = run_path / "player_assignments.json"

        if not assignments_path.exists():
            raise HTTPException(status_code=404, detail="Player assignments not found")

        with open(assignments_path) as f:
            assignments = json.load(f)

        return assignments

    # ── Team CRUD endpoints ────────────────────────────────────────────────

    @app.post("/api/teams")
    async def create_team(body: CreateTeamBody):
        """Create a new team."""
        db_path = get_player_db_path()
        try:
            from src.identity import PlayerDatabase
            with PlayerDatabase(db_path) as db:
                team = db.create_team(name=body.name, short_name=body.short_name)
                return {"success": True, "team": team.model_dump()}
        except Exception as exc:
            if "UNIQUE constraint" in str(exc):
                raise HTTPException(status_code=409, detail=f"Team '{body.name}' already exists")
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/api/teams")
    async def list_teams():
        """List all teams with kits and player counts."""
        db_path = get_player_db_path()
        if not db_path.exists():
            return {"teams": [], "count": 0}
        try:
            from src.identity import PlayerDatabase
            with PlayerDatabase(db_path) as db:
                teams = db.list_teams()
                return {
                    "teams": [t.model_dump() for t in teams],
                    "count": len(teams),
                }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/api/teams/{team_id}")
    async def get_team(team_id: int):
        """Get a team with kits and players."""
        db_path = get_player_db_path()
        if not db_path.exists():
            raise HTTPException(status_code=404, detail="Database not found")
        try:
            from src.identity import PlayerDatabase
            with PlayerDatabase(db_path) as db:
                team = db.get_team(team_id)
                if team is None:
                    raise HTTPException(status_code=404, detail="Team not found")
                kits = db.get_kits_for_team(team_id)
                players = [p for p in db.list_players() if p.team_id == team_id]
                result = team.model_dump()
                result["kits"] = [k.model_dump() for k in kits]
                result["players"] = [p.model_dump() for p in players]
                return result
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.patch("/api/teams/{team_id}")
    async def update_team(team_id: int, body: UpdateTeamBody):
        """Update team metadata."""
        db_path = get_player_db_path()
        if not db_path.exists():
            raise HTTPException(status_code=404, detail="Database not found")
        try:
            from src.identity import PlayerDatabase
            with PlayerDatabase(db_path) as db:
                team = db.update_team(team_id, name=body.name, short_name=body.short_name)
                if team is None:
                    raise HTTPException(status_code=404, detail="Team not found")
                return {"success": True, "team": team.model_dump()}
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.delete("/api/teams/{team_id}")
    async def delete_team(team_id: int):
        """Delete a team (cascade kits, unlink players)."""
        db_path = get_player_db_path()
        if not db_path.exists():
            raise HTTPException(status_code=404, detail="Database not found")
        try:
            from src.identity import PlayerDatabase
            with PlayerDatabase(db_path) as db:
                deleted = db.delete_team(team_id)
                if not deleted:
                    raise HTTPException(status_code=404, detail="Team not found")
                return {"success": True, "team_id": team_id}
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    # ── Kit management endpoints ───────────────────────────────────────────

    @app.post("/api/teams/{team_id}/kits/{kit_type}")
    async def upload_kit(team_id: int, kit_type: str, file: UploadFile = File(...)):
        """Upload a kit image and extract colors."""
        if kit_type not in {"home", "away", "third"}:
            raise HTTPException(status_code=400, detail="kit_type must be home, away, or third")

        db_path = get_player_db_path()
        try:
            from src.identity import PlayerDatabase
            with PlayerDatabase(db_path) as db:
                team = db.get_team(team_id)
                if team is None:
                    raise HTTPException(status_code=404, detail="Team not found")

                # Save uploaded file
                kits_dir = project_root / "data" / "team_kits" / str(team_id)
                kits_dir.mkdir(parents=True, exist_ok=True)
                # Remove any existing kit file (may have different extension)
                for old in kits_dir.glob(f"{kit_type}.*"):
                    old.unlink()
                ext = Path(file.filename).suffix if file.filename else ".jpg"
                dest = kits_dir / f"{kit_type}{ext}"
                content = await file.read()
                dest.write_bytes(content)

                # Extract colors
                import numpy as _np
                dominant_hsv = None
                secondary_hsv = None
                color_hex = None
                secondary_hex = None
                try:
                    from src.vision.team.kit_colors import extract_kit_colors, hsv_to_hex
                    primary, secondary = extract_kit_colors(dest)
                    dominant_hsv = primary.tolist()
                    color_hex = hsv_to_hex(primary)
                    if secondary is not None:
                        secondary_hsv = secondary.tolist()
                        secondary_hex = hsv_to_hex(secondary)
                except Exception:
                    pass

                kit = db.upsert_kit(
                    team_id=team_id,
                    kit_type=kit_type,
                    image_path=str(dest.relative_to(project_root)),
                    dominant_color_hsv=_np.array(dominant_hsv, dtype=_np.float32) if dominant_hsv else None,
                    secondary_color_hsv=_np.array(secondary_hsv, dtype=_np.float32) if secondary_hsv else None,
                    color_hex=color_hex,
                    secondary_color_hex=secondary_hex,
                )
                return {"success": True, "kit": kit.model_dump()}
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/api/teams/{team_id}/kits/{kit_type}/image")
    async def get_kit_image(team_id: int, kit_type: str):
        """Serve kit image file."""
        db_path = get_player_db_path()
        if not db_path.exists():
            raise HTTPException(status_code=404, detail="Database not found")
        try:
            from src.identity import PlayerDatabase
            with PlayerDatabase(db_path) as db:
                kit = db.get_kit(team_id, kit_type)
                if kit is None or kit.image_path is None:
                    raise HTTPException(status_code=404, detail="Kit image not found")
                img_path = project_root / kit.image_path
                if not img_path.exists():
                    # Fallback: look for file with different extension
                    candidates = list(img_path.parent.glob(f"{img_path.stem}.*")) if img_path.parent.exists() else []
                    if not candidates:
                        # Clear stale DB reference
                        db.upsert_kit(team_id=team_id, kit_type=kit_type, image_path=None)
                        raise HTTPException(status_code=404, detail="Kit image file missing")
                    img_path = candidates[0]
                    db.upsert_kit(team_id=team_id, kit_type=kit_type, image_path=str(img_path.relative_to(project_root)))
                return FileResponse(str(img_path))
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.delete("/api/teams/{team_id}/kits/{kit_type}")
    async def delete_kit(team_id: int, kit_type: str):
        """Delete a kit and its image."""
        db_path = get_player_db_path()
        if not db_path.exists():
            raise HTTPException(status_code=404, detail="Database not found")
        try:
            from src.identity import PlayerDatabase
            with PlayerDatabase(db_path) as db:
                kit = db.get_kit(team_id, kit_type)
                if kit and kit.image_path:
                    img_path = project_root / kit.image_path
                    if img_path.exists():
                        img_path.unlink()
                deleted = db.delete_kit(team_id, kit_type)
                if not deleted:
                    raise HTTPException(status_code=404, detail="Kit not found")
                return {"success": True}
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    # ── Team logo endpoints ─────────────────────────────────────────────

    @app.post("/api/teams/{team_id}/logo")
    async def upload_team_logo(team_id: int, file: UploadFile = File(...)):
        """Upload a team logo image."""
        db_path = get_player_db_path()
        try:
            from src.identity import PlayerDatabase
            with PlayerDatabase(db_path) as db:
                team = db.get_team(team_id)
                if team is None:
                    raise HTTPException(status_code=404, detail="Team not found")

                logos_dir = project_root / "data" / "team_logos" / str(team_id)
                logos_dir.mkdir(parents=True, exist_ok=True)
                # Remove any existing logo file (may have different extension)
                for old in logos_dir.glob("logo.*"):
                    old.unlink()
                ext = Path(file.filename).suffix if file.filename else ".png"
                dest = logos_dir / f"logo{ext}"
                content = await file.read()
                dest.write_bytes(content)

                rel_path = str(dest.relative_to(project_root))
                db.set_team_logo(team_id, rel_path)
                return {"success": True, "logo_path": rel_path}
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/api/teams/{team_id}/logo")
    async def get_team_logo(team_id: int):
        """Serve team logo image."""
        db_path = get_player_db_path()
        if not db_path.exists():
            raise HTTPException(status_code=404, detail="Database not found")
        try:
            from src.identity import PlayerDatabase
            with PlayerDatabase(db_path) as db:
                team = db.get_team(team_id)
                if team is None or team.logo_path is None:
                    raise HTTPException(status_code=404, detail="Team logo not found")
                img_path = project_root / team.logo_path
                if not img_path.exists():
                    # Fallback: look for file with different extension
                    candidates = list(img_path.parent.glob(f"{img_path.stem}.*")) if img_path.parent.exists() else []
                    if not candidates:
                        # Clear stale DB reference
                        db.set_team_logo(team_id, None)
                        raise HTTPException(status_code=404, detail="Logo file missing")
                    img_path = candidates[0]
                    db.set_team_logo(team_id, str(img_path.relative_to(project_root)))
                return FileResponse(str(img_path))
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.delete("/api/teams/{team_id}/logo")
    async def delete_team_logo(team_id: int):
        """Delete a team logo."""
        db_path = get_player_db_path()
        if not db_path.exists():
            raise HTTPException(status_code=404, detail="Database not found")
        try:
            from src.identity import PlayerDatabase
            with PlayerDatabase(db_path) as db:
                team = db.get_team(team_id)
                if team is None:
                    raise HTTPException(status_code=404, detail="Team not found")
                if team.logo_path:
                    img_path = project_root / team.logo_path
                    if img_path.exists():
                        img_path.unlink()
                db.set_team_logo(team_id, None)
                return {"success": True}
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    # ── Player photo endpoints ────────────────────────────────────────────

    @app.post("/api/players/{player_id}/photo")
    async def upload_player_photo(player_id: int, file: UploadFile = File(...)):
        """Upload a player photo."""
        db_path = get_player_db_path()
        try:
            from src.identity import PlayerDatabase
            with PlayerDatabase(db_path) as db:
                player = db.get_player(player_id)
                if player is None:
                    raise HTTPException(status_code=404, detail="Player not found")

                photos_dir = project_root / "data" / "player_photos" / str(player_id)
                photos_dir.mkdir(parents=True, exist_ok=True)
                # Remove any existing photo file (may have different extension)
                for old in photos_dir.glob("photo.*"):
                    old.unlink()
                ext = Path(file.filename).suffix if file.filename else ".jpg"
                dest = photos_dir / f"photo{ext}"
                content = await file.read()
                dest.write_bytes(content)

                rel_path = str(dest.relative_to(project_root))
                updated = db.set_player_photo(player_id, rel_path)
                return {"success": True, "player": updated.model_dump()}
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/api/players/{player_id}/photo")
    async def get_player_photo(player_id: int):
        """Serve player photo file."""
        db_path = get_player_db_path()
        if not db_path.exists():
            raise HTTPException(status_code=404, detail="Database not found")
        try:
            from src.identity import PlayerDatabase
            with PlayerDatabase(db_path) as db:
                player = db.get_player(player_id)
                if player is None or player.photo_path is None:
                    raise HTTPException(status_code=404, detail="Player photo not found")
                img_path = project_root / player.photo_path
                if not img_path.exists():
                    # Fallback: look for file with different extension
                    candidates = list(img_path.parent.glob(f"{img_path.stem}.*")) if img_path.parent.exists() else []
                    if not candidates:
                        # Clear stale DB reference so frontend shows placeholder next time
                        db.set_player_photo(player_id, None)
                        raise HTTPException(status_code=404, detail="Player photo file missing")
                    img_path = candidates[0]
                    db.set_player_photo(player_id, str(img_path.relative_to(project_root)))
                return FileResponse(str(img_path))
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.delete("/api/players/{player_id}/photo")
    async def delete_player_photo(player_id: int):
        """Delete a player's photo."""
        db_path = get_player_db_path()
        if not db_path.exists():
            raise HTTPException(status_code=404, detail="Database not found")
        try:
            from src.identity import PlayerDatabase
            with PlayerDatabase(db_path) as db:
                player = db.get_player(player_id)
                if player is None:
                    raise HTTPException(status_code=404, detail="Player not found")
                if player.photo_path:
                    img_path = project_root / player.photo_path
                    if img_path.exists():
                        img_path.unlink()
                db.set_player_photo(player_id, None)
                return {"success": True}
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    # ── Training images + embedding endpoints ────────────────────────────

    @app.post("/api/players/{player_id}/training-images")
    async def upload_training_images(player_id: int, files: list[UploadFile] = File(...)):
        """Upload training images, generate face embeddings, then delete images."""
        db_path = get_player_db_path()
        try:
            from src.identity import PlayerDatabase
            with PlayerDatabase(db_path) as db:
                player = db.get_player(player_id)
                if player is None:
                    raise HTTPException(status_code=404, detail="Player not found")

                player_dir = project_root / "data" / "player_photos" / str(player_id)
                training_dir = player_dir / "training"
                training_dir.mkdir(parents=True, exist_ok=True)

                # Save uploaded files temporarily
                saved = []
                for idx, f in enumerate(files):
                    ext = Path(f.filename).suffix if f.filename else ".jpg"
                    dest = training_dir / f"tmp_{idx:03d}{ext}"
                    content = await f.read()
                    dest.write_bytes(content)
                    saved.append(dest)

                # Generate embeddings for new images, merge with existing pkl
                from src.identity.embedding_generator import add_embeddings_from_images
                pkl_path = player_dir / "embeddings.pkl"
                payload = add_embeddings_from_images(
                    player_id, saved, pkl_path, player_name=player.name,
                )

                # Delete training images — only the pkl is kept
                for img_path in saved:
                    try:
                        img_path.unlink()
                    except OSError:
                        pass
                # Clean up any other leftover images in training dir
                if training_dir.exists():
                    for p in training_dir.iterdir():
                        try:
                            p.unlink()
                        except OSError:
                            pass
                    try:
                        training_dir.rmdir()
                    except OSError:
                        pass

                # Update DB centroid
                if payload["averaged_encoding"] is not None:
                    db.set_player_centroid_direct(
                        player_id, payload["averaged_encoding"], payload["num_encodings"],
                    )
                else:
                    db.set_player_centroid_direct(
                        player_id,
                        [0.0] * 512,
                        0,
                    )

                return {
                    "success": True,
                    "stats": payload["stats"],
                    "embedding_count": payload["num_encodings"],
                    "training_image_count": payload["stats"]["total_images_processed"],
                }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/api/players/{player_id}/training-images")
    async def list_training_images(player_id: int):
        """List embeddings info for a player (images are not kept after training)."""
        db_path = get_player_db_path()
        try:
            from src.identity import PlayerDatabase
            with PlayerDatabase(db_path) as db:
                player = db.get_player(player_id)
                if player is None:
                    raise HTTPException(status_code=404, detail="Player not found")

            pkl_path = project_root / "data" / "player_photos" / str(player_id) / "embeddings.pkl"
            has_embeddings = pkl_path.exists()
            embedding_count = player.embedding_count if player else 0

            # Report per-encoding source image names from pkl metadata
            encodings_info = []
            if has_embeddings:
                from src.identity.embedding_generator import load_embeddings_pkl
                payload = load_embeddings_pkl(pkl_path)
                if payload and payload.get("encodings"):
                    for enc in payload["encodings"]:
                        encodings_info.append({
                            "image_name": enc.get("image_name", "unknown"),
                            "timestamp": enc.get("timestamp"),
                            "model": enc.get("model", "Facenet512"),
                        })

            return {
                "images": [],
                "count": 0,
                "has_embeddings": has_embeddings,
                "embedding_count": embedding_count,
                "encodings": encodings_info,
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/api/players/{player_id}/training-images/{filename}")
    async def serve_training_image(player_id: int, filename: str):
        """Serve a single training image."""
        safe_name = Path(filename).name  # path traversal protection
        img_path = project_root / "data" / "player_photos" / str(player_id) / "training" / safe_name
        if not img_path.exists():
            raise HTTPException(status_code=404, detail="Training image not found")
        return FileResponse(str(img_path))

    @app.delete("/api/players/{player_id}/training-images/{filename}")
    async def delete_training_image(player_id: int, filename: str):
        """Remove a specific embedding by source image name from the pkl."""
        db_path = get_player_db_path()
        try:
            from src.identity import PlayerDatabase
            with PlayerDatabase(db_path) as db:
                player = db.get_player(player_id)
                if player is None:
                    raise HTTPException(status_code=404, detail="Player not found")

                pkl_path = project_root / "data" / "player_photos" / str(player_id) / "embeddings.pkl"
                from src.identity.embedding_generator import load_embeddings_pkl, write_embeddings_pkl
                payload = load_embeddings_pkl(pkl_path)
                if not payload or not payload.get("encodings"):
                    raise HTTPException(status_code=404, detail="No embeddings found")

                safe_name = Path(filename).name
                original_count = len(payload["encodings"])
                payload["encodings"] = [
                    e for e in payload["encodings"]
                    if e.get("image_name") != safe_name
                ]
                if len(payload["encodings"]) == original_count:
                    raise HTTPException(status_code=404, detail="Embedding not found for image")

                # Recompute averaged encoding
                import numpy as np
                if payload["encodings"]:
                    all_vecs = np.stack([e["encoding"] for e in payload["encodings"]])
                    mean_vec = all_vecs.mean(axis=0)
                    norm = float(np.linalg.norm(mean_vec))
                    payload["averaged_encoding"] = (mean_vec / norm).astype(np.float32) if norm > 1e-8 else None
                else:
                    payload["averaged_encoding"] = None
                payload["num_encodings"] = len(payload["encodings"])

                write_embeddings_pkl(payload, pkl_path)

                if payload["averaged_encoding"] is not None:
                    db.set_player_centroid_direct(
                        player_id, payload["averaged_encoding"], payload["num_encodings"],
                    )
                else:
                    db.set_player_centroid_direct(player_id, [0.0] * 512, 0)

                return {
                    "success": True,
                    "embedding_count": payload["num_encodings"],
                }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    # ── Run-team association endpoints ─────────────────────────────────────

    @app.post("/api/runs/{run_name}/teams")
    async def set_run_teams(run_name: str, body: SetRunTeamsBody):
        """Associate home + away teams with a run."""
        db_path = get_player_db_path()
        try:
            from src.identity import PlayerDatabase
            with PlayerDatabase(db_path) as db:
                assocs = db.set_run_teams(
                    run_name, body.home_team_id, body.away_team_id,
                    body.home_kit, body.away_kit,
                )
                return {"success": True, "associations": [a.model_dump() for a in assocs]}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/api/runs/{run_name}/teams")
    async def get_run_teams(run_name: str):
        """Get team associations for a run."""
        db_path = get_player_db_path()
        if not db_path.exists():
            return {"associations": []}
        try:
            from src.identity import PlayerDatabase
            with PlayerDatabase(db_path) as db:
                assocs = db.get_run_teams(run_name)
                # Enrich with team names
                result = []
                for a in assocs:
                    entry = a.model_dump()
                    team = db.get_team(a.team_id)
                    entry["team_name"] = team.name if team else None
                    entry["team_short_name"] = team.short_name if team else None
                    kit = db.get_kit(a.team_id, a.active_kit)
                    entry["color_hex"] = kit.color_hex if kit else None
                    result.append(entry)
                return {"associations": result}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/runs/{run_name}/teams/remap")
    async def remap_run_teams(run_name: str, body: RemapRunTeamsBody):
        """Swap cluster-to-team mapping in teams.json and DB."""
        run_path = runs_dir / run_name
        teams_path = run_path / "teams.json"
        if not teams_path.exists():
            raise HTTPException(status_code=404, detail="teams.json not found for this run")

        with open(teams_path) as f:
            team_info = json.load(f)

        cluster_to_role = team_info.get("cluster_to_role")
        db_team_ids = team_info.get("db_team_ids")
        if not cluster_to_role or not db_team_ids:
            raise HTTPException(status_code=400, detail="No team mapping to swap")

        # Swap roles
        swapped = {}
        for cid_str, role in cluster_to_role.items():
            swapped[cid_str] = "away" if role == "home" else "home"
        team_info["cluster_to_role"] = swapped

        # Swap team names
        home_id = db_team_ids.get("home")
        away_id = db_team_ids.get("away")
        db_team_ids["home"] = away_id
        db_team_ids["away"] = home_id
        team_info["db_team_ids"] = db_team_ids

        # Update team_names mapping
        try:
            from src.identity import PlayerDatabase
            db_path = get_player_db_path()
            with PlayerDatabase(db_path) as db:
                home_team = db.get_team(db_team_ids["home"]) if db_team_ids.get("home") else None
                away_team = db.get_team(db_team_ids["away"]) if db_team_ids.get("away") else None
                for cid_str, role in swapped.items():
                    team = home_team if role == "home" else away_team
                    if team:
                        team_info.setdefault("team_names", {})[cid_str] = team.name
                # Update DB run_teams cluster mapping
                for cid_str, role in swapped.items():
                    db.update_run_team_cluster(run_name, role, int(cid_str))
        except Exception:
            pass

        with open(teams_path, "w") as f:
            json.dump(team_info, f, indent=2)

        return {"success": True, "cluster_to_role": swapped}

    return app


def main(host: str = "127.0.0.1", port: int = 8000, runs_dir: str = "runs"):
    """Run the web server."""
    app = create_app(Path(runs_dir))
    print("\n🚀 Starting Veo Soccer Analysis UI")
    print(f"📂 Runs directory: {runs_dir}")
    print(f"🌐 Open your browser to: http://{host}:{port}\n")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import sys

    # Simple CLI
    runs_dir = sys.argv[1] if len(sys.argv) > 1 else "runs"
    main(runs_dir=runs_dir)
