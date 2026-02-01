"""FastAPI server for local web UI."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
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


class AssignAppearanceBody(BaseModel):
    """Request body for assigning an appearance to a player."""
    confidence: float = 1.0


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
    app = FastAPI(
        title="Veo Soccer Analysis UI",
        description="Local web interface for soccer match analysis",
        version="0.4.0",
    )

    # Serve static files (HTML, JS, CSS)
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def root():
        """Serve main UI page."""
        index_path = static_dir / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return HTMLResponse(
            content="<h1>Veo Soccer Analysis UI</h1><p>Frontend not found. Check src/ui/static/</p>"
        )

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

            # Load basic info
            run_info = {
                "name": run_path.name,
                "path": str(run_path),
                "has_manifest": manifest_path.exists(),
                "has_events": events_path.exists(),
                "has_timeline": timeline_path.exists(),
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

            runs.append(run_info)

        return {"runs": runs}

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

        return result

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
            # Expand bbox array into separate columns
            bbox_expanded = pd.DataFrame(df["bbox"].tolist(), index=df.index, columns=["x1", "y1", "x2", "y2"])
            df = pd.concat([df.drop("bbox", axis=1), bbox_expanded], axis=1)

        # Filter out rows with NaN or Inf in coordinates (invalid tracks from Kalman filter)
        coord_cols = ["x1", "y1", "x2", "y2"]
        for col in coord_cols:
            if col in df.columns:
                df = df[~(df[col].isna() | np.isinf(df[col]))]

        # Replace any remaining NaN/Inf values with None for JSON serialization
        df = df.replace([np.inf, -np.inf], None)
        df = df.where(pd.notna(df), None)

        # Convert numpy types to Python types for JSON serialization
        def convert_types(val):
            if val is None:
                return None
            elif isinstance(val, (np.integer, np.floating)):
                if np.isnan(val) or np.isinf(val):
                    return None
                return float(val)
            elif isinstance(val, np.ndarray):
                return val.tolist()
            return val

        # Convert to records for JSON
        tracks = df.to_dict(orient="records")
        tracks = [{k: convert_types(v) for k, v in track.items()} for track in tracks]

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

    # Player identity endpoints

    def get_player_db_path() -> Path:
        """Get path to player database."""
        # Check for database in runs directory first
        db_path = runs_dir / "players.db"
        if db_path.exists():
            return db_path
        # Fallback to current directory
        return Path("players.db")

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
                return {
                    "players": [p.model_dump() for p in players],
                    "count": len(players),
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

    return app


def main(host: str = "127.0.0.1", port: int = 8000, runs_dir: str = "runs"):
    """Run the web server."""
    app = create_app(Path(runs_dir))
    print(f"\n🚀 Starting Veo Soccer Analysis UI")
    print(f"📂 Runs directory: {runs_dir}")
    print(f"🌐 Open your browser to: http://{host}:{port}\n")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import sys

    # Simple CLI
    runs_dir = sys.argv[1] if len(sys.argv) > 1 else "runs"
    main(runs_dir=runs_dir)
