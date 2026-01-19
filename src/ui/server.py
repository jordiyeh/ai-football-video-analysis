"""FastAPI server for local web UI."""

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn


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
        """Get all events for a specific run."""
        run_path = runs_dir / run_name
        events_path = run_path / "events.jsonl"

        if not events_path.exists():
            raise HTTPException(status_code=404, detail="Events not found")

        events = []
        with open(events_path) as f:
            for line in f:
                events.append(json.loads(line))

        return {"events": events, "count": len(events)}

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
    async def get_video(run_name: str):
        """Stream the overlay video."""
        run_path = runs_dir / run_name
        video_path = run_path / "overlay.mp4"

        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Video not found")

        return FileResponse(
            video_path,
            media_type="video/mp4",
            headers={"Accept-Ranges": "bytes"},
        )

    @app.get("/api/runs/{run_name}/tracks")
    async def get_tracks(run_name: str, limit: int = 1000):
        """Get track data for a specific run (with pagination)."""
        import pandas as pd

        run_path = runs_dir / run_name
        tracks_path = run_path / "tracks.parquet"

        if not tracks_path.exists():
            raise HTTPException(status_code=404, detail="Tracks not found")

        # Load tracks (limit for performance)
        df = pd.read_parquet(tracks_path)

        # Sample if too large
        if len(df) > limit:
            df = df.sample(n=limit, random_state=42).sort_values("frame_idx")

        return {
            "tracks": df.to_dict(orient="records"),
            "count": len(df),
            "total": len(pd.read_parquet(tracks_path)),
        }

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
