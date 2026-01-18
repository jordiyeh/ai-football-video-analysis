# AGENTS.md — Veo-style Soccer Match Analysis (Local-First, Apple Silicon)

This repo analyzes a full soccer match from a Veo-recorded MP4 (ball-following, may zoom) and produces:
- player + ball tracking in pixel space
- team identification (our team vs opponent)
- high-priority events: shots and goals with score timeline
- a local UI (timeline + player) plus export options
Metadata (spreadsheet) can optionally assist with team colors, rosters, jersey numbers, and names.

## Principles
- **Quality first** for v1 (accuracy/robustness > speed).
- **Local-first**: run on Apple Silicon (M-series) using PyTorch MPS where possible.
- **Confidence + provenance** for every detection/event.
- **Duplicate proven patterns first** (Veo/pro-league style features) before novel analytics.
- **Cache everything** to support “vibe coding” iteration loops.

---

## Target platforms
- Primary: macOS on Apple Silicon (M1–M4). Use MPS backend for PyTorch where possible.
- Secondary: CPU-only fallback (slower, but functional).
- Docker: optional; avoid making it mandatory.

---

## Repo layout (required)
- `src/`
  - `cli.py` entrypoint
  - `config/` config schemas + defaults
  - `pipeline/` stage orchestration + caching
  - `video/` decode/encode, frame sampling, clip extraction
  - `vision/`
    - `detect/` player/ball detectors
    - `track/` MOT backends (ByteTrack/BoT-SORT/DeepSORT interchangeable)
    - `team/` jersey-color clustering + team assignment
    - `field/` (phase 2) keypoints/lines detection for stabilization/normalization
  - `events/` shot/goal inference + scoring timeline
  - `export/` JSONL/CSV/MP4 overlay writers
  - `ui/` local web app (FastAPI + static frontend)
- `configs/` YAML configs
- `data/samples/` tiny clips for tests (do not commit full matches)
- `runs/` outputs (gitignored)
- `models/` weights cache (gitignored)
- `tests/` unit + golden tests
- `docs/` feature specs and decisions

---

## Run contract (artifacts)
Each analysis run MUST produce:
- `run_manifest.json` (config snapshot + git commit + environment info)
- `detections.parquet` (or `.jsonl`) with per-frame detections
- `tracks.parquet` with trajectories and track quality metrics
- `events.jsonl` (one event per line: shots, goals, etc.)
- `score_timeline.json` (time-indexed score and confidence)
- `summary.json` (aggregate stats)
- `overlay.mp4` (annotated video)
- `ui_index.json` (index for UI to load this run)

All artifacts must include schema versioning.

---

## Definitions (do not change without migration)
- `t`: timestamp in seconds (float)
- `frame_idx`: integer
- `track_id`: stable within a match (best effort; may fragment)
- `team_id`: `ours` / `opponent` (mapped from colors or roster order)
- `confidence`: float [0, 1]
- Coordinates:
  - `image_xy`: pixel coords
  - (phase 2+) `norm_xy`: zoom/field-normalized coords

---

## Pipeline stages (v1)
### Stage A — Ingest
- Validate MP4, extract fps/duration, build frame index.
- Sampling strategy must be configurable (e.g., analyze every frame vs every N frames).

### Stage B — Detect
- Detect players + ball with open-source weights.
- Store raw detections with confidences.

### Stage C — Track
- Multi-object tracking for players + ball.
- Output smoothed tracks, gaps filled when reasonable.
- Track quality score per track.

### Stage D — Team assignment
- Jersey color clustering → `team_id`.
- Provide UI hooks to correct team labeling.

### Stage E — Events (v1)
- Shots (priority): inferred from ball speed/trajectory near penalty area (pixel heuristics first).
- Goals: inferred from ball entering goal region + subsequent restart patterns; store confidence and allow UI confirmation.
- Score timeline updated with confidence.

### Stage F — Export + UI
- Overlay render with toggles: ball, player boxes, track trails, team colors, labels.
- Local UI: timeline of events, click to jump, filter, export buttons.

---

## Feature backlog (keep as TODO checklists)
### Phase 1 (v1)
- [TODO] High-quality ball detection + tracking
- [TODO] Team clustering + “ours vs opponent” mapping
- [TODO] Shot detection + confidence
- [TODO] Goal detection + score timeline + confidence + UI confirm
- [TODO] Solid local UI (timeline + player + exports)

### Phase 2
- [TODO] Automatic field keypoints/lines for zoom-aware normalization/stabilization
- [TODO] Improved goal-mouth localization
- [TODO] Better possession segmentation

### Phase 3
- [TODO] Spreadsheet metadata import (rosters, numbers, names, colors, faces)
- [TODO] Semi-auto `track_id → player` mapping
- [TODO] Jersey number OCR assist
- [TODO] Persistent identity across videos (embedding DB + confirmation)

### Phase 4
- [TODO] Telestration tools + saved annotations
- [TODO] Heatmaps, pass networks, tactical summaries
- [TODO] Auto highlights and clip templates

---

## Engineering rules for agents
1. **Implement end-to-end slices** over isolated components.
2. Every new stage must:
   - write a versioned artifact
   - be cacheable and resumable
3. Add at least one test (unit or golden) for each non-trivial change.
4. Never block the pipeline on optional metadata.
5. Cloud calls must be opt-in and abstracted behind interfaces.

### Interfaces (keep clean)
- `Detector.detect(frames) -> Detections`
- `Tracker.update(detections) -> Tracks`
- `TeamAssigner.assign(tracks, frames) -> team_labels`
- `EventInferencer.infer(tracks, context) -> events + score_timeline`

---

## Metadata (spreadsheet) — v1 format proposal
Spreadsheet columns (minimum):
- `team` (ours/opponent)
- `number` (int, optional)
- `first_name` (optional)
- `last_name` (optional)
- `notes` (optional)

Optional:
- `primary_color` / `secondary_color` (hex)
- `face_image_path` (local path)

Metadata ingestion must output a normalized `metadata.json`.

---

## Definition of Done (v1)
Given a Veo MP4, the system:
- runs locally on Apple Silicon
- produces overlay video, events JSONL, score timeline, summary JSON
- exposes a local UI to review shots/goals and export results
