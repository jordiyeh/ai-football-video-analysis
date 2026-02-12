# Veo-style Soccer Match Analysis

A local-first soccer video analysis system optimized for Apple Silicon that detects players, tracks the ball, identifies teams, and recognizes key events like shots and goals.

**Status**: ✅ Milestones 1-5 complete - Full pipeline with local web UI (M1 MacBook Air)

**Latest**: Pipeline resume after crash — reuse cached artifacts with one-click Resume button

## Features

### ✅ Currently Available (v0.5 - Milestones 1-5)

- **Player & Ball Detection** - YOLOv8-based detection with confidence scores
- **Multi-Object Tracking** - ByteTrack for stable player/ball tracking across frames
- **Team Identification** - Automatic team assignment via jersey color clustering
- **Team Analytics (NEW)** - Possession, territory occupancy, inferred pass-network, and pressing metrics
- **Cross-Match Reporting (NEW)** - Season trends and coach/player report templates across runs
- **Event Detection** - Shot and goal detection with confidence scores and score timeline
- **Automatic Highlight Generation (NEW)** - Select top segments from goals/shots + crowd audio spikes + action intensity
- **Local Web UI** - Interactive timeline, video player, event review, and per-player reel browsing
- **Video Analysis** - Process full 90-minute matches with configurable frame sampling
- **Annotated Overlays** - Videos with team-colored boxes, track IDs, and movement trails
- **Data Export** - Detections, tracks, team assignments, and events in Parquet, CSV, JSONL, and JSON
- **Apple Silicon Optimized** - MPS (Metal Performance Shaders) GPU acceleration
- **CLI Interface** - Rich progress bars and status output
- **Analysis Tools** - Built-in tools to explore detections, tracks, team assignments, and events

### 🆕 Recent Updates (v0.5.10)

- **Player identity persistence** (NEW): Track players across videos using ReID embeddings
  - OSNet-x0.25 ReID model (~2MB, MPS-compatible) extracts 512D player embeddings
  - SQLite database stores player identities with centroid embeddings
  - Automatic player matching: auto (>0.85), suggested (>0.70), new_player (<0.60)
  - Per-run `player_assignments.json` with track-to-player mapping
  - API endpoints: list/get/update players, assign appearances, merge duplicates
  - Configurable via `reid` and `identity` sections in YAML
  - 55 new unit tests for ReID, database, and matching

### 🆕 v0.5.11

- **Highlight engine v1**: Generates ranked highlights from fused event/audio/action signals
  - Emits `highlight_candidates.jsonl`, `highlights.json`, `highlights.csv`, and `highlights_manifest.json`
  - Optional MP4 clip export to `runs/<run>/clips/`
  - Generates per-player reels from fused player assignments + highlight segments
  - Emits `player_highlights.json`, `player_highlights.csv`, and `player_highlights_manifest.json`
  - Optional per-player clips to `runs/<run>/player_clips/`
  - Configurable via new `highlights` section in YAML
- **Player reels in UI**:
  - New endpoints: `GET /api/runs/{run}/player_reels`, `GET /api/runs/{run}/player_reels/{player_id}`
  - Segment clip endpoint: `GET /api/runs/{run}/player_reels/{player_id}/segments/{segment_id}/clip`
  - Alias endpoint: `GET /api/runs/{run}/player_highlights`
  - Export package endpoints: `POST /api/runs/{run}/player_reels/actions/export_package`, `GET /api/runs/{run}/player_reels/exports/{export_name}`
  - Right-panel Player Reels browser with team/score/top-N filters and sorting
  - Click-to-seek segment playback in main video plus optional direct clip playback when `clip_path` exists
  - Export filtered reels as ZIP (JSON + CSV + manifest + optional clip files), optionally restricted to selected player IDs
- **Identity Review in UI**:
  - Endpoint: `GET /api/runs/{run}/identity_review`
  - Suggestions endpoint: `GET /api/runs/{run}/identity_suggestions` (builds `profile_match_suggestions.json` from fused assignments)
  - Apply suggestions endpoint: `POST /api/runs/{run}/identity_suggestions/actions/apply`
  - Apply + preview endpoint: `POST /api/runs/{run}/identity_suggestions/actions/apply_and_preview`
  - Apply + recompute endpoint: `POST /api/runs/{run}/identity_suggestions/actions/apply_and_recompute`
  - Approve preview endpoint: `POST /api/runs/{run}/player_reels/actions/approve_preview`
  - Edit history endpoint: `GET /api/runs/{run}/identity_review/edits`
  - Create player endpoint: `POST /api/players`
  - Single assign/unassign endpoint: `POST /api/runs/{run}/identity_review/actions/assign`
  - Bulk assign endpoint: `POST /api/runs/{run}/identity_review/actions/bulk_assign`
  - Undo last edit endpoint: `POST /api/runs/{run}/identity_review/actions/undo`
  - Undo specific operation endpoint: `POST /api/runs/{run}/identity_review/actions/undo/{op_id}`
  - Recompute endpoint: `POST /api/runs/{run}/player_reels/actions/recompute`
  - Recompute preview endpoint: `POST /api/runs/{run}/player_reels/actions/recompute_preview`
  - Edit player metadata, merge players, review profile/body fusion suggestions, inspect lock/unlock state and multimodal evidence (face/jersey), apply selected suggestions, one-click apply + preview diff / apply + recompute reels, explicitly approve stored previews to persist artifacts, toggle preserve-existing-clips behavior, reassign tracks (single or multi-select bulk), bulk unassign selected tracks, preview reel deltas, inspect assignment audit history, undo the last edit or a specific operation, then recompute reels from updated assignments
- **Profile ingestion stage**: Ingest external player profile bundles for future dynamic tagging
  - Reads profile folders with photos and optional `.pkl` embeddings
  - Emits `profile_registry.json` and `profile_embeddings.parquet`
  - Seeds player identities from profile photos in OSNet embedding space
  - Fuses body-ReID + profile evidence (body_only / profile_only / profile_override / agreement_boost)
  - Emits `profile_player_links.json` and enriched `player_assignments.json` fusion metadata
  - Configurable via `identity.profile_ingestion`
  - Includes ready local config: `configs/profile_ingestion.local.yaml`

### 🆕 v0.5.12

- **Run contract artifacts**:
  - Pipeline now emits `summary.json` (aggregate run counts + score + artifact index + timing summary)
  - Pipeline now emits `ui_index.json` (compact run index for UI discovery/loading)
  - Both artifacts include explicit schema versions

### 🆕 v0.5.13

- **Dynamic tagging quality stage (multimodal)**:
  - Adds Facenet512 profile face embeddings from profile photos (with histogram fallback if model backend is unavailable)
  - Adds jersey-number OCR evidence (when OCR backend is available) and fuses it with body/profile matching
  - Adds substitution-aware identity lock/unlock logic to reduce identity flips on overlapping/conflicting tracks
  - Emits `identity_multimodal_summary.json` for per-run multimodal diagnostics
  - Enriches `player_assignments.json` with lock fields and multimodal fusion metadata

### 🆕 v0.5.14

- **Field normalization stage (norm_xy)**:
  - New pipeline stage computes zoom-aware normalized coordinates from dynamic player-spread viewports
  - Adds `image_xy` and `norm_xy` fields to track rows for tactical analytics that are more stable under camera zoom
  - Emits `field_normalization.json` and `field_viewports.parquet` artifacts with schema/config/summary

### 🆕 v0.5.15

- **Team analytics stage**:
  - New pipeline stage computes team-level metrics from tracks + optional fused assignments
  - Possession: nearest-player ball ownership timeline with smoothed team possession shares
  - Territory: normalized occupancy/centroid/spread summaries and zone-control shares
  - Pass network: inferred same-team carrier transitions with pass edges and counts
  - Pressing: defender-to-carrier proximity pressure timeline and high-press episode metrics
  - Emits `team_analytics.json`, `team_possession_timeline.csv`, `team_territory_zones.csv`, `team_pass_network.csv`, and `team_pressing_timeline.csv`

### 🆕 v0.5.17

- **Pipeline resume after crash**:
  - New "Resume" button in UI for failed/cancelled jobs — reuses the same run directory so all cached stage artifacts are preserved
  - `POST /api/pipeline/jobs/{job_id}/resume` endpoint with conflict checking
  - OverlayStage now supports resume caching (skips re-render when `overlay.mp4` already exists)
  - Atomic writes for Parquet and JSONL artifacts — prevents corrupted partial files on crash
  - Fixed context reconstitution gaps in PlayerIdentityStage, PlayerHighlightReelsStage, and CrossMatchReportingStage cache paths

### 🆕 v0.5.16

- **Cross-match reporting/export stage**:
  - Aggregates multiple runs in the same `runs/` root into season trend summaries
  - Produces coach-ready report template with trend prompts and recent-window metrics
  - Produces player report templates from per-player reel trends across matches
  - Emits `cross_match_report.json`, `cross_match_match_trends.csv`, `cross_match_player_trends.csv`, `coach_report_template.md`, and `player_report_templates.md`
- **Season Trends in UI**:
  - New panel to browse season-level match aggregates, team trends, top players, and recent-window trajectories
  - New endpoints:
    - `GET /api/runs/{run}/cross_match`
    - `GET /api/runs/{run}/cross_match/artifacts/{artifact_id}`
    - `POST /api/runs/{run}/cross_match/actions/export_package`
    - `GET /api/runs/{run}/cross_match/exports/{export_name}`

### v0.5.9

- **Celebration detection**: Detect player celebrations as goal confirmation signal
  - Arms-up pose detection via bbox aspect ratio changes
  - Group huddle detection via player convergence analysis
  - Integrated as new weighted signal (0.15) in shot fusion engine
  - Works when ball tracking fails at critical moments
  - Configurable via `events.alternative_shot.celebration` in YAML
- **Ball-specific detection model**: Specialized soccer ball detector for dramatically improved ball detection
  - `BallSpecialistDetector` wraps HuggingFace soccer ball model (~6MB, auto-downloads)
  - `DetectorEnsemble` combines YOLO + specialist with Weighted Box Fusion (WBF)
  - Confidence boost when multiple detectors agree on same ball
  - Enable with `--config configs/ball_specialist.yaml`
  - Backward compatible (disabled by default)
- **Visual goal region detection**: Replaces hardcoded 15% edge margins with visual detection
  - Hough line transforms detect crossbars and goal lines
  - HSV white threshold detects goalposts
  - Temporal smoothing with outlier rejection handles camera movement
  - Graceful fallback to heuristic when visual detection fails
  - Configurable via `events.goal_region` in YAML
- **Event confirmation UI**: Approve/reject auto-detected events, add manual events at any timestamp
  - Status badges (pending/confirmed/rejected) on all events
  - User corrections stored in `events_confirmed.jsonl` (original data preserved)
  - Manual events with dashed border styling
  - Persistent across sessions
- **Alternative shot detection**: Detect shots even with sparse ball data (works with <1% ball detection rate!)
  - Kick detection: finds ball near player foot regions
  - Goal area entry: detects ball entering goal zones
  - Goalkeeper dive detection: identifies diving motion from bbox changes
  - Player clustering: analyzes attacking formations
  - Celebration detection: confirms goals via player reactions
  - Multi-signal fusion with weighted confidence scoring
- **Ball detection boosting**: Multi-scale detection, temporal filtering, and candidate tracking to dramatically improve ball detection rate
  - Multi-scale (0.5x, 1.0x, 1.5x) catches small balls that single-scale misses
  - Temporal filter rejects single-frame spurious detections
  - Candidate tracker with Kalman prediction bridges short occlusions
  - Lower confidence threshold (0.15 default) with intelligent filtering
- **Soft-NMS merging**: Improved detection merging across scales without hard elimination
- **Configurable ball settings**: New `ball:` section in YAML configs for fine-tuning
- **Physics-based ball interpolation**: Extended from 45 frames to 300+ frames using Kalman filter with bidirectional blending
- **Resume mode**: Skip completed stages with `--resume` flag
- **Dynamic overlay**: UI renders overlay in JavaScript (original video + canvas)

### 🆕 v0.5.8: Performance Profiling

- **Per-stage timing in manifest**: `run_manifest.json` (schema 1.1) now includes detailed timing metrics
  - Duration, items processed, items/sec for each pipeline stage
  - Custom metrics per stage (detector type, detection counts, etc.)
  - Device info (mps/cuda/cpu) and platform details
- **Profiling script**: `scripts/profile_pipeline.py` for deep performance analysis
  - cProfile integration with `.prof` output and summary
  - py-spy flame graph generation (SVG)
  - Profile analysis tools
- **Timing summary**: Pipeline now prints percentage breakdown by stage
- **Documented optimizations**: See `docs/profiling.md` for optimization opportunities

### 📋 Planned

- Field keypoint detection for normalization
- Jersey number OCR and player identification
- See `docs/ENHANCEMENTS.md` for full roadmap
- See `docs/FEATURE_ROADMAP.md` for expanded team/player analytics, dynamic tagging from profile photos, and highlight generation planning

## System Requirements

- macOS with Apple Silicon (M1-M4) recommended
- Python 3.11 or higher
- 8GB+ RAM (16GB recommended for full matches)

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd ai_video_analysis
```

### 2. Create virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -e ".[dev]"
```

This will install:
- PyTorch with MPS support for Apple Silicon
- YOLOv8 (Ultralytics)
- OpenCV for video I/O
- All other required dependencies

### 4. Verify installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}')"
```

You should see:
```
PyTorch: 2.x.x
MPS available: True
```

## Quick Start

### Basic Usage

Analyze a soccer video with default settings:

```bash
veo-analyze --video path/to/match.mp4 --output runs/my_analysis
```

### With Custom Configuration

```bash
veo-analyze --video path/to/match.mp4 --output runs/my_analysis --config configs/custom.yaml
```

### Resume from Cache

Skip completed stages (useful for iterating on later stages):

```bash
veo-analyze --video path/to/match.mp4 --output runs/my_analysis --resume
```

### Skip Overlay Rendering

The overlay stage is slow (~40 min for 90-min match). Use dynamic UI rendering instead:

```bash
veo-analyze --video path/to/match.mp4 --output runs/my_analysis --no-overlay
```

### Using Improved Detection Config

For better event detection (lower thresholds, longer interpolation):

```bash
veo-analyze --video path/to/match.mp4 --output runs/my_analysis --config configs/improved_detection.yaml
```

### Using Profile Ingestion Config (Your Team Pictures)

The repo includes a local config preset at `configs/profile_ingestion.local.yaml`:

```bash
veo-analyze \
  --video path/to/match.mp4 \
  --output runs/my_analysis \
  --config configs/profile_ingestion.local.yaml
```

### Output Structure

After running analysis, you'll find:

```
runs/my_analysis/
├── run_manifest.json       # Configuration snapshot and runtime info
├── video_metadata.json     # Video properties (fps, resolution, duration)
├── detections.parquet      # All detections with bbox, confidence, timestamps
├── tracks.parquet          # Stable tracks with IDs and team assignments
├── teams.json              # Team colors and assignments
├── field_normalization.json # Field-view normalization config + summary
├── field_viewports.parquet # Per-frame dynamic viewport bounds used for norm_xy
├── team_analytics.json     # Team-level possession/territory/pass-network/pressing summary
├── team_possession_timeline.csv # Frame-level ball ownership timeline
├── team_territory_zones.csv # Team occupancy and zone-control ratios
├── team_pass_network.csv   # Inferred pass edges between carriers
├── team_pressing_timeline.csv # Per-frame pressing pressure observations
├── cross_match_report.json # Season trends aggregated across run directories
├── cross_match_match_trends.csv # Flat per-match trend export
├── cross_match_player_trends.csv # Flat per-player season trend export
├── coach_report_template.md # Coach report template with season metrics
├── player_report_templates.md # Player report templates for top players
├── profile_registry.json   # Ingested player profile metadata (if enabled)
├── profile_embeddings.parquet # Normalized profile embeddings (if enabled)
├── profile_player_links.json # Profile→player mapping from seeding/fusion (if enabled)
├── player_assignments.json # Track-to-player identity mapping (ReID)
├── events.jsonl            # Detected events (shots, goals) with confidence
├── events_confirmed.jsonl  # User confirmations/rejections/manual events (UI)
├── score_timeline.json     # Score progression with timestamps
├── summary.json            # Aggregate run counts, score snapshot, and artifact index
├── ui_index.json           # Compact run index for UI discovery/loading
├── identity_multimodal_summary.json # Multimodal identity diagnostics (face/OCR/locking)
├── highlight_candidates.jsonl # Candidate highlight triggers with reasons/scores
├── highlights.json         # Selected highlight segments
├── highlights.csv          # Flat CSV export of selected highlight segments
├── highlights_manifest.json # Highlight config and summary
├── player_highlights.json  # Per-player reel segments from fused assignments
├── player_highlights.csv   # Flat CSV export of per-player reel segments
├── player_highlights_manifest.json # Player reel config and summary
├── clips/                  # Optional rendered highlight mp4 clips
├── player_clips/           # Optional rendered per-player clip mp4s
└── overlay.mp4             # Annotated video with team-colored boxes, IDs, and trails
```

### Working with Detection Data

The `detections.parquet` file contains all player and ball detections:

```python
import pandas as pd

# Load detections
df = pd.read_parquet("runs/my_analysis/detections.parquet")

# Columns: object_type, bbox, center, confidence, class_id,
#          width, height, area, frame_idx, timestamp

# Example queries
players = df[df.object_type == 'player']
ball = df[df.object_type == 'ball']
high_confidence = df[df.confidence > 0.8]

# Export to CSV for Excel/Numbers
df.to_csv("detections.csv", index=False)
```

Or use the built-in analysis tool:

```bash
python explore_detections.py
```

This generates:
- `detections.csv` - Full export for spreadsheets
- `ball_detections.csv` - Ball-only tracking data
- `frame_summary.csv` - Per-frame statistics
- Analysis of detection quality and patterns

### Working with Track Data

The `tracks.parquet` file contains stable tracks with persistent IDs:

```python
import pandas as pd

# Load tracks
df = pd.read_parquet("runs/my_analysis/tracks.parquet")

# Columns: track_id, frame_idx, timestamp, object_type, bbox,
#          confidence, age, hits, time_since_update,
#          image_x, image_y, image_xy, norm_x, norm_y, norm_xy, norm_source

# Example queries
unique_players = df[df.object_type == 'player']['track_id'].nunique()
ball_trajectory = df[df.object_type == 'ball']
long_tracks = df.groupby('track_id').size()[lambda x: x > 100]

# Export to CSV
df.to_csv("tracks.csv", index=False)
```

Or use the built-in track analysis tool:

```bash
python explore_tracks.py runs/my_analysis
```

This generates:
- `tracks.csv` - Full track export
- `track_summary.csv` - Per-track statistics (length, quality, etc.)
- `ball_tracks.csv` - Ball trajectory data
- `player_trajectories.csv` - Player positions over time for heatmaps
- Track quality analysis (fragmentation, coverage)

### Testing Multi-Object Tracking

Run a full analysis with tracking enabled:

```bash
# Activate virtual environment
source .venv/bin/activate

# Run analysis with fast config (recommended for first test)
python -m src.cli \
  --video "path/to/match.mp4" \
  --output runs/tracking_test \
  --config configs/fast_test.yaml
```

The pipeline will:
1. **Ingest** - Extract video metadata
2. **Detect** - Find all players and ball in sampled frames
3. **Track** - Associate detections into stable tracks with IDs
4. **Team Assignment** - Cluster jersey colors into team labels
5. **Field Normalization** - Compute zoom-aware `norm_xy` coordinates
6. **Overlay** - Render video with bounding boxes, track IDs, and trails

Then analyze the tracking results:

```bash
python explore_tracks.py runs/tracking_test
```

Expected output:
- Unique tracks for each player visible in the video
- Stable track IDs maintained across frames
- Track trails showing player movement
- Quality metrics showing fragmentation and coverage

### Working with Team Data

The `teams.json` file contains team assignments and colors:

```python
import json
import pandas as pd

# Load team info
with open("runs/my_analysis/teams.json") as f:
    teams = json.load(f)

# Load tracks with team assignments
df = pd.read_parquet("runs/my_analysis/tracks.parquet")

# Analyze by team
team_a = df[df.team_name == 'team_A']
team_b = df[df.team_name == 'team_B']

print(f"Team A players: {team_a.track_id.nunique()}")
print(f"Team B players: {team_b.track_id.nunique()}")

# Export by team
team_a.to_csv("team_A_tracks.csv", index=False)
team_b.to_csv("team_B_tracks.csv", index=False)
```

Or use the built-in team analysis tool:

```bash
python explore_teams.py runs/my_analysis
```

This generates:
- `tracks_team_A.csv` - Team A player tracks
- `tracks_team_B.csv` - Team B player tracks
- `team_summary.csv` - Per-team statistics
- Team balance analysis and consistency checks

### Working with Event Data

The `events.jsonl` file contains detected shots and goals with confidence scores:

```python
import json

# Load events
events = []
with open("runs/my_analysis/events.jsonl", "r") as f:
    for line in f:
        events.append(json.loads(line))

# Filter by event type
shots = [e for e in events if e["event_type"] == "shot"]
goals = [e for e in events if e["event_type"] == "goal"]

# High confidence events
high_conf = [e for e in events if e["confidence"] > 0.8]

print(f"Total shots: {len(shots)}")
print(f"Total goals: {len(goals)}")
```

The `score_timeline.json` shows score progression:

```python
import json

with open("runs/my_analysis/score_timeline.json", "r") as f:
    timeline = json.load(f)

print(f"Final score: {timeline['final_score']}")
print(f"Total goals: {timeline['goals']}")

# Goal timestamps
for entry in timeline['timeline']:
    print(f"{entry['timestamp']:.1f}s - {entry['score']}")
```

Or use the built-in event analysis tool:

```bash
python explore_events.py runs/my_analysis
```

This generates:
- `events.csv` - Full event export
- `shots.csv` - Shot events only
- `goals.csv` - Goal events only
- `event_timeline.csv` - Timeline for visualization
- Event statistics and confidence analysis

### Working with Highlights

The highlight stage creates ranked segments from:
- event signals (goals/shots)
- crowd audio spikes (RMS + robust z-score)
- action intensity (ball speed burst + nearby player pressure)

```python
import json

with open("runs/my_analysis/highlights.json", "r") as f:
    highlights = json.load(f)

print("Segments:", len(highlights["segments"]))
for seg in highlights["segments"][:5]:
    print(
        f"{seg['segment_id']} "
        f"{seg['start_time']:.1f}-{seg['end_time']:.1f}s "
        f"score={seg['score']:.2f} reasons={seg['reasons']}"
    )
```

If `highlights.export.save_clips=true`, rendered clips are saved to `runs/my_analysis/clips/`.

### Working with Per-Player Reels

Per-player reels are generated by combining:
- selected highlight segments (`highlights.json`)
- fused player assignments (`player_assignments.json`)
- player presence/activity inside each segment

```python
import json

with open("runs/my_analysis/player_highlights.json", "r") as f:
    player_reels = json.load(f)

for player in player_reels["players"][:3]:
    print(player["player_id"], player.get("player_name"), player["segment_count"])
    for seg in player["segments"][:2]:
        print(
            f"  {seg['segment_id']} "
            f"{seg['start_time']:.1f}-{seg['end_time']:.1f}s "
            f"player_score={seg['player_segment_score']:.2f}"
        )
```

If `highlights.player_reels.save_clips=true`, per-player clips are exported to `runs/my_analysis/player_clips/`.

## Using the Web UI

After running analysis, launch the local web interface to review events:

```bash
# Start the UI server
python run_ui.py

# Or specify a custom runs directory
python run_ui.py path/to/runs
```

The UI will open at http://localhost:8000 with:

**Features:**
- Browse all analysis runs
- Interactive video player with overlay
- Click events to jump to that moment
- Browse per-player reels (from fused player assignments + highlight segments)
- Click any player segment to seek/play that exact window in the main video
- Filter reels by team, min score, top-N segments, and sort order
- Play exported per-player clip files directly when available
- Browse season trends across runs and download cross-match report artifacts/ZIP packages
- Review identity assignments (player metadata + track mapping)
- Create players for split corrections, merge duplicates, and reassign tracks
- Multi-select track assignments and bulk-assign to a target player or bulk-unassign to `None`
- Recompute `player_highlights.json` directly from updated identity assignments
- Visual timeline with shot/goal markers
- Score display and event confidence
- Frame-accurate seeking
- Event confirmation (approve/reject auto-detected events)
- Manual event addition at current video position

**Usage:**
1. Click a run from the list to load it
2. Video plays with team-colored boxes and track IDs
3. Click events in the right panel to seek to that moment
4. Click markers on the timeline to jump to events
5. Review confidence scores and event details
6. Click ✓ to confirm or ✗ to reject auto-detected events
7. Click "+ Add Event" to manually add shots or goals at current timestamp
8. Use "Player Reels" to jump directly to player-specific highlight segments

**Event Confirmation:**
- Pending events show approve/reject buttons
- Confirmed events are slightly dimmed with green "confirmed" badge
- Rejected events are strikethrough with orange "rejected" badge
- Manual events have dashed border and can be deleted
- All corrections saved to `events_confirmed.jsonl` (original data preserved)

The UI automatically loads:
- Video overlay (`overlay.mp4`)
- Detected events (`events.jsonl`)
- Score timeline (`score_timeline.json`)
- Per-player reels (`player_highlights.json`)
- Run metadata

## Configuration

Edit `configs/default.yaml` to customize:

### Video Settings

```yaml
video:
  sampling_strategy: "every_frame"  # or "every_2nd", "every_nth"
  sampling_interval: 1              # for "every_nth"
  target_fps: null                  # optional: resample to specific fps
```

### Detection Settings

```yaml
detection:
  model_name: "yolov8x.pt"         # x = extra large (best accuracy)
  device: "mps"                     # mps, cuda, or cpu
  confidence_threshold: 0.5         # minimum confidence for player detections
  batch_size: 8                     # frames per batch
  # Ball-specific settings for improved detection
  ball:
    confidence_threshold: 0.15      # lower threshold for small ball
    enable_multiscale: true         # detect at multiple scales
    scales: [0.5, 1.0, 1.5]         # scale factors
    enable_temporal_filter: true    # reject spurious detections
    enable_candidate_tracking: true # soft-track before committing
  # Ball specialist model (optional, disabled by default)
  ball_specialist:
    enabled: false                  # enable for better ball detection
    model_source: "keremberke/yolov8n-soccer-ball-detection"
  # Detector ensemble (combines YOLO + specialist)
  ensemble:
    enabled: false                  # enable when using ball_specialist
    fusion_type: "wbf"              # weighted box fusion
```

Or use the pre-configured ball specialist config:
```bash
veo-analyze --video match.mp4 --output runs/test --config configs/ball_specialist.yaml
```

### Field Normalization Settings

```yaml
field:
  enabled: true
  min_players_per_frame: 6
  player_percentile_low: 0.10
  player_percentile_high: 0.90
  margin_ratio: 0.12
  smoothing_alpha: 0.25
  min_viewport_width_ratio: 0.35
  min_viewport_height_ratio: 0.35
  clip_norm: true
```

### Team Analytics Settings

```yaml
team_analytics:
  enabled: true
  use_norm_coordinates: true
  possession_max_ball_distance_px: 140.0
  possession_smoothing_frames: 3
  possession_min_stable_frames: 3
  possession_min_segment_frames: 4
  pass_min_gap_seconds: 0.15
  pass_max_gap_seconds: 2.5
  territory_x_bins: 3
  territory_y_bins: 3
  pressure_radius_norm: 0.10
  high_press_threshold: 0.65
  high_press_min_frames: 8
```

### Cross-Match Reporting Settings

```yaml
cross_match:
  enabled: true
  runs_root: null               # defaults to parent of current run directory
  include_current_run: true
  max_runs: 60
  last_n_window: 5
  top_players: 15
  min_player_segment_score: 0.25
```

### Profile Ingestion Settings

```yaml
identity:
  profile_ingestion:
    enabled: true
    profiles_root: "/Users/yehj10/iCloud/vs/download_icloud_images/team_pictures"
    recursive_image_scan: false
    enable_body_embedding_seed: true
    max_images_per_profile_for_reid: 5
    profile_match_enabled: true
    profile_match_auto_threshold: 0.82
    profile_match_suggest_threshold: 0.68
  multimodal:
    enabled: true
    face:
      enabled: true
      min_track_support_frames: 2
      suggest_threshold: 0.68
    jersey_ocr:
      enabled: true
      min_ocr_confidence: 0.45
      min_track_support_frames: 2
    locking:
      enabled: true
      lock_confidence_threshold: 0.82
      overlap_conflict_frames: 45
      substitution_gap_frames: 150
```

### Highlight Settings

```yaml
highlights:
  enabled: true
  event:
    include_goals: true
    include_shots: true
  audio:
    enabled: true
    min_z_score: 2.0
  action:
    enabled: true
    score_quantile: 0.9
  segment:
    pre_roll_seconds: 8.0
    post_roll_seconds: 12.0
    top_n: 20
  export:
    save_clips: false
  player_reels:
    enabled: true
    max_segments_per_player: 8
    min_presence_seconds: 1.5
    min_player_segment_score: 0.2
    min_assignment_confidence: 0.6
    include_suggested_assignments: true
    save_clips: false
```

### Overlay Settings

```yaml
overlay:
  bbox_thickness: 2
  show_confidence: true
  show_track_ids: true
  player_color: "#00FF00"          # green
  ball_color: "#FF0000"            # red
  trail_length: 30                 # frames
```

## Development

### Running Tests

```bash
# All tests (400+ tests, ~60% coverage)
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Golden/regression tests
pytest tests/golden/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Generate HTML coverage report
pytest tests/ --cov=src --cov-report=html
open coverage_html/index.html
```

**Test Coverage**: ~60% (3,345 lines covered)
- Unit tests: ByteTrack, Kalman filter, event detection, team clustering, colors, pipeline, video reader
- Integration tests: detection-to-tracking flow, team assignment pipeline, event detection pipeline
- Golden tests: regression testing for deterministic outputs

### Code Formatting

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/
```

### Type Checking

```bash
mypy src/
```

## Performance Tips

### For Full Matches (90 minutes)

Use the included `configs/fast_test.yaml` for quick testing:

```bash
veo-analyze --video match.mp4 --output runs/test --config configs/fast_test.yaml
```

This configuration:
- Samples every 10th frame (~10x speedup)
- Uses YOLOv8m (medium model, good balance)
- Batch size 16 for efficiency
- MPS GPU acceleration

### Custom Optimization

1. **Adjust frame sampling** to balance speed vs detail:
   ```yaml
   video:
     sampling_strategy: "every_nth"
     sampling_interval: 10  # every 10th frame = ~10x faster
   ```

2. **Choose model size** based on accuracy needs:
   ```yaml
   detection:
     model_name: "yolov8m.pt"  # m=medium, x=extra-large, n=nano
   ```

3. **Increase batch size** if you have enough memory:
   ```yaml
   detection:
     batch_size: 16  # Higher = faster but more memory
   ```

### Tested Performance (Real-world Results)

**M1 MacBook Air - 96-minute match (173K frames)**

With `configs/fast_test.yaml` (every 10th frame, YOLOv8m, MPS):
- **Processing time**: 49 minutes
- **Frames analyzed**: 17,285 (sampled)
- **Detections**: 175,399 (174,784 players, 615 ball)
- **Average**: ~10 players per frame
- **Ball detection rate**: 3.6% of frames

### Expected Times by Model & Sampling

| Model | Sampling | M1 Air | M1 Pro/Max | Notes |
|-------|----------|--------|------------|-------|
| YOLOv8m | every 10th | ~50 min | ~35 min | Recommended for fast testing |
| YOLOv8m | every frame | ~8 hours | ~5 hours | Best for production |
| YOLOv8x | every frame | ~12 hours | ~7 hours | Maximum accuracy |
| YOLOv8n | every frame | ~4 hours | ~2.5 hours | Fast but less accurate |

*Times for 90-minute matches at 30 FPS source. Add 30% for overlay rendering.*

## Known Limitations

### Ball Detection Rate

Ball detection can be challenging due to:
- Small ball size relative to frame
- YOLOv8's generic "sports ball" class (not soccer-specific)
- Veo camera zoom/motion blur

**Mitigation**:
- v0.5.3 adds ball detection boosting with multi-scale detection, temporal filtering, and candidate tracking
- v0.5.7 adds specialized soccer ball detector with ensemble fusion for significantly improved detection
- Use `configs/ball_specialist.yaml` for best ball detection (requires `pip install huggingface_hub`)
- Use `configs/improved_detection.yaml` for aggressive settings without specialist model

**Impact**: Event detection accuracy depends on ball tracking quality. Physics-based interpolation (300+ frames) bridges gaps.

### Goal Region Detection

Goal regions use visual detection with heuristic fallback (v0.5.6+):
- **Visual detection**: Hough lines for crossbars, HSV threshold for goalposts
- **Temporal smoothing**: Handles camera movement and zoom
- **Fallback**: Uses heuristic (top/bottom 15%) when visual detection confidence is low

**Limitations**:
- Visual detection works best with clearly visible white goalposts
- Assumes goals are at top/bottom edges (broadcast-style view)
- May fall back to heuristic in low-contrast or heavily zoomed footage

### Virtual Environment

The `.venv` symlinks may break after Homebrew Python updates. See CLAUDE.md for workarounds.

## Troubleshooting

### MPS Backend Issues

**Note**: MPS (GPU) works reliably on M1-M4 Macs with PyTorch 2.9+. Model initialization can take 1-2 minutes - this is normal, not hanging.

If you experience crashes or errors:

1. Fallback to CPU (slower but stable):
   ```yaml
   detection:
     device: "cpu"
   ```

2. Update PyTorch:
   ```bash
   pip install --upgrade torch torchvision
   ```

3. Check Activity Monitor - high CPU usage during "detection" stage means it's working, not stuck

### Memory Issues

For large videos:

1. Reduce batch size:
   ```yaml
   detection:
     batch_size: 4
   ```

2. Use frame sampling:
   ```yaml
   video:
     sampling_strategy: "every_2nd"
   ```

### Video Codec Issues

If overlay video won't play:

1. Try different codec:
   ```yaml
   export:
     video_codec: "avc1"  # or "H264", "XVID"
   ```

2. Convert with FFmpeg:
   ```bash
   ffmpeg -i overlay.mp4 -c:v libx264 overlay_h264.mp4
   ```

## Project Structure

```
ai_video_analysis/
├── src/
│   ├── cli.py              # Command-line interface
│   ├── config/             # Configuration schemas
│   ├── pipeline/           # Pipeline orchestration
│   ├── video/              # Video I/O
│   ├── vision/
│   │   ├── detect/         # Player/ball detection
│   │   │   ├── yolo.py     # YOLOv8 detector
│   │   │   ├── ball_specialist.py  # Soccer ball specialist
│   │   │   ├── ensemble.py # Detector ensemble + WBF
│   │   │   └── ball_boost.py # Multi-scale + temporal filtering
│   │   ├── track/          # Multi-object tracking (ByteTrack)
│   │   ├── team/           # Team identification (color clustering)
│   │   ├── field/          # Goal region detection
│   │   └── reid/           # Player re-identification (OSNet)
│   ├── identity/           # Player identity persistence (SQLite)
│   ├── events/             # Shot/goal detection & ball trajectory
│   ├── ui/                 # Web UI server and static files
│   └── export/             # Overlay rendering & exports
├── configs/                # YAML configurations
├── tests/                  # Unit and integration tests
├── docs/                   # Enhancement docs and specs
├── data/samples/           # Sample videos (gitignored)
├── runs/                   # Output artifacts (gitignored)
└── models/                 # Cached model weights (gitignored)
```

## Roadmap

### ✅ Milestone 1: "Hello World" (v0.1 - Completed)
- ✅ Video ingestion and metadata extraction
- ✅ Player and ball detection with YOLOv8
- ✅ MPS (GPU) acceleration for Apple Silicon
- ✅ Annotated video overlay generation
- ✅ CLI interface with rich progress bars
- ✅ Parquet/CSV data export
- ✅ Detection analysis tools
- ✅ Tested on full 96-minute match

### ✅ Milestone 2: "It Tracks" (v0.2 - Completed)
- ✅ ByteTrack multi-object tracking implementation
- ✅ Stable track IDs across frames with Kalman filtering
- ✅ Track quality metrics (age, hits, fragmentation)
- ✅ Track trails in overlay visualization
- ✅ Handle occlusions and tentative/confirmed tracks
- ✅ Track analysis and export tools

### ✅ Milestone 3: "It Knows Teams" (v0.3 - Completed)
- ✅ Jersey color extraction from player bounding boxes
- ✅ K-means clustering for team separation (HSV color space)
- ✅ Automatic team assignment to tracks
- ✅ Team-colored overlays and labels in video
- ✅ Team analysis tools and export by team
- ✅ Team consistency validation

### ✅ Milestone 4: "It Detects Events" (v0.4 - Completed)
- ✅ Ball trajectory analysis with velocity and direction
- ✅ Shot detection (ball velocity + trajectory towards goal)
- ✅ Goal detection (ball in goal region after shot)
- ✅ Score timeline with confidence scores
- ✅ Events JSONL export with metadata
- ✅ Event analysis and export tools

### ✅ Milestone 5: "It Has a UI" (v0.5 - Completed)
- ✅ FastAPI backend server with REST API
- ✅ Local web interface (HTML/CSS/JS)
- ✅ Interactive timeline with event markers (shots, goals)
- ✅ Video player with frame-accurate seeking
- ✅ Event list with click-to-seek
- ✅ Score display and confidence indicators
- ✅ Event confirmation and editing (approve/reject, manual add)
- [ ] Export and sharing functionality (deferred to v0.6)

### Milestone 6: "It's Production Ready"
- [ ] Export functionality from UI
- [x] Caching and resumable pipeline
- [x] Error recovery and validation
- [x] Performance profiling and optimization
- [x] Golden regression test suite
- [ ] Comprehensive documentation

## Contributing

This is a research project following the "vibe coding" philosophy with heavy caching for fast iteration. See `AGENTS.md` for detailed architecture and engineering rules.

## License

TBD

## Acknowledgments

- Built with [YOLOv8](https://github.com/ultralytics/ultralytics) by Ultralytics
- Inspired by [Veo](https://www.veo.co/) soccer camera technology
- Tracking algorithms: ByteTrack, BoT-SORT, DeepSORT
