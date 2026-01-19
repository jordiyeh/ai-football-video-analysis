# Veo-style Soccer Match Analysis

A local-first soccer video analysis system optimized for Apple Silicon that detects players, tracks the ball, identifies teams, and recognizes key events like shots and goals.

**Status**: ✅ Milestones 1-5 complete - Full pipeline with local web UI (M1 MacBook Air)

## Features

### ✅ Currently Available (v0.5 - Milestones 1-5)

- **Player & Ball Detection** - YOLOv8-based detection with confidence scores
- **Multi-Object Tracking** - ByteTrack for stable player/ball tracking across frames
- **Team Identification** - Automatic team assignment via jersey color clustering
- **Event Detection** - Shot and goal detection with confidence scores and score timeline
- **Local Web UI** - Interactive timeline, video player, and event review interface
- **Video Analysis** - Process full 90-minute matches with configurable frame sampling
- **Annotated Overlays** - Videos with team-colored boxes, track IDs, and movement trails
- **Data Export** - Detections, tracks, team assignments, and events in Parquet, CSV, JSONL, and JSON
- **Apple Silicon Optimized** - MPS (Metal Performance Shaders) GPU acceleration
- **CLI Interface** - Rich progress bars and status output
- **Analysis Tools** - Built-in tools to explore detections, tracks, team assignments, and events

### 📋 Planned

- Event confirmation and editing in UI
- Field keypoint detection for normalization
- Jersey number OCR and player identification

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

### Output Structure

After running analysis, you'll find:

```
runs/my_analysis/
├── run_manifest.json       # Configuration snapshot and runtime info
├── video_metadata.json     # Video properties (fps, resolution, duration)
├── detections.parquet      # All detections with bbox, confidence, timestamps
├── tracks.parquet          # Stable tracks with IDs and team assignments
├── teams.json              # Team colors and assignments
├── events.jsonl            # Detected events (shots, goals) with confidence
├── score_timeline.json     # Score progression with timestamps
└── overlay.mp4            # Annotated video with team-colored boxes, IDs, and trails
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
#          confidence, age, hits, time_since_update

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
4. **Overlay** - Render video with bounding boxes, track IDs, and trails

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
- Visual timeline with shot/goal markers
- Score display and event confidence
- Frame-accurate seeking

**Usage:**
1. Click a run from the list to load it
2. Video plays with team-colored boxes and track IDs
3. Click events in the right panel to seek to that moment
4. Click markers on the timeline to jump to events
5. Review confidence scores and event details

The UI automatically loads:
- Video overlay (`overlay.mp4`)
- Detected events (`events.jsonl`)
- Score timeline (`score_timeline.json`)
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
  confidence_threshold: 0.5         # minimum confidence for detections
  batch_size: 8                     # frames per batch
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
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

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
│   │   ├── track/          # Multi-object tracking (ByteTrack)
│   │   └── team/           # Team identification (color clustering)
│   ├── events/             # Shot/goal detection & ball trajectory
│   └── export/             # Overlay rendering & exports
├── configs/                # YAML configurations
├── tests/                  # Unit and integration tests
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
- [ ] Event confirmation and editing (deferred to v0.6)
- [ ] Export and sharing functionality (deferred to v0.6)

### Milestone 6: "It's Production Ready"
- [ ] Event confirmation and manual editing in UI
- [ ] Export functionality from UI
- [ ] Caching and resumable pipeline
- [ ] Error recovery and validation
- [ ] Performance profiling and optimization
- [ ] Golden regression test suite
- [ ] Comprehensive documentation

## Contributing

This is a research project following the "vibe coding" philosophy with heavy caching for fast iteration. See `AGENTS.md` for detailed architecture and engineering rules.

## License

TBD

## Acknowledgments

- Built with [YOLOv8](https://github.com/ultralytics/ultralytics) by Ultralytics
- Inspired by [Veo](https://www.veo.co/) soccer camera technology
- Tracking algorithms: ByteTrack, BoT-SORT, DeepSORT
