# Veo-style Soccer Match Analysis

A local-first soccer video analysis system optimized for Apple Silicon that detects players, tracks the ball, identifies teams, and recognizes key events like shots and goals.

## Features (v1)

- Player and ball detection using YOLOv8
- Real-time tracking with visualization
- Team identification via jersey color clustering
- Shot and goal detection
- Annotated video overlay generation
- Local web UI for event review
- Multiple export formats (Parquet, JSONL, CSV)

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
├── run_manifest.json       # Configuration and metadata
├── video_metadata.json     # Video properties (fps, resolution, etc.)
├── detections.parquet      # Per-frame detections
├── overlay.mp4            # Annotated video with bounding boxes
└── summary.json           # Aggregate statistics (TODO)
```

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

1. **Use frame sampling** to reduce processing time:
   ```yaml
   video:
     sampling_strategy: "every_2nd"  # Process every 2nd frame
   ```

2. **Lower model size** for faster inference:
   ```yaml
   detection:
     model_name: "yolov8m.pt"  # medium model (faster than yolov8x)
   ```

3. **Increase batch size** if you have enough memory:
   ```yaml
   detection:
     batch_size: 16  # default is 8
   ```

### Expected Processing Times (M1 Max)

- **YOLOv8x**: ~15-20 FPS (3-4 hours for full match)
- **YOLOv8m**: ~30-40 FPS (1.5-2 hours for full match)
- **YOLOv8n**: ~60-80 FPS (~1 hour for full match)

*Times with every_frame sampling at 30 FPS source video*

## Troubleshooting

### MPS Backend Issues

If you see MPS-related errors:

1. Fallback to CPU:
   ```yaml
   detection:
     device: "cpu"
   ```

2. Update PyTorch:
   ```bash
   pip install --upgrade torch torchvision
   ```

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
│   │   ├── track/          # Multi-object tracking (TODO)
│   │   └── team/           # Team identification (TODO)
│   ├── events/             # Shot/goal detection (TODO)
│   └── export/             # Overlay rendering & exports
├── configs/                # YAML configurations
├── tests/                  # Unit and integration tests
├── data/samples/           # Sample videos (gitignored)
├── runs/                   # Output artifacts (gitignored)
└── models/                 # Cached model weights (gitignored)
```

## Roadmap

### Milestone 1: "Hello World" (Current)
- ✅ Video ingestion and validation
- ✅ Player and ball detection with YOLOv8
- ✅ Annotated video overlay generation
- ✅ CLI interface with progress bars

### Milestone 2: "It Tracks"
- [ ] ByteTrack implementation
- [ ] Stable track IDs across frames
- [ ] Track quality metrics
- [ ] Track trails in overlay

### Milestone 3: "It Knows Teams"
- [ ] Jersey color clustering
- [ ] Team assignment logic
- [ ] Team-colored overlays
- [ ] Manual correction UI

### Milestone 4: "It Detects Events"
- [ ] Shot detection (ball velocity heuristics)
- [ ] Goal detection (goal region analysis)
- [ ] Score timeline generation
- [ ] Events JSONL export

### Milestone 5: "It Has a UI"
- [ ] FastAPI backend
- [ ] Local web interface
- [ ] Timeline with event markers
- [ ] Video player with jump-to functionality

### Milestone 6: "It's Polished"
- [ ] Error recovery and validation
- [ ] Performance optimization
- [ ] Golden regression tests
- [ ] Documentation and examples

## Contributing

This is a research project following the "vibe coding" philosophy with heavy caching for fast iteration. See `AGENTS.md` for detailed architecture and engineering rules.

## License

TBD

## Acknowledgments

- Built with [YOLOv8](https://github.com/ultralytics/ultralytics) by Ultralytics
- Inspired by [Veo](https://www.veo.co/) soccer camera technology
- Tracking algorithms: ByteTrack, BoT-SORT, DeepSORT
