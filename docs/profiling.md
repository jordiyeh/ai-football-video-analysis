# Performance Profiling Guide

This document describes how to profile the video analysis pipeline and interpret the results.

## Quick Start

### View Timing in Run Manifest

After any pipeline run, timing metrics are automatically saved to `run_manifest.json`:

```bash
# Run the pipeline
PYTHONPATH=".venv/lib/python3.11/site-packages" /opt/homebrew/bin/python3.11 src/cli.py \
  --video match.mp4 --output runs/test

# View timing breakdown
cat runs/test/run_manifest.json | python -m json.tool | grep -A 30 '"metrics"'
```

### Run cProfile Profiling

For detailed function-level profiling:

```bash
PYTHONPATH=".venv/lib/python3.11/site-packages" /opt/homebrew/bin/python3.11 \
  scripts/profile_pipeline.py \
  --video data/samples/short_clip.mp4 \
  --output runs/profile \
  --mode cprofile
```

This generates:
- `profile.prof` - Binary profile data
- `profile_summary.txt` - Human-readable summary

### Generate Flame Graph

For visual profiling with py-spy (requires installation):

```bash
pip install py-spy

PYTHONPATH=".venv/lib/python3.11/site-packages" /opt/homebrew/bin/python3.11 \
  scripts/profile_pipeline.py \
  --video data/samples/short_clip.mp4 \
  --output runs/profile \
  --mode flamegraph
```

This generates:
- `flamegraph.svg` - Visual flame graph
- `profile.speedscope.json` - Speedscope-compatible profile

View the speedscope profile at https://www.speedscope.app/

## Understanding the Metrics

### Run Manifest Metrics (schema 1.1)

The `run_manifest.json` includes a `metrics` section:

```json
{
  "schema_version": "1.1",
  "metrics": {
    "total_duration_seconds": 120.5,
    "stages": {
      "ingest": {
        "start_time": "2024-01-15T10:30:00",
        "end_time": "2024-01-15T10:30:02",
        "duration_seconds": 2.1,
        "items_processed": 1,
        "items_per_second": 0.47,
        "custom_metrics": {
          "fps": 30.0,
          "total_frames": 5400
        }
      },
      "detection": {
        "duration_seconds": 95.2,
        "items_processed": 5400,
        "items_per_second": 56.7,
        "custom_metrics": {
          "detector_type": "yolo",
          "ball_detections": 4200,
          "player_detections": 108000
        }
      }
    },
    "device": "mps",
    "python_version": "3.11.9"
  }
}
```

### Stage Timing Breakdown

The pipeline prints a timing summary after completion:

```
Timing Summary (total: 2m 0s)
  detection       ==================== 95.2s (79.3%)
  overlay         ===                  18.3s (15.2%)
  tracking        =                    4.2s (3.5%)
  team_assignment                      1.5s (1.3%)
  event_detection                      1.2s (1.0%)
  ingest                               0.1s (0.1%)
```

## Interpreting Results

### What to Look For

1. **Stage Breakdown**: Identify which stage takes the most time
2. **Items Per Second**: Compare throughput across runs
3. **Custom Metrics**: Understand stage-specific details

### Common Bottlenecks

| Stage | Typical % | Optimization Opportunity |
|-------|-----------|-------------------------|
| Detection | 60-80% | Batch processing, smaller model |
| Overlay | 10-25% | Skip overlay, use --no-overlay |
| Tracking | 2-5% | Reduce track buffer |
| Team Assignment | 1-3% | Reduce sampled frames |
| Event Detection | 1-2% | Usually fast |

## Optimization Opportunities

Based on profiling, here are known optimization paths:

### 1. Batch Detection (High Impact)

The `YOLODetector.detect_batch()` method exists but is currently unused. Enabling batch processing could provide 1.5-3x speedup:

```python
# In DetectionStage.run():
batch_size = 8  # Adjust based on GPU memory
detections = detector.detect_batch(frames, batch_size)
```

**Trade-off**: Incompatible with temporal filtering (requires sequential processing).

### 2. Skip Overlay Generation

Use `--no-overlay` flag to skip video rendering:

```bash
python src/cli.py --video match.mp4 --output runs/test --no-overlay
```

The UI can dynamically render overlays from track data.

### 3. Hardware Acceleration

Ensure MPS (Apple Silicon) or CUDA is being used:

```python
# Check device in manifest
cat runs/test/run_manifest.json | grep '"device"'
# Should show: "device": "mps" or "device": "cuda"
```

### 4. Sampling Strategy

Reduce frame processing rate:

```yaml
# In config YAML
video:
  sampling_strategy: "interval"
  sampling_interval: 2  # Process every 2nd frame
```

### 5. Smaller Detection Model

Use a faster YOLO model:

```yaml
detection:
  model_name: "yolov8n.pt"  # nano (fastest)
  # vs "yolov8m.pt" (medium) or "yolov8x.pt" (extra large)
```

## Analyzing Profile Data

### Load Profile in Python

```python
import pstats

stats = pstats.Stats("runs/profile/profile.prof")
stats.sort_stats("cumulative")
stats.print_stats(30)  # Top 30 functions
```

### Find Hot Functions

```python
# Functions called >1000 times with significant time
stats.sort_stats("tottime")
stats.print_stats("detect", 10)  # Filter by name
```

### Compare Profiles

```python
import pstats

stats1 = pstats.Stats("runs/baseline/profile.prof")
stats2 = pstats.Stats("runs/optimized/profile.prof")

# Compare totals
print(f"Baseline: {stats1.total_tt:.1f}s")
print(f"Optimized: {stats2.total_tt:.1f}s")
```

## Profiling Tips

1. **Use Short Clips**: Profile on 10-30 second clips for quick iteration
2. **Disable Cache**: Add `--no-cache` or delete cache directory for accurate timing
3. **Multiple Runs**: Profile multiple times to account for variance
4. **Isolate Changes**: Profile before and after optimization changes
5. **Check Device**: Ensure GPU is being used (check manifest metrics)
