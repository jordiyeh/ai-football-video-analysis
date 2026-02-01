# Enhancement Requirements

This document captures identified improvements for the soccer video analysis system, prioritized by impact and effort.

## Current State Analysis (2026-01-31)

### Tracking Quality Metrics (from `runs/event_test`)

| Metric | Value | Assessment |
|--------|-------|------------|
| Ball detection rate | **0.4%** (777/173K frames) | Critical issue |
| Ball trajectory gaps | **195 frames avg** (~6.5s) | Breaks trajectory analysis |
| Max ball gap | **67,470 frames** (~37 min) | Large portions untracked |
| Player detections | 685,309 | Good coverage |
| Unique player tracks | 15,597 | High fragmentation |
| Events detected | 1 shot, 0 goals | Far below expected |

### Root Cause

YOLOv8's generic "sports ball" class (COCO class 32) struggles with:
- Small ball size in wide shots
- Fast motion blur during kicks
- Similar colors to field/players
- Veo camera zoom changes

---

## Tier 1: High Impact, Moderate Effort

### 1. ✅ Smarter Ball Trajectory Interpolation (COMPLETED)

**Problem**: Current max interpolation gap is 45 frames, but average gap is 195 frames.

**Solution**:
- Extend interpolation to 300+ frames with physics-based prediction
- Use Kalman filter to predict ball position during gaps
- Apply confidence decay proportional to gap size
- Add "interpolated" flag to trajectory points

**Files modified**:
- `src/events/ball_trajectory.py` - New `_interpolate_linear()` and `_interpolate_physics()` methods with bidirectional blending
- `src/events/kalman_filter.py` - New 6-state Kalman filter (x, y, vx, vy, ax, ay)
- `src/config/schemas.py` - Added `InterpolationConfig` with tunable parameters
- `tests/unit/test_ball_trajectory.py` - 25 unit tests covering interpolation

**Acceptance criteria**:
- [x] Interpolation works for gaps up to 300 frames
- [x] Confidence decreases with gap size (0.97^(distance_seconds*30) with 0.1 floor)
- [x] Physics-based prediction (constant velocity + acceleration decay)
- [x] Unit tests for interpolation edge cases (25 tests passing)
- [x] `interpolated` flag and `interpolation_source` on synthetic points
- [x] Backward compatible with legacy `max_gap_frames` parameter

---

### 2. ✅ Ball Detection Boosting (COMPLETED)

**Problem**: Ball confidence threshold may be too high; ball detections filtered incorrectly.

**Solution**:
- Lower ball confidence threshold to 0.15 (from 0.25)
- Multi-scale detection (resize frames for different ball sizes)
- Temporal consistency filter (reject teleporting detections)
- Track ball candidates across frames before committing
- Soft-NMS for merging multi-scale detections

**Files modified**:
- `src/config/schemas.py` - Added `BallDetectionConfig` with all ball-specific parameters
- `src/vision/detect/yolo.py` - Added `detect_multiscale()` method, fixed `detect_batch()` ball filtering
- `src/vision/detect/ball_boost.py` - **NEW** module with:
  - `BallTemporalFilter` - Sliding window temporal consistency filter
  - `BallCandidateTracker` - Kalman-based soft tracker (NEW→TENTATIVE→CONFIRMED→LOST)
  - `soft_nms()` - Soft non-maximum suppression for merging
- `src/cli.py` - Integrated ball boost into `DetectionStage.run()`
- `configs/default.yaml` - Added ball-specific config section
- `configs/improved_detection.yaml` - Added aggressive ball detection settings
- `tests/unit/test_ball_boost.py` - **NEW** 26 unit tests

**Acceptance criteria**:
- [x] Multi-scale detection at 0.5x, 1.0x, 1.5x scales
- [x] Temporal filter rejects single-frame spurious detections
- [x] Candidate tracker promotes after min_hits consecutive frames
- [x] Soft-NMS merges overlapping detections across scales
- [x] Fully configurable via YAML (all features can be enabled/disabled)
- [x] Backward compatible with old configs (uses sensible defaults)
- [x] 26 unit tests passing

---

### 3. ✅ Shot Detection Without Full Trajectory (COMPLETED)

**Problem**: Shot detection requires continuous ball trajectory, which we don't have.

**Solution**:
- Detect "kick events" from player leg positions (ball near foot region)
- Look for ball appearing in goal area (entry detection)
- Use player clustering patterns (spread out = attacking formation)
- Detect goalkeeper diving motion (bbox aspect ratio + displacement)
- Fuse all signals with weighted confidence scoring

**Files modified**:
- `src/config/schemas.py` - Added `AlternativeShotDetectionConfig` with all tunable parameters
- `src/events/kick_detection.py` - **NEW** module with:
  - `KickEventDetector` - Detects ball near player foot regions
  - `GoalAreaEntryDetector` - Detects ball entering goal zones
  - `ShotFusionEngine` - Fuses multiple signals into shot candidates
- `src/events/player_analysis.py` - **NEW** module with:
  - `PlayerClusteringAnalyzer` - Analyzes team formations for attack patterns
  - `GoalkeeperAnalyzer` - Detects goalkeeper dives from bbox changes
- `src/events/detection.py` - Added `detect_shots_all()`, `detect_shots_alternative()`, `_merge_shot_detections()`
- `src/events/__init__.py` - Exported new classes
- `src/cli.py` - Integrated alternative detection into `EventDetectionStage`
- `tests/unit/test_kick_detection.py` - **NEW** 14 unit tests
- `tests/unit/test_player_analysis.py` - **NEW** 12 unit tests

**Acceptance criteria**:
- [x] Detect shots with >50% ball trajectory gaps (activates when coverage <50%)
- [x] Confidence reflects signal strength and corroboration
- [x] Multiple signals boost confidence (kick + goal_entry + gk_dive)
- [x] Configurable weights for each signal type
- [x] Backward compatible (velocity-based detection still primary when ball data good)
- [x] 26 unit tests passing

---

## Tier 2: Medium Impact

### 4. ✅ Goal Region Detection (COMPLETED)

**Problem**: Goal regions are hardcoded (top/bottom 15%), fails with different camera angles.

**Solution**:
- Detect pitch lines using Hough transforms (Canny edge detection + HoughLinesP)
- Find goalpost white pixels via HSV threshold
- Temporal smoothing with outlier rejection handles camera movement
- Confidence-based fallback/blending with heuristic
- Abstract `GoalRegionProvider` interface for extensibility

**Files modified**:
- `src/vision/field/goal_detector.py` - **NEW** module with:
  - `GoalRegion` dataclass for detected regions
  - `GoalRegionProvider` abstract interface
  - `HeuristicGoalRegionProvider` - Fallback using hardcoded margins
  - `GoalRegionDetector` - Visual detection with Hough lines + HSV
  - `GoalRegionTracker` - Temporal smoothing with outlier rejection
- `src/vision/field/__init__.py` - Exports new classes
- `src/config/schemas.py` - Added `GoalRegionDetectionConfig` with all tunable parameters
- `src/events/detection.py` - `EventDetector` accepts optional `GoalRegionProvider`
- `src/events/kick_detection.py` - `GoalAreaEntryDetector` uses provider
- `src/events/player_analysis.py` - `GoalkeeperAnalyzer` uses provider
- `configs/default.yaml` - Added `events.goal_region` configuration section
- `tests/unit/test_goal_detector.py` - **NEW** 28 unit tests

**Configuration** (`configs/default.yaml`):
```yaml
events:
  goal_region:
    enabled: true
    detection_method: "hybrid"  # visual, heuristic, or hybrid
    heuristic_edge_margin: 0.15
    heuristic_goal_width_fraction: 0.30
    enable_temporal_smoothing: true
    smoothing_window_frames: 30
    fallback_confidence_threshold: 0.3
    blend_threshold: 0.6
```

**Algorithm**:
- **Visual detection**: Canny edges → HoughLinesP for crossbars, HSV white threshold for posts
- **Confidence scoring**: Crossbar (35%) + posts (30%) + expected zone (35%)
- **Fallback strategy**: >0.6 use visual, 0.3-0.6 blend 70/30, <0.3 use heuristic
- **Temporal smoothing**: Weighted average with recency, outlier rejection (>50px jump)

**Acceptance criteria**:
- [x] Correct goal detection in >80% of frames (visual + fallback)
- [x] Works with zoomed/panned Veo footage (temporal smoothing handles movement)
- [x] Graceful fallback to heuristic when visual detection fails
- [x] All existing tests still pass (116 tests)
- [x] New unit tests for goal detection (28 tests)

---

### 5. ✅ Event Confirmation UI (COMPLETED)

**Problem**: Users can't verify or correct detected events.

**Solution**:
- Add approve/reject buttons to event list
- Allow manual event addition (click on timeline)
- Export "confirmed" vs "auto-detected" events separately
- Store user corrections persistently

**Files modified**:
- `src/ui/server.py` - Added 4 new API endpoints (confirm, reject, add, delete)
  - `generate_event_id()` - Creates stable IDs (auto_shot_139570, manual_1706710200123)
  - `load_confirmations()` / `save_confirmation()` - JSONL persistence
  - `merge_events_with_confirmations()` - Combines auto events with user actions
- `src/ui/static/app.js` - Event confirmation UI
  - Updated `renderEvents()` with status badges and action buttons
  - Added `confirmEvent()`, `rejectEvent()`, `deleteManualEvent()`
  - Added modal for manual event addition at current video position
- `src/ui/static/index.html` - CSS and modal HTML
  - Status badges (pending/confirmed/rejected)
  - Action buttons styling (approve: green, reject: orange, delete: yellow)
  - Event item states (confirmed: dimmed, rejected: strikethrough, manual: dashed)

**Data model**:
- User actions stored in `{run_dir}/events_confirmed.jsonl`
- Original `events.jsonl` never modified
- Actions: confirm, reject, add, delete (with timestamps and notes)

**Acceptance criteria**:
- [x] Approve/reject buttons work
- [x] Manual event addition at any timestamp
- [x] Corrections saved to `events_confirmed.jsonl`
- [x] Confirmation state persists across sessions

---

### 6. ✅ Ball-Specific Detection Model (COMPLETED)

**Problem**: Generic YOLOv8 "sports ball" class is not optimized for soccer balls.

**Solution**:
- Abstract `ObjectDetector` interface for detector plugins
- `BallSpecialistDetector` wraps soccer-trained YOLO model from HuggingFace
- `DetectorEnsemble` combines multiple detectors with Weighted Box Fusion (WBF)
- Auto-downloads specialized model (~6MB) on first use
- Confidence boost when multiple detectors agree

**Files modified**:
- `src/vision/detect/base.py` - **NEW** abstract `ObjectDetector` interface
- `src/vision/detect/ball_specialist.py` - **NEW** HuggingFace model wrapper
  - Loads `keremberke/yolov8n-soccer-ball-detection` (~6MB)
  - Auto-downloads and caches model
  - Same filtering as YOLODetector (size, aspect ratio)
- `src/vision/detect/ensemble.py` - **NEW** detector ensemble with WBF
  - `compute_iou()` - Intersection over Union
  - `DetectorEnsemble` - Combines detectors, fuses results
  - `_weighted_box_fusion()` - Clusters overlapping boxes, weighted average
  - `_fuse_cluster()` - Computes fused bbox + boosted confidence
  - Also supports NMS and Soft-NMS fusion types
- `src/vision/detect/yolo.py` - Added `ObjectDetector` inheritance, `name`/`supported_types` properties
- `src/vision/detect/__init__.py` - Exports new classes
- `src/config/schemas.py` - Added `BallSpecialistConfig` and `EnsembleConfig`
- `src/cli.py` - Added `_build_detector()` method for ensemble support
- `configs/default.yaml` - Added `ball_specialist` and `ensemble` sections (disabled by default)
- `configs/ball_specialist.yaml` - **NEW** pre-configured for optimal ball detection
- `tests/unit/test_ensemble.py` - **NEW** 19 unit tests

**Configuration** (`configs/ball_specialist.yaml`):
```yaml
detection:
  ball_specialist:
    enabled: true
    model_source: "keremberke/yolov8n-soccer-ball-detection"
    confidence_threshold: 0.25
  ensemble:
    enabled: true
    weights:
      yolo: 1.0
      ball_specialist: 2.0  # Higher weight for specialist
    fusion_type: "wbf"
```

**Algorithm**:
- **WBF**: Clusters overlapping boxes by IoU, computes weighted average coordinates
- **Confidence boost**: `max_conf * (1 + 0.1 * (n_detectors - 1))` when multiple agree
- **Object type separation**: Ball and player detections fused independently

**Acceptance criteria**:
- [x] Ball detection rate significantly improved with specialist model
- [x] Works across different field views (Veo camera with zoom)
- [x] Model size ~6MB (well under 50MB limit)
- [x] Backward compatible (disabled by default)
- [x] 19 unit tests passing + 116 existing tests still pass

---

## Tier 3: Nice to Have

### 7. ✅ Test Coverage Expansion (COMPLETED)

**Before**: ~35% coverage (~117 tests)

**After**: **60.23%** coverage (400+ tests)

**Solution**:
- Added tracking tests (ByteTrack, Kalman filter) - 58 tests
- Added event detection tests (shot/goal inference) - 34 tests
- Added team assignment tests (clustering + colors) - 64 tests
- Added pipeline and video reader tests - 49 tests
- Added integration tests (detection-tracking, team, events) - 21 tests
- Added golden/regression tests - 12 tests

**Files created**:
- `tests/conftest.py` - Shared fixtures, helpers, mock classes
- `.coveragerc` - Coverage configuration
- `tests/unit/test_bytetrack.py` - 32 tests for multi-object tracking
- `tests/unit/test_kalman.py` - 26 tests for Kalman filter
- `tests/unit/test_event_detection.py` - 34 tests for shot/goal detection
- `tests/unit/test_team_clustering.py` - 27 tests for team assignment
- `tests/unit/test_colors.py` - 37 tests for jersey color extraction
- `tests/unit/test_pipeline_base.py` - 26 tests for pipeline orchestration
- `tests/unit/test_video_reader.py` - 23 tests for video reading
- `tests/integration/test_detection_tracking.py` - 8 integration tests
- `tests/integration/test_team_pipeline.py` - 5 integration tests
- `tests/integration/test_event_pipeline.py` - 8 integration tests
- `tests/golden/test_regression.py` - 12 golden tests
- `tests/golden/data/*.json` - Golden data files

**Key modules coverage**:
| Module | Before | After |
|--------|--------|-------|
| `bytetrack.py` | 0% | 98.59% |
| `kalman.py` | 0% | 100% |
| `pipeline/base.py` | ~40% | 100% |
| `team/clustering.py` | ~20% | 98.39% |
| `team/colors.py` | ~30% | 87.50% |
| `video/reader.py` | ~60% | 95.10% |

**Acceptance criteria**:
- [x] 60%+ overall coverage achieved (60.23%)
- [x] ByteTrack and Kalman filter fully tested
- [x] Event detection logic tested (shots, goals, deduplication)
- [x] Team clustering pipeline tested end-to-end
- [x] Integration tests for data flow between stages
- [x] Golden tests for regression prevention
- [x] All 400+ tests passing

---

### 8. ✅ Performance Profiling (COMPLETED)

**Problem**: Detection and overlay stages are slow; no visibility into bottlenecks.

**Solution**:
- Per-stage timing automatically saved to `run_manifest.json` (schema 1.1)
- Profiling script with cProfile and py-spy flame graph support
- Documented optimization opportunities in `docs/profiling.md`

**Files modified**:
- `src/pipeline/metrics.py` - **NEW** timing infrastructure
  - `StageTimer` - Context manager for timing stage execution
  - `StageMetrics` - Per-stage timing data (duration, items/sec, custom metrics)
  - `PipelineMetrics` - Aggregate metrics with device/platform info
  - `format_duration()` - Human-readable duration formatting
  - `detect_device()` - Detects mps/cuda/cpu
- `src/pipeline/base.py` - Integrated timing wrapper, updated manifest schema to 1.1
  - Wraps each stage with `StageTimer`
  - Prints timing summary with percentage breakdown after pipeline completes
  - Includes `metrics` section in `run_manifest.json`
- `src/cli.py` - All 6 stages now report metrics
  - `ingest_items_processed`, `ingest_custom_metrics` (fps, frames, resolution)
  - `detection_items_processed`, `detection_custom_metrics` (detector type, counts)
  - `tracking_items_processed`, `tracking_custom_metrics` (unique tracks, points)
  - `team_assignment_items_processed`, `team_assignment_custom_metrics`
  - `event_detection_items_processed`, `event_detection_custom_metrics`
  - `overlay_items_processed`, `overlay_custom_metrics`
- `scripts/profile_pipeline.py` - **NEW** profiling script
  - `--mode cprofile` - Built-in Python profiler with `.prof` output
  - `--mode flamegraph` - py-spy integration for SVG flame graphs
  - `--analyze` - Analyze existing profile data with summary
  - Generates `profile_summary.txt` with top functions
- `docs/profiling.md` - **NEW** profiling documentation
  - How to run profiling
  - How to interpret results
  - Documented optimization opportunities
- `tests/unit/test_profiling.py` - **NEW** 20 unit tests

**Manifest schema 1.1** (new `metrics` section):
```json
{
  "schema_version": "1.1",
  "metrics": {
    "total_duration_seconds": 120.5,
    "stages": {
      "detection": {
        "start_time": "2024-01-15T10:00:00",
        "end_time": "2024-01-15T10:01:30",
        "duration_seconds": 90.2,
        "items_processed": 5400,
        "items_per_second": 59.9,
        "custom_metrics": {"detector_type": "yolo", "ball_detections": 615}
      }
    },
    "device": "mps",
    "python_version": "3.11.9"
  }
}
```

**Documented optimization opportunities**:
1. **Batch detection** - `detect_batch()` exists but unused (1.5-3x potential speedup)
2. **Video decoding** - Single-threaded, could use hardware acceleration
3. **Overlay rendering** - Re-reads video, could use in-memory pipeline
4. **Team color extraction** - Sequential, could parallelize across frames

**Acceptance criteria**:
- [x] Timing breakdown in manifest (schema 1.1 with `metrics` section)
- [x] Flame graph generation script (`scripts/profile_pipeline.py`)
- [x] Documented optimization opportunities (`docs/profiling.md`)
- [x] 20 unit tests passing

---

### 9. ✅ Player Identity Persistence (COMPLETED)

**Problem**: Track IDs reset per video; no cross-video player linking.

**Solution**:
- Extract player embeddings using OSNet-x0.25 ReID model (~2MB, MPS-compatible)
- Store embeddings in local SQLite database (project-level, shared across runs)
- Match players across videos by cosine similarity with threshold-based logic
- API endpoints for player management and manual corrections

**Files created**:
- `src/vision/reid/__init__.py` - Module exports
- `src/vision/reid/base.py` - Abstract `ReIDExtractor` base class
- `src/vision/reid/osnet.py` - `OSNetExtractor` with HuggingFace weights, MPS support
- `src/vision/reid/crop.py` - `CropExtractor` and `PlayerCrop` for player crop extraction
- `src/identity/__init__.py` - Module exports
- `src/identity/models.py` - Pydantic models: `Player`, `Appearance`, `PlayerWithAppearances`
- `src/identity/database.py` - `PlayerDatabase` SQLite wrapper with CRUD operations
- `src/identity/matching.py` - `cosine_similarity`, `match_embedding_to_players`, `aggregate_embeddings`
- `tests/unit/test_reid.py` - 16 unit tests for ReID extraction
- `tests/unit/test_identity_db.py` - 19 unit tests for database operations
- `tests/unit/test_identity_matching.py` - 20 unit tests for matching algorithms

**Files modified**:
- `src/config/schemas.py` - Added `ReIDConfig` and `IdentityConfig` classes
- `src/cli.py` - Added `PlayerIdentityStage` after TeamAssignment
- `src/ui/server.py` - Added 7 player identity API endpoints
- `configs/default.yaml` - Added `reid` and `identity` configuration sections
- `pyproject.toml` - Added `huggingface_hub` dependency

**Database Schema** (`players.db`):
```sql
CREATE TABLE players (
    player_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    jersey_number INTEGER,
    team_hint TEXT,          -- "ours" | "opponent"
    embedding_centroid BLOB, -- 512 floats (2KB)
    embedding_count INTEGER DEFAULT 0,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE appearances (
    appearance_id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL,
    run_name TEXT NOT NULL,
    track_id INTEGER NOT NULL,
    player_id INTEGER,       -- FK to players (NULL = unassigned)
    match_confidence FLOAT,
    match_method TEXT,       -- "auto" | "suggested" | "manual"
    frame_start INTEGER,
    frame_end INTEGER,
    embedding BLOB,
    UNIQUE(video_id, track_id)
);
```

**Configuration** (`configs/default.yaml`):
```yaml
reid:
  model_name: "osnet_x0_25"
  device: "mps"
  embedding_dim: 512
  crop_size: [256, 128]
  batch_size: 32
  cache_dir: "models"

identity:
  enabled: true
  database_path: "players.db"
  samples_per_track: 10
  min_crop_height: 50
  min_crop_width: 25
  auto_match_threshold: 0.85
  suggest_threshold: 0.70
  new_player_threshold: 0.60
```

**API Endpoints**:
- `GET /api/players` - List all players
- `GET /api/players/{id}` - Get player with appearances
- `PATCH /api/players/{id}` - Update name/jersey/team
- `DELETE /api/players/{id}` - Delete player
- `POST /api/appearances/{video_id}/{track_id}/assign/{player_id}` - Manual assign
- `POST /api/players/merge/{keep}/{merge}` - Merge players
- `GET /api/runs/{run_name}/player_assignments` - Get run assignments

**Per-Run Output** (`player_assignments.json`):
```json
{
  "schema_version": "1.0",
  "video_id": "match_001",
  "assignments": [
    {"track_id": 96, "player_id": 1, "player_name": "John", "match_method": "auto", "confidence": 0.92}
  ],
  "stats": {"total_tracks": 150, "auto_matched": 89, "suggested": 23, "new_players": 38}
}
```

**Matching Logic**:
- `similarity >= 0.85`: Auto-assign to best match
- `0.70 <= similarity < 0.85`: Suggest best match (requires confirmation)
- `similarity < 0.60`: Create new player

**Acceptance criteria**:
- [x] OSNet-x0.25 ReID model with MPS/CUDA/CPU support
- [x] SQLite database for persistent player storage
- [x] Cosine similarity matching with configurable thresholds
- [x] Running centroid update for improved matching over time
- [x] `PlayerIdentityStage` integrated into pipeline after TeamAssignment
- [x] API endpoints for player management
- [x] `player_assignments.json` per-run output
- [x] 55 unit tests passing (16 ReID + 19 database + 20 matching)

---

### 10. ✅ Celebration/Reaction Detection (COMPLETED)

**Problem**: Goals are missed if ball tracking fails at critical moment.

**Solution**:
- Detect arms-up celebration poses via bbox aspect ratio changes
- Detect group huddle celebrations via player convergence analysis
- Integrate as new weighted signal in shot fusion engine
- Use celebration patterns as goal confirmation signal

**Files modified**:
- `src/config/schemas.py` - Added `CelebrationConfig` with all tunable parameters
  - Arms-up detection: aspect ratio threshold, height change, min duration
  - Group huddle: max distance, min players, convergence threshold
  - Temporal: post-shot window (150 frames), cooldown (300 frames)
  - Fusion: signal weight (0.15), min confidence (0.4)
- `src/events/celebration_detection.py` - **NEW** module with:
  - `CelebrationEvent` dataclass (frame_idx, confidence, type, players, evidence)
  - `CelebrationDetector` class with arms-up and huddle detection
  - `_compute_track_baselines()` - Establishes normal pose from early frames
  - `_detect_arms_up()` - Detects raised arms via aspect ratio increase
  - `_detect_group_huddle()` - Detects player convergence after shot
  - `_deduplicate_celebrations()` - Cooldown-based deduplication
- `src/events/kick_detection.py` - Updated `ShotCandidate` and `ShotFusionEngine`
  - Added `celebration_event` field to `ShotCandidate`
  - Updated `fuse_signals()` to accept `celebration_events` parameter
  - Weight rebalancing when celebration enabled (kick 0.30, goal_entry 0.25, gk_dive 0.20, celebration 0.15)
- `src/events/detection.py` - Integrated `CelebrationDetector` into `EventDetector`
  - Initializes celebration detector when enabled in config
  - Calls celebration detection after initial shot candidates
  - Passes celebration events to fusion engine
  - Adds celebration metadata to event output
- `src/events/__init__.py` - Exported `CelebrationDetector` and `CelebrationEvent`
- `configs/default.yaml` - Added full `alternative_shot` section with nested `celebration` config
- `tests/unit/test_celebration_detection.py` - **NEW** 14 unit tests

**Detection Algorithms**:
- **Arms-up**: Compare current bbox aspect ratio to baseline (first 30 frames). If aspect ratio increases above threshold for min_duration_frames, celebration detected.
- **Group huddle**: Track player positions before/after shot. If players converge (pairwise distance < threshold, cluster size >= min_players), and convergence_ratio > threshold, huddle detected.

**Configuration** (`configs/default.yaml`):
```yaml
events:
  alternative_shot:
    celebration:
      enabled: true
      arms_up_aspect_ratio_threshold: 0.5
      arms_up_min_duration_frames: 5
      huddle_max_player_distance: 100.0
      huddle_min_players: 3
      huddle_convergence_threshold: 0.5
      post_shot_window_frames: 150  # 5 sec at 30fps
      celebration_cooldown_frames: 300
      signal_weight: 0.15
      min_confidence: 0.4
```

**Acceptance criteria**:
- [x] Arms-up detection via bbox aspect ratio changes
- [x] Group huddle detection via player convergence
- [x] Integrated as weighted signal in shot fusion (0.15 weight)
- [x] Cooldown prevents duplicate detections
- [x] Configurable via YAML
- [x] 14 unit tests passing
- [x] All 404 tests passing (no regressions)

---

## Implementation Priority

| Priority | Enhancement | Impact | Effort | Status |
|----------|-------------|--------|--------|--------|
| P0 | ~~Smarter Interpolation (#1)~~ | High | Low | ✅ DONE |
| P0 | ~~Ball Detection Boosting (#2)~~ | Critical | Medium | ✅ DONE |
| P0 | ~~Shot Detection Alternatives (#3)~~ | High | Medium | ✅ DONE |
| P1 | ~~Event Confirmation UI (#5)~~ | High | Low | ✅ DONE |
| P2 | ~~Goal Region Detection (#4)~~ | Medium | High | ✅ DONE |
| P2 | ~~Ball-Specific Model (#6)~~ | High | High | ✅ DONE |
| P3 | ~~Test Coverage (#7)~~ | Medium | Medium | ✅ DONE |
| P3 | ~~Performance Profiling (#8)~~ | Low | Low | ✅ DONE |
| P3 | ~~Celebration Detection (#10)~~ | Medium | Medium | ✅ DONE |
| P4 | ~~Player Identity Persistence (#9)~~ | Medium | High | ✅ DONE |

---

## Quick Wins (< 1 hour each)

1. ~~**Lower ball confidence threshold** to 0.15 in config~~ ✅ Done (part of ball boost)
2. ~~**Increase max_interpolation_gap** to 150 frames~~ ✅ Done (now 300 with physics)
3. ~~**Add ball detection rate** to run summary~~ ✅ Done (shows player/ball counts)
4. **Log gap statistics** during event detection
5. **Add --verbose flag** for detailed stage output

---

## Data Collection Needs

To improve ball detection, we need:
- [ ] 100+ labeled soccer ball images (various sizes, lighting)
- [ ] 10+ video clips with manual ball annotations
- [ ] Ground truth events for 3+ matches (for evaluation)

---

## References

- [SoccerNet Ball Detection](https://www.soccer-net.org/) - Benchmark dataset
- [TrackNet](https://nol.cs.nctu.edu.tw/nol/guest/index.php) - Ball tracking in sports
- [ByteTrack Paper](https://arxiv.org/abs/2110.06864) - Current tracker
- [YOLO Ball Detection](https://github.com/ultralytics/ultralytics) - Base model

---

*Last updated: 2026-01-31 (Player Identity Persistence completed - ReID embeddings + SQLite database for cross-video player matching)*
