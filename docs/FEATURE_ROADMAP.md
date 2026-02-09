# Feature Roadmap: Flexible Soccer Video Analysis

This document expands the roadmap for two primary perspectives:
- team analytics (coaches, analysts)
- individual player analytics (player development, recruiting)

It also defines a practical path for dynamic player mapping/tagging from player profile photos and `.pkl` embeddings, and a "highlight game" generator from goals, audio excitement, and action quality.

## Current Baseline (Already in Repo)

- Player + ball detection (YOLO + optional specialist/ensemble)
- Multi-object tracking (ByteTrack)
- Team clustering from jersey colors
- Shot and goal detection (with sparse-ball fallback and celebration signal)
- Player ReID embeddings (OSNet) + persistent SQLite identities
- Local UI for run browsing and event confirmation

## Team Perspective Features

### Match control and territory
- [DONE] Possession estimate by minute and by phase (build-up, middle third, final third)
- [TODO] Territory dominance map (team centroid, ball zone occupancy)
- [TODO] Final-third entries and penalty-box entries timeline
- [TODO] Pressing intensity proxy (distance-to-ball and convergence speed)
- [TODO] Team compactness (horizontal and vertical spread)
- [TODO] Defensive line height estimate over time
- [TODO] Transition detection (defense->attack, attack->defense) with timestamps

### Passing and chance creation
- [TODO] Pass candidate detection (ball carrier switch + trajectory handoff)
- [TODO] Pass network graph (edges weighted by pass count/completion)
- [TODO] Progressive passes and line-breaking passes
- [TODO] Crosses, cutbacks, and through-ball heuristics
- [TODO] Sequence-level chance creation chains (N actions before shot)
- [TODO] Shot map by region and footedness proxy (if detectable)
- [TODO] Expected threat proxy (xT-like grid from normalized field coordinates)

### Team performance KPIs
- [TODO] Team pressing regains and forced turnovers
- [TODO] Counterattack detection (rapid vertical progression after regain)
- [TODO] Build-up speed and directness metrics
- [TODO] Set-piece event detection (corners, free kicks, throw-ins) heuristics
- [TODO] Goalkeeper distribution profile (long/short, target zones)
- [TODO] Attack channel usage (left/center/right)
- [TODO] Opponent-adjusted metrics across multiple matches

### Team reporting and workflow
- [TODO] Match report export (PDF/HTML/JSON)
- [TODO] Coach playlist builder by tactical theme
- [TODO] Side-by-side half comparison dashboard
- [TODO] Batch run comparisons (season trends)
- [TODO] Analyst annotation and telestration save/load

## Individual Player Perspective Features

### Core player timelines
- [TODO] Touch timeline with context (pressure level, zone, outcome)
- [TODO] Involvement rate by minute (on-ball and near-ball actions)
- [TODO] Ball receipt map and release map
- [TODO] Shot involvement and assist-chain involvement
- [TODO] Defensive action timeline (interceptions, blocks, recoveries)
- [TODO] Sprint/high-intensity run proxy events
- [TODO] Fatigue trend proxy (speed/acceleration drop in late game)

### Role-based evaluation
- [TODO] Position heatmap and role consistency score
- [TODO] Off-ball run classification (overlap, underlap, diagonal, support)
- [TODO] Marking/pressing responsibility proxy by nearest-opponent patterns
- [TODO] Defender duel outcomes and clearance quality
- [TODO] Midfielder progression contribution (carries + progressive passes)
- [TODO] Forward chance quality and box presence metrics
- [TODO] Goalkeeper shot-stopping and distribution metrics

### Individual exports
- [TODO] Auto-generate per-player highlight reel (best actions)
- [TODO] Auto-generate per-player "all touches" cut
- [TODO] Auto-generate per-player defensive-only cut
- [TODO] Player report card with confidence/provenance for each stat
- [TODO] Multi-match player trend report

## Dynamic Player Mapping and Tagging from Profile Pictures

### Why this needs multimodal matching

The current identity flow uses OSNet body embeddings from in-game crops.  
Your profile file example uses FaceNet-style face embeddings (`Facenet512` in `.pkl`).  
These embedding spaces are different, so direct cosine matching between current track embeddings and `.pkl` face vectors is not reliable.

### Input profile bundle format (supported target)

Example folder pattern:
- `/Users/yehj10/iCloud/vs/download_icloud_images/team_pictures/10_Nicholas_Oestringer/`

Expected files:
- [TODO] `*.jpg` / `*.png` reference photos
- [TODO] Optional `*.pkl` with keys like `player_id`, `encodings`, `averaged_encoding`, `model`
- [TODO] Optional metadata (`team`, `number`, `name`, `position`)

### Proposed pipeline additions

### Stage P1: Profile ingestion
- [TODO] New module `src/identity/profile_ingest.py`
- [TODO] Parse profile directories and normalize player metadata
- [TODO] Read `.pkl` safely and extract face embeddings + model metadata
- [TODO] Generate `profile_registry.json` (schema-versioned)
- [TODO] Generate `profile_embeddings.parquet` with modality tags (`face`, `body`, `jersey_number`)

### Stage P2: In-video evidence extraction
- [TODO] Face detector on close-up frames (only when face size threshold is met)
- [TODO] Face embedding extraction with same model family as profiles (Facenet512 path)
- [TODO] Jersey number OCR candidate extraction (EasyOCR/Tesseract backends)
- [TODO] Team-color consistency score per track
- [TODO] Body ReID embedding quality score per track segment

### Stage P3: Dynamic assignment and tagging
- [TODO] Multi-signal scorer per track:
- [TODO] `score = w_body * sim_body + w_face * sim_face + w_number * number_match + w_team * team_match + w_color * color_match + w_temp * temporal_consistency`
- [TODO] Adaptive weights by evidence quality (face-only boost when close-up, body-only boost in wide shots)
- [TODO] State machine: `unknown -> candidate -> locked -> review`
- [TODO] Re-scoring on each new strong observation, with hysteresis to avoid identity flipping
- [TODO] Substitution detection hook to unlock stale identities automatically

### Stage P4: Human-in-the-loop corrections
- [TODO] UI panel for unresolved/suggested identities
- [TODO] One-click assign/merge/split track identity
- [TODO] Correction log persisted (`identity_corrections.jsonl`)
- [TODO] Online learning: manual corrections update player centroids and thresholds

## New identity artifacts (versioned)

- [TODO] `profile_registry.json`
- [TODO] `profile_embeddings.parquet`
- [TODO] `track_identity_timeline.jsonl` (identity confidence over time, per `track_id`)
- [TODO] `identity_corrections.jsonl`

## Highlight Game (Auto Segment Selection)

### Target outcome

Auto-select high-value match segments for:
- all-match highlights
- team-only highlights (ours/opponent)
- player-only highlights

### Signals to fuse

### Event signals
- [TODO] Goals (highest weight)
- [TODO] Shots on target / near-goals
- [TODO] Big chances from trajectory + crowd reaction
- [TODO] Goalkeeper saves and last-man defensive actions

### Audio excitement signals
- [TODO] Crowd loudness spikes (RMS/peak over rolling baseline)
- [TODO] Spectral flux spikes (sudden excitement bursts)
- [TODO] Sustained elevated volume windows (build-up moments)
- [TODO] Whistle and restart cues for segment boundaries

### Action quality signals
- [TODO] Ball speed bursts and rapid directional changes
- [TODO] Multi-player convergence near penalty areas
- [TODO] Fast transition windows (end-to-end movement)
- [TODO] Repeated touches in dangerous zones

## Segment assembly

- [TODO] Candidate windows with configurable pre/post roll (for example, 8s before and 12s after trigger)
- [TODO] Overlap merge and deduplication across nearby triggers
- [TODO] Rank by weighted excitement score with reason codes
- [TODO] Export top-N, all goals, and per-player/per-team playlists
- [TODO] Emit both machine score and human-readable reasons

## Highlight artifacts (versioned)

- [TODO] `highlight_candidates.jsonl`
- [TODO] `highlights.json` (selected segments, scores, reasons)
- [TODO] `highlights.csv`
- [TODO] `highlights_manifest.json`
- [TODO] `clips/` directory with rendered MP4 segments

## UI Extensions for Flexibility

- [TODO] Multi-view dashboard: team view, player view, tactical view
- [TODO] Filter bar by player/team/event/confidence/time window
- [TODO] Live confidence/provenance badges on every metric/event
- [TODO] Identity resolution queue with unresolved tags
- [TODO] Highlight tuning controls (aggressiveness, audio weight, minimum quality)
- [TODO] One-click export templates (coach, player, scout)

## Suggested Implementation Order (High Impact First)

1. [TODO] Profile ingestion + schema artifacts (`profile_registry.json`, `profile_embeddings.parquet`)
2. [TODO] Highlight scoring from existing events + basic audio spikes
3. [TODO] Jersey number OCR + multimodal identity scoring
4. [TODO] Player-focused highlight reels and report cards
5. [TODO] Team tactical analytics and cross-match trend reports

## Scope Questions to Finalize Before Build

- [TODO] Confirm primary user mode priority: coach/team analysis vs player development vs recruiting
- [TODO] Confirm whether profile `.pkl` files are always Facenet512 and whether raw photos are always present
- [TODO] Confirm if FFmpeg/audio extraction dependencies are acceptable in default local setup
- [TODO] Confirm if we should support per-half/per-phase tactical templates out of the box
- [TODO] Confirm if identity tagging should default to conservative (fewer wrong tags) or aggressive (more auto-tags)
