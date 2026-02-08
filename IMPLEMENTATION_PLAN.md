# Implementation Plan

> Veo Football Video Analysis App — feature parity and hardening roadmap.
> Delete this file to return to working directly from specs.
> Project root: /Users/yehj10/Local/personal/ai_video_analysis

## Tasks

### Phase 0: Baseline Hardening and Gap Matrix

- [x] [HIGH] Step 00 — Stabilize test harness: Fix pytest collection reliability (Torch import crash on collection). Ensure tests that do not require model loading can run. Test: `pytest --collect-only -q` and `pytest tests/unit/test_config.py tests/unit/test_pipeline_base.py -q`
- [x] [HIGH] Step 01 — Create guide-to-code gap matrix: Add `docs/veo_guide_gap_matrix.md` mapping each Veo guide feature to implemented/partial/missing status with current module paths. Test: `test -f docs/veo_guide_gap_matrix.md`

### Phase 1: Event Detection Parity

- [x] [HIGH] Step 02 — Event schema extension: Extend event typing in `src/events/detection.py` to support pass and set-piece event families with schema-versioned metadata. Test: `pytest tests/unit/test_event_detection.py -q`
- [x] [HIGH] Step 03 — Pass detection module: Create `src/events/passes.py` with deterministic pass inference from possession handoff, including confidence and provenance fields. Test: `pytest tests/unit/test_event_detection.py -q`
- [x] [HIGH] Step 04 — Set-piece detection module: Create `src/events/set_pieces.py` for kickoff, throw-in, corner, free kick, goal kick heuristics with configurable confidence thresholds. Test: `pytest tests/unit -q -k "set_piece or corner or throw or free_kick or goal_kick or kickoff"`
- [x] [HIGH] Step 05 — Integrate pass + set-pieces into pipeline: Wire new detectors into event stage in `src/cli.py`, ensure `events.jsonl` and `summary.json` include new counts. Test: `pytest tests/integration/test_event_pipeline.py -q`

### Phase 2: Match Stats and Visualization

- [ ] [HIGH] Step 06 — Unified match stats artifact: Create `src/analytics/match_stats.py` to emit `match_stats.json` with team-level totals (shots, goals, passes, set-pieces, possession). Test: `pytest tests/unit -q -k "match_stats or team_analytics or event_detection"`
- [ ] [HIGH] Step 07 — Stats API + UI panel: Add backend endpoints in `src/ui/server.py` to serve match stats, add UI rendering for stats panel with team comparison in `src/ui/static/app.js` and `src/ui/static/index.html`. Test: `pytest tests/unit/test_ui_server.py -q`
- [ ] [HIGH] Step 08 — Visualization framework scaffolding: Create package `src/export/visualizations/` with rendering interfaces and shared field canvas utilities. Test: `pytest tests/unit -q -k "visualization or field_normalization"`
- [ ] [HIGH] Step 09 — Shot map visualization: Implement `src/export/visualizations/shot_map.py` with per-team and per-player filters. Test: `pytest tests/unit -q -k "shot_map or event_detection"`
- [ ] [HIGH] Step 10 — Heat map visualization: Implement `src/export/visualizations/heat_map.py` from track/normalized coordinates, supporting team and player modes. Test: `pytest tests/unit -q -k "heat_map or field_normalization or team_analytics"`
- [ ] [HIGH] Step 11 — Pass map and tactical map visualizations: Implement `src/export/visualizations/pass_map.py` and `src/export/visualizations/tactical_map.py`, expose via API in `src/ui/server.py` and UI toggles. Test: `pytest tests/unit -q -k "pass_map or tactical_map or team_analytics"`

### Phase 3: Tactical Event Classification

- [ ] [HIGH] Step 12 — Tactical event classification: Create `src/events/tactical.py` for build-up, pressing, defending, and transition events with confidence/provenance in `events.jsonl`. Test: `pytest tests/unit -q -k "tactical or pressing or team_analytics"`

### Phase 4: Tagging and Annotation System

- [ ] [HIGH] Step 13 — DB migration for tags + match metadata: Extend SQLite schema in `src/identity/database.py` with match metadata table and tags table, add idempotent migrations. Test: `pytest tests/unit/test_identity_db.py -q`
- [ ] [HIGH] Step 14 — Tagging API + UI flows: Add tags CRUD/filter endpoints in `src/ui/server.py`, add manual tagging and filtering controls in `src/ui/static/app.js`. Test: `pytest tests/unit/test_ui_server.py -q`

### Phase 5: Player and Season Analytics

- [ ] [HIGH] Step 15 — Per-player analytics module: Create `src/analytics/player.py` for per-player stats across runs (minutes, distance, sprints, events), emit player analytics artifact and API payloads. Test: `pytest tests/unit -q -k "player_analysis or player_reels or team_analytics"`
- [ ] [HIGH] Step 16 — Season analytics enhancements: Create `src/analytics/season.py`, extend cross-match outputs with possession trend, W/L/D, match-type filter, formation filter, and radar-ready aggregates. Test: `pytest tests/unit/test_cross_match_reporting.py -q`

### Phase 6: Optional Integrations

- [ ] [MEDIUM] Step 17 — Coach assist (opt-in): Create `src/analytics/coach_assist.py` with provider abstraction, disabled by default, no cloud calls unless explicitly enabled. Test: `pytest tests/unit -q -k "coach_assist or analytics"`
- [ ] [MEDIUM] Step 18 — Veo API integration client (opt-in): Create `src/integrations/veo_api.py` with auth + basic read/write operations behind interface boundaries. Test: `pytest tests/unit -q -k "veo_api or integration_client"`

### Phase 7: Video UX and Final Hardening

- [ ] [MEDIUM] Step 19 — Video UX parity: Add speedrun playback mode (skip low-action windows) and multi-view layout toggle in `src/ui/static/app.js` and `src/ui/server.py`. Test: `pytest tests/unit/test_ui_server.py -q`
- [ ] [HIGH] Step 20 — Final regression + contract validation: Full artifact contract validation for all required outputs, ensure schema versions and cache/resume behavior are consistent. Test: `pytest tests/unit -q && pytest tests/integration -q && pytest tests/golden -q`

## Cross-Phase Quality Gates

> Apply every phase — agent should verify these before outputting DONE.

- Every new artifact is schema-versioned and resumable/cache-safe
- Every non-trivial change adds unit or golden coverage
- Optional metadata/integrations never block core pipeline
- Preserve local-first defaults; cloud/API features are explicitly opt-in
- Keep guide parity tracked in docs/veo_guide_gap_matrix.md

## Completed
