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

- [x] [HIGH] Step 06 — Unified match stats artifact: Create `src/analytics/match_stats.py` to emit `match_stats.json` with team-level totals (shots, goals, passes, set-pieces, possession). Test: `pytest tests/unit -q -k "match_stats or team_analytics or event_detection"`
- [x] [HIGH] Step 07 — Stats API + UI panel: Add backend endpoints in `src/ui/server.py` to serve match stats, add UI rendering for stats panel with team comparison in `src/ui/static/app.js` and `src/ui/static/index.html`. Test: `pytest tests/unit/test_ui_server.py -q`
- [x] [HIGH] Step 08 — Visualization framework scaffolding: Create package `src/export/visualizations/` with rendering interfaces and shared field canvas utilities. Test: `pytest tests/unit -q -k "visualization or field_normalization"`
- [x] [HIGH] Step 09 — Shot map visualization: Implement `src/export/visualizations/shot_map.py` with per-team and per-player filters. Test: `pytest tests/unit -q -k "shot_map or event_detection"`
- [x] [HIGH] Step 10 — Heat map visualization: Implement `src/export/visualizations/heat_map.py` from track/normalized coordinates, supporting team and player modes. Test: `pytest tests/unit -q -k "heat_map or field_normalization or team_analytics"`
- [x] [HIGH] Step 11 — Pass map and tactical map visualizations: Implement `src/export/visualizations/pass_map.py` and `src/export/visualizations/tactical_map.py`, expose via API in `src/ui/server.py` and UI toggles. Test: `pytest tests/unit -q -k "pass_map or tactical_map or team_analytics"`

### Phase 3: Tactical Event Classification

- [x] [HIGH] Step 12 — Tactical event classification: Create `src/events/tactical.py` for build-up, pressing, defending, and transition events with confidence/provenance in `events.jsonl`. Test: `pytest tests/unit -q -k "tactical or pressing or team_analytics"`

### Phase 4: Tagging and Annotation System

- [x] [HIGH] Step 13 — DB migration for tags + match metadata: Extend SQLite schema in `src/identity/database.py` with match metadata table and tags table, add idempotent migrations. Test: `pytest tests/unit/test_identity_db.py -q`
- [x] [HIGH] Step 14 — Tagging API + UI flows: Add tags CRUD/filter endpoints in `src/ui/server.py`, add manual tagging and filtering controls in `src/ui/static/app.js`. Test: `pytest tests/unit/test_ui_server.py -q`

### Phase 5: Player and Season Analytics

- [x] [HIGH] Step 15 — Per-player analytics module: Create `src/analytics/player.py` for per-player stats across runs (minutes, distance, sprints, events), emit player analytics artifact and API payloads. Test: `pytest tests/unit -q -k "player_analysis or player_reels or team_analytics"`
- [x] [HIGH] Step 16 — Season analytics enhancements: Create `src/analytics/season.py`, extend cross-match outputs with possession trend, W/L/D, match-type filter, formation filter, and radar-ready aggregates. Test: `pytest tests/unit/test_cross_match_reporting.py -q`

### Phase 6: Optional Integrations

- [x] [MEDIUM] Step 17 — Coach assist (opt-in): Create `src/analytics/coach_assist.py` with provider abstraction, disabled by default, no cloud calls unless explicitly enabled. Test: `pytest tests/unit -q -k "coach_assist or analytics"`
- [x] [MEDIUM] Step 18 — Veo API integration client (opt-in): Create `src/integrations/veo_api.py` with auth + basic read/write operations behind interface boundaries. Test: `pytest tests/unit -q -k "veo_api or integration_client"`

### Phase 7: Video UX and Final Hardening

- [x] [MEDIUM] Step 19 — Video UX parity: Add speedrun playback mode (skip low-action windows) and multi-view layout toggle in `src/ui/static/app.js` and `src/ui/server.py`. Test: `pytest tests/unit/test_ui_server.py -q`
- [x] [HIGH] Step 20 — Final regression + contract validation: Full artifact contract validation for all required outputs, ensure schema versions and cache/resume behavior are consistent. Test: `pytest tests/unit -q && pytest tests/integration -q && pytest tests/golden -q`
- [x] [HIGH] Step 21 — VideoReader cleanup regression fix: Ensure `VideoReader` cleanup is safe when initialization exits before `cap` exists; add regression coverage. Test: `pytest tests/unit/test_video_reader_cleanup.py -q && pytest tests/unit -q && pytest tests/integration -q && pytest tests/golden -q`

## Cross-Phase Quality Gates

> Apply every phase — agent should verify these before outputting DONE.

- Every new artifact is schema-versioned and resumable/cache-safe
- Every non-trivial change adds unit or golden coverage
- Optional metadata/integrations never block core pipeline
- Preserve local-first defaults; cloud/API features are explicitly opt-in
- Keep guide parity tracked in docs/veo_guide_gap_matrix.md

## Phase 8 — Veo Feature Parity (2026-02-09)

- [x] [SMALL] Step 22 — Shot Map & Heat Map API endpoints: Wire existing `ShotMapRenderer` and `HeatMapRenderer` to API endpoints in `src/ui/server.py`, add UI dropdown options in `index.html`, and route in `app.js`. Test: `pytest tests/unit/test_shot_map.py tests/unit/test_heat_map.py tests/unit/test_ui_server.py -q`
- [x] [SMALL] Step 23 — Per-player visualization filter: Add player dropdown filter to all visualization panels in `app.js` using dynamic `KNOWN_VIZ_TYPES` routing. Test: `pytest tests/unit/test_ui_server.py -q`
- [x] [MEDIUM] Step 24 — Momentum Graph visualization: Create `src/export/visualizations/momentum_graph.py` (`MomentumGraphRenderer`) combining possession share and territorial control into time-series area chart. Add API endpoint and UI integration. Test: `pytest tests/unit/test_momentum_graph.py -q`
- [x] [MEDIUM] Step 25 — Pass Strings visualization: Create `src/export/visualizations/pass_strings.py` (`PassStringsRenderer`) to visualize consecutive passing chains on field canvas. Add API endpoint and UI integration. Test: `pytest tests/unit/test_pass_strings.py -q`
- [x] [MEDIUM] Step 26 — Radial/Comparison Chart: Create `src/export/visualizations/radial_chart.py` (`RadialChartRenderer`) for spider/radar team comparison. Add API endpoint and UI integration. Test: `pytest tests/unit/test_radial_chart.py -q`
- [x] [SMALL] Step 27 — Scoreboard Overlay: Add `draw_scoreboard()` method to `OverlayRenderer` in `src/export/overlay.py` for score + match time display. Test: `pytest tests/unit/test_overlay.py -q`
- [x] [MEDIUM] Step 28 — Shot on/off target classification: Add `shot_on_target` and `shot_off_target` to `EventType` and `EVENT_TYPE_TO_FAMILY` in `src/events/detection.py`. Add `_classify_shot_target()` to `EventDetector`. Test: `pytest tests/unit/test_event_detection.py -q`
- [x] [SMALL] Step 29 — Possession Won metric: Add possession transition counting to `build_team_analytics()` in `src/analytics/team.py`. Test: `pytest tests/unit/test_team_analytics.py -q`
- [x] [SMALL] Step 30 — Possession Minutes: Add absolute possession minutes (seconds/60) alongside percentage in `build_team_analytics()`. Test: `pytest tests/unit/test_team_analytics.py -q`
- [x] [MEDIUM] Step 31 — Penalty Detection: Add penalty scoring to `SetPieceInferencer._classify_candidate()` in `src/events/set_pieces.py` using penalty spot geometry. Test: `pytest tests/unit/test_set_piece_detection.py -q`
- [x] [SMALL] Step 32 — Counter Press Detection: Add `_infer_counter_press_events()` to `TacticalInferencer` in `src/events/tactical.py`. Test: `pytest tests/unit/test_tactical_detection.py -q`
- [x] [MEDIUM] Step 33 — Defending Subtypes: Add `defending_box`, `defending_success`, `defending_poor_recovery` subtypes to `TacticalInferencer` in `src/events/tactical.py`. Test: `pytest tests/unit/test_tactical_detection.py -q`
- [x] [MEDIUM] Step 34 — Formation/Lineup Authoring UI: Add GET/PUT `/api/runs/{run_name}/lineup` endpoints and UI panel in `app.js`/`index.html`. Test: `pytest tests/unit/test_ui_server.py -q`
- [x] [SMALL] Step 35 — Coach Notes/Journal: Add GET/POST/DELETE `/api/runs/{run_name}/notes` CRUD endpoints and UI panel. Test: `pytest tests/unit/test_ui_server.py -q`
- [x] [MEDIUM] Step 36 — Multi-Match Aggregated Maps: Add `/api/multi-run/visualizations/{viz_type}` endpoint in `src/ui/server.py` for cross-match visualization aggregation. Test: `pytest tests/unit/test_ui_server.py -q`
- [x] [SMALL] Step 37 — Player Spotlight Sensitivity: Add GET/PUT `/api/runs/{run_name}/spotlight_config` endpoints with adjustable ball distance, time-on-ball, and buffer parameters. Test: `pytest tests/unit/test_ui_server.py -q`
- [x] [MEDIUM] Step 38 — Player Progress Charts: Create `src/export/visualizations/progress_chart.py` (`ProgressChartRenderer`) for cross-match player development visualization. Add API endpoint. Test: `pytest tests/unit/test_progress_chart.py -q`
- [x] [SMALL] Step 39 — Player Share Links: Add `/api/runs/{run_name}/share/{player_id}` endpoint for generating shareable player highlight permalinks. Test: `pytest tests/unit/test_ui_server.py -q`

## Completed
