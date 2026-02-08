# Veo Guide to Code Gap Matrix

Last verified: 2026-02-08
Source guide: `veo_analytics_implementation_guide.md`

## Status legend
- `Implemented`: production code path exists and is wired into pipeline/UI.
- `Partial`: foundational code exists, but parity is incomplete (missing subtype granularity, visualization, UX, or export integration).
- `Missing`: no direct implementation in current codebase.

## Summary
- Features mapped: `75`
- `Implemented`: `14`
- `Partial`: `14`
- `Missing`: `47`

## 1) Match-level statistics

| Veo feature | Status | Current code path(s) | Notes |
|---|---|---|---|
| Possession % | Implemented | `src/analytics/team.py`, `src/cli.py` (`TeamAnalyticsStage`) | Possession summary + timeline exported in `team_analytics.json` and `team_possession_timeline.csv`. |
| Passes Completed | Partial | `src/analytics/team.py` | Inferred pass network exists, but completed/attempted pass accounting is not explicit event-level pass detection. |
| Shots (count) | Implemented | `src/events/detection.py`, `src/cli.py` (`EventDetectionStage`) | Shot events are emitted to `events.jsonl` and counted in `summary.json`. |
| Goals (count) | Implemented | `src/events/detection.py`, `src/vision/field/goal_detector.py`, `src/cli.py` | Goal events are emitted and added to `score_timeline.json`. |
| Free Kicks | Missing | — | No set-piece detector module for free kicks. |
| Throw-ins | Missing | — | No throw-in detector module. |
| Corners | Missing | — | No corner detector module. |
| Penalties | Missing | — | No penalty event detector. |
| Score Tracking (live) | Partial | `src/cli.py` (`EventDetectionStage`), `src/ui/server.py` (`/timeline`), `src/ui/static/app.js` | Score timeline exists post-analysis, but there is no live/streaming score inference and team scoring attribution is still heuristic. |

## 2) Visualizations (Analytics 2.0 style)

| Veo feature | Status | Current code path(s) | Notes |
|---|---|---|---|
| Shot Map (field diagram) | Missing | — | No `src/export/visualizations/shot_map.py` equivalent yet. |
| Pass Location Map | Missing | — | No pass-map renderer/export yet. |
| Possession Location (by thirds) | Implemented | `src/analytics/team.py`, `src/cli.py` (`TeamAnalyticsStage`) | Territory bins and control shares are exported in `team_territory_zones.csv`. |
| Pass Strings | Missing | — | No sequential pass-chain visualization. |
| Heat Map | Partial | `src/vision/field/normalization.py`, `src/cli.py` (`FieldNormalizationStage`) | Normalized coordinates exist (`norm_xy`), but no heat-map rendering/export module. |
| Tactical Map (live positions) | Partial | `src/ui/server.py` (`/tracks`), `src/ui/static/app.js` | Track overlays and timeline playback exist, but no dedicated tactical map artifact/view. |
| Comparison Radial Chart | Missing | — | No radar/spider chart implementation in export or UI. |
| Possession % Trend Chart | Partial | `src/export/cross_match.py`, `src/ui/static/app.js` (season panel) | Cross-match trend data is exported; charted trend visualization is not implemented. |

## 3) Event detection (AI-tagged)

| Veo event type | Status | Current code path(s) | Notes |
|---|---|---|---|
| Goal | Implemented | `src/events/detection.py`, `src/cli.py` (`EventDetectionStage`) | Goal events generated with confidence and timeline updates. |
| Shot on Goal | Partial | `src/events/detection.py` | Shot direction/target metadata exists, but no distinct `shot_on_goal` event class. |
| Shot (off target) | Partial | `src/events/detection.py` | General `shot` events exist, but not split into explicit on/off target taxonomy. |
| Celebration | Implemented | `src/events/celebration_detection.py`, `src/events/detection.py` | Celebration signal is used in alternative shot/goal fusion flow. |
| Kickoff | Missing | — | No kickoff detector. |
| Build Up | Missing | — | No build-up event classifier. |
| Pressing | Partial | `src/analytics/team.py` | Pressing metrics/episodes exist, but not emitted as standalone event tags in `events.jsonl`. |
| Defending | Missing | — | No defending event classifier. |
| Defending the Box | Missing | — | No box-defense event classifier. |
| Defending (Success) | Missing | — | No success/failure defending event taxonomy. |
| Defending, Poor Recovery | Missing | — | No poor-recovery event tagging. |
| Counter Press | Missing | — | No counter-press event detector. |
| Transition to Defend | Missing | — | No transition classifier for attack->defense phase changes. |
| Body Shape | Missing | — | No body-orientation analytics module. |
| Corner | Missing | — | No corner event detector. |
| Corner Attacking | Missing | — | No attacking-corner scenario detector. |
| Throw-in | Missing | — | No throw-in event detector. |
| Free Kick | Missing | — | No free-kick event detector. |
| Goal Kick | Missing | — | No goal-kick event detector. |

## 4) Tagging and annotation system

| Veo feature | Status | Current code path(s) | Notes |
|---|---|---|---|
| Manual clip tagging | Partial | `src/ui/server.py` (`/events` confirm/reject/add/delete), `src/ui/static/app.js` | Manual event confirmation/addition exists, but no general clip tag entity/taxonomy. |
| Tactical tag taxonomy | Missing | — | No persisted tag taxonomy or tactical tag catalog. |
| Clip creation (time range) | Partial | `src/events/highlights.py`, `src/cli.py` (`HighlightGenerationStage`) | Automatic clip extraction exists; manual arbitrary range clipping is not exposed. |
| Clip annotations (text notes) | Partial | `src/ui/server.py` (`events_confirmed.jsonl` notes), `src/ui/static/app.js` | Notes exist for event confirmations, not a generalized clip annotation model. |
| Tag-based filtering | Missing | — | Filtering exists by event type/status, not by user-defined tags. |
| Visibility controls (creator/team/all) | Missing | — | No ACL/visibility model for tags or clips. |

## 5) Player-level analytics

| Veo feature | Status | Current code path(s) | Notes |
|---|---|---|---|
| Player Moments (per-player clips) | Implemented | `src/events/player_reels.py`, `src/cli.py` (`PlayerHighlightReelsStage`), `src/ui/server.py` | Per-player reels and optional clips are generated and served via API/UI. |
| Player stats across matches | Partial | `src/export/cross_match.py`, `src/ui/static/app.js` (season panel) | Cross-match player trend summaries exist, but full per-player stat model is not implemented. |
| Player share links | Missing | — | No share-link generation flow. |
| Jersey number detection | Implemented | `src/identity/multimodal.py` | Jersey OCR evidence is integrated in multimodal identity fusion (backend availability dependent). |
| Player identity persistence | Implemented | `src/identity/database.py`, `src/identity/matching.py`, `src/cli.py` (`PlayerIdentityStage`) | SQLite-backed cross-run identity persistence and assignment workflows are implemented. |
| Per-player shot map | Missing | — | No player-filtered shot-map renderer. |
| Per-player heat map | Missing | — | No player-filtered heat-map renderer. |
| Per-player pass network | Partial | `src/analytics/team.py` | Pass edges include player/track linkage but no dedicated per-player network artifact/UI view. |

## 6) Cross-match / season analytics

| Veo feature | Status | Current code path(s) | Notes |
|---|---|---|---|
| Key Metrics Dashboard | Implemented | `src/export/cross_match.py`, `src/cli.py` (`CrossMatchReportingStage`), `src/ui/static/app.js` | Season-level aggregates and top-player summaries are exported and displayed. |
| Possession % Trend (over N matches) | Partial | `src/export/cross_match.py` | Trend data is available; dedicated chart rendering is not present. |
| Game Maps aggregated across matches | Missing | — | No multi-match shot/heat/pass map aggregation. |
| Team Comparison Radial | Missing | — | No team radar comparison view. |
| Match Type Filtering | Missing | — | Match metadata lacks type dimension in cross-match reporting. |
| Formation Filtering | Missing | — | Formation metadata/filter pipeline not implemented. |
| Coach Notes (per-team journal) | Missing | — | No coach journal storage or UI workflow. |
| W/L/D Results Tracking | Missing | — | No explicit result tracking layer in cross-match reports. |

## 7) Coach assist (AI)

| Veo feature | Status | Current code path(s) | Notes |
|---|---|---|---|
| Coach Assist tactical insights | Missing | — | No opt-in LLM coach assistant module yet. |
| Automated tactical analysis text | Missing | — | No automatic natural-language tactical summary generation. |

## 8) Video features

| Veo feature | Status | Current code path(s) | Notes |
|---|---|---|---|
| Video playback with overlay | Implemented | `src/export/overlay.py`, `src/ui/server.py` (`/video`), `src/ui/static/app.js` | Overlay playback is supported with canvas-based UI overlays. |
| Speedrun mode | Missing | — | No low-action skip/accelerated segment mode. |
| Video download | Implemented | `src/ui/server.py` (`/api/runs/{run}/video`), run artifacts (`overlay.mp4`) | Download/stream of original or overlay video is available. |
| Highlights download | Implemented | `src/events/highlights.py`, `src/ui/server.py` (player/cross-match export package endpoints) | Highlight artifacts and zip exports are available. |
| Forward/backward navigation | Implemented | `src/ui/static/index.html`, `src/ui/static/app.js` | Frame-step controls, keyboard shortcuts, and timeline seek/jump are available. |
| Multi-view layout toggle | Missing | — | No split/multi-view layout mode in current UI. |

## 9) Lineup and formation

| Veo feature | Status | Current code path(s) | Notes |
|---|---|---|---|
| Formation Selection | Missing | — | No formation metadata selection in run config/UI. |
| Starting Lineup | Missing | — | No starting lineup model stored per run. |
| Formation tracking per match | Missing | — | No formation persistence/filtering across runs. |

## 10) Veo API integration

| Veo API capability | Status | Current code path(s) | Notes |
|---|---|---|---|
| Videos API integration | Partial | `src/integrations/veo_api.py` | Opt-in Veo client now supports basic video list/get/create/update operations via interface-boundary transport. |
| Users API integration | Missing | — | No user sync integration. |
| Groups/Communities API integration | Missing | — | No organization/group sync integration. |
| Comments API integration | Missing | — | No comment sync flow. |
| Tagsets API integration | Missing | — | No tag import/export bridge. |
| OAuth/Bearer auth client | Partial | `src/integrations/veo_api.py` | Bearer-token auth is implemented; full OAuth flow/token refresh is still missing. |

## Immediate parity targets (from current gaps)

1. Add set-piece + pass event families (`passes.py`, `set_pieces.py`) and wire into `events.jsonl`.
2. Add visualization package (`src/export/visualizations/`) for shot/heat/pass/tactical maps.
3. Extend cross-match with result + match metadata dimensions (W/L/D, match type, formation).
4. Introduce tag persistence and tag filtering endpoints/UI (separate from event confirmations).
