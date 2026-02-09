# Veo Guide to Code Gap Matrix

Last verified: 2026-02-09
Source guide: `veo_analytics_implementation_guide.md`

## Status legend
- `Implemented`: production code path exists and is wired into pipeline/UI.
- `Partial`: foundational code exists, but parity is incomplete (missing subtype granularity, visualization, UX, or export integration).
- `Missing`: no direct implementation in current codebase.

## Summary
- Features mapped: `75`
- `Implemented`: `38`
- `Partial`: `19`
- `Missing`: `18`

## 1) Match-level statistics

| Veo feature | Status | Current code path(s) | Notes |
|---|---|---|---|
| Possession % | Implemented | `src/analytics/team.py`, `src/cli.py` (`TeamAnalyticsStage`) | Possession summary + timeline exported in `team_analytics.json` and `team_possession_timeline.csv`. |
| Passes Completed | Partial | `src/events/passes.py`, `src/events/detection.py`, `src/analytics/match_stats.py`, `src/cli.py` (`EventDetectionStage`, `MatchStatsStage`) | Pass events and totals are inferred and exported, but attempted/completed distinction is still heuristic. |
| Shots (count) | Implemented | `src/events/detection.py`, `src/cli.py` (`EventDetectionStage`) | Shot events are emitted to `events.jsonl` and counted in `summary.json` and `match_stats.json`. |
| Goals (count) | Implemented | `src/events/detection.py`, `src/vision/field/goal_detector.py`, `src/cli.py` | Goal events are emitted and added to `score_timeline.json`. |
| Free Kicks | Implemented | `src/events/set_pieces.py`, `src/events/detection.py`, `src/cli.py` (`EventDetectionStage`) | Free-kick events are inferred as a set-piece subtype with confidence and provenance. |
| Throw-ins | Implemented | `src/events/set_pieces.py`, `src/events/detection.py`, `src/cli.py` (`EventDetectionStage`) | Throw-in events are inferred and included in event + stats artifacts. |
| Corners | Implemented | `src/events/set_pieces.py`, `src/events/detection.py`, `src/cli.py` (`EventDetectionStage`) | Corner-kick events are inferred and included in event + stats artifacts. |
| Penalties | Missing | — | No explicit penalty detector/event subtype yet. |
| Score Tracking (live) | Partial | `src/cli.py` (`EventDetectionStage`), `src/ui/server.py` (`/timeline`), `src/ui/static/app.js` | Score timeline exists post-analysis, but there is no real-time streaming inference. |

## 2) Visualizations (Analytics 2.0 style)

| Veo feature | Status | Current code path(s) | Notes |
|---|---|---|---|
| Shot Map (field diagram) | Partial | `src/export/visualizations/shot_map.py`, `src/export/visualizations/__init__.py` | Shot map renderer exists with team/player filtering, but it is not yet exposed as a first-class UI/API endpoint. |
| Pass Location Map | Implemented | `src/export/visualizations/pass_map.py`, `src/ui/server.py` (`/api/runs/{run_name}/visualizations/pass_map`), `src/ui/static/app.js` | Pass map renderer and UI toggles are implemented. |
| Possession Location (by thirds) | Implemented | `src/analytics/team.py`, `src/cli.py` (`TeamAnalyticsStage`) | Territory bins and control shares are exported in `team_territory_zones.csv`. |
| Pass Strings | Missing | — | No dedicated pass-chain string visualization artifact yet. |
| Heat Map | Partial | `src/export/visualizations/heat_map.py`, `src/export/visualizations/__init__.py`, `src/vision/field/normalization.py` | Heat-map rendering exists from normalized coordinates, but UI/API integration is incomplete. |
| Tactical Map (live positions) | Implemented | `src/export/visualizations/tactical_map.py`, `src/ui/server.py` (`/api/runs/{run_name}/visualizations/tactical_map`), `src/ui/static/app.js` | Tactical map renderer and UI integration are implemented. |
| Comparison Radial Chart | Partial | `src/analytics/season.py`, `src/export/cross_match.py` | Radar-ready aggregates are generated, but no dedicated radial chart rendering exists in UI/export. |
| Possession % Trend Chart | Partial | `src/analytics/season.py`, `src/export/cross_match.py`, `src/ui/static/app.js` | Possession trend data is exported and summarized; a dedicated chart component is still limited. |

## 3) Event detection (AI-tagged)

| Veo event type | Status | Current code path(s) | Notes |
|---|---|---|---|
| Goal | Implemented | `src/events/detection.py`, `src/cli.py` (`EventDetectionStage`) | Goal events generated with confidence and timeline updates. |
| Shot on Goal | Partial | `src/events/detection.py` | Shot direction/target metadata exists, but there is no distinct `shot_on_goal` event type. |
| Shot (off target) | Partial | `src/events/detection.py` | General `shot` events exist, but not split into explicit on/off-target classes. |
| Celebration | Implemented | `src/events/celebration_detection.py`, `src/events/detection.py` | Celebration signal is integrated in shot/goal inference flow. |
| Kickoff | Implemented | `src/events/set_pieces.py`, `src/events/detection.py` | Kickoff is emitted as a set-piece subtype with confidence/provenance. |
| Build Up | Implemented | `src/events/tactical.py`, `src/events/detection.py` | Build-up events are inferred and written to `events.jsonl`. |
| Pressing | Implemented | `src/events/tactical.py`, `src/events/detection.py` | Pressing events are inferred from pressing timeline segments and emitted as events. |
| Defending | Implemented | `src/events/tactical.py`, `src/events/detection.py` | Defensive-shape events are inferred and emitted as tactical events. |
| Defending the Box | Missing | — | No explicit box-defense subtype yet. |
| Defending (Success) | Missing | — | No success/failure defending outcome taxonomy yet. |
| Defending, Poor Recovery | Missing | — | No poor-recovery defensive subtype yet. |
| Counter Press | Missing | — | No explicit counter-press subtype yet. |
| Transition to Defend | Partial | `src/events/tactical.py`, `src/events/detection.py` | Generic transition events exist, but no explicit attack-to-defend directional subtype. |
| Body Shape | Missing | — | No body-orientation analytics module yet. |
| Corner | Implemented | `src/events/set_pieces.py`, `src/events/detection.py` | Corner-kick set-piece events are emitted with metadata. |
| Corner Attacking | Missing | — | No dedicated attacking-corner scenario classifier yet. |
| Throw-in | Implemented | `src/events/set_pieces.py`, `src/events/detection.py` | Throw-in set-piece events are emitted with metadata. |
| Free Kick | Implemented | `src/events/set_pieces.py`, `src/events/detection.py` | Free-kick set-piece events are emitted with metadata. |
| Goal Kick | Implemented | `src/events/set_pieces.py`, `src/events/detection.py` | Goal-kick set-piece events are emitted with metadata. |

## 4) Tagging and annotation system

| Veo feature | Status | Current code path(s) | Notes |
|---|---|---|---|
| Manual clip tagging | Implemented | `src/identity/database.py`, `src/ui/server.py` (`/api/runs/{run_name}/tags` CRUD), `src/ui/static/app.js` | Persisted tag entities with run/time/category metadata are created/updated/deleted via API and UI. |
| Tactical tag taxonomy | Partial | `src/identity/database.py`, `src/ui/server.py`, `src/ui/static/app.js` | Categories are supported, but there is no enforced tactical taxonomy catalog yet. |
| Clip creation (time range) | Partial | `src/events/highlights.py`, `src/cli.py` (`HighlightGenerationStage`) | Automatic clip extraction exists; manual arbitrary clip export ranges are still limited. |
| Clip annotations (text notes) | Implemented | `src/identity/database.py` (`tags.notes`), `src/ui/server.py`, `src/ui/static/app.js` | Text notes are persisted on tags and surfaced in the tagging UI. |
| Tag-based filtering | Implemented | `src/ui/server.py` (`/api/runs/{run_name}/tags` filters), `src/ui/static/app.js` | Filtering by tag label/category/source is implemented in API and UI. |
| Visibility controls (creator/team/all) | Missing | — | No ACL/visibility model for tags or clips. |

## 5) Player-level analytics

| Veo feature | Status | Current code path(s) | Notes |
|---|---|---|---|
| Player Moments (per-player clips) | Implemented | `src/events/player_reels.py`, `src/cli.py` (`PlayerHighlightReelsStage`), `src/ui/server.py` | Per-player reels and optional clips are generated and served via API/UI. |
| Player stats across matches | Implemented | `src/analytics/player.py`, `src/cli.py` (`PlayerAnalyticsStage`), `src/export/cross_match.py`, `src/ui/static/app.js` | Per-player metrics are generated and surfaced in cross-match/season views. |
| Player share links | Missing | — | No share-link generation flow. |
| Jersey number detection | Implemented | `src/identity/multimodal.py` | Jersey OCR evidence is integrated in multimodal identity fusion (backend availability dependent). |
| Player identity persistence | Implemented | `src/identity/database.py`, `src/identity/matching.py`, `src/cli.py` (`PlayerIdentityStage`) | SQLite-backed cross-run identity persistence and assignment workflows are implemented. |
| Per-player shot map | Partial | `src/export/visualizations/shot_map.py`, `src/export/visualizations/__init__.py` | Renderer supports player-level filtering, but no dedicated per-player shot-map UI flow is wired. |
| Per-player heat map | Partial | `src/export/visualizations/heat_map.py`, `src/export/visualizations/__init__.py` | Renderer supports player/team modes, but no dedicated per-player heat-map UI flow is wired. |
| Per-player pass network | Partial | `src/export/visualizations/pass_map.py`, `src/analytics/team.py` | Player-linked pass context exists, but full per-player network UX is still limited. |

## 6) Cross-match / season analytics

| Veo feature | Status | Current code path(s) | Notes |
|---|---|---|---|
| Key Metrics Dashboard | Implemented | `src/export/cross_match.py`, `src/cli.py` (`CrossMatchReportingStage`), `src/ui/static/app.js` | Season-level aggregates and top-player summaries are exported and displayed. |
| Possession % Trend (over N matches) | Partial | `src/analytics/season.py`, `src/export/cross_match.py`, `src/ui/static/app.js` | Trend values are generated and surfaced, but visualization polish is still limited. |
| Game Maps aggregated across matches | Missing | — | No multi-match shot/heat/pass map aggregation artifact yet. |
| Team Comparison Radial | Partial | `src/analytics/season.py`, `src/export/cross_match.py` | Radar-ready team aggregates exist, but no dedicated radial chart view yet. |
| Match Type Filtering | Implemented | `src/analytics/season.py`, `src/export/cross_match.py`, `src/ui/static/app.js` | Match-type metadata filtering is supported in season/cross-match outputs. |
| Formation Filtering | Implemented | `src/analytics/season.py`, `src/export/cross_match.py`, `src/ui/static/app.js` | Formation filtering is supported when formation metadata is present. |
| Coach Notes (per-team journal) | Missing | — | No coach journal/notebook persistence workflow yet. |
| W/L/D Results Tracking | Implemented | `src/analytics/season.py`, `src/export/cross_match.py`, `src/ui/static/app.js` | W/L/D summaries and result-derived metrics are exported and displayed. |

## 7) Coach assist (AI)

| Veo feature | Status | Current code path(s) | Notes |
|---|---|---|---|
| Coach Assist tactical insights | Implemented | `src/analytics/coach_assist.py`, `src/cli.py` (`CoachAssistStage`) | Opt-in provider abstraction is implemented and disabled by default. |
| Automated tactical analysis text | Implemented | `src/analytics/coach_assist.py` (`HeuristicCoachAssistProvider`) | Deterministic natural-language recommendations are generated from artifacts. |

## 8) Video features

| Veo feature | Status | Current code path(s) | Notes |
|---|---|---|---|
| Video playback with overlay | Implemented | `src/export/overlay.py`, `src/ui/server.py` (`/video`), `src/ui/static/app.js` | Overlay playback is supported with canvas-based UI overlays. |
| Speedrun mode | Implemented | `src/ui/server.py` (`/api/runs/{run_name}/playback/speedrun`), `src/ui/static/app.js` | High-action playback windows and low-action skipping mode are implemented. |
| Video download | Implemented | `src/ui/server.py` (`/api/runs/{run_name}/video`), run artifacts (`overlay.mp4`) | Download/stream of original or overlay video is available. |
| Highlights download | Implemented | `src/events/highlights.py`, `src/ui/server.py` (player/cross-match export package endpoints) | Highlight artifacts and zip exports are available. |
| Forward/backward navigation | Implemented | `src/ui/static/index.html`, `src/ui/static/app.js` | Frame-step controls, keyboard shortcuts, and timeline seek/jump are available. |
| Multi-view layout toggle | Implemented | `src/ui/static/app.js`, `src/ui/static/index.html` | Split/stacked viewer layout toggle is implemented. |

## 9) Lineup and formation

| Veo feature | Status | Current code path(s) | Notes |
|---|---|---|---|
| Formation Selection | Missing | — | No first-class formation selection workflow in run setup UI/config. |
| Starting Lineup | Missing | — | No starting-lineup model persisted per run. |
| Formation tracking per match | Partial | `src/export/cross_match.py`, `src/analytics/season.py` | Formation metadata can be ingested/filterable when present, but no automatic formation tracking pipeline exists. |

## 10) Veo API integration

| Veo API capability | Status | Current code path(s) | Notes |
|---|---|---|---|
| Videos API integration | Partial | `src/integrations/veo_api.py` | Opt-in Veo client supports list/get/create/update operations behind interface boundaries. |
| Users API integration | Missing | — | No user sync integration. |
| Groups/Communities API integration | Missing | — | No organization/group sync integration. |
| Comments API integration | Missing | — | No comment sync flow. |
| Tagsets API integration | Missing | — | No tag import/export bridge. |
| OAuth/Bearer auth client | Partial | `src/integrations/veo_api.py` | Bearer-token auth is implemented; full OAuth flow/token refresh remains missing. |

## Immediate parity targets (from current gaps)

1. Add penalty + advanced defensive subtype detection (box defense, counter-press, defensive outcomes).
2. Add dedicated radial/trend chart components and pass-string visualization in the UI/export layer.
3. Add formation and lineup authoring/persistence workflows (not only metadata filtering).
4. Expand Veo API coverage beyond videos/auth (users, groups, comments, tagsets).
