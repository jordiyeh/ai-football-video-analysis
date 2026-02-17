# Veo Analytics Features — Implementation Guide for Codebase

**Date:** February 8, 2026
**Codebase:** veo-soccer-analysis v0.5.14
**Source:** Live analysis of app.veo.co (Arlington Soccer club) + developer.veo.co.uk API docs

---

## Executive Summary

After thorough analysis of the Veo platform and your existing codebase, I identified **38 distinct analytics features** across the Veo platform. Your codebase already implements **14 of these** in some form. Below is a complete breakdown organized by implementation priority, with specific guidance on which source files to extend.

---

## Feature Matrix: Veo vs. Your Codebase

### Legend
- ✅ **Implemented** — Feature exists in codebase
- 🟡 **Partial** — Foundation exists but incomplete
- ❌ **Missing** — Not implemented, can be added
- 🔗 **API** — Requires Veo API integration

---

## 1. MATCH-LEVEL STATISTICS (Stats Panel)

| Veo Feature | Status | Codebase Location | Notes |
|---|---|---|---|
| Possession % | ✅ | `src/analytics/team.py` → `compute_possession()` | Already computes per-team possession from ball tracking data |
| Passes Completed | 🟡 | `src/events/` | Pass detection is referenced in config but not fully implemented as a standalone metric |
| Shots (count) | ✅ | `src/events/shot.py` | Shot detection via ball trajectory analysis |
| Goals (count) | ✅ | `src/events/shot.py` + `src/vision/field/goal.py` | Goal detection via net entry + goal region detection |
| Free Kicks | ❌ | — | Veo uses AI to detect set-piece events; not in codebase |
| Throw-ins | ❌ | — | Veo auto-tags throw-in events; not in codebase |
| Corners | ❌ | — | Veo auto-tags corner events; not in codebase |
| Penalties | ❌ | — | Veo auto-tags penalty events; not in codebase |
| Score Tracking (live) | 🟡 | `src/events/shot.py` | Goal events detected but no running score display |

**Implementation Priority: HIGH**
Your codebase has possession and shot/goal detection. Adding pass completion counting, set-piece detection (free kicks, throw-ins, corners), and a unified stats comparison view would bring parity with Veo's Stats panel.

**Recommended approach:**
- Extend `src/events/` with new detectors: `freekick.py`, `throwin.py`, `corner.py`
- Set-piece detection can leverage ball position near sidelines/corner arcs combined with play stoppages
- Add a stats aggregation module in `src/analytics/match_stats.py` that compiles all event counts per team

---

## 2. VISUALIZATIONS (Analytics 2.0 Panel)

| Veo Feature | Status | Codebase Location | Notes |
|---|---|---|---|
| Shot Map (field diagram) | 🟡 | `src/vision/field/normalization.py` | Field normalization exists; no shot map visualization |
| Pass Location Map | ❌ | — | Requires pass detection + field-normalized coordinates |
| Possession Location (by thirds) | ✅ | `src/analytics/team.py` → `compute_territory()` | Territory occupancy already bins the field |
| Pass Strings | ❌ | — | Sequential pass chain visualization; not implemented |
| Heat Map | 🟡 | `src/vision/field/normalization.py` | Normalized coordinates available; no heat map renderer |
| Tactical Map (live positions) | 🟡 | `src/vision/track/` + `src/vision/field/` | Tracking + field normalization exist; no real-time overlay |
| Comparison Radial Chart | ❌ | — | Team comparison radar/spider chart; purely a visualization |
| Possession % Trend Chart | ❌ | — | Possession over last N matches; needs cross-match reporting |

**Implementation Priority: HIGH**
Your field normalization (`src/vision/field/normalization.py`) provides the coordinate foundation for most of these visualizations. The main gap is rendering logic.

**Recommended approach:**
- Create `src/export/visualizations/` module with: `shot_map.py`, `heat_map.py`, `pass_map.py`, `tactical_map.py`
- Use matplotlib or Pillow to render field diagrams with overlaid data points
- Territory computation already splits field into thirds; add a visual renderer
- For the web UI, add corresponding endpoints in `src/ui/server.py` and D3.js/Chart.js components

---

## 3. EVENT DETECTION (AI-Tagged Events)

| Veo Event Type | Status | Codebase Location | Notes |
|---|---|---|---|
| Goal | ✅ | `src/events/shot.py` | Detected via ball-net entry |
| Shot on Goal | ✅ | `src/events/shot.py` | Detected via ball trajectory toward goal |
| Shot (off target) | ✅ | `src/events/shot.py` | Detected with lower confidence |
| Celebration | ✅ | `src/events/celebration.py` | Detected via pose cues |
| Kickoff | ❌ | — | Center circle + game start detection |
| Build Up | ❌ | — | Sustained possession in midfield moving forward |
| Pressing | 🟡 | `src/analytics/team.py` → `compute_pressing()` | Pressing metrics computed but not as tagged events |
| Defending | ❌ | — | Defensive actions as tagged events |
| Defending the Box | ❌ | — | Defensive actions in the penalty area |
| Defending (Success) | ❌ | — | Successful defensive recovery |
| Defending, Poor Recovery | ❌ | — | Failed defensive recovery |
| Counter Press | ❌ | — | Immediate press after losing possession |
| Transition to Defend | ❌ | — | Phase transition from attack to defense |
| Body Shape | ❌ | — | Player body orientation analysis |
| Corner | ❌ | — | Corner kick detection |
| Corner Attacking | ❌ | — | Offensive corner kick situations |
| Throw-in | ❌ | — | Throw-in detection from sideline |
| Free Kick | ❌ | — | Free kick set piece detection |
| Goal Kick | ❌ | — | Goal kick detection |

**Implementation Priority: MEDIUM-HIGH**
Your pressing analytics compute metrics but don't create timestamped tagged events. Most of these events can be derived from existing detection + tracking + possession data.

**Recommended approach:**
- Create `src/events/set_pieces.py` for: kickoff, throw-in, corner, free kick, goal kick (detect via ball position near field boundaries + play stoppages)
- Create `src/events/tactical.py` for: build up, pressing events, defending events, transitions (derive from team possession zones, ball movement patterns, player formation changes)
- Convert pressing metrics in `src/analytics/team.py` into frame-level event annotations
- Each event should have: timestamp, team, event_type, confidence, field_coordinates

---

## 4. TAGGING & ANNOTATION SYSTEM

| Veo Feature | Status | Codebase Location | Notes |
|---|---|---|---|
| Manual clip tagging | ❌ | — | Coach can tag clips with custom labels |
| Tactical tag taxonomy | ❌ | — | 50+ predefined tags (2v1, 3rd man run, Attacking Shape, Ball circulation, etc.) |
| Clip creation (time range) | 🟡 | `src/events/highlights.py` | Highlights generated from events; no manual clip creation |
| Clip annotations (text notes) | ❌ | — | Coach text comments on clips |
| Tag-based filtering | ❌ | — | Filter events/clips by tag |
| Visibility controls (creator/team/all) | ❌ | — | Permission levels on clips/tags |

**Veo's Full Tag Taxonomy (observed):**
2v1, 3rd man run, 50/50, Assist, Attack, Attacking in the final third, Attacking Run, Attacking Shape, Attacking Team Shape, Balance, Ball circulation, Ball Watching, Forward as Priority, Free Player, Freekick, Freekick save, FWD Runs, Getting compact to reduce space, GK - Coming out, GK - DS - SP, GK - Feedback, GK - SPRT - AG, Press (Successful), Press (unsuccessful), Press After Loss, Pressing, Pressing Phase, Pressing wide player, Pressing Recovery and Defending, Preventative Defensive Shape, Progression, and more...

**Implementation Priority: MEDIUM**
This is a user-facing feature set rather than analytics computation. It would enhance the web UI significantly.

**Recommended approach:**
- Add a `tags` table to the SQLite database alongside `players.db`
- Create `src/ui/api/tags.py` with CRUD endpoints for clip tags
- Extend `src/ui/server.py` with tag filtering on events/clips endpoints
- Add the full tactical tag taxonomy as a default configuration in `configs/`
- Store annotations as JSON metadata linked to time ranges in the match

---

## 5. PLAYER-LEVEL ANALYTICS

| Veo Feature | Status | Codebase Location | Notes |
|---|---|---|---|
| Player Moments (per-player clips) | ✅ | `src/pipeline/stages.py` → PlayerReelsStage | Player reels already generated |
| Player stats across matches | 🟡 | `src/identity/` | Identity persistence exists; no aggregated stats |
| Player share links | ❌ | — | Shareable player highlight links |
| Jersey number detection | ✅ | `src/identity/multimodal.py` | Jersey OCR already implemented |
| Player identity persistence | ✅ | `src/identity/persistence.py` | SQLite-based cross-match identity |
| Per-player shot map | ❌ | — | Shot map filtered to individual player |
| Per-player heat map | ❌ | — | Heat map filtered to individual player |
| Per-player pass network | 🟡 | `src/analytics/team.py` → `compute_pass_network()` | Pass network exists team-level; not per-player |

**Implementation Priority: MEDIUM**
Your identity system is robust. The gap is aggregating per-player statistics across matches and creating player-specific visualizations.

**Recommended approach:**
- Add per-player stat aggregation to `src/identity/persistence.py`
- Create `src/analytics/player.py` for individual player metrics
- Extend the pass network to track per-player connections
- Add player-filtered views to the visualization modules

---

## 6. CROSS-MATCH / SEASON ANALYTICS (Analytics Studio)

| Veo Feature | Status | Codebase Location | Notes |
|---|---|---|---|
| Key Metrics Dashboard (avg possession, goals, shots, passes) | 🟡 | `src/analytics/` | Individual match metrics exist; no multi-match aggregation dashboard |
| Possession % Trend (bar chart over N matches) | ❌ | — | Time-series visualization across matches |
| Game Maps aggregated across matches | ❌ | — | Shot maps, possession maps across multiple matches |
| Team Comparison Radial | ❌ | — | Spider/radar chart comparing two teams across metrics |
| Match Type Filtering | ❌ | — | Filter analytics by match type (league, friendly, etc.) |
| Formation Filtering | ❌ | — | Filter analytics by formation used |
| Coach Notes (per-team journal) | ❌ | — | Free-text notes attached to analytics views |
| W/L/D Results Tracking | ❌ | — | Match results (win/loss/draw) tracking |

**Implementation Priority: MEDIUM**
Your cross-match reporting foundation exists (mentioned in AGENTS.md as a feature), but the aggregation and visualization layer is thin.

**Recommended approach:**
- Create `src/analytics/season.py` for multi-match metric aggregation
- Add match metadata (result, type, formation) to the run configuration
- Build a dashboard view in the web UI aggregating metrics across `runs/` directory
- Use Chart.js in the frontend for trend charts and radar charts

---

## 7. COACH ASSIST (AI)

| Veo Feature | Status | Codebase Location | Notes |
|---|---|---|---|
| Coach Assist (tactical insights from AI) | ❌ | — | AI-generated text insights like "team exhibited inconsistency in pressing when opponents advanced (60'-74')" |
| Automated tactical analysis text | ❌ | — | Natural language summaries of match patterns |

**Implementation Priority: LOW-MEDIUM**
This is Veo's "Alpha" feature using LLMs to generate tactical insights from match data.

**Recommended approach:**
- Create `src/analytics/coach_assist.py` that feeds match statistics to an LLM API
- Input: possession data, event timeline, pressing metrics, territorial control
- Output: natural language tactical insights and recommendations
- Could use OpenAI/Anthropic API to generate coaching summaries

---

## 8. VIDEO FEATURES

| Veo Feature | Status | Codebase Location | Notes |
|---|---|---|---|
| Video playback with overlay | ✅ | `src/pipeline/stages.py` → OverlayRenderStage | Annotated video rendering |
| Speedrun mode | ❌ | — | Fast-forward through non-action portions |
| Video download | ✅ | `src/export/` | Export to MP4/clips |
| Highlights download | ✅ | `src/events/highlights.py` | Highlight generation |
| Forward/backward navigation | ❌ | — | UI feature for event-to-event navigation |
| Multi-view layout toggle | ❌ | — | Side-by-side/split view modes |

**Implementation Priority: LOW**
These are primarily UI/UX features. The core video processing is solid.

---

## 9. LINEUP & FORMATION

| Veo Feature | Status | Codebase Location | Notes |
|---|---|---|---|
| Formation Selection | ❌ | — | Choose team formation (4-3-3, 4-4-2, etc.) |
| Starting Lineup | ❌ | — | Track which players started |
| Formation tracking per match | ❌ | — | Associate formation with match for filtering |

**Implementation Priority: LOW**
This is metadata management rather than computer vision analytics.

**Recommended approach:**
- Add formation and lineup fields to match metadata in config YAML
- Store in run metadata JSON for cross-match filtering

---

## 10. VEO API INTEGRATION

| API Resource | Capability | Implementation Opportunity |
|---|---|---|
| Videos API | Upload/download videos, get metadata, transcoding status | Import Veo recordings directly into your pipeline |
| Users API | Manage users, roles, organizations | Sync player identities with Veo accounts |
| Groups/Communities API | Manage team groups | Map to your team management |
| Comments API | Read/write comments on videos | Sync coach annotations |
| Tagsets API | Access tag data | Import/export tagged events between platforms |
| Authentication | OAuth Bearer tokens | Programmatic API access |

**API Base URL:** `https://api.veo.co.uk/api/`
**Auth:** Bearer token via OAuth
**Swagger docs:** Available at developer.veo.co.uk

**Recommended approach:**
- Create `src/integrations/veo_api.py` as a Veo API client
- Import match videos directly from Veo into your pipeline
- Export your AI-detected events back to Veo as tags/clips
- Sync player identities between your SQLite DB and Veo's user system

---

## Implementation Roadmap

### Phase 1 — High Impact, Leveraging Existing Code (2-3 weeks)
1. **Pass completion detection** — extend `src/events/` with pass detection using ball trajectory + team assignment
2. **Set-piece event detection** — add free kick, throw-in, corner, goal kick detectors
3. **Shot map visualization** — render field diagram with shot locations using normalized coordinates
4. **Heat map visualization** — aggregate player positions into heat map using tracking data
5. **Unified stats panel** — create match stats summary (all event counts by team)

### Phase 2 — New Capabilities (3-4 weeks)
6. **Tactical event classification** — build up, pressing events, defending events, transitions
7. **Pass strings visualization** — sequential pass chains on field diagram
8. **Per-player analytics** — individual player stats, shot maps, heat maps
9. **Tagging system** — SQLite-backed tag management with UI
10. **Cross-match dashboard** — season-level metrics with trend charts

### Phase 3 — Advanced Features (4-6 weeks)
11. **Coach Assist AI** — LLM-powered tactical insights from match data
12. **Veo API integration** — video import, event sync, player identity sync
13. **Formation tracking** — lineup management and formation-based filtering
14. **Comparison radial charts** — team vs team radar visualizations
15. **Clip annotation system** — coach notes on time-ranged clips

---

## Architecture Recommendations

### New Modules to Create
```
src/
├── analytics/
│   ├── match_stats.py      # Unified match statistics aggregation
│   ├── player.py           # Per-player statistics
│   ├── season.py           # Cross-match/season aggregation
│   └── coach_assist.py     # LLM-powered tactical insights
├── events/
│   ├── set_pieces.py       # Free kick, throw-in, corner, goal kick, kickoff
│   ├── tactical.py         # Build up, pressing events, defending, transitions
│   └── passes.py           # Pass detection and pass chain analysis
├── export/
│   └── visualizations/
│       ├── shot_map.py     # Shot location field diagram
│       ├── heat_map.py     # Player position heat map
│       ├── pass_map.py     # Pass location/strings visualization
│       └── tactical_map.py # Live position overlay
├── integrations/
│   └── veo_api.py          # Veo API client (videos, users, tags, comments)
└── ui/
    └── api/
        └── tags.py         # Tagging/annotation CRUD endpoints
```

### Database Extensions
```sql
-- Add to players.db or create analytics.db
CREATE TABLE match_metadata (
    run_id TEXT PRIMARY KEY,
    result TEXT,           -- 'W', 'L', 'D'
    score_home INTEGER,
    score_away INTEGER,
    match_type TEXT,        -- 'league', 'friendly', 'cup'
    formation TEXT,         -- '4-3-3', '4-4-2', etc.
    opponent TEXT,
    date TEXT
);

CREATE TABLE tags (
    id INTEGER PRIMARY KEY,
    run_id TEXT,
    start_frame INTEGER,
    end_frame INTEGER,
    tag_name TEXT,
    category TEXT,          -- 'tactical', 'set_piece', 'player', 'custom'
    annotation TEXT,
    created_by TEXT,
    visibility TEXT,        -- 'creator', 'team', 'all'
    FOREIGN KEY (run_id) REFERENCES match_metadata(run_id)
);

CREATE TABLE player_match_stats (
    id INTEGER PRIMARY KEY,
    run_id TEXT,
    player_id TEXT,
    minutes_played REAL,
    shots INTEGER,
    goals INTEGER,
    passes_completed INTEGER,
    passes_attempted INTEGER,
    FOREIGN KEY (run_id) REFERENCES match_metadata(run_id)
);
```

---

## Sources

- **Veo Platform:** https://app.veo.co (live exploration of Arlington Soccer club account)
- **Veo Developer API:** https://developer.veo.co.uk
- **Veo API Overview:** https://developer.veo.co.uk/apioverview
- **Veo Camera API Reference:** https://next.api.prod.camera.veo.co/docs/reference
