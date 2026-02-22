# UI/UX Readiness Checklist

This checklist tracks concrete end-user UX gaps found in the web app and the implementation status.

## Pipeline Execution UX

- [x] Users can start a **full pipeline run** from the web UI (not just review existing runs).
- [x] Users can queue **multiple videos** in one action.
- [x] Users can choose config, resume mode, and overlay mode directly in UI.
- [x] Users can see **live job state** (queued/running/succeeded/failed) with stage progress.
- [x] Users can open completed runs directly from job cards.

## Pipeline Job Reliability UX

- [x] Job history is persisted to disk and survives server restarts.
- [x] Interrupted queued/running jobs are surfaced clearly after restart.
- [x] Users can cancel queued jobs immediately.
- [x] Users can request cancellation for running jobs (applies before next stage).
- [x] Users can retry failed/cancelled jobs with one click.
- [x] Users can duplicate any completed job settings to run again.

## Visual UX

- [x] Added a dedicated **Pipeline Studio** section with clear controls.
- [x] Updated visual hierarchy (cards, badges, progress bars) for scanability.
- [x] Improved typography + color system for modern look and readability.
- [x] Responsive behavior maintained for mobile/tablet layouts.

## Safety/Robustness UX

- [x] Run names are sanitized and de-duplicated to avoid collisions.
- [x] Conflicting queued/running run names are blocked with actionable errors.
- [x] Empty/missing file input and missing config paths return clear user-facing errors.
- [x] UI run selection no longer relies on implicit global browser `event` object.

## Validation

- [x] Unit tests cover queue/config endpoints and multi-video queue behavior.
- [x] Full project test suite passes after UI/backend job UX changes.
