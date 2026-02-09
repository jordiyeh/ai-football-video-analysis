# Task 002: Bump `actions/setup-python` to v6

## Source
- GitHub open work item: `#2`
- URL: https://github.com/jordiyeh/ai-football-video-analysis/pull/2

## Requirements
- Update CI workflows from `actions/setup-python@v5` to `actions/setup-python@v6`.
- Verify the workflow file remains valid.

## Verification
- `python3 - <<'PY' ... yaml.safe_load('.github/workflows/ci.yml') ... PY`
- `.venv_linux/bin/python -m pytest -q` (719 passed, 3 skipped)

## Status: COMPLETE
