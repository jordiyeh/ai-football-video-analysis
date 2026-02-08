"""Unit tests for opt-in coach assist analytics."""

from __future__ import annotations

from src.analytics.coach_assist import build_coach_assist_report
from src.config.schemas import PipelineConfig


def _sample_match_stats() -> dict:
    return {
        "schema_version": "1.0",
        "teams": {
            "ours": {"shots": 8, "goals": 1, "set_pieces": 6},
            "opponent": {"shots": 4, "goals": 0, "set_pieces": 2},
        },
        "totals": {"shots": 12, "goals": 1, "passes": 41, "set_pieces": 8},
    }


def _sample_team_analytics() -> dict:
    return {
        "schema_version": "1.0",
        "possession": {
            "teams": {
                "ours": {"share": 0.41},
                "opponent": {"share": 0.59},
            }
        },
        "pressing": {
            "teams": {
                "ours": {"high_press_rate": 0.32},
                "opponent": {"high_press_rate": 0.48},
            }
        },
    }


def test_build_coach_assist_report_disabled_by_default() -> None:
    """Coach assist should return schema-valid disabled payload unless enabled."""
    report = build_coach_assist_report(
        match_stats={},
        team_analytics={},
        events=[],
        config={},
    )

    assert report["enabled"] is False
    assert report["insights"] == []
    assert report["summary"]["status"] == "disabled"
    assert report["summary"]["reason"] == "config_disabled"


def test_pipeline_config_disables_coach_assist_by_default() -> None:
    """Pipeline config should default coach assist to disabled and cloud-safe."""
    cfg = PipelineConfig()

    assert cfg.coach_assist.enabled is False
    assert cfg.coach_assist.allow_cloud is False
    assert cfg.coach_assist.provider == "heuristic"


def test_build_coach_assist_report_generates_local_heuristic_insights() -> None:
    """Enabled heuristic provider should emit deterministic tactical insights."""
    report = build_coach_assist_report(
        match_stats=_sample_match_stats(),
        team_analytics=_sample_team_analytics(),
        events=[{"event_type": "transition"} for _ in range(7)],
        config={
            "enabled": True,
            "provider": "heuristic",
            "allow_cloud": False,
            "max_insights": 4,
            "min_confidence": 0.45,
        },
    )

    assert report["enabled"] is True
    assert report["provider"] == "heuristic"
    assert report["summary"]["status"] == "ready"
    assert report["summary"]["cloud_used"] is False
    assert report["summary"]["insights_generated"] == len(report["insights"])
    assert 1 <= len(report["insights"]) <= 4
    assert all(row["confidence"] >= 0.45 for row in report["insights"])


def test_cloud_provider_is_blocked_when_allow_cloud_false() -> None:
    """Cloud providers must never be called unless allow_cloud is enabled."""

    class ExplodingCloudProvider:
        name = "cloud_test"
        requires_cloud = True

        def generate(self, context, *, max_insights, min_confidence):  # pragma: no cover - should never run
            raise AssertionError("cloud provider should not be called")

    report = build_coach_assist_report(
        match_stats=_sample_match_stats(),
        team_analytics=_sample_team_analytics(),
        events=[],
        config={"enabled": True, "provider": "cloud", "allow_cloud": False},
        provider=ExplodingCloudProvider(),
    )

    assert report["summary"]["status"] == "skipped"
    assert report["summary"]["reason"] == "cloud_provider_not_allowed"
    assert report["summary"]["cloud_used"] is False
    assert report["insights"] == []


def test_cloud_provider_runs_only_when_explicitly_enabled() -> None:
    """Cloud provider may run only when both enabled and allow_cloud are true."""

    class StubCloudProvider:
        name = "cloud_test"
        requires_cloud = True

        def __init__(self) -> None:
            self.called = False

        def generate(self, context, *, max_insights, min_confidence):
            self.called = True
            return [
                {
                    "title": "Cloud insight",
                    "priority": "high",
                    "confidence": 0.77,
                    "recommendation": "Compress central space earlier.",
                    "evidence": {"source": "stub"},
                }
            ]

    provider = StubCloudProvider()
    report = build_coach_assist_report(
        match_stats=_sample_match_stats(),
        team_analytics=_sample_team_analytics(),
        events=[],
        config={"enabled": True, "provider": "cloud", "allow_cloud": True},
        provider=provider,
    )

    assert provider.called is True
    assert report["summary"]["status"] == "ready"
    assert report["summary"]["cloud_used"] is True
    assert report["summary"]["insights_generated"] == 1
    assert report["insights"][0]["title"] == "Cloud insight"


def test_cloud_request_callback_is_not_invoked_when_cloud_disallowed() -> None:
    """Cloud request callback should remain untouched when allow_cloud is false."""
    called = {"count": 0}

    def request_fn(payload: dict) -> dict:
        called["count"] += 1
        return {"insights": [{"title": "Should not run"}]}

    report = build_coach_assist_report(
        match_stats=_sample_match_stats(),
        team_analytics=_sample_team_analytics(),
        events=[],
        config={"enabled": True, "provider": "cloud", "allow_cloud": False},
        cloud_request=request_fn,
    )

    assert called["count"] == 0
    assert report["summary"]["status"] == "skipped"
    assert report["summary"]["reason"] == "cloud_provider_not_allowed"
