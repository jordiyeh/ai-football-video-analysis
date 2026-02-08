"""Optional coach-assist insight generation with provider abstraction."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable


COACH_ASSIST_SCHEMA_VERSION = "1.0"


def _cfg_value(config: Any, key: str, default: Any) -> Any:
    """Read config key from model/object/dict with fallback."""
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely cast arbitrary values to float."""
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    """Safely cast arbitrary values to int."""
    try:
        return int(value)
    except Exception:
        return default


def _clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp value to [minimum, maximum]."""
    return max(minimum, min(maximum, value))


def _priority_rank(priority: str) -> int:
    """Return sortable rank for insight priority."""
    normalized = str(priority).strip().lower()
    if normalized == "high":
        return 0
    if normalized == "medium":
        return 1
    return 2


def _normalize_priority(value: Any) -> str:
    """Normalize arbitrary priority labels to high/medium/low."""
    normalized = str(value).strip().lower()
    if normalized in {"high", "h"}:
        return "high"
    if normalized in {"medium", "med", "m"}:
        return "medium"
    return "low"


def _as_dict(value: Any) -> dict[str, Any]:
    """Return value when dict-like, else empty dict."""
    if isinstance(value, dict):
        return value
    return {}


@dataclass(frozen=True)
class CoachAssistContext:
    """Input bundle for provider-backed coach assist generation."""

    match_stats: dict[str, Any]
    team_analytics: dict[str, Any]
    events: list[dict[str, Any]]


@runtime_checkable
class CoachAssistProvider(Protocol):
    """Provider abstraction for tactical insight generation."""

    name: str
    requires_cloud: bool

    def generate(
        self,
        context: CoachAssistContext,
        *,
        max_insights: int,
        min_confidence: float,
    ) -> list[dict[str, Any]]:
        """Generate coach insights from analysis artifacts."""
        ...


class HeuristicCoachAssistProvider:
    """Deterministic local provider that never performs network I/O."""

    name = "heuristic"
    requires_cloud = False

    def generate(
        self,
        context: CoachAssistContext,
        *,
        max_insights: int,
        min_confidence: float,
    ) -> list[dict[str, Any]]:
        """Generate insights from match_stats, team_analytics, and events."""
        match_stats = _as_dict(context.match_stats)
        team_analytics = _as_dict(context.team_analytics)
        events = context.events if isinstance(context.events, list) else []

        teams = _as_dict(match_stats.get("teams"))
        ours = _as_dict(teams.get("ours"))

        totals = _as_dict(match_stats.get("totals"))
        ours_shots = max(0, _safe_int(ours.get("shots"), _safe_int(totals.get("shots"), 0)))
        ours_goals = max(0, _safe_int(ours.get("goals"), _safe_int(totals.get("goals"), 0)))
        ours_set_pieces = max(0, _safe_int(ours.get("set_pieces"), _safe_int(totals.get("set_pieces"), 0)))

        insights: list[dict[str, Any]] = []

        if ours_shots >= 4:
            conversion = ours_goals / max(1.0, float(ours_shots))
            if conversion < 0.18:
                confidence = _clamp(0.58 + min(0.25, (ours_shots - 4) * 0.03), 0.0, 1.0)
                if confidence >= min_confidence:
                    insights.append(
                        {
                            "title": "Improve shot quality in the final third",
                            "priority": "high",
                            "confidence": confidence,
                            "recommendation": (
                                "Prioritize one-touch combinations or cutback patterns before finishing "
                                "to increase conversion quality."
                            ),
                            "evidence": {
                                "shots": ours_shots,
                                "goals": ours_goals,
                                "conversion_rate": round(conversion, 3),
                            },
                            "provenance": {"rule": "finishing_efficiency"},
                        }
                    )

        possession = _as_dict(team_analytics.get("possession"))
        possession_teams = _as_dict(possession.get("teams"))
        ours_possession_share = _safe_float(
            _as_dict(possession_teams.get("ours")).get("share"),
            default=-1.0,
        )
        if 0.0 <= ours_possession_share < 0.47:
            possession_deficit = 0.5 - ours_possession_share
            confidence = _clamp(0.54 + possession_deficit * 1.3, 0.0, 1.0)
            if confidence >= min_confidence:
                insights.append(
                    {
                        "title": "Protect buildup under pressure",
                        "priority": "medium",
                        "confidence": confidence,
                        "recommendation": (
                            "Use a support triangle around the first receiver and create a clear third-man "
                            "option to reduce direct turnovers."
                        ),
                        "evidence": {
                            "ours_possession_share": round(ours_possession_share, 3),
                            "target_share": 0.5,
                        },
                        "provenance": {"rule": "possession_control"},
                    }
                )

        pressing = _as_dict(team_analytics.get("pressing"))
        pressing_teams = _as_dict(pressing.get("teams"))
        ours_press_rate = _safe_float(
            _as_dict(pressing_teams.get("ours")).get("high_press_rate"),
            default=-1.0,
        )
        opponent_press_rate = _safe_float(
            _as_dict(pressing_teams.get("opponent")).get("high_press_rate"),
            default=-1.0,
        )
        if (
            0.0 <= ours_press_rate <= 1.0
            and 0.0 <= opponent_press_rate <= 1.0
            and ours_press_rate + 0.06 < opponent_press_rate
        ):
            press_gap = opponent_press_rate - ours_press_rate
            confidence = _clamp(0.52 + press_gap * 1.2, 0.0, 1.0)
            if confidence >= min_confidence:
                insights.append(
                    {
                        "title": "Raise first-line pressing intensity",
                        "priority": "medium",
                        "confidence": confidence,
                        "recommendation": (
                            "Coordinate the front line trigger so the nearest midfielder steps early and "
                            "closes central passing lanes."
                        ),
                        "evidence": {
                            "ours_high_press_rate": round(ours_press_rate, 3),
                            "opponent_high_press_rate": round(opponent_press_rate, 3),
                        },
                        "provenance": {"rule": "pressing_gap"},
                    }
                )

        if ours_set_pieces >= 5 and ours_goals == 0:
            confidence = _clamp(0.5 + min(0.2, ours_set_pieces * 0.02), 0.0, 1.0)
            if confidence >= min_confidence:
                insights.append(
                    {
                        "title": "Increase set-piece conversion threat",
                        "priority": "medium",
                        "confidence": confidence,
                        "recommendation": (
                            "Rehearse one near-post routine and one late-arriving back-post routine, "
                            "then rotate takers to vary delivery profile."
                        ),
                        "evidence": {
                            "set_pieces": ours_set_pieces,
                            "goals": ours_goals,
                        },
                        "provenance": {"rule": "set_piece_return"},
                    }
                )

        event_counts = Counter()
        for row in events:
            if not isinstance(row, dict):
                continue
            event_type = row.get("event_type")
            if event_type is None:
                continue
            event_counts[str(event_type)] += 1

        transitions = int(event_counts.get("transition", 0))
        if transitions >= 5:
            confidence = _clamp(0.46 + min(0.25, transitions * 0.02), 0.0, 1.0)
            if confidence >= min_confidence:
                insights.append(
                    {
                        "title": "Stabilize rest-defense during transitions",
                        "priority": "low",
                        "confidence": confidence,
                        "recommendation": (
                            "Keep one holding midfielder deeper when fullbacks advance to protect central "
                            "counter lanes."
                        ),
                        "evidence": {
                            "transitions_detected": transitions,
                        },
                        "provenance": {"rule": "transition_volume"},
                    }
                )

        if not insights:
            insights.append(
                {
                    "title": "No major tactical outlier detected",
                    "priority": "low",
                    "confidence": min_confidence,
                    "recommendation": (
                        "Use video review to validate off-ball spacing and communication cues before "
                        "changing match plan."
                    ),
                    "evidence": {"rules_triggered": 0},
                    "provenance": {"rule": "fallback"},
                }
            )

        insights.sort(key=lambda row: (_priority_rank(str(row.get("priority", "low"))), -_safe_float(row.get("confidence"))))
        return insights[:max_insights]


class CloudCoachAssistProvider:
    """
    Cloud-capable provider facade.

    Network behavior is delegated to an injected callable. This class itself
    performs no direct HTTP calls.
    """

    requires_cloud = True

    def __init__(
        self,
        request_fn: Callable[[dict[str, Any]], Any] | None = None,
        *,
        provider_name: str = "cloud",
    ) -> None:
        self._request_fn = request_fn
        self.name = provider_name

    def generate(
        self,
        context: CoachAssistContext,
        *,
        max_insights: int,
        min_confidence: float,
    ) -> list[dict[str, Any]]:
        """Generate insights through injected cloud request callback."""
        if self._request_fn is None:
            return []

        response = self._request_fn(
            {
                "match_stats": context.match_stats,
                "team_analytics": context.team_analytics,
                "events": context.events,
                "max_insights": max_insights,
                "min_confidence": min_confidence,
            }
        )

        rows: list[dict[str, Any]]
        if isinstance(response, dict):
            response_rows = response.get("insights", [])
            rows = response_rows if isinstance(response_rows, list) else []
        elif isinstance(response, list):
            rows = [row for row in response if isinstance(row, dict)]
        else:
            rows = []

        return rows[:max_insights]


def create_coach_assist_provider(
    config: Any,
    *,
    cloud_request: Callable[[dict[str, Any]], Any] | None = None,
) -> CoachAssistProvider:
    """Create provider implementation from config."""
    provider_name = str(_cfg_value(config, "provider", "heuristic")).strip().lower()
    if provider_name == "cloud":
        return CloudCoachAssistProvider(request_fn=cloud_request, provider_name="cloud")
    return HeuristicCoachAssistProvider()


def _normalize_insight_row(
    row: dict[str, Any],
    *,
    idx: int,
    provider_name: str,
    min_confidence: float,
) -> dict[str, Any] | None:
    """Normalize provider output row into stable coach-assist insight schema."""
    confidence = _clamp(_safe_float(row.get("confidence"), default=min_confidence), 0.0, 1.0)
    if confidence < min_confidence:
        return None

    evidence = _as_dict(row.get("evidence"))
    provenance = _as_dict(row.get("provenance"))
    title = str(row.get("title", "")).strip() or f"Insight {idx}"
    recommendation = str(row.get("recommendation", "")).strip()
    priority = _normalize_priority(row.get("priority", "low"))

    normalized_provenance = {
        "provider": provider_name,
        **provenance,
    }

    return {
        "insight_id": str(row.get("insight_id", f"insight_{idx:03d}")),
        "title": title,
        "priority": priority,
        "confidence": round(confidence, 3),
        "recommendation": recommendation,
        "evidence": evidence,
        "provenance": normalized_provenance,
    }


def build_coach_assist_report(
    *,
    match_stats: dict[str, Any] | None,
    team_analytics: dict[str, Any] | None,
    events: list[dict[str, Any]] | None,
    config: Any,
    provider: CoachAssistProvider | None = None,
    cloud_request: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """
    Build coach-assist insight artifact payload.

    When disabled, returns a schema-valid payload with no insights.
    Cloud providers are blocked unless `allow_cloud` is explicitly true.
    """
    enabled = bool(_cfg_value(config, "enabled", False))
    allow_cloud = bool(_cfg_value(config, "allow_cloud", False))
    max_insights = max(1, _safe_int(_cfg_value(config, "max_insights", 5), default=5))
    min_confidence = _clamp(
        _safe_float(_cfg_value(config, "min_confidence", 0.45), default=0.45),
        0.0,
        1.0,
    )

    resolved_provider = provider or create_coach_assist_provider(
        config,
        cloud_request=cloud_request,
    )
    provider_name = getattr(resolved_provider, "name", "heuristic")
    provider_cloud = bool(getattr(resolved_provider, "requires_cloud", False))

    report: dict[str, Any] = {
        "schema_version": COACH_ASSIST_SCHEMA_VERSION,
        "enabled": enabled,
        "provider": provider_name,
        "allow_cloud": allow_cloud,
        "insights": [],
        "summary": {
            "status": "disabled" if not enabled else "pending",
            "reason": "config_disabled" if not enabled else None,
            "insights_generated": 0,
            "cloud_used": False,
        },
    }

    if not enabled:
        return report

    if provider_cloud and not allow_cloud:
        report["summary"] = {
            "status": "skipped",
            "reason": "cloud_provider_not_allowed",
            "insights_generated": 0,
            "cloud_used": False,
        }
        return report

    try:
        context = CoachAssistContext(
            match_stats=_as_dict(match_stats),
            team_analytics=_as_dict(team_analytics),
            events=events if isinstance(events, list) else [],
        )
        generated_rows = resolved_provider.generate(
            context,
            max_insights=max_insights,
            min_confidence=min_confidence,
        )
    except Exception as exc:
        report["summary"] = {
            "status": "error",
            "reason": "provider_error",
            "error": str(exc),
            "insights_generated": 0,
            "cloud_used": False,
        }
        return report

    normalized: list[dict[str, Any]] = []
    for idx, row in enumerate(generated_rows, start=1):
        if not isinstance(row, dict):
            continue
        normalized_row = _normalize_insight_row(
            row,
            idx=idx,
            provider_name=provider_name,
            min_confidence=min_confidence,
        )
        if normalized_row is not None:
            normalized.append(normalized_row)

    normalized.sort(
        key=lambda row: (_priority_rank(row["priority"]), -_safe_float(row["confidence"])),
    )
    normalized = normalized[:max_insights]

    report["insights"] = normalized
    report["summary"] = {
        "status": "ready",
        "reason": None,
        "insights_generated": len(normalized),
        "cloud_used": provider_cloud,
    }
    return report
