"""Unit tests for opt-in Veo API integration client."""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from src.config.schemas import PipelineConfig
from src.integrations.veo_api import (
    VeoAPIClient,
    VeoAPICloudDisabledError,
    VeoAPIConfigurationError,
    VeoAPIDisabledError,
    VeoAPIRequestError,
    VeoAPIResponse,
    VeoAPIUnauthorizedError,
)


class RecordingTransport:
    """Test transport that records requests and returns queued responses."""

    def __init__(self, responses: list[VeoAPIResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def request(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        params: Mapping[str, Any] | None = None,
        json_body: Any = None,
        timeout_seconds: float = 15.0,
    ) -> VeoAPIResponse:
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers),
                "params": dict(params) if params else None,
                "json_body": json_body,
                "timeout_seconds": timeout_seconds,
            }
        )
        if self._responses:
            return self._responses.pop(0)
        return VeoAPIResponse(status_code=200, data={"ok": True}, headers={})


def test_pipeline_config_disables_veo_api_by_default() -> None:
    """Pipeline config should default Veo API integration to cloud-safe off."""
    cfg = PipelineConfig()

    assert cfg.veo_api.enabled is False
    assert cfg.veo_api.allow_cloud is False
    assert cfg.veo_api.base_url == "https://api.veo.co"
    assert cfg.veo_api.api_token is None


def test_client_blocks_when_integration_is_disabled() -> None:
    """Client should reject requests when integration is disabled."""
    transport = RecordingTransport(responses=[])
    client = VeoAPIClient(
        config={
            "enabled": False,
            "allow_cloud": True,
            "api_token": "token",
        },
        transport=transport,
    )

    with pytest.raises(VeoAPIDisabledError):
        client.list_videos()
    assert transport.calls == []


def test_client_blocks_when_cloud_access_not_allowed() -> None:
    """Client should reject requests when allow_cloud is false."""
    transport = RecordingTransport(responses=[])
    client = VeoAPIClient(
        config={
            "enabled": True,
            "allow_cloud": False,
            "api_token": "token",
        },
        transport=transport,
    )

    with pytest.raises(VeoAPICloudDisabledError):
        client.get_video("video-1")
    assert transport.calls == []


def test_client_requires_token_for_authenticated_requests() -> None:
    """Client should fail fast when auth token is missing."""
    transport = RecordingTransport(responses=[])
    client = VeoAPIClient(
        config={
            "enabled": True,
            "allow_cloud": True,
            "api_token": None,
        },
        transport=transport,
    )

    with pytest.raises(VeoAPIConfigurationError):
        client.list_videos()
    assert transport.calls == []


def test_client_performs_basic_read_and_write_operations_with_auth() -> None:
    """Client should send bearer auth and support basic video read/write ops."""
    transport = RecordingTransport(
        responses=[
            VeoAPIResponse(status_code=200, data={"videos": []}, headers={}),
            VeoAPIResponse(status_code=200, data={"id": "v1"}, headers={}),
            VeoAPIResponse(status_code=201, data={"id": "v2"}, headers={}),
            VeoAPIResponse(status_code=200, data={"id": "v2", "title": "Updated"}, headers={}),
        ]
    )
    client = VeoAPIClient(
        config={
            "enabled": True,
            "allow_cloud": True,
            "api_token": "secret-token",
            "base_url": "https://api.veo.test/",
            "timeout_seconds": 7.5,
        },
        transport=transport,
    )

    videos = client.list_videos(limit=25, cursor="next-cursor")
    video = client.get_video("v1")
    created = client.create_video({"title": "Created"})
    updated = client.update_video("v2", {"title": "Updated"})

    assert videos == {"videos": []}
    assert video == {"id": "v1"}
    assert created == {"id": "v2"}
    assert updated == {"id": "v2", "title": "Updated"}

    assert len(transport.calls) == 4
    first = transport.calls[0]
    assert first["method"] == "GET"
    assert first["url"] == "https://api.veo.test/videos"
    assert first["params"] == {"limit": 25, "cursor": "next-cursor"}
    assert first["headers"]["Authorization"] == "Bearer secret-token"
    assert first["timeout_seconds"] == 7.5

    second = transport.calls[1]
    assert second["method"] == "GET"
    assert second["url"] == "https://api.veo.test/videos/v1"
    assert second["params"] is None

    third = transport.calls[2]
    assert third["method"] == "POST"
    assert third["url"] == "https://api.veo.test/videos"
    assert third["json_body"] == {"title": "Created"}

    fourth = transport.calls[3]
    assert fourth["method"] == "PATCH"
    assert fourth["url"] == "https://api.veo.test/videos/v2"
    assert fourth["json_body"] == {"title": "Updated"}


def test_client_raises_unauthorized_error_for_401_or_403() -> None:
    """401/403 responses should raise VeoAPIUnauthorizedError."""
    transport = RecordingTransport(
        responses=[VeoAPIResponse(status_code=401, data={"message": "Unauthorized"}, headers={})]
    )
    client = VeoAPIClient(
        config={
            "enabled": True,
            "allow_cloud": True,
            "api_token": "token",
        },
        transport=transport,
    )

    with pytest.raises(VeoAPIUnauthorizedError) as exc_info:
        client.get_video("v1")

    assert exc_info.value.status_code == 401
    assert exc_info.value.response_data == {"message": "Unauthorized"}


def test_client_raises_request_error_for_non_success_responses() -> None:
    """4xx/5xx responses should raise VeoAPIRequestError."""
    transport = RecordingTransport(
        responses=[VeoAPIResponse(status_code=500, data={"error": "upstream unavailable"}, headers={})]
    )
    client = VeoAPIClient(
        config={
            "enabled": True,
            "allow_cloud": True,
            "api_token": "token",
        },
        transport=transport,
    )

    with pytest.raises(VeoAPIRequestError) as exc_info:
        client.list_videos()

    assert exc_info.value.status_code == 500
    assert exc_info.value.response_data == {"error": "upstream unavailable"}
    assert "upstream unavailable" in str(exc_info.value)
