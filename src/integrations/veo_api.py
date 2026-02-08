"""Opt-in Veo API client with auth, transport abstraction, and safe defaults."""

from __future__ import annotations

import json
import os
import ssl
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


VEO_API_CLIENT_SCHEMA_VERSION = "1.0"


def _cfg_value(config: Any, key: str, default: Any) -> Any:
    """Read config key from model/object/dict with fallback."""
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _coerce_bool(value: Any, *, default: bool) -> bool:
    """Convert arbitrary values into booleans with a stable default."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _coerce_float(value: Any, *, default: float) -> float:
    """Convert arbitrary values into a float with fallback."""
    try:
        return float(value)
    except Exception:
        return default


def _sanitize_base_url(value: Any, *, default: str) -> str:
    """Ensure base URL is a non-empty string without trailing slash."""
    text = str(value).strip()
    if not text:
        return default.rstrip("/")
    return text.rstrip("/")


def _serialize_query_params(params: Mapping[str, Any] | None) -> str:
    """Serialize query parameters while skipping null values."""
    if not params:
        return ""
    cleaned: dict[str, Any] = {}
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, bool):
            cleaned[key] = "true" if value else "false"
            continue
        cleaned[key] = value
    return urlencode(cleaned, doseq=True)


def _extract_error_message(data: Any, *, fallback: str) -> str:
    """Best-effort extraction of human-readable API error messages."""
    if isinstance(data, dict):
        for key in ("message", "error", "detail", "description"):
            value = data.get(key)
            if value is not None:
                text = str(value).strip()
                if text:
                    return text
    if isinstance(data, str):
        text = data.strip()
        if text:
            return text
    return fallback


def _parse_response_body(raw_body: str, *, content_type: str) -> Any:
    """Parse API body as JSON when possible, fallback to raw text."""
    if raw_body == "":
        return None
    lower_content_type = content_type.lower()
    if "application/json" in lower_content_type:
        try:
            return json.loads(raw_body)
        except json.JSONDecodeError:
            return {"raw": raw_body}
    try:
        return json.loads(raw_body)
    except json.JSONDecodeError:
        return raw_body


class VeoAPIError(RuntimeError):
    """Base exception for Veo API client failures."""


class VeoAPIDisabledError(VeoAPIError):
    """Raised when Veo API usage is disabled in config."""


class VeoAPICloudDisabledError(VeoAPIError):
    """Raised when cloud calls are blocked by allow_cloud safety gate."""


class VeoAPIConfigurationError(VeoAPIError):
    """Raised for invalid or incomplete Veo API configuration."""


class VeoAPITransportError(VeoAPIError):
    """Raised when transport-level connection or I/O errors occur."""


class VeoAPIRequestError(VeoAPIError):
    """Raised for non-success HTTP responses."""

    def __init__(self, message: str, *, status_code: int, response_data: Any = None) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.response_data = response_data


class VeoAPIUnauthorizedError(VeoAPIRequestError):
    """Raised for explicit authorization failures (HTTP 401/403)."""


@dataclass(frozen=True)
class VeoAPIResponse:
    """Transport response container."""

    status_code: int
    data: Any
    headers: dict[str, str]
    raw_body: str = ""


@runtime_checkable
class VeoAPITransport(Protocol):
    """Transport boundary for Veo API requests."""

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
        """Execute one HTTP request and return a normalized response."""
        ...


class UrllibVeoAPITransport:
    """Default stdlib HTTP transport for Veo API interactions."""

    def __init__(self, *, verify_tls: bool = True) -> None:
        self.verify_tls = bool(verify_tls)

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
        """Perform an HTTP request with urllib and normalize response payloads."""
        query = _serialize_query_params(params)
        request_url = url
        if query:
            joiner = "&" if "?" in request_url else "?"
            request_url = f"{request_url}{joiner}{query}"

        request_headers = {str(key): str(value) for key, value in headers.items()}
        body_bytes: bytes | None = None
        if json_body is not None:
            body_bytes = json.dumps(json_body).encode("utf-8")
            has_content_type = any(key.lower() == "content-type" for key in request_headers)
            if not has_content_type:
                request_headers["Content-Type"] = "application/json"

        request = Request(
            request_url,
            data=body_bytes,
            headers=request_headers,
            method=method.upper(),
        )

        ssl_context = None
        if not self.verify_tls:
            ssl_context = ssl._create_unverified_context()  # noqa: S323 - explicit opt-out knob

        try:
            with urlopen(request, timeout=timeout_seconds, context=ssl_context) as response:
                status_code = int(getattr(response, "status", 200))
                raw_body = response.read().decode("utf-8", errors="replace")
                content_type = response.headers.get("Content-Type", "")
                parsed_data = _parse_response_body(raw_body, content_type=content_type)
                return VeoAPIResponse(
                    status_code=status_code,
                    data=parsed_data,
                    headers=dict(response.headers.items()),
                    raw_body=raw_body,
                )
        except HTTPError as err:
            raw_body = err.read().decode("utf-8", errors="replace")
            content_type = ""
            if err.headers is not None:
                content_type = err.headers.get("Content-Type", "")
            parsed_data = _parse_response_body(raw_body, content_type=content_type)
            return VeoAPIResponse(
                status_code=int(err.code),
                data=parsed_data,
                headers=dict(err.headers.items()) if err.headers else {},
                raw_body=raw_body,
            )
        except URLError as err:
            raise VeoAPITransportError(f"failed to connect to Veo API: {err.reason}") from err


@dataclass(frozen=True)
class VeoAPIClientConfig:
    """Runtime config for Veo API integration client."""

    enabled: bool = False
    allow_cloud: bool = False
    base_url: str = "https://api.veo.co"
    api_token: str | None = None
    timeout_seconds: float = 15.0
    verify_tls: bool = True
    user_agent: str = "veo-soccer-analysis/0.1"

    @classmethod
    def from_config(cls, config: Any) -> "VeoAPIClientConfig":
        """Build client config from dict/object while preserving safe defaults."""
        token_value = _cfg_value(config, "api_token", None) or os.getenv("VEO_API_TOKEN")
        token = str(token_value).strip() if token_value is not None else None
        if token == "":
            token = None

        return cls(
            enabled=_coerce_bool(_cfg_value(config, "enabled", False), default=False),
            allow_cloud=_coerce_bool(_cfg_value(config, "allow_cloud", False), default=False),
            base_url=_sanitize_base_url(
                _cfg_value(config, "base_url", "https://api.veo.co"),
                default="https://api.veo.co",
            ),
            api_token=token,
            timeout_seconds=_coerce_float(
                _cfg_value(config, "timeout_seconds", 15.0),
                default=15.0,
            ),
            verify_tls=_coerce_bool(_cfg_value(config, "verify_tls", True), default=True),
            user_agent=str(
                _cfg_value(config, "user_agent", "veo-soccer-analysis/0.1"),
            ).strip()
            or "veo-soccer-analysis/0.1",
        )


@runtime_checkable
class VeoAPIClientProtocol(Protocol):
    """Interface boundary for Veo API read/write operations."""

    def list_videos(self, *, limit: int | None = None, cursor: str | None = None) -> Any:
        """Return paginated videos list."""
        ...

    def get_video(self, video_id: str) -> Any:
        """Return one video record."""
        ...

    def create_video(self, payload: Mapping[str, Any]) -> Any:
        """Create one video resource."""
        ...

    def update_video(self, video_id: str, payload: Mapping[str, Any]) -> Any:
        """Update one video resource."""
        ...


class VeoAPIClient:
    """Opt-in Veo API client with explicit cloud safety gates."""

    schema_version = VEO_API_CLIENT_SCHEMA_VERSION

    def __init__(
        self,
        config: VeoAPIClientConfig | Mapping[str, Any] | Any | None = None,
        *,
        transport: VeoAPITransport | None = None,
    ) -> None:
        self.config = (
            config
            if isinstance(config, VeoAPIClientConfig)
            else VeoAPIClientConfig.from_config(config)
        )
        self.transport = transport or UrllibVeoAPITransport(verify_tls=self.config.verify_tls)

    def _assert_cloud_allowed(self) -> None:
        """Reject remote access unless explicitly enabled and allowed."""
        if not self.config.enabled:
            raise VeoAPIDisabledError("veo_api integration is disabled")
        if not self.config.allow_cloud:
            raise VeoAPICloudDisabledError("veo_api cloud access is not allowed")
        if not self.config.api_token:
            raise VeoAPIConfigurationError("veo_api token is required for authenticated requests")

    @staticmethod
    def _normalize_path(path: str) -> str:
        normalized = str(path).strip()
        if normalized == "":
            return "/"
        if normalized.startswith("http://") or normalized.startswith("https://"):
            return normalized
        return "/" + normalized.lstrip("/")

    def _build_headers(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        """Build request headers including bearer token auth."""
        self._assert_cloud_allowed()
        token = self.config.api_token
        assert token is not None  # Narrowed by _assert_cloud_allowed

        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
            "User-Agent": self.config.user_agent,
        }
        if extra:
            for key, value in extra.items():
                headers[str(key)] = str(value)
        return headers

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        payload: Any = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        """Execute one API request and return parsed JSON/text payload."""
        normalized_path = self._normalize_path(path)
        if normalized_path.startswith("http://") or normalized_path.startswith("https://"):
            url = normalized_path
        else:
            url = f"{self.config.base_url}{normalized_path}"

        response = self.transport.request(
            method=method.upper(),
            url=url,
            headers=self._build_headers(headers),
            params=params,
            json_body=payload,
            timeout_seconds=self.config.timeout_seconds,
        )

        if response.status_code in {401, 403}:
            message = _extract_error_message(
                response.data,
                fallback="veo api authorization failed",
            )
            raise VeoAPIUnauthorizedError(
                message,
                status_code=response.status_code,
                response_data=response.data,
            )
        if response.status_code >= 400:
            message = _extract_error_message(
                response.data,
                fallback=f"veo api request failed with status {response.status_code}",
            )
            raise VeoAPIRequestError(
                message,
                status_code=response.status_code,
                response_data=response.data,
            )
        return response.data

    def get(self, path: str, *, params: Mapping[str, Any] | None = None) -> Any:
        """Send a GET request to the Veo API."""
        return self.request("GET", path, params=params)

    def post(self, path: str, *, payload: Any = None) -> Any:
        """Send a POST request to the Veo API."""
        return self.request("POST", path, payload=payload)

    def patch(self, path: str, *, payload: Any = None) -> Any:
        """Send a PATCH request to the Veo API."""
        return self.request("PATCH", path, payload=payload)

    def put(self, path: str, *, payload: Any = None) -> Any:
        """Send a PUT request to the Veo API."""
        return self.request("PUT", path, payload=payload)

    def delete(self, path: str) -> Any:
        """Send a DELETE request to the Veo API."""
        return self.request("DELETE", path)

    def get_api_status(self) -> Any:
        """Read API status/health endpoint."""
        return self.get("/status")

    def list_videos(self, *, limit: int | None = None, cursor: str | None = None) -> Any:
        """Read paginated videos."""
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = int(limit)
        if cursor:
            params["cursor"] = str(cursor)
        return self.get("/videos", params=params or None)

    def get_video(self, video_id: str) -> Any:
        """Read one video by identifier."""
        video_id_str = str(video_id).strip()
        if video_id_str == "":
            raise ValueError("video_id must be non-empty")
        return self.get(f"/videos/{video_id_str}")

    def create_video(self, payload: Mapping[str, Any]) -> Any:
        """Create one video resource."""
        return self.post("/videos", payload=dict(payload))

    def update_video(self, video_id: str, payload: Mapping[str, Any]) -> Any:
        """Update one video resource."""
        video_id_str = str(video_id).strip()
        if video_id_str == "":
            raise ValueError("video_id must be non-empty")
        return self.patch(f"/videos/{video_id_str}", payload=dict(payload))
