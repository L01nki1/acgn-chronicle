"""Contract errors raised by the web host and its backend seam.

Owner: C | Spec: contracts/KnowledgePort.md §5

The status codes are the contract, not a UI preference: unknown snapshot -> 404, retired ->
410, invalid filter -> 422, host not serving -> 503 with `Retry-After`. The body keeps FastAPI's
`{"detail": ...}` shape so a client has one place to read the reason.
"""

from __future__ import annotations

from collections.abc import Mapping


class PortError(Exception):
    """An error that maps to a contract status code and a JSON `detail` body."""

    status_code = 500

    def __init__(self, detail: str, *, headers: Mapping[str, str] | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        self.headers = dict(headers or {})


class UnknownSnapshot(PortError):
    """The id was never known to this host."""

    status_code = 404


class RetiredSnapshot(PortError):
    """The id was known, but it is outside retention now."""

    status_code = 410


class InvalidFilter(PortError):
    """The filter is malformed or reversed; retrying it cannot help."""

    status_code = 422


class NotServing(PortError):
    """The host has no completed snapshot to serve yet."""

    status_code = 503

    def __init__(self, detail: str, *, retry_after_seconds: int = 30) -> None:
        super().__init__(detail, headers={"Retry-After": str(retry_after_seconds)})
        self.retry_after_seconds = retry_after_seconds


__all__ = [
    "InvalidFilter",
    "NotServing",
    "PortError",
    "RetiredSnapshot",
    "UnknownSnapshot",
]
