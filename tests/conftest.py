"""Global fixtures: no real network access by default.

Rule (docs/ONBOARDING.md §4): neither CI nor a normal local run touches the network.
Tests that genuinely need it are marked `@pytest.mark.network`, which the default addopts exclude.
"""

from __future__ import annotations

import socket
from collections.abc import Callable
from typing import Any, NoReturn

import pytest

# Windows has no native `socket.socketpair`, so CPython emulates it with a loopback TCP
# connection. asyncio builds every event loop's self-pipe with `socketpair`, which means a
# blanket connect guard makes an event loop unstartable there: Starlette's TestClient dies
# before the first request. A loopback connection never leaves the machine, so it is not the
# external traffic this guard exists to stop.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class NetworkBlocked(RuntimeError):
    """Raised when a test attempts a real outbound connection."""


def is_loopback(address: object) -> bool:
    """True when a connect() target stays on the local machine."""
    host = address[0] if isinstance(address, tuple) and address else None
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    return isinstance(host, str) and host in _LOOPBACK_HOSTS


@pytest.fixture(autouse=True)
def block_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("network"):
        return

    def guard() -> NoReturn:
        raise NetworkBlocked(
            "Network access is blocked in tests: mock HTTP with respx/fixtures, "
            "or mark the test @pytest.mark.network and run it locally."
        )

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_create_connection = socket.create_connection

    def guarded_connect(self: socket.socket, address: Any, *args: Any, **kwargs: Any) -> None:
        if is_loopback(address):
            real_connect(self, address, *args, **kwargs)
            return
        guard()

    def guarded_connect_ex(self: socket.socket, address: Any, *args: Any, **kwargs: Any) -> int:
        if is_loopback(address):
            return real_connect_ex(self, address, *args, **kwargs)
        guard()

    def guarded_create_connection(address: Any, *args: Any, **kwargs: Any) -> socket.socket:
        if is_loopback(address):
            return real_create_connection(address, *args, **kwargs)
        guard()

    monkeypatch.setattr(socket.socket, "connect", guarded_connect, raising=True)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex, raising=True)
    monkeypatch.setattr(socket, "create_connection", guarded_create_connection, raising=True)


@pytest.fixture()
def fake_embedder() -> Callable[[list[str]], list[list[float]]]:
    """Deterministic fake vectors: no model download, no network."""

    def encode(texts: list[str], dim: int = 8) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            seed = sum(ord(c) * (i + 1) for i, c in enumerate(text)) or 1
            out.append([((seed * (j + 3)) % 97) / 97 for j in range(dim)])
        return out

    return encode
