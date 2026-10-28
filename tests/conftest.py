"""Global fixtures: no real network access by default.

Rule (docs/ONBOARDING.md §4): neither CI nor a normal local run touches the network.
Tests that genuinely need it are marked `@pytest.mark.network`, which the default addopts exclude.
"""

from __future__ import annotations

import socket
from collections.abc import Callable

import pytest


class NetworkBlocked(RuntimeError):
    """Raised when a test attempts a real outbound connection."""


@pytest.fixture(autouse=True)
def block_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("network"):
        return

    def guard(*_args: object, **_kwargs: object) -> None:
        raise NetworkBlocked(
            "Network access is blocked in tests: mock HTTP with respx/fixtures, "
            "or mark the test @pytest.mark.network and run it locally."
        )

    monkeypatch.setattr(socket.socket, "connect", guard, raising=True)
    monkeypatch.setattr(socket.socket, "connect_ex", guard, raising=True)
    monkeypatch.setattr(socket, "create_connection", guard, raising=True)


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
