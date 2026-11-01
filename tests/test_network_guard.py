"""The offline guard must stop external connects without breaking loopback plumbing.

On Windows, `socket.socketpair` is emulated with a loopback TCP connection and asyncio builds
every event loop's self-pipe with it; a blanket connect guard makes TestClient unstartable there.
See `tests/conftest.py` for the guard itself.
"""

from __future__ import annotations

import socket

import pytest
from fastapi.testclient import TestClient

from acgn_rag.web.api import create_app


def test_external_connect_is_blocked() -> None:
    with socket.socket() as sock, pytest.raises(RuntimeError, match="Network access is blocked"):
        sock.connect(("example.com", 80))


def test_loopback_connect_is_allowed() -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        address = listener.getsockname()
        with socket.socket() as client:
            client.settimeout(5)
            client.connect(address)
            accepted, _ = listener.accept()
            accepted.close()


def test_event_loop_self_pipe_works_with_the_windows_socketpair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reproduce Windows' TCP-emulated socketpair on every platform.

    `socket._fallback_socketpair` is the exact helper CPython uses where `_socket` has no
    native socketpair. Patching it in makes the POSIX build emulate Windows here, so a guard
    that would break the Windows event loop fails locally too.
    """
    monkeypatch.setattr(socket, "socketpair", socket._fallback_socketpair)

    with TestClient(create_app()) as client:
        response = client.get("/snapshots")
    assert response.status_code == 200
