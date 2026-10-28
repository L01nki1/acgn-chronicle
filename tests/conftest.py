"""全局测试夹具：默认禁止真实网络。

规则（docs/ONBOARDING.md §4）：CI 与本地常规测试都不联网。
需要真网的用例请标 `@pytest.mark.network`，默认 addopts 会把它排除。
"""

from __future__ import annotations

import socket
from collections.abc import Callable

import pytest


class NetworkBlocked(RuntimeError):
    """测试期间任何真实出站连接都会命中这个异常。"""


@pytest.fixture(autouse=True)
def block_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("network"):
        return

    def guard(*_args: object, **_kwargs: object) -> None:
        raise NetworkBlocked(
            "测试禁止真实网络访问：请用 respx/fixture 模拟 HTTP，"
            "或给用例标 @pytest.mark.network 并在本地手动运行。"
        )

    monkeypatch.setattr(socket.socket, "connect", guard, raising=True)
    monkeypatch.setattr(socket.socket, "connect_ex", guard, raising=True)
    monkeypatch.setattr(socket, "create_connection", guard, raising=True)


@pytest.fixture()
def fake_embedder() -> Callable[[list[str]], list[list[float]]]:
    """确定性假向量：不下载模型、不联网，供切片/检索单测使用。"""

    def encode(texts: list[str], dim: int = 8) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            seed = sum(ord(c) * (i + 1) for i, c in enumerate(text)) or 1
            out.append([((seed * (j + 3)) % 97) / 97 for j in range(dim)])
        return out

    return encode
