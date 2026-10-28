"""冒烟用例：包可导入、配置可加载、默认值符合文档口径。"""

from __future__ import annotations

import acgn_rag
from acgn_rag.config import DEFAULT_UA, load_settings


def test_version() -> None:
    assert acgn_rag.__version__


def test_settings_defaults() -> None:
    settings = load_settings()
    assert settings.store.backend == "chroma"
    assert settings.store.over_fetch_factor >= 2
    assert settings.embed.query_prefix == "query: "
    assert settings.embed.passage_prefix == "passage: "
    assert settings.embed.device == "cpu"  # 跨机器指标可比的前提


def test_user_agent_is_self_describing() -> None:
    # 01 §5.2：自描述 UA 是合规姿态，不是伪装
    assert DEFAULT_UA.startswith("AcgnRagBot/")
    assert "CS5481" in DEFAULT_UA


def test_paths_stay_under_data_dir() -> None:
    paths = load_settings().paths
    for p in (paths.raw, paths.facts, paths.index, paths.snapshots):
        assert str(p).startswith(str(paths.root))
