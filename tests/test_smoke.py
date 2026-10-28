"""Smoke tests: the package imports, settings load, defaults match the documented values."""

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
    assert settings.embed.device == "cpu"  # identical device is what keeps metrics comparable


def test_user_agent_is_self_describing() -> None:
    # 01 §5.2: a self-describing UA is a compliance stance, not a disguise
    assert DEFAULT_UA.startswith("AcgnRagBot/")
    assert "CS5481" in DEFAULT_UA


def test_paths_stay_under_data_dir() -> None:
    paths = load_settings().paths
    for p in (paths.raw, paths.facts, paths.index, paths.snapshots):
        assert str(p).startswith(str(paths.root))
