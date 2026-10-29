"""EmbedCache: round trip, hit/miss stats, atomic writes, two-level sharded layout.

Owner: F | Spec: RUNBOOK §3.4 (identity-keyed cache)
"""

from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")  # [embed] extra; CI installs [dev] only and skips this module

from acgn_rag.core.embed import EmbedCache  # noqa: E402

KEY = "ab" + "c" * 62  # a sha256-shaped key: first two levels are "ab" and "cc"


def test_round_trip_and_disk_persistence(tmp_path: Path) -> None:
    cache = EmbedCache(tmp_path)
    vector = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
    assert cache.get(KEY) is None
    cache.put(KEY, vector)
    loaded = cache.get(KEY)
    np.testing.assert_array_equal(loaded, vector)
    # a fresh instance over the same root serves from disk, not from the dead in-memory dict
    reopened = EmbedCache(tmp_path)
    from_disk = reopened.get(KEY)
    assert from_disk is not None
    np.testing.assert_array_equal(from_disk, vector)
    assert from_disk.dtype == np.float32


def test_stats_count_hits_and_misses(tmp_path: Path) -> None:
    cache = EmbedCache(tmp_path)
    assert cache.stats() == {"hits": 0, "misses": 0}
    assert cache.get(KEY) is None  # miss
    cache.put(KEY, np.ones(4, dtype=np.float32))
    assert cache.get(KEY) is not None  # hit, from memory
    assert cache.get(KEY) is not None  # hit, still memory
    assert cache.stats() == {"hits": 2, "misses": 1}


def test_layout_is_two_level_sharded(tmp_path: Path) -> None:
    cache = EmbedCache(tmp_path)
    cache.put(KEY, np.zeros(3, dtype=np.float32))
    assert (tmp_path / KEY[:2] / KEY[2:4] / f"{KEY}.npy").is_file()


def test_write_leaves_exactly_one_complete_file(tmp_path: Path) -> None:
    cache = EmbedCache(tmp_path)
    cache.put(KEY, np.ones(5, dtype=np.float32))
    files = sorted(p for p in tmp_path.rglob("*") if p.is_file())
    # no temp/partial file survives: the tree holds exactly the one final .npy
    assert files == [tmp_path / KEY[:2] / KEY[2:4] / f"{KEY}.npy"]
    np.testing.assert_array_equal(np.load(files[0]), np.ones(5, dtype=np.float32))


def test_put_copies_the_callers_array(tmp_path: Path) -> None:
    cache = EmbedCache(tmp_path)
    vector = np.ones(4, dtype=np.float32)
    cache.put(KEY, vector)
    vector[0] = 99.0  # mutating the caller's array must not poison the cache
    np.testing.assert_array_equal(cache.get(KEY), np.ones(4, dtype=np.float32))


def test_short_key_is_rejected(tmp_path: Path) -> None:
    cache = EmbedCache(tmp_path)
    with pytest.raises(ValueError, match="at least 4 characters"):
        cache.get("abc")
    with pytest.raises(ValueError, match="at least 4 characters"):
        cache.put("abc", np.ones(2, dtype=np.float32))
