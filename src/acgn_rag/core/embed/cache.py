"""Identity-keyed embedding cache: one .npy per key, atomic writes, no expiry.

Owner: F | Spec: RUNBOOK §3.4 (cache key + encoder identity)

Keys are `encoder.cache_key()` digests - they already bind model, family mode, prefix/prompt,
header version, header, body and the full encoder identity, so this class never inspects them.
Two levels of two-hex sharding keep any single directory small enough to list on a 4x40k-chunk
corpus. No network, no expiry: a stale vector is impossible because the key changes with every
input that could change the vector. numpy ships with the [embed] extra.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


class EmbedCache:
    """On-disk vector cache: `root/<key[:2]>/<key[2:4]>/<key>.npy` plus an in-memory dict.

    `get` serves from memory first, then disk; `put` writes a temp file in the target directory
    and `os.replace`s it into place, so a concurrent reader sees either the old vector or the
    complete new one, never a partial file.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._mem: dict[str, np.ndarray] = {}
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> np.ndarray | None:
        """Return the cached vector, or None; every lookup counts toward stats()."""
        self._check_key(key)
        if key in self._mem:
            self._hits += 1
            return self._mem[key]
        path = self._path(key)
        if path.is_file():
            vector = np.load(path)
            self._mem[key] = vector
            self._hits += 1
            return vector
        self._misses += 1
        return None

    def put(self, key: str, vector: np.ndarray) -> None:
        """Store in memory and on disk; the caller's array is copied, not aliased."""
        self._check_key(key)
        self._mem[key] = np.array(vector, copy=True)
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".tmp-{os.getpid()}-{path.name}")
        np.save(tmp, self._mem[key])
        os.replace(tmp, path)

    def stats(self) -> dict[str, int]:
        """Hits/misses so far; enough to report a cache hit rate for a build run."""
        return {"hits": self._hits, "misses": self._misses}

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / key[2:4] / f"{key}.npy"

    @staticmethod
    def _check_key(key: str) -> None:
        if len(key) < 4:
            raise ValueError(
                f"cache keys need at least 4 characters for the two shard levels, got {key!r}"
            )


__all__ = ["EmbedCache"]
