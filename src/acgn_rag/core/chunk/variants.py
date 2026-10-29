"""A'(256/0) and A0(512/0) are parameter variants of A, not separate pipelines (issue #8).

Owner: F | Spec: 03 §2, issue #8
Status: active - CHUNK_VARIANTS is the single table every chunk consumer reads; adding a
config here and only here is what keeps the A/A'/A0 comparison on one code path.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkVariant:
    """One row of the chunking experiment: the variant only supplies (max_tokens, overlap)."""

    name: str
    max_tokens: int
    overlap: int


CHUNK_VARIANTS: dict[str, ChunkVariant] = {
    "A": ChunkVariant(name="A", max_tokens=512, overlap=64),
    "A'": ChunkVariant(name="A'", max_tokens=256, overlap=0),
    "A0": ChunkVariant(name="A0", max_tokens=512, overlap=0),
}

__all__ = [
    "CHUNK_VARIANTS",
    "ChunkVariant",
]
