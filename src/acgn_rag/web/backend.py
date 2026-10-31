"""The seam between the HTTP host and a KnowledgePort implementation.

Owner: C | Spec: contracts/KnowledgePort.md §4

The endpoint layer may call exactly these four operations. G1a supplies a mock implementation;
W2 replaces it with the real host without changing the page or the routes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from acgn_rag.core.ports import (
    ChunkExcerpt,
    CorpusSnapshot,
    GroundedAnswer,
    RetrievalQuery,
    SnapshotInventory,
)


class KnowledgePortBackend(Protocol):
    """What a backend must expose to the web host."""

    def inventory(self) -> SnapshotInventory:
        """Return the current snapshot and the ids still queryable."""
        ...

    def snapshot(self, snapshot_id: str | None = None) -> CorpusSnapshot:
        """Return the current snapshot when `snapshot_id` is omitted."""
        ...

    def retrieve(self, query: RetrievalQuery) -> Sequence[ChunkExcerpt]:
        """Return excerpts only; never full article bodies."""
        ...

    def ask(self, query: RetrievalQuery) -> GroundedAnswer:
        """Return a grounded answer, or an explicit refusal."""
        ...


__all__ = ["KnowledgePortBackend"]
