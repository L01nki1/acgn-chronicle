"""WP3 retrieval: vector top-k with filters over the merged Chroma store.

Owner: E | Spec: `04 §1.3` item 3, `docs/RUNBOOK.md` §3.1, `contracts/G1aReview.md` §2

This module is a thin adapter: it builds the Chroma where-clause, asks the store for top-k
(the store over-fetches, asserts the row count and truncates), maps the stored payload to the
internal `RetrievedChunk`, and returns a deterministic order. It never serializes the full body.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Protocol

from acgn_rag.core.ports import (
    Citation,
    RetrievalFilters,
    RetrievalQuery,
    RetrievedChunk,
)
from acgn_rag.core.store.chroma_store import ScoredPayload, media_filter
from acgn_rag.core.store.payload import iso_to_epoch

_DAY_SECONDS = 86_400


class EmbedderPort(Protocol):
    """The query-embedding surface E consumes from `core/embed`."""

    def encode_queries(self, texts: Sequence[str]) -> Any: ...


class StorePort(Protocol):
    """The bound-store surface E consumes; the store owns over-fetch and truncation."""

    def query(
        self,
        embedding: Sequence[float],
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[ScoredPayload]: ...


def _is_date_only(value: str) -> bool:
    return "T" not in value and " " not in value


def _until_epoch(value: str) -> int:
    """Date-only `until` is exclusive at the next UTC midnight (G1aReview §2)."""
    epoch = iso_to_epoch(value)
    return epoch + _DAY_SECONDS if _is_date_only(value) else epoch


def build_where(filters: RetrievalFilters | None) -> dict[str, Any] | None:
    """Translate `RetrievalFilters` into a Chroma where-clause.

    Empty lists mean "no constraint in that family"; families are ANDed, values inside a
    family are ORed; `until` is exclusive (next midnight for a date-only value).
    """
    if filters is None:
        return None
    if (
        filters.since
        and filters.until
        and iso_to_epoch(filters.since) >= _until_epoch(filters.until)
    ):
        raise ValueError("since must be earlier than until")

    clauses: list[dict[str, Any]] = []
    if filters.source_types:
        clauses.append({"source_type": {"$in": list(filters.source_types)}})
    if filters.since:
        clauses.append({"published_ts": {"$gte": iso_to_epoch(filters.since)}})
    if filters.until:
        clauses.append({"published_ts": {"$lt": _until_epoch(filters.until)}})
    if filters.media_ids:
        media = media_filter(list(filters.media_ids))
        if media is not None:
            clauses.append(media)

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def to_retrieved(hit: ScoredPayload, snapshot_id: str) -> RetrievedChunk:
    """Map one stored hit to the internal chunk; the store score is already cosine."""
    payload = hit.payload
    if payload.snapshot_id != snapshot_id:
        raise ValueError(
            f"hit belongs to snapshot {payload.snapshot_id!r}, expected {snapshot_id!r}"
        )
    citation = Citation(
        source_type=payload.source_type,
        publisher=payload.publisher,
        title=payload.title,
        url=payload.url,
        published_at=datetime.fromtimestamp(payload.published_ts, UTC).date().isoformat(),
        chunk_id=payload.chunk_id,
    )
    return RetrievedChunk(
        chunk_id=payload.chunk_id,
        doc_id=payload.doc_id,
        snapshot_id=payload.snapshot_id,
        text=payload.body,
        score=hit.score,
        citation=citation,
    )


def search(
    query: RetrievalQuery,
    *,
    store: StorePort,
    embedder: EmbedderPort,
) -> list[RetrievedChunk]:
    """Embed the query, retrieve top-k through the bound store, return deterministic order."""
    if query.top_k < 1:
        raise ValueError("top_k must be >= 1")
    where = build_where(query.filters)
    vectors = embedder.encode_queries([query.text])
    if not len(vectors):
        raise ValueError("embedder returned no vector for the query")
    hits = store.query(vectors[0], query.top_k, where)
    chunks = [to_retrieved(hit, query.snapshot_id) for hit in hits]
    chunks.sort(key=lambda chunk: (-chunk.score, chunk.chunk_id))
    return chunks


__all__ = ["EmbedderPort", "StorePort", "build_where", "search", "to_retrieved"]
