"""WP3 retrieval: vector top-k with filters over the merged Chroma store.

Owner: E | Spec: `04 §1.3` item 3, `docs/RUNBOOK.md` §3.1, `contracts/G1aReview.md` §2

This module is a thin adapter: it builds the Chroma where-clause, asks the store for top-k
(the store over-fetches, asserts the row count and truncates), maps the stored payload to the
internal `RetrievedChunk`, and returns a deterministic order. It never serializes the full body.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from acgn_rag.core.ports import (
    Citation,
    RetrievalFilters,
    RetrievalQuery,
    RetrievedChunk,
)
from acgn_rag.core.store.chroma_store import ScoredPayload, media_filter

_ONE_DAY = timedelta(days=1)
# Any fractional part, whether attached to the time or to a numeric timezone offset. The raw
# string is checked because `datetime.fromisoformat` silently drops digits beyond microseconds,
# and drops an offset fraction entirely.
_FRACTION_RE = re.compile(r"[.,](\d+)")


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
    return "T" not in value and "t" not in value and " " not in value


def _parse_bound(value: str) -> tuple[datetime, bool]:
    """Parse an ISO bound as UTC; return (instant, is_date_only).

    Sub-second bounds are rejected: `published_ts` is stored as whole epoch seconds, so a
    fractional bound cannot be honoured. Chroma truncates a float operand against an integer
    metadata field, which made `since=...00:00.5Z` match a record stored at `...00:00`. The raw
    fraction is checked because `datetime.fromisoformat` drops digits beyond microseconds.
    """
    for match in _FRACTION_RE.finditer(value):
        if any(digit != "0" for digit in match.group(1)):
            raise ValueError(
                "fractional-second bounds are not supported: published_ts is stored in whole "
                f"seconds, so {value!r} cannot be compared exactly"
            )
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    parsed = parsed.astimezone(UTC)
    if parsed.microsecond:
        raise ValueError(
            "fractional-second bounds are not supported: published_ts is stored in whole "
            f"seconds, so {value!r} cannot be compared exactly"
        )
    return parsed, _is_date_only(value)


def build_where(filters: RetrievalFilters | None) -> dict[str, Any] | None:
    """Translate `RetrievalFilters` into a Chroma where-clause.

    Empty lists mean "no constraint in that family"; families are ANDed, values inside a
    family are ORed. Date-only bounds are whole UTC days (upper exclusive at next midnight);
    ISO datetimes are half-open at their stated whole-second instants. Fractional-second
    bounds are rejected because the store keeps whole seconds. Equal datetimes are an empty
    interval, while equal date-only values select one day.
    """
    if filters is None:
        return None

    since = _parse_bound(filters.since) if filters.since else None
    until = _parse_bound(filters.until) if filters.until else None

    if since is not None and until is not None:
        since_dt, since_date_only = since
        until_dt, until_date_only = until
        if since_date_only and until_date_only:
            if since_dt.date() > until_dt.date():
                raise ValueError("since must be earlier than until")
        else:
            upper = until_dt + _ONE_DAY if until_date_only else until_dt
            if since_dt > upper:
                raise ValueError("since must be earlier than until")

    clauses: list[dict[str, Any]] = []
    if filters.source_types:
        clauses.append({"source_type": {"$in": list(filters.source_types)}})
    if since is not None:
        clauses.append({"published_ts": {"$gte": since[0].timestamp()}})
    if until is not None:
        upper = until[0] + _ONE_DAY if until[1] else until[0]
        clauses.append({"published_ts": {"$lt": upper.timestamp()}})
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
    if not query.text.strip():
        # An empty question has no evidence; embedding it would return arbitrary chunks.
        return []
    vectors = embedder.encode_queries([query.text])
    if not len(vectors):
        raise ValueError("embedder returned no vector for the query")
    hits = store.query(vectors[0], query.top_k, where)
    chunks = [to_retrieved(hit, query.snapshot_id) for hit in hits]
    chunks.sort(key=lambda chunk: (-chunk.score, chunk.chunk_id))
    return chunks


__all__ = ["EmbedderPort", "StorePort", "build_where", "search", "to_retrieved"]
