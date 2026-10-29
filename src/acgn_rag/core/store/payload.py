"""Chroma metadata payload: the flat, typed shape one chunk takes inside a collection.

Owner: F | Spec: 04 §7 D4, docs/RUNBOOK.md §3.1 (chromadb >=1.0,<2)

Chroma 1.x metadata accepts scalars or flat same-type arrays - nesting is still forbidden -
so a chunk is stored as `documents=<body>` plus a flat metadata dict built by `to_payload`:
`published_ts` is an int epoch (never an ISO string) and `media_ids` stays a native `int[]`
filtered with `$contains` (see `chroma_store.media_filter`).

Adapter seam: `payload_from_fields` is the ONLY place that changes when A's NormalizedItem
schema lands (interface issue still open). Everything downstream sees `ChunkPayload`, whose
field list is settled independently of A's crawl-side representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from acgn_rag.core.ports import SourceType


@dataclass(frozen=True)
class ChunkPayload:
    """Flat per-chunk record stored as exactly one Chroma row (document + metadata)."""

    chunk_id: str
    doc_id: str
    snapshot_id: str
    title: str
    url: str
    publisher: str
    source_type: SourceType
    published_ts: int  # epoch seconds, UTC; never an ISO string (RUNBOOK §3.1 pitfall 2)
    lang: str
    body: str
    media_ids: tuple[int, ...]
    ingest_category: str | None = None  # ingest-side tag (RUNBOOK §3.6); not a SourceType


def iso_to_epoch(value: str) -> int:
    """ISO-8601 string -> int epoch seconds.

    A timestamp without an offset is read as UTC - an explicit convention, not an accident:
    the sources are trilingual and their feeds mix offsets, so a silent local-time read
    (Python's default for naive datetimes) would shift `since`/`until` filters by hours.
    """
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp())


def to_payload(payload: ChunkPayload) -> dict[str, Any]:
    """Build the Chroma metadata dict for one chunk: flat scalars + `media_ids` as int[].

    `body` is deliberately absent: it is stored as the Chroma document (`documents=`), and
    copying it into the metadata would double every row for zero query value. `to_payload`
    and `payload_from_metadata` are inverses, so a query round-trips the full payload.
    `ingest_category` is omitted when None - Chroma has no null metadata value, and the
    absent key maps back to None on reconstruction.
    """
    meta: dict[str, Any] = {
        "chunk_id": payload.chunk_id,
        "doc_id": payload.doc_id,
        "snapshot_id": payload.snapshot_id,
        "title": payload.title,
        "url": payload.url,
        "publisher": payload.publisher,
        "source_type": payload.source_type,
        "published_ts": payload.published_ts,
        "lang": payload.lang,
        "media_ids": list(payload.media_ids),
    }
    if payload.ingest_category is not None:
        meta["ingest_category"] = payload.ingest_category
    return meta


def payload_from_fields(**kwargs: Any) -> ChunkPayload:
    """Build a `ChunkPayload` from NormalizedItem-shaped kwargs - THE adapter seam.

    This is the only function to edit when A's NormalizedItem schema lands (issue open at
    PR3 time): field renames and type coercions happen here, nowhere else. Accepts either
    `published_ts` (int, passed through) or `published_at` (ISO string, converted via
    `iso_to_epoch`); `media_ids` accepts any iterable of ints and is frozen into a tuple.
    Unknown keys raise the dataclass TypeError, so schema drift fails loudly here instead
    of silently producing half-populated rows at query time.
    """
    published_at = kwargs.pop("published_at", None)
    if "published_ts" not in kwargs:
        if published_at is None:
            raise TypeError("ChunkPayload requires published_ts (int) or published_at (ISO string)")
        kwargs["published_ts"] = iso_to_epoch(published_at)
    kwargs["media_ids"] = tuple(kwargs.get("media_ids", ()))
    return ChunkPayload(**kwargs)


def payload_from_metadata(metadata: dict[str, Any], document: str | None = None) -> ChunkPayload:
    """Rebuild a `ChunkPayload` from a query result row (metadata dict + stored document)."""
    return ChunkPayload(
        chunk_id=str(metadata["chunk_id"]),
        doc_id=str(metadata["doc_id"]),
        snapshot_id=str(metadata["snapshot_id"]),
        title=str(metadata["title"]),
        url=str(metadata["url"]),
        publisher=str(metadata["publisher"]),
        source_type=metadata["source_type"],
        published_ts=int(metadata["published_ts"]),
        lang=str(metadata["lang"]),
        body=document if document is not None else "",
        media_ids=tuple(int(m) for m in metadata.get("media_ids", ())),
        ingest_category=metadata.get("ingest_category"),
    )


__all__ = [
    "ChunkPayload",
    "iso_to_epoch",
    "payload_from_fields",
    "payload_from_metadata",
    "to_payload",
]
