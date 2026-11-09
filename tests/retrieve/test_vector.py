"""WP3 retrieval tests: filter translation, payload mapping and deterministic order.

No network, no Chroma: a fake store records the where-clause and returns configured hits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest

from acgn_rag.core.ports import RetrievalFilters, RetrievalQuery
from acgn_rag.core.retrieve.vector import build_where, search, to_retrieved
from acgn_rag.core.store.chroma_store import ScoredPayload
from acgn_rag.core.store.payload import ChunkPayload, iso_to_epoch

DAY = 86_400
SNAPSHOT = "20261028abcdef"


def _ts(value: str) -> float:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def _hit(
    chunk_id: str = "c1",
    *,
    score: float = 0.9,
    snapshot_id: str = SNAPSHOT,
    media_ids: tuple[int, ...] = (7,),
) -> ScoredPayload:
    return ScoredPayload(
        payload=ChunkPayload(
            chunk_id=chunk_id,
            doc_id="d1",
            snapshot_id=snapshot_id,
            title="Title",
            url="https://example.com/news/1",
            publisher="Publisher",
            source_type="official",
            published_ts=iso_to_epoch("2026-10-28"),
            lang="en",
            body="The body text.",
            media_ids=media_ids,
        ),
        score=score,
    )


@dataclass
class FakeStore:
    hits: list[ScoredPayload]
    calls: list[tuple[int, dict[str, Any] | None]] = field(default_factory=list)

    def query(
        self,
        embedding: Any,
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[ScoredPayload]:
        self.calls.append((top_k, where))
        return self.hits[:top_k]


class FakeEmbedder:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def encode_queries(self, texts: Any) -> list[list[float]]:
        self.texts.extend(texts)
        return [[0.1, 0.2] for _ in texts]


class EmptyEmbedder:
    def encode_queries(self, texts: Any) -> list[list[float]]:
        return []


def test_build_where_none() -> None:
    assert build_where(None) is None


def test_build_where_empty_lists_mean_no_filter() -> None:
    assert build_where(RetrievalFilters(source_types=[], media_ids=[])) is None


def test_build_where_source_types() -> None:
    assert build_where(RetrievalFilters(source_types=["official"])) == {
        "source_type": {"$in": ["official"]}
    }


def test_build_where_media_single_and_multiple() -> None:
    assert build_where(RetrievalFilters(media_ids=[7])) == {"media_ids": {"$contains": 7}}
    assert build_where(RetrievalFilters(media_ids=[7, 8])) == {
        "$or": [{"media_ids": {"$contains": 7}}, {"media_ids": {"$contains": 8}}]
    }


def test_build_where_date_bounds_are_half_open_utc_days() -> None:
    where = build_where(RetrievalFilters(since="2026-10-28", until="2026-10-28"))
    start = iso_to_epoch("2026-10-28")
    assert where == {
        "$and": [
            {"published_ts": {"$gte": start}},
            {"published_ts": {"$lt": start + DAY}},
        ]
    }


def test_build_where_combines_families_with_and() -> None:
    where = build_where(
        RetrievalFilters(source_types=["official", "media"], since="2026-10-01", media_ids=[7])
    )
    assert where == {
        "$and": [
            {"source_type": {"$in": ["official", "media"]}},
            {"published_ts": {"$gte": iso_to_epoch("2026-10-01")}},
            {"media_ids": {"$contains": 7}},
        ]
    }


def test_build_where_since_only_and_until_only() -> None:
    assert build_where(RetrievalFilters(since="2026-10-01")) == {
        "published_ts": {"$gte": iso_to_epoch("2026-10-01")}
    }
    assert build_where(RetrievalFilters(until="2026-10-28")) == {
        "published_ts": {"$lt": iso_to_epoch("2026-10-28") + DAY}
    }


def test_build_where_full_iso_until_is_exact() -> None:
    assert build_where(RetrievalFilters(until="2026-10-28T12:00:00+00:00")) == {
        "published_ts": {"$lt": iso_to_epoch("2026-10-28T12:00:00+00:00")}
    }


def test_build_where_equal_datetime_instants_is_empty_interval() -> None:
    value = "2026-10-28T12:00:00+00:00"
    assert build_where(RetrievalFilters(since=value, until=value)) == {
        "$and": [
            {"published_ts": {"$gte": _ts(value)}},
            {"published_ts": {"$lt": _ts(value)}},
        ]
    }


def test_build_where_reversed_datetimes_raise() -> None:
    with pytest.raises(ValueError, match="since must be earlier"):
        build_where(
            RetrievalFilters(
                since="2026-10-28T12:00:00+00:00",
                until="2026-10-28T11:00:00+00:00",
            )
        )


def test_build_where_offset_until_is_exact_utc() -> None:
    assert build_where(RetrievalFilters(until="2026-10-29T08:00:00+08:00")) == {
        "published_ts": {"$lt": _ts("2026-10-29T00:00:00+00:00")}
    }


def test_build_where_naive_datetime_is_utc() -> None:
    assert build_where(RetrievalFilters(since="2026-10-28T12:00:00")) == {
        "published_ts": {"$gte": _ts("2026-10-28T12:00:00+00:00")}
    }


def test_build_where_rejects_fractional_bounds() -> None:
    fractional = (
        RetrievalFilters(since="2026-10-28T12:00:00.3Z"),
        RetrievalFilters(until="2026-10-28T12:00:00.500000+00:00"),
        RetrievalFilters(since="2026-10-28T21:00:00.5+09:00"),
    )
    for filters in fractional:
        with pytest.raises(ValueError, match="fractional-second"):
            build_where(filters)


def test_build_where_whole_second_datetime_stays_exact() -> None:
    assert build_where(RetrievalFilters(since="2026-10-28T12:00:00.000Z")) == {
        "published_ts": {"$gte": _ts("2026-10-28T12:00:00+00:00")}
    }


def test_build_where_empty_source_types_keeps_other_families() -> None:
    assert build_where(RetrievalFilters(source_types=[], since="2026-10-01")) == {
        "published_ts": {"$gte": iso_to_epoch("2026-10-01")}
    }


def test_build_where_rejects_since_after_until() -> None:
    with pytest.raises(ValueError, match="since must be earlier"):
        build_where(RetrievalFilters(since="2026-10-29", until="2026-10-28"))


def test_build_where_rejects_invalid_date() -> None:
    with pytest.raises(ValueError):
        build_where(RetrievalFilters(since="not-a-date"))


def test_to_retrieved_maps_citation_and_body() -> None:
    chunk = to_retrieved(_hit(), SNAPSHOT)
    assert chunk.chunk_id == "c1"
    assert chunk.text == "The body text."
    assert chunk.citation.url == "https://example.com/news/1"
    assert chunk.citation.chunk_id == "c1"
    assert chunk.citation.source_type == "official"
    assert chunk.citation.publisher == "Publisher"
    assert chunk.citation.title == "Title"
    assert chunk.citation.published_at == "2026-10-28"


def test_to_retrieved_rejects_snapshot_mismatch() -> None:
    with pytest.raises(ValueError, match="snapshot"):
        to_retrieved(_hit(snapshot_id="other"), SNAPSHOT)


def test_search_embeds_query_maps_and_orders_deterministically() -> None:
    store = FakeStore([_hit("c2", score=0.5), _hit("c1", score=0.9), _hit("c3", score=0.5)])
    embedder = FakeEmbedder()

    chunks = search(
        RetrievalQuery(text="when does it air?", snapshot_id=SNAPSHOT),
        store=store,
        embedder=embedder,
    )

    assert [chunk.chunk_id for chunk in chunks] == ["c1", "c2", "c3"]
    assert store.calls == [(5, None)]
    assert embedder.texts == ["when does it air?"]


def test_search_passes_where_and_top_k() -> None:
    store = FakeStore([_hit()])
    filters = RetrievalFilters(source_types=["official"], since="2026-10-01")

    chunks = search(
        RetrievalQuery(text="q", top_k=3, snapshot_id=SNAPSHOT, filters=filters),
        store=store,
        embedder=FakeEmbedder(),
    )

    assert len(chunks) == 1
    assert store.calls == [
        (
            3,
            {
                "$and": [
                    {"source_type": {"$in": ["official"]}},
                    {"published_ts": {"$gte": iso_to_epoch("2026-10-01")}},
                ]
            },
        )
    ]


def test_search_returns_empty_when_store_has_no_hits() -> None:
    chunks = search(
        RetrievalQuery(text="q", snapshot_id=SNAPSHOT),
        store=FakeStore([]),
        embedder=FakeEmbedder(),
    )
    assert chunks == []


def test_search_returns_available_hits_when_fewer_than_top_k() -> None:
    chunks = search(
        RetrievalQuery(text="q", top_k=10, snapshot_id=SNAPSHOT),
        store=FakeStore([_hit("c1")]),
        embedder=FakeEmbedder(),
    )
    assert [chunk.chunk_id for chunk in chunks] == ["c1"]


def test_search_rejects_empty_embedding() -> None:
    with pytest.raises(ValueError, match="no vector"):
        search(
            RetrievalQuery(text="q", snapshot_id=SNAPSHOT),
            store=FakeStore([_hit()]),
            embedder=EmptyEmbedder(),
        )


def test_search_rejects_top_k_zero() -> None:
    with pytest.raises(ValueError, match="top_k"):
        search(
            RetrievalQuery(text="q", top_k=0, snapshot_id=SNAPSHOT),
            store=FakeStore([]),
            embedder=FakeEmbedder(),
        )


def test_search_rejects_blank_query_before_embedding() -> None:
    store = FakeStore([_hit("c1")])
    embedder = FakeEmbedder()

    chunks = search(
        RetrievalQuery(text="   ", snapshot_id=SNAPSHOT),
        store=store,
        embedder=embedder,
    )

    assert chunks == []
    assert store.calls == []
    assert embedder.texts == []
