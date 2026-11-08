"""E -> F filter integration: `build_where` clauses exercised on a real Chroma collection.

Owner: E | Gate: G1b filter/over-fetch pass condition; G2 hard check 9.

`tests/retrieve/test_vector.py` pins the where-clause shape with a fake store. This file
closes the other half: the E-produced clause must be accepted by Chroma 1.x and return the
same chunks the translation promises. It deliberately does not re-test F's store-level
`media_filter` / no-match / under-return cases (`tests/test_store_chroma.py` owns those).

Chroma rejects empty metadata lists. The main corpus uses non-empty media IDs; the
unresolved-item tests separately cover empty IDs and clearing an earlier link.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

chromadb = pytest.importorskip("chromadb")

from acgn_rag.config import StoreConfig  # noqa: E402
from acgn_rag.core.ports import RetrievalFilters, RetrievalQuery  # noqa: E402
from acgn_rag.core.retrieve.vector import search  # noqa: E402
from acgn_rag.core.store.chroma_store import ChromaStore  # noqa: E402
from acgn_rag.core.store.payload import ChunkPayload, payload_from_fields  # noqa: E402

SNAP = "20261028abcdef1234"
ALL_IDS = {"c1", "c2", "c3", "c4", "c5", "c6"}


def _payload(
    chunk_id: str,
    *,
    source_type: str,
    published_at: str,
    media_ids: tuple[int, ...],
) -> ChunkPayload:
    return payload_from_fields(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        snapshot_id=SNAP,
        title=f"Title {chunk_id}",
        url=f"https://example.com/{chunk_id}",
        publisher="Publisher",
        source_type=source_type,
        published_at=published_at,
        lang="en",
        body=f"Body {chunk_id}",
        media_ids=media_ids,
    )


def _corpus() -> list[ChunkPayload]:
    return [
        _payload("c1", source_type="official", published_at="2026-10-28T00:00:00Z", media_ids=(1,)),
        _payload("c2", source_type="media", published_at="2026-10-28T12:00:00Z", media_ids=(2,)),
        _payload(
            "c3", source_type="metadata", published_at="2026-10-29T00:00:00Z", media_ids=(1, 2)
        ),
        _payload("c4", source_type="official", published_at="2026-10-27T00:00:00Z", media_ids=(4,)),
        _payload("c5", source_type="media", published_at="2026-10-30T00:00:00Z", media_ids=(3,)),
        _payload("c6", source_type="official", published_at="2026-10-28T00:00:00Z", media_ids=(1,)),
    ]


_VECTORS = [
    [1.0, 0.0],  # c1
    [0.9, 0.1],  # c2
    [0.0, 1.0],  # c3
    [0.5, 0.5],  # c4
    [0.2, 0.8],  # c5
    [1.0, 0.0],  # c6 (ties c1; ordering is the chunk_id tie-break)
]


class _QueryEmbedder:
    def encode_queries(self, texts: Any) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class _SpyStore:
    """Count store queries so a rejected filter can be proven to stop before retrieval."""

    def __init__(self, inner: ChromaStore) -> None:
        self.inner = inner
        self.calls = 0

    def query(
        self,
        embedding: Any,
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[Any]:
        self.calls += 1
        return self.inner.query(embedding, top_k, where)


@pytest.fixture()
def store(tmp_path: Path) -> ChromaStore:
    store = ChromaStore(tmp_path / "chroma", StoreConfig(over_fetch_factor=5))
    store.bind(SNAP, "A")
    store.upsert_chunks(_corpus(), _VECTORS)
    return store


def _search(
    store: Any,
    filters: RetrievalFilters | None = None,
    *,
    top_k: int = 10,
) -> list[Any]:
    return search(
        RetrievalQuery(text="q", top_k=top_k, snapshot_id=SNAP, filters=filters),
        store=store,
        embedder=_QueryEmbedder(),
    )


def _ids(chunks: list[Any]) -> set[str]:
    return {chunk.chunk_id for chunk in chunks}


def test_empty_filters_return_all(store: ChromaStore) -> None:
    assert _ids(_search(store)) == ALL_IDS
    assert _ids(_search(store, RetrievalFilters(source_types=[], media_ids=[]))) == ALL_IDS


def test_source_type_filter(store: ChromaStore) -> None:
    assert _ids(_search(store, RetrievalFilters(source_types=["official"]))) == {
        "c1",
        "c4",
        "c6",
    }
    assert _ids(_search(store, RetrievalFilters(source_types=["official", "media"]))) == {
        "c1",
        "c2",
        "c4",
        "c5",
        "c6",
    }


def test_date_only_same_day_is_one_utc_day(store: ChromaStore) -> None:
    assert _ids(_search(store, RetrievalFilters(since="2026-10-28", until="2026-10-28"))) == {
        "c1",
        "c2",
        "c6",
    }


def test_offset_until_is_exact_utc_instant(store: ChromaStore) -> None:
    # 2026-10-29T08:00:00+08:00 == 2026-10-29T00:00:00Z; the c3 row is exactly at the
    # exclusive upper bound and must not be returned.
    assert _ids(_search(store, RetrievalFilters(until="2026-10-29T08:00:00+08:00"))) == {
        "c1",
        "c2",
        "c4",
        "c6",
    }


def test_media_ids_any_of(store: ChromaStore) -> None:
    assert _ids(_search(store, RetrievalFilters(media_ids=[1]))) == {"c1", "c3", "c6"}
    assert _ids(_search(store, RetrievalFilters(media_ids=[1, 3]))) == {"c1", "c3", "c5", "c6"}


def test_filter_families_combine_with_and(store: ChromaStore) -> None:
    filters = RetrievalFilters(
        source_types=["official", "media"],
        since="2026-10-28",
        until="2026-10-28",
        media_ids=[1, 2],
    )
    assert _ids(_search(store, filters)) == {"c1", "c2", "c6"}


def test_restrictive_no_match_returns_empty(store: ChromaStore) -> None:
    assert _search(store, RetrievalFilters(media_ids=[999])) == []
    assert _search(store, RetrievalFilters(source_types=["metadata"], since="2026-10-30")) == []
    # The population is untouched: no filter was dropped and no fallback query ran.
    assert store.count() == len(ALL_IDS)


def test_top_k_truncates_ranked_results(store: ChromaStore) -> None:
    chunks = _search(store, top_k=3)
    assert [chunk.chunk_id for chunk in chunks] == ["c1", "c6", "c2"]


def test_top_k_above_available_returns_all(store: ChromaStore) -> None:
    assert _ids(_search(store, top_k=50)) == ALL_IDS


def test_equal_scores_order_by_chunk_id(store: ChromaStore) -> None:
    chunks = _search(store, top_k=2)
    assert [chunk.chunk_id for chunk in chunks] == ["c1", "c6"]


def test_reversed_dates_raise_before_store_query(store: ChromaStore) -> None:
    spy = _SpyStore(store)
    with pytest.raises(ValueError, match="since must be earlier"):
        _search(spy, RetrievalFilters(since="2026-10-29", until="2026-10-28"))
    assert spy.calls == 0


def test_empty_media_ids_payload_is_storable_and_not_media_matchable(
    store: ChromaStore,
) -> None:
    """Real production payload: an unresolved entity row carries media_ids=[]."""
    unresolved = _payload(
        "c7",
        source_type="official",
        published_at="2026-10-28T00:00:00Z",
        media_ids=(),
    )
    store.upsert_chunks([unresolved], [[1.0, 0.0]])
    assert "c7" in _ids(_search(store))
    assert "c7" not in _ids(_search(store, RetrievalFilters(media_ids=[1])))


def test_media_ids_can_become_unresolved_on_upsert(store: ChromaStore) -> None:
    """A linked item that later becomes unresolved must not retain its old media link."""
    linked = _payload(
        "c7",
        source_type="official",
        published_at="2026-10-28T00:00:00Z",
        media_ids=(1,),
    )
    store.upsert_chunks([linked], [[1.0, 0.0]])
    assert "c7" in _ids(_search(store, RetrievalFilters(media_ids=[1])))

    unresolved = _payload(
        "c7",
        source_type="official",
        published_at="2026-10-28T00:00:00Z",
        media_ids=(),
    )
    store.upsert_chunks([unresolved], [[1.0, 0.0]])
    assert "c7" in _ids(_search(store))
    assert "c7" not in _ids(_search(store, RetrievalFilters(media_ids=[1])))

    hits = store.query([1.0, 0.0], top_k=10)
    c7 = next(hit for hit in hits if hit.payload.chunk_id == "c7")
    assert c7.payload.media_ids == ()
