"""ChromaStore on an embedded chromadb 1.x instance: offline, tmp_path, telemetry off.

Requires the [store] extra; skipped whole-module without it, so a bare
`pip install -e ".[dev]"` clone still passes (RUNBOOK §1). The supported band starts at
1.5.0 for typed array metadata and the `$contains` filter.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

chromadb = pytest.importorskip("chromadb")

from acgn_rag.config import StoreConfig  # noqa: E402
from acgn_rag.core.store.chroma_store import (  # noqa: E402
    ChromaStore,
    collection_name,
    media_filter,
)
from acgn_rag.core.store.payload import ChunkPayload, payload_from_fields, to_payload  # noqa: E402

SNAP = "20261028abcdef1234"
VECTORS = [[1.0, 0.0], [0.0, 1.0]]


def _payload(chunk_id: str, media_ids: tuple[int, ...], **overrides: Any) -> ChunkPayload:
    fields: dict[str, Any] = {
        "chunk_id": chunk_id,
        "doc_id": "doc-1",
        "snapshot_id": SNAP,
        "title": f"Title {chunk_id}",
        "url": f"https://example.com/{chunk_id}",
        "publisher": "Capcom",
        "source_type": "official",
        "published_at": "2026-10-28T00:00:00+00:00",
        "lang": "ja",
        "body": f"Body {chunk_id}",
        "media_ids": media_ids,
    }
    fields.update(overrides)
    return payload_from_fields(**fields)


def _two_chunks() -> list[ChunkPayload]:
    return [
        _payload("doc-1::A::0", (1, 2), ingest_category="corporate"),
        _payload("doc-1::A::1", (3,)),
    ]


@pytest.fixture()
def store(tmp_path: Path) -> ChromaStore:
    return ChromaStore(
        tmp_path / "chroma",
        StoreConfig(backend="chroma", keep_snapshots=2, over_fetch_factor=2),
    )


@pytest.fixture()
def loaded(store: ChromaStore) -> ChromaStore:
    store.bind(SNAP, "A")
    store.upsert_chunks(_two_chunks(), VECTORS)
    return store


def test_chromadb_is_in_supported_band() -> None:
    major, minor = (int(part) for part in chromadb.__version__.split(".")[:2])
    assert (1, 5) <= (major, minor) < (2, 0)


def test_collection_name_format() -> None:
    assert collection_name(SNAP, "A") == f"snap_full_{SNAP}_A"
    assert collection_name("short", "A0") == "snap_full_short_A0"


def test_media_filter_shape() -> None:
    assert media_filter([]) is None  # absent filter = no media constraint, not "match nothing"
    # 1.x rejects an $or with fewer than two clauses, so a single id returns the bare condition
    assert media_filter([3]) == {"media_ids": {"$contains": 3}}
    assert media_filter([1, 3]) == {
        "$or": [{"media_ids": {"$contains": 1}}, {"media_ids": {"$contains": 3}}]
    }


def test_operations_before_bind_raise(store: ChromaStore) -> None:
    with pytest.raises(RuntimeError, match="bind"):
        store.count()
    with pytest.raises(RuntimeError, match="bind"):
        store.query([1.0, 0.0], top_k=1)


def test_upsert_rejects_length_mismatch(loaded: ChromaStore) -> None:
    with pytest.raises(ValueError, match="lengths differ"):
        loaded.upsert_chunks(_two_chunks(), [VECTORS[0]])


def test_upsert_and_query_accept_numpy_embeddings(loaded: ChromaStore) -> None:
    # Encoder output is numpy by contract; chromadb 1.x rejects numpy scalars inside
    # embeddings, so the store must convert at the boundary (found live by build_index).
    np = pytest.importorskip("numpy")
    vectors = [np.array(vector, dtype=np.float32) for vector in VECTORS]
    loaded.upsert_chunks(_two_chunks(), vectors)  # idempotent overwrite of the same ids
    hits = loaded.query(np.array([1.0, 0.0], dtype=np.float32), top_k=2)
    assert [hit.payload.chunk_id for hit in hits] == ["doc-1::A::0", "doc-1::A::1"]


def test_cosine_scores_with_known_vectors(loaded: ChromaStore) -> None:
    hits = loaded.query([1.0, 0.0], top_k=2)
    assert [hit.payload.chunk_id for hit in hits] == ["doc-1::A::0", "doc-1::A::1"]
    # score = 1 - cosine distance: identical vectors -> 1, orthogonal -> 0. Under the default
    # l2 space the second score would be 1 - sqrt(2) < 0, so this pins hnsw:space="cosine".
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert hits[1].score == pytest.approx(0.0, abs=1e-5)


def test_upsert_query_round_trips_payloads(loaded: ChromaStore) -> None:
    hits = loaded.query([0.0, 1.0], top_k=1)
    assert len(hits) == 1
    assert hits[0].payload == _payload("doc-1::A::1", (3,))
    assert hits[0].payload.body == "Body doc-1::A::1"
    assert hits[0].payload.media_ids == (3,)


def test_upsert_is_idempotent_on_chunk_ids(loaded: ChromaStore) -> None:
    assert loaded.count() == 2
    loaded.upsert_chunks(_two_chunks(), VECTORS)  # same chunk_ids: upsert, not duplicate add
    assert loaded.count() == 2


def test_mixed_empty_and_linked_media_ids_round_trip(store: ChromaStore) -> None:
    store.bind(SNAP, "A")
    empty = _payload("empty", ())
    linked = _payload("linked", (7,))
    store.upsert_chunks([empty, linked], VECTORS)

    hits = store.query([1.0, 0.0], top_k=2)
    assert {hit.payload.chunk_id: hit.payload for hit in hits} == {
        "empty": empty,
        "linked": linked,
    }
    assert [hit.payload for hit in store.query([1.0, 0.0], top_k=2, where=media_filter([7]))] == [
        linked
    ]


def test_upsert_clears_and_relinks_optional_metadata(store: ChromaStore) -> None:
    store.bind(SNAP, "A")
    original = _payload("changed", (7,), ingest_category="corporate")
    unaffected = _payload("other", (42,), ingest_category="other")
    store.upsert_chunks([original, unaffected], VECTORS)

    cleared = _payload("changed", ())
    store.upsert_chunks([cleared], [VECTORS[0]])
    hits = store.query([1.0, 0.0], top_k=2)
    assert {hit.payload.chunk_id: hit.payload for hit in hits} == {
        "changed": cleared,
        "other": unaffected,
    }
    stored = store.client.get_collection(collection_name(SNAP, "A")).get(
        ids=["changed"], include=["metadatas"]
    )["metadatas"][0]
    assert "media_ids" not in stored
    assert "ingest_category" not in stored
    assert store.query([1.0, 0.0], top_k=2, where=media_filter([7])) == []
    assert store.query([1.0, 0.0], top_k=2, where={"ingest_category": {"$eq": "corporate"}}) == []
    assert [hit.payload for hit in store.query([1.0, 0.0], top_k=2, where=media_filter([42]))] == [
        unaffected
    ]

    relinked = _payload("changed", (9,), ingest_category="relinked")
    store.upsert_chunks([relinked], [VECTORS[0]])
    assert [hit.payload for hit in store.query([1.0, 0.0], top_k=2, where=media_filter([9]))] == [
        relinked
    ]
    assert store.query([1.0, 0.0], top_k=2, where=media_filter([7])) == []
    assert store.count() == 2


@pytest.mark.parametrize(
    ("media_ids", "category"),
    [((7,), None), ((), "corporate")],
    ids=["clear-category-only", "clear-ids-only"],
)
def test_upsert_clears_only_the_requested_metadata(
    store: ChromaStore, media_ids: tuple[int, ...], category: str | None
) -> None:
    store.bind(SNAP, "A")
    store.upsert_chunks([_payload("changed", (7,), ingest_category="corporate")], [VECTORS[0]])
    updated = _payload("changed", media_ids, ingest_category=category)
    store.upsert_chunks([updated], [VECTORS[0]])

    assert store.query([1.0, 0.0], top_k=1)[0].payload == updated
    id_hits = store.query([1.0, 0.0], top_k=1, where=media_filter([7]))
    category_hits = store.query(
        [1.0, 0.0], top_k=1, where={"ingest_category": {"$eq": "corporate"}}
    )
    assert [hit.payload for hit in id_hits] == ([updated] if media_ids else [])
    assert [hit.payload for hit in category_hits] == ([updated] if category else [])


def test_media_filter_contains_actually_filters(loaded: ChromaStore) -> None:
    hits = loaded.query([1.0, 0.0], top_k=5, where=media_filter([3]))
    assert [hit.payload.chunk_id for hit in hits] == ["doc-1::A::1"]
    hits_both = loaded.query([1.0, 0.0], top_k=5, where=media_filter([1, 3]))
    assert {hit.payload.chunk_id for hit in hits_both} == {"doc-1::A::0", "doc-1::A::1"}


def test_top_k_above_available_returns_all_without_raising(loaded: ChromaStore) -> None:
    # 2 rows in the collection, top_k=5: the assert is >= min(top_k, available) = 2, so the
    # over-fetch machinery must NOT raise on a small collection.
    hits = loaded.query([1.0, 0.0], top_k=5)
    assert len(hits) == 2


def test_query_raises_when_hnsw_under_returns(
    loaded: ChromaStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # HNSW ef/M under-return cannot be triggered deterministically, so simulate it: drop one
    # row from the raw fetch while _available still reports 2 matching rows.
    real_fetch = loaded._fetch_rows

    def drop_one(embedding: list[float], n_results: int, where: dict[str, Any] | None):
        return real_fetch(embedding, n_results, where)[:-1]

    monkeypatch.setattr(loaded, "_fetch_rows", drop_one)
    with pytest.raises(RuntimeError, match="under-returned"):
        loaded.query([1.0, 0.0], top_k=5)


def test_filtered_availability_probe_is_bounded(
    store: ChromaStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection = store.bind(SNAP, "A")
    payloads = [_payload(f"doc-{index}::A::1", (1 if index < 30 else 2,)) for index in range(2000)]
    store.upsert_chunks(payloads, [[1.0, 0.0]] * len(payloads))
    calls: list[dict[str, Any]] = []
    original_get = type(collection).get

    def observed_get(self, *args, **kwargs):
        calls.append(kwargs)
        return original_get(self, *args, **kwargs)

    monkeypatch.setattr(type(collection), "get", observed_get)
    # This regression measures the real filtered probe, not approximate-neighbor recall.
    # Equal vectors can make HNSW under-return nondeterministically; its guard has a separate test.
    monkeypatch.setattr(
        store,
        "_fetch_rows",
        lambda *_args: [(p.chunk_id, p.body, to_payload(p), 0.0) for p in payloads[:6]],
    )
    assert len(store.query([1.0, 0.0], top_k=3, where=media_filter([1]))) == 3
    assert len(calls) == 1
    assert calls[0]["include"] == []
    assert 0 < calls[0].get("limit", 2000) <= 3 * store.settings.over_fetch_factor + 1


def test_empty_replacement_collection_is_an_error(loaded: ChromaStore) -> None:
    loaded.drop_collection(SNAP, "A")
    loaded.bind(SNAP, "A")
    with pytest.raises(RuntimeError, match="empty"):
        loaded.query([1.0, 0.0], top_k=1)
    loaded.upsert_chunks(_two_chunks(), VECTORS)
    assert len(loaded.query([1.0, 0.0], top_k=1)) == 1


def test_no_filter_match_is_empty_success_on_a_populated_collection(loaded: ChromaStore) -> None:
    assert loaded.query([1.0, 0.0], top_k=3, where=media_filter([9999])) == []


@pytest.mark.parametrize("top_k", [0, -1])
def test_query_rejects_nonpositive_result_count(loaded: ChromaStore, top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k"):
        loaded.query([1.0, 0.0], top_k=top_k)


def test_prune_keeps_newest_snapshots(store: ChromaStore) -> None:
    store.client.get_or_create_collection(name="unrelated")  # non-snap_* must never be pruned
    snapshots = ("20260101aaaa1111", "20260102bbbb2222", "20260103cccc3333")
    for snapshot_id in snapshots:
        store.bind(snapshot_id, "A")
        store.upsert_chunks(
            [_payload(f"{snapshot_id}::A::0", (1,), snapshot_id=snapshot_id)], [[1.0, 0.0]]
        )
    publication_times = {
        snapshot_id: datetime(2026, 1, index, tzinfo=UTC)
        for index, snapshot_id in enumerate(snapshots, 1)
    }
    deleted = store.prune(keep=2, publication_times=publication_times)
    assert deleted == [collection_name(snapshots[0], "A")]
    remaining = {
        col if isinstance(col, str) else col.name for col in store.client.list_collections()
    }
    assert remaining == {
        collection_name(snapshots[1], "A"),
        collection_name(snapshots[2], "A"),
        "unrelated",
    }


def test_prune_keeps_everything_within_retention(store: ChromaStore) -> None:
    store.bind("20260101aaaa1111", "A")
    store.upsert_chunks(
        [_payload("20260101aaaa1111::A::0", (1,), snapshot_id="20260101aaaa1111")],
        [[1.0, 0.0]],
    )
    assert store.prune(keep=2) == []
    assert store.count() == 1  # the bound collection survived


def test_drop_collection_clears_the_binding_only_for_the_dropped_pair(store: ChromaStore) -> None:
    """Dropping a sibling pair must not invalidate the live binding (as in ``prune``); only
    dropping the bound pair itself clears it, and the next ``bind`` recreates it empty."""
    store.bind(SNAP, "A")
    store.upsert_chunks(_two_chunks(), VECTORS)
    store.bind(SNAP, "A0")
    store.upsert_chunks([_payload("doc-1::A0::0", (1,))], [[1.0, 0.0]])
    store.bind(SNAP, "A")  # the working binding is A; A0 is a sibling collection

    assert store.drop_collection(SNAP, "A0") is True  # the sibling existed and is gone
    assert {
        col if isinstance(col, str) else col.name for col in store.client.list_collections()
    } == {collection_name(SNAP, "A")}
    assert store.count() == 2  # the A binding survived the sibling's deletion

    assert store.drop_collection(SNAP, "no-such-config") is False  # nothing to delete
    assert store.count() == 2  # ... and the binding still survives

    assert store.drop_collection(SNAP, "A") is True  # the bound pair itself
    with pytest.raises(RuntimeError, match="must be called before"):
        store.count()  # the binding was cleared with it
    store.bind(SNAP, "A")  # the following bind recreates the collection, empty
    assert store.count() == 0
