"""ChromaStore on an embedded chromadb 1.x instance: offline, tmp_path, telemetry off.

Requires the [store] extra; skipped whole-module without it, so a bare
`pip install -e ".[dev]"` clone still passes (RUNBOOK §1). Written for the 1.x API
(`configuration=`, typed array metadata): the drafting venv still carries the old 0.6.3
band, where only the version anchor fails by design - see NOTES.md (verification) for what
was exercised on which band.
"""

from __future__ import annotations

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
from acgn_rag.core.store.payload import ChunkPayload, payload_from_fields  # noqa: E402

SNAP = "20261028abcdef1234"  # longer than 12 chars, so naming truncation is observable
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


def test_chromadb_is_1x() -> None:
    # The store is written for the 1.x API (configuration=, typed array metadata); see NOTES.
    assert int(chromadb.__version__.split(".")[0]) == 1


def test_collection_name_format() -> None:
    assert collection_name(SNAP, "A") == f"snap_{SNAP[:12]}_A"
    assert collection_name("short", "A0") == "snap_short_A0"


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


def test_prune_keeps_newest_snapshots(store: ChromaStore) -> None:
    store.client.get_or_create_collection(name="unrelated")  # non-snap_* must never be pruned
    for snapshot_id in ("20260101aaaa1111", "20260102bbbb2222", "20260103cccc3333"):
        store.bind(snapshot_id, "A")
        store.upsert_chunks([_payload(f"{snapshot_id}::A::0", (1,))], [[1.0, 0.0]])
    deleted = store.prune(keep=2)  # keep_n defaults to settings.keep_snapshots when omitted
    assert deleted == ["snap_20260101aaaa_A"]
    remaining = {
        col if isinstance(col, str) else col.name for col in store.client.list_collections()
    }
    assert remaining == {"snap_20260102bbbb_A", "snap_20260103cccc_A", "unrelated"}


def test_prune_keeps_everything_within_retention(store: ChromaStore) -> None:
    store.bind("20260101aaaa1111", "A")
    store.upsert_chunks([_payload("20260101aaaa1111::A::0", (1,))], [[1.0, 0.0]])
    assert store.prune(keep=2) == []
    assert store.count() == 1  # the bound collection survived
