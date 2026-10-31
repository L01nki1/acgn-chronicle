"""Snapshot identity and retention safeguards against collection-name aliases."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

pytest.importorskip("chromadb")

from acgn_rag.config import StoreConfig
from acgn_rag.core.store.chroma_store import ChromaStore, collection_name
from acgn_rag.core.store.payload import ChunkPayload, payload_from_fields, to_payload

SNAP_A = "20261028-abc000000000"
SNAP_B = "20261028-abc111111111"
LEGACY_NAME = "snap_20261028-abc_A"


@pytest.fixture()
def store(tmp_path: Path) -> ChromaStore:
    return ChromaStore(tmp_path / "index" / "chroma", StoreConfig(keep_snapshots=2))


def _payload(snapshot_id: str, body: str) -> ChunkPayload:
    return payload_from_fields(
        chunk_id="doc-1::A::0",
        doc_id="doc-1",
        snapshot_id=snapshot_id,
        title="Example",
        url="https://example.com/article",
        publisher="Example",
        source_type="official",
        published_at="2026-10-28T00:00:00+00:00",
        lang="en",
        body=body,
        media_ids=[1],
    )


def _legacy_collection(store: ChromaStore, rows: list[ChunkPayload]):
    collection = store.client.create_collection(
        name=LEGACY_NAME,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"created_ts": 1},
    )
    if rows:
        collection.upsert(
            ids=[f"legacy-{index}" for index in range(len(rows))],
            embeddings=[[1.0, 0.0] for _ in rows],
            documents=[row.body for row in rows],
            metadatas=[to_payload(row) for row in rows],
        )
    return collection


def _names(store: ChromaStore) -> set[str]:
    return {
        collection if isinstance(collection, str) else collection.name
        for collection in store.client.list_collections()
    }


def _write_completed_sidecars(store: ChromaStore, snapshot_id: str, name: str) -> tuple[Path, Path]:
    folder = store.root.parent / snapshot_id / "A"
    folder.mkdir(parents=True, exist_ok=True)
    identity = {
        "snapshot_id": snapshot_id,
        "config": "A",
        "label": "A",
        "collection": name,
        "identity": {
            "model": "example",
            "backend": "torch",
            "device": "cpu",
            "precision": "fp32",
            "identity_slug": "example-torch-cpu-fp32",
            "batch_size": 32,
        },
    }
    build = identity | {"stored": 1, "chunks": 1, "timestamp": "2026-10-31T00:00:00+00:00"}
    identity_path = folder / "identity.json"
    build_path = folder / "build.json"
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    build_path.write_text(json.dumps(build), encoding="utf-8")
    return identity_path, build_path


def test_bind_existing_does_not_create_an_absent_store_directory(store: ChromaStore) -> None:
    paths = _write_completed_sidecars(store, SNAP_A, collection_name(SNAP_A, "A"))
    original = [path.read_bytes() for path in paths]
    assert not store.root.exists()
    with pytest.raises(LookupError, match="collection"):
        store.bind_existing(SNAP_A, "A")
    assert not store.root.exists()
    assert [path.read_bytes() for path in paths] == original


def test_bind_existing_does_not_initialize_an_empty_store_directory(store: ChromaStore) -> None:
    store.root.mkdir(parents=True)
    with pytest.raises(LookupError, match="collection"):
        store.bind_existing(SNAP_A, "A")
    assert list(store.root.iterdir()) == []


def test_bind_existing_rejects_a_non_file_database_path(store: ChromaStore) -> None:
    database = store.root / "chroma.sqlite3"
    database.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="invalid snapshot database"):
        store.bind_existing(SNAP_A, "A")
    assert list(database.iterdir()) == []


def test_bind_existing_reports_an_inaccessible_database(
    store: ChromaStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = store.root / "chroma.sqlite3"
    original_stat = Path.stat

    def stat_with_denied_database(path: Path, *args, **kwargs):
        if path == database:
            raise PermissionError("access denied")
        return original_stat(path, *args, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(Path, "stat", stat_with_denied_database)
        with pytest.raises(RuntimeError, match="cannot inspect the snapshot database"):
            store.bind_existing(SNAP_A, "A")
    assert not store.root.exists()


def test_bind_existing_missing_collection_does_not_create_with_completed_sidecars(
    store: ChromaStore,
) -> None:
    paths = _write_completed_sidecars(store, SNAP_A, collection_name(SNAP_A, "A"))
    original = [path.read_bytes() for path in paths]
    assert _names(store) == set()

    with pytest.raises(LookupError, match="collection"):
        store.bind_existing(SNAP_A, "A")

    assert _names(store) == set()
    assert [path.read_bytes() for path in paths] == original


def test_bind_existing_missing_snapshot_preserves_previous_binding(store: ChromaStore) -> None:
    store.bind(SNAP_A, "A")
    store.upsert_chunks([_payload(SNAP_A, "first snapshot")], [[1.0, 0.0]])
    _write_completed_sidecars(store, SNAP_B, collection_name(SNAP_B, "A"))
    before = _names(store)

    with pytest.raises(LookupError, match="collection"):
        store.bind_existing(SNAP_B, "A")

    assert _names(store) == before
    assert store.query([1.0, 0.0], top_k=1)[0].payload.snapshot_id == SNAP_A


def test_bind_existing_opens_a_verified_full_collection(store: ChromaStore) -> None:
    store.bind(SNAP_A, "A")
    store.upsert_chunks([_payload(SNAP_A, "first snapshot")], [[1.0, 0.0]])
    before = _names(store)

    assert store.bind_existing(SNAP_A, "A").name == collection_name(SNAP_A, "A")
    assert store.query([1.0, 0.0], top_k=1)[0].payload.snapshot_id == SNAP_A
    assert _names(store) == before


def test_bind_existing_uses_verified_legacy_name_from_completed_sidecars(
    store: ChromaStore,
) -> None:
    _legacy_collection(store, [_payload(SNAP_A, "legacy A")])
    _write_completed_sidecars(store, SNAP_A, LEGACY_NAME)
    before = _names(store)

    assert store.bind_existing(SNAP_A, "A").name == LEGACY_NAME
    assert store.query([1.0, 0.0], top_k=1)[0].payload.snapshot_id == SNAP_A
    assert _names(store) == before


def test_bind_existing_rejects_full_and_legacy_for_same_owner(store: ChromaStore) -> None:
    store.bind(SNAP_A, "A")
    _legacy_collection(store, [_payload(SNAP_A, "legacy A")])
    _write_completed_sidecars(store, SNAP_A, LEGACY_NAME)
    before = _names(store)

    with pytest.raises(RuntimeError, match="ambiguous snapshot"):
        store.bind_existing(SNAP_A, "A")

    assert _names(store) == before


def test_full_ids_with_the_same_old_prefix_remain_isolated(store: ChromaStore) -> None:
    assert collection_name(SNAP_A, "A") != collection_name(SNAP_B, "A")
    store.bind(SNAP_A, "A")
    store.upsert_chunks([_payload(SNAP_A, "first snapshot")], [[1.0, 0.0]])
    store.bind(SNAP_B, "A")
    store.upsert_chunks([_payload(SNAP_B, "second snapshot")], [[1.0, 0.0]])

    assert store.count() == 1
    assert store.query([1.0, 0.0], top_k=1)[0].payload.body == "second snapshot"
    store.bind(SNAP_A, "A")
    assert store.count() == 1
    assert store.query([1.0, 0.0], top_k=1)[0].payload.body == "first snapshot"


def test_drop_does_not_delete_or_invalidate_a_same_prefix_sibling(store: ChromaStore) -> None:
    store.bind(SNAP_A, "A")
    store.upsert_chunks([_payload(SNAP_A, "first snapshot")], [[1.0, 0.0]])
    store.bind(SNAP_B, "A")
    store.upsert_chunks([_payload(SNAP_B, "second snapshot")], [[1.0, 0.0]])

    assert store.drop_collection(SNAP_A, "A")
    assert store.count() == 1
    assert store.query([1.0, 0.0], top_k=1)[0].payload.snapshot_id == SNAP_B
    assert collection_name(SNAP_B, "A") in _names(store)


def test_upsert_rejects_payload_for_a_different_snapshot_before_writing(store: ChromaStore) -> None:
    store.bind(SNAP_A, "A")
    with pytest.raises(ValueError, match="snapshot"):
        store.upsert_chunks([_payload(SNAP_B, "wrong snapshot")], [[1.0, 0.0]])
    assert store.count() == 0


def test_legacy_collection_with_one_proven_owner_is_preserved(store: ChromaStore) -> None:
    _legacy_collection(store, [_payload(SNAP_A, "legacy A")])
    assert store.bind(SNAP_A, "A").name == LEGACY_NAME
    assert store.count() == 1

    assert store.bind(SNAP_B, "A").name != LEGACY_NAME
    store.upsert_chunks([_payload(SNAP_B, "new B")], [[1.0, 0.0]])
    assert store.drop_collection(SNAP_B, "A")
    assert LEGACY_NAME in _names(store)
    assert store.bind(SNAP_A, "A").count() == 1


def test_dropping_verified_legacy_does_not_invalidate_full_id_sibling(store: ChromaStore) -> None:
    _legacy_collection(store, [_payload(SNAP_A, "legacy A")])
    store.bind(SNAP_B, "A")
    store.upsert_chunks([_payload(SNAP_B, "new B")], [[1.0, 0.0]])
    assert store.drop_collection(SNAP_A, "A")
    assert store.count() == 1
    assert store.query([1.0, 0.0], top_k=1)[0].payload.snapshot_id == SNAP_B


def test_mixed_legacy_collection_fails_before_bind_or_drop(store: ChromaStore) -> None:
    legacy = _legacy_collection(store, [_payload(SNAP_A, "A"), _payload(SNAP_B, "B")])
    before = _names(store)
    with pytest.raises(RuntimeError, match="ambiguous legacy"):
        store.bind(SNAP_A, "A")
    with pytest.raises(RuntimeError, match="ambiguous legacy"):
        store.drop_collection(SNAP_A, "A")
    assert _names(store) == before
    assert legacy.count() == 2


def test_empty_unowned_legacy_collection_fails_before_mutation(store: ChromaStore) -> None:
    legacy = _legacy_collection(store, [])
    before = _names(store)
    with pytest.raises(RuntimeError, match="ambiguous legacy"):
        store.bind(SNAP_A, "A")
    with pytest.raises(RuntimeError, match="ambiguous legacy"):
        store.drop_collection(SNAP_A, "A")
    assert _names(store) == before
    assert legacy.count() == 0


def test_conflicting_legacy_sidecar_claims_fail_before_mutation(store: ChromaStore) -> None:
    legacy = _legacy_collection(store, [_payload(SNAP_A, "A")])
    _write_completed_sidecars(store, SNAP_B, LEGACY_NAME)

    before = _names(store)
    with pytest.raises(RuntimeError, match="ambiguous legacy"):
        store.bind(SNAP_A, "A")
    with pytest.raises(RuntimeError, match="ambiguous legacy"):
        store.drop_collection(SNAP_A, "A")
    assert _names(store) == before
    assert legacy.count() == 1


def test_legacy_scan_checks_rows_beyond_the_first_page(store: ChromaStore) -> None:
    legacy = _legacy_collection(store, [])
    legacy.upsert(
        ids=[f"row-{index:04d}" for index in range(501)],
        embeddings=[[1.0, 0.0]] * 501,
        metadatas=[{"snapshot_id": SNAP_A}] * 500 + [{"snapshot_id": SNAP_B}],
    )
    with pytest.raises(RuntimeError, match="ambiguous legacy"):
        store.bind(SNAP_A, "A")
    assert legacy.count() == 501


def test_legacy_row_without_snapshot_identity_is_rejected(store: ChromaStore) -> None:
    legacy = _legacy_collection(store, [])
    legacy.upsert(ids=["unknown"], embeddings=[[1.0, 0.0]], metadatas=[{"doc_id": "doc-1"}])
    with pytest.raises(RuntimeError, match="ambiguous legacy"):
        store.bind(SNAP_A, "A")
    assert legacy.count() == 1


def test_legacy_with_an_incomplete_sidecar_fails_before_mutation(store: ChromaStore) -> None:
    legacy = _legacy_collection(store, [_payload(SNAP_A, "A")])
    folder = store.root.parent / SNAP_A / "A"
    folder.mkdir(parents=True)
    (folder / "identity.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="ambiguous legacy"):
        store.drop_collection(SNAP_A, "A")
    assert legacy.count() == 1


def test_rejected_legacy_bind_keeps_previous_binding_working(store: ChromaStore) -> None:
    other = "20261027-xyz000000000"
    store.bind(other, "A")
    store.upsert_chunks([_payload(other, "other")], [[1.0, 0.0]])
    _legacy_collection(store, [_payload(SNAP_A, "A"), _payload(SNAP_B, "B")])
    with pytest.raises(RuntimeError, match="ambiguous legacy"):
        store.bind(SNAP_A, "A")
    assert store.count() == 1
    assert store.query([1.0, 0.0], top_k=1)[0].payload.snapshot_id == other


def test_full_collection_with_wrong_owner_metadata_is_rejected(store: ChromaStore) -> None:
    name = collection_name(SNAP_A, "A")
    collection = store.client.create_collection(
        name=name,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"snapshot_id": SNAP_B, "chunk_config": "A"},
    )
    with pytest.raises(RuntimeError, match="owner"):
        store.bind(SNAP_A, "A")
    with pytest.raises(RuntimeError, match="owner"):
        store.bind_existing(SNAP_A, "A")
    with pytest.raises(RuntimeError, match="owner"):
        store.drop_collection(SNAP_A, "A")
    assert name in _names(store)
    assert collection.count() == 0


def test_duplicate_full_and_legacy_collections_are_not_silently_selected(
    store: ChromaStore,
) -> None:
    store.bind(SNAP_A, "A")
    _legacy_collection(store, [_payload(SNAP_A, "legacy")])
    before = _names(store)
    with pytest.raises(RuntimeError, match="ambiguous snapshot"):
        store.bind(SNAP_A, "A")
    with pytest.raises(RuntimeError, match="ambiguous snapshot"):
        store.drop_collection(SNAP_A, "A")
    assert _names(store) == before


def test_prune_refuses_to_guess_publication_order(store: ChromaStore) -> None:
    store.bind(SNAP_A, "A")
    store.bind(SNAP_B, "A")
    before = _names(store)
    with pytest.raises(RuntimeError, match="publication"):
        store.prune(keep=1)
    assert _names(store) == before


def test_prune_uses_validated_publication_times_not_hash_order(store: ChromaStore) -> None:
    store.bind(SNAP_A, "A")
    store.bind(SNAP_B, "A")
    published = {
        SNAP_A: datetime(2026, 11, 1, tzinfo=UTC),
        SNAP_B: datetime(2026, 10, 31, tzinfo=UTC),
    }
    assert store.prune(keep=1, publication_times=published) == [collection_name(SNAP_B, "A")]
    assert _names(store) == {collection_name(SNAP_A, "A")}


def test_prune_rejects_equal_publication_times_before_deleting(store: ChromaStore) -> None:
    store.bind(SNAP_A, "A")
    store.bind(SNAP_B, "A")
    before = _names(store)
    instant = datetime(2026, 10, 31, tzinfo=UTC)
    with pytest.raises(RuntimeError, match="publication"):
        store.prune(keep=1, publication_times={SNAP_A: instant, SNAP_B: instant})
    assert _names(store) == before


def test_prune_rejects_missing_or_naive_publication_times(store: ChromaStore) -> None:
    store.bind(SNAP_A, "A")
    store.bind(SNAP_B, "A")
    before = _names(store)
    with pytest.raises(RuntimeError, match="publication"):
        store.prune(keep=1, publication_times={SNAP_A: datetime(2026, 10, 31, tzinfo=UTC)})
    with pytest.raises(RuntimeError, match="publication"):
        store.prune(
            keep=1,
            publication_times={
                SNAP_A: datetime(2026, 10, 31, tzinfo=UTC),
                SNAP_B: datetime(2026, 11, 1),
            },
        )
    assert _names(store) == before


def test_prune_keeps_all_configs_of_the_newer_snapshot(store: ChromaStore) -> None:
    for snapshot in (SNAP_A, SNAP_B):
        for config in ("A", "A0"):
            store.bind(snapshot, config)
    published = {
        SNAP_A: datetime(2026, 11, 1, tzinfo=UTC),
        SNAP_B: datetime(2026, 10, 31, tzinfo=UTC),
    }
    assert set(store.prune(keep=1, publication_times=published)) == {
        collection_name(SNAP_B, "A"),
        collection_name(SNAP_B, "A0"),
    }
    assert _names(store) == {collection_name(SNAP_A, "A"), collection_name(SNAP_A, "A0")}


def test_ambiguous_collection_name_segments_are_rejected() -> None:
    with pytest.raises(ValueError):
        collection_name("snapshot_with_separator", "A")
    with pytest.raises(ValueError):
        collection_name("snapshot", "A_with_separator")
