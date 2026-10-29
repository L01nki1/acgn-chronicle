"""Chroma 1.x embedded vector store: one collection per (snapshot, chunk config).

Owner: F | Spec: 04 §7 D4, docs/RUNBOOK.md §3.1 (chromadb >=1.0,<2 rewrite), PLAN PR3

API surface used (all 1.x; nothing here runs on 0.5.x):
  - ``PersistentClient(path=..., settings=Settings(anonymized_telemetry=False))``: embedded,
    offline, telemetry off; the client is created lazily so importing this module costs
    nothing without the ``[store]`` extra.
  - ``get_or_create_collection(configuration={"hnsw": {"space": "cosine"}})``: the default
    space is still ``l2`` and would turn ``score`` into a distance while the contract
    promises a similarity (ports.SCORE_SEMANTICS).
  - Typed array metadata: ``media_ids`` is stored natively as ``int[]`` and filtered with
    ``$contains`` (``media_filter``).

Where-filtering is pre-KNN in 1.x, but HNSW can still under-return when ``ef``/``M`` are
small relative to the filtered candidate set, so every query over-fetches by
``StoreConfig.over_fetch_factor`` and asserts the row count (RUNBOOK §3.1 pitfall 3).
``score = 1 - cosine_distance`` (RUNBOOK §3.1).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from acgn_rag.config import StoreConfig
from acgn_rag.core.store.payload import ChunkPayload, payload_from_metadata, to_payload

_SNAP_PREFIX = "snap_"
_CREATED_TS_KEY = "created_ts"


@dataclass(frozen=True)
class ScoredPayload:
    """One query hit: the round-tripped chunk plus its score.

    ``score`` is cosine similarity (1 - Chroma cosine distance); it is ordinal within one
    response only - never compare across queries, chunk configurations or snapshots
    (ports.SCORE_SEMANTICS).
    """

    payload: ChunkPayload
    score: float


def collection_name(snapshot_id: str, chunk_config: str) -> str:
    """``snap_<snapshot_id[:12]>_<chunk_config>`` - the naming convention in config.py."""
    return f"{_SNAP_PREFIX}{snapshot_id[:12]}_{chunk_config}"


def _snapshot_key(name: str) -> str:
    """Invert `collection_name`: the snapshot part of a snap_* collection name.

    Assumes the chunk-config label contains no underscore (labels in use: A, A-prime, A0, B).
    """
    return name[len(_SNAP_PREFIX) :].rsplit("_", 1)[0]


def _floats(embedding: Sequence[float]) -> list[float]:
    """Plain-Python floats: chromadb 1.x rejects numpy scalars inside embeddings.

    ``list(np.array)`` passes ``numpy.float32`` objects straight through, so callers that
    hand over encoder output (numpy by contract, PR #20) must be converted here once.
    """
    tolist = getattr(embedding, "tolist", None)
    if callable(tolist):
        return [float(value) for value in tolist()]
    return [float(value) for value in embedding]


def media_filter(media_ids: list[int]) -> dict[str, Any] | None:
    """Any-of filter over the native ``media_ids`` int[] via the 1.x ``$contains`` operator.

    An empty list returns None: an absent filter must mean "no media constraint", not
    "match nothing". A single id returns the bare ``$contains`` condition - 1.x rejects an
    ``$or`` with fewer than two clauses ("list with at least two where expressions").
    Sized for the small any-of lists ``RetrievalFilters.media_ids`` implies; long any-of
    lists would justify the SQLite reverse-table option noted in RUNBOOK §3.1.
    """
    if not media_ids:
        return None
    contains = [{"media_ids": {"$contains": int(media_id)}} for media_id in media_ids]
    if len(contains) == 1:
        return contains[0]
    return {"$or": contains}


def _field(result: Any, key: str) -> Any:
    """Read one field off a Chroma result: 1.x returns models, older wheels returned dicts."""
    if isinstance(result, dict):
        return result[key]
    return getattr(result, key)


def _assert_min_rows(returned: int, top_k: int, available: int, name: str) -> None:
    """The RUNBOOK §3.1 pitfall-3 assert: over-fetch must satisfy min(top_k, available)."""
    expected = min(top_k, available)
    if returned < expected:
        raise RuntimeError(
            f"Chroma under-returned from collection {name!r}: got {returned} rows, expected "
            f">= {expected} (top_k={top_k}, matching rows={available}). Raise ACGN_OVERFETCH "
            "(StoreConfig.over_fetch_factor) or rebuild the index; see RUNBOOK §3.1 pitfall 3."
        )


class ChromaStore:
    """Embedded Chroma client bound to one (snapshot, chunk config) collection at a time.

    ``bind`` selects (creating if needed) the collection for a snapshot + chunk config;
    ``upsert_chunks`` / ``query`` / ``count`` operate on the bound collection, and ``prune``
    manages retention across all snap_* collections in the root.
    """

    def __init__(self, root: Path, settings: StoreConfig) -> None:
        self.root = Path(root)
        self.settings = settings
        self._client: Any | None = None
        self._collection: Any | None = None

    @property
    def client(self) -> Any:
        """The persistent embedded client, created lazily (chromadb import is not cheap)."""
        if self._client is None:
            import chromadb
            from chromadb.config import Settings

            self._client = chromadb.PersistentClient(
                path=str(self.root),
                settings=Settings(anonymized_telemetry=False),
            )
        return self._client

    def bind(self, snapshot_id: str, chunk_config: str) -> Any:
        """Select the collection for (snapshot, chunk config), creating it with cosine HNSW.

        Creation also stamps a ``created_ts`` int-epoch marker into the collection metadata
        so ``prune`` can order snapshots even if snapshot ids ever stop being date-first.
        Some 1.x minors reject ``configuration`` and ``metadata`` together; that retry drops
        the stamp, and prune falls back to name order (date-first ids sort chronologically).
        """
        name = collection_name(snapshot_id, chunk_config)
        configuration = {"hnsw": {"space": "cosine"}}
        try:
            collection = self.client.get_or_create_collection(
                name=name,
                configuration=configuration,
                metadata={_CREATED_TS_KEY: int(time.time())},
            )
        except (TypeError, ValueError):
            collection = self.client.get_or_create_collection(
                name=name, configuration=configuration
            )
        self._collection = collection
        return collection

    def _require_collection(self) -> Any:
        if self._collection is None:
            raise RuntimeError(
                "ChromaStore.bind(snapshot_id, chunk_config) must be called before "
                "upsert_chunks / query / count."
            )
        return self._collection

    def upsert_chunks(
        self, payloads: Sequence[ChunkPayload], embeddings: Sequence[Sequence[float]]
    ) -> None:
        """Upsert chunks into the bound collection: ids = chunk_id, documents = body.

        Metadata rows come from ``to_payload`` (flat scalars + media_ids as int[]). Upsert,
        not add: rebuilding the same snapshot must be idempotent, not duplicate-keyed.
        """
        collection = self._require_collection()
        if len(payloads) != len(embeddings):
            raise ValueError(
                f"payloads and embeddings lengths differ: {len(payloads)} != {len(embeddings)}"
            )
        if not payloads:
            return
        collection.upsert(
            ids=[payload.chunk_id for payload in payloads],
            embeddings=[_floats(embedding) for embedding in embeddings],
            documents=[payload.body for payload in payloads],
            metadatas=[to_payload(payload) for payload in payloads],
        )

    def _fetch_rows(
        self, embedding: Sequence[float], n_results: int, where: dict[str, Any] | None
    ) -> list[tuple[str, str, dict[str, Any], float]]:
        result = self._require_collection().query(
            query_embeddings=[_floats(embedding)],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = _field(result, "ids")[0] or []
        documents = _field(result, "documents")[0] or []
        metadatas = _field(result, "metadatas")[0] or []
        distances = _field(result, "distances")[0] or []
        return list(zip(ids, documents, metadatas, distances, strict=True))

    def _available(self, where: dict[str, Any] | None) -> int:
        """Rows matching the filter: the denominator of the pitfall-3 assert."""
        collection = self._require_collection()
        if where is None:
            return int(collection.count())
        return len(_field(collection.get(where=where, include=[]), "ids"))

    def query(
        self, embedding: Sequence[float], top_k: int, where: dict[str, Any] | None = None
    ) -> list[ScoredPayload]:
        """Nearest chunks by cosine similarity: over-fetch, assert the row count, truncate.

        Chroma 1.x filters pre-KNN, but HNSW can still under-return (ef/M), so the internal
        request is ``top_k x over_fetch_factor``; the assert then guarantees at least
        ``min(top_k, available_after_filter)`` rows before truncation to ``top_k``.
        """
        collection = self._require_collection()
        n_results = top_k * max(self.settings.over_fetch_factor, 1)
        rows = self._fetch_rows(embedding, n_results, where)
        _assert_min_rows(len(rows), top_k, self._available(where), collection.name)
        hits = [
            ScoredPayload(payload=payload_from_metadata(metadata, document), score=1.0 - distance)
            for (_chunk_id, document, metadata, distance) in rows
        ]
        return hits[:top_k]

    def count(self) -> int:
        """Rows in the bound collection."""
        return int(self._require_collection().count())

    def drop_collection(self, snapshot_id: str, chunk_config: str) -> bool:
        """Delete exactly the (snapshot, chunk config) collection; True when it existed.

        The scoped replacement primitive behind ``build_index.py --fresh``: the name comes
        from ``collection_name``, so only that pair's ``snap_*`` collection can ever be
        targeted and no sibling snapshot or config is touched. A binding to the dropped pair
        is cleared so the following ``bind`` recreates the collection with cosine HNSW and a
        fresh ``created_ts`` stamp (build_index does both immediately after); a binding to a
        different pair stays valid, exactly as in ``prune``.
        """
        name = collection_name(snapshot_id, chunk_config)
        # collection_name always emits snap_* names; the assert pins the contract this
        # deletion relies on instead of a check that could never fire.
        assert name.startswith(_SNAP_PREFIX)
        existed = any(
            (collection if isinstance(collection, str) else collection.name) == name
            for collection in self.client.list_collections()
        )
        if existed:
            self.client.delete_collection(name)
        if getattr(self._collection, "name", None) == name:
            self._collection = None  # the bound collection was itself dropped (as in prune)
        return existed

    def prune(self, keep: int | None = None) -> list[str]:
        """Delete snap_* collections of snapshots beyond retention; return the deleted names.

        Retention counts SNAPSHOTS (ACGN_KEEP_SNAPSHOTS, RUNBOOK §3.2), not collections: a
        snapshot owns one collection per chunk config, and pruning removes the oldest
        snapshots' collections wholesale. Snapshots are ordered by the newest created_ts
        among their collections (0 when unstamped), tie-broken by name - snapshot ids are
        date-first (e.g. 20261028abcdef), so name order is chronological order. Collections
        outside the snap_* prefix are never touched.
        """
        keep_n = self.settings.keep_snapshots if keep is None else keep
        if keep_n < 0:
            raise ValueError("keep must be >= 0")
        groups: dict[str, list[str]] = {}
        stamps: dict[str, int] = {}
        for collection in self.client.list_collections():
            name = collection if isinstance(collection, str) else collection.name
            if not name.startswith(_SNAP_PREFIX):
                continue
            metadata = {} if isinstance(collection, str) else dict(collection.metadata or {})
            key = _snapshot_key(name)
            groups.setdefault(key, []).append(name)
            stamps[key] = max(stamps.get(key, 0), int(metadata.get(_CREATED_TS_KEY, 0)))
        if len(groups) <= keep_n:
            return []
        ordered = sorted(groups, key=lambda key: (stamps[key], key))
        doomed = [name for key in ordered[: len(groups) - keep_n] for name in groups[key]]
        for name in doomed:
            self.client.delete_collection(name)
        if getattr(self._collection, "name", None) in doomed:
            self._collection = None  # the bound collection was itself retired
        return sorted(doomed)


__all__ = ["ChromaStore", "ScoredPayload", "collection_name", "media_filter"]
