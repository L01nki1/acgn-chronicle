"""Chroma 1.x embedded vector store: one collection per (snapshot, chunk config).

Owner: F | Spec: 04 §7 D4, docs/RUNBOOK.md §3.1 (chromadb >=1.0,<2 rewrite), PLAN PR3

API surface used (all 1.x; nothing here runs on 0.5.x):
  - ``PersistentClient(path=..., settings=Settings(anonymized_telemetry=False))``: embedded,
    offline, telemetry off; the client is created lazily so importing this module costs
    nothing without the ``[store]`` extra.
  - ``create_collection(configuration={"hnsw": {"space": "cosine"}})`` for a new
    verified name: the default space is still ``l2`` and would turn ``score`` into a
    distance while the contract promises a similarity (ports.SCORE_SEMANTICS).
  - Typed array metadata: ``media_ids`` is stored natively as ``int[]`` and filtered with
    ``$contains`` (``media_filter``).

Where-filtering is pre-KNN in 1.x, but HNSW can still under-return when ``ef``/``M`` are
small relative to the filtered candidate set, so every query over-fetches by
``StoreConfig.over_fetch_factor`` and asserts the row count (RUNBOOK §3.1 pitfall 3).
``score = 1 - cosine_distance`` (RUNBOOK §3.1).
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from stat import S_ISREG
from typing import Any

from acgn_rag.config import StoreConfig
from acgn_rag.core.store.payload import ChunkPayload, payload_from_metadata, to_payload

_SNAP_PREFIX = "snap_"
_FULL_PREFIX = "snap_full_"
_CREATED_TS_KEY = "created_ts"
_SNAPSHOT_ID_KEY = "snapshot_id"
_CHUNK_CONFIG_KEY = "chunk_config"
_NAMING_VERSION_KEY = "snapshot_name_version"
_NAMING_VERSION = 2
_SEGMENT = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?")
_LEGACY_PAGE_SIZE = 500
_PERSISTENT_DB = "chroma.sqlite3"


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
    """Return a disjoint, reversible name containing the complete snapshot identity."""
    _validate_segment(snapshot_id, "snapshot_id")
    _validate_segment(chunk_config, "chunk_config")
    name = f"{_FULL_PREFIX}{snapshot_id}_{chunk_config}"
    if len(name) > 512:
        raise ValueError(
            "snapshot_id and chunk_config produce a collection name over 512 characters"
        )
    return name


def _validate_segment(value: str, label: str) -> None:
    if not isinstance(value, str) or not _SEGMENT.fullmatch(value) or ".." in value:
        raise ValueError(f"{label} must contain only letters, digits, dots or hyphens")


def _legacy_collection_name(snapshot_id: str, chunk_config: str) -> str:
    return f"{_SNAP_PREFIX}{snapshot_id[:12]}_{chunk_config}"


def _parse_collection_name(name: str) -> tuple[bool, str, str]:
    """Parse full names and legacy short names without inferring a legacy owner."""
    if name.startswith(_FULL_PREFIX):
        parts = name[len(_FULL_PREFIX) :].rsplit("_", 1)
        if len(parts) == 2:
            try:
                if collection_name(parts[0], parts[1]) == name:
                    return True, parts[0], parts[1]
            except ValueError:
                pass
    if name.startswith(_SNAP_PREFIX):
        parts = name[len(_SNAP_PREFIX) :].rsplit("_", 1)
        if len(parts) == 2:
            prefix, chunk_config = parts
            try:
                _validate_segment(prefix, "legacy prefix")
                _validate_segment(chunk_config, "chunk_config")
            except ValueError as exc:
                raise RuntimeError(f"invalid legacy collection name {name!r}") from exc
            if len(prefix) <= 12:
                return False, prefix, chunk_config
    raise RuntimeError(f"unrecognized snapshot collection name {name!r}")


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
    ``bind_existing`` opens only a verified existing collection for a read;
    ``upsert_chunks`` / ``query`` / ``count`` operate on the bound collection, and ``prune``
    manages retention across all snap_* collections in the root.
    """

    def __init__(self, root: Path, settings: StoreConfig) -> None:
        self.root = Path(root)
        self.settings = settings
        self._client: Any | None = None
        self._collection: Any | None = None
        self._bound_snapshot_id: str | None = None

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

    def _collections(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for collection in self.client.list_collections():
            if isinstance(collection, str):
                result[collection] = self.client.get_collection(collection)
            else:
                result[collection.name] = collection
        return result

    @staticmethod
    def _sidecar_exists(path: Path, collection_name: str) -> bool:
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise RuntimeError(
                f"ambiguous legacy collection {collection_name!r}: inaccessible sidecars"
            ) from exc
        if not S_ISREG(mode):
            raise RuntimeError(f"ambiguous legacy collection {collection_name!r}: invalid sidecar")
        return True

    def _legacy_sidecar_owners(self, name: str, chunk_config: str, count: int) -> set[str]:
        """Find completed sidecars that claim a legacy physical collection."""
        owners: set[str] = set()
        index_root = self.root.parent
        if not index_root.is_dir():
            return owners
        for candidate in index_root.iterdir():
            snapshot_id = candidate.name
            try:
                if _legacy_collection_name(snapshot_id, chunk_config) != name:
                    continue
                collection_name(snapshot_id, chunk_config)
            except ValueError as exc:
                raise RuntimeError(
                    f"ambiguous legacy collection {name!r}: invalid sidecar path"
                ) from exc
            if candidate.is_symlink():
                raise RuntimeError(f"ambiguous legacy collection {name!r}: symlinked index path")
            if not candidate.is_dir():
                continue
            folder = candidate / chunk_config
            build_path = folder / "build.json"
            identity_path = folder / "identity.json"
            if folder.is_symlink():
                raise RuntimeError(f"ambiguous legacy collection {name!r}: symlinked sidecars")
            build_present = self._sidecar_exists(build_path, name)
            identity_present = self._sidecar_exists(identity_path, name)
            if not build_present and not identity_present:
                continue
            if not build_present or not identity_present:
                raise RuntimeError(f"ambiguous legacy collection {name!r}: incomplete sidecars")
            try:
                build = json.loads(build_path.read_text(encoding="utf-8"))
                identity = json.loads(identity_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"ambiguous legacy collection {name!r}: unreadable sidecars"
                ) from exc
            if not isinstance(build, dict) or not isinstance(identity, dict):
                raise RuntimeError(f"ambiguous legacy collection {name!r}: invalid sidecars")
            for record in (build, identity):
                if record.get("snapshot_id") != snapshot_id or record.get("label") != chunk_config:
                    raise RuntimeError(
                        f"ambiguous legacy collection {name!r}: sidecar identity mismatch"
                    )
            config = build.get("config")
            expected_label = "A-prime" if config == "A'" else config
            encoder = build.get("identity")
            if (
                not isinstance(config, str)
                or expected_label != chunk_config
                or config != identity.get("config")
                or build.get("collection") != identity.get("collection")
                or not isinstance(encoder, dict)
                or encoder != identity.get("identity")
                or not isinstance(build.get("stored"), int)
                or isinstance(build.get("stored"), bool)
                or build["stored"] <= 0
                or not isinstance(build.get("chunks"), int)
                or isinstance(build.get("chunks"), bool)
                or build["chunks"] <= 0
            ):
                raise RuntimeError(f"ambiguous legacy collection {name!r}: inconsistent sidecars")
            for field in ("model", "backend", "device", "precision", "identity_slug"):
                if not isinstance(encoder.get(field), str) or not encoder[field]:
                    raise RuntimeError(
                        f"ambiguous legacy collection {name!r}: invalid encoder identity"
                    )
            batch_size = encoder.get("batch_size")
            if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
                raise RuntimeError(
                    f"ambiguous legacy collection {name!r}: invalid encoder identity"
                )
            recorded_name = build["collection"]
            if recorded_name not in {name, collection_name(snapshot_id, chunk_config)}:
                raise RuntimeError(
                    f"ambiguous legacy collection {name!r}: unexpected sidecar collection"
                )
            try:
                stamp = datetime.fromisoformat(build["timestamp"])
                if stamp.tzinfo is None or stamp.utcoffset() is None:
                    raise ValueError("timestamp has no timezone")
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"ambiguous legacy collection {name!r}: invalid build timestamp"
                ) from exc
            if recorded_name == name:
                if build["stored"] != count:
                    raise RuntimeError(
                        f"ambiguous legacy collection {name!r}: sidecar count mismatch"
                    )
                owners.add(snapshot_id)
        return owners

    def _legacy_owner(self, name: str, chunk_config: str, collection: Any) -> str:
        """Establish one full owner from metadata, every row and any matching sidecars."""
        count = int(collection.count())
        owners = self._legacy_sidecar_owners(name, chunk_config, count)
        metadata = dict(collection.metadata or {})
        recorded_config = metadata.get(_CHUNK_CONFIG_KEY)
        if recorded_config is not None and recorded_config != chunk_config:
            raise RuntimeError(f"ambiguous legacy collection {name!r}: config mismatch")
        recorded_owner = metadata.get(_SNAPSHOT_ID_KEY)
        if recorded_owner is not None:
            if not isinstance(recorded_owner, str):
                raise RuntimeError(f"ambiguous legacy collection {name!r}: invalid owner")
            owners.add(recorded_owner)

        offset = 0
        while offset < count:
            page = collection.get(limit=_LEGACY_PAGE_SIZE, offset=offset, include=["metadatas"])
            ids = _field(page, "ids") or []
            metadatas = _field(page, "metadatas") or []
            if not ids or len(ids) != len(metadatas):
                raise RuntimeError(f"ambiguous legacy collection {name!r}: incomplete row metadata")
            for row in metadatas:
                if not isinstance(row, dict) or not isinstance(row.get(_SNAPSHOT_ID_KEY), str):
                    raise RuntimeError(
                        f"ambiguous legacy collection {name!r}: row has no full owner"
                    )
                owners.add(row[_SNAPSHOT_ID_KEY])
            offset += len(ids)
        if offset != count or int(collection.count()) != count:
            raise RuntimeError(
                f"ambiguous legacy collection {name!r}: rows changed during inspection"
            )
        if len(owners) != 1:
            raise RuntimeError(f"ambiguous legacy collection {name!r}: no unique full owner")
        owner = next(iter(owners))
        try:
            collection_name(owner, chunk_config)
        except ValueError as exc:
            raise RuntimeError(f"ambiguous legacy collection {name!r}: invalid full owner") from exc
        if _legacy_collection_name(owner, chunk_config) != name:
            raise RuntimeError(f"ambiguous legacy collection {name!r}: owner does not match name")
        return owner

    @staticmethod
    def _check_full_owner(collection: Any, snapshot_id: str, chunk_config: str) -> None:
        metadata = dict(collection.metadata or {})
        if (
            metadata.get(_SNAPSHOT_ID_KEY) != snapshot_id
            or metadata.get(_CHUNK_CONFIG_KEY) != chunk_config
            or metadata.get(_NAMING_VERSION_KEY) != _NAMING_VERSION
        ):
            raise RuntimeError(f"full-id collection {collection.name!r} has no matching owner")

    def _resolve_collection(self, snapshot_id: str, chunk_config: str) -> Any | None:
        full_name = collection_name(snapshot_id, chunk_config)
        legacy_name = _legacy_collection_name(snapshot_id, chunk_config)
        collections = self._collections()
        full = collections.get(full_name)
        legacy = collections.get(legacy_name)
        if full is not None:
            self._check_full_owner(full, snapshot_id, chunk_config)
        if legacy is not None:
            owner = self._legacy_owner(legacy_name, chunk_config, legacy)
            if owner == snapshot_id:
                if full is not None:
                    raise RuntimeError(
                        f"ambiguous snapshot {snapshot_id!r}: both full and legacy collections exist"
                    )
                return legacy
        return full

    def bind(self, snapshot_id: str, chunk_config: str) -> Any:
        """Select one full snapshot/config pair, rejecting ambiguous legacy collections."""
        name = collection_name(snapshot_id, chunk_config)
        collection = self._resolve_collection(snapshot_id, chunk_config)
        if collection is None:
            configuration = {"hnsw": {"space": "cosine"}}
            metadata = {
                _CREATED_TS_KEY: int(time.time()),
                _SNAPSHOT_ID_KEY: snapshot_id,
                _CHUNK_CONFIG_KEY: chunk_config,
                _NAMING_VERSION_KEY: _NAMING_VERSION,
            }
            try:
                collection = self.client.create_collection(
                    name=name, configuration=configuration, metadata=metadata
                )
            except (TypeError, ValueError):
                # Some Chroma 1.x minors reject configuration and metadata together.
                collection = self.client.create_collection(name=name, configuration=configuration)
                collection.modify(metadata=metadata)
            self._check_full_owner(collection, snapshot_id, chunk_config)
        self._collection = collection
        self._bound_snapshot_id = snapshot_id
        return collection

    def bind_existing(self, snapshot_id: str, chunk_config: str) -> Any:
        """Bind a verified existing collection without creating one.

        Missing physical storage raises ``LookupError``; the caller must consult the catalog
        to distinguish unknown from retired snapshots. On failure the previous binding remains.
        This does not enforce read-only access or protect against concurrent deletion.
        """
        name = collection_name(snapshot_id, chunk_config)
        if self._client is None:
            # PersistentClient initializes an empty database if its storage file is missing.
            try:
                mode = (self.root / _PERSISTENT_DB).stat().st_mode
            except FileNotFoundError as exc:
                raise LookupError(f"no existing snapshot collection {name!r}") from exc
            except OSError as exc:
                raise RuntimeError(f"cannot inspect the snapshot database at {self.root}") from exc
            if not S_ISREG(mode):
                raise RuntimeError(f"invalid snapshot database at {self.root}")
        collection = self._resolve_collection(snapshot_id, chunk_config)
        if collection is None:
            raise LookupError(f"no existing snapshot collection {name!r}")
        self._collection = collection
        self._bound_snapshot_id = snapshot_id
        return collection

    def validate_binding(self, snapshot_id: str, chunk_config: str) -> None:
        """Check ownership before a build performs cache, collection or row writes."""
        self._resolve_collection(snapshot_id, chunk_config)

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
        if any(payload.snapshot_id != self._bound_snapshot_id for payload in payloads):
            raise ValueError("payload snapshot_id must match the bound snapshot")
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
        """Delete only the verified snapshot/config pair, preserving any other binding."""
        collection = self._resolve_collection(snapshot_id, chunk_config)
        if collection is None:
            return False
        name = collection.name
        self.client.delete_collection(name)
        if getattr(self._collection, "name", None) == name:
            self._collection = None
            self._bound_snapshot_id = None
        return True

    def prune(
        self,
        keep: int | None = None,
        *,
        publication_times: Mapping[str, datetime] | None = None,
    ) -> list[str]:
        """Prune only with a complete, unambiguous publication order from a validated catalog.

        The caller must keep active pins out of the deletion set; lease coordination is W2 work.
        """
        keep_n = self.settings.keep_snapshots if keep is None else keep
        if keep_n < 0:
            raise ValueError("keep must be >= 0")
        groups: dict[str, list[str]] = {}
        pairs: set[tuple[str, str]] = set()
        for name, collection in self._collections().items():
            if not name.startswith(_SNAP_PREFIX):
                continue
            is_full, snapshot_or_prefix, chunk_config = _parse_collection_name(name)
            if is_full:
                snapshot_id = snapshot_or_prefix
                self._check_full_owner(collection, snapshot_id, chunk_config)
            else:
                snapshot_id = self._legacy_owner(name, chunk_config, collection)
            pair = (snapshot_id, chunk_config)
            if pair in pairs:
                raise RuntimeError(f"ambiguous snapshot {snapshot_id!r}: duplicate collections")
            pairs.add(pair)
            groups.setdefault(snapshot_id, []).append(name)
        if len(groups) <= keep_n:
            return []
        if publication_times is None:
            raise RuntimeError("publication times from a validated catalog are required to prune")
        times: dict[str, datetime] = {}
        for snapshot_id in groups:
            value = publication_times.get(snapshot_id)
            if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
                raise RuntimeError(f"missing or invalid publication time for {snapshot_id!r}")
            times[snapshot_id] = value.astimezone(UTC)
        if len(set(times.values())) != len(times):
            raise RuntimeError("publication times do not establish an unambiguous order")
        ordered = sorted(groups, key=times.__getitem__)
        doomed = [name for key in ordered[: len(groups) - keep_n] for name in groups[key]]
        for name in doomed:
            self.client.delete_collection(name)
        if getattr(self._collection, "name", None) in doomed:
            self._collection = None
            self._bound_snapshot_id = None
        return sorted(doomed)


__all__ = ["ChromaStore", "ScoredPayload", "collection_name", "media_filter"]
