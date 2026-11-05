"""Canonical snapshot manifest minting, atomic staging/publication, idempotency verification,
and snapshot reader functions for WP2 (`build_index.py`, `extract_facts.py`, `strategy_b.py`)
and WP3 (`run_eval.py`).

Implements `docs/INGEST_CONTRACT.md` §3-§4:
  - `manifest_sha256 = sha256(canonical_json(manifest)).hexdigest()`
  - `snapshot_id = window_end.replace("-", "") + "-" + manifest_sha256[:12]`
  - Rejects same-short-ID / different-full-digest collisions before writing or overwriting.
  - Stages files in a temporary directory and atomically publishes `catalog.json` (only
    flipping the serving pointer `CURRENT` when `set_current=True`).

Owner: A (WP1) | Spec: docs/INGEST_CONTRACT.md §3-§4, Issue #9, Issue #23, Issue #30, Issue #31
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from acgn_rag.core.ingest.dedup import DedupCluster, DedupPolicy, DedupResult, cluster_items
from acgn_rag.core.ingest.facts_link import (
    EntityLinker,
    EntityRegistry,
    SourceRegistry,
    build_default_source_registry,
    build_entity_registry_from_raw,
)
from acgn_rag.core.ingest.normalize import (
    NORMALIZATION_VERSION,
    NormalizedItem,
    QuarantinedItem,
    RawIngestItem,
    canonical_json,
    classify_content_category,
    compute_doc_id,
    normalize_batch,
    normalize_url,
)
from acgn_rag.core.ports import CorpusSnapshot

MANIFEST_VERSION = "1"
_DATE_STR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class SnapshotError(RuntimeError):
    """Base error for snapshot minting, validation, or publication failures."""


class SnapshotCollisionError(SnapshotError):
    """Raised when a short 12-hex snapshot_id collides with a different full manifest SHA-256."""


def _validate_window_str(value: date | datetime | str, *, name: str) -> str:
    if isinstance(value, datetime):
        s = value.date().isoformat()
    elif isinstance(value, date):
        s = value.isoformat()
    else:
        s = str(value).strip()
    if not _DATE_STR_RE.match(s):
        raise ValueError(f"{name} must be a UTC calendar date YYYY-MM-DD, got {value!r}")
    date.fromisoformat(s)
    return s


def build_manifest_dict(
    items: Sequence[NormalizedItem],
    *,
    window_start: date | str,
    window_end: date | str,
    entity_registry_sha256: str,
    source_registry_sha256: str,
    dedup_policy: DedupPolicy | Mapping[str, Any],
    manifest_version: str = MANIFEST_VERSION,
    normalization_version: str = NORMALIZATION_VERSION,
) -> dict[str, Any]:
    """Build the canonical manifest v1 dictionary (docs/INGEST_CONTRACT.md §3)."""
    w_start = _validate_window_str(window_start, name="window_start")
    w_end = _validate_window_str(window_end, name="window_end")
    if w_start > w_end:
        raise ValueError(f"window_start ({w_start}) must not exceed window_end ({w_end})")
    if not items:
        raise SnapshotError("Cannot build a snapshot manifest from 0 items")

    sorted_items = sorted(items, key=lambda it: it.doc_id)
    seen_doc_ids: set[str] = set()
    manifest_items: list[dict[str, Any]] = []

    for item in sorted_items:
        if item.doc_id in seen_doc_ids:
            raise SnapshotError(f"Duplicate doc_id in manifest items: {item.doc_id}")
        seen_doc_ids.add(item.doc_id)
        pub_day = item.published_at[:10]
        if pub_day < w_start or pub_day > w_end:
            raise SnapshotError(
                f"Item {item.doc_id} published_at ({item.published_at}) is outside frozen window "
                f"{w_start}..{w_end}"
            )
        manifest_items.append(item.to_manifest_item())

    policy_dict = (
        dedup_policy.to_dict() if isinstance(dedup_policy, DedupPolicy) else dict(dedup_policy)
    )

    return {
        "dedup_policy": policy_dict,
        "entity_registry_sha256": str(entity_registry_sha256).lower(),
        "items": manifest_items,
        "manifest_version": str(manifest_version),
        "normalization_version": str(normalization_version),
        "source_registry_sha256": str(source_registry_sha256).lower(),
        "window_end": w_end,
        "window_start": w_start,
    }


def compute_snapshot_id_from_manifest(
    manifest: Mapping[str, Any],
) -> tuple[str, str, bytes]:
    """Return `(snapshot_id, manifest_sha256, canonical_manifest_bytes)` for a manifest object.

    Formula (INGEST_CONTRACT.md §3):
      `manifest_sha256 = sha256(canonical_json(manifest)).hexdigest()`
      `snapshot_id = window_end.replace("-", "") + "-" + manifest_sha256[:12]`
    """
    w_end = _validate_window_str(str(manifest["window_end"]), name="window_end")
    items = list(manifest.get("items") or [])
    sorted_items = sorted(items, key=lambda row: str(row["doc_id"]))
    normalized_manifest = dict(manifest)
    normalized_manifest["items"] = sorted_items

    manifest_bytes = canonical_json(normalized_manifest)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    date_prefix = w_end.replace("-", "")
    snapshot_id = f"{date_prefix}-{manifest_sha256[:12]}"
    return snapshot_id, manifest_sha256, manifest_bytes


@dataclass(frozen=True)
class BuiltSnapshot:
    """In-memory and on-disk representation of a minted frozen snapshot."""

    snapshot_id: str
    manifest_sha256: str
    manifest_bytes: bytes
    manifest: dict[str, Any]
    corpus_snapshot: CorpusSnapshot
    items: list[NormalizedItem]
    clusters: list[DedupCluster]
    entity_registry: EntityRegistry
    source_registry: SourceRegistry
    snapshot_dir: Path | None = None
    quarantined: tuple[QuarantinedItem, ...] = ()
    ingest_report: dict[str, Any] = field(default_factory=dict)

    @property
    def representatives(self) -> list[NormalizedItem]:
        rep_ids = {c.representative_doc_id for c in self.clusters}
        return [it for it in self.items if it.doc_id in rep_ids]


def build_snapshot_from_items(
    items: Sequence[NormalizedItem],
    *,
    window_start: date | str,
    window_end: date | str,
    entity_registry: EntityRegistry | None = None,
    source_registry: SourceRegistry | None = None,
    dedup_policy: DedupPolicy | None = None,
    created_at: str | None = None,
    quarantined: Sequence[QuarantinedItem] = (),
    ingest_report: Mapping[str, Any] | None = None,
    external_refs_by_doc_id: Mapping[str, Sequence[tuple[str, str]]] | None = None,
    crawl_summary: Sequence[Mapping[str, Any]] | None = None,
    partial: bool | None = None,
) -> BuiltSnapshot:
    """Link entities, cluster items, and mint the canonical manifest and `CorpusSnapshot`."""
    ent_reg = entity_registry if entity_registry is not None else EntityRegistry()
    pol = dedup_policy if dedup_policy is not None else DedupPolicy()

    linker = EntityLinker(ent_reg)
    linked_items, link_stats = linker.link_normalized_items(
        items,
        external_refs_by_doc_id=external_refs_by_doc_id,
    )
    dedup_res: DedupResult = cluster_items(linked_items, policy=pol)

    distinct_sources = {it.source_key for it in dedup_res.items if it.source_key}
    if source_registry is not None:
        src_reg = source_registry
    else:
        src_reg = build_default_source_registry().filter_used(distinct_sources)

    manifest_dict = build_manifest_dict(
        dedup_res.items,
        window_start=window_start,
        window_end=window_end,
        entity_registry_sha256=ent_reg.sha256,
        source_registry_sha256=src_reg.sha256,
        dedup_policy=dedup_res.policy,
    )
    snapshot_id, manifest_sha256, manifest_bytes = compute_snapshot_id_from_manifest(manifest_dict)

    op_created_at = (
        created_at
        if created_at is not None
        else datetime.now(tz=UTC).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    corpus_snap = CorpusSnapshot(
        snapshot_id=snapshot_id,
        created_at=op_created_at,
        item_count=len(dedup_res.items),
        deduped_count=len(dedup_res.clusters),
        source_count=len(distinct_sources),
        window_start=str(manifest_dict["window_start"]),
        window_end=str(manifest_dict["window_end"]),
    )

    report_dict: dict[str, Any] = dict(ingest_report) if ingest_report is not None else {}
    if crawl_summary is not None:
        report_dict["crawl_summary"] = [dict(row) for row in crawl_summary]
    if partial is not None:
        report_dict["partial"] = bool(partial)
    report_dict["linking_stats"] = link_stats.to_dict()
    report_dict.setdefault("quarantined_count", len(quarantined))

    return BuiltSnapshot(
        snapshot_id=snapshot_id,
        manifest_sha256=manifest_sha256,
        manifest_bytes=manifest_bytes,
        manifest=manifest_dict,
        corpus_snapshot=corpus_snap,
        items=dedup_res.items,
        clusters=dedup_res.clusters,
        entity_registry=ent_reg,
        source_registry=src_reg,
        snapshot_dir=None,
        quarantined=tuple(quarantined),
        ingest_report=report_dict,
    )


def build_snapshot_from_raw(
    raw_items: Sequence[RawIngestItem],
    *,
    window_start: date | str,
    window_end: date | str,
    entity_registry: EntityRegistry | None = None,
    source_registry: SourceRegistry | None = None,
    dedup_policy: DedupPolicy | None = None,
    created_at: str | None = None,
    persist_entity_registry_path: Path | None = None,
    crawl_summary: Sequence[Mapping[str, Any]] | None = None,
    partial: bool = False,
) -> BuiltSnapshot:
    """End-to-end deterministic builder: `RawIngestItem`s -> normalize -> link -> dedup -> manifest."""
    w_start = _validate_window_str(window_start, name="window_start")
    w_end = _validate_window_str(window_end, name="window_end")

    ent_reg = (
        entity_registry
        if entity_registry is not None
        else build_entity_registry_from_raw(
            raw_items,
            persist_path=persist_entity_registry_path,
        )
    )

    ext_refs_by_doc: dict[str, list[tuple[str, str]]] = {}
    for raw in raw_items:
        if raw.external_media_refs:
            try:
                did = compute_doc_id(raw.url)
                ext_refs_by_doc.setdefault(did, []).extend(raw.external_media_refs)
            except Exception:
                continue

    norm_report = normalize_batch(
        raw_items,
        window_start=w_start,
        window_end=w_end,
        quarantine_on_error=True,
    )
    if not norm_report.items:
        raise SnapshotError(
            f"0 normalized items survived in window {w_start}..{w_end} "
            f"(quarantined={len(norm_report.quarantined)}, "
            f"out_of_window={norm_report.out_of_window_count})"
        )

    quarantined_by_source = dict(
        sorted(Counter(q.source_key or "unknown" for q in norm_report.quarantined).items())
    )
    ingest_report: dict[str, Any] = {
        "exact_duplicates_collapsed": norm_report.exact_duplicates_collapsed,
        "out_of_window_count": norm_report.out_of_window_count,
        "partial": bool(partial),
        "quarantined_by_source": quarantined_by_source,
        "quarantined_count": len(norm_report.quarantined),
        "raw_input_count": len(raw_items),
    }
    if crawl_summary is not None:
        ingest_report["crawl_summary"] = [dict(row) for row in crawl_summary]

    return build_snapshot_from_items(
        norm_report.items,
        window_start=w_start,
        window_end=w_end,
        entity_registry=ent_reg,
        source_registry=source_registry,
        dedup_policy=dedup_policy,
        created_at=created_at,
        quarantined=norm_report.quarantined,
        ingest_report=ingest_report,
        external_refs_by_doc_id=ext_refs_by_doc,
    )


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def publish_snapshot(
    built: BuiltSnapshot,
    *,
    snapshots_root: Path = Path("data/snapshots"),
    set_current: bool = False,
    keep_snapshots: int = 2,
) -> Path:
    """Publish a `BuiltSnapshot` to `data/snapshots/<snapshot_id>/` via atomic staging.

    - Updates `catalog.json` on every publication.
    - Only flips the serving pointer `CURRENT` when `set_current=True` (`INGEST_CONTRACT.md` §4:
      `CURRENT` is the final serving switch after the A/C/F completion protocol).
    - Rejects short-ID collisions (`SnapshotCollisionError`) before touching existing files.
    """
    snapshots_root.mkdir(parents=True, exist_ok=True)
    target_dir = snapshots_root / built.snapshot_id

    if target_dir.exists():
        existing_manifest_path = target_dir / "manifest.json"
        if existing_manifest_path.exists():
            existing_bytes = existing_manifest_path.read_bytes()
            existing_sha256 = hashlib.sha256(existing_bytes).hexdigest()
            if existing_sha256 != built.manifest_sha256:
                raise SnapshotCollisionError(
                    f"Snapshot ID collision on {built.snapshot_id!r}: existing manifest_sha256="
                    f"{existing_sha256} != new manifest_sha256={built.manifest_sha256}"
                )
            # Identical manifest bytes: idempotent rebuild of the exact same snapshot
            _update_current_and_catalog(
                snapshots_root,
                built.snapshot_id,
                published_at=built.corpus_snapshot.created_at,
                keep_snapshots=keep_snapshots,
                set_current=set_current,
            )
            return target_dir

    staging_dir = snapshots_root / f".staging_{built.snapshot_id}_{os.getpid()}"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. items.jsonl (all members sorted by doc_id)
        items_lines = b"".join(canonical_json(item.to_dict()) for item in built.items)
        (staging_dir / "items.jsonl").write_bytes(items_lines)

        # 2. representatives.jsonl (one representative per cluster sorted by doc_id, for Config A/A'/A0)
        rep_lines = b"".join(canonical_json(item.to_dict()) for item in built.representatives)
        (staging_dir / "representatives.jsonl").write_bytes(rep_lines)

        # 3. clusters.jsonl (sorted by representative_doc_id)
        cluster_lines = b"".join(canonical_json(cluster.to_dict()) for cluster in built.clusters)
        (staging_dir / "clusters.jsonl").write_bytes(cluster_lines)

        # 4. quarantine.jsonl (malformed / rejected rows with source and reason)
        quarantine_lines = b"".join(
            canonical_json(
                {
                    "reason": q.reason,
                    "source_key": q.source_key,
                    "url": q.url,
                }
            )
            for q in built.quarantined
        )
        (staging_dir / "quarantine.jsonl").write_bytes(quarantine_lines)

        # 5. entities.json and sources.json
        (staging_dir / "entities.json").write_bytes(built.entity_registry.to_canonical_bytes())
        (staging_dir / "sources.json").write_bytes(built.source_registry.to_canonical_bytes())

        # 6. manifest.json (exact canonical bytes)
        (staging_dir / "manifest.json").write_bytes(built.manifest_bytes)

        # 7. snapshot.json (CorpusSnapshot wire metadata + provenance + ingest_report)
        source_type_counts = {"official": 0, "media": 0, "metadata": 0}
        category_counts = {"anime": 0, "manga": 0, "game": 0, "light_novel": 0}
        corporate_count = 0
        for item in built.items:
            source_type_counts[item.source_type] = source_type_counts.get(item.source_type, 0) + 1
            cat = classify_content_category(item.title, item.body, source_key=item.source_key)
            category_counts[cat] = category_counts.get(cat, 0) + 1
            if item.ingest_category == "corporate":
                corporate_count += 1

        pub_dates = sorted(it.published_at[:10] for it in built.items)
        cluster_rep_map = {c.cluster_id: c.representative_doc_id for c in built.clusters}
        url_to_rep = {
            item.url: cluster_rep_map[item.cluster_id]
            for item in built.items
            if item.cluster_id in cluster_rep_map
        }

        snapshot_meta = built.corpus_snapshot.model_dump(mode="json") | {
            "category_counts": category_counts,
            "corporate_count": corporate_count,
            "ingest_report": dict(built.ingest_report),
            "manifest_sha256": built.manifest_sha256,
            "observed_earliest": pub_dates[0] if pub_dates else "",
            "observed_latest": pub_dates[-1] if pub_dates else "",
            "partial": bool(built.ingest_report.get("partial", False)),
            "source_type_counts": source_type_counts,
            "url_to_representative": url_to_rep,
        }
        (staging_dir / "snapshot.json").write_bytes(canonical_json(snapshot_meta))

        if target_dir.exists():
            shutil.rmtree(target_dir)
        os.replace(staging_dir, target_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)

    _update_current_and_catalog(
        snapshots_root,
        built.snapshot_id,
        published_at=built.corpus_snapshot.created_at,
        keep_snapshots=keep_snapshots,
        set_current=set_current,
    )
    return target_dir


def _update_current_and_catalog(
    snapshots_root: Path,
    snapshot_id: str,
    *,
    published_at: str,
    keep_snapshots: int = 2,
    set_current: bool = True,
) -> None:
    """Atomically update `catalog.json` ordered by publication time, and flip `CURRENT` when
    `set_current=True`.

    Enforces `docs/INGEST_CONTRACT.md` §4:
      - `retained` and `retired` are strictly disjoint (re-publishing an older snapshot removes
        it from `retired`).
      - A corrupt `catalog.json` raises `SnapshotError` rather than silently wiping retirement
        history.
    """
    catalog_path = snapshots_root / "catalog.json"
    catalog: dict[str, Any] = {
        "current": None,
        "latest_corpus": snapshot_id,
        "publications": [],
        "retained": [],
        "retired": [],
    }
    if catalog_path.exists():
        try:
            loaded = json.loads(catalog_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise SnapshotError(f"Corrupt catalog.json at {catalog_path}: {exc}") from exc
        if not isinstance(loaded, dict):
            raise SnapshotError(
                f"Corrupt catalog.json at {catalog_path}: root is not a JSON object"
            )
        catalog.update(loaded)

    raw_pubs: list[dict[str, str]] = [
        p for p in (catalog.get("publications") or []) if isinstance(p, dict)
    ]
    existing_pub = next((p for p in raw_pubs if p.get("snapshot_id") == snapshot_id), None)
    eff_published_at = (
        str(existing_pub["published_at"])
        if existing_pub is not None and existing_pub.get("published_at")
        else published_at
    )
    keep_n = max(1, keep_snapshots)
    currently_retained_ids = {str(p.get("snapshot_id")) for p in raw_pubs[:keep_n]}

    if existing_pub is not None and snapshot_id in currently_retained_ids and not set_current:
        pubs = list(raw_pubs)
    else:
        pubs = [p for p in raw_pubs if p.get("snapshot_id") != snapshot_id]
        pubs.insert(0, {"published_at": eff_published_at, "snapshot_id": snapshot_id})

    retained = [p["snapshot_id"] for p in pubs[:keep_n]]
    retained_set = set(retained)

    # Remove any retained snapshot_id (e.g. a re-published older snapshot) from `retired`
    retired_list: list[str] = [
        str(sid)
        for sid in (catalog.get("retired") or [])
        if str(sid) and str(sid) not in retained_set
    ]
    for p in pubs[keep_n:]:
        sid = str(p["snapshot_id"])
        if sid not in retained_set and sid not in retired_list:
            retired_list.append(sid)

    catalog["latest_corpus"] = pubs[0]["snapshot_id"] if pubs else snapshot_id
    if set_current:
        catalog["current"] = snapshot_id
    catalog["publications"] = pubs
    catalog["retained"] = retained
    catalog["retired"] = retired_list

    _atomic_write_bytes(catalog_path, canonical_json(catalog))
    if set_current:
        _atomic_write_bytes(snapshots_root / "CURRENT", f"{snapshot_id}\n".encode())


def resolve_snapshot_dir(
    snapshot_dir_or_id: Path | str,
    *,
    data_root: Path = Path("data"),
) -> Path:
    """Resolve a snapshot directory Path or full `snapshot_id` under `<data_root>/snapshots/`."""
    candidate = Path(snapshot_dir_or_id)
    if candidate.is_dir() and (candidate / "manifest.json").exists():
        return candidate
    by_id = data_root / "snapshots" / str(snapshot_dir_or_id).strip()
    if by_id.is_dir() and (by_id / "manifest.json").exists():
        return by_id
    raise FileNotFoundError(
        f"Snapshot {snapshot_dir_or_id!r} not found under {data_root / 'snapshots'}"
    )


def verify_snapshot_integrity(snap_dir: Path) -> str:
    """Verify `manifest.json` SHA-256 against the directory name / `snapshot.json` and return
    `manifest_sha256`. Raises `SnapshotError` if tampered.
    """
    manifest_path = snap_dir / "manifest.json"
    if not manifest_path.exists():
        raise SnapshotError(f"Missing manifest.json in {snap_dir}")
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    dir_name = snap_dir.name
    if "-" in dir_name and len(dir_name.split("-", 1)[1]) == 12:
        expected_prefix = dir_name.split("-", 1)[1]
        if not manifest_sha.startswith(expected_prefix):
            raise SnapshotError(
                f"Snapshot manifest digest mismatch in {snap_dir}: "
                f"dir prefix={expected_prefix!r} vs sha256={manifest_sha[:12]!r}"
            )
    snap_json_path = snap_dir / "snapshot.json"
    if snap_json_path.exists():
        meta = json.loads(snap_json_path.read_text(encoding="utf-8"))
        recorded_sha = meta.get("manifest_sha256")
        if recorded_sha and recorded_sha != manifest_sha:
            raise SnapshotError(
                f"Snapshot metadata manifest_sha256 mismatch in {snap_dir}: "
                f"{recorded_sha} != {manifest_sha}"
            )
    return manifest_sha


def iter_snapshot_items(
    snapshot_dir_or_id: Path | str,
    *,
    data_root: Path = Path("data"),
    representatives_only: bool = False,
    verify: bool = True,
) -> Iterator[NormalizedItem]:
    """Iterate `NormalizedItem` rows from a frozen snapshot directory.

    - `representatives_only=True`: yields one representative per cluster (`representatives.jsonl`),
      used by ordinary single-article chunk configs (`A`, `A'`, `A0`).
    - `representatives_only=False`: yields all cluster member rows (`items.jsonl`), used by
      Config `B` event-cluster aggregation and gold URL resolution.
    """
    snap_dir = resolve_snapshot_dir(snapshot_dir_or_id, data_root=data_root)
    manifest_by_doc_id: dict[str, dict[str, Any]] = {}
    if verify:
        verify_snapshot_integrity(snap_dir)
        manifest_data = json.loads((snap_dir / "manifest.json").read_text(encoding="utf-8"))
        manifest_by_doc_id = {
            str(m["doc_id"]): m for m in manifest_data.get("items", []) if isinstance(m, dict)
        }
    filename = "representatives.jsonl" if representatives_only else "items.jsonl"
    jsonl_path = snap_dir / filename
    if not jsonl_path.exists() and representatives_only:
        clusters = load_snapshot_clusters(snap_dir, data_root=data_root, verify=False)
        rep_ids = {c.representative_doc_id for c in clusters}
        for item in iter_snapshot_items(
            snap_dir, data_root=data_root, representatives_only=False, verify=verify
        ):
            if item.doc_id in rep_ids:
                yield item
        return

    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            raw_row = json.loads(stripped)
            item = NormalizedItem.from_dict(raw_row)
            if verify:
                raw_doc_id = raw_row.get("doc_id")
                if raw_doc_id is not None and raw_doc_id != item.doc_id:
                    raise SnapshotError(
                        f"Tampered doc_id in {jsonl_path}: {raw_doc_id!r} != {item.doc_id!r}"
                    )
                raw_body_sha = raw_row.get("body_sha256")
                if raw_body_sha is not None and raw_body_sha != item.body_sha256:
                    raise SnapshotError(
                        f"Tampered body_sha256 for {item.doc_id} in {jsonl_path}: "
                        f"{raw_body_sha!r} != {item.body_sha256!r}"
                    )
                manifest_row = manifest_by_doc_id.get(item.doc_id)
                if manifest_row is None:
                    raise SnapshotError(
                        f"Tampered item {item.doc_id} in {jsonl_path}: missing from verified manifest.json"
                    )
                if manifest_row != item.to_manifest_item():
                    raise SnapshotError(
                        f"Tampered item {item.doc_id} in {jsonl_path}: row disagrees with verified manifest.json"
                    )
            yield item


def load_snapshot_items(
    snapshot_dir_or_id: Path | str,
    *,
    data_root: Path = Path("data"),
    representatives_only: bool = False,
    verify: bool = True,
) -> list[NormalizedItem]:
    """Load all (or representative-only) `NormalizedItem` records from a frozen snapshot."""
    return list(
        iter_snapshot_items(
            snapshot_dir_or_id,
            data_root=data_root,
            representatives_only=representatives_only,
            verify=verify,
        )
    )


def load_snapshot_clusters(
    snapshot_dir_or_id: Path | str,
    *,
    data_root: Path = Path("data"),
    verify: bool = True,
) -> list[DedupCluster]:
    """Load `DedupCluster` records from `clusters.jsonl` in a frozen snapshot."""
    snap_dir = resolve_snapshot_dir(snapshot_dir_or_id, data_root=data_root)
    if verify:
        verify_snapshot_integrity(snap_dir)
    clusters_path = snap_dir / "clusters.jsonl"
    clusters: list[DedupCluster] = []
    with clusters_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            clusters.append(DedupCluster.from_dict(json.loads(stripped)))
    return clusters


def load_corpus_snapshot(
    snapshot_dir_or_id: Path | str,
    *,
    data_root: Path = Path("data"),
    verify: bool = True,
) -> CorpusSnapshot:
    """Load the wire-compatible `CorpusSnapshot` model from `snapshot.json`."""
    snap_dir = resolve_snapshot_dir(snapshot_dir_or_id, data_root=data_root)
    if verify:
        verify_snapshot_integrity(snap_dir)
    raw = json.loads((snap_dir / "snapshot.json").read_text(encoding="utf-8"))
    return CorpusSnapshot.model_validate(
        {
            "snapshot_id": raw["snapshot_id"],
            "created_at": raw["created_at"],
            "item_count": raw["item_count"],
            "deduped_count": raw["deduped_count"],
            "source_count": raw["source_count"],
            "window_start": raw["window_start"],
            "window_end": raw["window_end"],
        }
    )


def resolve_url_in_snapshot(
    snapshot_dir_or_id: Path | str,
    url: str,
    *,
    data_root: Path = Path("data"),
) -> str | None:
    """Resolve any member URL (including syndicated non-representative copies) to its cluster's
    `representative_doc_id` (INGEST_CONTRACT.md §3).
    """
    snap_dir = resolve_snapshot_dir(snapshot_dir_or_id, data_root=data_root)
    canonical = normalize_url(url)
    meta_path = snap_dir / "snapshot.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        mapping = meta.get("url_to_representative")
        if isinstance(mapping, dict) and canonical in mapping:
            return str(mapping[canonical])

    clusters = {
        c.cluster_id: c.representative_doc_id
        for c in load_snapshot_clusters(snap_dir, data_root=data_root)
    }
    for item in iter_snapshot_items(snap_dir, data_root=data_root, representatives_only=False):
        if item.url == canonical and item.cluster_id in clusters:
            return clusters[item.cluster_id]
    return None


__all__ = [
    "MANIFEST_VERSION",
    "BuiltSnapshot",
    "SnapshotCollisionError",
    "SnapshotError",
    "build_manifest_dict",
    "build_snapshot_from_items",
    "build_snapshot_from_raw",
    "compute_snapshot_id_from_manifest",
    "iter_snapshot_items",
    "load_corpus_snapshot",
    "load_snapshot_clusters",
    "load_snapshot_items",
    "publish_snapshot",
    "resolve_snapshot_dir",
    "resolve_url_in_snapshot",
    "verify_snapshot_integrity",
]
