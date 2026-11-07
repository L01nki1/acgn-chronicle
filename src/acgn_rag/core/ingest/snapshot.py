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
REQUIRED_SNAPSHOT_FILES: tuple[str, ...] = (
    "manifest.json",
    "snapshot.json",
    "items.jsonl",
    "representatives.jsonl",
    "clusters.jsonl",
    "entities.json",
    "sources.json",
    "quarantine.jsonl",
)


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
        rows_list = [dict(row) for row in crawl_summary]
        ingest_report["crawl_summary"] = rows_list
        gaps = [
            f"{row.get('source_key')}:{row.get('coverage_mode')}"
            for row in rows_list
            if row.get("coverage_mode") and row.get("coverage_mode") != "full_window_archive"
        ]
        ingest_report["coverage_gaps"] = gaps
        ingest_report["coverage_complete"] = len(gaps) == 0 and not bool(partial)

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


def _parse_catalog_published_at(value: Any, *, context: str = "published_at") -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise SnapshotError(f"Invalid or missing {context}: {value!r}")
    raw = value.strip()
    iso_str = raw[:-1] + "+00:00" if raw.endswith(("Z", "z")) else raw
    try:
        dt = datetime.fromisoformat(iso_str)
    except ValueError as exc:
        raise SnapshotError(f"Malformed ISO-8601 {context} {value!r}: {exc}") from exc
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise SnapshotError(f"{context} {value!r} must be timezone-aware UTC")
    return dt.astimezone(UTC)


def _next_unique_publish_timestamp(snapshots_root: Path) -> str:
    now_dt = datetime.now(tz=UTC)
    catalog_path = snapshots_root / "catalog.json"
    if catalog_path.exists():
        try:
            loaded = json.loads(catalog_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing_dts = [
                    _parse_catalog_published_at(p["published_at"])
                    for key in ("publications", "partial_publications")
                    for p in (loaded.get(key) or [])
                    if isinstance(p, dict) and p.get("published_at")
                ]
                if existing_dts:
                    max_existing = max(existing_dts)
                    if now_dt <= max_existing:
                        from datetime import timedelta

                        now_dt = max_existing + timedelta(microseconds=1)
        except Exception:
            pass
    return now_dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _classify_item_body_level(item: NormalizedItem) -> str:
    """Classify an item's body depth (`full`, `blurb`, or `headline`).

    - `metadata` sources are always `blurb` (catalog synopses, never counted as full articles).
    - Items whose body equals the title (or adds <= 15 characters beyond the title) are `headline`.
    - `full` requires an article body of >= 300 characters AND at least 2 distinct non-title
      lines/paragraphs so a single-paragraph listing card or RSS summary blurb is never
      misclassified as a full article.
    """
    if item.source_type == "metadata":
        return "blurb"
    body_clean = item.body.strip()
    title_clean = item.title.strip()
    if body_clean == title_clean or len(body_clean) <= len(title_clean) + 15:
        return "headline"
    non_title_paragraphs = [
        line.strip()
        for line in body_clean.splitlines()
        if line.strip() and line.strip() != title_clean
    ]
    if len(body_clean) >= 300 and len(non_title_paragraphs) >= 2:
        return "full"
    return "blurb"


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
    - Rejects short-ID collisions (`SnapshotCollisionError`) and incomplete existing snapshot
      directories (`SnapshotError`) before touching existing files.
    """
    snapshots_root.mkdir(parents=True, exist_ok=True)
    target_dir = snapshots_root / built.snapshot_id
    is_partial = bool(built.ingest_report.get("partial", False))
    pub_timestamp = _next_unique_publish_timestamp(snapshots_root)

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
        missing_files = [f for f in REQUIRED_SNAPSHOT_FILES if not (target_dir / f).exists()]
        if missing_files:
            raise SnapshotError(
                f"Existing snapshot directory {target_dir} is incomplete (missing {missing_files}); "
                f"refusing to overwrite or treat as published"
            )
        # Verify full integrity of existing snapshot + sidecars before idempotent return
        verify_snapshot_integrity(target_dir)
        load_corpus_snapshot(target_dir, data_root=snapshots_root.parent, verify=True)
        load_snapshot_items(
            target_dir, data_root=snapshots_root.parent, representatives_only=False, verify=True
        )
        load_snapshot_items(
            target_dir, data_root=snapshots_root.parent, representatives_only=True, verify=True
        )
        _update_current_and_catalog(
            snapshots_root,
            built.snapshot_id,
            published_at=pub_timestamp,
            keep_snapshots=keep_snapshots,
            set_current=set_current,
            partial=is_partial,
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
        body_level_counts = {"full": 0, "blurb": 0, "headline": 0}
        body_level_by_source: dict[str, dict[str, int]] = {}
        corporate_count = 0
        for item in built.items:
            source_type_counts[item.source_type] = source_type_counts.get(item.source_type, 0) + 1
            cat = classify_content_category(item.title, item.body, source_key=item.source_key)
            category_counts[cat] = category_counts.get(cat, 0) + 1
            blevel = _classify_item_body_level(item)
            body_level_counts[blevel] = body_level_counts.get(blevel, 0) + 1
            src_blevel = body_level_by_source.setdefault(
                item.source_key, {"full": 0, "blurb": 0, "headline": 0}
            )
            src_blevel[blevel] = src_blevel.get(blevel, 0) + 1
            if item.ingest_category == "corporate":
                corporate_count += 1

        pub_dates = sorted(it.published_at[:10] for it in built.items)
        cluster_rep_map = {c.cluster_id: c.representative_doc_id for c in built.clusters}
        url_to_rep = {
            item.url: cluster_rep_map[item.cluster_id]
            for item in built.items
            if item.cluster_id in cluster_rep_map
        }
        coverage_gaps = list(built.ingest_report.get("coverage_gaps") or [])
        if not coverage_gaps:
            for row in built.ingest_report.get("crawl_summary") or []:
                if (
                    isinstance(row, dict)
                    and row.get("coverage_mode")
                    and row.get("coverage_mode") != "full_window_archive"
                ):
                    coverage_gaps.append(f"{row.get('source_key')}:{row.get('coverage_mode')}")
        coverage_complete = bool(
            built.ingest_report.get(
                "coverage_complete",
                len(coverage_gaps) == 0 and not is_partial,
            )
        )

        snapshot_meta = built.corpus_snapshot.model_dump(mode="json") | {
            "body_level_by_source": body_level_by_source,
            "body_level_counts": body_level_counts,
            "category_counts": category_counts,
            "category_counts_heuristic": category_counts,
            "corporate_count": corporate_count,
            "coverage_complete": coverage_complete,
            "coverage_gaps": coverage_gaps,
            "dedup_calibration_status": str(
                built.ingest_report.get("dedup_calibration_status", "provisional_pending_g1b_sweep")
            ),
            "full_body_item_count": body_level_counts["full"],
            "ingest_report": dict(built.ingest_report),
            "manifest_sha256": built.manifest_sha256,
            "observed_earliest": pub_dates[0] if pub_dates else "",
            "observed_latest": pub_dates[-1] if pub_dates else "",
            "partial": is_partial,
            "source_type_counts": source_type_counts,
            "url_to_representative": url_to_rep,
        }
        (staging_dir / "snapshot.json").write_bytes(canonical_json(snapshot_meta))

        os.replace(staging_dir, target_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)

    _update_current_and_catalog(
        snapshots_root,
        built.snapshot_id,
        published_at=pub_timestamp,
        keep_snapshots=keep_snapshots,
        set_current=set_current,
        partial=is_partial,
    )
    return target_dir


def _update_current_and_catalog(
    snapshots_root: Path,
    snapshot_id: str,
    *,
    published_at: str,
    keep_snapshots: int = 2,
    set_current: bool = True,
    partial: bool = False,
) -> None:
    """Atomically update `catalog.json` ordered by publication time, and flip `CURRENT` when
    `set_current=True`.

    Enforces `docs/INGEST_CONTRACT.md` §4:
      - `published_at` is validated as timezone-aware UTC; exact timestamp ties between distinct
        `snapshot_id`s raise `SnapshotError`.
      - `publications` is sorted strictly by `published_at` descending.
      - `partial=True` snapshots are recorded in `partial_publications` and excluded from
        `publications`, `retained`, `retired`, `latest_corpus`, and `CURRENT`.
      - `catalog["current"]` is unconditionally unioned into `retained` so the active serving
        snapshot is never retired.
      - `retained` and `retired` are strictly disjoint (re-publishing an older snapshot removes
        it from `retired`).
      - A corrupt `catalog.json` raises `SnapshotError` rather than silently wiping retirement
        history.
    """
    if not snapshot_id or not str(snapshot_id).strip():
        raise SnapshotError("snapshot_id must not be empty")
    _parse_catalog_published_at(published_at, context="published_at")

    if partial and set_current:
        raise SnapshotError(f"Cannot set CURRENT to partial snapshot {snapshot_id!r}")

    catalog_path = snapshots_root / "catalog.json"
    catalog: dict[str, Any] = {
        "current": None,
        "latest_corpus": None if partial else snapshot_id,
        "partial_publications": [],
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

    raw_pubs_any = catalog.get("publications") or []
    if not isinstance(raw_pubs_any, list):
        raise SnapshotError(f"Corrupt catalog.json at {catalog_path}: publications is not a list")
    raw_pubs: list[dict[str, str]] = []
    for entry in raw_pubs_any:
        if not isinstance(entry, dict) or not entry.get("snapshot_id"):
            raise SnapshotError(f"Corrupt publication entry in {catalog_path}: {entry!r}")
        _parse_catalog_published_at(
            entry.get("published_at"), context="catalog.publications.published_at"
        )
        raw_pubs.append(
            {
                "published_at": str(entry["published_at"]),
                "snapshot_id": str(entry["snapshot_id"]),
            }
        )

    raw_partial_any = catalog.get("partial_publications") or []
    if not isinstance(raw_partial_any, list):
        raise SnapshotError(
            f"Corrupt catalog.json at {catalog_path}: partial_publications is not a list"
        )
    raw_partials: list[dict[str, str]] = []
    for entry in raw_partial_any:
        if not isinstance(entry, dict) or not entry.get("snapshot_id"):
            raise SnapshotError(f"Corrupt partial_publication entry in {catalog_path}: {entry!r}")
        _parse_catalog_published_at(
            entry.get("published_at"), context="catalog.partial_publications.published_at"
        )
        raw_partials.append(
            {
                "published_at": str(entry["published_at"]),
                "snapshot_id": str(entry["snapshot_id"]),
            }
        )

    keep_n = max(1, keep_snapshots)
    sorted_existing_pubs = sorted(
        raw_pubs,
        key=lambda p: _parse_catalog_published_at(p["published_at"]),
        reverse=True,
    )
    currently_retained_ids = {str(p["snapshot_id"]) for p in sorted_existing_pubs[:keep_n]}
    if catalog.get("current"):
        currently_retained_ids.add(str(catalog["current"]))

    if partial:
        existing_partial = next((p for p in raw_partials if p["snapshot_id"] == snapshot_id), None)
        eff_pub_at = (
            existing_partial["published_at"] if existing_partial is not None else published_at
        )
        eff_dt = _parse_catalog_published_at(eff_pub_at)
        for p in [*raw_pubs, *raw_partials]:
            if (
                p["snapshot_id"] != snapshot_id
                and _parse_catalog_published_at(p["published_at"]) == eff_dt
            ):
                raise SnapshotError(
                    f"Ambiguous tied published_at={eff_pub_at!r} between {snapshot_id!r} "
                    f"and {p['snapshot_id']!r}"
                )
        updated_partials = [p for p in raw_partials if p["snapshot_id"] != snapshot_id]
        updated_partials.append({"published_at": eff_pub_at, "snapshot_id": snapshot_id})
        updated_partials.sort(
            key=lambda p: _parse_catalog_published_at(p["published_at"]),
            reverse=True,
        )
        catalog["partial_publications"] = updated_partials
        catalog["publications"] = sorted_existing_pubs
        catalog["latest_corpus"] = (
            sorted_existing_pubs[0]["snapshot_id"] if sorted_existing_pubs else None
        )
        _atomic_write_bytes(catalog_path, canonical_json(catalog))
        return

    existing_pub = next((p for p in sorted_existing_pubs if p["snapshot_id"] == snapshot_id), None)
    if existing_pub is not None and snapshot_id in currently_retained_ids and not set_current:
        eff_published_at = existing_pub["published_at"]
    else:
        eff_published_at = published_at

    eff_dt = _parse_catalog_published_at(eff_published_at)
    for p in [*sorted_existing_pubs, *raw_partials]:
        if (
            p["snapshot_id"] != snapshot_id
            and _parse_catalog_published_at(p["published_at"]) == eff_dt
        ):
            raise SnapshotError(
                f"Ambiguous tied published_at={eff_published_at!r} between {snapshot_id!r} "
                f"and {p['snapshot_id']!r}"
            )

    pubs = [p for p in sorted_existing_pubs if p["snapshot_id"] != snapshot_id]
    pubs.append({"published_at": eff_published_at, "snapshot_id": snapshot_id})
    pubs.sort(
        key=lambda p: _parse_catalog_published_at(p["published_at"]),
        reverse=True,
    )

    if set_current:
        catalog["current"] = snapshot_id

    retained = [p["snapshot_id"] for p in pubs[:keep_n]]
    active_current = catalog.get("current")
    if isinstance(active_current, str) and active_current and active_current not in retained:
        retained.append(active_current)
    retained_set = set(retained)

    # Remove any retained snapshot_id (e.g. a re-published older snapshot or active CURRENT) from `retired`
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
    catalog["partial_publications"] = raw_partials
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
    """Verify all required snapshot files exist, verify `manifest.json` SHA-256 against the
    directory name / `snapshot.json`, and verify `entities.json` and `sources.json` digests
    against `manifest.json`. Raises `SnapshotError` if tampered or incomplete.
    """
    missing = [f for f in REQUIRED_SNAPSHOT_FILES if not (snap_dir / f).exists()]
    if missing:
        raise SnapshotError(
            f"Incomplete snapshot directory {snap_dir}: missing required files {missing}"
        )
    manifest_path = snap_dir / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    try:
        manifest_data = json.loads(manifest_bytes.decode("utf-8"))
    except Exception as exc:
        raise SnapshotError(f"Corrupt manifest.json in {snap_dir}: {exc}") from exc
    if not isinstance(manifest_data, dict):
        raise SnapshotError(f"Corrupt manifest.json in {snap_dir}: root is not a JSON object")

    computed_id, computed_sha, _ = compute_snapshot_id_from_manifest(manifest_data)
    if computed_sha != manifest_sha:
        raise SnapshotError(
            f"Non-canonical manifest.json in {snap_dir}: raw sha256={manifest_sha[:12]} "
            f"!= canonical sha256={computed_sha[:12]}"
        )

    dir_name = snap_dir.name
    if "-" in dir_name and len(dir_name.split("-", 1)[1]) == 12:
        expected_prefix = dir_name.split("-", 1)[1]
        if not manifest_sha.startswith(expected_prefix) or dir_name != computed_id:
            raise SnapshotError(
                f"Snapshot manifest digest/ID mismatch in {snap_dir}: "
                f"dir={dir_name!r} vs computed_id={computed_id!r} (sha256={manifest_sha[:12]!r})"
            )

    ent_sha = hashlib.sha256((snap_dir / "entities.json").read_bytes()).hexdigest()
    expected_ent_sha = str(manifest_data.get("entity_registry_sha256") or "").lower()
    if ent_sha != expected_ent_sha:
        raise SnapshotError(
            f"Tampered entities.json in {snap_dir}: sha256={ent_sha} != "
            f"manifest.entity_registry_sha256={expected_ent_sha}"
        )

    src_sha = hashlib.sha256((snap_dir / "sources.json").read_bytes()).hexdigest()
    expected_src_sha = str(manifest_data.get("source_registry_sha256") or "").lower()
    if src_sha != expected_src_sha:
        raise SnapshotError(
            f"Tampered sources.json in {snap_dir}: sha256={src_sha} != "
            f"manifest.source_registry_sha256={expected_src_sha}"
        )

    snap_json_path = snap_dir / "snapshot.json"
    try:
        meta = json.loads(snap_json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SnapshotError(f"Corrupt snapshot.json in {snap_dir}: {exc}") from exc
    recorded_sha = meta.get("manifest_sha256")
    if recorded_sha != manifest_sha:
        raise SnapshotError(
            f"Snapshot metadata manifest_sha256 mismatch in {snap_dir}: "
            f"{recorded_sha!r} != {manifest_sha!r}"
        )
    return manifest_sha


def _validate_and_load_jsonl_items(
    snap_dir: Path,
    *,
    data_root: Path,
    representatives_only: bool,
    verify: bool,
) -> list[NormalizedItem]:
    filename = "representatives.jsonl" if representatives_only else "items.jsonl"
    jsonl_path = snap_dir / filename

    if not verify:
        if not jsonl_path.exists() and representatives_only:
            clusters = load_snapshot_clusters(snap_dir, data_root=data_root, verify=False)
            rep_ids = {c.representative_doc_id for c in clusters}
            return [
                it
                for it in _validate_and_load_jsonl_items(
                    snap_dir, data_root=data_root, representatives_only=False, verify=False
                )
                if it.doc_id in rep_ids
            ]
        out: list[NormalizedItem] = []
        with jsonl_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    out.append(NormalizedItem.from_dict(json.loads(stripped)))
        return out

    verify_snapshot_integrity(snap_dir)
    manifest_data = json.loads((snap_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest_items = [m for m in manifest_data.get("items", []) if isinstance(m, dict)]
    manifest_by_doc_id: dict[str, dict[str, Any]] = {str(m["doc_id"]): m for m in manifest_items}
    verified_clusters = load_snapshot_clusters(snap_dir, data_root=data_root, verify=True)

    def _validate_single_file(
        target_path: Path,
        expected_ids: list[str],
        *,
        items_line_map: dict[str, str] | None = None,
        record_lines_into: dict[str, str] | None = None,
    ) -> list[NormalizedItem]:
        rows: list[NormalizedItem] = []
        seen_ids: list[str] = []
        with target_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    raw_row = json.loads(stripped)
                except Exception as exc:
                    raise SnapshotError(f"Malformed JSON row in {target_path}: {exc}") from exc
                item = NormalizedItem.from_dict(raw_row)
                raw_doc_id = raw_row.get("doc_id")
                if raw_doc_id is not None and raw_doc_id != item.doc_id:
                    raise SnapshotError(
                        f"Tampered doc_id in {target_path}: {raw_doc_id!r} != {item.doc_id!r}"
                    )
                raw_body_sha = raw_row.get("body_sha256")
                if raw_body_sha is not None and raw_body_sha != item.body_sha256:
                    raise SnapshotError(
                        f"Tampered body_sha256 for {item.doc_id} in {target_path}: "
                        f"{raw_body_sha!r} != {item.body_sha256!r}"
                    )
                manifest_row = manifest_by_doc_id.get(item.doc_id)
                if manifest_row is None:
                    raise SnapshotError(
                        f"Tampered item {item.doc_id} in {target_path}: missing from verified manifest.json"
                    )
                if manifest_row != item.to_manifest_item():
                    raise SnapshotError(
                        f"Tampered item {item.doc_id} in {target_path}: row disagrees with verified manifest.json"
                    )
                if items_line_map is not None:
                    matching_items_line = items_line_map.get(item.doc_id)
                    if matching_items_line is None or matching_items_line != stripped:
                        raise SnapshotError(
                            f"Tampered representative {item.doc_id} in {target_path}: "
                            f"row is not byte-identical to items.jsonl"
                        )
                if record_lines_into is not None:
                    record_lines_into[item.doc_id] = stripped
                seen_ids.append(item.doc_id)
                rows.append(item)

        if seen_ids != expected_ids:
            missing_ids = set(expected_ids) - set(seen_ids)
            extra_ids = set(seen_ids) - set(expected_ids)
            raise SnapshotError(
                f"Incomplete, duplicate, or out-of-order rows in {target_path}: "
                f"expected {len(expected_ids)} rows, got {len(seen_ids)} "
                f"(missing={len(missing_ids)}, extra={len(extra_ids)})"
            )
        return rows

    # Always validate BOTH items.jsonl (all members) and representatives.jsonl (cluster
    # representatives) on either read path so Config A/A'/A0 and Config B readers share an
    # identical whole-snapshot completeness boundary (E Blocker 1, F #6666814959).
    expected_all_doc_ids = [str(m["doc_id"]) for m in manifest_items]
    expected_rep_doc_ids = sorted(c.representative_doc_id for c in verified_clusters)
    items_line_by_doc_id: dict[str, str] = {}

    validated_all = _validate_single_file(
        snap_dir / "items.jsonl",
        expected_all_doc_ids,
        record_lines_into=items_line_by_doc_id,
    )
    validated_reps = _validate_single_file(
        snap_dir / "representatives.jsonl",
        expected_rep_doc_ids,
        items_line_map=items_line_by_doc_id,
    )

    return validated_reps if representatives_only else validated_all


def iter_snapshot_items(
    snapshot_dir_or_id: Path | str,
    *,
    data_root: Path = Path("data"),
    representatives_only: bool = False,
    verify: bool = True,
) -> Iterator[NormalizedItem]:
    """Iterate `NormalizedItem` rows from a frozen snapshot directory.

    When `verify=True`, the entire file is validated against `manifest.json` and `clusters.jsonl`
    before the first row is yielded so downstream index builders never ingest a truncated stream.
    """
    snap_dir = resolve_snapshot_dir(snapshot_dir_or_id, data_root=data_root)
    validated = _validate_and_load_jsonl_items(
        snap_dir,
        data_root=data_root,
        representatives_only=representatives_only,
        verify=verify,
    )
    yield from validated


def load_snapshot_items(
    snapshot_dir_or_id: Path | str,
    *,
    data_root: Path = Path("data"),
    representatives_only: bool = False,
    verify: bool = True,
) -> list[NormalizedItem]:
    """Load all (or representative-only) `NormalizedItem` records from a frozen snapshot."""
    snap_dir = resolve_snapshot_dir(snapshot_dir_or_id, data_root=data_root)
    return _validate_and_load_jsonl_items(
        snap_dir,
        data_root=data_root,
        representatives_only=representatives_only,
        verify=verify,
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
            try:
                clusters.append(DedupCluster.from_dict(json.loads(stripped)))
            except Exception as exc:
                raise SnapshotError(f"Malformed cluster row in {clusters_path}: {exc}") from exc

    if verify:
        manifest_data = json.loads((snap_dir / "manifest.json").read_text(encoding="utf-8"))
        manifest_by_doc_id = {
            str(m["doc_id"]): m for m in manifest_data.get("items", []) if isinstance(m, dict)
        }
        seen_members: set[str] = set()
        rep_order: list[str] = []
        for cluster in clusters:
            rep_order.append(cluster.representative_doc_id)
            if cluster.representative_doc_id not in cluster.member_doc_ids:
                raise SnapshotError(
                    f"Tampered cluster {cluster.cluster_id} in {clusters_path}: "
                    f"representative {cluster.representative_doc_id!r} not in member_doc_ids"
                )
            for mid in cluster.member_doc_ids:
                if mid in seen_members:
                    raise SnapshotError(
                        f"Overlapping cluster membership for {mid!r} in {clusters_path}"
                    )
                seen_members.add(mid)
                mrow = manifest_by_doc_id.get(mid)
                if mrow is None:
                    raise SnapshotError(
                        f"Cluster member {mid!r} in {clusters_path} missing from manifest.json"
                    )
                if mrow.get("cluster_id") != cluster.cluster_id:
                    raise SnapshotError(
                        f"Cluster mismatch for {mid!r} in {clusters_path}: "
                        f"cluster={cluster.cluster_id!r} vs manifest={mrow.get('cluster_id')!r}"
                    )
        if seen_members != set(manifest_by_doc_id.keys()):
            raise SnapshotError(
                f"Incomplete cluster partition in {clusters_path}: covered {len(seen_members)} "
                f"of {len(manifest_by_doc_id)} manifest items"
            )
        if rep_order != sorted(rep_order):
            raise SnapshotError(
                f"Clusters in {clusters_path} are not sorted by representative_doc_id"
            )

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
    if verify:
        manifest_data = json.loads((snap_dir / "manifest.json").read_text(encoding="utf-8"))
        manifest_items = [m for m in manifest_data.get("items", []) if isinstance(m, dict)]
        expected_id, _, _ = compute_snapshot_id_from_manifest(manifest_data)
        verified_clusters = load_snapshot_clusters(snap_dir, data_root=data_root, verify=True)
        expected_item_count = len(manifest_items)
        expected_deduped_count = len(verified_clusters)
        expected_source_count = len(
            {str(m["source_key"]) for m in manifest_items if m.get("source_key")}
        )
        expected_w_start = str(manifest_data.get("window_start") or "")
        expected_w_end = str(manifest_data.get("window_end") or "")
        if (
            raw.get("snapshot_id") != expected_id
            or raw.get("item_count") != expected_item_count
            or raw.get("deduped_count") != expected_deduped_count
            or raw.get("source_count") != expected_source_count
            or raw.get("window_start") != expected_w_start
            or raw.get("window_end") != expected_w_end
        ):
            raise SnapshotError(
                f"Tampered snapshot.json in {snap_dir}: metadata fields disagree with verified "
                f"manifest.json / clusters.jsonl"
            )
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
    verify_snapshot_integrity(snap_dir)
    load_corpus_snapshot(snap_dir, data_root=data_root, verify=True)
    canonical = normalize_url(url)
    clusters = {
        c.cluster_id: c.representative_doc_id
        for c in load_snapshot_clusters(snap_dir, data_root=data_root, verify=True)
    }
    manifest_data = json.loads((snap_dir / "manifest.json").read_text(encoding="utf-8"))
    expected_url_map: dict[str, str] = {}
    for m in manifest_data.get("items", []):
        if isinstance(m, dict):
            m_url = str(m.get("url") or "")
            m_cid = str(m.get("cluster_id") or "")
            if m_url and m_cid in clusters:
                expected_url_map[m_url] = clusters[m_cid]

    meta_path = snap_dir / "snapshot.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        mapping = meta.get("url_to_representative")
        if mapping is not None and mapping != expected_url_map:
            raise SnapshotError(
                f"Tampered url_to_representative in {meta_path}: disagrees with verified manifest.json"
            )
    return expected_url_map.get(canonical)


__all__ = [
    "MANIFEST_VERSION",
    "REQUIRED_SNAPSHOT_FILES",
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
