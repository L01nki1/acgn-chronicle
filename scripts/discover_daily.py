"""Incremental daily discovery and source health check (`scripts/discover_daily.py`).

Implements `docs/INGEST_CONTRACT.md` §5, `docs/DECISIONS.md` ("Two corpora, never mixed"),
and Issue #9 G1b requirements:
  - Calls `adapter.discover(since)` across registered W1 sources.
  - Distinguishes legitimate `304 Not Modified` / zero new items from broken selectors or HTTP errors,
    and always writes `health_<stamp>.json` even when one or more sources error.
  - When verified against an existing frozen snapshot (`--base-snapshot`), inserts only genuinely
    new `doc_id`s; already-ingested `doc_id`s (including cross-mode RSS vs archive copies of the
    same canonical URL) are counted as `unchanged_count` so no ghost snapshot is minted.
  - Daily increments are never written into the frozen evaluation `CURRENT` lineage.

Owner: A (WP1) | Spec: docs/INGEST_CONTRACT.md §5, docs/ROADMAP.md G1b, Issue #9
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acgn_rag.config import load_settings
from acgn_rag.core.ingest.adapters import SourceAdapter, build_default_registry
from acgn_rag.core.ingest.normalize import (
    NormalizedItem,
    RawIngestItem,
    canonical_json,
    compute_doc_id,
    normalize_batch,
    parse_utc_timestamp,
)
from acgn_rag.core.ingest.snapshot import (
    BuiltSnapshot,
    build_snapshot_from_items,
    load_corpus_snapshot,
    load_snapshot_items,
    publish_snapshot,
    resolve_snapshot_dir,
)


def _coerce_reference_date(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.astimezone(UTC).date() if value.tzinfo else value.date()
    if isinstance(value, date):
        return value
    s = str(value).strip()
    return date.fromisoformat(s[:10])


@dataclass(frozen=True)
class IncrementalRunResult:
    """Result of an incremental discovery + reconciliation pass against a base snapshot."""

    base_snapshot_id: str | None
    discovered_raw_count: int
    inserted_count: int
    updated_count: int
    unchanged_count: int
    new_snapshot_id: str | None
    source_health: dict[str, dict[str, Any]]
    out_of_window_count: int = 0
    quarantined_count: int = 0
    metadata_drift_count: int = 0
    metadata_drift_doc_ids: tuple[str, ...] = ()
    metadata_drift_doc_ids_truncated_to: int = 20
    window_start: str | None = None
    window_end: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_snapshot_id": self.base_snapshot_id,
            "discovered_raw_count": self.discovered_raw_count,
            "inserted_count": self.inserted_count,
            "metadata_drift_count": self.metadata_drift_count,
            "metadata_drift_doc_ids": list(self.metadata_drift_doc_ids),
            "metadata_drift_doc_ids_truncated_to": self.metadata_drift_doc_ids_truncated_to,
            "new_snapshot_id": self.new_snapshot_id,
            "out_of_window_count": self.out_of_window_count,
            "quarantined_count": self.quarantined_count,
            "source_health": self.source_health,
            "unchanged_count": self.unchanged_count,
            "updated_count": self.updated_count,
            "window_end": self.window_end,
            "window_start": self.window_start,
        }


def run_incremental_discovery(
    *,
    since: date | datetime | str | None = None,
    reference_date: date | datetime | str | None = None,
    registry: Mapping[str, SourceAdapter] | None = None,
    selected_sources: Sequence[str] | None = None,
    base_snapshot_dir_or_id: Path | str | None = None,
    data_root: Path = Path("data"),
    publish_if_changed: bool = False,
    allow_partial: bool = False,
    allow_body_updates: bool = False,
    strict: bool = False,
) -> tuple[IncrementalRunResult, BuiltSnapshot | None]:
    """Run `discover(since)` on registered adapters and reconcile against `base_snapshot_dir_or_id`.

    - Uses a rolling 14-day daily window `[ref_end - 13d, ref_end]` so daily snapshots under
      `data/snapshots_daily/` never absorb the 3-year historical evaluation corpus (`DECISIONS.md`
      "Two corpora, never mixed") and never allow future metadata air dates (e.g. 2027) to shift
      the snapshot date prefix.
    - Uses `base_snapshot_dir_or_id` to recognize already-ingested `doc_id`s; metadata/body drift
      on known `doc_id`s when `allow_body_updates=False` is counted in `metadata_drift_count`.
    """
    active_registry = (
        registry
        if registry is not None
        else build_default_registry(
            conditional_cache_path=data_root / "raw" / "http_conditional_cache.json"
        )
    )
    source_names = list(selected_sources) if selected_sources else list(active_registry.keys())

    raw_discovered: list[RawIngestItem] = []
    health: dict[str, dict[str, Any]] = {}

    for name in source_names:
        adapter = active_registry.get(name)
        if adapter is None:
            raise KeyError(f"Unknown source adapter key: {name!r}")
        try:
            items = adapter.discover(since=since)
            raw_discovered.extend(items)
            not_mod = bool(getattr(adapter, "last_not_modified", False))
            skipped = int(getattr(adapter, "last_skipped_entries", 0) or 0)
            entry_health: dict[str, Any] = {
                "count": len(items),
                "status": "not_modified" if not_mod else ("ok" if items else "no_new_items"),
            }
            if skipped > 0:
                entry_health["skipped_entries"] = skipped
                skip_reasons = getattr(adapter, "last_skip_reasons", None)
                if isinstance(skip_reasons, Mapping) and skip_reasons:
                    entry_health["skip_reasons"] = dict(skip_reasons)
            health[name] = entry_health
        except Exception as exc:
            if type(exc).__name__ == "RobotsDisallowedError" and bool(
                getattr(exc, "explicit_disallow", False)
            ):
                health[name] = {
                    "count": 0,
                    "note": f"{type(exc).__name__}: {exc}",
                    "status": "skipped_robots_disallow",
                }
                continue
            health[name] = {
                "count": 0,
                "error": f"{type(exc).__name__}: {exc}",
                "status": "error",
            }
            if strict:
                raise

    has_errors = any(v.get("status") == "error" for v in health.values())
    if publish_if_changed and has_errors and not allow_partial:
        raise RuntimeError(
            "Incremental discovery encountered source errors; refusing to publish partial snapshot"
        )

    if base_snapshot_dir_or_id is None:
        if reference_date is not None:
            ref_end = _coerce_reference_date(reference_date)
            ref_start = ref_end - timedelta(days=13)
            norm = normalize_batch(
                raw_discovered,
                window_start=ref_start.isoformat(),
                window_end=ref_end.isoformat(),
                quarantine_on_error=True,
            )
            w_start_str: str | None = ref_start.isoformat()
            w_end_str: str | None = ref_end.isoformat()
        else:
            norm = normalize_batch(raw_discovered, quarantine_on_error=True)
            w_start_str = None
            w_end_str = None
        res = IncrementalRunResult(
            base_snapshot_id=None,
            discovered_raw_count=len(raw_discovered),
            inserted_count=len(norm.items),
            updated_count=0,
            unchanged_count=0,
            new_snapshot_id=None,
            source_health=health,
            out_of_window_count=norm.out_of_window_count,
            quarantined_count=len(norm.quarantined),
            window_start=w_start_str,
            window_end=w_end_str,
        )
        return res, None

    snap_dir = resolve_snapshot_dir(base_snapshot_dir_or_id, data_root=data_root)
    base_meta = load_corpus_snapshot(snap_dir, data_root=data_root)
    base_items = load_snapshot_items(snap_dir, data_root=data_root, representatives_only=False)
    by_doc_id: dict[str, NormalizedItem] = {it.doc_id: it for it in base_items}

    base_end_date = date.fromisoformat(base_meta.window_end)
    if reference_date is not None:
        ref_end = _coerce_reference_date(reference_date)
    else:
        today_utc = datetime.now(tz=UTC).date()
        max_allowed_end = max(today_utc, base_end_date + timedelta(days=1))
        valid_discovered_dates: list[date] = []
        for raw_it in raw_discovered:
            try:
                pub_iso = parse_utc_timestamp(raw_it.published_at, source_tz=raw_it.source_tz)
                d_val = date.fromisoformat(pub_iso[:10])
                if d_val <= max_allowed_end:
                    valid_discovered_dates.append(d_val)
            except Exception:
                continue
        if valid_discovered_dates:
            ref_end = max(base_end_date, max(valid_discovered_dates))
        else:
            ref_end = max(base_end_date, today_utc)

    ref_start = ref_end - timedelta(days=13)
    w_start_iso = ref_start.isoformat()
    w_end_iso = ref_end.isoformat()

    norm = normalize_batch(
        raw_discovered,
        window_start=w_start_iso,
        window_end=w_end_iso,
        quarantine_on_error=True,
    )

    inserted = 0
    updated = 0
    unchanged = 0
    drift_doc_ids: list[str] = []
    for item in norm.items:
        prior = by_doc_id.get(item.doc_id)
        if prior is None:
            by_doc_id[item.doc_id] = item
            inserted += 1
            continue

        prior_sig = (
            prior.title,
            prior.body_sha256,
            prior.published_at,
            prior.publisher,
            prior.source_type,
            prior.source_key,
            prior.lang,
            prior.ingest_category,
            prior.event_type,
        )
        item_sig = (
            item.title,
            item.body_sha256,
            item.published_at,
            item.publisher,
            item.source_type,
            item.source_key,
            item.lang,
            item.ingest_category,
            item.event_type,
        )
        has_drift = prior_sig != item_sig

        if allow_body_updates and has_drift:
            by_doc_id[item.doc_id] = item
            updated += 1
        else:
            unchanged += 1
            if has_drift:
                drift_doc_ids.append(item.doc_id)

    drift_tuple = tuple(sorted(set(drift_doc_ids)))

    if inserted == 0 and updated == 0:
        return (
            IncrementalRunResult(
                base_snapshot_id=base_meta.snapshot_id,
                discovered_raw_count=len(raw_discovered),
                inserted_count=0,
                updated_count=0,
                unchanged_count=unchanged,
                new_snapshot_id=None,
                source_health=health,
                out_of_window_count=norm.out_of_window_count,
                quarantined_count=len(norm.quarantined),
                metadata_drift_count=len(drift_tuple),
                metadata_drift_doc_ids=drift_tuple[:20],
                window_start=w_start_iso,
                window_end=w_end_iso,
            ),
            None,
        )

    from acgn_rag.core.ingest.facts_link import (
        EntityRegistry,
        build_default_source_registry,
        build_entity_registry_from_raw,
    )

    base_ent_reg = EntityRegistry.load(snap_dir / "entities.json")
    ent_reg = build_entity_registry_from_raw(raw_discovered, base_registry=base_ent_reg)

    ext_refs_by_doc: dict[str, list[tuple[str, str]]] = {}
    for raw_it in raw_discovered:
        if raw_it.external_media_refs:
            try:
                did = compute_doc_id(raw_it.url)
                ext_refs_by_doc.setdefault(did, []).extend(raw_it.external_media_refs)
            except Exception:
                continue

    daily_window_items = [
        it for it in by_doc_id.values() if w_start_iso <= it.published_at[:10] <= w_end_iso
    ]
    used_sources = {it.source_key for it in daily_window_items}
    src_reg = build_default_source_registry().filter_used(used_sources)
    has_errors = any(v.get("status") == "error" for v in health.values())
    crawl_rows = [{"source_key": k, **v} for k, v in sorted(health.items())]

    built = build_snapshot_from_items(
        daily_window_items,
        window_start=w_start_iso,
        window_end=w_end_iso,
        entity_registry=ent_reg,
        source_registry=src_reg,
        external_refs_by_doc_id=ext_refs_by_doc,
        crawl_summary=crawl_rows,
        partial=has_errors,
    )
    if publish_if_changed and (not has_errors or allow_partial):
        # DECISIONS 2026-10-28 ("Two corpora, never mixed"): daily increments are published to
        # `snapshots_daily` and never flip the frozen evaluation `CURRENT` pointer.
        publish_snapshot(
            built,
            snapshots_root=data_root / "snapshots_daily",
            set_current=False,
        )

    return (
        IncrementalRunResult(
            base_snapshot_id=base_meta.snapshot_id,
            discovered_raw_count=len(raw_discovered),
            inserted_count=inserted,
            updated_count=updated,
            unchanged_count=unchanged,
            new_snapshot_id=built.snapshot_id if (not has_errors or allow_partial) else None,
            source_health=health,
            out_of_window_count=norm.out_of_window_count,
            quarantined_count=len(norm.quarantined),
            metadata_drift_count=len(drift_tuple),
            metadata_drift_doc_ids=drift_tuple[:20],
            window_start=w_start_iso,
            window_end=w_end_iso,
        ),
        built if (not has_errors or allow_partial) else None,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run daily incremental discovery across W1 adapters and check source health."
    )
    parser.add_argument("--since", default=None, help="ISO-8601 date or UTC timestamp cursor")
    parser.add_argument(
        "--reference-date",
        default=None,
        help="Optional UTC reference date YYYY-MM-DD bounding the 14-day rolling daily window",
    )
    parser.add_argument(
        "--base-snapshot",
        default=None,
        help="Optional existing snapshot ID/path to reconcile increments against",
    )
    parser.add_argument(
        "--sources",
        default="",
        help="Optional comma-separated source_keys (default: all registered)",
    )
    parser.add_argument(
        "--data-root",
        default="",
        help="Data directory root (default: ACGN_DATA_DIR / data)",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish a daily incremental snapshot under data/snapshots_daily/ if new items exist",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    settings = load_settings()
    data_root = Path(args.data_root) if args.data_root else settings.paths.root
    selected = [s.strip() for s in args.sources.split(",") if s.strip()] if args.sources else None

    result, _built = run_incremental_discovery(
        since=args.since,
        reference_date=args.reference_date,
        selected_sources=selected,
        base_snapshot_dir_or_id=args.base_snapshot,
        data_root=data_root,
        publish_if_changed=args.publish,
        strict=False,
    )

    health_dir = data_root / "raw" / "daily_health"
    health_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    (health_dir / f"health_{stamp}.json").write_bytes(canonical_json(result.to_dict()))
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))

    errored = [k for k, v in result.source_health.items() if v.get("status") == "error"]
    return 1 if errored else 0


if __name__ == "__main__":
    raise SystemExit(main())
