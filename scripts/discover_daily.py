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
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acgn_rag.config import load_settings
from acgn_rag.core.ingest.adapters import SourceAdapter, build_default_registry
from acgn_rag.core.ingest.normalize import (
    NormalizedItem,
    RawIngestItem,
    canonical_json,
    normalize_batch,
)
from acgn_rag.core.ingest.snapshot import (
    BuiltSnapshot,
    build_snapshot_from_items,
    load_corpus_snapshot,
    load_snapshot_items,
    publish_snapshot,
    resolve_snapshot_dir,
)


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_snapshot_id": self.base_snapshot_id,
            "discovered_raw_count": self.discovered_raw_count,
            "inserted_count": self.inserted_count,
            "new_snapshot_id": self.new_snapshot_id,
            "source_health": self.source_health,
            "unchanged_count": self.unchanged_count,
            "updated_count": self.updated_count,
        }


def run_incremental_discovery(
    *,
    since: date | datetime | str | None = None,
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

    If `base_snapshot_dir_or_id` is provided and all discovered `doc_id`s are already present
    in the base snapshot (or `discovered_raw_count == 0`), returns `inserted_count=0,
    updated_count=0, new_snapshot_id=None` without minting a new snapshot.
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
            health[name] = {
                "count": len(items),
                "status": "not_modified" if not_mod else ("ok" if items else "no_new_items"),
            }
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
        norm = normalize_batch(raw_discovered, quarantine_on_error=True)
        res = IncrementalRunResult(
            base_snapshot_id=None,
            discovered_raw_count=len(raw_discovered),
            inserted_count=len(norm.items),
            updated_count=0,
            unchanged_count=0,
            new_snapshot_id=None,
            source_health=health,
        )
        return res, None

    snap_dir = resolve_snapshot_dir(base_snapshot_dir_or_id, data_root=data_root)
    base_meta = load_corpus_snapshot(snap_dir, data_root=data_root)
    base_items = load_snapshot_items(snap_dir, data_root=data_root, representatives_only=False)
    by_doc_id: dict[str, NormalizedItem] = {it.doc_id: it for it in base_items}

    norm = normalize_batch(
        raw_discovered,
        window_start=base_meta.window_start,
        window_end=base_meta.window_end,
        quarantine_on_error=True,
    )

    inserted = 0
    updated = 0
    unchanged = 0
    for item in norm.items:
        prior = by_doc_id.get(item.doc_id)
        if prior is None:
            by_doc_id[item.doc_id] = item
            inserted += 1
        elif allow_body_updates and (
            prior.body_sha256 != item.body_sha256 or prior.title != item.title
        ):
            by_doc_id[item.doc_id] = item
            updated += 1
        else:
            # Already-known doc_id (including cross-mode RSS vs archive observations of the same URL)
            unchanged += 1

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
            ),
            None,
        )

    from acgn_rag.core.ingest.facts_link import EntityRegistry, build_default_source_registry

    ent_reg = EntityRegistry.load(snap_dir / "entities.json")
    all_items = list(by_doc_id.values())
    used_sources = {it.source_key for it in all_items}
    src_reg = build_default_source_registry().filter_used(used_sources)
    has_errors = any(v.get("status") == "error" for v in health.values())
    crawl_rows = [{"source_key": k, **v} for k, v in sorted(health.items())]

    built = build_snapshot_from_items(
        all_items,
        window_start=base_meta.window_start,
        window_end=base_meta.window_end,
        entity_registry=ent_reg,
        source_registry=src_reg,
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
        ),
        built if (not has_errors or allow_partial) else None,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run daily incremental discovery across W1 adapters and check source health."
    )
    parser.add_argument("--since", default=None, help="ISO-8601 date or UTC timestamp cursor")
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
