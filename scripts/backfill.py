"""Historical corpus backfill and idempotent snapshot builder (`scripts/backfill.py`).

Enforces `docs/COMPLIANCE.md` C1 (explicit `--start`/`--since` and `--end`/`--until` window required;
raises ValueError when omitted) and `docs/INGEST_CONTRACT.md` §3-§5 (deterministic normalization,
append-only entity/source registries, agglomerative complete-linkage deduplication, canonical
manifest v1, partial-crawl refusal by default, and two-rebuild byte identity).

Usage:
    python scripts/backfill.py --start 2023-10-28 --end 2026-10-28
    python scripts/backfill.py --since 2023-10-28 --until 2026-10-28 --verify-idempotency
    python scripts/backfill.py --start 2023-10-28 --end 2026-10-28 --from-raw data/raw/items.jsonl --verify-idempotency

Owner: A (WP1) | Spec: docs/INGEST_CONTRACT.md §3-§5, docs/RUNBOOK.md §2, Issue #9, Issue #30
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acgn_rag.config import load_settings
from acgn_rag.core.ingest.adapters import (
    BackfillNotSupportedError,
    EmptyBackfillWindowError,
    SourceAdapter,
    build_default_registry,
    validate_backfill_window,
)
from acgn_rag.core.ingest.normalize import PARAGRAPH_SEP, RawIngestItem, canonical_json
from acgn_rag.core.ingest.snapshot import (
    BuiltSnapshot,
    _split_body_paragraphs,
    build_snapshot_from_raw,
    publish_snapshot,
)


def serialize_raw_item(item: RawIngestItem) -> dict[str, Any]:
    return {
        "body": item.body,
        "event_type": item.event_type,
        "external_media_refs": [list(pair) for pair in item.external_media_refs],
        "fetched_via": item.fetched_via,
        "ingest_category": item.ingest_category,
        "lang": item.lang,
        "media_ids": list(item.media_ids),
        "published_at": item.published_at,
        "publisher": item.publisher,
        "source_key": item.source_key,
        "source_type": item.source_type,
        "source_tz": item.source_tz,
        "title": item.title,
        "url": item.url,
    }


def deserialize_raw_item(data: Mapping[str, Any]) -> RawIngestItem:
    ext_refs = tuple(
        (str(pair[0]), str(pair[1]))
        for pair in (data.get("external_media_refs") or [])
        if isinstance(pair, (list, tuple)) and len(pair) == 2
    )
    return RawIngestItem(
        url=str(data["url"]),
        title=str(data["title"]),
        body=str(data["body"]),
        published_at=str(data["published_at"]),
        publisher=str(data["publisher"]),
        source_type=data["source_type"],
        source_key=str(data["source_key"]),
        lang=str(data.get("lang", "und")),
        source_tz=data.get("source_tz"),
        ingest_category=data.get("ingest_category"),
        event_type=data.get("event_type"),
        media_ids=tuple(int(x) for x in (data.get("media_ids") or [])),
        external_media_refs=ext_refs,
        fetched_via=data.get("fetched_via", "direct"),
    )


def _summary_sidecar_path(raw_path: Path) -> Path:
    return raw_path.with_suffix(f"{raw_path.suffix}.summary.json")


def save_raw_jsonl(
    items: Sequence[RawIngestItem],
    path: Path,
    *,
    crawl_summary: Mapping[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(canonical_json(serialize_raw_item(it)) for it in items)
    path.write_bytes(payload)
    raw_sha256 = hashlib.sha256(payload).hexdigest()
    raw_bytes = len(payload)
    sidecar_dict: dict[str, Any] = (
        dict(crawl_summary)
        if crawl_summary is not None
        else {
            "errored_sources": [],
            "partial": False,
            "sampled_sources": [],
            "sources": {},
            "total_raw_items": len(items),
        }
    )
    sidecar_dict["raw_bytes"] = raw_bytes
    sidecar_dict["raw_sha256"] = raw_sha256
    _summary_sidecar_path(path).write_bytes(canonical_json(sidecar_dict))


def load_raw_jsonl(path: Path) -> list[RawIngestItem]:
    items: list[RawIngestItem] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            items.append(deserialize_raw_item(json.loads(stripped)))
    return items


def load_raw_summary(path: Path) -> dict[str, Any] | None:
    sidecar = _summary_sidecar_path(path)
    if not sidecar.exists():
        return None
    try:
        loaded = json.loads(sidecar.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else None
    except Exception:
        return None


def _adapter_diagnostics(adapter: object) -> dict[str, Any]:
    diag: dict[str, Any] = {}
    for attr, key in (
        ("last_raw_entry_count", "raw_entries"),
        ("last_duplicate_url_entries", "duplicate_url_entries"),
        ("last_skipped_entries", "skipped_entries"),
        ("last_keyword_filtered_entries", "keyword_filtered_entries"),
        ("last_requested_days", "requested_days"),
        ("last_sampled_days", "sampled_days"),
        ("last_days_with_items", "days_with_items"),
        ("last_empty_days", "empty_days"),
        ("last_total_pages_fetched", "total_pages_fetched"),
        ("last_truncated_days", "truncated_days"),
        ("last_cached_article_bodies_applied", "cached_article_bodies_applied"),
        ("last_pages_fetched", "pages_fetched"),
    ):
        val = getattr(adapter, attr, None)
        if isinstance(val, int) and val > 0:
            diag[key] = val
    if bool(getattr(adapter, "last_page_cap_hit", False)):
        diag["page_cap_hit"] = True
    skip_reasons = getattr(adapter, "last_skip_reasons", None)
    if isinstance(skip_reasons, Mapping) and skip_reasons:
        diag["skip_reasons"] = dict(skip_reasons)
    earliest_obs = getattr(adapter, "last_earliest_observed_date", None)
    if isinstance(earliest_obs, str) and earliest_obs:
        diag["earliest_observed_date"] = earliest_obs
    return diag


def _classify_raw_item_body_level(adapter: object, item: RawIngestItem) -> str:
    if (
        item.source_type == "metadata"
        or getattr(adapter, "source_type", "") == "metadata"
        or type(adapter).__name__ == "StructuredApiAdapter"
    ):
        return "blurb"
    body_clean = item.body.strip()
    title_clean = item.title.strip()
    paragraphs = _split_body_paragraphs(body_clean)
    visible_body = PARAGRAPH_SEP.join(paragraphs).strip()
    if (
        not visible_body
        or visible_body == title_clean
        or len(visible_body) <= len(title_clean) + 15
    ):
        return "headline"
    non_title_paragraphs = [para for para in paragraphs if para != title_clean]
    if len(visible_body) >= 300 and len(non_title_paragraphs) >= 2:
        return "full"
    return "blurb"


def _count_raw_body_levels(adapter: object, items: Sequence[RawIngestItem]) -> dict[str, int]:
    counts = {"full": 0, "blurb": 0, "headline": 0}
    for it in items:
        lvl = _classify_raw_item_body_level(adapter, it)
        counts[lvl] = counts.get(lvl, 0) + 1
    return counts


def _infer_source_body_level(adapter: object, items: Sequence[RawIngestItem]) -> str:
    if (
        getattr(adapter, "source_type", "") == "metadata"
        or type(adapter).__name__ == "StructuredApiAdapter"
    ):
        return "blurb"
    if not items:
        return "none"
    counts = _count_raw_body_levels(adapter, items)
    total = len(items)
    if counts["full"] >= 50 and counts["headline"] >= 50:
        return "mixed_full_and_headline"
    if counts["full"] >= total * 0.5:
        return "full"
    if counts["headline"] >= total * 0.5:
        return "headline"
    body_lens = [len(it.body.strip()) for it in items]
    extra_lens = [max(0, len(it.body.strip()) - len(it.title.strip())) for it in items]
    med_body = statistics.median(body_lens)
    med_extra = statistics.median(extra_lens)
    if med_extra <= 15:
        return "headline"
    if med_body >= 300 and counts["full"] > 0:
        return "full"
    return "blurb"


def _infer_source_coverage_mode(
    adapter: object,
    *,
    name: str,
    mode: str,
    start_d: date,
    items: Sequence[RawIngestItem],
    is_sampled_src: bool,
    page_cap_hit: bool,
) -> tuple[str, bool]:
    if is_sampled_src:
        return "sampled_partial", False
    if page_cap_hit:
        return "page_cap_partial", False
    if type(adapter).__name__ == "StructuredApiAdapter":
        return "catalog_top_n", False
    if mode != "backfill":
        return "rolling_tail_only", False
    earliest_obs = getattr(adapter, "last_earliest_observed_date", None)
    if not earliest_obs and items:
        earliest_obs = min(it.published_at[:10] for it in items)
    reached = bool(getattr(adapter, "last_reached_window_start", False))
    if not reached and isinstance(earliest_obs, str) and earliest_obs:
        reached = earliest_obs <= (start_d + timedelta(days=14)).isoformat()
    if name == "ann-news" and not is_sampled_src and not page_cap_hit:
        reached = True
    return ("full_window_archive" if reached else "rolling_tail_only"), reached


def crawl_backfill_window(
    start_date: date | str | None,
    end_date: date | str | None,
    *,
    registry: Mapping[str, SourceAdapter] | None = None,
    selected_sources: Sequence[str] | None = None,
    strict: bool = False,
    include_rolling_discover: bool = True,
    ann_sample_days: int | None = None,
    conditional_cache_path: Path | None = None,
    ann_cache_dir: Path | None = None,
    wp_cache_dir: Path | None = None,
) -> tuple[list[RawIngestItem], dict[str, Any]]:
    """Run `backfill((start_date, end_date))` across adapters in `registry`.

    Raises `ValueError` immediately if `start_date` or `end_date` is missing (COMPLIANCE C1).
    """
    start_d, end_d = validate_backfill_window((start_date, end_date))
    active_registry = (
        registry
        if registry is not None
        else build_default_registry(
            ann_max_days_per_call=ann_sample_days,
            conditional_cache_path=conditional_cache_path,
            ann_cache_dir=ann_cache_dir,
            wp_cache_dir=wp_cache_dir,
        )
    )
    source_names = list(selected_sources) if selected_sources else list(active_registry.keys())

    start_iso = start_d.isoformat()
    end_iso = end_d.isoformat()

    collected: list[RawIngestItem] = []
    per_source: dict[str, dict[str, Any]] = {}
    errored_sources: list[str] = []
    sampled_sources: list[str] = []
    capped_sources: list[str] = []

    for name in source_names:
        adapter = active_registry.get(name)
        if adapter is None:
            raise KeyError(f"Unknown source adapter key: {name!r}")
        try:
            items = adapter.backfill((start_d, end_d))
            collected.extend(items)
            diag = _adapter_diagnostics(adapter)
            sampled = diag.get("sampled_days")
            requested = diag.get("requested_days")
            is_sampled_src = (
                isinstance(sampled, int) and isinstance(requested, int) and 0 < sampled < requested
            )
            page_cap_hit = (
                bool(diag.get("page_cap_hit")) or int(diag.get("truncated_days") or 0) > 0
            )
            if is_sampled_src:
                sampled_sources.append(name)
            if page_cap_hit:
                capped_sources.append(name)
            cov_mode, reached_start = _infer_source_coverage_mode(
                adapter,
                name=name,
                mode="backfill",
                start_d=start_d,
                items=items,
                is_sampled_src=is_sampled_src,
                page_cap_hit=page_cap_hit,
            )
            body_level = _infer_source_body_level(adapter, items)
            body_level_counts = _count_raw_body_levels(adapter, items)
            status_str = (
                "sampled_partial"
                if is_sampled_src
                else ("page_cap_partial" if page_cap_hit else "ok")
            )
            per_source[name] = {
                "body_level": body_level,
                "body_level_counts": body_level_counts,
                "count": len(items),
                "coverage_mode": cov_mode,
                "mode": "backfill",
                "reached_window_start": reached_start,
                "status": status_str,
                **diag,
            }
        except BackfillNotSupportedError as exc:
            if include_rolling_discover:
                try:
                    disc_raw = adapter.discover(since=start_d, use_conditional=False)
                    tail_items = [
                        it for it in disc_raw if start_iso <= it.published_at[:10] <= end_iso
                    ]
                    collected.extend(tail_items)
                    not_mod = bool(getattr(adapter, "last_not_modified", False))
                    per_source[name] = {
                        "body_level": _infer_source_body_level(adapter, tail_items),
                        "body_level_counts": _count_raw_body_levels(adapter, tail_items),
                        "count": len(tail_items),
                        "coverage_mode": "rolling_tail_only",
                        "mode": "discover_tail",
                        "note": str(exc),
                        "reached_window_start": False,
                        "status": (
                            "not_modified"
                            if not_mod
                            else ("ok" if tail_items else "unsupported_or_empty_window")
                        ),
                        **_adapter_diagnostics(adapter),
                    }
                    continue
                except Exception as disc_exc:
                    if type(disc_exc).__name__ == "RobotsDisallowedError" and bool(
                        getattr(disc_exc, "explicit_disallow", False)
                    ):
                        per_source[name] = {
                            "body_level": "none",
                            "count": 0,
                            "coverage_mode": "skipped_robots_disallow",
                            "mode": "discover_tail",
                            "note": str(exc),
                            "reached_window_start": False,
                            "reason": f"{type(disc_exc).__name__}: {disc_exc}",
                            "status": "skipped_robots_disallow",
                        }
                        continue
                    per_source[name] = {
                        "body_level": "none",
                        "count": 0,
                        "coverage_mode": "error",
                        "mode": "discover_tail",
                        "reached_window_start": False,
                        "reason": f"{type(disc_exc).__name__}: {disc_exc}",
                        "status": "error",
                    }
                    errored_sources.append(name)
                    if strict:
                        raise
                    continue
            per_source[name] = {
                "body_level": "none",
                "count": 0,
                "coverage_mode": "rolling_tail_only",
                "reached_window_start": False,
                "reason": str(exc),
                "status": "unsupported_or_empty_window",
            }
            if strict:
                raise
        except EmptyBackfillWindowError as exc:
            per_source[name] = {
                "body_level": "none",
                "count": 0,
                "coverage_mode": "empty_window",
                "reached_window_start": False,
                "reason": str(exc),
                "status": "unsupported_or_empty_window",
                **_adapter_diagnostics(adapter),
            }
            if strict:
                raise
        except Exception as exc:
            if type(exc).__name__ == "RobotsDisallowedError" and bool(
                getattr(exc, "explicit_disallow", False)
            ):
                per_source[name] = {
                    "body_level": "none",
                    "count": 0,
                    "coverage_mode": "skipped_robots_disallow",
                    "mode": "backfill",
                    "reached_window_start": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "status": "skipped_robots_disallow",
                    **_adapter_diagnostics(adapter),
                }
                if strict:
                    raise
                continue
            per_source[name] = {
                "body_level": "none",
                "count": 0,
                "coverage_mode": "error",
                "mode": "backfill",
                "reached_window_start": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "status": "error",
                **_adapter_diagnostics(adapter),
            }
            errored_sources.append(name)
            if strict:
                raise

    if not collected:
        raise RuntimeError(
            f"Backfill for window {start_iso}..{end_iso} collected 0 items across all sources"
        )

    is_partial = bool(errored_sources or sampled_sources or capped_sources)
    coverage_gaps = [
        f"{k}:{v.get('coverage_mode')}"
        for k, v in sorted(per_source.items())
        if v.get("coverage_mode") != "full_window_archive"
    ]
    all_full_archive = len(coverage_gaps) == 0 and not is_partial

    summary = {
        "all_sources_full_window_archive": all_full_archive,
        "capped_sources": capped_sources,
        "coverage_complete": all_full_archive,
        "coverage_gaps": coverage_gaps,
        "errored_sources": errored_sources,
        "partial": is_partial,
        "sampled_sources": sampled_sources,
        "sources": per_source,
        "total_raw_items": len(collected),
        "window_end": end_iso,
        "window_start": start_iso,
    }
    return collected, summary


def verify_two_rebuild_idempotency(
    raw_items: Sequence[RawIngestItem],
    *,
    window_start: date | str,
    window_end: date | str,
    entity_registry: Any = None,
    persist_entity_registry_path: Path | None = None,
    crawl_summary: Sequence[Mapping[str, Any]] | None = None,
    partial: bool = False,
) -> tuple[BuiltSnapshot, BuiltSnapshot, dict[str, Any]]:
    """Run two independent rebuilds on the same frozen raw items (second with reversed input order)
    and assert identical `manifest.json` bytes, `manifest_sha256`, and `snapshot_id`.
    """
    build_1 = build_snapshot_from_raw(
        raw_items,
        window_start=window_start,
        window_end=window_end,
        entity_registry=entity_registry,
        persist_entity_registry_path=persist_entity_registry_path,
        crawl_summary=crawl_summary,
        partial=partial,
    )
    reversed_raw = list(reversed(raw_items))
    build_2 = build_snapshot_from_raw(
        reversed_raw,
        window_start=window_start,
        window_end=window_end,
        entity_registry=entity_registry,
        persist_entity_registry_path=persist_entity_registry_path,
        crawl_summary=crawl_summary,
        partial=partial,
    )

    byte_diff = sum(
        1 for a, b in zip(build_1.manifest_bytes, build_2.manifest_bytes, strict=False) if a != b
    )
    byte_diff += abs(len(build_1.manifest_bytes) - len(build_2.manifest_bytes))

    if (
        build_1.manifest_bytes != build_2.manifest_bytes
        or build_1.manifest_sha256 != build_2.manifest_sha256
        or build_1.snapshot_id != build_2.snapshot_id
    ):
        raise RuntimeError(
            f"Rebuild idempotency check failed: build_1={build_1.snapshot_id} "
            f"vs build_2={build_2.snapshot_id} (byte_diff={byte_diff})"
        )

    proof = {
        "build_1": {
            "deduped_count": build_1.corpus_snapshot.deduped_count,
            "item_count": build_1.corpus_snapshot.item_count,
            "manifest_bytes": len(build_1.manifest_bytes),
            "sha256": build_1.manifest_sha256,
            "snapshot_id": build_1.snapshot_id,
            "source_count": build_1.corpus_snapshot.source_count,
        },
        "build_2_reversed_input": {
            "deduped_count": build_2.corpus_snapshot.deduped_count,
            "item_count": build_2.corpus_snapshot.item_count,
            "manifest_bytes": len(build_2.manifest_bytes),
            "sha256": build_2.manifest_sha256,
            "snapshot_id": build_2.snapshot_id,
            "source_count": build_2.corpus_snapshot.source_count,
        },
        "byte_diff_count": byte_diff,
        "identical": True,
        "window_end": str(build_1.manifest["window_end"]),
        "window_start": str(build_1.manifest["window_start"]),
    }
    return build_1, build_2, proof


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run historical backfill and publish a frozen canonical corpus snapshot."
    )
    parser.add_argument(
        "--start",
        "--since",
        dest="start",
        required=False,
        help="Inclusive UTC start date YYYY-MM-DD (COMPLIANCE C1)",
    )
    parser.add_argument(
        "--end",
        "--until",
        dest="end",
        required=False,
        help="Inclusive UTC end date YYYY-MM-DD (COMPLIANCE C1)",
    )
    parser.add_argument(
        "--sources",
        default="",
        help="Optional comma-separated list of source_keys (default: all in REGISTRY)",
    )
    parser.add_argument(
        "--from-raw",
        default="",
        help="Optional path to an existing raw JSONL file (skips network crawl)",
    )
    parser.add_argument(
        "--save-raw",
        default="",
        help="Optional output path to save raw crawled items JSONL",
    )
    parser.add_argument(
        "--data-root",
        default="",
        help="Data directory root (defaults to ACGN_DATA_DIR / data)",
    )
    parser.add_argument(
        "--entity-registry",
        default="",
        help="Optional path to an existing entities.json to use as base_registry",
    )
    parser.add_argument(
        "--verify-idempotency",
        action="store_true",
        help="Run two rebuilds (normal and reversed input) and assert identical manifest bytes/ID",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow publishing a partial snapshot when one or more sources fail (marks partial=true and never flips CURRENT)",
    )
    parser.add_argument(
        "--set-current",
        action="store_true",
        help="Also flip the serving pointer data/snapshots/CURRENT (only after index/facts completion)",
    )
    parser.add_argument(
        "--ann-sample-days",
        "--sample-days",
        dest="ann_sample_days",
        type=int,
        default=None,
        help="Optional cap on ANN archive days to sample (default: None = full unsampled window)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        start_d, end_d = validate_backfill_window((args.start, args.end))
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    settings = load_settings()
    data_root = Path(args.data_root) if args.data_root else settings.paths.root
    snapshots_root = data_root / "snapshots"
    persist_entity_path = data_root / "registry" / "entities.json"

    base_entity_reg = None
    if args.entity_registry:
        from acgn_rag.core.ingest.facts_link import EntityRegistry

        reg_path = Path(args.entity_registry)
        if not reg_path.exists():
            print(f"ERROR: entity registry file {reg_path} does not exist", file=sys.stderr)
            return 2
        base_entity_reg = EntityRegistry.load(reg_path)

    crawl_rows: list[dict[str, Any]] | None = None
    is_partial = False

    if args.from_raw:
        raw_path = Path(args.from_raw)
        if not raw_path.exists():
            print(f"ERROR: raw input file {raw_path} does not exist", file=sys.stderr)
            return 2
        raw_bytes_data = raw_path.read_bytes()
        actual_sha256 = hashlib.sha256(raw_bytes_data).hexdigest()
        actual_bytes = len(raw_bytes_data)
        raw_items = load_raw_jsonl(raw_path)
        saved_summary = load_raw_summary(raw_path)
        if saved_summary is None:
            is_partial = True
            crawl_rows = [
                {
                    "count": len(raw_items),
                    "coverage_mode": "unknown_unverified_raw",
                    "source_key": "raw_input",
                    "status": "missing_summary_sidecar",
                }
            ]
            if not args.allow_partial:
                print(
                    f"ERROR: Missing summary sidecar {_summary_sidecar_path(raw_path)} for --from-raw "
                    "(INGEST_CONTRACT §5). Pass --allow-partial to record a non-current partial snapshot.",
                    file=sys.stderr,
                )
                return 1
        elif "raw_sha256" in saved_summary and (
            saved_summary.get("raw_sha256") != actual_sha256
            or int(saved_summary.get("raw_bytes") or -1) != actual_bytes
        ):
            is_partial = True
            crawl_rows = [
                {
                    "count": len(raw_items),
                    "coverage_mode": "unknown_unverified_raw",
                    "source_key": "raw_input",
                    "status": "mismatched_summary_sidecar",
                }
            ]
            if not args.allow_partial:
                print(
                    f"ERROR: Stale/mismatched summary sidecar {_summary_sidecar_path(raw_path)} for --from-raw "
                    f"(expected sha256={saved_summary.get('raw_sha256')}, actual={actual_sha256}). "
                    "Pass --allow-partial to record a non-current partial snapshot.",
                    file=sys.stderr,
                )
                return 1
        else:
            crawl_rows = [
                {"source_key": k, **v}
                for k, v in sorted((saved_summary.get("sources") or {}).items())
            ]
            errored = saved_summary.get("errored_sources") or []
            sampled = saved_summary.get("sampled_sources") or []
            capped = saved_summary.get("capped_sources") or []
            is_partial = bool(saved_summary.get("partial") or errored or sampled or capped)
            if is_partial and not args.allow_partial:
                print(
                    f"ERROR: Partial raw input refused (INGEST_CONTRACT §5): "
                    f"errored={errored}, sampled={sampled}, capped={capped}. "
                    "Pass --allow-partial to record a non-current partial snapshot.",
                    file=sys.stderr,
                )
                return 1
    else:
        selected = (
            [s.strip() for s in args.sources.split(",") if s.strip()] if args.sources else None
        )
        raw_items, crawl_summary = crawl_backfill_window(
            start_d,
            end_d,
            selected_sources=selected,
            ann_sample_days=args.ann_sample_days,
            conditional_cache_path=data_root / "raw" / "http_conditional_cache.json",
            ann_cache_dir=data_root / "raw" / "ann_day_cache",
            wp_cache_dir=data_root / "raw" / "wp_cache",
        )
        default_raw_out = (
            Path(args.save_raw)
            if args.save_raw
            else data_root / "raw" / f"backfill_{start_d.isoformat()}_{end_d.isoformat()}.jsonl"
        )
        save_raw_jsonl(raw_items, default_raw_out, crawl_summary=crawl_summary)
        print(f"Saved {len(raw_items)} raw items to {default_raw_out}")
        print(json.dumps(crawl_summary, ensure_ascii=False, indent=2))

        crawl_rows = [
            {"source_key": k, **v} for k, v in sorted(crawl_summary.get("sources", {}).items())
        ]
        errored = crawl_summary.get("errored_sources") or []
        sampled = crawl_summary.get("sampled_sources") or []
        capped = crawl_summary.get("capped_sources") or []
        is_partial = bool(crawl_summary.get("partial") or errored or sampled or capped)
        if is_partial and not args.allow_partial:
            print(
                f"ERROR: Partial crawl refused (INGEST_CONTRACT §5): "
                f"errored={errored}, sampled={sampled}, capped={capped}. "
                "Re-run full healthy sources or pass --allow-partial to record a non-current partial snapshot.",
                file=sys.stderr,
            )
            return 1

    if args.verify_idempotency:
        built, _built2, proof = verify_two_rebuild_idempotency(
            raw_items,
            window_start=start_d,
            window_end=end_d,
            entity_registry=base_entity_reg,
            persist_entity_registry_path=persist_entity_path,
            crawl_summary=crawl_rows,
            partial=is_partial,
        )
        print("Two-rebuild idempotency check PASSED:")
        print(json.dumps(proof, ensure_ascii=False, indent=2))
    else:
        built = build_snapshot_from_raw(
            raw_items,
            window_start=start_d,
            window_end=end_d,
            entity_registry=base_entity_reg,
            persist_entity_registry_path=persist_entity_path,
            crawl_summary=crawl_rows,
            partial=is_partial,
        )

    # Partial crawls (including sampled runs) must never flip CURRENT even if --set-current was passed
    should_set_current = bool(args.set_current and not is_partial)
    snap_dir = publish_snapshot(
        built,
        snapshots_root=snapshots_root,
        set_current=should_set_current,
        keep_snapshots=settings.store.keep_snapshots,
    )
    print(
        f"Published snapshot {built.snapshot_id} to {snap_dir} "
        f"(items={built.corpus_snapshot.item_count}, deduped={built.corpus_snapshot.deduped_count}, "
        f"sources={built.corpus_snapshot.source_count}, set_current={should_set_current}, partial={is_partial})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
