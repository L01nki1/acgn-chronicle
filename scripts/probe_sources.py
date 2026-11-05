"""Source-depth probe: how many items each source can yield at each look-back depth.

Owner: A | Spec: docs/SOURCES.md 6, docs/ROADMAP.md 2 (G1a)

Why this exists: the evaluation window is sized by measurement, not by extrapolating a published
items/day figure. A rolling RSS feed describes *new* items and cannot be paged at all, so a backfill
is bounded by how far back each source actually keeps an archive. This script answers one question
per source: "if I ask for three years, how many items come back, and how far back does it reach?"

It deliberately stops before normalize / dedup / chunk / embed / index: the point is to size the
window in minutes, not to build anything.

Output (tracked, small): eval/source_depth_probe.csv, one row per (source, depth).

Usage:
    python scripts/probe_sources.py            # all depths
    python scripts/probe_sources.py --depth 3y # one depth only
"""

from __future__ import annotations

import csv
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

# Look-back depths, in days. "all" uses ARCHIVE_FLOOR (2005-01-01, predating Toei 2008-10 and
# Capcom 2014-03) so every backfill() call still carries an explicit (since, until) bound (C1).
ARCHIVE_FLOOR = date(2005, 1, 1)
DEPTHS: dict[str, int | None] = {"90d": 90, "1y": 365, "2y": 730, "3y": 1095, "all": None}

FIELDS = [
    "source",
    "adapter",
    "depth",
    "since",
    "items",
    "earliest",
    "latest",
    "official",
    "media",
    "metadata",
    "note",
]

OUT_PATH = Path("eval/source_depth_probe.csv")
TARGET_ITEMS = 8_000


def _registry(*, ann_max_days_per_call: int | None = None):
    """Load the adapter registry.

    A owns `acgn_rag.core.ingest.adapters.REGISTRY`; this script only consumes it. Keeping the lookup
    in one place means the probe never needs to know how a source is fetched.
    """
    try:
        from acgn_rag.core.ingest.adapters import build_default_registry
    except (ImportError, AttributeError):
        return {}
    return build_default_registry(
        ann_max_days_per_call=ann_max_days_per_call,
        ann_cache_dir=Path("data/raw/ann_day_cache"),
    )


def _count(items) -> tuple[int, int, int]:
    """Count items per SourceType; anything unlabelled counts as metadata."""
    counts = {"official": 0, "media": 0, "metadata": 0}
    for item in items:
        label = getattr(item, "source_type", "metadata")
        counts[label if label in counts else "metadata"] += 1
    return counts["official"], counts["media"], counts["metadata"]


def _probe_one(
    adapter,
    name: str,
    depth: str,
    days: int | None,
    *,
    until: date | None = None,
) -> dict[str, object]:
    upper = until if until is not None else datetime.now(tz=UTC).date()
    since = upper - timedelta(days=days) if days is not None else ARCHIVE_FLOOR
    row: dict[str, object] = {
        "source": name,
        "adapter": type(adapter).__name__,
        "depth": depth,
        "since": since.isoformat(),
        "items": 0,
        "earliest": "",
        "latest": "",
        "official": 0,
        "media": 0,
        "metadata": 0,
        "note": "",
    }
    extra_note = ""
    try:
        # backfill() is the historical entry point; a source that cannot page must raise, not return
        # everything (docs/SOURCES.md 7).
        if (
            name == "ann-news"
            and depth == "all"
            and getattr(adapter, "max_days_per_call", None) is None
        ):
            items_3y = list(
                adapter.backfill(date_range=(upper - timedelta(days=1095), upper)) or []
            )
            if len(items_3y) >= TARGET_ITEMS:
                items = items_3y
                adapter.last_sampled_days = 1096
                adapter.last_requested_days = 1096
                extra_note = (
                    "3y already >= 8000 target (1096/1096 days); pre-2023 archive not needed"
                )
            else:
                items = adapter.backfill(date_range=(since, upper))
        else:
            items = adapter.backfill(date_range=(since, upper))
    except Exception as exc:
        note = f"{type(exc).__name__}: {exc}"
        if type(exc).__name__ == "BackfillNotSupportedError":
            try:
                tail = list(adapter.discover(since=since, use_conditional=False) or [])
                tail_dates = sorted(
                    str(getattr(i, "published_at", ""))[:10]
                    for i in tail
                    if getattr(i, "published_at", None)
                )
                span = f"{tail_dates[0]}..{tail_dates[-1]}" if tail_dates else "-"
                note = f"BackfillNotSupportedError: rolling feed; discover() tail={len(tail)} items ({span})"
            except Exception:
                pass
        row["note"] = note[:160]
        return row

    items = list(items or [])
    row["items"] = len(items)
    dates = sorted(
        str(getattr(i, "published_at", "")) for i in items if getattr(i, "published_at", None)
    )
    if dates:
        row["earliest"], row["latest"] = dates[0][:10], dates[-1][:10]
    official, media, meta = _count(items)
    row["official"], row["media"], row["metadata"] = official, media, meta

    notes: list[str] = []
    if extra_note:
        notes.append(extra_note)
    sampled = getattr(adapter, "last_sampled_days", 0)
    requested = getattr(adapter, "last_requested_days", 0)
    if isinstance(sampled, int) and isinstance(requested, int) and 0 < sampled < requested:
        notes.append(f"sampled_days={sampled}/{requested} (partial sample)")
    raw_entries = getattr(adapter, "last_raw_entry_count", 0)
    skipped_entries = getattr(adapter, "last_skipped_entries", 0)
    dup_urls = getattr(adapter, "last_duplicate_url_entries", 0)
    if isinstance(raw_entries, int) and raw_entries > len(items):
        notes.append(
            f"feed_total={raw_entries}, dup_urls_collapsed={dup_urls}, skipped={skipped_entries}"
        )
    if notes:
        row["note"] = "; ".join(notes)[:160]
    return row


def _suggest_window(rows: list[dict[str, object]]) -> str:
    """Smallest depth (>= 3y per DECISIONS D2 / SOURCES §6) whose total meets the target."""
    if any("sampled_days=" in str(r.get("note", "")) for r in rows):
        return "partial sample run (sampled_days < requested_days); run unsampled probe before freezing window"
    totals: dict[str, int] = {}
    for depth in DEPTHS:
        totals[depth] = sum(int(r["items"]) for r in rows if r["depth"] == depth)
    for depth in ("3y", "all"):
        if depth in totals and totals[depth] >= TARGET_ITEMS:
            return f"{depth} is enough ({totals[depth]} items); freeze it in docs/DECISIONS.md"
    best = max(totals, key=lambda d: totals[d])
    return (
        f"no depth reaches {TARGET_ITEMS}; the best is {best} with {totals[best]} items. "
        "Either widen further, add sources, or take the downgrade path in docs/SOURCES.md 6"
    )


def main(argv: list[str]) -> int:
    only = None
    if "--depth" in argv:
        only = argv[argv.index("--depth") + 1]
        if only not in DEPTHS:
            print(f"unknown depth {only!r}; choose from {', '.join(DEPTHS)}", file=sys.stderr)
            return 2

    ann_sample_days: int | None = None
    for flag in ("--sample-days", "--ann-sample-days"):
        if flag in argv:
            ann_sample_days = int(argv[argv.index(flag) + 1])
            break

    registry = _registry(ann_max_days_per_call=ann_sample_days)
    if not registry:
        print(
            "No adapters registered yet.\n"
            "A: fill acgn_rag.core.ingest.adapters.REGISTRY with {name: adapter_instance} and re-run.\n"
            "The probe stops before normalize/dedup/embed, so it costs only the requests themselves.",
            file=sys.stderr,
        )
        return 1

    depths = {only: DEPTHS[only]} if only else DEPTHS
    rows = []
    for name, adapter in registry.items():
        for depth, days in depths.items():
            row = _probe_one(adapter, name, depth, days)
            rows.append(row)
            print(
                f"{name:28} {depth:4} items={row['items']:>6}  earliest={row['earliest'] or '-':10} {row['note']}"
            )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nwrote {OUT_PATH}")
    print("window:", _suggest_window(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
