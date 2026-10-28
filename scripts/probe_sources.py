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
from datetime import date, timedelta
from pathlib import Path

# Look-back depths, in days. "all" means "no lower bound": whatever the source can page back to.
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


def _registry():
    """Load the adapter registry.

    A owns `acgn_rag.core.ingest.adapters.REGISTRY`; this script only consumes it. Keeping the lookup
    in one place means the probe never needs to know how a source is fetched.
    """
    try:
        from acgn_rag.core.ingest.adapters import REGISTRY
    except (ImportError, AttributeError):
        return {}
    return REGISTRY


def _count(items) -> tuple[int, int, int]:
    """Count items per SourceType; anything unlabelled counts as metadata."""
    counts = {"official": 0, "media": 0, "metadata": 0}
    for item in items:
        label = getattr(item, "source_type", "metadata")
        counts[label if label in counts else "metadata"] += 1
    return counts["official"], counts["media"], counts["metadata"]


def _probe_one(adapter, name: str, depth: str, days: int | None) -> dict[str, object]:
    until = date.today()
    since = until - timedelta(days=days) if days is not None else None
    row: dict[str, object] = {
        "source": name,
        "adapter": type(adapter).__name__,
        "depth": depth,
        "since": since.isoformat() if since else "",
        "items": 0,
        "earliest": "",
        "latest": "",
        "official": 0,
        "media": 0,
        "metadata": 0,
        "note": "",
    }
    try:
        # backfill() is the historical entry point; a source that cannot page must raise, not return
        # everything (docs/SOURCES.md 7).
        items = adapter.backfill(date_range=(since, until) if since else None)
    except Exception as exc:
        row["note"] = f"{type(exc).__name__}: {exc}"[:160]
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
    return row


def _suggest_window(rows: list[dict[str, object]]) -> str:
    """Smallest depth whose total already meets the target, in depth order."""
    totals: dict[str, int] = {}
    for depth in DEPTHS:
        totals[depth] = sum(int(r["items"]) for r in rows if r["depth"] == depth)
    for depth in DEPTHS:
        if totals[depth] >= TARGET_ITEMS:
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

    registry = _registry()
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
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nwrote {OUT_PATH}")
    print("window:", _suggest_window(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
