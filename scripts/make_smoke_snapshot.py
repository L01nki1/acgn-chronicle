"""Build a tiny synthetic frozen snapshot so the offline smoke can run without the real corpus.

Owner: F (WP2) | Spec: docs/RUNBOOK.md §2, docs/INGEST_CONTRACT.md §3

The real 36k-item snapshot is distributed out of band (RUNBOOK §2) and is never committed.
This fixture writes a handful of *synthetic* announcements through the same publisher the
backfill uses, so `scripts/smoke_e2e.py` can prove the snapshot -> index -> facts -> /ask
seams on any machine. The text is invented; nothing here is corpus evidence.

    python scripts/make_smoke_snapshot.py --data-root .tmp_smoke/data --count 12
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date, timedelta
from pathlib import Path

from acgn_rag.core.ingest.dedup import DedupPolicy, cluster_items
from acgn_rag.core.ingest.normalize import NormalizedItem
from acgn_rag.core.ingest.snapshot import publish_snapshot

#: Synthetic announcements: (title, publisher, source_key, source_type, lang, body).
SEEDS: tuple[tuple[str, str, str, str, str, str], ...] = (
    (
        "Synthetic Anime A Announces October Premiere",
        "Synthetic Studio",
        "synthetic-official",
        "official",
        "en",
        "Synthetic Studio announced that the anime adaptation of Synthetic Novel A will "
        "premiere in October 2026 on three terrestrial channels. The director is a fictional "
        "name used for fixture purposes only.",
    ),
    (
        "Synthetic Game B Gets Release Date",
        "Synthetic Publisher",
        "synthetic-official",
        "official",
        "en",
        "Synthetic Publisher confirmed that Synthetic Game B launches on 2026-03-14 for two "
        "home consoles and PC. A demo is planned for the previous month.",
    ),
    (
        "Synthetic Manga C Reaches Final Chapter",
        "Synthetic Magazine",
        "synthetic-media",
        "media",
        "en",
        "The serialization of Synthetic Manga C will end with its final chapter in the "
        "December issue, the magazine announced. The collected edition runs to twelve volumes.",
    ),
    (
        "Synthetic Light Novel D Anime Adaptation Confirmed",
        "Synthetic Media Outlet",
        "synthetic-media",
        "media",
        "en",
        "A television anime adaptation of Synthetic Light Novel D was confirmed at a stage "
        "event, with the studio and broadcast season to be announced later.",
    ),
    (
        "Synthetic Studio E Opens New Animation Department",
        "Synthetic Studio",
        "synthetic-official",
        "official",
        "en",
        "Synthetic Studio E established a dedicated 2D animation department and plans to hire "
        "thirty animators over the next fiscal year, according to its announcement.",
    ),
    (
        "Synthetic Franchise F Collaboration Event Announced",
        "Synthetic Publisher",
        "synthetic-official",
        "official",
        "en",
        "Synthetic Franchise F and a fictional beverage brand will run a collaboration event "
        "from 2026-07-01 to 2026-07-31, with limited merchandise at participating stores.",
    ),
)


def _doc(index: int, seed: tuple[str, str, str, str, str, str]) -> NormalizedItem:
    title, publisher, source_key, source_type, lang, body = seed
    url = f"https://example.invalid/synthetic/{index}"
    published = (date(2026, 1, 5) + timedelta(days=index * 7)).isoformat()
    return NormalizedItem(
        doc_id="doc_" + hashlib.sha256(url.encode("utf-8")).hexdigest(),
        title=title,
        body=body,
        url=url,
        publisher=publisher,
        source_type=source_type,  # type: ignore[arg-type]
        published_at=f"{published}T00:00:00Z",
        lang=lang,
        media_ids=[],
        source_key=source_key,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish a synthetic frozen snapshot")
    parser.add_argument("--data-root", required=True, help="data root to publish under")
    parser.add_argument("--count", type=int, default=len(SEEDS), help="documents to write")
    args = parser.parse_args(argv)
    if args.count < 1:
        print("--count must be positive", file=sys.stderr)
        return 2

    items = [_doc(index, SEEDS[index % len(SEEDS)]) for index in range(args.count)]
    clustered = cluster_items(items, policy=DedupPolicy(algorithm="singleton-example-v0"))
    built = clustered  # DedupResult carries the clustered items
    from acgn_rag.core.ingest.snapshot import build_snapshot_from_items

    snapshot = build_snapshot_from_items(
        built.items,
        window_start="2026-01-01",
        window_end="2026-11-05",
        dedup_policy=clustered.policy,
    )
    path = publish_snapshot(snapshot, snapshots_root=Path(args.data_root) / "snapshots")
    print(json.dumps({"snapshot_id": snapshot.snapshot_id, "path": str(path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
