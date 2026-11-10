"""Reproduce one threshold of the recorded dedup sweep from the frozen snapshot.

Owner: F (WP2) | Spec: docs/RUNBOOK.md §3.3, docs/INGEST_CONTRACT.md §2

`eval/snapshot_evidence.json` records a `threshold_sweep_20261105` whose `total_items` is
36,168 while the frozen snapshot holds 36,167 members and 36,121 clusters, and whose 0.86
`cluster_count` is 36,127. Re-running `cluster_items` on the snapshot at 0.86 reproduces the
snapshot's own 36,121 clusters, so the recorded table was produced by a different clustering
run than the one that published the snapshot. This script shows exactly where the two differ,
pair by pair, without guessing which is right:

    python scripts/dedup_diff.py --snapshot 20261105-d797b5f0ae8e --threshold 0.86

It prints the recorded clusters that do not exist in the reproduction, the reproduced
non-singleton clusters that the record does not contain, and the per-pair reason the current
`_pair_merge_score` gives for each recorded-but-reproduced-apart pair (time gap, language,
media_ids, event_type, corporate guard, cosine).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from acgn_rag.core.ingest.dedup import (
    DEFAULT_SIMILARITY_THRESHOLD,
    DedupPolicy,
    _default_item_similarity,
    _has_conflicting_headline_numbers,
    char_ngram_cosine,
    cluster_items,
)
from acgn_rag.core.ingest.normalize import NormalizedItem
from acgn_rag.core.ingest.snapshot import iter_snapshot_items, load_snapshot_clusters
from acgn_rag.core.store.payload import iso_to_epoch


def _reason(first: NormalizedItem, second: NormalizedItem, threshold: float) -> str:
    """Why the current policy would not merge this pair, in evaluation order."""
    gap = abs(iso_to_epoch(first.published_at) - iso_to_epoch(second.published_at))
    if gap > 72 * 3600:
        return f"time_gap_{gap // 3600}h"
    if first.ingest_category == "corporate" or second.ingest_category == "corporate":
        return "corporate_guard"
    media_a, media_b = set(first.media_ids), set(second.media_ids)
    if media_a and media_b and media_a != media_b:
        return "media_ids_differ"
    if first.event_type and second.event_type and first.event_type != second.event_type:
        return "event_type_differs"
    if (
        media_a
        and media_a == media_b
        and first.event_type
        and first.event_type == second.event_type
    ):
        if first.lang == second.lang:
            if _has_conflicting_headline_numbers(first.title, second.title):
                return "primary_channel_headline_numbers"
            title_sim = char_ngram_cosine(first.title, second.title, n=3)
            if title_sim < 0.30:
                return f"primary_channel_title_sim_{title_sim:.2f}"
        return "primary_channel_should_merge"
    if first.lang != second.lang:
        return "language_differs"
    cosine = _default_item_similarity(first, second)
    return f"aux_cosine_{cosine:.4f}_{'below' if cosine < threshold else 'above'}_threshold"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diff a recorded dedup sweep against the snapshot")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD)
    parser.add_argument("--evidence", default="eval/snapshot_evidence.json")
    parser.add_argument("--limit", type=int, default=40, help="rows printed per section")
    args = parser.parse_args(argv)

    data_root = Path(args.data_root)
    items = list(iter_snapshot_items(args.snapshot, data_root=data_root, verify=True))
    by_doc = {item.doc_id: item for item in items}
    frozen = load_snapshot_clusters(args.snapshot, data_root=data_root)
    frozen_multi = {c.cluster_id: c for c in frozen if len(c.member_doc_ids) > 1}

    result = cluster_items(
        items, policy=DedupPolicy(thresholds={"same_lang_cosine": args.threshold})
    )
    reproduced_multi = {c.cluster_id: c for c in result.clusters if len(c.member_doc_ids) > 1}

    evidence = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
    sweep = evidence.get("threshold_sweep_20261105", {})
    recorded_row = next(
        (
            row
            for row in sweep.get("sweep_table", [])
            if abs(float(row["threshold"]) - args.threshold) < 1e-9
        ),
        None,
    )
    recorded_samples = [
        row for row in sweep.get("sample_merged_clusters_at_0_86", []) if args.threshold == 0.86
    ]

    report: dict[str, Any] = {
        "snapshot_id": args.snapshot,
        "threshold": args.threshold,
        "snapshot": {
            "items": len(items),
            "clusters": len(frozen),
            "non_singleton": len(frozen_multi),
            "collapsed": len(items) - len(frozen),
        },
        "reproduced": {
            "items": len(result.items),
            "clusters": len(result.clusters),
            "non_singleton": len(reproduced_multi),
            "collapsed": len(result.items) - len(result.clusters),
        },
        "recorded_sweep_row": recorded_row,
        "matches_snapshot": len(reproduced_multi) == len(frozen_multi),
    }

    only_in_record = [cid for cid in frozen_multi if cid not in reproduced_multi]
    only_in_reproduction = [cid for cid in reproduced_multi if cid not in frozen_multi]
    report["clusters_only_in_snapshot"] = len(only_in_record)
    report["clusters_only_in_reproduction"] = len(only_in_reproduction)

    print(json.dumps({k: v for k, v in report.items()}, indent=2))

    print("\n== recorded 0.86 sample clusters vs the frozen snapshot ==")
    reasons: Counter[str] = Counter()
    shown = 0
    for row in recorded_samples:
        cluster_id = row["cluster_id"]
        frozen_cluster = frozen_multi.get(cluster_id)
        if frozen_cluster is None:
            reasons["recorded_cluster_absent_from_snapshot"] += 1
            if shown < args.limit:
                shown += 1
                print(
                    f"{cluster_id}: recorded as merged at {row['char3_cosine']}, but the frozen "
                    f"clusters.jsonl has no such cluster\n"
                    f"  recorded A {row['representative_title'][:70]}\n"
                    f"  recorded B {row['other_member_title'][:70]}"
                )
            continue
        members = [by_doc[doc_id] for doc_id in frozen_cluster.member_doc_ids]
        for index, first in enumerate(members):
            for second in members[index + 1 :]:
                reason = _reason(first, second, args.threshold)
                reasons[reason.rsplit("_0.", 1)[0]] += 1
                if shown < args.limit:
                    shown += 1
                    print(
                        f"{cluster_id} {reason}\n"
                        f"  A [{first.lang}/{first.source_key}] {first.title[:70]}\n"
                        f"  B [{second.lang}/{second.source_key}] {second.title[:70]}"
                    )
    print("\nreason histogram:", dict(reasons))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
