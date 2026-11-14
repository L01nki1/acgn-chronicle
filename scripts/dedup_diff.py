"""Reproduce one threshold of the recorded dedup sweep from the frozen snapshot.

Owner: F (WP2) | Spec: docs/RUNBOOK.md §3.3, docs/INGEST_CONTRACT.md §2

Historical note: the sweep table first recorded in `eval/snapshot_evidence.json` was stale
(36,168 items / 36,127 clusters, from a different clustering run than the one that published
the snapshot); PR #76 re-issued it from the frozen snapshot (36,167 items / 36,121 clusters)
on 2026-11-12. This script is the reproduction evidence behind that correction — and the
standing way to check any future sweep table. It re-runs `cluster_items` on the frozen
snapshot at the requested threshold and compares the reproduction against the record
**strictly**:

* `cluster_id_set_identical` — the non-singleton cluster ids match exactly;
* `member_sets_identical` — and so does every cluster's member set;
* `item_counts_match` — item and cluster counts agree;
* `matches_snapshot` — all three, i.e. the record reproduces at the pair level, not merely
  in aggregate counts.

    python scripts/dedup_diff.py --snapshot 20261105-d797b5f0ae8e --threshold 0.86 \
        --data-root data --out eval/dedup_g1b/findings.json

For recorded sample pairs the current policy would not merge, it prints the reason
`dedup._pair_merge_score`'s own verdict implies (time gap, corporate guard, media_ids,
event_type, language, headline numbers, cosine) — the verdict comes from that function
itself, so a policy change moves the diff with the code.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from acgn_rag.core.ingest.dedup import (
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_WINDOW_HOURS,
    DedupPolicy,
    _default_item_similarity,
    _has_conflicting_headline_numbers,
    _pair_merge_score,
    char_ngram_cosine,
    cluster_items,
)
from acgn_rag.core.ingest.normalize import NormalizedItem
from acgn_rag.core.ingest.snapshot import iter_snapshot_items, load_snapshot_clusters
from acgn_rag.core.store.payload import iso_to_epoch

#: Mirrors the inline title-similarity floor in `dedup._pair_merge_score`'s primary branch;
#: dedup does not export it yet (E's #75 review, P6). Pinned by the consistency test.
_PRIMARY_TITLE_FLOOR = 0.30


def _reason(first: NormalizedItem, second: NormalizedItem, threshold: float) -> str:
    """Why the current policy would not merge this pair, per `_pair_merge_score`'s verdict.

    The verdict comes from `dedup._pair_merge_score` itself, so a policy change moves the
    diff with the code; the sub-reasons only label which rejection fired, in that function's
    own branch order.
    """
    score = _pair_merge_score(
        first,
        second,
        ts_a=iso_to_epoch(first.published_at),
        ts_b=iso_to_epoch(second.published_at),
        max_gap_s=DEFAULT_WINDOW_HOURS * 3600,
        sim_threshold=threshold,
        similarity_fn=_default_item_similarity,
    )
    if score is not None:
        if score >= 1.0:
            return "primary_channel_should_merge"
        if score >= threshold:
            return "aux_above_threshold"
        return f"aux_cosine_{score:.4f}_below_threshold"
    # Rejection sub-reasons mirror _pair_merge_score's branch order.
    gap = abs(iso_to_epoch(first.published_at) - iso_to_epoch(second.published_at))
    if gap > DEFAULT_WINDOW_HOURS * 3600:
        return f"time_gap_{gap // 3600}h"
    if first.ingest_category == "corporate" or second.ingest_category == "corporate":
        return "corporate_guard"
    media_a, media_b = set(first.media_ids), set(second.media_ids)
    if media_a and media_b and media_a != media_b:
        return "media_ids_differ"
    if (
        first.event_type is not None
        and second.event_type is not None
        and first.event_type != second.event_type
    ):
        return "event_type_differs"
    if (
        media_a
        and media_a == media_b
        and first.event_type is not None
        and first.event_type == second.event_type
        and first.lang == second.lang
    ):
        if _has_conflicting_headline_numbers(first.title, second.title):
            return "primary_channel_headline_numbers"
        title_sim = char_ngram_cosine(first.title, second.title, n=3)
        if title_sim < _PRIMARY_TITLE_FLOOR:
            return f"primary_channel_title_sim_{title_sim:.2f}"
    if first.lang != second.lang:
        return "language_differs"
    cosine = _default_item_similarity(first, second)
    return f"aux_cosine_{cosine:.4f}_below_threshold"


def _compare(
    frozen: list[Any],
    reproduced: list[Any],
    *,
    frozen_items: int,
    reproduced_items: int,
) -> dict[str, Any]:
    """The strict cluster-level comparison: ids, member sets and item counts.

    Aggregate counts alone cannot say "reproduces" (E's #75 review, P2): a clustering with
    the same items/clusters/non-singleton counts but a different partition must compare as
    a mismatch, so `matches_snapshot` requires the cluster-id set, every member set and the
    item counts to agree.
    """
    frozen_multi = {c.cluster_id: c for c in frozen if len(c.member_doc_ids) > 1}
    reproduced_multi = {c.cluster_id: c for c in reproduced if len(c.member_doc_ids) > 1}
    frozen_ids = set(frozen_multi)
    reproduced_ids = set(reproduced_multi)
    cluster_id_set_identical = frozen_ids == reproduced_ids
    member_sets_identical = cluster_id_set_identical and all(
        set(frozen_multi[cid].member_doc_ids) == set(reproduced_multi[cid].member_doc_ids)
        for cid in frozen_ids
    )
    item_counts_match = (
        frozen_items == reproduced_items
        and len(frozen) == len(reproduced)
        and len(frozen_multi) == len(reproduced_multi)
    )
    return {
        "cluster_id_set_identical": cluster_id_set_identical,
        "member_sets_identical": member_sets_identical,
        "item_counts_match": item_counts_match,
        "matches_snapshot": cluster_id_set_identical
        and member_sets_identical
        and item_counts_match,
        "clusters_only_in_snapshot": len(frozen_ids - reproduced_ids),
        "clusters_only_in_reproduction": len(reproduced_ids - frozen_ids),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diff a recorded dedup sweep against the snapshot")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD)
    parser.add_argument("--evidence", default="eval/snapshot_evidence.json")
    parser.add_argument("--out", default=None, help="write the machine-readable report JSON here")
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
    }
    report.update(
        _compare(
            frozen=frozen,
            reproduced=result.clusters,
            frozen_items=len(items),
            reproduced_items=len(result.items),
        )
    )

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
    report["reason_histogram"] = dict(reasons)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
