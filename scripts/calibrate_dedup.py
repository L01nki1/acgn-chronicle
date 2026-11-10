"""G1b dedup calibration: the 0.82 / 0.84 / 0.86 / 0.88 sweep and its hand-label sheet.

Owner: F (WP2) | Spec: docs/RUNBOOK.md §3.3, docs/INGEST_CONTRACT.md §2, docs/DECISIONS.md 2026-11-05

RUNBOOK §3.3 asks for three things before the 0.86 threshold freezes:

1. the sweep on the **frozen corpus** (cluster counts per threshold),
2. **20 hand-checked pairs per value**, and
3. one line stating what the threshold optimizes.

This script produces (1) and the sheet for (2). It never invents a label: the ``decision``
column stays empty until a human reads the pair, and ``--score`` refuses to compute anything
until every row is labelled.

    python scripts/calibrate_dedup.py --snapshot 20261105-d797b5f0ae8e --out eval/dedup_g1b
    # ... label eval/dedup_g1b/label_sheet.csv by hand ...
    python scripts/calibrate_dedup.py --score eval/dedup_g1b/label_sheet.csv

The sweep is deterministic: ``cluster_items`` sorts by ``doc_id`` and the auxiliary channel is a
character-trigram cosine, so the same snapshot yields the same numbers on any machine.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

from acgn_rag.core.ingest.dedup import (
    SWEEP_THRESHOLDS,
    DedupPolicy,
    _default_item_similarity,
    cluster_items,
)
from acgn_rag.core.ingest.normalize import NormalizedItem
from acgn_rag.core.ingest.snapshot import iter_snapshot_items

#: Pairs sampled per threshold (RUNBOOK §3.3: "20 hand-checked pairs per value").
PAIRS_PER_THRESHOLD = 20
#: Rows written to the sheet per threshold: the required 20 plus 10 extra candidates, so a
#: pair the labeller cannot judge (dead link, truncated body) can be skipped without a rerun.
SHEET_ROWS = 30
#: Rows of the sheet taken from the top of the cosine range; the rest are spread evenly.
HEAD_ROWS = 10

LABEL_COLUMNS = (
    "threshold",
    "pair_id",
    "cosine",
    "cluster_id",
    "same_language",
    "event_type_a",
    "event_type_b",
    "media_ids_a",
    "media_ids_b",
    "title_a",
    "title_b",
    "url_a",
    "url_b",
    "decision",
    "note",
)


def _load(snapshot: str, data_root: Path) -> list[NormalizedItem]:
    """Every normalized member row of the frozen snapshot (items.jsonl, not representatives)."""
    started = time.monotonic()
    items = list(iter_snapshot_items(snapshot, data_root=data_root, verify=True))
    print(f"loaded {len(items)} items in {time.monotonic() - started:.1f}s", file=sys.stderr)
    return items


def _sweep(items: list[NormalizedItem]) -> list[dict[str, Any]]:
    """Cluster counts per threshold, plus the non-singleton clusters at each value."""
    rows: list[dict[str, Any]] = []
    by_doc = {item.doc_id: item for item in items}
    for threshold in SWEEP_THRESHOLDS:
        started = time.monotonic()
        result = cluster_items(
            items, policy=DedupPolicy(thresholds={"same_lang_cosine": float(threshold)})
        )
        multi = [c for c in result.clusters if len(c.member_doc_ids) > 1]
        rows.append(
            {
                "threshold": float(threshold),
                "item_count": len(result.items),
                "cluster_count": len(result.clusters),
                "non_singleton_clusters": len(multi),
                "collapsed": len(result.items) - len(result.clusters),
                "largest_cluster": max((len(c.member_doc_ids) for c in multi), default=0),
                "seconds": round(time.monotonic() - started, 1),
                "_clusters": multi,
                "_by_doc": by_doc,
            }
        )
        print(
            f"threshold {threshold}: clusters={len(result.clusters)} "
            f"non_singleton={len(multi)} collapsed={len(result.items) - len(result.clusters)} "
            f"({rows[-1]['seconds']}s)",
            file=sys.stderr,
        )
    return rows


def _pairs(clusters: list[Any], by_doc: dict[str, NormalizedItem]) -> list[dict[str, Any]]:
    """Member pairs of every non-singleton cluster, highest cosine first."""
    pairs: list[dict[str, Any]] = []
    for cluster in clusters:
        members = [by_doc[doc_id] for doc_id in cluster.member_doc_ids]
        for index, first in enumerate(members):
            for second in members[index + 1 :]:
                pairs.append(
                    {
                        "cosine": _default_item_similarity(first, second),
                        "cluster_id": cluster.cluster_id,
                        "a": first,
                        "b": second,
                    }
                )
    pairs.sort(key=lambda row: (-row["cosine"], row["a"].doc_id, row["b"].doc_id))
    return pairs


def _sample(pairs: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    """``HEAD_ROWS`` highest cosines plus an evenly spaced spread over the rest.

    Taking only the top rows would hand the labeller the easiest calls (identical syndicated
    headlines) and inflate precision at every threshold. The spread is where a threshold is
    actually decided: the pairs that one value merges and the next one does not.
    """
    head = pairs[:HEAD_ROWS]
    rest = pairs[HEAD_ROWS:]
    remaining = max(size - len(head), 0)
    if not rest or remaining == 0:
        return head
    if len(rest) <= remaining:
        return head + rest
    step = len(rest) / remaining
    return head + [rest[min(int(index * step), len(rest) - 1)] for index in range(remaining)]


def _sheet_row(threshold: float, rank: int, pair: dict[str, Any]) -> dict[str, Any]:
    first: NormalizedItem = pair["a"]
    second: NormalizedItem = pair["b"]
    return {
        "threshold": f"{threshold:.2f}",
        "pair_id": f"{threshold:.2f}-{rank:02d}",
        "cosine": f"{pair['cosine']:.4f}",
        "cluster_id": pair["cluster_id"],
        "same_language": "yes" if first.lang == second.lang else "no",
        "event_type_a": first.event_type or "",
        "event_type_b": second.event_type or "",
        "media_ids_a": " ".join(str(value) for value in first.media_ids),
        "media_ids_b": " ".join(str(value) for value in second.media_ids),
        "title_a": first.title,
        "title_b": second.title,
        "url_a": first.url,
        "url_b": second.url,
        "decision": "",
        "note": "",
    }


def _write_sheet(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _score(path: Path) -> int:
    """Score a completed sheet: precision per threshold, and the value that maximizes it."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        print(f"{path}: no rows", file=sys.stderr)
        return 2
    unlabelled = [row["pair_id"] for row in rows if not (row.get("decision") or "").strip()]
    if unlabelled:
        print(
            f"{path}: {len(unlabelled)} unlabelled row(s), first: {unlabelled[0]}. "
            "Every row needs duplicate / not_duplicate / skip before scoring.",
            file=sys.stderr,
        )
        return 2

    by_threshold: dict[str, dict[str, int]] = {}
    for row in rows:
        decision = row["decision"].strip().lower()
        if decision not in {"duplicate", "not_duplicate", "skip"}:
            print(f"{row['pair_id']}: unknown decision {row['decision']!r}", file=sys.stderr)
            return 2
        bucket = by_threshold.setdefault(row["threshold"], {"duplicate": 0, "not_duplicate": 0})
        if decision != "skip":
            bucket[decision] += 1

    report: dict[str, Any] = {"sheet": str(path), "thresholds": {}}
    for threshold in sorted(by_threshold):
        counts = by_threshold[threshold]
        judged = counts["duplicate"] + counts["not_duplicate"]
        precision = counts["duplicate"] / judged if judged else None
        report["thresholds"][threshold] = {
            "judged": judged,
            "duplicate": counts["duplicate"],
            "not_duplicate": counts["not_duplicate"],
            "precision": None if precision is None else round(precision, 4),
        }
    print(json.dumps(report, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="G1b dedup threshold sweep and label sheet")
    parser.add_argument("--snapshot", help="frozen snapshot id to sweep")
    parser.add_argument("--data-root", default=None, help="data root; default ACGN_DATA_DIR")
    parser.add_argument("--out", default="eval/dedup_g1b", help="output directory")
    parser.add_argument("--score", default=None, help="score a completed label sheet and exit")
    args = parser.parse_args(argv)

    if args.score:
        return _score(Path(args.score))
    if not args.snapshot:
        print("--snapshot is required unless --score is used", file=sys.stderr)
        return 2

    data_root = Path(args.data_root) if args.data_root else Path("data")
    out = Path(args.out)
    items = _load(args.snapshot, data_root)
    sweep = _sweep(items)

    sheet: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for row in sweep:
        pairs = _pairs(row["_clusters"], row["_by_doc"])
        summary.append(
            {
                "threshold": row["threshold"],
                "item_count": row["item_count"],
                "cluster_count": row["cluster_count"],
                "non_singleton_clusters": row["non_singleton_clusters"],
                "collapsed": row["collapsed"],
                "largest_cluster": row["largest_cluster"],
                "member_pairs": len(pairs),
            }
        )
        for rank, pair in enumerate(_sample(pairs, SHEET_ROWS), 1):
            sheet.append(_sheet_row(row["threshold"], rank, pair))

    out.mkdir(parents=True, exist_ok=True)
    (out / "sweep.json").write_text(
        json.dumps(
            {
                "snapshot_id": args.snapshot,
                "policy": "entity-event-72h+agglomerative-complete:char3-tf-cosine-v2",
                "thresholds": summary,
                "note": "cluster counts from cluster_items on the frozen corpus; hand labels pending",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_sheet(sheet, out / "label_sheet.csv")
    print(f"wrote {out / 'sweep.json'} and {out / 'label_sheet.csv'} ({len(sheet)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
