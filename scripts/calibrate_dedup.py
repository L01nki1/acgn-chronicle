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

from acgn_rag.config import load_settings
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
#: RUNBOOK §3.3 sweeps all four values. Gate evidence must cover every one of them: a sheet
#: that labels only 0.86 is not a sweep, even if that one value has 20 judgements.
REQUIRED_THRESHOLDS = tuple(f"{value:.2f}" for value in SWEEP_THRESHOLDS)

LABEL_COLUMNS = (
    "threshold",
    "pair_id",
    "channel",
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


def _pairs(
    clusters: list[Any], by_doc: dict[str, NormalizedItem], threshold: float
) -> list[dict[str, Any]]:
    """Member pairs of every non-singleton cluster, highest cosine first, channel-tagged.

    ``aux`` pairs (cosine >= threshold) are the ones the sweep's auxiliary channel decides:
    a threshold change moves them, so they are what the hand labels must calibrate.
    ``primary`` pairs (cosine < threshold) sit in the cluster only because the primary
    channel's compatibility floor (>= 1.0) is threshold-independent; they cannot
    discriminate one sweep value from another (E's #75 review, P3).
    """
    pairs: list[dict[str, Any]] = []
    for cluster in clusters:
        members = [by_doc[doc_id] for doc_id in cluster.member_doc_ids]
        for index, first in enumerate(members):
            for second in members[index + 1 :]:
                cosine = _default_item_similarity(first, second)
                pairs.append(
                    {
                        "cosine": cosine,
                        "channel": "aux" if cosine >= threshold else "primary",
                        "cluster_id": cluster.cluster_id,
                        "a": first,
                        "b": second,
                    }
                )
    pairs.sort(key=lambda row: (-row["cosine"], row["a"].doc_id, row["b"].doc_id))
    return pairs


def _sample(pairs: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    """Aux pairs first — the ``HEAD_ROWS`` highest cosines, then a spread up to the gate
    minimum — and an even spread of primary pairs for the remaining slots.

    Taking only the top rows would hand the labeller the easiest calls (identical syndicated
    headlines) and inflate precision at every threshold; the spread is where a threshold is
    actually decided. Primary pairs fill only what is left after the aux requirement: they
    carry policy precision, not threshold precision.
    """
    aux = [pair for pair in pairs if pair["channel"] == "aux"]
    primary = [pair for pair in pairs if pair["channel"] == "primary"]

    def spread(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
        if count <= 0:
            return []
        if len(rows) <= count:
            return list(rows)
        step = len(rows) / count
        return [rows[min(int(index * step), len(rows) - 1)] for index in range(count)]

    head = aux[:HEAD_ROWS]
    top_up = spread(aux[HEAD_ROWS:], max(PAIRS_PER_THRESHOLD - len(head), 0))
    primary_rows = spread(primary, max(size - len(head) - len(top_up), 0))
    return head + top_up + primary_rows


def _sheet_row(threshold: float, rank: int, pair: dict[str, Any]) -> dict[str, Any]:
    first: NormalizedItem = pair["a"]
    second: NormalizedItem = pair["b"]
    return {
        "threshold": f"{threshold:.2f}",
        "pair_id": f"{threshold:.2f}-{rank:02d}",
        "channel": pair["channel"],
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


def _score(path: Path, *, required: int = PAIRS_PER_THRESHOLD, exploratory: bool = False) -> int:
    """Score a completed sheet: aux precision per threshold, and whether it is gate evidence.

    Gate evidence requires **all four** RUNBOOK §3.3 sweep values and ``required`` distinct
    non-skip **aux** judgements for each (20 by default): aux pairs (cosine >= threshold) are
    the ones a threshold change actually moves, while primary pairs (cosine < threshold,
    merged by the primary channel's threshold-independent floor) are reported separately as
    policy precision and never count towards the gate. Repeated ``pair_id`` rows count once;
    conflicting decisions or channels for one pair are an error, not extra evidence.
    ``--exploratory`` or a requirement below 20 is explicitly non-gate exploration and never
    fails the gate.
    """
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

    by_threshold: dict[str, dict[str, Any]] = {}
    for row in rows:
        decision = row["decision"].strip().lower()
        if decision not in {"duplicate", "not_duplicate", "skip"}:
            print(f"{row['pair_id']}: unknown decision {row['decision']!r}", file=sys.stderr)
            return 2
        channel = (row.get("channel") or "").strip().lower()
        if channel not in {"aux", "primary"}:
            print(
                f"{row['pair_id']}: channel {row.get('channel')!r} is not aux/primary - "
                "regenerate the sheet with the current tooling (RUNBOOK §3.3)",
                file=sys.stderr,
            )
            return 2
        bucket = by_threshold.setdefault(row["threshold"], {"pairs": {}, "skipped": 0})
        if decision == "skip":
            bucket["skipped"] += 1
            continue
        pair_id = row["pair_id"]
        previous = bucket["pairs"].get(pair_id)
        if previous is not None and previous != (decision, channel):
            print(
                f"{pair_id}: conflicting decisions/channels {previous} vs "
                f"{(decision, channel)} for one pair",
                file=sys.stderr,
            )
            return 2
        bucket["pairs"][pair_id] = (decision, channel)

    gate_mode = not exploratory and required >= PAIRS_PER_THRESHOLD
    if exploratory:
        reason: str | None = "--exploratory"
    elif required < PAIRS_PER_THRESHOLD:
        reason = f"--required-judged {required} is below the gate minimum {PAIRS_PER_THRESHOLD}"
    else:
        reason = None
    report: dict[str, Any] = {
        "sheet": str(path),
        "mode": "gate" if gate_mode else "exploratory — not gate evidence",
        "mode_reason": reason,
        "required_judged_per_threshold": required,
        "required_thresholds": list(REQUIRED_THRESHOLDS),
        "thresholds": {},
    }
    missing = [value for value in REQUIRED_THRESHOLDS if value not in by_threshold]
    insufficient: list[str] = []
    for threshold in sorted(set(REQUIRED_THRESHOLDS) | set(by_threshold)):
        bucket = by_threshold.get(threshold, {"pairs": {}, "skipped": 0})
        pairs: dict[str, tuple[str, str]] = bucket["pairs"]
        aux = [value for value in pairs.values() if value[1] == "aux"]
        primary = [value for value in pairs.values() if value[1] == "primary"]
        aux_duplicate = sum(1 for value in aux if value[0] == "duplicate")
        aux_judged = len(aux)
        aux_precision = aux_duplicate / aux_judged if aux_judged else None
        primary_duplicate = sum(1 for value in primary if value[0] == "duplicate")
        primary_judged = len(primary)
        primary_precision = primary_duplicate / primary_judged if primary_judged else None
        enough = aux_judged >= required
        if not enough:
            insufficient.append(threshold)
        report["thresholds"][threshold] = {
            "judged_aux": aux_judged,
            "aux_duplicate": aux_duplicate,
            "aux_precision": None if aux_precision is None else round(aux_precision, 4),
            "judged_primary": primary_judged,
            "primary_duplicate": primary_duplicate,
            "primary_precision": None if primary_precision is None else round(primary_precision, 4),
            "skipped": bucket["skipped"],
            "insufficient": not enough,
        }
    report["missing_thresholds"] = missing
    report["gate"] = (
        "NOT_GATE" if not gate_mode else ("INSUFFICIENT" if insufficient or missing else "PASS")
    )
    if gate_mode and (insufficient or missing):
        detail = ", ".join(
            f"{t}: aux judged {report['thresholds'][t]['judged_aux']}/{required}"
            for t in insufficient
        )
        print(
            f"{path}: insufficient hand labels for gate reporting ({detail}); "
            f"all of {', '.join(REQUIRED_THRESHOLDS)} need {required} distinct non-skip AUX "
            "pairs (RUNBOOK §3.3), or pass --exploratory to score outside the gate",
            file=sys.stderr,
        )
    print(json.dumps(report, indent=2))
    if gate_mode and (insufficient or missing):
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="G1b dedup threshold sweep and label sheet")
    parser.add_argument("--snapshot", help="frozen snapshot id to sweep")
    parser.add_argument("--data-root", default=None, help="data root; default ACGN_DATA_DIR")
    parser.add_argument("--out", default="eval/dedup_g1b", help="output directory")
    parser.add_argument("--score", default=None, help="score a completed label sheet and exit")
    parser.add_argument(
        "--required-judged",
        type=int,
        default=20,
        help="minimum non-skip judgements per threshold for gate scoring (RUNBOOK §3.3)",
    )
    parser.add_argument(
        "--exploratory",
        action="store_true",
        help="score any sheet for exploration; the report is tagged as not gate evidence",
    )
    args = parser.parse_args(argv)

    if args.score:
        return _score(Path(args.score), required=args.required_judged, exploratory=args.exploratory)
    if not args.snapshot:
        print("--snapshot is required unless --score is used", file=sys.stderr)
        return 2

    data_root = Path(args.data_root) if args.data_root else load_settings().paths.root
    out = Path(args.out)
    items = _load(args.snapshot, data_root)
    sweep = _sweep(items)

    sheet: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for row in sweep:
        pairs = _pairs(row["_clusters"], row["_by_doc"], row["threshold"])
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
