"""Apply human answer labels to a run archive and write the correctness block into metrics.json.

Owner: D (WP5) | Spec: docs/EVAL_PROTOCOL.md 5

The runner always writes an empty `labels.jsonl`, and nothing ever read it back, so correctness
stayed `null` no matter how much labelling was done. This closes that loop:

    .venv/Scripts/python.exe scripts/score_labels.py --run evalout/20261120-qd-on-r2

It reads `answers.jsonl` + the QA set + `labels.jsonl`, rebuilds the same `CaseResult` rows the
runner builds (same eligibility rules), and rewrites only the `correctness` and
`reporting.correctness` blocks of `metrics.json`. Everything else in that file is left byte-
identical, so a re-run of the pipeline never has to be repeated to record a human judgement.

`answer_correctness` itself stays the only thing that computes the numbers - this script does
not duplicate the aggregation, it only supplies `answer_label`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from acgn_rag.core.eval import metrics

CORRECTNESS_LABELS = ("fully_correct", "partially_correct", "incorrect")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def load_labels(path: Path) -> dict[str, str]:
    """qa_id -> label, validating the vocabulary so a typo cannot pass as a judgement."""
    labels: dict[str, str] = {}
    problems: list[str] = []
    for row in read_jsonl(path):
        qa_id = str(row.get("qa_id") or "")
        label = row.get("label")
        if not qa_id:
            problems.append("a label row has no qa_id")
            continue
        if label is None:
            continue  # exported but not yet judged
        if label not in CORRECTNESS_LABELS:
            problems.append(f"{qa_id}: unknown label {label!r}")
            continue
        if qa_id in labels:
            problems.append(f"{qa_id}: labelled twice")
            continue
        labels[qa_id] = label
    if problems:
        raise ValueError("labels.jsonl is not usable:\n  " + "\n  ".join(problems))
    return labels


def resolve(gold: dict[str, dict[str, Any]], qa_id: str) -> dict[str, Any] | None:
    """Find the QA row for an archived id, tolerating the `delivery-` namespace rename."""
    for candidate in (qa_id, f"delivery-{qa_id}", qa_id.removeprefix("delivery-")):
        if candidate in gold:
            return gold[candidate]
    return None


def build_cases(
    answers: list[dict[str, Any]],
    qa: dict[str, dict[str, Any]],
    labels: dict[str, str],
) -> tuple[list[metrics.CaseResult], list[str]]:
    """Mirror the runner's CaseResult construction for the fields correctness depends on."""
    cases: list[metrics.CaseResult] = []
    unresolved: list[str] = []
    for answer in answers:
        qa_id = str(answer["qa_id"])
        item = resolve(qa, qa_id)
        if item is None:
            unresolved.append(qa_id)
            continue
        evidence = item.get("gold_evidence") or {}
        cases.append(
            metrics.CaseResult(
                qa_id=str(item["qa_id"]),
                lang=str(item.get("lang") or "en"),
                answerable=bool(item.get("answerable", True)),
                gold_urls=tuple(evidence.get("urls") or ()),
                retrieved_urls=tuple(answer.get("retrieved_urls") or ()),
                retrieved_chunk_ids=tuple(answer.get("retrieved_chunk_ids") or ()),
                cited_chunk_ids=tuple(answer.get("cited_chunk_ids") or ()),
                dropped=bool(item.get("dropped", False)),
                calibration=bool(item.get("calibration", False)),
                answer_label=labels.get(str(item["qa_id"])) or labels.get(qa_id),
                abstained=bool(answer.get("abstained")),
            )
        )
    return cases, unresolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply human answer labels to a run archive")
    parser.add_argument("--run", required=True, help="run archive directory")
    parser.add_argument("--qa", default="eval/qa_set_delivery.jsonl")
    parser.add_argument("--dry-run", action="store_true", help="print the result, write nothing")
    args = parser.parse_args(argv)

    run_dir = Path(args.run)
    answers_path = run_dir / "answers.jsonl"
    metrics_path = run_dir / "metrics.json"
    labels_path = run_dir / "labels.jsonl"
    if not answers_path.is_file() or not metrics_path.is_file():
        print(f"{run_dir}: needs answers.jsonl and metrics.json", file=sys.stderr)
        return 2

    answers = read_jsonl(answers_path)
    qa = {str(row["qa_id"]): row for row in read_jsonl(Path(args.qa))}
    _probe, missing_qa = build_cases(answers, qa, {})
    if missing_qa:
        print(
            f"{len(missing_qa)} answer row(s) are not in {args.qa} (first: {missing_qa[:3]}). "
            "The archive may predate a QA-set rename; pass the set it was run against.",
            file=sys.stderr,
        )
        return 2

    try:
        labels = load_labels(labels_path)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    cases = build_cases(answers, qa, labels)[0]
    correctness = metrics.answer_correctness(cases)

    document = json.loads(metrics_path.read_text(encoding="utf-8"))
    document["correctness"] = correctness
    document.setdefault("reporting", {})["correctness"] = {
        "available": correctness["rates"]["all"]["value"] is not None,
        "n": correctness["rates"]["all"]["n"],
    }

    graded = correctness["rates"]["all"]["n"]
    counted = sum(correctness["counts"].values())
    print(f"{run_dir.name}: {len(labels)} label(s) read, {counted} graded, {graded} in the rate")
    print("  counts:", correctness["counts"])
    for key in ("all", "en", "zh"):
        block = correctness["rates"].get(key) or {}
        value = block.get("value")
        print(f"  rate[{key}] = {'n/a' if value is None else f'{value:.4f}'}  (n={block.get('n')})")
    unanswered = [
        a["qa_id"] for a in answers if not a.get("abstained") and str(a["qa_id"]) not in labels
    ]
    if unanswered:
        print(
            f"  NOT yet labelled ({len(unanswered)}): {', '.join(unanswered[:8])}"
            + (" ..." if len(unanswered) > 8 else "")
        )

    if args.dry_run:
        print("dry run: metrics.json not written")
        return 0
    metrics_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {metrics_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
