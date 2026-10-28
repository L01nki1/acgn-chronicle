"""QA set loading and rule enforcement (`docs/EVAL_PROTOCOL.md` §2-§3).

Owner: D | Spec: EVAL_PROTOCOL §2 / §3

The protocol states the quotas in prose; this module makes them executable, so "the set is frozen" means
"it passed the checker" rather than "nobody objected". Two rules are easy to get wrong by hand and are
therefore enforced here:

  - type A must stay at or below one third of the total, or the set stops measuring time-sensitive
    questions and starts measuring repetition;
  - the abstention calibration set must be disjoint from the QA set, or the reported abstention number is
    fitted on the same items it is reported for and comes out optimistic by construction.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

QA_TYPES = ("A", "B", "C", "D")
# (minimum, maximum) per type; None means no upper bound
TYPE_QUOTAS: dict[str, tuple[int, int | None]] = {
    "A": (10, 12),
    "B": (8, 10),
    "C": (6, 8),
    "D": (8, None),
}
MIN_TOTAL = 30
MIN_UNANSWERABLE = 5
OVERLAP_COUNT = 10
CALIBRATION_SIZE = {"answerable": 20, "unanswerable": 10}
REQUIRED_FIELDS = (
    "qa_id",
    "type",
    "answerable",
    "question",
    "gold_answer",
    "gold_evidence",
    "key_entities",
    "window",
    "annotator",
    "verified_by",
)
QA_ID_PATTERN = re.compile(r"^[ABCD]-\d{3}$")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping blank lines. Raises on malformed JSON so a typo never passes silently."""
    items: list[dict[str, Any]] = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from exc
    return items


def _urls(item: dict[str, Any]) -> list[str]:
    evidence = item.get("gold_evidence")
    if not isinstance(evidence, dict):
        return []
    urls = evidence.get("urls")
    return [u for u in urls if isinstance(u, str)] if isinstance(urls, list) else []


def _check_item(item: dict[str, Any], index: int) -> list[str]:
    where = str(item.get("qa_id") or f"line {index + 1}")
    problems: list[str] = []

    missing = [f for f in REQUIRED_FIELDS if f not in item]
    if missing:
        return [f"{where}: missing field(s) {missing}"]

    qa_id = str(item["qa_id"])
    if not QA_ID_PATTERN.match(qa_id):
        problems.append(f"{where}: qa_id must look like A-007")
    if item["type"] not in QA_TYPES:
        problems.append(f"{where}: type must be one of {QA_TYPES}")
    elif QA_ID_PATTERN.match(qa_id) and qa_id[0] != item["type"]:
        problems.append(f"{where}: qa_id prefix disagrees with type {item['type']}")
    if not isinstance(item["answerable"], bool):
        problems.append(f"{where}: answerable must be true or false (never inferred from type)")
    if item["answerable"] is False and item["type"] != "D":
        problems.append(f"{where}: an unanswerable item must be type D")
    if item["answerable"] is True:
        urls = _urls(item)
        if not urls:
            problems.append(f"{where}: answerable item needs at least one gold_evidence url")
        for url in urls:
            if not url.startswith("https://"):
                problems.append(f"{where}: gold url must be https: {url}")
        if not item["key_entities"]:
            problems.append(f"{where}: answerable item needs key_entities to grade against")
    window = item["window"]
    if not (isinstance(window, list) and len(window) == 2 and window[0] <= window[1]):
        problems.append(f"{where}: window must be [start, end] with start <= end")
    if item["annotator"] == item["verified_by"]:
        problems.append(
            f"{where}: annotator and verified_by must differ (self-review is not review)"
        )
    return problems


def validate(
    items: list[dict[str, Any]],
    *,
    calibration: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Return every rule violation. An empty list means the set may be frozen."""
    problems: list[str] = []
    for index, item in enumerate(items):
        problems.extend(_check_item(item, index))

    ids = [item.get("qa_id") for item in items]
    duplicates = sorted({str(i) for i, n in Counter(ids).items() if n > 1})
    if duplicates:
        problems.append(f"duplicate qa_id(s): {duplicates}")

    total = len(items)
    if total < MIN_TOTAL:
        problems.append(f"total {total} < {MIN_TOTAL} required by the protocol")

    counts = Counter(item.get("type") for item in items if item.get("type") in QA_TYPES)
    for qa_type, (low, high) in TYPE_QUOTAS.items():
        got = counts.get(qa_type, 0)
        if got < low:
            problems.append(f"type {qa_type}: {got} items, need >= {low}")
        if high is not None and got > high:
            problems.append(f"type {qa_type}: {got} items, need <= {high}")

    if total and counts.get("A", 0) > total / 3:
        problems.append(
            f"type A is {counts.get('A', 0)}/{total} = more than one third; the set stops measuring "
            "time-sensitive questions and starts measuring repetition"
        )

    unanswerable = sum(1 for item in items if item.get("answerable") is False)
    if unanswerable < MIN_UNANSWERABLE:
        problems.append(f"only {unanswerable} unanswerable item(s), need >= {MIN_UNANSWERABLE}")

    overlap = sum(1 for item in items if item.get("overlap") is True)
    if overlap != OVERLAP_COUNT:
        problems.append(
            f"{overlap} items marked overlap, need exactly {OVERLAP_COUNT} so agreement is measurable"
        )

    if calibration is not None:
        problems.extend(_check_calibration(items, calibration))
    return problems


def _check_calibration(items: list[dict[str, Any]], calibration: list[dict[str, Any]]) -> list[str]:
    problems: list[str] = []
    answerable = sum(1 for item in calibration if item.get("answerable") is True)
    unanswerable = sum(1 for item in calibration if item.get("answerable") is False)
    if answerable != CALIBRATION_SIZE["answerable"]:
        problems.append(
            f"calibration: {answerable} answerable, need {CALIBRATION_SIZE['answerable']} "
            "(the false-refusal rate cannot be measured without them)"
        )
    if unanswerable != CALIBRATION_SIZE["unanswerable"]:
        problems.append(
            f"calibration: {unanswerable} unanswerable, need {CALIBRATION_SIZE['unanswerable']}"
        )

    qa_questions = {" ".join(str(item.get("question", "")).lower().split()) for item in items}
    shared = [
        item.get("question")
        for item in calibration
        if " ".join(str(item.get("question", "")).lower().split()) in qa_questions
    ]
    if shared:
        problems.append(
            f"calibration shares {len(shared)} question(s) with the QA set: choosing the threshold on "
            "the items you then report on makes the reported rate optimistic, by construction"
        )
    return problems


def summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts for the report and for a quick sanity check before freezing."""
    counts = Counter(item.get("type") for item in items)
    return {
        "total": len(items),
        "by_type": {qa_type: counts.get(qa_type, 0) for qa_type in QA_TYPES},
        "unanswerable": sum(1 for item in items if item.get("answerable") is False),
        "answerable": sum(1 for item in items if item.get("answerable") is True),
        "overlap": sum(1 for item in items if item.get("overlap") is True),
        "urls": sum(len(_urls(item)) for item in items),
    }


__all__ = [
    "CALIBRATION_SIZE",
    "MIN_TOTAL",
    "MIN_UNANSWERABLE",
    "OVERLAP_COUNT",
    "QA_TYPES",
    "REQUIRED_FIELDS",
    "TYPE_QUOTAS",
    "load_jsonl",
    "summary",
    "validate",
]
