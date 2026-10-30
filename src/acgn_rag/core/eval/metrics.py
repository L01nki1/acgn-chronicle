"""Rule-based evaluation metrics (`docs/EVAL_PROTOCOL.md` §4-§8).

Owner: E | Spec: `docs/EVAL_PROTOCOL.md` §4-§8, issue #7

The runner supplies one `CaseResult` per QA item. These functions are pure: they never call a
model, never touch the network and never decide an answer label — `answer_correctness` is a
human label; code only aggregates it and computes the mechanical flags.

Rules encoded here:
- retrieval hit = **any** returned `citation.url` is in `gold_evidence.urls`;
- URL comparison normalises only the fragment, a trailing slash and host case;
- `gold_unavailable` items are excluded from denominators and counted;
- retrieval metrics are reported for `all` / `en` / `zh`;
- abstention reports refusal rate (unanswerable) and false-refusal rate (answerable);
- a cited chunk must be a subset of the retrieved chunk ids.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

LANGS = ("en", "zh")
CORRECTNESS_LABELS = ("fully_correct", "partially_correct", "incorrect")


@dataclass(frozen=True)
class CaseResult:
    """One QA item after retrieval and (optionally) generation.

    `gold_unavailable` is set by the runner when the gold URL is absent from the snapshot.
    `answer_label` and `abstained` stay `None` until a human labels the run.
    """

    qa_id: str
    lang: str
    answerable: bool
    gold_urls: tuple[str, ...]
    retrieved_urls: tuple[str, ...]
    retrieved_chunk_ids: tuple[str, ...] = ()
    cited_chunk_ids: tuple[str, ...] = ()
    gold_unavailable: bool = False
    answer_label: str | None = None
    abstained: bool | None = None


def normalize_url(url: str) -> str:
    """Normalise only what the protocol allows: fragment, trailing slash, host case."""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path.rstrip("/"), parts.query, ""))


def _gold_set(urls: Sequence[str]) -> set[str]:
    return {normalize_url(url) for url in urls}


def hit_at_k(case: CaseResult) -> int:
    """1 if any retrieved URL is a gold URL, else 0. Not graded on answer text."""
    if case.gold_unavailable:
        return 0
    gold = _gold_set(case.gold_urls)
    if not gold:
        return 0
    return int(any(normalize_url(url) in gold for url in case.retrieved_urls))


def evidence_recall(case: CaseResult) -> float:
    """Distinct gold URLs retrieved / all gold URLs. Distinct: duplicates do not help."""
    if case.gold_unavailable:
        return 0.0
    gold = _gold_set(case.gold_urls)
    if not gold:
        return 0.0
    found = {normalize_url(url) for url in case.retrieved_urls} & gold
    return len(found) / len(gold)


def mrr_at_k(case: CaseResult) -> float:
    """Reciprocal rank of the first gold-URL hit; 0.0 when there is none."""
    if case.gold_unavailable:
        return 0.0
    gold = _gold_set(case.gold_urls)
    if not gold:
        return 0.0
    for rank, url in enumerate(case.retrieved_urls, start=1):
        if normalize_url(url) in gold:
            return 1.0 / rank
    return 0.0


def _retrieval_eligible(case: CaseResult) -> bool:
    return case.answerable and not case.gold_unavailable and bool(case.gold_urls)


def _retrieval_block(cases: Sequence[CaseResult]) -> dict[str, float | int | None]:
    n = len(cases)
    if n == 0:
        return {"n": 0, "retrieval_hit_rate": None, "evidence_recall": None, "mrr": None}
    return {
        "n": n,
        "retrieval_hit_rate": sum(hit_at_k(case) for case in cases) / n,
        "evidence_recall": sum(evidence_recall(case) for case in cases) / n,
        "mrr": sum(mrr_at_k(case) for case in cases) / n,
    }


def retrieval_metrics(cases: Sequence[CaseResult]) -> dict[str, Any]:
    """Per-language retrieval blocks plus the excluded-item counts."""
    eligible = [case for case in cases if _retrieval_eligible(case)]
    blocks = {key: _retrieval_block(block) for key, block in _blocks(eligible)}
    return {
        "retrieval_hit_rate": {key: block["retrieval_hit_rate"] for key, block in blocks.items()},
        "evidence_recall": {key: block["evidence_recall"] for key, block in blocks.items()},
        "mrr": {key: block["mrr"] for key, block in blocks.items()},
        "n": {key: block["n"] for key, block in blocks.items()},
        "excluded": {
            "unanswerable": sum(1 for case in cases if not case.answerable),
            "gold_unavailable": sum(1 for case in cases if case.gold_unavailable),
            "missing_gold": sum(
                1
                for case in cases
                if case.answerable and not case.gold_unavailable and not case.gold_urls
            ),
        },
    }


def _blocks(cases: Sequence[CaseResult]) -> list[tuple[str, list[CaseResult]]]:
    return [
        ("all", list(cases)),
        ("en", [case for case in cases if case.lang == "en"]),
        ("zh", [case for case in cases if case.lang == "zh"]),
    ]


def answer_correctness(cases: Sequence[CaseResult]) -> dict[str, Any]:
    """Aggregate the human labels; an unknown label is a bug, not a default."""
    counts = dict.fromkeys(CORRECTNESS_LABELS, 0)
    labeled = 0
    for case in cases:
        if case.answer_label is None:
            continue
        if case.answer_label not in CORRECTNESS_LABELS:
            raise ValueError(
                f"{case.qa_id}: unknown answer_label {case.answer_label!r}; "
                f"expected one of {CORRECTNESS_LABELS}"
            )
        counts[case.answer_label] += 1
        labeled += 1
    return {
        "counts": counts,
        "labeled": labeled,
        "fully_correct_rate": counts["fully_correct"] / labeled if labeled else None,
    }


def abstention_accuracy(cases: Sequence[CaseResult]) -> dict[str, Any]:
    """Refusal rate on unanswerable items and false-refusal rate on answerable items."""
    usable = [case for case in cases if not case.gold_unavailable and case.abstained is not None]
    answerable = [case for case in usable if case.answerable]
    unanswerable = [case for case in usable if not case.answerable]

    false_refusals = sum(1 for case in answerable if case.abstained)
    correct_refusals = sum(1 for case in unanswerable if case.abstained)

    return {
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "false_refusals": false_refusals,
        "correct_refusals": correct_refusals,
        "false_refusal_rate": false_refusals / len(answerable) if answerable else None,
        "refusal_rate": correct_refusals / len(unanswerable) if unanswerable else None,
    }


def citation_containment(cases: Sequence[CaseResult]) -> dict[str, Any]:
    """Every cited chunk must have been retrieved for that query."""
    checked = [case for case in cases if case.cited_chunk_ids and case.retrieved_chunk_ids]
    violations = sum(
        1 for case in checked if not set(case.cited_chunk_ids).issubset(case.retrieved_chunk_ids)
    )
    return {"checked": len(checked), "violations": violations}


def aggregate(cases: Sequence[CaseResult]) -> dict[str, Any]:
    """Full metrics object for `eval/runs/<run_id>/metrics.json`."""
    retrieval = retrieval_metrics(cases)
    return {
        "retrieval_hit_rate": retrieval["retrieval_hit_rate"],
        "evidence_recall": retrieval["evidence_recall"],
        "mrr": retrieval["mrr"],
        "n": retrieval["n"],
        "excluded": retrieval["excluded"],
        "answer_correctness": answer_correctness(cases),
        "abstention_accuracy": abstention_accuracy(cases),
        "citation_containment": citation_containment(cases),
    }


__all__ = [
    "CORRECTNESS_LABELS",
    "LANGS",
    "CaseResult",
    "abstention_accuracy",
    "aggregate",
    "answer_correctness",
    "citation_containment",
    "evidence_recall",
    "hit_at_k",
    "mrr_at_k",
    "normalize_url",
    "retrieval_metrics",
]
