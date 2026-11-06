"""Rule-based evaluation metrics (`docs/EVAL_PROTOCOL.md` §4-§8).

Owner: E | Spec: `docs/EVAL_PROTOCOL.md` §4-§8, issue #7, D's #30 ruling

The runner supplies one `CaseResult` per QA item. These functions are pure: they never call a
model, never touch the network and never decide an answer label — `answer_correctness` is a
human label; code only aggregates it and computes the mechanical flags.

Denominators (D's #30 ruling, 2026-10-31): `deduped_count` is provenance and the bound on what
the index covers; it is **not** a divisor. Every rate is computed over its own eligible item set
and prints that set's `n` beside the value:

- retrieval: answerable items whose gold url resolves in this snapshot (`gold_unavailable`
  items are excluded and counted; `gold_member_not_indexed` items are excluded and counted
  separately until the E/F/D representative-alias rule is agreed);
- correctness: answerable items actually graded (dropped items are excluded and counted);
- abstention: the QA set only — unanswerable items for the refusal rate, answerable items for
  the false-refusal rate; the calibration set is reported separately as in-sample.

Rules encoded here:
- retrieval hit = **any** returned `citation.url` is in `gold_evidence.urls`;
- URL comparison normalises only the fragment, a trailing slash and host case;
- retrieval metrics are reported for `all` / `en` / `zh`;
- a cited chunk must be a subset of the retrieved chunk ids.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

LANGS = ("en", "zh")
BLOCKS = ("all", *LANGS)
CORRECTNESS_LABELS = ("fully_correct", "partially_correct", "incorrect")


@dataclass(frozen=True)
class CaseResult:
    """One QA item after retrieval and (optionally) generation.

    `gold_unavailable` is set by the runner when the gold URL is absent from the snapshot.
    `gold_member_not_indexed` is set when the gold URL is a snapshot member whose cluster is
    represented in the index by another member; eligibility is pending the E/F/D rule.
    `dropped` is set when an item is excluded after adjudication (§5); it stays counted.
    `calibration` marks an item from `eval/abstain_calibration.jsonl`, never a reporting item.
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
    gold_member_not_indexed: bool = False
    gold_unresolved: bool = False
    dropped: bool = False
    calibration: bool = False
    answer_label: str | None = None
    abstained: bool | None = None


def normalize_url(url: str) -> str:
    """Normalise only what the protocol allows: fragment, trailing slash, host case."""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path.rstrip("/"), parts.query, ""))


def _gold_set(urls: Sequence[str]) -> set[str]:
    return {normalize_url(url) for url in urls}


def _rate(value: float | None, n: int) -> dict[str, float | int | None]:
    """A rate never travels without the size of the set it was computed on."""
    return {"value": value, "n": n}


def _mean(values: Sequence[float]) -> dict[str, float | int | None]:
    n = len(values)
    return _rate(sum(values) / n if n else None, n)


def _retrieved_at_k(case: CaseResult, k: int | None) -> tuple[str, ...]:
    if k is None:
        return case.retrieved_urls
    if k < 1:
        raise ValueError("k must be >= 1")
    return case.retrieved_urls[:k]


def hit_at_k(case: CaseResult, *, k: int | None = None) -> int:
    """1 if any retrieved URL within `k` is a gold URL, else 0. Not graded on answer text."""
    if case.gold_unavailable or case.gold_member_not_indexed:
        return 0
    gold = _gold_set(case.gold_urls)
    if not gold:
        return 0
    return int(any(normalize_url(url) in gold for url in _retrieved_at_k(case, k)))


def evidence_recall(case: CaseResult, *, k: int | None = None) -> float:
    """Distinct gold URLs retrieved within `k` / all gold URLs. Duplicates do not help."""
    if case.gold_unavailable or case.gold_member_not_indexed:
        return 0.0
    gold = _gold_set(case.gold_urls)
    if not gold:
        return 0.0
    found = {normalize_url(url) for url in _retrieved_at_k(case, k)} & gold
    return len(found) / len(gold)


def mrr_at_k(case: CaseResult, *, k: int | None = None) -> float:
    """Reciprocal rank of the first gold-URL hit within `k`; 0.0 when there is none."""
    if case.gold_unavailable or case.gold_member_not_indexed:
        return 0.0
    gold = _gold_set(case.gold_urls)
    if not gold:
        return 0.0
    for rank, url in enumerate(_retrieved_at_k(case, k), start=1):
        if normalize_url(url) in gold:
            return 1.0 / rank
    return 0.0


def _retrieval_eligible(case: CaseResult) -> bool:
    return (
        case.answerable
        and not case.gold_unavailable
        and not case.gold_member_not_indexed
        and not case.gold_unresolved
        and not case.dropped
        and not case.calibration
        and bool(case.gold_urls)
    )


def _blocks(cases: Sequence[CaseResult]) -> dict[str, list[CaseResult]]:
    return {
        "all": list(cases),
        "en": [case for case in cases if case.lang == "en"],
        "zh": [case for case in cases if case.lang == "zh"],
    }


def retrieval_metrics(cases: Sequence[CaseResult], *, k: int = 5) -> dict[str, Any]:
    """Per-language retrieval rates (each with its own `n`) plus the exclusion counters."""
    eligible = [case for case in cases if _retrieval_eligible(case)]
    excluded = [case for case in cases if not case.calibration]
    blocks = _blocks(eligible)
    return {
        "k": k,
        "hit": {
            key: _mean([hit_at_k(case, k=k) for case in block]) for key, block in blocks.items()
        },
        "mrr": {
            key: _mean([mrr_at_k(case, k=k) for case in block]) for key, block in blocks.items()
        },
        "evidence_recall": {
            key: _mean([evidence_recall(case, k=k) for case in block])
            for key, block in blocks.items()
        },
        "excluded": {
            "unanswerable": sum(1 for case in excluded if not case.answerable),
            "gold_unavailable": sum(1 for case in excluded if case.gold_unavailable),
            "gold_member_not_indexed": sum(1 for case in excluded if case.gold_member_not_indexed),
            "gold_unresolved": sum(1 for case in excluded if case.gold_unresolved),
            "missing_gold": sum(
                1
                for case in excluded
                if case.answerable
                and not case.gold_unavailable
                and not case.gold_member_not_indexed
                and not case.gold_urls
            ),
            "dropped": sum(1 for case in excluded if case.dropped),
        },
    }


def answer_correctness(cases: Sequence[CaseResult]) -> dict[str, Any]:
    """Aggregate the human labels; an unknown label is a bug, not a default."""
    graded = [
        case
        for case in cases
        if case.answer_label is not None and not case.dropped and not case.calibration
    ]
    counts = dict.fromkeys(CORRECTNESS_LABELS, 0)
    for case in graded:
        if case.answer_label not in CORRECTNESS_LABELS:
            raise ValueError(
                f"{case.qa_id}: unknown answer_label {case.answer_label!r}; "
                f"expected one of {CORRECTNESS_LABELS}"
            )
        counts[case.answer_label] += 1

    rates = {
        key: _mean([1.0 if case.answer_label == "fully_correct" else 0.0 for case in block])
        for key, block in _blocks(graded).items()
    }
    return {"counts": counts, "rates": rates}


def abstention_accuracy(
    cases: Sequence[CaseResult], *, threshold: float | None = None
) -> dict[str, Any]:
    """Refusal rate on unanswerable items and false-refusal rate on answerable items.

    Only QA-set items are reporting denominators. Calibration items are reported separately and
    labelled in-sample so they can never be mistaken for the result (§6).
    """
    reporting = [
        case
        for case in cases
        if not case.calibration
        and not case.dropped
        and not case.gold_unavailable
        and not case.gold_member_not_indexed
        and case.abstained is not None
    ]
    answerable = [case for case in reporting if case.answerable]
    unanswerable = [case for case in reporting if not case.answerable]

    false_refusals = sum(1 for case in answerable if case.abstained)
    correct_refusals = sum(1 for case in unanswerable if case.abstained)

    calibration = [case for case in cases if case.calibration and case.abstained is not None]
    calibration_correct = sum(
        1 for case in calibration if bool(case.abstained) is not case.answerable
    )

    return {
        "threshold": threshold,
        "refusal_rate": _rate(
            correct_refusals / len(unanswerable) if unanswerable else None, len(unanswerable)
        ),
        "false_refusal_rate": _rate(
            false_refusals / len(answerable) if answerable else None, len(answerable)
        ),
        "calibration_in_sample": _rate(
            calibration_correct / len(calibration) if calibration else None, len(calibration)
        ),
    }


def citation_containment(cases: Sequence[CaseResult]) -> dict[str, Any]:
    """Every non-abstained answer must cite only retrieved chunks, and cite at least one."""
    checked = [case for case in cases if case.abstained is False and not case.calibration]
    violations = sum(
        1
        for case in checked
        if not case.cited_chunk_ids
        or not set(case.cited_chunk_ids).issubset(case.retrieved_chunk_ids)
    )
    abstained = [case for case in cases if case.abstained is True and not case.calibration]
    abstained_violations = sum(1 for case in abstained if case.cited_chunk_ids)
    return {
        "containment_rate": _rate(1 - violations / len(checked) if checked else None, len(checked)),
        "violations": violations,
        "abstained_violations": abstained_violations,
    }


def aggregate(
    cases: Sequence[CaseResult],
    *,
    k: int = 5,
    deduped_count: int | None = None,
    indexed_chunks: int | None = None,
    threshold: float | None = None,
    provisional: bool = False,
) -> dict[str, Any]:
    """Full metrics object for `eval/runs/<run_id>/metrics.json`.

    `deduped_count` and `indexed_chunks` are provenance: they bound what the index covers and
    are never used as a rate denominator. The counts block describes the QA set only; calibration
    items are reported through `abstention.calibration_in_sample` and never inflate `qa_total`.
    """
    qa_cases = [case for case in cases if not case.calibration]
    return {
        "counts": {
            "qa_total": len(qa_cases),
            "answerable": sum(1 for case in qa_cases if case.answerable),
            "unanswerable": sum(1 for case in qa_cases if not case.answerable),
            "gold_unavailable": sum(1 for case in qa_cases if case.gold_unavailable),
            "gold_member_not_indexed": sum(1 for case in qa_cases if case.gold_member_not_indexed),
            "gold_unresolved": sum(1 for case in qa_cases if case.gold_unresolved),
            "dropped": sum(1 for case in qa_cases if case.dropped),
            "deduped_count": deduped_count,
            "indexed_chunks": indexed_chunks,
        },
        "retrieval": retrieval_metrics(cases, k=k),
        "correctness": answer_correctness(cases),
        "abstention": abstention_accuracy(cases, threshold=threshold),
        "citation": citation_containment(cases),
        "provisional": provisional,
    }
