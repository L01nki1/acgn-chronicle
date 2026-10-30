"""Tests for the rule-based metrics in `core/eval/metrics.py`."""

from __future__ import annotations

import pytest

from acgn_rag.core.eval.metrics import (
    CaseResult,
    abstention_accuracy,
    aggregate,
    answer_correctness,
    citation_containment,
    evidence_recall,
    hit_at_k,
    mrr_at_k,
    normalize_url,
    retrieval_metrics,
)

GOLD_A = "https://example.com/news/a"
GOLD_B = "https://example.com/news/b"
OTHER = "https://example.com/news/c"


def _case(**overrides: object) -> CaseResult:
    values: dict[str, object] = {
        "qa_id": "A-001",
        "lang": "en",
        "answerable": True,
        "gold_urls": (GOLD_A,),
        "retrieved_urls": (GOLD_A,),
    }
    values.update(overrides)
    return CaseResult(**values)  # type: ignore[arg-type]


def test_normalize_url_only_fragment_trailing_slash_and_host_case() -> None:
    assert normalize_url("https://Example.COM/news/?q=1#frag") == "https://example.com/news?q=1"
    assert normalize_url("https://example.com/news") == "https://example.com/news"
    assert normalize_url("https://example.com/news/") == "https://example.com/news"


def test_hit_at_k_any_gold_url_counts() -> None:
    assert hit_at_k(_case(retrieved_urls=(OTHER, GOLD_B), gold_urls=(GOLD_A, GOLD_B))) == 1
    assert hit_at_k(_case(retrieved_urls=(OTHER,), gold_urls=(GOLD_A,))) == 0
    assert hit_at_k(_case(gold_urls=())) == 0
    assert hit_at_k(_case(gold_unavailable=True)) == 0


def test_evidence_recall_distinct_gold_urls() -> None:
    assert evidence_recall(_case(retrieved_urls=(GOLD_A,), gold_urls=(GOLD_A, GOLD_B))) == 0.5
    assert evidence_recall(_case(retrieved_urls=(GOLD_A, GOLD_A), gold_urls=(GOLD_A,))) == 1.0
    assert evidence_recall(_case(gold_urls=())) == 0.0
    assert evidence_recall(_case(gold_unavailable=True)) == 0.0


def test_mrr_at_k_first_hit_only() -> None:
    assert mrr_at_k(_case(retrieved_urls=(OTHER, GOLD_A), gold_urls=(GOLD_A,))) == 0.5
    assert mrr_at_k(_case(retrieved_urls=(OTHER,), gold_urls=(GOLD_A,))) == 0.0
    assert mrr_at_k(_case(gold_urls=())) == 0.0


def test_retrieval_metrics_per_language_and_exclusions() -> None:
    cases = [
        _case(qa_id="A-001", retrieved_urls=(GOLD_A,)),
        _case(qa_id="A-002", lang="zh", retrieved_urls=(OTHER,)),
        _case(qa_id="A-003", gold_unavailable=True),
        _case(qa_id="D-001", answerable=False, gold_urls=()),
        _case(qa_id="A-004", gold_urls=()),
    ]

    metrics = retrieval_metrics(cases)

    assert metrics["n"] == {"all": 2, "en": 1, "zh": 1}
    assert metrics["retrieval_hit_rate"] == {"all": 0.5, "en": 1.0, "zh": 0.0}
    assert metrics["evidence_recall"] == {"all": 0.5, "en": 1.0, "zh": 0.0}
    assert metrics["mrr"] == {"all": 0.5, "en": 1.0, "zh": 0.0}
    assert metrics["excluded"] == {
        "unanswerable": 1,
        "gold_unavailable": 1,
        "missing_gold": 1,
    }


def test_answer_correctness_aggregates_human_labels() -> None:
    cases = [
        _case(qa_id="A-001", answer_label="fully_correct"),
        _case(qa_id="A-002", answer_label="partially_correct"),
        _case(qa_id="A-003", answer_label="incorrect"),
        _case(qa_id="A-004", answer_label=None),
    ]

    metrics = answer_correctness(cases)

    assert metrics["counts"] == {
        "fully_correct": 1,
        "partially_correct": 1,
        "incorrect": 1,
    }
    assert metrics["labeled"] == 3
    assert metrics["fully_correct_rate"] == pytest.approx(1 / 3)


def test_answer_correctness_rejects_unknown_label() -> None:
    with pytest.raises(ValueError, match="unknown answer_label"):
        answer_correctness([_case(answer_label="great")])


def test_abstention_accuracy_reports_both_rates() -> None:
    cases = [
        _case(qa_id="A-001", abstained=False),
        _case(qa_id="A-002", abstained=True),
        _case(qa_id="D-001", answerable=False, gold_urls=(), abstained=True),
        _case(qa_id="D-002", answerable=False, gold_urls=(), abstained=False),
    ]

    metrics = abstention_accuracy(cases)

    assert metrics["answerable"] == 2
    assert metrics["unanswerable"] == 2
    assert metrics["false_refusals"] == 1
    assert metrics["correct_refusals"] == 1
    assert metrics["false_refusal_rate"] == 0.5
    assert metrics["refusal_rate"] == 0.5


def test_abstention_accuracy_none_when_no_usable_items() -> None:
    metrics = abstention_accuracy([_case(abstained=None)])
    assert metrics["false_refusal_rate"] is None
    assert metrics["refusal_rate"] is None


def test_citation_containment_flags_outside_chunks() -> None:
    metrics = citation_containment(
        [
            _case(
                qa_id="A-001",
                retrieved_chunk_ids=("c1", "c2"),
                cited_chunk_ids=("c1",),
            ),
            _case(
                qa_id="A-002",
                retrieved_chunk_ids=("c1",),
                cited_chunk_ids=("c2",),
            ),
        ]
    )
    assert metrics == {"checked": 2, "violations": 1}


def test_aggregate_contains_every_report_block() -> None:
    metrics = aggregate([_case()])
    assert set(metrics) == {
        "retrieval_hit_rate",
        "evidence_recall",
        "mrr",
        "n",
        "excluded",
        "answer_correctness",
        "abstention_accuracy",
        "citation_containment",
    }
