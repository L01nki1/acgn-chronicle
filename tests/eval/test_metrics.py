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

    assert metrics["k"] == 5
    assert metrics["hit"] == {
        "all": {"value": 0.5, "n": 2},
        "en": {"value": 1.0, "n": 1},
        "zh": {"value": 0.0, "n": 1},
    }
    assert metrics["evidence_recall"]["all"] == {"value": 0.5, "n": 2}
    assert metrics["mrr"]["all"] == {"value": 0.5, "n": 2}
    assert metrics["excluded"] == {
        "unanswerable": 1,
        "gold_unavailable": 1,
        "gold_unresolved": 0,
        "missing_gold": 1,
        "dropped": 0,
    }


def test_every_rate_carries_its_own_n() -> None:
    empty = retrieval_metrics([_case(gold_unavailable=True)])
    for family in ("hit", "mrr", "evidence_recall"):
        assert empty[family]["all"] == {"value": None, "n": 0}

    one = retrieval_metrics([_case()])
    assert one["hit"]["all"] == {"value": 1.0, "n": 1}


def test_dropped_and_calibration_items_leave_the_reporting_denominators() -> None:
    cases = [
        _case(qa_id="A-001"),
        _case(qa_id="A-002", dropped=True),
        _case(qa_id="C-001", calibration=True),
    ]
    metrics = retrieval_metrics(cases)
    assert metrics["hit"]["all"] == {"value": 1.0, "n": 1}
    assert metrics["excluded"]["dropped"] == 1

    counts = aggregate(cases)["counts"]
    assert counts["qa_total"] == 2
    assert counts["dropped"] == 1


def test_answer_correctness_aggregates_human_labels() -> None:
    cases = [
        _case(qa_id="A-001", answer_label="fully_correct"),
        _case(qa_id="A-002", lang="zh", answer_label="partially_correct"),
        _case(qa_id="A-003", answer_label="incorrect"),
        _case(qa_id="A-004", answer_label=None),
    ]

    metrics = answer_correctness(cases)

    assert metrics["counts"] == {
        "fully_correct": 1,
        "partially_correct": 1,
        "incorrect": 1,
    }
    assert metrics["rates"]["all"] == {"value": pytest.approx(1 / 3), "n": 3}
    assert metrics["rates"]["en"] == {"value": 0.5, "n": 2}
    assert metrics["rates"]["zh"] == {"value": 0.0, "n": 1}


def test_answer_correctness_excludes_dropped_items_from_denominator() -> None:
    metrics = answer_correctness(
        [
            _case(qa_id="A-001", answer_label="fully_correct"),
            _case(qa_id="A-002", answer_label="incorrect", dropped=True),
        ]
    )
    assert metrics["counts"]["incorrect"] == 0
    assert metrics["rates"]["all"] == {"value": 1.0, "n": 1}


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

    metrics = abstention_accuracy(cases, threshold=0.5)

    assert metrics["threshold"] == 0.5
    assert metrics["false_refusal_rate"] == {"value": 0.5, "n": 2}
    assert metrics["refusal_rate"] == {"value": 0.5, "n": 2}
    assert metrics["calibration_in_sample"] == {"value": None, "n": 0}


def test_abstention_accuracy_none_when_no_usable_items() -> None:
    metrics = abstention_accuracy([_case(abstained=None)])
    assert metrics["false_refusal_rate"] == {"value": None, "n": 0}
    assert metrics["refusal_rate"] == {"value": None, "n": 0}


def test_calibration_set_is_labelled_in_sample_not_a_reporting_denominator() -> None:
    metrics = abstention_accuracy(
        [
            _case(qa_id="D-001", answerable=False, gold_urls=(), abstained=True),
            _case(qa_id="C-001", answerable=True, abstained=True, calibration=True),
            _case(qa_id="C-002", answerable=False, gold_urls=(), abstained=False, calibration=True),
        ]
    )
    # Reporting rates stay on the QA set only.
    assert metrics["refusal_rate"] == {"value": 1.0, "n": 1}
    assert metrics["false_refusal_rate"] == {"value": None, "n": 0}
    # Calibration is quoted separately, with its own n.
    assert metrics["calibration_in_sample"] == {"value": 0.0, "n": 2}


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
    assert metrics == {"containment_rate": {"value": 0.5, "n": 2}, "violations": 1}


def test_citation_containment_counts_a_citation_with_no_retrieved_chunks() -> None:
    metrics = citation_containment(
        [_case(qa_id="A-001", retrieved_chunk_ids=(), cited_chunk_ids=("c1",))]
    )
    assert metrics == {"containment_rate": {"value": 0.0, "n": 1}, "violations": 1}


def test_aggregate_contains_every_report_block() -> None:
    metrics = aggregate(
        [_case()],
        k=5,
        deduped_count=6402,
        indexed_chunks=9120,
    )
    assert set(metrics) == {
        "counts",
        "retrieval",
        "correctness",
        "abstention",
        "citation",
        "provisional",
    }
    assert metrics["provisional"] is False
    assert metrics["counts"] == {
        "qa_total": 1,
        "answerable": 1,
        "unanswerable": 0,
        "gold_unavailable": 0,
        "gold_unresolved": 0,
        "dropped": 0,
        "deduped_count": 6402,
        "indexed_chunks": 9120,
    }
    assert metrics["retrieval"]["k"] == 5
    assert metrics["retrieval"]["hit"]["all"] == {"value": 1.0, "n": 1}
    assert metrics["citation"]["containment_rate"] == {"value": None, "n": 0}


def test_unresolved_gold_is_excluded_and_counted() -> None:
    cases = [
        _case(qa_id="A-001"),
        _case(qa_id="A-002", gold_unresolved=True),
    ]
    metrics = aggregate(cases, provisional=True)
    assert metrics["retrieval"]["hit"]["all"] == {"value": 1.0, "n": 1}
    assert metrics["counts"]["gold_unresolved"] == 1
    assert metrics["retrieval"]["excluded"]["gold_unresolved"] == 1
    assert metrics["provisional"] is True


def test_retrieval_excluded_counts_ignore_calibration_items() -> None:
    cases = [
        _case(qa_id="A-001", gold_unavailable=True),
        _case(qa_id="C-001", gold_unavailable=True, calibration=True),
    ]
    metrics = retrieval_metrics(cases)
    assert metrics["excluded"]["gold_unavailable"] == 1
