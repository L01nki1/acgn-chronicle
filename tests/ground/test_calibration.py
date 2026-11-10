"""Deterministic behaviour of the pure abstention-threshold calibration utility."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence

import pytest

from acgn_rag.core.ground.calibration import (
    CalibrationSample,
    ThresholdPoint,
    choose_threshold,
    sweep_thresholds,
)

STRATEGIES = ("balanced_accuracy", "target_missed_refusal")


def _sample(score: float, answerable: bool) -> CalibrationSample:
    return CalibrationSample(retrieval_top_score=score, answerable=answerable)


def _points_by_threshold(
    samples: Sequence[CalibrationSample],
) -> dict[float, ThresholdPoint]:
    return {point.threshold: point for point in sweep_thresholds(samples)}


def _overlapping_samples() -> list[CalibrationSample]:
    return [
        _sample(0.9, True),
        _sample(0.7, True),
        _sample(0.5, True),
        _sample(0.6, False),
        _sample(0.4, False),
    ]


def test_perfectly_separable_scores_reach_the_perfect_point() -> None:
    samples = [
        _sample(0.9, True),
        _sample(0.8, True),
        _sample(0.7, True),
        _sample(0.2, False),
        _sample(0.1, False),
    ]

    result = choose_threshold(samples)

    assert result.strategy == "balanced_accuracy"
    assert result.point.threshold == 0.7
    assert 0.2 < result.point.threshold <= 0.7
    assert result.point.refused_answerable == 0
    assert result.point.missed_unanswerable == 0
    assert result.point.false_refusal_rate == 0.0
    assert result.point.missed_refusal_rate == 0.0
    assert result.point.balanced_accuracy == 1.0
    assert result.total_samples == 5
    assert result.answerable_samples == 3
    assert result.unanswerable_samples == 2

    perfect = [point for point in sweep_thresholds(samples) if point.balanced_accuracy == 1.0]
    assert [point.threshold for point in perfect] == [0.7]


def test_overlapping_scores_produce_expected_counts_and_rates() -> None:
    samples = _overlapping_samples()
    points = _points_by_threshold(samples)
    sentinel = math.nextafter(0.9, math.inf)

    expected = {
        # threshold: (refused_answerable, missed_unanswerable, false rate, missed rate, BA)
        0.4: (0, 2, 0.0, 1.0, 0.5),
        0.5: (0, 1, 0.0, 0.5, 0.75),
        0.6: (1, 1, 1 / 3, 0.5, 7 / 12),
        0.7: (1, 0, 1 / 3, 0.0, 5 / 6),
        0.9: (2, 0, 2 / 3, 0.0, 2 / 3),
        sentinel: (3, 0, 1.0, 0.0, 0.5),
    }
    assert set(points) == set(expected)
    for threshold, (refused, missed, false_rate, missed_rate, balanced) in expected.items():
        point = points[threshold]
        assert point.refused_answerable == refused, threshold
        assert point.missed_unanswerable == missed, threshold
        assert point.false_refusal_rate == pytest.approx(false_rate), threshold
        assert point.missed_refusal_rate == pytest.approx(missed_rate), threshold
        assert point.balanced_accuracy == pytest.approx(balanced), threshold

    result = choose_threshold(samples)

    assert result.point.threshold == 0.7
    assert result.point.refused_answerable == 1
    assert result.point.missed_unanswerable == 0
    assert result.point.false_refusal_rate == pytest.approx(1 / 3)
    assert result.point.missed_refusal_rate == 0.0
    assert result.point.balanced_accuracy == pytest.approx(5 / 6)


def test_a_sample_at_the_threshold_is_answered_not_refused() -> None:
    samples = [_sample(0.5, True), _sample(0.5, False)]
    points = _points_by_threshold(samples)

    at_threshold = points[0.5]
    assert at_threshold.refused_answerable == 0
    assert at_threshold.missed_unanswerable == 1
    assert at_threshold.false_refusal_rate == 0.0
    assert at_threshold.missed_refusal_rate == 1.0

    just_above = points[math.nextafter(0.5, math.inf)]
    assert just_above.refused_answerable == 1
    assert just_above.missed_unanswerable == 0


def test_balanced_accuracy_tie_prefers_lower_false_refusal() -> None:
    samples = [
        _sample(0.1, False),
        _sample(0.5, True),
        _sample(0.6, False),
        _sample(0.8, True),
    ]
    points = _points_by_threshold(samples)
    assert points[0.5].balanced_accuracy == pytest.approx(0.75)
    assert points[0.8].balanced_accuracy == pytest.approx(0.75)

    result = choose_threshold(samples)

    assert result.point.threshold == 0.5
    assert result.point.false_refusal_rate == 0.0
    assert result.point.missed_refusal_rate == 0.5
    assert result.point.balanced_accuracy == pytest.approx(0.75)


def test_target_missed_refusal_uses_the_target_to_reject_missed_refusals() -> None:
    samples = _overlapping_samples()

    strict = choose_threshold(
        samples,
        strategy="target_missed_refusal",
        target_missed_refusal_rate=0.2,
    )
    assert strict.strategy == "target_missed_refusal"
    assert strict.point.threshold == 0.7
    assert strict.point.refused_answerable == 1
    assert strict.point.missed_unanswerable == 0
    assert strict.point.false_refusal_rate == pytest.approx(1 / 3)
    assert strict.point.missed_refusal_rate == 0.0

    relaxed = choose_threshold(
        samples,
        strategy="target_missed_refusal",
        target_missed_refusal_rate=0.5,
    )
    assert relaxed.point.threshold == 0.5
    assert relaxed.point.refused_answerable == 0
    assert relaxed.point.missed_unanswerable == 1
    assert relaxed.point.false_refusal_rate == 0.0
    assert relaxed.point.missed_refusal_rate == 0.5


def test_target_missed_refusal_ties_prefer_the_higher_threshold() -> None:
    samples = [
        _sample(0.1, False),
        _sample(0.5, False),
        _sample(0.6, True),
        _sample(0.9, True),
    ]

    result = choose_threshold(
        samples,
        strategy="target_missed_refusal",
        target_missed_refusal_rate=0.5,
    )

    assert result.point.threshold == 0.6
    assert result.point.false_refusal_rate == 0.0
    assert result.point.missed_refusal_rate == 0.0
    assert result.point.balanced_accuracy == 1.0


def test_target_missed_refusal_with_no_feasible_point_raises() -> None:
    with pytest.raises(ValueError, match="no threshold satisfies"):
        choose_threshold(
            _overlapping_samples(),
            strategy="target_missed_refusal",
            target_missed_refusal_rate=-0.1,
        )


def test_unknown_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown calibration strategy"):
        choose_threshold(
            _overlapping_samples(),
            strategy="majority_vote",  # type: ignore[arg-type]
        )


def test_empty_samples_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        sweep_thresholds([])
    with pytest.raises(ValueError, match="at least one"):
        choose_threshold([])


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
def test_non_finite_scores_are_rejected(score: float) -> None:
    samples = [_sample(score, True), _sample(0.5, False)]

    with pytest.raises(ValueError, match="finite"):
        sweep_thresholds(samples)
    with pytest.raises(ValueError, match="finite"):
        choose_threshold(samples)


@pytest.mark.parametrize("answerable", [1, 0, "true", None])
def test_non_bool_answerable_is_rejected(answerable: object) -> None:
    sample = CalibrationSample(0.5, answerable)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="answerable must be bool"):
        sweep_thresholds([sample])


def test_boolean_score_is_not_a_real_number() -> None:
    sample = CalibrationSample(True, True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="finite real number"):
        sweep_thresholds([sample])


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize(
    "samples",
    [
        [_sample(0.9, True), _sample(0.8, True)],
        [_sample(0.1, False)],
    ],
)
def test_choose_requires_both_classes(samples: list[CalibrationSample], strategy: str) -> None:
    with pytest.raises(ValueError, match="at least one answerable and one unanswerable"):
        choose_threshold(samples, strategy=strategy)  # type: ignore[arg-type]


def test_sweep_is_ordered_deduplicated_and_input_order_independent() -> None:
    samples = [
        _sample(0.3, False),
        _sample(0.9, True),
        _sample(0.3, True),
        _sample(0.1, False),
    ]

    forward = sweep_thresholds(samples)
    backward = sweep_thresholds(list(reversed(samples)))

    assert forward == backward
    assert [point.threshold for point in forward] == [
        0.1,
        0.3,
        0.9,
        math.nextafter(0.9, math.inf),
    ]


def test_sweep_represents_no_refusal_at_the_minimum_and_all_refusal_at_the_sentinel() -> None:
    samples = [
        _sample(0.9, True),
        _sample(0.7, True),
        _sample(0.2, False),
        _sample(0.1, False),
    ]
    points = sweep_thresholds(samples)
    first, last = points[0], points[-1]

    assert first.threshold == 0.1
    assert first.refused_answerable == 0
    assert first.missed_unanswerable == 2
    assert first.false_refusal_rate == 0.0
    assert first.missed_refusal_rate == 1.0

    assert last.threshold == math.nextafter(0.9, math.inf)
    assert last.refused_answerable == 2
    assert last.missed_unanswerable == 0
    assert last.false_refusal_rate == 1.0
    assert last.missed_refusal_rate == 0.0
    assert last.balanced_accuracy == 0.5


def test_one_class_sweep_leaves_undefined_rates_none() -> None:
    answerable_points = sweep_thresholds([_sample(0.9, True), _sample(0.1, True)])
    assert all(point.missed_refusal_rate is None for point in answerable_points)
    assert all(point.balanced_accuracy is None for point in answerable_points)
    assert answerable_points[0].false_refusal_rate == 0.0
    assert answerable_points[-1].false_refusal_rate == 1.0

    unanswerable_points = sweep_thresholds([_sample(0.9, False), _sample(0.1, False)])
    assert all(point.false_refusal_rate is None for point in unanswerable_points)
    assert all(point.balanced_accuracy is None for point in unanswerable_points)
    assert unanswerable_points[0].missed_refusal_rate == 1.0
    assert unanswerable_points[-1].missed_refusal_rate == 0.0


def test_metadata_records_the_in_sample_calibration_only() -> None:
    result = choose_threshold([_sample(0.9, True), _sample(0.7, True), _sample(0.5, False)])
    metadata = result.to_metadata()

    assert metadata["in_sample"] is True
    assert metadata["threshold"] == 0.7
    assert metadata["strategy"] == "balanced_accuracy"
    assert metadata["counts"] == {
        "total": 3,
        "answerable": 2,
        "unanswerable": 1,
    }
    assert metadata["refused_answerable"] == 0
    assert metadata["missed_unanswerable"] == 0
    assert metadata["false_refusal_rate"] == 0.0
    assert metadata["missed_refusal_rate"] == 0.0
    assert metadata["balanced_accuracy"] == 1.0
    assert "abstention_accuracy" not in metadata
    assert "retrieval_hit_rate" not in metadata
    assert json.loads(json.dumps(metadata, allow_nan=False)) == metadata


def test_selected_threshold_round_trips_into_the_strict_decision_rule() -> None:
    samples = [
        _sample(0.9, True),
        _sample(0.7, True),
        _sample(0.2, False),
        _sample(0.1, False),
    ]
    result = choose_threshold(samples)
    threshold = result.point.threshold

    # The answerer's optional min_retrieval_score gate uses the same strict comparison:
    # a score strictly below the threshold refuses; equality answers.
    def refuses(score: float, limit: float) -> bool:
        return score < limit

    for sample in samples:
        assert refuses(sample.retrieval_top_score, threshold) is (not sample.answerable)
    assert refuses(0.7, threshold) is False
    assert refuses(math.nextafter(0.7, -math.inf), threshold) is True
