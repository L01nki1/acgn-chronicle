"""Deterministic abstention-threshold calibration from labelled retrieval scores.

The answerer refuses a request when the request's best retrieval score is strictly below its
configured threshold ("score < threshold"); a score equal to the threshold still answers.
This module turns labelled per-item top retrieval scores into the operating point that rule
should use. It is pure: it runs no retrieval, calls no generation model and reads no files.

Scope and interpretation
------------------------
* Calibration is in-sample. The selected point and every rate in
  CalibrationResult.to_metadata describe the calibration set only. Quoting them as QA-set
  results would be leakage; the disjoint QA set is what measures abstention
  (docs/EVAL_PROTOCOL.md section 6).
* A retrieval score is ordinal within one response only (ports.SCORE_SEMANTICS). The
  selected threshold therefore belongs to the exact snapshot, chunk configuration, retrieval
  mode and encoder identity that produced these scores. It must be re-calibrated when any of
  those changes and must never be ported across retrieval modes.
* At a fixed threshold, refusals are part of the protocol's deterministic tier
  (docs/EVAL_PROTOCOL.md section 8): the sweep order, tie-breaks and rate arithmetic below
  are deterministic for a given sample sequence.

Threshold candidates
--------------------
One candidate is produced for each distinct observed score, ascending, so every distinct
answer/refuse partition induced by the samples is considered. Because a sample whose score
equals the candidate is answered, the lowest candidate answers every sample. One sentinel
candidate, math.nextafter above the maximum score, refuses every sample and therefore
represents the all-refused endpoint.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

Strategy = Literal["balanced_accuracy", "target_missed_refusal"]


@dataclass(frozen=True, slots=True)
class CalibrationSample:
    """One labelled calibration item and its best retrieval score for one request."""

    retrieval_top_score: float
    answerable: bool


@dataclass(frozen=True, slots=True)
class ThresholdPoint:
    """One candidate threshold and the refusal counts it produces.

    The decision rule is strict: an item is refused exactly when
    retrieval_top_score < threshold; a score equal to the threshold is answered.
    refused_answerable counts false refusals and missed_unanswerable counts unanswerable
    items that were answered instead of refused (missed refusals).

    false_refusal_rate is refused_answerable / answerable_count; missed_refusal_rate is
    missed_unanswerable / unanswerable_count. A rate is None when its class is absent.
    balanced_accuracy is None unless both classes are present; when defined it is
    0.5 * ((1 - missed_refusal_rate) + (1 - false_refusal_rate)), the mean of refusal
    recall on unanswerable items and answer recall on answerable items.
    """

    threshold: float
    refused_answerable: int
    missed_unanswerable: int
    false_refusal_rate: float | None
    missed_refusal_rate: float | None
    balanced_accuracy: float | None


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """The selected operating point and the sample counts an evaluation run records."""

    point: ThresholdPoint
    strategy: Strategy
    total_samples: int
    answerable_samples: int
    unanswerable_samples: int

    def to_metadata(self) -> dict[str, object]:
        """Return the in-sample calibration record for a run's metadata.

        in_sample is always true. These are calibration-set counts and rates; they are not
        QA-set abstention results and must not be reported as such.
        """
        return {
            "threshold": self.point.threshold,
            "strategy": self.strategy,
            "in_sample": True,
            "counts": {
                "total": self.total_samples,
                "answerable": self.answerable_samples,
                "unanswerable": self.unanswerable_samples,
            },
            "refused_answerable": self.point.refused_answerable,
            "missed_unanswerable": self.point.missed_unanswerable,
            "false_refusal_rate": self.point.false_refusal_rate,
            "missed_refusal_rate": self.point.missed_refusal_rate,
            "balanced_accuracy": self.point.balanced_accuracy,
        }


def sweep_thresholds(samples: Sequence[CalibrationSample]) -> tuple[ThresholdPoint, ...]:
    """Evaluate every candidate threshold, ascending, including the all-refused sentinel.

    Candidates are the distinct observed scores in ascending order plus
    math.nextafter(max(scores), math.inf). At the lowest candidate every sample is answered
    (score < threshold is false); at the sentinel every sample is refused. Identical scores
    produce one candidate, not one per sample.

    Raises:
        ValueError: if samples is empty, a score is not a finite real number, or an
            answerable value is not a real bool.
    """
    values = _validate_samples(samples)
    return _sweep(values)


def choose_threshold(
    samples: Sequence[CalibrationSample],
    *,
    strategy: Strategy = "balanced_accuracy",
    target_missed_refusal_rate: float = 0.2,
) -> CalibrationResult:
    """Select one threshold operating point from labelled top retrieval scores.

    Both strategies require at least one answerable and one unanswerable sample; a threshold
    chosen without both classes says nothing about the trade-off it is supposed to balance.

    balanced_accuracy maximises
    0.5 * ((1 - missed_refusal_rate) + (1 - false_refusal_rate)). Ties prefer the lower
    false-refusal rate (do not refuse answerable items), then the higher threshold (fewer
    missed refusals once false refusals are equal).

    target_missed_refusal keeps only points whose missed-refusal rate is at most
    target_missed_refusal_rate, minimises the false-refusal rate among them, and breaks ties
    toward the higher threshold. Because the sweep includes the all-refused sentinel, every
    non-negative target is feasible; a ValueError therefore indicates a target no point
    satisfies (typically a negative or NaN target).

    Raises:
        ValueError: if samples is empty or invalid, either class is absent, the strategy is
            unknown, or no point satisfies the target.
    """
    values = _validate_samples(samples)
    answerable = sum(1 for sample in values if sample.answerable)
    unanswerable = len(values) - answerable
    if not answerable or not unanswerable:
        raise ValueError(
            "threshold selection requires at least one answerable and one unanswerable "
            f"sample; got {answerable} answerable and {unanswerable} unanswerable"
        )

    points = _sweep(values)
    if strategy == "balanced_accuracy":
        chosen = max(points, key=_balanced_accuracy_rank)
    elif strategy == "target_missed_refusal":
        feasible = [
            point
            for point in points
            if point.missed_refusal_rate is not None
            and point.missed_refusal_rate <= target_missed_refusal_rate
        ]
        if not feasible:
            lowest = min(
                point.missed_refusal_rate
                for point in points
                if point.missed_refusal_rate is not None
            )
            raise ValueError(
                "no threshold satisfies "
                f"target_missed_refusal_rate={target_missed_refusal_rate!r}; "
                f"the lowest achievable missed-refusal rate is {lowest!r}"
            )
        chosen = max(feasible, key=_target_missed_refusal_rank)
    else:
        raise ValueError(
            f"unknown calibration strategy {strategy!r}; "
            "expected 'balanced_accuracy' or 'target_missed_refusal'"
        )

    return CalibrationResult(
        point=chosen,
        strategy=strategy,
        total_samples=len(values),
        answerable_samples=answerable,
        unanswerable_samples=unanswerable,
    )


def _validate_samples(samples: Sequence[CalibrationSample]) -> tuple[CalibrationSample, ...]:
    """Return the samples as a tuple after rejecting values that cannot be calibrated."""
    values = tuple(samples)
    if not values:
        raise ValueError("at least one calibration sample is required")
    for index, sample in enumerate(values):
        if not isinstance(sample, CalibrationSample):
            raise TypeError(
                f"samples[{index}] must be a CalibrationSample, got {type(sample).__name__}"
            )
        if type(sample.answerable) is not bool:
            raise ValueError(f"samples[{index}].answerable must be bool, got {sample.answerable!r}")
        score = sample.retrieval_top_score
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(
                f"samples[{index}].retrieval_top_score must be a finite real number, got {score!r}"
            )
        try:
            numeric_score = float(score)
        except OverflowError as exc:
            raise ValueError(
                f"samples[{index}].retrieval_top_score is outside the finite float range: {score!r}"
            ) from exc
        if not math.isfinite(numeric_score):
            raise ValueError(f"samples[{index}].retrieval_top_score must be finite, got {score!r}")
    return values


def _sweep(values: tuple[CalibrationSample, ...]) -> tuple[ThresholdPoint, ...]:
    """Compute the candidate points for already-validated samples."""
    # Adding 0.0 canonicalises -0.0 to +0.0, so candidate identity does not depend on the
    # order in which otherwise-equal zero scores arrived.
    thresholds = sorted({float(sample.retrieval_top_score) + 0.0 for sample in values})
    thresholds.append(math.nextafter(thresholds[-1], math.inf))

    answerable = sum(1 for sample in values if sample.answerable)
    unanswerable = len(values) - answerable
    points: list[ThresholdPoint] = []
    for threshold in thresholds:
        refused_answerable = 0
        missed_unanswerable = 0
        for sample in values:
            if float(sample.retrieval_top_score) < threshold:
                if sample.answerable:
                    refused_answerable += 1
            elif not sample.answerable:
                missed_unanswerable += 1

        false_refusal_rate = refused_answerable / answerable if answerable else None
        missed_refusal_rate = missed_unanswerable / unanswerable if unanswerable else None
        if false_refusal_rate is None or missed_refusal_rate is None:
            balanced_accuracy = None
        else:
            balanced_accuracy = 0.5 * ((1.0 - missed_refusal_rate) + (1.0 - false_refusal_rate))
        points.append(
            ThresholdPoint(
                threshold=threshold,
                refused_answerable=refused_answerable,
                missed_unanswerable=missed_unanswerable,
                false_refusal_rate=false_refusal_rate,
                missed_refusal_rate=missed_refusal_rate,
                balanced_accuracy=balanced_accuracy,
            )
        )
    return tuple(points)


def _balanced_accuracy_rank(point: ThresholdPoint) -> tuple[float, float, float]:
    """Rank by higher balanced accuracy, lower false refusals, then higher threshold."""
    assert point.balanced_accuracy is not None
    assert point.false_refusal_rate is not None
    return (point.balanced_accuracy, -point.false_refusal_rate, point.threshold)


def _target_missed_refusal_rank(point: ThresholdPoint) -> tuple[float, float]:
    """Rank by lower false refusals, then higher threshold."""
    assert point.false_refusal_rate is not None
    return (-point.false_refusal_rate, point.threshold)
