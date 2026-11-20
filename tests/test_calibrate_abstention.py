"""Tests for the abstention calibration archive builder."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import calibrate_abstention  # noqa: E402

from acgn_rag.core.ground.calibration import (  # noqa: E402
    CalibrationSample,
    choose_threshold,
)
from acgn_rag.core.ground.generate import ANSWERER_VERSION  # noqa: E402


def _result():
    return choose_threshold(
        [
            CalibrationSample(0.90, True),
            CalibrationSample(0.80, True),
            CalibrationSample(0.70, True),
            CalibrationSample(0.60, True),
            CalibrationSample(0.30, False),
            CalibrationSample(0.20, False),
            CalibrationSample(0.10, False),
            CalibrationSample(0.05, False),
        ]
    )


def _identity() -> dict[str, object]:
    return {
        "snapshot_id": "20261028abcdef",
        "chunk_config": "A",
        "chunker": {
            "version": "composed-input-v2",
            "header_version": "h1",
            "config": "A",
            "max_tokens": 512,
            "overlap": 64,
            "max_seq_length": 512,
        },
        "retrieval": {"mode": "vector", "top_k": 5, "over_fetch": 5},
        "embed": {
            "model": "fake",
            "backend": "torch",
            "device": "cuda",
            "precision": "fp32",
            "library_versions": {"torch": "2.15.0"},
            "batch": 32,
        },
        "encoder_identity_slug": "fake-torch-cuda-fp32-12345678",
        # The calibration lane runs the historical single-pool search with no query transform,
        # under the answer line this tree ships with.
        "query_transform": "none",
        "official_score_weight": 1.0,
        "answerer_version": ANSWERER_VERSION,
    }


def test_samples_pair_rows_with_scores() -> None:
    rows = [
        {"qa_id": "X-1", "answerable": True},
        {"qa_id": "X-2", "answerable": False},
    ]

    samples = calibrate_abstention._samples(rows, [0.9, None])

    assert [sample.retrieval_top_score for sample in samples] == [0.9, None]
    assert [sample.answerable for sample in samples] == [True, False]


def test_samples_reject_missing_answerable() -> None:
    with pytest.raises(ValueError, match="answerable"):
        calibrate_abstention._samples([{"qa_id": "X-1"}], [0.5])


def test_samples_reject_nonfinite_score() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        calibrate_abstention._samples([{"qa_id": "X-1", "answerable": True}], [float("nan")])


def test_artifact_records_the_threshold_and_identity() -> None:
    result = _result()

    artifact = calibrate_abstention._artifact(
        run_id="20261115-cal-A-fake",
        snapshot_id="20261028abcdef",
        config="A",
        identity=_identity(),
        result=result,
    )

    assert artifact["threshold"] == result.point.threshold
    assert artifact["calibration"]["in_sample"] is True
    assert artifact["calibration_identity"] == _identity()
    assert artifact["snapshot_id"] == "20261028abcdef"


def test_artifact_requires_an_encoder_slug() -> None:
    identity = _identity()
    identity["encoder_identity_slug"] = ""

    with pytest.raises(ValueError, match="encoder_identity_slug"):
        calibrate_abstention._artifact(
            run_id="20261115-cal-A-fake",
            snapshot_id="20261028abcdef",
            config="A",
            identity=identity,
            result=_result(),
        )


def test_run_id_is_filesystem_safe() -> None:
    run_id = calibrate_abstention._run_id("A'", "fake-torch-cuda-fp32-12345678")

    assert run_id.endswith("-cal-A-prime-fake-torch-cuda-fp32-12345678")
    assert calibrate_abstention._RUN_ID.fullmatch(run_id)


def _rows(n: int = 30, answerable: int = 20) -> list[dict[str, object]]:
    return [
        {
            "qa_id": f"X-{index:03d}",
            "question": "q",
            "answerable": index < answerable,
            "window": ["2023-10-28", "2026-10-28"],
        }
        for index in range(n)
    ]


def test_filters_reads_the_item_window() -> None:
    filters = calibrate_abstention._filters(_rows(1)[0])

    assert filters.since == "2023-10-28"
    assert filters.until == "2026-10-28"


def test_filters_require_a_window() -> None:
    with pytest.raises(ValueError, match="window"):
        calibrate_abstention._filters({"qa_id": "X-1", "question": "q"})


def test_set_problems_accepts_the_gate_shape() -> None:
    assert calibrate_abstention._set_problems(_rows(), allow_any_set=False) == []


def test_set_problems_rejects_a_short_gate_set() -> None:
    problems = calibrate_abstention._set_problems(_rows(29), allow_any_set=False)

    assert any("30 items" in problem for problem in problems)


def test_set_problems_rejects_duplicate_ids() -> None:
    rows = _rows()
    rows[1]["qa_id"] = rows[0]["qa_id"]

    problems = calibrate_abstention._set_problems(rows, allow_any_set=False)

    assert any("duplicate" in problem for problem in problems)


def test_set_problems_allows_an_exploration_set() -> None:
    assert calibrate_abstention._set_problems(_rows(10, 5), allow_any_set=True) == []


def test_lane_flags_reach_the_calibration_identity() -> None:
    """A threshold is bound to the lane it was measured on, so the flags must reach the identity.

    `query_transform` and `official_score_weight` used to be hard-coded to "none"/1.0 here, so a
    calibration produced for a bonus or translation lane could never pair with a live run on that
    lane: `_validate_abstention` rejected it on `official_score_weight` before the run started.
    """
    from acgn_rag.core.eval.runner import run_identity

    chunker = {
        "version": "composed-input-v2",
        "header_version": "h1",
        "config": "A",
        "max_tokens": 512,
        "overlap": 64,
        "max_seq_length": 512,
    }
    identity = run_identity(
        snapshot_id="20261105-d797b5f0ae8e",
        config="A",
        chunker=chunker,
        top_k=5,
        over_fetch=5,
        build_identity={"identity_slug": "fake-torch-cuda-fp32-12345678"},
        query_transform="translate-zh",
        official_score_weight=1.08,
        answerer_version=ANSWERER_VERSION,
    )

    assert identity["query_transform"] == "translate-zh"
    assert identity["official_score_weight"] == 1.08
