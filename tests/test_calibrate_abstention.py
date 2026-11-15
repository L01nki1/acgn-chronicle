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
