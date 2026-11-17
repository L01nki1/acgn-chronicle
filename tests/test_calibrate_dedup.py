"""G1b dedup calibration tooling: sweep reproduction, sheet shape, and the label gate.

Owner: F (WP2) | Spec: docs/RUNBOOK.md §3.3

These tests pin the *tooling* around the hand-labelling, never a label: the sweep must be
deterministic on a frozen snapshot, the sheet must sample the whole cosine range instead of
its top with enough **aux** pairs to meet the gate, and the scorer must refuse to invent
precision from an unlabelled, sparse, or channel-less sheet.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import calibrate_dedup  # noqa: E402


def _publish(root: Path, count: int = 8) -> str:
    import make_smoke_snapshot

    assert make_smoke_snapshot.main(["--data-root", str(root), "--count", str(count)]) == 0
    snapshots = sorted(path for path in (root / "snapshots").iterdir() if path.is_dir())
    return snapshots[0].name


@pytest.fixture()
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "data"
    monkeypatch.setenv("ACGN_DATA_DIR", str(root))
    monkeypatch.delenv("ACGN_EMBED_MODEL", raising=False)
    return root


def test_sweep_and_sheet_on_a_synthetic_snapshot(data_root: Path, tmp_path: Path) -> None:
    snapshot = _publish(data_root, count=8)
    out = tmp_path / "calib"
    code = calibrate_dedup.main(
        ["--snapshot", snapshot, "--data-root", str(data_root), "--out", str(out)]
    )
    assert code == 0

    sweep = json.loads((out / "sweep.json").read_text(encoding="utf-8"))
    thresholds = [row["threshold"] for row in sweep["thresholds"]]
    assert thresholds == [0.82, 0.84, 0.86, 0.88]
    for row in sweep["thresholds"]:
        # Synthetic documents are distinct, so nothing merges at any threshold.
        assert row["item_count"] == 8
        assert row["cluster_count"] == 8
        assert row["collapsed"] == 0
        assert row["member_pairs"] == 0

    with (out / "label_sheet.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [], "no non-singleton cluster means no pair to label"


def test_score_refuses_an_unlabelled_sheet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sheet = tmp_path / "sheet.csv"
    sheet.write_text(
        "threshold,pair_id,channel,decision\n0.86,0.86-01,aux,\n0.86,0.86-02,aux,duplicate\n",
        encoding="utf-8",
    )
    assert calibrate_dedup._score(sheet) == 2
    assert "unlabelled" in capsys.readouterr().err


def test_score_rejects_an_unknown_decision(tmp_path: Path) -> None:
    sheet = tmp_path / "sheet.csv"
    sheet.write_text(
        "threshold,pair_id,channel,decision\n0.86,0.86-01,aux,maybe\n", encoding="utf-8"
    )
    assert calibrate_dedup._score(sheet) == 2


def test_score_rejects_a_sheet_without_channels(tmp_path: Path) -> None:
    """E's #75 review, P3: without the channel column the gate cannot be aux-scoped."""
    sheet = tmp_path / "sheet.csv"
    sheet.write_text("threshold,pair_id,decision\n0.86,0.86-01,duplicate\n", encoding="utf-8")
    assert calibrate_dedup._score(sheet) == 2
    # The error is not raised in exploratory mode; the report just cannot be gate evidence.
    assert calibrate_dedup._score(sheet, exploratory=True) == 2


def test_score_reports_aux_precision_per_threshold(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sheet = tmp_path / "sheet.csv"
    sheet.write_text(
        "threshold,pair_id,channel,decision\n"
        "0.82,0.82-01,aux,duplicate\n"
        "0.82,0.82-02,aux,not_duplicate\n"
        "0.82,0.82-03,primary,skip\n"
        "0.82,0.82-04,primary,duplicate\n"
        "0.86,0.86-01,aux,duplicate\n"
        "0.86,0.86-02,aux,duplicate\n",
        encoding="utf-8",
    )
    assert calibrate_dedup._score(sheet, exploratory=True) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mode"].startswith("exploratory")
    assert report["thresholds"]["0.82"] == {
        "judged_aux": 2,
        "aux_duplicate": 1,
        "aux_precision": 0.5,
        "judged_primary": 1,
        "primary_duplicate": 1,
        "primary_precision": 1.0,
        "skipped": 1,
        "insufficient": True,
    }
    assert report["thresholds"]["0.86"]["aux_precision"] == 1.0


def test_sample_keeps_the_aux_head_and_spreads_the_rest() -> None:
    pairs = [{"cosine": 1.0 - index / 100, "channel": "aux"} for index in range(100)] + [
        {"cosine": 0.1, "channel": "primary"} for index in range(40)
    ]
    sample = calibrate_dedup._sample(pairs, 30)
    assert len(sample) == 30
    aux_rows = [pair for pair in sample if pair["channel"] == "aux"]
    primary_rows = [pair for pair in sample if pair["channel"] == "primary"]
    # The gate minimum is aux pairs: 20 of them (head + spread), never swapped away.
    assert len(aux_rows) == calibrate_dedup.PAIRS_PER_THRESHOLD
    assert aux_rows[0]["cosine"] == 1.0
    # The aux spread reaches below the head instead of stopping at the top of the range.
    assert aux_rows[-1]["cosine"] < pairs[calibrate_dedup.HEAD_ROWS]["cosine"]
    assert len(primary_rows) == 10


def test_sample_returns_everything_when_the_cluster_has_few_pairs() -> None:
    pairs = [
        {"cosine": 0.9, "channel": "aux"},
        {"cosine": 0.5, "channel": "primary"},
    ]
    assert calibrate_dedup._sample(pairs, 30) == pairs


def test_sample_takes_every_aux_pair_when_the_threshold_has_fewer_than_the_minimum() -> None:
    """At 0.88 many pairs are primary-only; the sheet must still carry every aux pair."""
    pairs = [{"cosine": 0.9, "channel": "aux"} for _ in range(6)]
    pairs.extend({"cosine": 0.1, "channel": "primary"} for _ in range(40))
    sample = calibrate_dedup._sample(pairs, 30)
    assert len(sample) == 30
    assert sum(1 for pair in sample if pair["channel"] == "aux") == 6


def test_score_marks_an_insufficient_sheet_for_the_gate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """E's #75 review, P3: primary pairs never count towards the aux gate."""
    rows = ["threshold,pair_id,channel,decision"]
    rows.append("0.86,0.86-01,aux,duplicate")
    rows.extend(f"0.86,0.86-{index:02d},aux,skip" for index in range(2, 26))
    rows.extend(f"0.86,0.86-{index:02d},primary,duplicate" for index in range(26, 31))
    sheet = tmp_path / "sheet.csv"
    sheet.write_text("\n".join(rows) + "\n", encoding="utf-8")

    assert calibrate_dedup._score(sheet) == 2
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["mode"] == "gate"
    assert report["gate"] == "INSUFFICIENT"
    assert report["required_judged_per_threshold"] == 20
    threshold = report["thresholds"]["0.86"]
    assert threshold["judged_aux"] == 1
    assert threshold["judged_primary"] == 5
    assert threshold["skipped"] == 24
    assert threshold["insufficient"] is True
    assert "aux judged 1/20" in captured.err


def test_score_passes_a_sufficient_gate_sheet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Gate evidence needs all four sweep values, each with distinct judged aux pairs."""
    rows = ["threshold,pair_id,channel,decision"]
    for threshold in calibrate_dedup.REQUIRED_THRESHOLDS:
        rows.extend(f"{threshold},{threshold}-{index:02d},aux,duplicate" for index in range(1, 21))
        rows.extend(
            f"{threshold},{threshold}-{index:02d},aux,not_duplicate" for index in range(21, 26)
        )
        rows.extend(
            f"{threshold},{threshold}-{index:02d},primary,duplicate" for index in range(26, 31)
        )
    sheet = tmp_path / "sheet.csv"
    sheet.write_text("\n".join(rows) + "\n", encoding="utf-8")

    assert calibrate_dedup._score(sheet) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "gate"
    assert report["gate"] == "PASS"
    assert report["missing_thresholds"] == []
    for threshold in calibrate_dedup.REQUIRED_THRESHOLDS:
        entry = report["thresholds"][threshold]
        assert entry["judged_aux"] == 25
        assert entry["aux_duplicate"] == 20
        assert entry["aux_precision"] == 0.8
        assert entry["judged_primary"] == 5
        assert entry["primary_precision"] == 1.0
        assert entry["insufficient"] is False


def test_score_rejects_a_single_threshold_gate_sheet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """F's #75 re-review, P2: 20 judgements for 0.86 alone is not a four-value sweep."""
    rows = ["threshold,pair_id,channel,decision"]
    rows.extend(f"0.86,0.86-{index:02d},aux,duplicate" for index in range(1, 21))
    sheet = tmp_path / "sheet.csv"
    sheet.write_text("\n".join(rows) + "\n", encoding="utf-8")

    assert calibrate_dedup._score(sheet) == 2
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["gate"] == "INSUFFICIENT"
    assert report["missing_thresholds"] == ["0.82", "0.84", "0.88"]
    assert report["thresholds"]["0.86"]["judged_aux"] == 20
    assert "insufficient hand labels" in captured.err


def test_score_counts_distinct_pairs_not_rows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One pair_id repeated 20 times must not read as 20 hand-checked pairs."""
    rows = ["threshold,pair_id,channel,decision"]
    for threshold in calibrate_dedup.REQUIRED_THRESHOLDS:
        rows.extend(f"{threshold},{threshold}-01,aux,duplicate" for _ in range(20))
    sheet = tmp_path / "sheet.csv"
    sheet.write_text("\n".join(rows) + "\n", encoding="utf-8")

    assert calibrate_dedup._score(sheet) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["gate"] == "INSUFFICIENT"
    for threshold in calibrate_dedup.REQUIRED_THRESHOLDS:
        assert report["thresholds"][threshold]["judged_aux"] == 1


def test_score_rejects_conflicting_decisions_for_one_pair(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sheet = tmp_path / "sheet.csv"
    sheet.write_text(
        "threshold,pair_id,channel,decision\n"
        "0.86,0.86-01,aux,duplicate\n"
        "0.86,0.86-01,aux,not_duplicate\n",
        encoding="utf-8",
    )
    assert calibrate_dedup._score(sheet) == 2
    assert "conflicting decisions" in capsys.readouterr().err


def test_score_required_judged_is_configurable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sheet = tmp_path / "sheet.csv"
    sheet.write_text(
        "threshold,pair_id,channel,decision\n"
        "0.86,0.86-01,aux,duplicate\n"
        "0.86,0.86-02,aux,not_duplicate\n",
        encoding="utf-8",
    )
    # Below the RUNBOOK minimum the report must be explicitly non-gate, not a gate PASS.
    assert calibrate_dedup._score(sheet, required=2) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mode"].startswith("exploratory")
    assert report["gate"] == "NOT_GATE"
    assert report["mode_reason"].startswith("--required-judged")
    assert report["thresholds"]["0.86"]["insufficient"] is False
