"""G1b dedup calibration tooling: sweep reproduction, sheet shape, and the label gate.

Owner: F (WP2) | Spec: docs/RUNBOOK.md §3.3

These tests pin the *tooling* around the hand-labelling, never a label: the sweep must be
deterministic on a frozen snapshot, the sheet must sample the whole cosine range instead of
its top, and the scorer must refuse to invent precision from an unlabelled or incomplete sheet.
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
        "threshold,pair_id,decision\n0.86,0.86-01,\n0.86,0.86-02,duplicate\n", encoding="utf-8"
    )
    assert calibrate_dedup._score(sheet) == 2
    assert "unlabelled" in capsys.readouterr().err


def test_score_rejects_an_unknown_decision(tmp_path: Path) -> None:
    sheet = tmp_path / "sheet.csv"
    sheet.write_text("threshold,pair_id,decision\n0.86,0.86-01,maybe\n", encoding="utf-8")
    assert calibrate_dedup._score(sheet) == 2


def test_score_reports_precision_per_threshold(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sheet = tmp_path / "sheet.csv"
    sheet.write_text(
        "threshold,pair_id,decision\n"
        "0.82,0.82-01,duplicate\n"
        "0.82,0.82-02,not_duplicate\n"
        "0.82,0.82-03,skip\n"
        "0.86,0.86-01,duplicate\n"
        "0.86,0.86-02,duplicate\n",
        encoding="utf-8",
    )
    assert calibrate_dedup._score(sheet) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["thresholds"]["0.82"] == {
        "judged": 2,
        "duplicate": 1,
        "not_duplicate": 1,
        "precision": 0.5,
    }
    assert report["thresholds"]["0.86"]["precision"] == 1.0


def test_sample_keeps_the_head_and_spreads_the_rest() -> None:
    pairs = [{"cosine": 1.0 - index / 100} for index in range(100)]
    sample = calibrate_dedup._sample(pairs, 30)
    assert len(sample) == 30
    assert sample[: calibrate_dedup.HEAD_ROWS] == pairs[: calibrate_dedup.HEAD_ROWS]
    # The spread reaches well below the head instead of stopping at the top of the range.
    assert sample[-1]["cosine"] < pairs[calibrate_dedup.HEAD_ROWS]["cosine"]


def test_sample_returns_everything_when_the_cluster_has_few_pairs() -> None:
    pairs = [{"cosine": 0.9}, {"cosine": 0.5}]
    assert calibrate_dedup._sample(pairs, 30) == pairs
