"""Headline-only QA representatives: does the title itself state the gold answer?

Owner: D (WP5 tooling) | Spec: docs/EVAL_PROTOCOL.md, issue #6 comment 6743557815, issue #44

F's URL precheck found answerable QA items whose only URL-matched representative is a
headline-only record. These tests pin the *checking tool*, not the verdicts: the verdicts are a
reading aid for D's QA adjudication, and a test that froze them would turn a judgement call into
a regression.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import qa_headline_check  # noqa: E402


def test_norm_folds_case_and_width() -> None:
    # Full-width Latin + ideographic space collapse to the ASCII form NFKC produces.
    assert qa_headline_check._norm(
        "\uff24\uff32\uff21\uff27\uff2f\uff2e\u3000\uff22\uff21\uff2c\uff2c"
    ) == (qa_headline_check._norm("dragon ball"))
    assert qa_headline_check._contains("第97回アカデミー賞", "97th") is False
    assert qa_headline_check._contains("第97回アカデミー賞", "第97回") is True


def test_answer_forms_cover_the_known_readings() -> None:
    # The gold value and the form the headline actually uses must both be listed.
    assert "2025年11月14日" in qa_headline_check.ANSWER_FORMS["2025-11-14"]
    assert "70周年" in qa_headline_check.ANSWER_FORMS["70th"]
    assert "第97回" in qa_headline_check.ANSWER_FORMS["97th"]
    # A question asking "which platform" is not answered by a premiere date.
    assert "Netflix" not in qa_headline_check.ANSWER_FORMS
    # A bare month would accept a headline about a different year.
    assert "October" not in qa_headline_check.ANSWER_FORMS["October 2025"]


def test_headline_only_map_is_well_formed() -> None:
    assert len(qa_headline_check.HEADLINE_ONLY) == 11
    for qa_id, doc_id in qa_headline_check.HEADLINE_ONLY.items():
        assert qa_id.count("-") == 1, qa_id
        assert doc_id.startswith("doc_") and len(doc_id) == 68, (qa_id, doc_id)
    assert set(qa_headline_check.SECOND_REPRESENTATIVE) <= set(qa_headline_check.HEADLINE_ONLY)


def test_load_qa_reads_the_committed_set() -> None:
    qa = qa_headline_check.load_qa(Path("eval/qa_set.jsonl"))
    assert "A-001" in qa
    assert all("key_entities" in item for item in qa.values())


def test_load_qa_keeps_unicode_line_separators_inside_records(tmp_path: Path) -> None:
    """U+2028 / U+2029 / U+0085 are legal inside a JSON string, not record boundaries."""
    path = tmp_path / "qa.jsonl"
    rows = [
        {"qa_id": "A-001", "key_entities": ["2025年10月"], "gold_answer": "10月\u2028放送開始"},
        {"qa_id": "A-002", "key_entities": ["Netflix"], "gold_answer": "配信\u2029決定"},
        {"qa_id": "A-003", "key_entities": ["70th"], "gold_answer": "70周年\u0085記念"},
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    qa = qa_headline_check.load_qa(path)

    assert set(qa) == {"A-001", "A-002", "A-003"}
    assert qa["A-001"]["gold_answer"] == "10月\u2028放送開始"
    assert qa["A-002"]["gold_answer"] == "配信\u2029決定"
    assert qa["A-003"]["gold_answer"] == "70周年\u0085記念"


def test_check_rejects_headline_ids_missing_from_the_qa_set(tmp_path: Path, monkeypatch) -> None:
    """A renumbered QA item must fail loudly, not become a silent empty row."""
    qa = tmp_path / "qa.jsonl"
    qa.write_text(
        json.dumps({"qa_id": "A-001", "key_entities": [], "gold_evidence": {"spans": []}}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        qa_headline_check,
        "HEADLINE_ONLY",
        {"A-001": "doc_" + "0" * 64, "D-911": "doc_" + "1" * 64},
    )
    monkeypatch.setattr(qa_headline_check, "SECOND_REPRESENTATIVE", {})

    with pytest.raises(ValueError, match="absent from the QA set: D-911"):
        qa_headline_check.check("20261105-d797b5f0ae8e", tmp_path, qa)


def test_check_reports_a_missing_representative_without_raising(
    tmp_path: Path, monkeypatch
) -> None:
    """A doc_id that is absent from the snapshot is reported, not crashed on."""
    qa = tmp_path / "qa.jsonl"
    qa.write_text(
        json.dumps(
            {
                "qa_id": "A-001",
                "key_entities": ["2025年10月"],
                "gold_evidence": {"spans": []},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(qa_headline_check, "HEADLINE_ONLY", {"A-001": "doc_" + "0" * 64})
    monkeypatch.setattr(qa_headline_check, "SECOND_REPRESENTATIVE", {})
    monkeypatch.setattr(qa_headline_check, "iter_snapshot_items", lambda *a, **k: iter(()))

    report = qa_headline_check.check("20261105-d797b5f0ae8e", tmp_path, qa)

    assert report["checked"] == 1
    assert report["rows"][0]["status"] == "representative_not_found"
    assert report["needs_adjudication"] == 1


def test_cli_requires_a_snapshot() -> None:
    try:
        qa_headline_check.main([])
    except SystemExit as exc:
        assert exc.code == 2
