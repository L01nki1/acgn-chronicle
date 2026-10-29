"""QA set rules (docs/EVAL_PROTOCOL.md §3): the quotas are only real if something enforces them."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from acgn_rag.core.eval.qa import load_jsonl, summary, validate


def _item(
    qa_id: str,
    qa_type: str,
    *,
    answerable: bool = True,
    overlap: bool = False,
    **overrides: Any,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "qa_id": qa_id,
        "type": qa_type,
        "lang": "en",
        "answerable": answerable,
        "question": f"question {qa_id}",
        "gold_answer": "2026-10-05",
        "gold_evidence": {
            "urls": ["https://www.toei-anim.co.jp/news/1"],
            "publisher": "Toei Animation",
            "spans": ["will begin airing on 2026-10-05"],
        },
        "key_entities": ["2026-10-05"],
        "window": ["2023-10-28", "2026-10-28"],
        "annotator": "D",
        "verified_by": "F",
        "overlap": overlap,
    }
    item.update(overrides)
    return item


def _valid_set() -> list[dict[str, Any]]:
    """11 A + 9 B + 7 C + 8 D = 35 items; A is 31% of the total; 5 unanswerable; 10 overlap."""
    items: list[dict[str, Any]] = []
    suffix = 1
    for qa_type, count in (("A", 11), ("B", 9), ("C", 7), ("D", 8)):
        for _ in range(count):
            id_prefix = qa_type
            items.append(
                _item(
                    f"{id_prefix}-{suffix:03d}",
                    qa_type,
                    answerable=not (qa_type == "D" and suffix > 30),
                )
            )
            suffix += 1
    for index, item in enumerate(items[:10]):
        items[index] = {**item, "overlap": True}
    for index in range(8):  # the language quota needs 8 zh items in a valid set
        items[index] = {**items[index], "lang": "zh"}
    return items


def _calibration(size: tuple[int, int] = (20, 10)) -> list[dict[str, Any]]:
    answerable, unanswerable = size
    items = [_item(f"CAL-{i:03d}", "A") for i in range(answerable)] + [
        _item(f"CAL-{i:03d}", "D", answerable=False) for i in range(unanswerable)
    ]
    for index, item in enumerate(items):
        item["question"] = f"calibration question {index}"
        item["qa_id"] = f"CAL-{index:03d}"
    return items


def test_a_valid_set_passes() -> None:
    items = _valid_set()
    assert validate(items, calibration=_calibration()) == []


def test_summary_counts_what_the_report_needs() -> None:
    counts = summary(_valid_set())
    assert counts["total"] == 35
    assert counts["by_type"] == {"A": 11, "B": 9, "C": 7, "D": 8}
    assert counts["by_lang"] == {"en": 27, "zh": 8}
    assert counts["unanswerable"] == 5
    assert counts["overlap"] == 10


def test_total_below_thirty_fails() -> None:
    problems = validate(_valid_set()[:20])
    assert any("total 20 < 30" in p for p in problems), problems


def test_type_a_above_one_third_fails() -> None:
    items = _valid_set()
    for index in range(4):  # 11 A -> 15 A of 35 = 43%
        items.append(_item(f"A-{900 + index:03d}", "A"))
    problems = validate(items)
    assert any("one third" in p for p in problems), problems


def test_unanswerable_minimum_is_enforced() -> None:
    items = [_item(i["qa_id"], i["type"], overlap=i["overlap"]) for i in _valid_set()]
    problems = validate(items)
    assert any("unanswerable" in p for p in problems), problems


def test_overlap_must_be_exactly_ten() -> None:
    items = _valid_set()
    items[0] = {**items[0], "overlap": False}
    problems = validate(items)
    assert any("need exactly 10" in p for p in problems), problems


def test_answerable_item_without_url_fails() -> None:
    items = _valid_set()
    items[0] = {**items[0], "gold_evidence": {"urls": [], "publisher": "x", "spans": []}}
    problems = validate(items)
    assert any("at least one gold_evidence url" in p for p in problems), problems


def test_url_must_be_https() -> None:
    items = _valid_set()
    items[0] = {**items[0], "gold_evidence": {"urls": ["http://example.com/a"]}}
    problems = validate(items)
    assert any("must be https" in p for p in problems), problems


def test_unanswerable_must_be_type_d() -> None:
    items = _valid_set()
    items[0] = {**items[0], "answerable": False}
    problems = validate(items)
    assert any("must be type D" in p for p in problems), problems


def test_missing_lang_fails() -> None:
    items = _valid_set()
    del items[0]["lang"]
    problems = validate(items)
    assert any("'lang'" in p for p in problems), problems


def test_lang_outside_en_zh_fails() -> None:
    items = _valid_set()
    items[0] = {**items[0], "lang": "jp"}
    problems = validate(items)
    assert any("lang must be one of" in p for p in problems), problems


def test_language_quota_is_enforced() -> None:
    # the same set with every zh mark dropped: the quota fails, not just the wording
    items = [_item(i["qa_id"], i["type"], overlap=i["overlap"]) for i in _valid_set()]
    problems = validate(items)
    assert any("lang='zh'" in p for p in problems), problems


def test_annotator_cannot_verify_themselves() -> None:
    items = _valid_set()
    items[0] = {**items[0], "verified_by": "D"}
    problems = validate(items)
    assert any("self-review" in p for p in problems), problems


def test_duplicate_ids_fail() -> None:
    items = _valid_set()
    items[1] = {**items[1], "qa_id": items[0]["qa_id"]}
    problems = validate(items)
    assert any("duplicate qa_id" in p for p in problems), problems


def test_calibration_must_be_disjoint_or_the_number_is_optimistic() -> None:
    items = _valid_set()
    calibration = _calibration()
    calibration[0] = {**calibration[0], "question": items[0]["question"]}
    problems = validate(items, calibration=calibration)
    assert any("shares 1 question" in p for p in problems), problems


def test_calibration_sizes_are_checked() -> None:
    problems = validate(_valid_set(), calibration=_calibration((20, 5)))
    assert any("need 10" in p for p in problems), problems


def test_load_jsonl_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(_valid_set()[0]) + "\n{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_jsonl(path)


def test_cli_reports_and_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    qa_path = tmp_path / "qa_set.jsonl"
    qa_path.write_text(
        "\n".join(json.dumps(item) for item in _valid_set()) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        sys, "argv", ["check_qa.py", "--qa", str(qa_path), "--calibration", str(tmp_path / "none")]
    )
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from check_qa import main

    assert main() == 0
