"""The human-labelling loop: page inputs and the label -> metrics correctness path.

`EVAL_PROTOCOL.md` 5 makes answer correctness a human judgement, and before this the runner
wrote an empty `labels.jsonl` that nothing ever read, so `correctness` stayed null however much
labelling was done. These tests pin both halves: what the page is given, and that a finished
labels file actually moves `metrics.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import render_answer_labels  # noqa: E402
import score_labels  # noqa: E402

QA = [
    {
        "qa_id": "delivery-A-001",
        "type": "A",
        "lang": "en",
        "answerable": True,
        "question": "When does the film open?",
        "gold_answer": "January 30",
        "gold_evidence": {"urls": ["https://example.com/a"], "spans": ["opens on January 30"]},
        "key_entities": ["January 30"],
    },
    {
        "qa_id": "delivery-D-001",
        "type": "D",
        "lang": "en",
        "answerable": False,
        "question": "Unknown thing?",
        "gold_answer": "",
        "gold_evidence": {"urls": [], "spans": []},
        "key_entities": [],
    },
    {
        "qa_id": "delivery-A-002",
        "type": "A",
        "lang": "zh",
        "answerable": True,
        "question": "何时开播？",
        "gold_answer": "2027",
        "gold_evidence": {"urls": ["https://example.com/b"], "spans": ["in 2027"]},
        "key_entities": ["2027"],
    },
]


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )


def _archive(tmp_path: Path, answers: list[dict]) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    _write(run / "answers.jsonl", answers)
    (run / "metrics.json").write_text(
        json.dumps(
            {
                "correctness": {"counts": {}, "rates": {"all": {"value": None, "n": 0}}},
                "reporting": {"correctness": {"available": False, "n": 0}},
                "retrieval": {"hit": {"all": {"value": 1.0, "n": 2}}},
            }
        ),
        encoding="utf-8",
    )
    (run / "labels.jsonl").write_text("", encoding="utf-8")
    return run


def _answer(qa_id: str, *, answered: bool, url: str = "https://example.com/a") -> dict:
    return {
        "qa_id": qa_id,
        "retrieved_urls": [url],
        "retrieved_chunk_ids": [f"{qa_id}::A::1"],
        "cited_chunk_ids": [f"{qa_id}::A::1"] if answered else [],
        "answer": "January 30 [1]" if answered else "Not in the knowledge base.",
        "citations": [
            {
                "n": 1,
                "publisher": "ANN",
                "title": "Film opens",
                "url": url,
                "source_type": "media",
                "published_at": "2026-01-01",
            }
        ]
        if answered
        else [],
        "abstained": not answered,
    }


def test_page_lists_only_answered_answerable_items(tmp_path: Path) -> None:
    qa_path = tmp_path / "qa.jsonl"
    _write(qa_path, QA)
    run = _archive(
        tmp_path,
        [
            _answer("delivery-A-001", answered=True),
            _answer(
                "delivery-D-001", answered=True
            ),  # unanswerable: graded by abstention, not here
            _answer("delivery-A-002", answered=False),  # abstained: no answer text to judge
        ],
    )

    items = render_answer_labels.build_items(run, qa_path)

    assert [item["qa_id"] for item in items] == ["delivery-A-001"]


def test_page_resolves_archives_written_before_the_namespace_rename(tmp_path: Path) -> None:
    """A pre-rename archive says `A-001`; the QA set now says `delivery-A-001`."""
    qa_path = tmp_path / "qa.jsonl"
    _write(qa_path, QA)
    run = _archive(tmp_path, [_answer("A-001", answered=True)])

    items = render_answer_labels.build_items(run, qa_path)

    assert len(items) == 1
    assert items[0]["qa_id"] == "delivery-A-001"
    assert items[0]["archived_id"] == "A-001"


def test_page_marks_which_citations_are_gold(tmp_path: Path) -> None:
    qa_path = tmp_path / "qa.jsonl"
    _write(qa_path, QA)
    run = _archive(
        tmp_path, [_answer("delivery-A-001", answered=True, url="https://example.com/x")]
    )

    item = render_answer_labels.build_items(run, qa_path)[0]

    assert item["citations"][0]["in_gold"] is False
    assert item["gold_urls"] == ["https://example.com/a"]


def test_finished_labels_move_metrics_correctness(tmp_path: Path) -> None:
    qa_path = tmp_path / "qa.jsonl"
    _write(qa_path, QA)
    run = _archive(
        tmp_path,
        [
            _answer("delivery-A-001", answered=True),
            _answer("delivery-A-002", answered=True, url="https://example.com/b"),
        ],
    )
    _write(
        run / "labels.jsonl",
        [
            {"qa_id": "delivery-A-001", "label": "fully_correct"},
            {"qa_id": "delivery-A-002", "label": "incorrect"},
            {"qa_id": "delivery-A-001", "label": None},
        ],
    )

    assert score_labels.main(["--run", str(run), "--qa", str(qa_path)]) == 0

    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["correctness"]["counts"]["fully_correct"] == 1
    assert metrics["correctness"]["counts"]["incorrect"] == 1
    assert metrics["correctness"]["rates"]["all"] == {"value": 0.5, "n": 2}
    assert metrics["correctness"]["rates"]["zh"]["n"] == 1
    assert metrics["reporting"]["correctness"] == {"available": True, "n": 2}
    # Everything the runner wrote elsewhere is untouched.
    assert metrics["retrieval"]["hit"]["all"]["value"] == 1.0


def test_unknown_label_is_rejected_rather_than_counted(tmp_path: Path) -> None:
    labels = tmp_path / "labels.jsonl"
    _write(labels, [{"qa_id": "delivery-A-001", "label": "mostly_fine"}])

    with pytest.raises(ValueError, match="unknown label"):
        score_labels.load_labels(labels)


def test_unlabelled_items_leave_correctness_unavailable(tmp_path: Path) -> None:
    qa_path = tmp_path / "qa.jsonl"
    _write(qa_path, QA)
    run = _archive(tmp_path, [_answer("delivery-A-001", answered=True)])

    assert score_labels.main(["--run", str(run), "--qa", str(qa_path)]) == 0

    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["correctness"]["rates"]["all"] == {"value": None, "n": 0}
    assert metrics["reporting"]["correctness"]["available"] is False
