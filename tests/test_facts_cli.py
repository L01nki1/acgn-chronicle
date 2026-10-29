"""CLI contract for scripts/extract_facts.py: dry-run prints a summary, real runs write atomically."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


class FakeLLMClient:
    """Offline LLMClient: pops canned responses and records every call."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(self, system: str, user: str, *, max_tokens: int) -> str:
        self.calls.append({"system": system, "user": user, "max_tokens": max_tokens})
        return self.responses.pop(0)


def _row(index: int, text: str | None = None) -> dict[str, str]:
    return {
        "chunk_id": f"doc{index:02d}::config::0",
        "text": text if text is not None else f"Announcement {index}: the new season starts soon.",
        "url": f"https://example.com/news/{index}",
        "publisher": "Example Publisher",
    }


def _payload(count: int) -> str:
    items = [
        {
            "title": f"Work {i}",
            "release_date": "2026-10-05",
            "platforms": ["Switch", "PS5"],
            "event_type": "release",
            "relevance": 0.9,
        }
        for i in range(count)
    ]
    return json.dumps(items, ensure_ascii=False)


def _write_chunks(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _main(argv: list[str], *, client: Any = None) -> int:
    sys.path.insert(0, str(SCRIPTS_DIR))
    import extract_facts

    return extract_facts.main(argv, client=client)


def test_dry_run_prints_a_summary_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ACGN_DATA_DIR", str(tmp_path / "data"))
    rows = [_row(i) for i in range(12)]
    rows[3]["text"] = "第2四半期決算短信を公開しました"  # one IR item inside batch 1
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path, rows)

    code = _main(["--snapshot", "snap-dry", "--input", str(chunks_path), "--dry-run"])

    assert code == 0
    out = capsys.readouterr().out
    assert "chunks=12" in out and "batches=2" in out
    assert "planned_calls=2" in out and "ir_skipped=1" in out
    assert "nothing was written" in out
    assert not (tmp_path / "data" / "facts").exists()


def test_limit_caps_the_input_before_anything_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ACGN_DATA_DIR", str(tmp_path / "data"))
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path, [_row(i) for i in range(12)])

    code = _main(
        ["--snapshot", "snap-lim", "--input", str(chunks_path), "--limit", "5", "--dry-run"]
    )

    assert code == 0
    out = capsys.readouterr().out
    assert "chunks=5" in out and "batches=1" in out


def test_real_run_writes_facts_jsonl_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ACGN_DATA_DIR", str(tmp_path / "data"))
    rows = [_row(i) for i in range(11)]
    rows.append(_row(99, text="株主総会のお知らせ"))  # IR item in the second batch
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path, rows)

    client = FakeLLMClient([_payload(10), _payload(1)])  # batch 1: 10 candidates; batch 2: 1
    code = _main(["--snapshot", "snap-1", "--input", str(chunks_path)], client=client)

    assert code == 0
    facts_dir = tmp_path / "data" / "facts" / "snap-1"
    facts_path = facts_dir / "facts.jsonl"
    assert facts_path.exists()
    assert not facts_path.with_name("facts.jsonl.tmp").exists()  # atomic: no leftover temp file
    lines = [json.loads(line) for line in facts_path.read_text(encoding="utf-8").splitlines()]
    # 11 facts x (release_date + 2 platforms + event_type) = 44 rows
    assert len(lines) == 44
    required = {"media", "field", "value", "chunk_id", "url", "publisher"}
    for row in lines:
        assert required <= set(row)
        assert row["url"].startswith("https://example.com/news/")
    chunk_ids = {row["chunk_id"] for row in lines}
    assert chunk_ids <= {row["chunk_id"] for row in rows}
    assert "doc99::config::0" not in chunk_ids  # the IR chunk yields no fact rows

    skipped = [
        json.loads(line)
        for line in (facts_dir / "ir_skipped.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [(row["chunk_id"], row["ingest_category"]) for row in skipped] == [
        ("doc99::config::0", "corporate")
    ]

    out = capsys.readouterr().out
    assert "batches=2" in out and "calls=2" in out
    assert "extracted_rows=44" in out and "ir_skipped=1" in out


def test_missing_input_is_reported_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ACGN_DATA_DIR", str(tmp_path / "data"))
    code = _main(["--snapshot", "snap-x", "--input", str(tmp_path / "nope.jsonl"), "--dry-run"])
    assert code == 2
    assert "NOT FOUND" in capsys.readouterr().err
