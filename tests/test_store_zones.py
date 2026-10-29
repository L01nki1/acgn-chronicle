"""Zone boundary (C2): facts zone is publishable JSONL, the raw zone is refused."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from acgn_rag.core.store.zones import assert_publishable, write_facts_zone


def _row(media: str = "anime", value: str = "2027-04-05") -> dict:
    return {
        "media": media,
        "field": "release_date",
        "value": value,
        "chunk_id": "doc-1::A::0",
        "url": "https://example.com/1",
        "publisher": "Capcom",
    }


def test_assert_publishable_rejects_the_raw_zone() -> None:
    for path in ("data/raw/facts.jsonl", "data/raw/2026/facts.jsonl", "/x/data/raw/sub/f.jsonl"):
        with pytest.raises(ValueError, match="raw zone"):
            assert_publishable(Path(path))


def test_assert_publishable_accepts_non_raw_paths() -> None:
    # "raw-notes" is a file name, not a zone marker; a directory named raw is one.
    assert assert_publishable(Path("data/facts/facts.jsonl"))
    assert assert_publishable(Path("data/facts/raw-notes.jsonl"))
    with pytest.raises(ValueError, match="raw zone"):
        assert_publishable(Path("data/facts/raw/nested.jsonl"))


def test_write_facts_zone_round_trips_jsonl(tmp_path: Path) -> None:
    rows = [_row(), _row(media="game", value="Steam")]
    path = write_facts_zone(rows, tmp_path / "data" / "facts" / "facts.jsonl")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == rows


def test_write_facts_zone_is_atomic_no_tmp_left_behind(tmp_path: Path) -> None:
    write_facts_zone([_row()], tmp_path / "data" / "facts" / "facts.jsonl")
    assert list((tmp_path / "data" / "facts").glob("*.tmp")) == []


def test_write_facts_zone_validates_before_touching_disk(tmp_path: Path) -> None:
    target = write_facts_zone([_row()], tmp_path / "data" / "facts" / "facts.jsonl")
    before = target.read_text(encoding="utf-8")
    bad = _row()
    del bad["url"]  # FactUpdate.url is required: an invalid row must not half-replace the file
    with pytest.raises(ValidationError, match="url"):
        write_facts_zone([bad], target)
    assert target.read_text(encoding="utf-8") == before
    assert list(target.parent.glob("*.tmp")) == []


def test_write_facts_zone_refuses_raw_zone_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="raw zone"):
        write_facts_zone([_row()], tmp_path / "data" / "raw" / "facts.jsonl")


def test_write_facts_zone_empty_rows_write_an_empty_file(tmp_path: Path) -> None:
    path = write_facts_zone([], tmp_path / "data" / "facts" / "facts.jsonl")
    assert path.read_text(encoding="utf-8") == ""
