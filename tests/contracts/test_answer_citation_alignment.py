"""A validated answer must not describe two different sets of supporting chunks."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from acgn_rag.core.ports import GroundedAnswer


def _citation(chunk_id: str) -> dict[str, str]:
    return {
        "source_type": "official",
        "publisher": "Independent Studio",
        "title": "Release announcement",
        "url": "https://example.com/release",
        "published_at": "2026-10-31",
        "chunk_id": chunk_id,
    }


def _answer(cited_ids: list[str]) -> dict[str, object]:
    return {
        "answer": "The announcement gives the release date.",
        "citations": [_citation("a"), _citation("b")],
        "cited_chunk_ids": cited_ids,
        "abstained": False,
        "abstain_reason": None,
    }


@pytest.mark.parametrize("cited_ids", [["a"], ["a", "b", "extra"], ["a", "wrong"], ["a", "a", "b"]])
def test_wire_answer_rejects_missing_extra_wrong_or_duplicated_ids(cited_ids: list[str]) -> None:
    with pytest.raises(ValidationError, match="cited_chunk_ids"):
        GroundedAnswer.model_validate_json(json.dumps(_answer(cited_ids)))


def test_matching_ids_do_not_require_a_second_list_order() -> None:
    answer = GroundedAnswer.model_validate(_answer(["b", "a"]))
    assert [citation.chunk_id for citation in answer.citations] == ["a", "b"]
    assert list(answer.cited_chunk_ids) == ["b", "a"]


def test_empty_refusal_support_is_consistent() -> None:
    payload = _answer([])
    payload.update(citations=[], abstained=True, abstain_reason="No evidence supports the answer.")
    assert GroundedAnswer.model_validate(payload).abstained


def test_fresh_validation_detects_alignment_lost_after_construction() -> None:
    answer = GroundedAnswer.model_validate(_answer(["a", "b"]))
    payload = answer.model_dump(mode="json")
    payload["cited_chunk_ids"] = []
    with pytest.raises(ValidationError, match="cited_chunk_ids"):
        GroundedAnswer.model_validate(payload)
