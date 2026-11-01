"""Contract examples are synthetic consumer fixtures, not HTTP or LLM run evidence."""

from __future__ import annotations

import json
from pathlib import Path

from acgn_rag.core.ports import GroundedAnswer

EXAMPLES = Path(__file__).resolve().parents[2] / "contracts" / "examples"


def test_no_evidence_example_is_a_valid_explicit_refusal() -> None:
    payload = json.loads((EXAMPLES / "ask-abstained.json").read_text(encoding="utf-8"))
    answer = GroundedAnswer.model_validate(payload)

    assert answer.model_dump(mode="json") == payload
    assert answer.abstained is True
    assert answer.answer
    assert answer.abstain_reason
    assert list(answer.citations) == []
    assert list(answer.cited_chunk_ids) == []
