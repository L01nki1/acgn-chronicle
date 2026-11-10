"""Compliance test (docs/COMPLIANCE.md C2/C3): no exported answer carries raw-zone text.

`POST /ask` serves `GroundedAnswer`; retrieved chunk text is generation-only context. The real
answerer is driven here with a deterministic fake client, so no LLM, network or store is involved.
If an assertion turns red, fix the implementation - never expose the body to make it pass.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from acgn_rag.core.ground import (
    Completion,
    GroundedAnswerer,
    GroundingValidationError,
    validate_grounded_answer,
)
from acgn_rag.core.ports import Citation, GroundedAnswer, RetrievalQuery, RetrievedChunk

BODY_MARKER = "FULL-ARTICLE-BODY-" + "x" * 4000
# A clipped body would drop the full marker but keep this head, so both are checked.
BODY_HEAD = "FULL-ARTICLE-BODY-"
PARAPHRASE = "The title ships on 14 March 2027."

SNAPSHOT_ID = "20261104-abcdef012345"
CHUNK_ID = "c_0001"

ANSWER_FIELDS = frozenset({"answer", "citations", "cited_chunk_ids", "abstained", "abstain_reason"})
CITATION_FIELDS = frozenset(
    {"source_type", "publisher", "title", "url", "published_at", "chunk_id"}
)
# Field names that could carry article text; the wire types must have none of them.
BODY_FIELD_NAMES = frozenset({"text", "body", "raw_text", "full_text"})


class FakeAnswerClient:
    """Deterministic answer client that records the prompt it was given."""

    def __init__(self, claims: list[dict[str, object]]) -> None:
        self.response = json.dumps({"claims": claims})
        self.calls: list[tuple[str, str, int]] = []

    def complete(self, system: str, user: str, *, max_tokens: int) -> Completion:
        self.calls.append((system, user, max_tokens))
        return Completion(text=self.response, finish_reason="stop")


def _citation() -> Citation:
    return Citation(
        source_type="official",
        publisher="Capcom",
        title="Press release",
        url="https://www.capcom.co.jp/news/0001",
        published_at="2026-10-27",
        chunk_id=CHUNK_ID,
    )


def _body_chunk() -> RetrievedChunk:
    """A retrieved chunk whose text is the raw article body plus one citable sentence."""
    return RetrievedChunk(
        chunk_id=CHUNK_ID,
        doc_id="d_0001",
        snapshot_id=SNAPSHOT_ID,
        text=f"{BODY_MARKER} {PARAPHRASE}",
        score=0.87,
        citation=_citation(),
    )


def _query() -> RetrievalQuery:
    return RetrievalQuery(text="When does the title ship?", snapshot_id=SNAPSHOT_ID)


def _grounded_answer() -> tuple[GroundedAnswer, RetrievedChunk]:
    chunk = _body_chunk()
    client = FakeAnswerClient([{"text": PARAPHRASE, "evidence_indices": [1]}])
    answer = GroundedAnswerer(client).answer(_query(), [chunk])

    # The body really was in scope: it reached the generation prompt, so the negative
    # assertions below test serialization rather than an untaken path.
    assert BODY_MARKER in client.calls[0][1]
    return answer, chunk


def _serialized(answer: GroundedAnswer) -> tuple[str, dict[str, Any]]:
    raw = answer.model_dump_json()
    return raw, json.loads(raw)


def _keys(payload: object) -> set[str]:
    if isinstance(payload, dict):
        found = set(payload)
        for child in payload.values():
            found |= _keys(child)
        return found
    if isinstance(payload, list):
        found = set()
        for child in payload:
            found |= _keys(child)
        return found
    return set()


def _string_leaves(payload: object) -> list[str]:
    if isinstance(payload, str):
        return [payload]
    if isinstance(payload, dict):
        return [leaf for child in payload.values() for leaf in _string_leaves(child)]
    if isinstance(payload, list):
        return [leaf for child in payload for leaf in _string_leaves(child)]
    return []


def _assert_no_body_text(raw: str, payload: dict[str, Any]) -> None:
    assert BODY_MARKER not in raw
    assert BODY_HEAD not in raw
    assert not any(BODY_MARKER in leaf or BODY_HEAD in leaf for leaf in _string_leaves(payload))
    assert _keys(payload).isdisjoint(BODY_FIELD_NAMES)


def test_grounded_answer_does_not_serialize_the_retrieved_body() -> None:
    answer, _ = _grounded_answer()
    raw, payload = _serialized(answer)

    assert not answer.abstained
    assert PARAPHRASE in payload["answer"]
    assert set(payload) == ANSWER_FIELDS
    _assert_no_body_text(raw, payload)


def test_refusal_does_not_serialize_the_retrieved_body() -> None:
    client = FakeAnswerClient([])
    answer = GroundedAnswerer(client).answer(_query(), [_body_chunk()])
    raw, payload = _serialized(answer)

    assert answer.abstained
    assert BODY_MARKER in client.calls[0][1]
    assert payload["citations"] == []
    assert payload["cited_chunk_ids"] == []
    assert isinstance(payload["answer"], str) and payload["answer"].strip()
    assert isinstance(payload["abstain_reason"], str) and payload["abstain_reason"].strip()
    _assert_no_body_text(raw, payload)


def test_serialized_citations_expose_metadata_only() -> None:
    answer, chunk = _grounded_answer()
    _, payload = _serialized(answer)

    citations = payload["citations"]
    assert len(citations) == 1
    for citation in citations:
        # Structural, not behavioural: nothing but provenance can be serialized, so there is
        # no field a body text could occupy even if generation regressed.
        assert set(citation) == CITATION_FIELDS
        assert citation == chunk.citation.model_dump(mode="json")
        assert _keys(citation).isdisjoint(BODY_FIELD_NAMES)


def test_validator_rejects_an_answer_that_copies_the_body() -> None:
    chunk = _body_chunk()
    tampered = GroundedAnswer(
        answer=f"{BODY_MARKER} [1]",
        citations=[chunk.citation],
        cited_chunk_ids=[chunk.chunk_id],
        abstained=False,
    )
    assert BODY_MARKER in tampered.model_dump_json()

    with pytest.raises(GroundingValidationError, match="source quotation budget"):
        validate_grounded_answer(tampered, SNAPSHOT_ID, [chunk])
