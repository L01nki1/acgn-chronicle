"""Observable grounded-generation behavior over request-local retrieved chunks."""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from acgn_rag.core.ground import (
    GroundingValidationError,
    InvalidModelResponseError,
    InvalidRetrievedEvidenceError,
)
from acgn_rag.core.ground.generate import (
    NO_EVIDENCE_REASON,
    UNSUPPORTED_REASON,
    GroundedAnswerer,
    build_grounding_prompt,
    generate_grounded_answer,
    validate_grounded_answer,
)
from acgn_rag.core.ground.policy import MEDIA_SOURCE_RULE
from acgn_rag.core.ports import Citation, GroundedAnswer, RetrievalQuery, RetrievedChunk

type JsonValue = str | int | float | bool | list[JsonValue] | dict[str, JsonValue] | None


class FakeAnswerClient:
    """Deterministic answer client whose mutation records whether a call occurred."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, str, int]] = []

    def complete(self, system: str, user: str, *, max_tokens: int) -> str:
        self.calls.append((system, user, max_tokens))
        return self.response


def _citation(index: int, *, chunk_id: str | None = None) -> Citation:
    resolved_id = chunk_id or f"chunk-{index}"
    return Citation(
        source_type="official",
        publisher=f"Publisher {index}",
        title=f"Announcement {index}",
        url=f"https://example.com/news/{index}",
        published_at="2026-11-04",
        chunk_id=resolved_id,
    )


def _chunk(
    index: int,
    *,
    snapshot_id: str = "20261104-abcdef012345",
    citation_chunk_id: str | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"chunk-{index}",
        doc_id=f"doc-{index}",
        snapshot_id=snapshot_id,
        text=f"Evidence body {index} with a concrete fact.",
        score=0.9 - index / 10,
        citation=_citation(index, chunk_id=citation_chunk_id),
    )


def _query(snapshot_id: str = "20261104-abcdef012345") -> RetrievalQuery:
    return RetrievalQuery(
        text="What concrete facts were announced?",
        snapshot_id=snapshot_id,
    )


def _response(claims: Sequence[dict[str, JsonValue]]) -> str:
    return json.dumps({"claims": claims})


def test_valid_claims_map_local_indices_and_deduplicate_in_first_use_order() -> None:
    chunks = [_chunk(1), _chunk(2), _chunk(3)]
    client = FakeAnswerClient(
        _response(
            [
                {"text": "The second announcement confirms fact B.", "evidence_indices": [2, 1]},
                {"text": "The same source also confirms fact C.", "evidence_indices": [2]},
            ]
        )
    )

    answer = generate_grounded_answer(_query(), chunks, client)

    assert answer.answer == (
        "The second announcement confirms fact B. [1, 2] The same source also confirms fact C. [1]"
    )
    assert list(answer.cited_chunk_ids) == ["chunk-2", "chunk-1"]
    assert [citation.chunk_id for citation in answer.citations] == ["chunk-2", "chunk-1"]
    assert not answer.abstained


@pytest.mark.parametrize(
    "claims",
    [
        [{"text": "Unsupported.", "evidence_indices": [0]}],
        [{"text": "Unsupported.", "evidence_indices": [3]}],
    ],
)
def test_out_of_range_indices_produce_an_explicit_refusal(
    claims: Sequence[dict[str, JsonValue]],
) -> None:
    answer = generate_grounded_answer(
        _query(), [_chunk(1), _chunk(2)], FakeAnswerClient(_response(claims))
    )

    assert answer.abstained
    assert answer.answer == UNSUPPORTED_REASON
    assert answer.abstain_reason == UNSUPPORTED_REASON
    assert list(answer.citations) == []
    assert list(answer.cited_chunk_ids) == []


@pytest.mark.parametrize(
    "claims",
    [
        [{"text": "Malformed.", "evidence_indices": ["1"]}],
        [{"text": "Malformed.", "evidence_indices": [True]}],
    ],
)
def test_malformed_claim_schema_raises_component_error(
    claims: Sequence[dict[str, JsonValue]],
) -> None:
    with pytest.raises(InvalidModelResponseError):
        generate_grounded_answer(
            _query(), [_chunk(1), _chunk(2)], FakeAnswerClient(_response(claims))
        )


@pytest.mark.parametrize("response", ["", '{"claims":['])
def test_empty_or_malformed_provider_content_raises_component_error(response: str) -> None:
    with pytest.raises(InvalidModelResponseError):
        generate_grounded_answer(_query(), [_chunk(1)], FakeAnswerClient(response))


def test_valid_empty_claim_list_is_a_knowledge_base_refusal() -> None:
    answer = generate_grounded_answer(_query(), [_chunk(1)], FakeAnswerClient(_response([])))

    assert answer.abstained
    assert answer.answer == UNSUPPORTED_REASON


def test_empty_evidence_refuses_without_calling_the_provider() -> None:
    client = FakeAnswerClient(_response([]))

    answer = generate_grounded_answer(_query(), [], client)

    assert answer.abstained
    assert answer.answer == NO_EVIDENCE_REASON
    assert client.calls == []


def test_concrete_answerer_matches_the_evaluation_answer_shape() -> None:
    answerer = GroundedAnswerer(
        FakeAnswerClient(_response([{"text": "Supported fact.", "evidence_indices": [1]}]))
    )

    answer = answerer.answer(_query(), [_chunk(1)])

    assert isinstance(answer, GroundedAnswer)
    assert answer.answer == "Supported fact. [1]"


def test_mixed_supported_and_unsupported_claims_publish_only_supported_claims() -> None:
    client = FakeAnswerClient(
        _response(
            [
                {"text": "Supported fact.", "evidence_indices": [2]},
                {"text": "No evidence.", "evidence_indices": []},
                {"text": "Wrong request index.", "evidence_indices": [99]},
            ]
        )
    )

    answer = generate_grounded_answer(_query(), [_chunk(1), _chunk(2)], client)

    assert answer.answer == "Supported fact. [1]"
    assert list(answer.cited_chunk_ids) == ["chunk-2"]
    assert not answer.abstained


@pytest.mark.parametrize(
    "chunks",
    [
        [_chunk(1, snapshot_id="20261103-fedcba654321")],
        [_chunk(1, citation_chunk_id="another-request-chunk")],
        [_chunk(1), _chunk(1)],
    ],
)
def test_request_isolation_failure_raises_before_model_use(
    chunks: Sequence[RetrievedChunk],
) -> None:
    client = FakeAnswerClient(_response([{"text": "Fact.", "evidence_indices": [1]}]))

    with pytest.raises(InvalidRetrievedEvidenceError):
        generate_grounded_answer(_query(), chunks, client)

    assert client.calls == []


def test_prompt_contains_the_actual_question_evidence_and_shared_media_policy() -> None:
    query = _query()
    chunks = [_chunk(1), _chunk(2)]

    system, user = build_grounding_prompt(query, chunks)

    assert system.endswith(MEDIA_SOURCE_RULE)
    assert query.text in user
    assert chunks[0].text in user
    assert chunks[1].citation.url in user
    assert chunks[0].chunk_id not in user


def test_final_guard_rejects_a_non_request_citation() -> None:
    chunk = _chunk(1)
    foreign = GroundedAnswer(
        answer="Invented support.",
        citations=[_citation(9)],
        cited_chunk_ids=["chunk-9"],
        abstained=False,
        abstain_reason=None,
    )

    with pytest.raises(GroundingValidationError):
        validate_grounded_answer(foreign, _query().snapshot_id, [chunk])


@pytest.mark.parametrize(
    "answer",
    [
        GroundedAnswer(
            answer="No support.",
            citations=[_chunk(1).citation],
            cited_chunk_ids=[_chunk(1).chunk_id],
            abstained=True,
            abstain_reason="No support.",
        ),
        GroundedAnswer(
            answer="No support.",
            citations=[],
            cited_chunk_ids=[],
            abstained=True,
            abstain_reason="",
        ),
    ],
)
def test_final_guard_rejects_inconsistent_refusal(answer: GroundedAnswer) -> None:
    with pytest.raises(GroundingValidationError):
        validate_grounded_answer(answer, _query().snapshot_id, [_chunk(1)])


def test_final_guard_accepts_different_nonblank_refusal_explanations() -> None:
    refusal = GroundedAnswer(
        answer="The answer cannot be established.",
        citations=[],
        cited_chunk_ids=[],
        abstained=True,
        abstain_reason="The selected evidence lacks support.",
    )

    validated = validate_grounded_answer(refusal, _query().snapshot_id, [])

    assert validated.answer == "The answer cannot be established."
    assert validated.abstain_reason == "The selected evidence lacks support."


@pytest.mark.parametrize(
    "mutated_text",
    [
        "Fact one. [99] Fact two. [1]",
        "Fact two. [2] Fact one. [1]",
        "Fact one. [1]",
    ],
)
def test_final_guard_rejects_mutated_markers_on_generated_answer(mutated_text: str) -> None:
    chunks = [_chunk(1), _chunk(2)]
    answer = generate_grounded_answer(
        _query(),
        chunks,
        FakeAnswerClient(
            _response(
                [
                    {"text": "Fact one.", "evidence_indices": [1]},
                    {"text": "Fact two.", "evidence_indices": [2]},
                ]
            )
        ),
    )
    answer.answer = mutated_text

    with pytest.raises(GroundingValidationError):
        validate_grounded_answer(answer, _query().snapshot_id, chunks)
