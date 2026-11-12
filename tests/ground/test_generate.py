"""Observable grounded-generation behavior over request-local retrieved chunks."""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from acgn_rag.config import LLMConfig
from acgn_rag.core.ground import (
    ABSTAIN_ALL_CLAIMS_REJECTED,
    ABSTAIN_MODEL_EMPTY,
    ABSTAIN_NO_EVIDENCE,
    REJECTED_CLAIMS_REASON,
    Completion,
    GroundingValidationError,
    InvalidModelResponseError,
    InvalidRetrievedEvidenceError,
    PromptBudgetError,
)
from acgn_rag.core.ground.generate import (
    NO_EVIDENCE_REASON,
    UNSUPPORTED_REASON,
    GroundedAnswerer,
    build_grounding_prompt,
    generate_grounded_answer,
    generate_grounded_answer_run,
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

    def complete(self, system: str, user: str, *, max_tokens: int) -> Completion:
        self.calls.append((system, user, max_tokens))
        return Completion(text=self.response, finish_reason="stop")


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
    assert answer.answer == REJECTED_CLAIMS_REASON
    assert answer.abstain_reason == REJECTED_CLAIMS_REASON
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


def test_default_answer_budget_is_above_the_reasoning_floor_and_can_be_overridden() -> None:
    client = FakeAnswerClient(_response([]))
    generate_grounded_answer(_query(), [_chunk(1)], client)
    assert client.calls[0][2] == 2048

    override = FakeAnswerClient(_response([]))
    generate_grounded_answer(_query(), [_chunk(1)], override, max_tokens=3072)
    assert override.calls[0][2] == 3072


def test_run_diagnostics_record_completion_metadata_and_dropped_claims() -> None:
    class MetadataClient(FakeAnswerClient):
        def complete(self, system: str, user: str, *, max_tokens: int) -> Completion:
            self.calls.append((system, user, max_tokens))
            return Completion(
                text=_response(
                    [
                        {"text": "Supported fact.", "evidence_indices": [1]},
                        {"text": "Malformed claim.", "evidence_indices": ["1"]},
                    ]
                ),
                finish_reason="stop",
                usage={"prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 42},
            )

    run = generate_grounded_answer_run(_query(), [_chunk(1)], MetadataClient(""))

    assert run.answer.answer == "Supported fact. [1]"
    assert run.diagnostics.max_tokens == 2048
    assert run.diagnostics.finish_reason == "stop"
    assert run.diagnostics.usage == {
        "prompt_tokens": 30,
        "completion_tokens": 12,
        "total_tokens": 42,
    }
    assert run.diagnostics.dropped_claims == 1
    assert run.diagnostics.abstain_code is None


@pytest.mark.parametrize(
    "raw",
    [
        'Here is the JSON: {"claims": [{"text": "Fact one.", "evidence_indices": [1]}]} Thanks.',
        '\u0060\u0060\u0060json\n{"claims": [{"text": "Fact one.", "evidence_indices": [1]}]}\n\u0060\u0060\u0060',
    ],
)
def test_prose_and_fences_around_the_json_object_do_not_lose_valid_claims(raw: str) -> None:
    answer = generate_grounded_answer(_query(), [_chunk(1)], FakeAnswerClient(raw))

    assert answer.answer == "Fact one. [1]"


def test_one_malformed_claim_does_not_discard_the_valid_claims() -> None:
    client = FakeAnswerClient(
        _response(
            [
                {"text": "Fact one.", "evidence_indices": [1]},
                {"text": "Malformed.", "evidence_indices": ["1"]},
                {"text": "Fact two.", "evidence_indices": [2]},
            ]
        )
    )

    run = generate_grounded_answer_run(_query(), [_chunk(1), _chunk(2)], client)

    assert run.answer.answer == "Fact one. [1] Fact two. [2]"
    assert run.diagnostics.dropped_claims == 1


def test_extra_top_level_keys_do_not_discard_valid_claims() -> None:
    raw = json.dumps({"claims": [{"text": "Fact one.", "evidence_indices": [1]}], "note": "x"})

    answer = generate_grounded_answer(_query(), [_chunk(1)], FakeAnswerClient(raw))

    assert answer.answer == "Fact one. [1]"


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("这部作品什么时候播出?", "本次请求没有可用的检索证据。"),
        ("この作品はいつ放送されますか?", "このリクエストには検索で得られた根拠がありません。"),
        ("When is it broadcast?", NO_EVIDENCE_REASON),
    ],
)
def test_empty_evidence_refusal_uses_the_question_language(question: str, expected: str) -> None:
    query = RetrievalQuery(text=question, snapshot_id=_query().snapshot_id)

    answer = generate_grounded_answer(query, [], FakeAnswerClient(_response([])))

    assert answer.abstained
    assert answer.answer == expected
    assert answer.abstain_reason == expected


def test_abstention_diagnostics_distinguish_the_abstention_causes() -> None:
    no_evidence = generate_grounded_answer_run(_query(), [], FakeAnswerClient(_response([])))
    assert no_evidence.diagnostics.abstain_code == ABSTAIN_NO_EVIDENCE

    model_empty = generate_grounded_answer_run(
        _query(), [_chunk(1)], FakeAnswerClient(_response([]))
    )
    assert model_empty.diagnostics.abstain_code == ABSTAIN_MODEL_EMPTY

    rejected = generate_grounded_answer_run(
        _query(),
        [_chunk(1)],
        FakeAnswerClient(_response([{"text": "Unsupported.", "evidence_indices": [99]}])),
    )
    assert rejected.diagnostics.abstain_code == ABSTAIN_ALL_CLAIMS_REJECTED
    assert rejected.diagnostics.dropped_claims == 1


def test_from_config_passes_the_configured_answer_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ACGN_ANSWER_MAX_TOKENS", "4096")
    config = LLMConfig(answer_base_url="http://example.invalid", answer_model="m")
    client = FakeAnswerClient(_response([]))

    answerer = GroundedAnswerer.from_config(config, client)
    assert answerer.max_tokens == 4096
    answerer.answer(_query(), [_chunk(1)])
    assert client.calls[0][2] == 4096


def test_bracketed_year_is_allowed_in_claim_text() -> None:
    claim = "The [2024] edition launches in spring."

    answer = generate_grounded_answer(
        _query(),
        [_chunk(1)],
        FakeAnswerClient(_response([{"text": claim, "evidence_indices": [1]}])),
    )

    assert not answer.abstained
    assert "[2024]" in answer.answer


def test_prompt_evidence_budget_raises_instead_of_truncating() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id=f"chunk-{index}",
            doc_id=f"doc-{index}",
            snapshot_id=_query().snapshot_id,
            text="x" * 5000,
            score=0.5,
            citation=_citation(index),
        )
        for index in range(1, 4)
    ]
    client = FakeAnswerClient(_response([{"text": "A claim.", "evidence_indices": [1]}]))

    with pytest.raises(PromptBudgetError):
        generate_grounded_answer(_query(), chunks, client)
    assert client.calls == []


@pytest.mark.parametrize(
    "claim",
    [
        "See https://evil.example/fake for details.",
        "See doc-1::A::2 for details.",
    ],
)
def test_urls_and_chunk_ids_in_claim_text_are_dropped(claim: str) -> None:
    answer = generate_grounded_answer(
        _query(),
        [_chunk(1)],
        FakeAnswerClient(_response([{"text": claim, "evidence_indices": [1]}])),
    )

    assert answer.abstained
    assert "evil.example" not in answer.answer
    assert "doc-1::A::2" not in answer.answer


def test_broken_fence_before_a_valid_object_is_recovered() -> None:
    raw = "\u0060\u0060\u0060json\n{broken\n\u0060\u0060\u0060\nHere is the answer:\n" + _response(
        [{"text": "Fact one.", "evidence_indices": [1]}]
    )

    answer = generate_grounded_answer(_query(), [_chunk(1)], FakeAnswerClient(raw))

    assert answer.answer == "Fact one. [1]"


def test_malformed_first_object_does_not_hide_the_valid_one() -> None:
    raw = '{"claims": [}\n' + _response([{"text": "Fact one.", "evidence_indices": [1]}])

    answer = generate_grounded_answer(_query(), [_chunk(1)], FakeAnswerClient(raw))

    assert answer.answer == "Fact one. [1]"
