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
    generate,
)
from acgn_rag.core.ground.generate import (
    NO_EVIDENCE_REASON,
    SYSTEM_PROMPT,
    UNSUPPORTED_REASON,
    GroundedAnswerer,
    _refusal_message,
    build_grounding_prompt,
    generate_grounded_answer,
    generate_grounded_answer_run,
    validate_grounded_answer,
)
from acgn_rag.core.ground.policy import MEDIA_SOURCE_RULE, QUOTE_DISCIPLINE_RULE
from acgn_rag.core.ground.validation import (
    ClaimEvidence,
    allowed_media_quote,
    source_exposure_exceeds_budget,
)
from acgn_rag.core.ports import (
    MAX_CLAIM_CHARS,
    Citation,
    GroundedAnswer,
    RetrievalQuery,
    RetrievedChunk,
)

type JsonValue = str | int | float | bool | list[JsonValue] | dict[str, JsonValue] | None


class FakeAnswerClient:
    """Deterministic answer client whose mutation records whether a call occurred."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, str, int]] = []

    def complete(self, system: str, user: str, *, max_tokens: int) -> Completion:
        self.calls.append((system, user, max_tokens))
        return Completion(text=self.response)


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


def test_prompt_contains_the_actual_question_evidence_and_shared_media_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = _query()
    chunks = [_chunk(1), _chunk(2)]

    # The shipped prompt body must survive every mode. Two append-only rules exist, so the
    # "ends with the media rule" shape only holds with both off; with either on the media rule
    # must still be present verbatim.
    monkeypatch.setenv("ACGN_ANSWER_QUOTE_DISCIPLINE", "0")
    monkeypatch.setenv("ACGN_ANSWER_FOCUS", "0")
    system_off, user = build_grounding_prompt(query, chunks)
    assert system_off.endswith(MEDIA_SOURCE_RULE)

    monkeypatch.delenv("ACGN_ANSWER_QUOTE_DISCIPLINE", raising=False)
    monkeypatch.delenv("ACGN_ANSWER_FOCUS", raising=False)
    system_on, user_on = build_grounding_prompt(query, chunks)
    assert system_on.startswith(system_off)
    assert MEDIA_SOURCE_RULE in system_on
    assert user == user_on

    for system in (system_off, system_on):
        assert "answer only the facts asked for" in system
        assert "cite only those official blocks" in system
        assert "Keep verbatim quotes in their source language" in system
        assert "without an attribution prefix" in system
        assert "Cite only the media blocks that contain that excerpt" in system
    assert query.text in user
    assert chunks[0].text in user
    assert chunks[1].citation.url in user
    assert chunks[0].chunk_id not in user


def test_quote_discipline_is_on_by_default_and_the_escape_hatch_is_byte_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The adopted default appends the rule; disabling it restores the old prompt exactly."""
    # Isolate the quote-discipline switch from the other append-only rule.
    monkeypatch.setenv("ACGN_ANSWER_FOCUS", "0")
    monkeypatch.delenv("ACGN_ANSWER_QUOTE_DISCIPLINE", raising=False)
    assert generate.system_prompt() == SYSTEM_PROMPT + " " + QUOTE_DISCIPLINE_RULE

    monkeypatch.setenv("ACGN_ANSWER_QUOTE_DISCIPLINE", "0")
    assert generate.system_prompt() == SYSTEM_PROMPT

    # A blank value is not a literal "off": unset and empty both take the adopted default.
    monkeypatch.setenv("ACGN_ANSWER_QUOTE_DISCIPLINE", "")
    assert generate.system_prompt() == SYSTEM_PROMPT + " " + QUOTE_DISCIPLINE_RULE


def test_headline_quotes_are_charged_against_the_per_source_exposure_budget() -> None:
    """A title-only answer cannot spend the budget twice: four slices of one headline.

    `allowed_media_quote` accepts a claim that is a verbatim run of the chunk's body *or* its
    headline, so the exposure counter must charge both. Counting only the body left the
    headline free: this test would have passed every individual claim and reported a total of
    zero before the fix, letting one 215-character headline leave the source in full.
    """
    headline = (
        "The Idolmaster Cinderella Girls Announces New Anime Adaptation Of Its Starlight Stage "
        "Branch With A Returning Cast Panel Revealed At The Autumn Festival Broadcast And A "
        "Confirmed Two Season Run Beginning Next Spring With Production By A Returning Studio "
        "And An Original Soundtrack Release Announced Alongside The First Key Visual"
    )
    assert len(headline) > MAX_CLAIM_CHARS
    media_citation = Citation(
        source_type="media",
        publisher="ANN",
        title=headline,
        url="https://example.com/news/1",
        published_at="2026-11-04",
        chunk_id="chunk-1",
    )
    chunk = RetrievedChunk(
        chunk_id="chunk-1",
        doc_id="doc-1",
        snapshot_id="20261104-abcdef012345",
        text="Body text that shares nothing with the headline.",
        score=0.9,
        citation=media_citation,
    )
    slices = [headline[offset : offset + 60] for offset in range(0, len(headline), 60)]

    # Every slice is a legal individual quotation...
    for piece in slices:
        assert len(piece) <= MAX_CLAIM_CHARS
        # allowed_media_quote strips surrounding whitespace before comparing, so the returned
        # quotation is the piece without its leading space, not the raw slice.
        assert allowed_media_quote(piece, [chunk]) == piece.strip()

    # ...and together they exceed what one source may expose.
    claims = [ClaimEvidence(text=piece, chunks=(chunk,)) for piece in slices]
    assert sum(len(piece) for piece in slices) > MAX_CLAIM_CHARS
    assert source_exposure_exceeds_budget(claims, [chunk]) is True

    # A single slice of the same headline stays within budget.
    assert source_exposure_exceeds_budget(claims[:1], [chunk]) is False


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


@pytest.mark.parametrize(
    ("finish_reason", "usage"),
    [
        ("stop", {"prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 42}),
        (None, None),
    ],
)
def test_run_diagnostics_record_completion_metadata_and_dropped_claims(
    finish_reason: str | None,
    usage: dict[str, int] | None,
) -> None:
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
                finish_reason=finish_reason,
                usage=usage,
            )

    run = generate_grounded_answer_run(_query(), [_chunk(1)], MetadataClient(""))

    assert run.answer.answer == "Supported fact. [1]"
    assert run.diagnostics.max_tokens == 2048
    assert run.diagnostics.provider_completion is True
    assert run.diagnostics.finish_reason == finish_reason
    assert run.diagnostics.usage == usage
    assert run.diagnostics.dropped_claims == 1
    assert run.diagnostics.claims_seen == 2
    assert run.diagnostics.parse_ok == 1
    assert run.diagnostics.dropped_by_rule == {"invalid_shape": 1}
    assert run.diagnostics.abstain_code is None


def test_run_diagnostics_count_only_the_first_rule_that_drops_each_claim() -> None:
    quote = "放送直後よりNetflixにて独占配信"
    media = _chunk(2)
    media = media.model_copy(
        update={
            "text": f"放送・配信情報。{quote}。以上。",
            "citation": media.citation.model_copy(update={"source_type": "media"}),
        }
    )
    # Long enough to cross the per-source exposure budget on its own, whatever the limit is.
    copied_text = "The official release contains a detailed announcement of the schedule. " * (
        MAX_CLAIM_CHARS // 60 + 2
    )
    official = _chunk(3).model_copy(update={"text": copied_text})
    client = FakeAnswerClient(
        _response(
            [
                {"text": "Malformed index.", "evidence_indices": ["1"]},
                {"text": "Fabricated [1] at https://example.com/fact.", "evidence_indices": [99]},
                {"text": "See https://example.com/fact.", "evidence_indices": [99]},
                {"text": "Invalid reference.", "evidence_indices": [99]},
                {"text": "The series streams exclusively on Netflix.", "evidence_indices": [1, 2]},
                {"text": copied_text, "evidence_indices": [3]},
                {"text": quote, "evidence_indices": [2]},
            ]
        )
    )

    run = generate_grounded_answer_run(_query(), [_chunk(1), media, official], client)

    assert run.answer.answer == f'"{quote}" [1]'
    assert list(run.answer.cited_chunk_ids) == ["chunk-2"]
    assert run.diagnostics.claims_seen == 7
    assert run.diagnostics.parse_ok == 6
    assert run.diagnostics.dropped_by_rule == {
        "invalid_shape": 1,
        "citation_marker": 1,
        "forbidden_reference": 1,
        "bad_index": 1,
        "invalid_media_quote": 1,
        "exposure_budget": 1,
    }
    assert run.diagnostics.dropped_claims == sum(run.diagnostics.dropped_by_rule.values()) == 6


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
    assert no_evidence.diagnostics.provider_completion is False
    assert no_evidence.diagnostics.claims_seen == no_evidence.diagnostics.parse_ok == 0
    assert no_evidence.diagnostics.dropped_by_rule == {}

    model_empty = generate_grounded_answer_run(
        _query(), [_chunk(1)], FakeAnswerClient(_response([]))
    )
    assert model_empty.diagnostics.abstain_code == ABSTAIN_MODEL_EMPTY
    assert model_empty.diagnostics.provider_completion is True
    assert model_empty.diagnostics.claims_seen == model_empty.diagnostics.parse_ok == 0
    assert model_empty.diagnostics.dropped_by_rule == {}

    rejected = generate_grounded_answer_run(
        _query(),
        [_chunk(1)],
        FakeAnswerClient(_response([{"text": "Unsupported.", "evidence_indices": [99]}])),
    )
    assert rejected.diagnostics.abstain_code == ABSTAIN_ALL_CLAIMS_REJECTED
    assert rejected.diagnostics.dropped_claims == 1
    assert rejected.diagnostics.claims_seen == rejected.diagnostics.parse_ok == 1
    assert rejected.diagnostics.dropped_by_rule == {"bad_index": 1}
    assert rejected.diagnostics.provider_completion is True


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


# --- refusal-language detection (L4a) -------------------------------------------------

# The exact frozen-set items the raw kana rule misjudged (10/32 qa_set + 9/30 calibration): a
# Japanese work title carries kana into a question asked in English or Chinese, and the old
# rule answered those refusals in Japanese. Every row asserts the language its declared `lang`
# actually is, through the observable refusal message.
_MISJUDGED_QUESTIONS = [
    (
        "A-003",
        "en",
        "When does the second season of Let This Grieving Soul Retire! (嘆きの亡霊は引退したい) premiere?",
    ),
    ("A-004", "zh", "TV 动画《運命の巻戻士》宣布决定于何时播出？"),
    (
        "A-006",
        "zh",
        "《魔法つかいプリキュア！！～MIRAI DAYS～》宣布从哪一天开始在 ABC 电视台·朝日电视台系列深夜档播出？",
    ),
    ("A-008", "en", "When does MAPPA's Dorohedoro (ドロヘドロ) anime begin worldwide streaming?"),
    (
        "B-001",
        "en",
        "Which studio produced the short anime for Kenshi Yonezu's 2026 NHK soccer theme 「烏」?",
    ),
    ("B-005", "zh", "《らんま1/2》第3期在日本电视台播出后由哪个平台独家配信？"),
    (
        "B-007",
        "en",
        "Which edition of the Academy Awards nominated Toei Animation's Magic Candies (あめだま) for Best Animated Short?",
    ),
    (
        "D-004",
        "zh",
        "《進撃の巨人》The Final Season 在 MAPPA EXPO 展区每天的具体开放时间是几点到几点？",
    ),
    (
        "D-007",
        "zh",
        "『ざんねんないきもの事典』×《ドラゴンクエストモンスターズ４》的官方联动企划『ざんねんなまもの事典』定于哪一天开始公开？",
    ),
    (
        "D-008",
        "en",
        "Which venue hosted the first stop of The Powerpuff Girls POP UP STORE ガールズの休日, which opened on May 2, 2025?",
    ),
    (
        "A-901",
        "zh",
        "改编自朝日新闻出版《科学漫画生存》系列的 TV 动画《科学×冒険サバイバル！》在哪个电视台播出？",
    ),
    ("A-903", "zh", "にじさんじ 宣布出展的「Anime x Game Festival Korea 2026」于哪一天开幕？"),
    (
        "B-901",
        "en",
        "What did Toei Animation announce for Galaxy Express 999 (銀河鉄道999) in July 2026?",
    ),
    (
        "B-902",
        "en",
        "What is the title of the newly announced sequel to Expelled from Paradise (楽園追放)?",
    ),
    (
        "B-905",
        "en",
        "What did MAPPA unveil for the Chainsaw Man Assassin Arc (チェンソーマン 刺客篇)?",
    ),
    (
        "B-910",
        "en",
        "Which game engine does the announced Resident Evil Code: Veronica remake (バイオハザード RE:ベロニカ) use?",
    ),
    (
        "B-913",
        "zh",
        "TV动画《光が死んだ夏》×ヴィレッジヴァンガード的联动限定商品何时开始线上订购？",
    ),
    ("D-908", "en", "What is the exact total episode count of らんま1/2 season 3?"),
    ("D-909", "zh", "《サクラ大戦》30 周年纪念珠宝是否会在日本以外的市场发售？"),
]

# Mirror of the no-evidence refusal text per language; the test asserts the observable message
# rather than the private detector.
_NO_EVIDENCE_BY_LANG = {
    "en": NO_EVIDENCE_REASON,
    "zh": "本次请求没有可用的检索证据。",
    "ja": "このリクエストには検索で得られた根拠がありません。",
}

# The four items whose kana survives the quoted-span strip (the title is not inside any
# bracket); they pin the second-stage function-word fallback.
_BARE_KANA_IDS = ("D-008", "A-903", "B-913", "D-908")


@pytest.mark.parametrize(("qa_id", "lang", "question"), _MISJUDGED_QUESTIONS)
def test_refusal_language_matches_declared_lang(qa_id: str, lang: str, question: str) -> None:
    assert _refusal_message(ABSTAIN_NO_EVIDENCE, question) == _NO_EVIDENCE_BY_LANG[lang], qa_id


@pytest.mark.parametrize("qa_id", _BARE_KANA_IDS)
def test_bare_kana_title_is_resolved_by_function_words(qa_id: str) -> None:
    row = next(row for row in _MISJUDGED_QUESTIONS if row[0] == qa_id)

    assert _refusal_message(ABSTAIN_NO_EVIDENCE, row[2]) == _NO_EVIDENCE_BY_LANG[row[1]], qa_id


def test_refusal_language_falls_back_on_a_bare_title() -> None:
    # A question that is nothing but quoted material loses its frame; the raw script rule runs
    # on the original text, so a bare Japanese title still answers in Japanese.
    assert _refusal_message(ABSTAIN_NO_EVIDENCE, "プリキュア") == _NO_EVIDENCE_BY_LANG["ja"]


def test_refusal_language_keeps_genuine_japanese_questions_japanese() -> None:
    # ja input is not a target scenario (no ja QA items), but the defensive ja refusal texts
    # stay reachable for a real Japanese question.
    assert (
        _refusal_message(ABSTAIN_NO_EVIDENCE, "このゲームはいつ発売されますか？")
        == _NO_EVIDENCE_BY_LANG["ja"]
    )
