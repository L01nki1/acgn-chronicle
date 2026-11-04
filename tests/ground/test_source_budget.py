"""Per-source quotation-budget regressions at generation and final-validation boundaries."""

from __future__ import annotations

import json

import pytest

from acgn_rag.core.ground import (
    GroundedAnswerer,
    GroundingValidationError,
    validate_grounded_answer,
)
from acgn_rag.core.ports import (
    MAX_QUOTE_CHARS,
    Citation,
    RetrievalQuery,
    RetrievedChunk,
    SourceType,
)


class FakeAnswerClient:
    def __init__(self, claims: list[dict[str, str | list[int]]]) -> None:
        self.response = json.dumps({"claims": claims})

    def complete(self, system: str, user: str, *, max_tokens: int) -> str:
        return self.response


def _query() -> RetrievalQuery:
    return RetrievalQuery(text="What was announced?", snapshot_id="20261104-abcdef012345")


def _chunk(
    text: str,
    *,
    index: int = 1,
    doc_id: str | None = None,
    source_type: SourceType = "official",
    url: str = "https://example.com/source",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"chunk-{index}",
        doc_id=doc_id or f"doc-{index}",
        snapshot_id=_query().snapshot_id,
        text=text,
        score=0.9,
        citation=Citation(
            source_type=source_type,
            publisher="Studio Alpha",
            title="Announcement",
            url=url,
            published_at="2026-11-04",
            chunk_id=f"chunk-{index}",
        ),
    )


@pytest.mark.parametrize(
    "claims",
    [
        [
            {
                "text": ("Raw source sentence with exact wording.\n\t" * 20)[:563],
                "evidence_indices": [1],
            }
        ],
        [
            {"text": "a" * 110, "evidence_indices": [1]},
            {"text": "b" * 110, "evidence_indices": [1]},
        ],
    ],
)
def test_answerer_rejects_over_budget_verbatim_source(
    claims: list[dict[str, str | list[int]]],
) -> None:
    source = " ".join(str(claim["text"]) for claim in claims)

    with pytest.raises(GroundingValidationError):
        GroundedAnswerer(FakeAnswerClient(claims)).answer(_query(), [_chunk(source)])


def test_answerer_rejects_source_copied_across_a_wordless_claim_boundary() -> None:
    first, second = "a" * 180, "b" * 180
    claims = [
        {"text": first, "evidence_indices": [1]},
        {"text": second, "evidence_indices": [1]},
    ]

    with pytest.raises(GroundingValidationError):
        GroundedAnswerer(FakeAnswerClient(claims)).answer(_query(), [_chunk(first + second)])


def test_answerer_rejects_two_media_quotes_reconstructing_one_body() -> None:
    first = (
        "The studio confirmed that production is complete and that the new series is scheduled "
        "for a worldwide release during the coming spring season."
    )
    second = (
        "The announcement also states that distribution will include television and online "
        "streaming, with further scheduling details to be published separately."
    )
    claims = [
        {"text": first, "evidence_indices": [1]},
        {"text": second, "evidence_indices": [1]},
    ]

    with pytest.raises(GroundingValidationError):
        GroundedAnswerer(FakeAnswerClient(claims)).answer(
            _query(), [_chunk(f"{first} {second}", source_type="media")]
        )


def test_answerer_rejects_quotes_across_chunks_of_the_same_document() -> None:
    first, second = "a" * 180, "b" * 180
    claims = [
        {"text": first, "evidence_indices": [1]},
        {"text": second, "evidence_indices": [2]},
    ]
    chunks = [
        _chunk(first, index=1, doc_id="shared-doc", url="https://example.com/shared"),
        _chunk(second, index=2, doc_id="shared-doc", url="https://example.com/shared"),
    ]

    with pytest.raises(GroundingValidationError):
        GroundedAnswerer(FakeAnswerClient(claims)).answer(_query(), chunks)


def test_final_guard_rejects_mutated_source_budget_across_same_document_chunks() -> None:
    first, second = "a" * 180, "b" * 180
    chunks = [
        _chunk(first, index=1, doc_id="shared-doc", url="https://example.com/shared"),
        _chunk(second, index=2, doc_id="shared-doc", url="https://example.com/shared"),
    ]
    answer = GroundedAnswerer(
        FakeAnswerClient(
            [
                {"text": "First supported summary.", "evidence_indices": [1]},
                {"text": "Second supported summary.", "evidence_indices": [2]},
            ]
        )
    ).answer(_query(), chunks)
    answer.answer = f"{first} [1] {second} [2]"

    with pytest.raises(GroundingValidationError):
        validate_grounded_answer(answer, _query().snapshot_id, chunks)


@pytest.mark.parametrize("field", ["answer", "abstain_reason"])
def test_final_guard_rejects_full_source_in_refusal_fields(field: str) -> None:
    body = "The official announcement contains a detailed public notice. " * 6
    chunk = _chunk(body)
    answer = GroundedAnswerer(FakeAnswerClient([])).answer(_query(), [chunk])
    setattr(answer, field, f"The retrieved notice does not establish the date. {body}")

    with pytest.raises(GroundingValidationError):
        validate_grounded_answer(answer, _query().snapshot_id, [chunk])


def test_final_guard_allows_long_paraphrased_refusal_explanations() -> None:
    chunk = _chunk("The official announcement has no release date.")
    answer = GroundedAnswerer(FakeAnswerClient([])).answer(_query(), [chunk])
    answer.answer = "The selected records do not resolve the requested question. " * 5
    answer.abstain_reason = "The requested date is absent from the selected records."

    validated = validate_grounded_answer(answer, _query().snapshot_id, [chunk])

    assert validated.abstained
    assert len(validated.answer) > MAX_QUOTE_CHARS
    assert validated.answer != validated.abstain_reason


@pytest.mark.parametrize(
    ("length", "raises"),
    [(MAX_QUOTE_CHARS, False), (MAX_QUOTE_CHARS + 1, True)],
)
def test_literal_source_budget_accepts_200_and_rejects_201(length: int, raises: bool) -> None:
    quote = "x" * length
    answerer = GroundedAnswerer(FakeAnswerClient([{"text": quote, "evidence_indices": [1]}]))
    if raises:
        with pytest.raises(GroundingValidationError):
            answerer.answer(_query(), [_chunk(quote)])
    else:
        assert not answerer.answer(_query(), [_chunk(quote)]).abstained


@pytest.mark.parametrize("shared_identity", ["doc_id", "url"])
def test_source_budget_groups_chunks_by_document_or_trimmed_url(
    shared_identity: str,
) -> None:
    first, second = "a" * 110, "b" * 110
    chunks = [
        _chunk(
            first,
            index=1,
            doc_id="shared" if shared_identity == "doc_id" else "first",
            url=" https://example.com/shared "
            if shared_identity == "url"
            else "https://example.com/1",
        ),
        _chunk(
            second,
            index=2,
            doc_id="shared" if shared_identity == "doc_id" else "second",
            url="https://example.com/shared"
            if shared_identity == "url"
            else "https://example.com/2",
        ),
    ]
    claims = [
        {"text": first, "evidence_indices": [1]},
        {"text": second, "evidence_indices": [2]},
    ]

    with pytest.raises(GroundingValidationError):
        GroundedAnswerer(FakeAnswerClient(claims)).answer(_query(), chunks)


def test_overlapping_chunks_count_one_claim_once_per_source_group() -> None:
    quote = "x" * 150
    chunks = [
        _chunk(f"prefix {quote}", index=1, doc_id="shared"),
        _chunk(f"{quote} suffix", index=2, doc_id="shared"),
    ]

    answer = GroundedAnswerer(FakeAnswerClient([{"text": quote, "evidence_indices": [1]}])).answer(
        _query(), chunks
    )

    assert not answer.abstained
    assert answer.answer == f"{quote} [1]"


def test_same_document_long_paraphrase_is_not_limited_by_answer_length() -> None:
    source = (
        "The official notice says production is complete, the series returns in spring 2027, "
        "Studio Alpha published the announcement, television and streaming distribution are "
        "planned, and the launch schedule remains unchanged."
    )
    claims = [
        {
            "text": "Production has concluded ahead of the planned spring 2027 return.",
            "evidence_indices": [1],
        },
        {
            "text": "Studio Alpha disclosed the update through its own announcement channel.",
            "evidence_indices": [1],
        },
        {
            "text": "Viewers are expected to receive both broadcast and streaming availability.",
            "evidence_indices": [1],
        },
        {
            "text": "The subsequent notice leaves the previously stated launch timetable intact.",
            "evidence_indices": [1],
        },
    ]

    answer = GroundedAnswerer(FakeAnswerClient(claims)).answer(_query(), [_chunk(source)])

    assert not answer.abstained
    assert len(answer.answer) > MAX_QUOTE_CHARS
    assert list(answer.cited_chunk_ids) == ["chunk-1"]


def test_long_paraphrase_across_sources_is_not_limited_by_answer_length() -> None:
    sources = [
        "Official notice: production finished and the series returns in spring 2027.",
        "The announcement was published by Studio Alpha on its official site.",
        "Distribution includes television broadcast plus streaming availability.",
        "The follow-up says the previously announced launch plan is unchanged.",
    ]
    claims = [
        {
            "text": "The series is scheduled to return during spring 2027 after production concluded.",
            "evidence_indices": [1],
        },
        {
            "text": "Studio Alpha made the announcement through its official public channel.",
            "evidence_indices": [2],
        },
        {
            "text": "The release covers television distribution and a simultaneous streaming rollout.",
            "evidence_indices": [3],
        },
        {
            "text": "A later notice confirms the launch plan remains unchanged across both platforms.",
            "evidence_indices": [4],
        },
    ]
    chunks = [_chunk(source, index=index) for index, source in enumerate(sources, 1)]

    answer = GroundedAnswerer(FakeAnswerClient(claims)).answer(_query(), chunks)

    assert not answer.abstained
    assert len(answer.answer) > MAX_QUOTE_CHARS
    assert list(answer.cited_chunk_ids) == [f"chunk-{index}" for index in range(1, 5)]
