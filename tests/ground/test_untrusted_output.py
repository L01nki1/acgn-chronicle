"""Security regressions for treating model-authored grounded claims as untrusted input."""

from __future__ import annotations

import json

import pytest

from acgn_rag.core.ground import (
    GroundedAnswerer,
    GroundingValidationError,
    InvalidRetrievedEvidenceError,
    validate_grounded_answer,
)
from acgn_rag.core.ports import (
    Citation,
    RetrievalQuery,
    RetrievedChunk,
    SourceType,
)


class FakeAnswerClient:
    """Deterministic fake recording whether unsafe evidence reached the provider."""

    def __init__(self, claims: list[dict[str, str | list[int]]]) -> None:
        self.response = json.dumps({"claims": claims})
        self.calls = 0

    def complete(self, system: str, user: str, *, max_tokens: int) -> str:
        self.calls += 1
        return self.response


def _query() -> RetrievalQuery:
    return RetrievalQuery(
        text="What was announced and by whom?",
        snapshot_id="20261104-abcdef012345",
    )


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


@pytest.mark.parametrize("marker", ["[99]", "[99-100]", "[99; 100]", "[99, 100]"])
def test_answerer_rejects_model_authored_numeric_citation_markers(marker: str) -> None:
    answer = GroundedAnswerer(
        FakeAnswerClient([{"text": f"Untrusted claim {marker}", "evidence_indices": [1]}])
    ).answer(_query(), [_chunk("Supported fact")])

    assert answer.abstained
    assert marker not in answer.answer
    assert list(answer.citations) == []


def test_answerer_allows_ordinary_nonnumeric_bracket_text() -> None:
    claim = "The [Director's Cut] edition launches in spring."
    answer = GroundedAnswerer(FakeAnswerClient([{"text": claim, "evidence_indices": [1]}])).answer(
        _query(), [_chunk("A Director's Cut edition is scheduled for spring.")]
    )

    assert not answer.abstained
    assert answer.answer == f"{claim} [1]"


def test_answerer_rejects_media_only_commentary_paraphrase() -> None:
    source = "The reviewer called the episode the best of the season."
    paraphrase = "The review described the episode as the season's finest achievement."

    answer = GroundedAnswerer(
        FakeAnswerClient([{"text": paraphrase, "evidence_indices": [1]}])
    ).answer(_query(), [_chunk(source, source_type="media")])

    assert answer.abstained
    assert paraphrase not in answer.answer
    assert list(answer.citations) == []


@pytest.mark.parametrize("wrapper", ["", '"'])
def test_answerer_allows_attributed_in_budget_media_quote(wrapper: str) -> None:
    quote = "The release date is April 2."
    client = FakeAnswerClient([{"text": f"{wrapper}{quote}{wrapper}", "evidence_indices": [1]}])
    answer = GroundedAnswerer(client).answer(
        _query(), [_chunk(f"Report text: {quote} More reporting.", source_type="media")]
    )

    assert not answer.abstained
    assert answer.answer == f'"{quote}" [1]'
    assert answer.citations[0].source_type == "media"
    assert answer.citations[0].url == "https://example.com/source"


def test_final_guard_rejects_media_quote_without_enclosing_marks() -> None:
    quote = "The release date is April 2."
    chunk = _chunk(quote, source_type="media")
    client = FakeAnswerClient([{"text": quote, "evidence_indices": [1]}])
    answer = GroundedAnswerer(client).answer(_query(), [chunk])
    answer.answer = f"{quote} [1]"

    with pytest.raises(GroundingValidationError):
        validate_grounded_answer(answer, _query().snapshot_id, [chunk])


def test_final_guard_accepts_permuted_cited_ids_with_ordered_citations() -> None:
    chunks = [_chunk("Fact one.", index=1), _chunk("Fact two.", index=2)]
    client = FakeAnswerClient(
        [
            {"text": "Fact one.", "evidence_indices": [1]},
            {"text": "Fact two.", "evidence_indices": [2]},
        ]
    )
    answer = GroundedAnswerer(client).answer(_query(), chunks)
    answer.cited_chunk_ids = ["chunk-2", "chunk-1"]

    validated = validate_grounded_answer(answer, _query().snapshot_id, chunks)

    assert [citation.chunk_id for citation in validated.citations] == ["chunk-1", "chunk-2"]
    assert list(validated.cited_chunk_ids) == ["chunk-2", "chunk-1"]


def test_answerer_raises_for_blank_citation_url_before_provider_use() -> None:
    client = FakeAnswerClient([{"text": "Supported fact", "evidence_indices": [1]}])

    with pytest.raises(InvalidRetrievedEvidenceError):
        GroundedAnswerer(client).answer(_query(), [_chunk("Supported fact", url="  ")])

    assert client.calls == 0


def test_answerer_allows_publisher_name_as_a_supported_fact() -> None:
    claim = "Studio Alpha announced the release date."
    answer = GroundedAnswerer(FakeAnswerClient([{"text": claim, "evidence_indices": [1]}])).answer(
        _query(), [_chunk(claim)]
    )

    assert not answer.abstained
    assert answer.answer == f"{claim} [1]"
