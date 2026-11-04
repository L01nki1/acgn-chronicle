"""Grounding boundary validation and source-exposure policy."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from acgn_rag.core.ports import (
    MAX_QUOTE_CHARS,
    Citation,
    GroundedAnswer,
    RetrievalQuery,
    RetrievedChunk,
)

NUMERIC_CITATION_MARKER: Final = re.compile(r"\[\s*\d+(?:\s*(?:[,;]|[-\u2013\u2014])\s*\d+)*\s*\]")
_GENERATED_CITATION_MARKER: Final = re.compile(r"\[\s*(?P<refs>\d+(?:\s*,\s*\d+)*)\s*\]")
_QUOTE_PAIRS: Final = {
    '"': '"',
    "'": "'",
    "\u201c": "\u201d",
    "\u2018": "\u2019",
    "\u300c": "\u300d",
}


class GroundingComponentError(Exception):
    """A grounding component failed before producing a valid response."""


class InvalidRetrievedEvidenceError(GroundingComponentError):
    """Retrieved evidence is inconsistent with the current request."""


class InvalidModelResponseError(GroundingComponentError):
    """The provider returned content that cannot be parsed as the answer schema."""


class ProviderResponseError(GroundingComponentError):
    """The provider returned an incomplete or non-final completion response."""


class GroundingValidationError(GroundingComponentError):
    """The final payload is inconsistent with its request-local evidence."""


@dataclass(frozen=True, slots=True)
class ClaimEvidence:
    """One accepted claim before presentation markers are added."""

    text: str
    chunks: tuple[RetrievedChunk, ...]


def contains_numeric_citation_marker(text: str) -> bool:
    """Return whether untrusted model text contains a numeric citation-like marker."""
    return NUMERIC_CITATION_MARKER.search(text) is not None


def validate_retrieved_evidence(
    query: RetrievalQuery,
    chunks: Sequence[RetrievedChunk],
) -> None:
    """Raise when retrieved chunks cannot safely belong to this request."""
    if not query.text.strip():
        raise InvalidRetrievedEvidenceError("the request question must be non-blank")
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise InvalidRetrievedEvidenceError("retrieved chunk ids must be unique within one request")
    if any(chunk.snapshot_id != query.snapshot_id for chunk in chunks):
        raise InvalidRetrievedEvidenceError("retrieved chunks must match the requested snapshot")
    if any(chunk.citation.chunk_id != chunk.chunk_id for chunk in chunks):
        raise InvalidRetrievedEvidenceError("citation and retrieved chunk identity must agree")
    if any(not chunk.citation.url.strip() for chunk in chunks):
        raise InvalidRetrievedEvidenceError("retrieved citation URLs must be non-blank")


def allowed_media_quote(
    claim_text: str,
    chunks: Sequence[RetrievedChunk],
) -> str | None:
    """Return an in-budget verbatim media quotation, without optional enclosing quotes."""
    claim_text = _strip_optional_quotes(claim_text)
    if not claim_text or len(claim_text) > MAX_QUOTE_CHARS:
        return None
    return claim_text if any(_is_verbatim(claim_text, chunk.text) for chunk in chunks) else None


def source_exposure_exceeds_budget(
    claims: Sequence[ClaimEvidence],
    chunks: Sequence[RetrievedChunk],
) -> bool:
    """Count repeated verbatim exposure per source identity across claims and chunks."""
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    parents = {chunk_id: chunk_id for chunk_id in chunks_by_id}

    def find(chunk_id: str) -> str:
        while parents[chunk_id] != chunk_id:
            parents[chunk_id] = parents[parents[chunk_id]]
            chunk_id = parents[chunk_id]
        return chunk_id

    def union(first_id: str, second_id: str) -> None:
        first_root = find(first_id)
        second_root = find(second_id)
        if first_root != second_root:
            parents[second_root] = first_root

    chunk_list = list(chunks_by_id.values())
    for index, first in enumerate(chunk_list):
        for second in chunk_list[index + 1 :]:
            first_doc_id = first.doc_id.strip()
            second_doc_id = second.doc_id.strip()
            same_document = bool(first_doc_id) and first_doc_id == second_doc_id
            first_url = first.citation.url.strip()
            second_url = second.citation.url.strip()
            same_url = bool(first_url) and first_url == second_url
            if same_document or same_url:
                union(first.chunk_id, second.chunk_id)

    exposure_by_source: dict[str, int] = {}
    for claim in claims:
        text = _normalize(claim.text)
        matching_sources = {
            find(chunk.chunk_id) for chunk in chunk_list if _is_verbatim(text, chunk.text)
        }
        for source in matching_sources:
            exposure_by_source[source] = exposure_by_source.get(source, 0) + len(text)
    return any(
        exposure > MAX_QUOTE_CHARS for exposure in exposure_by_source.values()
    ) or _contains_over_budget_contiguous_copy(claims, chunk_list)


def validate_grounded_answer(
    answer: GroundedAnswer,
    snapshot_id: str,
    chunks: Sequence[RetrievedChunk],
) -> GroundedAnswer:
    """Freshly validate final citations, markers, source exposure and refusal shape."""
    validated = GroundedAnswer.model_validate(answer.model_dump(mode="json"))
    cited_ids = list(validated.cited_chunk_ids)
    if validated.abstained:
        if (
            not validated.answer.strip()
            or validated.citations
            or cited_ids
            or validated.abstain_reason is None
            or not validated.abstain_reason.strip()
        ):
            raise GroundingValidationError("an abstention must contain only a non-blank refusal")
        refusal_claims = [
            ClaimEvidence(text=validated.answer, chunks=tuple(chunks)),
            ClaimEvidence(text=validated.abstain_reason, chunks=tuple(chunks)),
        ]
        if source_exposure_exceeds_budget(refusal_claims, chunks):
            raise GroundingValidationError("an abstention exceeds a source quotation budget")
        return validated

    citation_ids = [citation.chunk_id for citation in validated.citations]
    if (
        not validated.answer.strip()
        or not validated.citations
        or not cited_ids
        or validated.abstain_reason is not None
        or len(cited_ids) != len(set(cited_ids))
        or set(cited_ids) != set(citation_ids)
    ):
        raise GroundingValidationError("a supported answer must have unique aligned citations")

    retrieved = {chunk.chunk_id: chunk for chunk in chunks}
    if len(retrieved) != len(chunks):
        raise GroundingValidationError("retrieved chunk ids must be unique within one request")
    for citation in validated.citations:
        _validate_citation(citation, snapshot_id, retrieved)

    claims = _parse_answer_claims(validated.answer, validated.citations, retrieved)
    if source_exposure_exceeds_budget(claims, chunks):
        raise GroundingValidationError("a supported answer exceeds a source quotation budget")
    return validated


def _validate_citation(
    citation: Citation,
    snapshot_id: str,
    retrieved: dict[str, RetrievedChunk],
) -> None:
    chunk = retrieved.get(citation.chunk_id)
    if (
        chunk is None
        or chunk.snapshot_id != snapshot_id
        or chunk.citation != citation
        or chunk.citation.chunk_id != chunk.chunk_id
        or not citation.url.strip()
    ):
        raise GroundingValidationError(
            "every citation must match a chunk retrieved for the requested snapshot"
        )


def _parse_answer_claims(
    answer: str,
    citations: Sequence[Citation],
    retrieved: dict[str, RetrievedChunk],
) -> list[ClaimEvidence]:
    broad_markers = list(NUMERIC_CITATION_MARKER.finditer(answer))
    markers = list(_GENERATED_CITATION_MARKER.finditer(answer))
    if [marker.span() for marker in broad_markers] != [marker.span() for marker in markers]:
        raise GroundingValidationError("answer citation markers must use comma-separated indices")
    if not markers:
        raise GroundingValidationError("a supported answer must reference its citations")

    claims: list[ClaimEvidence] = []
    first_use: list[int] = []
    seen: set[int] = set()
    segment_start = 0
    for marker in markers:
        text = answer[segment_start : marker.start()].strip()
        if not text:
            raise GroundingValidationError("every citation marker must follow a non-blank claim")
        refs = [int(value.strip()) for value in marker.group("refs").split(",")]
        if len(refs) != len(set(refs)) or any(ref < 1 or ref > len(citations) for ref in refs):
            raise GroundingValidationError(
                "answer citation markers must reference existing citations"
            )
        for ref in refs:
            if ref not in seen:
                seen.add(ref)
                first_use.append(ref)
        evidence = tuple(retrieved[citations[ref - 1].chunk_id] for ref in refs)
        source_text = _strip_optional_quotes(text)
        is_quoted = len(text) > 1 and _QUOTE_PAIRS.get(text[0]) == text[-1]
        if all(chunk.citation.source_type == "media" for chunk in evidence) and (
            not is_quoted or allowed_media_quote(source_text, evidence) is None
        ):
            raise GroundingValidationError(
                "media-only claims must be attributed in-budget verbatim quotations"
            )
        claims.append(ClaimEvidence(text=source_text, chunks=evidence))
        segment_start = marker.end()

    if answer[segment_start:].strip():
        raise GroundingValidationError("supported answer text must end with a citation marker")
    expected_first_use = list(range(1, len(citations) + 1))
    if first_use != expected_first_use:
        raise GroundingValidationError("citations must be ordered by first use and all be used")
    return claims


def _strip_optional_quotes(text: str) -> str:
    stripped = text.strip()
    if len(stripped) > 1 and _QUOTE_PAIRS.get(stripped[0]) == stripped[-1]:
        return stripped[1:-1].strip()
    return stripped


def _is_verbatim(claim_text: str, source_text: str) -> bool:
    claim = _normalize(claim_text)
    source = _normalize(source_text)
    return claim in source or claim.replace(" ", "") in source.replace(" ", "")


def _contains_over_budget_contiguous_copy(
    claims: Sequence[ClaimEvidence],
    chunks: Sequence[RetrievedChunk],
) -> bool:
    run_length = MAX_QUOTE_CHARS + 1
    for separator in (" ", ""):
        answer = separator.join(claim.text for claim in claims)
        answer = separator.join(answer.split())
        for chunk in chunks:
            source = separator.join(chunk.text.split())
            for start in range(len(source) - run_length + 1):
                if source[start : start + run_length] in answer:
                    return True
    return False


def _normalize(text: str) -> str:
    return " ".join(text.split())


__all__ = [
    "ClaimEvidence",
    "GroundingComponentError",
    "GroundingValidationError",
    "InvalidModelResponseError",
    "InvalidRetrievedEvidenceError",
    "ProviderResponseError",
    "allowed_media_quote",
    "contains_numeric_citation_marker",
    "source_exposure_exceeds_budget",
    "validate_grounded_answer",
    "validate_retrieved_evidence",
]
