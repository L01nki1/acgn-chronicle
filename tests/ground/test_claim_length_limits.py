"""The two length limits answer different questions and must stay independent.

`MAX_QUOTE_CHARS` is the **C3 outbound** limit: what a client can receive. It is a compliance
boundary and cannot move. `MAX_CLAIM_CHARS` bounds one model-written claim on the answer line.

They were the same constant, and that coupling caused a false refusal: a benchmark item's
baseline answer already ran to 201 characters, so any attempt to quote slightly more context
failed `allowed_media_quote` and the whole question abstained on exposure budget. Raising the
claim limit fixes that without changing what leaves the system.
"""

from __future__ import annotations

from acgn_rag.core.ground.validation import (
    ClaimEvidence,
    allowed_media_quote,
    source_exposure_exceeds_budget,
)
from acgn_rag.core.ports import MAX_CLAIM_CHARS, MAX_QUOTE_CHARS, Citation, RetrievedChunk


def _chunk(title: str, body: str = "unrelated body") -> RetrievedChunk:
    citation = Citation(
        source_type="media",
        publisher="ANN",
        title=title,
        url="https://example.com/a",
        published_at="2026-01-01",
        chunk_id="c1",
    )
    return RetrievedChunk(
        chunk_id="c1", doc_id="d", snapshot_id="s", text=body, score=0.9, citation=citation
    )


def test_the_c3_wire_limit_is_still_two_hundred() -> None:
    """A compliance boundary: raising the claim limit must not raise this."""
    assert MAX_QUOTE_CHARS == 200


def test_the_claim_limit_is_independent_and_larger() -> None:
    assert MAX_CLAIM_CHARS == 300
    assert MAX_CLAIM_CHARS > MAX_QUOTE_CHARS


def test_a_claim_between_the_two_limits_is_accepted() -> None:
    """The exact case that used to fail: 250 characters of a long media headline."""
    headline = "A" * 240 + " and Studio KAI animates it again"
    assert len(headline) > MAX_QUOTE_CHARS
    assert len(headline) <= MAX_CLAIM_CHARS

    assert allowed_media_quote(headline, [_chunk(headline)]) == headline


def test_a_claim_over_the_claim_limit_is_still_refused() -> None:
    headline = "B" * (MAX_CLAIM_CHARS + 1)
    assert allowed_media_quote(headline, [_chunk(headline)]) is None


def test_exposure_budget_follows_the_claim_limit() -> None:
    """One source may expose up to the claim limit, not up to the wire limit."""
    text = "Z" * (MAX_QUOTE_CHARS + 40)
    chunk = _chunk("irrelevant title", body=text)
    claim = ClaimEvidence(text=text, chunks=(chunk,))

    assert source_exposure_exceeds_budget([claim], [chunk]) is False

    over = ClaimEvidence(text="Y" * (MAX_CLAIM_CHARS + 2), chunks=(chunk,))
    assert source_exposure_exceeds_budget([over], [_chunk("t", body="Y" * (MAX_CLAIM_CHARS + 2))])
