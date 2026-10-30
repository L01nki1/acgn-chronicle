"""Compliance tests (06 §3 C3 / invariant 1 of spec 02): excerpts are <=200 chars with attribution.

These assertions are the automated gate for "full text never leaves the store". If one turns red,
fix the implementation - never raise the threshold.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from acgn_rag.core.ground.serialize import MAX_QUOTE_CHARS, clip_quote, to_excerpt
from acgn_rag.core.ports import ChunkExcerpt, Citation, RetrievedChunk


def _chunk(text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c_0001",
        doc_id="d_0001",
        snapshot_id="20261028abcdef",
        text=text,
        score=0.87,
        citation=Citation(
            source_type="official",
            publisher="Capcom",
            title="Press release",
            url="https://www.capcom.co.jp/news/0001",
            published_at="2026-10-27",
            chunk_id="c_0001",
        ),
    )


def test_limit_constant() -> None:
    assert MAX_QUOTE_CHARS == 200


def test_cjk_long_text_is_clipped_within_limit() -> None:
    text = "新作アニメの放送開始日が発表されました。" * 60
    quote = clip_quote(text)
    assert len(quote) <= MAX_QUOTE_CHARS
    assert quote.endswith("\u2026")


def test_ascii_long_text_is_clipped_by_chars_not_bytes() -> None:
    text = "a" * 1000
    quote = clip_quote(text)
    assert len(quote) == MAX_QUOTE_CHARS
    assert quote.endswith("\u2026")


def test_short_text_untouched() -> None:
    text = "Short notice: available today."
    assert clip_quote(text) == text


def test_boundary_exact_limit_has_no_ellipsis() -> None:
    text = "x" * MAX_QUOTE_CHARS
    quote = clip_quote(text)
    assert quote == text
    assert not quote.endswith("\u2026")


def test_whitespace_normalised() -> None:
    assert clip_quote("a\n\n  b\t c") == "a b c"


def test_excerpt_is_a_clipped_prefix_never_the_whole_body() -> None:
    body = "sentence about a release date. " * 200  # far longer than the excerpt budget
    excerpt = to_excerpt(_chunk(body))
    assert isinstance(excerpt, ChunkExcerpt)
    assert len(body) > MAX_QUOTE_CHARS * 10
    assert len(excerpt.quote) <= MAX_QUOTE_CHARS
    # the excerpt is a truncated prefix, never the whole body
    assert len(excerpt.quote) < len(body)
    assert body.startswith(excerpt.quote[:-1])


def test_excerpt_carries_a_usable_citation() -> None:
    excerpt = to_excerpt(_chunk("body " * 500))
    assert excerpt.citation.url.startswith("https://")
    assert excerpt.citation.chunk_id == excerpt.chunk_id
    assert excerpt.citation.publisher


def test_caller_cannot_raise_the_outbound_quote_budget() -> None:
    excerpt = to_excerpt(_chunk("a" * 1000), limit=500)
    assert len(excerpt.quote) == MAX_QUOTE_CHARS


@pytest.mark.parametrize("character", ["a", "日", "🎬"])
def test_wire_quote_budget_counts_unicode_characters(character: str) -> None:
    chunk = _chunk("")
    excerpt = ChunkExcerpt(
        chunk_id=chunk.chunk_id,
        score=chunk.score,
        quote=character * MAX_QUOTE_CHARS,
        citation=chunk.citation,
    )
    assert len(excerpt.quote) == MAX_QUOTE_CHARS


@pytest.mark.parametrize("character", ["a", "日", "🎬"])
def test_wire_model_rejects_an_oversized_quote(character: str) -> None:
    chunk = _chunk("")
    with pytest.raises(ValidationError, match="quote"):
        ChunkExcerpt(
            chunk_id=chunk.chunk_id,
            score=chunk.score,
            quote=character * (MAX_QUOTE_CHARS + 1),
            citation=chunk.citation,
        )


@pytest.mark.parametrize("character", ["a", "日", "🎬"])
def test_wire_quote_budget_survives_attribute_assignment(character: str) -> None:
    excerpt = to_excerpt(_chunk("original quote"))
    with pytest.raises(ValidationError, match="quote"):
        excerpt.quote = character * (MAX_QUOTE_CHARS + 1)
    assert excerpt.quote == "original quote"
    excerpt.quote = character * MAX_QUOTE_CHARS
    assert len(excerpt.quote) == MAX_QUOTE_CHARS


def test_wire_payload_has_no_field_for_the_body() -> None:
    """Structural, not behavioural: the wire type must have nowhere to put the article.

    A `text` field that is "always clipped" can regress; a type without a `text` field cannot.
    """
    payload = to_excerpt(_chunk("body " * 500)).model_dump()
    assert set(payload) == {"chunk_id", "score", "quote", "citation"}
    long_values = [
        key
        for key, value in payload.items()
        if isinstance(value, str) and len(value) > MAX_QUOTE_CHARS
    ]
    assert long_values == []
