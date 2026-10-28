"""Compliance tests (06 §3 C3 / invariant 1 of spec 02): excerpts are <=200 chars with attribution.

These assertions are the automated gate for "full text never leaves the store". If one turns red,
fix the implementation - never raise the threshold.
"""

from __future__ import annotations

from acgn_rag.core.ground.serialize import MAX_QUOTE_CHARS, clip_quote, serialize_chunk
from acgn_rag.core.ports import Citation, RetrievedChunk


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
    text = "New anime adaptation announced. " * 60  # ~1900 characters
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


def test_serialized_chunk_carries_source_and_flag() -> None:
    body = "sentence about a release date. " * 200  # far longer than the excerpt budget
    payload = serialize_chunk(_chunk(body))
    quote = payload["quote"]
    assert len(quote) <= MAX_QUOTE_CHARS
    assert len(body) > MAX_QUOTE_CHARS * 10
    # the excerpt is a truncated prefix, never the whole body
    assert len(quote) < len(body)
    assert body.startswith(quote[:-1])
    assert payload["source"]["url"].startswith("https://")
    assert payload["attribution_required"] is True
