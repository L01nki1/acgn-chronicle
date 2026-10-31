"""Compliance clipping for outbound payloads (06 §3 C3) - the last gate before the client.

Owner: B (front-end layer reviewed by C) | Spec: 06 §3 C3/C4, invariant 1 of spec 02

Clients may only receive four things: a summary, the source URL, a <=200 character excerpt, and a
mandatory attribution. Full text is never served - this module is the only place in the project
allowed to shorten an article body; do not slice text anywhere else.

Measurement unit: **Unicode code points** (not bytes or UTF-16 code units), so Japanese, Chinese
and English excerpts share one budget. Rationale in docs/COMPLIANCE.md.

The conversion here is also the type boundary: `RetrievedChunk` (internal, carries the body) goes
in, `ChunkExcerpt` (wire, carries only `quote`) comes out. Returning a plain dict instead would put
the wire shape outside the schema, where no test could check it.
"""

from __future__ import annotations

from acgn_rag.core.ports import MAX_QUOTE_CHARS, ChunkExcerpt, Citation, RetrievedChunk

ELLIPSIS = "\u2026"


def clip_quote(text: str, limit: int = MAX_QUOTE_CHARS) -> str:
    """Clip the body to at most `limit` characters, ending with an ellipsis that fits the budget.

    Whitespace normalisation only collapses runs of whitespace; it does not change wording.
    C4 forbids paraphrasing media commentary, not removing layout noise.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1] + ELLIPSIS


def to_excerpt(chunk: RetrievedChunk, limit: int = MAX_QUOTE_CHARS) -> ChunkExcerpt:
    """Turn an internal chunk into the wire unit of `/retrieve`.

    The body is dropped here rather than redacted later: after this call the only text in the
    object is `quote`, so no downstream handler can accidentally serialize the article.
    A caller may request a shorter quote, but cannot raise the outbound budget above 200.
    """
    citation: Citation = chunk.citation
    return ChunkExcerpt(
        chunk_id=chunk.chunk_id,
        score=chunk.score,
        quote=clip_quote(chunk.text, min(limit, MAX_QUOTE_CHARS)),
        citation=citation,
    )


__all__ = ["ELLIPSIS", "MAX_QUOTE_CHARS", "clip_quote", "to_excerpt"]
