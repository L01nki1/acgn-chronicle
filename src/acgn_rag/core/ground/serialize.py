"""Compliance clipping for outbound payloads (06 §3 C3) - the last gate before the client.

Owner: B (front-end layer reviewed by C) | Spec: 06 §3 C3/C4, invariant 1 of spec 02

Clients may only receive four things: a summary, the source URL, a <=200 character excerpt,
and a mandatory attribution. Full text is never served - this module is the only place in the
project allowed to shorten an article body; do not slice text anywhere else.

Measurement unit: **Unicode characters** (not bytes, not UTF-8 code units), so Japanese,
Chinese and English excerpts share one budget. Rationale in docs/COMPLIANCE.md.
"""

from __future__ import annotations

from typing import Any

from acgn_rag.core.ports import RetrievedChunk

MAX_QUOTE_CHARS = 200
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


def serialize_chunk(chunk: RetrievedChunk, limit: int = MAX_QUOTE_CHARS) -> dict[str, Any]:
    """Turn an internal chunk into a client-safe payload (the response unit of /retrieve)."""
    citation = chunk.citation
    return {
        "chunk_id": chunk.chunk_id,
        "score": chunk.score,
        "quote": clip_quote(chunk.text, limit),
        "source": {
            "type": citation.source_type,
            "publisher": citation.publisher,
            "title": citation.title,
            "url": citation.url,
            "published_at": citation.published_at,
        },
        # Mandatory attribution: the client must not render a citation without a url
        "attribution_required": True,
    }


__all__ = ["ELLIPSIS", "MAX_QUOTE_CHARS", "clip_quote", "serialize_chunk"]
