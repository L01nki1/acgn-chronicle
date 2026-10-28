"""合规单测（06 §3 C3 / 02 §3 不变量 1）：对外片段 ≤200 字符 + 强制出处。

这些断言是"原文不出库"的最后一道自动化闸门，任何一条变红都必须先修实现，不许改阈值。
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
            title="プレスリリース",
            url="https://www.capcom.co.jp/news/0001",
            published_at="2026-10-27",
            chunk_id="c_0001",
        ),
    )


def test_limit_constant() -> None:
    assert MAX_QUOTE_CHARS == 200


def test_cjk_long_text_is_clipped_within_limit() -> None:
    text = "『新作アニメ』の制作が決定した。" * 60  # 日文约 960 字符
    quote = clip_quote(text)
    assert len(quote) <= MAX_QUOTE_CHARS
    assert quote.endswith("…")


def test_ascii_long_text_is_clipped_by_chars_not_bytes() -> None:
    text = "a" * 1000
    quote = clip_quote(text)
    assert len(quote) == MAX_QUOTE_CHARS
    assert quote.endswith("…")


def test_short_text_untouched() -> None:
    text = "短公告：本日より配信開始。"
    assert clip_quote(text) == text


def test_boundary_exact_limit_has_no_ellipsis() -> None:
    text = "x" * MAX_QUOTE_CHARS
    quote = clip_quote(text)
    assert quote == text
    assert not quote.endswith("…")


def test_whitespace_normalised() -> None:
    assert clip_quote("a\n\n  b\t c") == "a b c"


def test_serialized_chunk_carries_source_and_flag() -> None:
    payload = serialize_chunk(_chunk("本文" * 500))
    assert len(payload["quote"]) <= MAX_QUOTE_CHARS
    assert payload["source"]["url"].startswith("https://")
    assert payload["attribution_required"] is True
    # 原文全文绝不出现在载荷里
    assert "本文" * 101 not in payload["quote"]
