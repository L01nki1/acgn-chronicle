"""对外序列化的合规裁剪（06 §3 C3）—— 前端之前的最后一道闸。

负责人：B（C 复核前端层）　|　依据：06 §3 C3/C4、02 §3 不变量 1

对外只允许输出四件东西：摘要 + 原文 URL + ≤200 字符片段 + 强制出处。
原文全文永不响应 —— 本模块是全项目唯一允许"缩短正文"的地方，别处不要自己 slice。

计量口径：**按 Unicode 字符计**（不是字节、不是 UTF-8 码元）。日中英混排统一用 len(str)，
理由见 docs/COMPLIANCE.md。
"""

from __future__ import annotations

from typing import Any

from acgn_rag.core.ports import RetrievedChunk

MAX_QUOTE_CHARS = 200
ELLIPSIS = "…"


def clip_quote(text: str, limit: int = MAX_QUOTE_CHARS) -> str:
    """把正文截到 limit 个字符以内，被截断时以省略号收尾且总长仍 ≤ limit。

    空白归一（连续空白压成一个空格）只做格式整理，不改变原意 —— C4 禁止的是转述媒体观点，
    不是去除排版噪声。
    """
    if limit <= 0:
        raise ValueError("limit must be positive")
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1] + ELLIPSIS


def serialize_chunk(chunk: RetrievedChunk, limit: int = MAX_QUOTE_CHARS) -> dict[str, Any]:
    """把内部 chunk 转成可直接响应的对外载荷（/retrieve 的返回单元）。"""
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
        # 强制出处：前端不许渲染没有 url 的引用
        "attribution_required": True,
    }


__all__ = ["ELLIPSIS", "MAX_QUOTE_CHARS", "clip_quote", "serialize_chunk"]
