"""Chunking configs: A(512/64) / A'(256/0) / A0(512/0) are parameters of one implementation.

Owner: F | Spec: 02 §4, issue #8
Status: strategy A implemented in `strategy_a.py`; strategy B (event aggregation) stays a
placeholder until A has cluster ids.
"""

from acgn_rag.core.chunk.strategy_a import Chunk, TokenCounter, budget_for, chunk_text
from acgn_rag.core.chunk.variants import CHUNK_VARIANTS, ChunkVariant

__all__ = [
    "CHUNK_VARIANTS",
    "Chunk",
    "ChunkVariant",
    "TokenCounter",
    "budget_for",
    "chunk_text",
]
