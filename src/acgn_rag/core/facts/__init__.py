"""WP2: L0 fact extraction from first-party posts (title / date / platform / event), always with chunkId + url.

Owner: F | Spec: 01 §4.4, RUNBOOK §3.6
"""

from acgn_rag.core.facts.extract import (
    BATCH_SIZE,
    IR_KEYWORDS,
    IR_TOKEN_PATTERN,
    MIN_MAX_TOKENS,
    SYSTEM_PROMPT,
    AnnouncementChunk,
    AnnouncementFacts,
    FactBatch,
    LLMClient,
    build_batch_prompt,
    expand_to_fact_updates,
    extract_facts,
    is_ir_release,
    make_openai_client,
)

__all__ = [
    "BATCH_SIZE",
    "IR_KEYWORDS",
    "IR_TOKEN_PATTERN",
    "MIN_MAX_TOKENS",
    "SYSTEM_PROMPT",
    "AnnouncementChunk",
    "AnnouncementFacts",
    "FactBatch",
    "LLMClient",
    "build_batch_prompt",
    "expand_to_fact_updates",
    "extract_facts",
    "is_ir_release",
    "make_openai_client",
]
