"""multilingual-e5-base encoding + cache. e5 requires the query:/passage: prefixes and L2 normalization.

Owner: F | Spec: 04 §7 D4 / RUNBOOK §3.4-§3.5

Encoder identity, model-family prefix/prompt policy, context headers, the identity-keyed cache
key and the throughput helper live here. Every index build records its identity so vectors from
different lanes never mix (RUNBOOK §3.4). numpy comes with [dev] as well as [embed], so CI runs
the fake-encoder and cache tests without torch or a model download.
"""

from acgn_rag.core.embed.cache import EmbedCache
from acgn_rag.core.embed.encoder import (
    HEADER_VERSION,
    MODEL_SPECS,
    Embedder,
    EncoderIdentity,
    ModelSpec,
    PassageInput,
    build_header,
    cache_key,
    measure_throughput,
    model_spec,
)

__all__ = [
    "HEADER_VERSION",
    "MODEL_SPECS",
    "EmbedCache",
    "Embedder",
    "EncoderIdentity",
    "ModelSpec",
    "PassageInput",
    "build_header",
    "cache_key",
    "measure_throughput",
    "model_spec",
]
