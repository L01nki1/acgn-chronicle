"""Two-zone storage: fact zone (publishable) / raw zone (internal retrieval only, never serialized).

Owner: F | Spec: 06 §3 C2, 04 §7 D4
Backend: chromadb >=1.0,<2, embedded PersistentClient, cosine HNSW (docs/RUNBOOK.md §3.1).
The adapter seam for A's NormalizedItem schema is `store.payload.payload_from_fields`.
"""

from acgn_rag.core.store.chroma_store import ChromaStore, collection_name, media_filter
from acgn_rag.core.store.payload import to_payload
from acgn_rag.core.store.zones import assert_publishable, write_facts_zone

__all__ = [
    "ChromaStore",
    "assert_publishable",
    "collection_name",
    "media_filter",
    "to_payload",
    "write_facts_zone",
]
