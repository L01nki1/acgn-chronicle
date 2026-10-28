"""KnowledgePort contract types - the single source of truth for the whole project (spec 05 §2).

Owner: B (contract arbitration) | Spec: 05 §2
Freeze policy (05 §5):
  1. This file is the source of truth; `contracts/knowledge_port.schema.json` is exported from it
     by `scripts/export_schema.py`. CI verifies both sides; any drift fails the build.
  2. Semantic changes (rename / retype / new required field) require a PR with a reason,
     a re-exported schema, and a notice to C (web) and F (Isla contact).
  3. New fields are always Optional with a default, so old consumers keep working.
  4. No breaking change from W2 onwards; fully frozen (including additions) from W3.

No implementation logic belongs here - types only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

SourceType = Literal["official", "media", "metadata"]

SCHEMA_VERSION = "0.1.0"


class CorpusSnapshot(BaseModel):
    """A frozen corpus state. A new crawl produces a new snapshot; old ones are immutable."""

    snapshot_id: str = Field(
        ..., description="content hash + date; rebuilding the same corpus must match"
    )
    created_at: str
    item_count: int
    deduped_count: int = Field(
        ..., description="items after deduplication - the denominator of every metric"
    )
    source_count: int
    window_start: str = Field(
        ..., description="corpus time span; required for reproducible evaluation"
    )
    window_end: str


class RetrievalFilters(BaseModel):
    source_types: Sequence[SourceType] | None = None
    since: str | None = Field(
        None,
        description='daily views ("today only") also use this; never add a time-bucketed index',
    )
    until: str | None = None
    media_ids: Sequence[int] | None = None


class RetrievalQuery(BaseModel):
    text: str
    top_k: int = 5
    snapshot_id: str = Field(
        ..., description="binds the query to a snapshot, which makes results reproducible"
    )
    filters: RetrievalFilters | None = None


class Citation(BaseModel):
    """A structured citation, not a concatenated string."""

    source_type: SourceType = Field(
        ..., description="drives display strength and compliance clipping"
    )
    publisher: str = Field(..., description="'Capcom' | 'ANN' | 'AniList' ...")
    title: str
    url: str
    published_at: str
    chunk_id: str = Field(..., description="a citation must be able to locate its chunk")


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    snapshot_id: str
    text: str = Field(
        ...,
        description="internal use; clipped to a <=200 character excerpt by ground/serialize for clients",
    )
    score: float
    citation: Citation


class GroundedAnswer(BaseModel):
    answer: str
    citations: Sequence[Citation]
    cited_chunk_ids: Sequence[str]
    abstained: bool
    abstain_reason: str | None = None


# ---- Secondary task ports (interface first; full logic lands later) ----


class DailyReportPort(BaseModel):
    """Daily report entry point (implemented in a later stage)."""

    snapshot_id: str
    date: str


class CategoryDigest(BaseModel):
    category: str = Field(..., description="anime / manga / game / novel")
    new_count: int
    cluster_count: int


class ClusterCard(BaseModel):
    """Daily hot cluster: one card = one deduplicated cluster."""

    cluster_id: str
    size: int = Field(..., description="cluster size as a heat proxy")
    media_ids: Sequence[int]
    representative_title: str
    first_seen: str
    citations: Sequence[Citation] = Field(
        ..., description="reuses Citation; the client applies compliance clipping"
    )


class FactUpdate(BaseModel):
    """A structured fact newly extracted today (L0)."""

    media: str
    field: str = Field(..., description="release date / platform / event type ...")
    value: str
    chunk_id: str
    url: str
    publisher: str


class DailyDigest(BaseModel):
    """Per-day candidate set shared by the board and the stage-2 text digest.

    Per-day retrieval still goes through RetrievalFilters.since/until; a time-bucketed index is
    forbidden because it would contaminate the chunking comparison (invariant 4, spec 02).
    """

    snapshot_id: str
    date: str
    by_category: Sequence[CategoryDigest]
    top_clusters: Sequence[ClusterCard]
    fact_updates: Sequence[FactUpdate]
    episode_table: Sequence[dict] = Field(..., description="Bilibili schedule passthrough")
    source_coverage: dict[str, int] = Field(
        ..., description="items per source for that day (also feed monitoring, 01 §5.5)"
    )


__all__ = [
    "SCHEMA_VERSION",
    "CategoryDigest",
    "Citation",
    "ClusterCard",
    "CorpusSnapshot",
    "DailyDigest",
    "DailyReportPort",
    "FactUpdate",
    "GroundedAnswer",
    "RetrievalFilters",
    "RetrievalQuery",
    "RetrievedChunk",
    "SourceType",
]
