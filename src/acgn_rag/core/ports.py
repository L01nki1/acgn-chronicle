"""KnowledgePort contract types - the single source of truth for the whole project (spec 05 §2).

Owner: B (contract arbitration) | Spec: 05 §2
Freeze policy (05 §5):
  1. This file is the source of truth for structure. `contracts/knowledge_port.schema.json` is
     exported from it by `scripts/export_schema.py`; CI fails on any drift. Prose (this module's
     docstrings, `contracts/KnowledgePort.md`) carries semantics only - it may never contradict
     the types, because prose is not machine-checked.
  2. Only `WIRE_MODELS` is exported. Internal types (e.g. `RetrievedChunk`) are deliberately kept
     out of the schema so a consumer cannot code against a shape that never crosses the boundary.
  3. Semantic changes (rename / retype / new required field) require a PR with a reason, a
     re-exported schema, and a notice to C (web) and F (Isla contact).
  4. New fields are always Optional with a default, so old consumers keep working.
  5. No breaking change from W2 onwards; fully frozen (including additions) from W3.

No implementation logic belongs here - types only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SourceType = Literal["official", "media", "metadata"]

SCHEMA_VERSION = "0.2.1"

# Shared by the wire constraint and the serializer; serialize.MAX_QUOTE_CHARS re-exports it.
MAX_QUOTE_CHARS = 200

# One definition, reused by both the internal and the wire chunk type: a score whose meaning is
# left implicit is worse than no score, because a client will average it with another config's.
SCORE_SEMANTICS = (
    "cosine similarity in [-1, 1], higher is better (1 - Chroma cosine distance). "
    "Ordinal within one response only: never compare across queries, chunk configurations or "
    "snapshots, and never average it as a quality metric"
)


class CorpusSnapshot(BaseModel):
    """A frozen corpus state. A new crawl produces a new snapshot; existing ones are immutable."""

    snapshot_id: str = Field(
        ...,
        description="YYYYMMDD-<12 lowercase SHA-256 hex>; date is the frozen window_end UTC day, "
        "not the rebuild day; hash is from A's canonical content manifest",
    )
    created_at: str
    item_count: int
    deduped_count: int = Field(
        ...,
        description="items after deduplication - the denominator of every metric",
    )
    source_count: int
    window_start: str = Field(
        ...,
        description="corpus time span; required for reproducible evaluation",
    )
    window_end: str


class SnapshotInventory(BaseModel):
    """What this host can serve right now (response of `GET /snapshots`).

    A client must not invent a snapshot id, and must not assume that the id it used yesterday is
    still retained. Availability is governed by retention, not by currency.
    """

    current: str = Field(
        ...,
        description="the snapshot the read path serves; flips atomically after a build succeeds",
    )
    retained: Sequence[str] = Field(
        ...,
        description="full ids still queryable, most recent first; a known completed snapshot "
        "retired from service answers 410, a never-known id 404; absence alone does not prove retirement",
    )


class RetrievalFilters(BaseModel):
    """AND between filter families, ANY within a list; absent/null/empty lists do not filter."""

    source_types: Sequence[SourceType] | None = Field(
        None,
        description="match ANY listed source type; absent, null or [] means no source-type filter",
    )
    since: str | None = Field(
        None,
        description="inclusive lower bound: a YYYY-MM-DD date starts at UTC midnight; an ISO-8601 "
        "datetime is converted to UTC (naive means UTC). Absent/null leaves the bound open",
    )
    until: str | None = Field(
        None,
        description="exclusive upper bound: a YYYY-MM-DD date includes that whole UTC day and "
        "ends at next midnight; an ISO-8601 datetime ends at that UTC instant (naive means UTC). "
        "Invalid or reversed bounds are 422 at the HTTP boundary; absent/null leaves it open",
    )
    media_ids: Sequence[int] | None = Field(
        None,
        description="match ANY listed canonical media id; absent, null or [] means no media filter. "
        "Apply membership within the requested snapshot; a non-empty selection with no matching "
        "documents matches nothing, never drops the filter. Combine filter families using AND",
    )


class RetrievalQuery(BaseModel):
    text: str
    top_k: int = 5
    snapshot_id: str = Field(
        ...,
        description=(
            "binds the query to a snapshot, which makes results reproducible; "
            "an unknown id answers 404, a retired id answers 410"
        ),
    )
    filters: RetrievalFilters | None = None


class Citation(BaseModel):
    """A structured citation, not a concatenated string."""

    source_type: SourceType = Field(
        ...,
        description="drives display strength and compliance clipping",
    )
    publisher: str = Field(..., description="'Capcom' | 'ANN' | 'AniList' ...")
    title: str
    url: str = Field(
        ...,
        description="canonical original URL, never an archive link; required, so a client cannot "
        "render a citation without attribution",
    )
    published_at: str
    chunk_id: str = Field(..., description="a citation must be able to locate its chunk")


class RetrievedChunk(BaseModel):
    """Internal retrieval result. NEVER serialized - it carries the full body.

    This type is intentionally absent from the exported schema. What crosses the boundary is
    `ChunkExcerpt`; `core/ground/serialize.py` is the only conversion point.
    """

    chunk_id: str
    doc_id: str
    snapshot_id: str
    text: str = Field(
        ...,
        description="FULL body text; internal use only, never leaves the process",
    )
    score: float = Field(..., description=SCORE_SEMANTICS)
    citation: Citation


class ChunkExcerpt(BaseModel):
    """The only chunk shape a client ever sees: an excerpt plus a citation, never the body.

    Response unit of `POST /retrieve`. `quote` is the complete text content of the response -
    there is no second field carrying more of the article.
    """

    model_config = ConfigDict(validate_assignment=True)

    chunk_id: str
    score: float = Field(..., description=SCORE_SEMANTICS)
    quote: str = Field(
        ...,
        max_length=MAX_QUOTE_CHARS,
        description="at most 200 Unicode characters, clipped by ground/serialize; the client "
        "must not re-expand it from the source page",
    )
    citation: Citation = Field(
        ...,
        description="url is required, so attribution is structural rather than a convention",
    )


class GroundedAnswer(BaseModel):
    answer: str = Field(
        ..., description="supported answer text, or an explicit refusal explanation when abstained"
    )
    citations: Sequence[Citation] = Field(
        ..., description="deduplicated in first-use order; empty when abstained"
    )
    cited_chunk_ids: Sequence[str] = Field(
        ...,
        description="must be a subset of the chunks actually retrieved for this query; empty when "
        "abstained",
    )
    abstained: bool = Field(
        ..., description="true when the selected snapshot and filters do not support an answer"
    )
    abstain_reason: str | None = Field(
        None,
        description="non-empty explanation of missing support when abstained; null otherwise",
    )


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

    `episode_table` was dropped in 0.2.0: `Sequence[dict]` has no shape, so it could neither be
    validated nor typed by a consumer, and its only user (the digest) has not been crawled yet.
    When the digest is green-lit it returns as `Sequence[EpisodeRow] | None = None`.
    """

    snapshot_id: str
    date: str
    by_category: Sequence[CategoryDigest]
    top_clusters: Sequence[ClusterCard]
    fact_updates: Sequence[FactUpdate]
    source_coverage: dict[str, int] = Field(
        ..., description="items per source for that day (also feed monitoring, 01 §5.5)"
    )


# What the exported schema contains. Anything absent from here is internal by definition, so the
# schema answers "what may a consumer depend on" without ambiguity.
WIRE_MODELS: tuple[str, ...] = (
    "CategoryDigest",
    "ChunkExcerpt",
    "Citation",
    "ClusterCard",
    "CorpusSnapshot",
    "DailyDigest",
    "DailyReportPort",
    "FactUpdate",
    "GroundedAnswer",
    "RetrievalFilters",
    "RetrievalQuery",
    "SnapshotInventory",
)

__all__ = [
    "MAX_QUOTE_CHARS",
    "SCHEMA_VERSION",
    "SCORE_SEMANTICS",
    "WIRE_MODELS",
    "CategoryDigest",
    "ChunkExcerpt",
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
    "SnapshotInventory",
    "SourceType",
]
