"""Official-source score bonus: one pool, official slightly favoured.

Owner: D | Spec: REBUILD_PLAN §1b (official sources are not retrievable)

``§1b`` measured why the official pages lose: ``capcom-news`` and ``toei-news`` store no full
body at all (55-57 characters of headline plus blurb), so on a single-pool vector search a
thousand-character ANN article on the same topic outranks them for essentially every fact
question. Measured on the frozen corpus, official gold URLs reached 0/20 in top-5 while media
gold URLs reached 20/26.

Two obvious fixes both fail:

  - **Filter to official only.** The official corpus is 470 items against media's 35 148, so
    most questions become unanswerable (measured: media gold 20/26 -> 0/26).
  - **Hard reservation** ("keep two official slots"). It does recover official recall, but it
    reserves those slots even when the official hits are unrelated, which cost a media hit
    (20/26 -> 19/26). A reservation cannot tell a relevant official page from an irrelevant
    one, because the score distributions overlap: the *lowest* official hit scored 0.7471 and
    the *highest* official miss scored 0.8337 on the same set.

So official chunks are ranked in the same pool with a multiplier instead. Nothing is
guaranteed a slot: an on-topic official page rises above a comparable media article, and an
irrelevant one still sinks below every better media hit. Media therefore cannot be starved by
construction, which was the requirement: never drop the media pool just because the official
pool came back empty or irrelevant.

Weight sweep on both sets (official gold / media gold):

    weight   calibration   delivery
    1.00     3/20          20/26
    1.05     6/20          20/26
    1.08     9/20          20/26   <- chosen
    1.12     10/20         16/26
    1.20     11/20          0/26

1.08 takes the full official gain available before media recall starts to fall.
"""

from __future__ import annotations

from dataclasses import dataclass

from acgn_rag.core.ports import RetrievalFilters, RetrievalQuery, RetrievedChunk
from acgn_rag.core.retrieve.vector import search

OFFICIAL = "official"
MEDIA = "media"

#: Multiplier applied to an official chunk's score before the single ranking.
#: 1.08 is the largest bonus that left media recall untouched on both evaluation sets.
OFFICIAL_SCORE_WEIGHT = 1.08


@dataclass(frozen=True, slots=True)
class SourceMix:
    """How many of the returned chunks came from each source type; archived for diagnosis."""

    official: int
    media: int
    other: int

    @property
    def total(self) -> int:
        return self.official + self.media + self.other


def _with_source_types(query: RetrievalQuery, source_types: tuple[str, ...]) -> RetrievalQuery:
    base = query.filters
    filters = RetrievalFilters(
        since=base.since if base else None,
        until=base.until if base else None,
        source_types=source_types,
        media_ids=base.media_ids if base else (),
    )
    return query.model_copy(update={"filters": filters})


def search_with_official_bonus(
    query: RetrievalQuery,
    *,
    store,
    embedder,
    official_weight: float = OFFICIAL_SCORE_WEIGHT,
) -> tuple[list[RetrievedChunk], SourceMix]:
    """Retrieve ``query.top_k`` chunks with official scores scaled by ``official_weight``.

    ``official_weight=1.0`` reproduces the single-pool behaviour exactly, so an on/off run
    differs in this one constant and nothing else.

    Only official and media are queried separately; any other source type is left out on
    purpose, because it is not evidence for a grounded answer and never was in the single-pool
    path either (metadata rows have no body to cite).
    """
    if official_weight == 1.0:
        chunks = search(query, store=store, embedder=embedder)
        return chunks, _mix(chunks)

    official = search(_with_source_types(query, (OFFICIAL,)), store=store, embedder=embedder)
    media = search(_with_source_types(query, (MEDIA,)), store=store, embedder=embedder)

    # One ranking over both tiers, each candidate keeping its own tier's score scale (which is
    # a cosine similarity in both cases, so the comparison is meaningful) except that official
    # candidates carry the bonus. Ties keep the earlier candidate, so the order is stable.
    scored: list[tuple[float, int, RetrievedChunk]] = [
        (chunk.score * official_weight, position, chunk) for position, chunk in enumerate(official)
    ] + [(chunk.score, len(official) + position, chunk) for position, chunk in enumerate(media)]
    scored.sort(key=lambda row: (-row[0], row[1]))

    picked: list[RetrievedChunk] = []
    seen: set[str] = set()
    for _, _, chunk in scored:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        picked.append(chunk)
        if len(picked) == query.top_k:
            break
    return picked, _mix(picked)


def _mix(chunks: list[RetrievedChunk]) -> SourceMix:
    official = sum(1 for c in chunks if c.citation.source_type == OFFICIAL)
    media = sum(1 for c in chunks if c.citation.source_type == MEDIA)
    return SourceMix(official=official, media=media, other=len(chunks) - official - media)


__all__ = [
    "MEDIA",
    "OFFICIAL",
    "OFFICIAL_SCORE_WEIGHT",
    "SourceMix",
    "search_with_official_bonus",
]
