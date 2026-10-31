# ruff: noqa: RUF001 - the mock returns human-readable Chinese copy; fullwidth punctuation is correct.
"""Contract-shaped mock data for the G1a query page.

Owner: C | Spec: docs/ROADMAP.md G1a, contracts/KnowledgePort.md §3-§4

This is deliberately **not** retrieval or generation. It returns a fixed, clearly fictional corpus
so the page and its compliance rules can be built before WP2/WP4 land. The page consumes the same
wire models a real host must return; replacing this adapter with the real host is the W2 wiring.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from acgn_rag.core.ports import (
    ChunkExcerpt,
    Citation,
    CorpusSnapshot,
    GroundedAnswer,
    RetrievalFilters,
    RetrievalQuery,
    SnapshotInventory,
)
from acgn_rag.web.errors import InvalidFilter, RetiredSnapshot, UnknownSnapshot

CURRENT_SNAPSHOT_ID = "20261030-abcdef012345"
RETAINED_SNAPSHOT_ID = "20261023-012345abcdef"
RETIRED_SNAPSHOT_ID = "20261016-fedcba987654"
MOCK_CONFIG = "A"

CURRENT_SNAPSHOT = CorpusSnapshot(
    snapshot_id=CURRENT_SNAPSHOT_ID,
    created_at="2026-10-30T09:00:00Z",
    item_count=8123,
    deduped_count=6402,
    source_count=12,
    window_start="2023-07-30",
    window_end="2026-10-30",
)
RETAINED_SNAPSHOT = CorpusSnapshot(
    snapshot_id=RETAINED_SNAPSHOT_ID,
    created_at="2026-10-23T09:00:00Z",
    item_count=7984,
    deduped_count=6110,
    source_count=12,
    window_start="2023-07-23",
    window_end="2026-10-23",
)

_PRIVATE_TERMS = ("电话", "住址", "未公开", "私人")


@dataclass(frozen=True, slots=True)
class MockRecord:
    excerpt: ChunkExcerpt
    media_ids: frozenset[int]


def _citation(
    *,
    source_type: str,
    publisher: str,
    title: str,
    url: str,
    published_at: str,
    chunk_id: str,
) -> Citation:
    return Citation(
        source_type=source_type,
        publisher=publisher,
        title=title,
        url=url,
        published_at=published_at,
        chunk_id=chunk_id,
    )


def _record(
    *,
    chunk_id: str,
    score: float,
    quote: str,
    media_ids: tuple[int, ...],
    citation: Citation,
) -> MockRecord:
    return MockRecord(
        excerpt=ChunkExcerpt(chunk_id=chunk_id, score=score, quote=quote, citation=citation),
        media_ids=frozenset(media_ids),
    )


def _build_records() -> tuple[MockRecord, ...]:
    """Five fixed records: three source types and two media ids, all fictional.

    `example.com` is used on purpose: a mock URL must never look like a real attribution.
    """
    return (
        _record(
            chunk_id="doc-90210::A::1",
            score=0.93,
            quote=(
                "电视动画《星海旅人》第二季确定于 2026 年 10 月 4 日开播。官方同时公开了首支预告片"
                "与主视觉图，制作阵容保持第一季原班人马。"
            ),
            media_ids=(90210,),
            citation=_citation(
                source_type="official",
                publisher="Aurora Animation",
                title="第二季开播公告",
                url="https://example.com/acgn-chronicle/mock/aurora/season-2",
                published_at="2026-09-12",
                chunk_id="doc-90210::A::1",
            ),
        ),
        _record(
            chunk_id="doc-90210::A::2",
            score=0.88,
            quote=(
                "Anime News Network reports that Star Sea Traveler season two will premiere on "
                "4 October 2026, with the original cast and studio returning."
            ),
            media_ids=(90210,),
            citation=_citation(
                source_type="media",
                publisher="Anime News Network",
                title="Star Sea Traveler season 2 premiere date",
                url="https://example.com/acgn-chronicle/mock/ann/season-2",
                published_at="2026-09-13",
                chunk_id="doc-90210::A::2",
            ),
        ),
        _record(
            chunk_id="doc-90210::A::3",
            score=0.84,
            quote=(
                "AniList lists Star Sea Traveler (media id 90210) as a TV series with 12 episodes, "
                "status upcoming, and an average score of 78."
            ),
            media_ids=(90210,),
            citation=_citation(
                source_type="metadata",
                publisher="AniList",
                title="Star Sea Traveler",
                url="https://example.com/acgn-chronicle/mock/anilist/90210",
                published_at="2026-09-01",
                chunk_id="doc-90210::A::3",
            ),
        ),
        _record(
            chunk_id="doc-90210::A::4",
            score=0.79,
            quote=(
                "Blu-ray 第一卷将于 2026 年 12 月 23 日发售，收录第 1 至 4 话；初回限定版附带"
                "设定集与原创广播剧 CD。"
            ),
            media_ids=(90210,),
            citation=_citation(
                source_type="official",
                publisher="Aurora Animation",
                title="Blu-ray 发售公告",
                url="https://example.com/acgn-chronicle/mock/aurora/blu-ray-1",
                published_at="2026-09-20",
                chunk_id="doc-90210::A::4",
            ),
        ),
        _record(
            chunk_id="doc-77123::A::1",
            score=0.71,
            quote=(
                "《月灯食堂》剧场版定于 2027 年 1 月 15 日在日本上映，官方公开了特报影像与"
                "追加声优名单。"
            ),
            media_ids=(77123,),
            citation=_citation(
                source_type="official",
                publisher="Moonlight Diner Committee",
                title="剧场版上映公告",
                url="https://example.com/acgn-chronicle/mock/moonlight/movie",
                published_at="2026-09-25",
                chunk_id="doc-77123::A::1",
            ),
        ),
    )


MOCK_RECORDS: tuple[MockRecord, ...] = _build_records()


def _parse_timestamp(value: str) -> datetime:
    """Parse a date or an ISO datetime; naive values mean UTC (KnowledgePort §3.3)."""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise InvalidFilter(f"Invalid ISO date or datetime: {value!r}.") from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _bound(value: str, *, end_of_day: bool) -> datetime:
    parsed = _parse_timestamp(value)
    if len(value) == 10 and end_of_day:
        return parsed + timedelta(days=1)
    return parsed


def _is_date_only(value: str) -> bool:
    return len(value) == 10


def _time_bounds(filters: RetrievalFilters) -> tuple[datetime | None, datetime | None]:
    """Validate bounds, expanding a date-only `until` to the next midnight.

    Two date-only bounds are compared as calendar dates *before* `until` expands, so an
    inverted adjacent pair (`since=10-05`, `until=10-04`) is a `422`, not a one-day interval.
    A mixed pair is compared after expansion, so `since=2026-09-25T12:00:00Z`,
    `until=2026-09-25` is valid: its upper instant is 2026-09-26T00:00Z.
    """
    since_text = filters.since
    until_text = filters.until
    since = _parse_timestamp(since_text) if since_text else None
    until = _parse_timestamp(until_text) if until_text else None
    if until is None or until_text is None:
        return since, None

    widened_until = _bound(until_text, end_of_day=True)
    if since is not None:
        both_date_only = _is_date_only(since_text) and _is_date_only(until_text)
        upper = until if both_date_only else widened_until
        if since > upper:
            raise InvalidFilter("`since` must not be later than `until`.")
    return since, widened_until


def _abstain(reason: str) -> GroundedAnswer:
    """The refusal shape: empty citations and a non-empty reason, never a guessed answer."""
    return GroundedAnswer(
        answer=reason,
        citations=[],
        cited_chunk_ids=[],
        abstained=True,
        abstain_reason=reason,
    )


class MockKnowledgePort:
    """Fixed backend implementing the four KnowledgePort reads for the G1a page."""

    def __init__(self, records: tuple[MockRecord, ...] = MOCK_RECORDS) -> None:
        self._records = records
        self._snapshots = {
            CURRENT_SNAPSHOT.snapshot_id: CURRENT_SNAPSHOT,
            RETAINED_SNAPSHOT.snapshot_id: RETAINED_SNAPSHOT,
        }

    def inventory(self) -> SnapshotInventory:
        return SnapshotInventory(
            current=CURRENT_SNAPSHOT.snapshot_id,
            retained=(CURRENT_SNAPSHOT.snapshot_id, RETAINED_SNAPSHOT.snapshot_id),
        )

    def snapshot(self, snapshot_id: str | None = None) -> CorpusSnapshot:
        if snapshot_id is None:
            return CURRENT_SNAPSHOT
        if snapshot_id == RETIRED_SNAPSHOT_ID:
            raise RetiredSnapshot(
                f"Snapshot {snapshot_id} is retired; re-read /snapshots and start a new request."
            )
        if snapshot_id not in self._snapshots:
            raise UnknownSnapshot(f"Snapshot {snapshot_id} is not known to this host.")
        return self._snapshots[snapshot_id]

    def retrieve(self, query: RetrievalQuery) -> list[ChunkExcerpt]:
        self.snapshot(query.snapshot_id)
        since, until = _time_bounds(query.filters or RetrievalFilters())
        if not query.text.strip():
            return []
        return self._select(query, since=since, until=until)

    def ask(self, query: RetrievalQuery) -> GroundedAnswer:
        self.snapshot(query.snapshot_id)
        since, until = _time_bounds(query.filters or RetrievalFilters())
        text = query.text.strip()
        if not text:
            return _abstain("问题为空，没有可检索的文本。")
        excerpts = self._select(query, since=since, until=until)
        if not excerpts:
            return _abstain("当前筛选条件下没有可用证据；系统拒绝在没有依据时作答。")
        if any(term in text for term in _PRIVATE_TERMS):
            return _abstain("问题涉及未公开的个人信息，知识库没有相关证据。")

        citations: list[Citation] = []
        seen: set[str] = set()
        for excerpt in excerpts:
            if excerpt.citation.chunk_id in seen:
                continue
            seen.add(excerpt.citation.chunk_id)
            citations.append(excerpt.citation)

        answer = (
            "（G1a 模拟回答）本次检索返回 "
            f"{len(excerpts)} 条片段，最相关来源为 {excerpts[0].citation.publisher}"
            f"（{excerpts[0].citation.source_type}）。以下引用均来自本次检索结果；"
            "客户端只展示，不补全或改写原文。"
        )
        return GroundedAnswer(
            answer=answer,
            citations=citations,
            cited_chunk_ids=[excerpt.chunk_id for excerpt in excerpts],
            abstained=False,
            abstain_reason=None,
        )

    def _select(
        self,
        query: RetrievalQuery,
        *,
        since: datetime | None,
        until: datetime | None,
    ) -> list[ChunkExcerpt]:
        filters = query.filters or RetrievalFilters()
        source_types = set(filters.source_types or ())
        media_ids = set(filters.media_ids or ())
        selected: list[ChunkExcerpt] = []
        for record in self._records:
            citation = record.excerpt.citation
            if source_types and citation.source_type not in source_types:
                continue
            if media_ids and not media_ids.intersection(record.media_ids):
                continue
            published_at = _parse_timestamp(citation.published_at)
            if since is not None and published_at < since:
                continue
            if until is not None and published_at >= until:
                continue
            selected.append(record.excerpt)

        selected.sort(key=lambda excerpt: (-excerpt.score, excerpt.chunk_id))
        if query.top_k <= 0:
            return []
        return selected[: query.top_k]


__all__ = [
    "CURRENT_SNAPSHOT",
    "CURRENT_SNAPSHOT_ID",
    "MOCK_CONFIG",
    "MOCK_RECORDS",
    "RETAINED_SNAPSHOT",
    "RETAINED_SNAPSHOT_ID",
    "RETIRED_SNAPSHOT_ID",
    "MockKnowledgePort",
    "MockRecord",
]
