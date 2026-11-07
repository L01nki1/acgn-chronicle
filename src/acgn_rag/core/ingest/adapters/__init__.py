"""Two-mode `SourceAdapter` protocol and W1 adapter registry (`REGISTRY`).

Every adapter implements both:
  - `discover(since)` — incremental acquisition since a timestamp/date cursor
  - `backfill(date_range)` — historical acquisition over an explicit `(start_date, end_date)` window
    (a windowless call raises `ValueError` per COMPLIANCE C1)

Owner: A (WP1) | Spec: docs/SOURCES.md §3 & §7, docs/COMPLIANCE.md C1, Issue #9
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from acgn_rag.core.ingest.adapters.html_list import HtmlListAdapter, parse_html_list_page
from acgn_rag.core.ingest.adapters.rss import PRTIMES_ACGN_KEYWORDS, RssAdapter
from acgn_rag.core.ingest.adapters.structured_api import (
    ANILIST_GRAPHQL_URL,
    BANGUMI_SUBJECTS_URL,
    BILIBILI_TIMELINE_URL,
    SYOSETU_NOVELAPI_URL,
    StructuredApiAdapter,
)
from acgn_rag.core.ingest.adapters.wp_paged import WpPagedAdapter
from acgn_rag.core.ingest.ann_archive import (
    ANN_BASE_URL,
    ANN_PUBLISHER,
    ANN_RSS_URL,
    ANN_SOURCE_KEY,
    AdapterError,
    AnnArchiveAdapter,
    AnnArticleParseResult,
    AnnDayParseResult,
    BackfillNotSupportedError,
    EmptyBackfillWindowError,
    PaginationLoopError,
    SelectorBrokenError,
    ann_day_url,
    parse_ann_article_html,
    parse_ann_day_html,
    validate_backfill_window,
)
from acgn_rag.core.ingest.http import PoliteHttpClient
from acgn_rag.core.ingest.normalize import RawIngestItem


@runtime_checkable
class SourceAdapter(Protocol):
    """Two-mode ingestion adapter contract."""

    source_key: str
    publisher: str

    def discover(
        self,
        since: date | datetime | str | None = None,
        *,
        use_conditional: bool = True,
    ) -> list[RawIngestItem]:
        """Return items published on or after `since` (or recent feed items when `since` is None)."""
        ...

    def backfill(
        self,
        date_range: tuple[date | datetime | str | None, date | datetime | str | None] | None = None,
    ) -> list[RawIngestItem]:
        """Return historical items within `date_range = (start_date, end_date)`.

        Must raise `ValueError` when `date_range` is None or missing either bound (COMPLIANCE C1).
        """
        ...


def build_default_registry(
    http: PoliteHttpClient | None = None,
    *,
    ann_max_days_per_call: int | None = None,
    conditional_cache_path: Path | None = None,
    ann_cache_dir: Path | None = None,
    ann_article_cache_path: Path | None = None,
    wp_cache_dir: Path | None = None,
) -> dict[str, SourceAdapter]:
    """Build the W1 source adapter registry ordered highest-yield first (Issue #9)."""
    client = (
        http
        if http is not None
        else PoliteHttpClient(conditional_cache_path=conditional_cache_path)
    )
    resolved_ann_article_cache = ann_article_cache_path
    if resolved_ann_article_cache is None and ann_cache_dir is not None:
        json_candidate = ann_cache_dir.parent / "ann_articles_cache.json"
        zip_candidate = ann_cache_dir.parents[1] / "raw" / "ann-article-cache-20261102.zip"
        if json_candidate.exists():
            resolved_ann_article_cache = json_candidate
        elif zip_candidate.exists():
            resolved_ann_article_cache = zip_candidate
    return {
        "toei-news": HtmlListAdapter(
            source_key="toei-news",
            list_url="https://www.toei-anim.co.jp/news/",
            publisher="Toei Animation",
            source_type="official",
            lang="ja",
            source_tz="Asia/Tokyo",
            max_pages=1,
            link_pattern=r"(?:/ja/press/[^/]+\.html|/news/detail/[^/]+/?)$",
            http=client,
        ),
        "ann-news": AnnArchiveAdapter(
            http=client,
            source_key=ANN_SOURCE_KEY,
            publisher=ANN_PUBLISHER,
            rss_url=ANN_RSS_URL,
            base_url=ANN_BASE_URL,
            max_days_per_call=ann_max_days_per_call,
            cache_dir=ann_cache_dir,
            article_cache_path=resolved_ann_article_cache,
        ),
        "nijimen-news": WpPagedAdapter(
            source_key="nijimen-news",
            site_url="https://nijimen.kusuguru.co.jp",
            publisher="にじめん",
            source_type="media",
            lang="ja",
            source_tz="Asia/Tokyo",
            mode="rest",
            per_page=100,
            max_pages=300,
            canonical_host="nijimen.kusuguru.co.jp",
            disk_cache_dir=wp_cache_dir,
            http=client,
        ),
        "realsound-acgn": WpPagedAdapter(
            source_key="realsound-acgn",
            site_url="https://realsound.jp",
            publisher="Real Sound",
            source_type="media",
            lang="ja",
            source_tz="Asia/Tokyo",
            mode="rest",
            per_page=100,
            max_pages=50,
            categories=(116583, 116584),
            disk_cache_dir=wp_cache_dir,
            http=client,
        ),
        "capcom-news": RssAdapter(
            source_key="capcom-news",
            feed_url="https://www.capcom.co.jp/rss/news.xml",
            publisher="Capcom",
            source_type="official",
            lang="ja",
            source_tz="Asia/Tokyo",
            supports_backfill=True,
            http=client,
        ),
        "mappa-news": WpPagedAdapter(
            source_key="mappa-news",
            site_url="https://www.mappa.co.jp",
            publisher="MAPPA",
            source_type="official",
            lang="ja",
            source_tz="Asia/Tokyo",
            mode="auto",
            per_page=100,
            max_pages=300,
            disk_cache_dir=wp_cache_dir,
            http=client,
        ),
        "bones-news": WpPagedAdapter(
            source_key="bones-news",
            site_url="https://www.bones.co.jp",
            publisher="BONES",
            source_type="official",
            lang="ja",
            source_tz="Asia/Tokyo",
            mode="auto",
            per_page=100,
            max_pages=300,
            disk_cache_dir=wp_cache_dir,
            http=client,
        ),
        "paworks-news": WpPagedAdapter(
            source_key="paworks-news",
            site_url="https://www.pa-works.jp",
            publisher="P.A.WORKS",
            source_type="official",
            lang="ja",
            source_tz="Asia/Tokyo",
            mode="auto",
            per_page=100,
            max_pages=300,
            disk_cache_dir=wp_cache_dir,
            http=client,
        ),
        "prtimes-acgn": RssAdapter(
            source_key="prtimes-acgn",
            feed_url="https://prtimes.jp/index.rdf",
            publisher="PR TIMES",
            source_type="media",
            lang="ja",
            source_tz="Asia/Tokyo",
            keyword_filter=PRTIMES_ACGN_KEYWORDS,
            supports_backfill=False,
            http=client,
        ),
        "gcores-news": RssAdapter(
            source_key="gcores-news",
            feed_url="https://www.gcores.com/rss",
            publisher="Gcores",
            source_type="media",
            lang="zh",
            source_tz="Asia/Shanghai",
            supports_backfill=False,
            http=client,
        ),
        "chuapp-news": RssAdapter(
            source_key="chuapp-news",
            feed_url="https://www.chuapp.com/feed",
            publisher="Chuapp",
            source_type="media",
            lang="zh",
            source_tz="Asia/Shanghai",
            supports_backfill=False,
            http=client,
        ),
        "yystv-news": RssAdapter(
            source_key="yystv-news",
            feed_url="https://www.yystv.cn/rss/feed",
            publisher="Yystv",
            source_type="media",
            lang="zh",
            source_tz="Asia/Shanghai",
            supports_backfill=False,
            http=client,
        ),
        "anilist-graphql": StructuredApiAdapter(
            source_key="anilist-graphql",
            provider="anilist",
            endpoint_url=ANILIST_GRAPHQL_URL,
            publisher="AniList",
            source_type="metadata",
            lang="en",
            source_tz="UTC",
            max_pages=3,
            per_page=50,
            http=client,
        ),
        "bangumi-api": StructuredApiAdapter(
            source_key="bangumi-api",
            provider="bangumi",
            endpoint_url=BANGUMI_SUBJECTS_URL,
            publisher="Bangumi",
            source_type="metadata",
            lang="zh",
            source_tz="Asia/Shanghai",
            max_pages=2,
            per_page=50,
            http=client,
        ),
        "bilibili-timeline": StructuredApiAdapter(
            source_key="bilibili-timeline",
            provider="bilibili",
            endpoint_url=BILIBILI_TIMELINE_URL,
            publisher="Bilibili",
            source_type="metadata",
            lang="zh",
            source_tz="Asia/Shanghai",
            http=client,
        ),
        "syosetu-novelapi": StructuredApiAdapter(
            source_key="syosetu-novelapi",
            provider="syosetu",
            endpoint_url=SYOSETU_NOVELAPI_URL,
            publisher="Syosetu",
            source_type="metadata",
            lang="ja",
            source_tz="Asia/Tokyo",
            http=client,
        ),
    }


REGISTRY: dict[str, SourceAdapter] = build_default_registry()

__all__ = [
    "ANILIST_GRAPHQL_URL",
    "ANN_BASE_URL",
    "ANN_PUBLISHER",
    "ANN_RSS_URL",
    "ANN_SOURCE_KEY",
    "BANGUMI_SUBJECTS_URL",
    "BILIBILI_TIMELINE_URL",
    "PRTIMES_ACGN_KEYWORDS",
    "REGISTRY",
    "SYOSETU_NOVELAPI_URL",
    "AdapterError",
    "AnnArchiveAdapter",
    "AnnArticleParseResult",
    "AnnDayParseResult",
    "BackfillNotSupportedError",
    "EmptyBackfillWindowError",
    "HtmlListAdapter",
    "PaginationLoopError",
    "RssAdapter",
    "SelectorBrokenError",
    "SourceAdapter",
    "StructuredApiAdapter",
    "WpPagedAdapter",
    "ann_day_url",
    "build_default_registry",
    "parse_ann_article_html",
    "parse_ann_day_html",
    "parse_html_list_page",
    "validate_backfill_window",
]
