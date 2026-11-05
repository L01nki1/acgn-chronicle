"""WordPress paged adapter (`WpPagedAdapter`) supporting WP REST API (`/wp-json/wp/v2/posts`)
and paged RSS feeds (`/feed/?paged=N`) for anime studios (MAPPA, BONES, P.A.WORKS).

Owner: A (WP1) | Spec: docs/SOURCES.md §3 & §7, Issue #9
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal

import feedparser

from acgn_rag.core.ingest.adapters.rss import _entry_published_utc, _fallback_parse_xml_items
from acgn_rag.core.ingest.ann_archive import (
    EmptyBackfillWindowError,
    PaginationLoopError,
    SelectorBrokenError,
    validate_backfill_window,
)
from acgn_rag.core.ingest.http import IngestHttpError, PoliteHttpClient
from acgn_rag.core.ingest.normalize import (
    RawIngestItem,
    clean_title,
    is_corporate_ir_text,
    normalize_url,
    parse_utc_timestamp,
    strip_html_boilerplate,
    utc_iso_at_or_after,
)
from acgn_rag.core.ports import SourceType


def _fingerprint(items: list[RawIngestItem]) -> str:
    payload = "\n".join(sorted(item.url for item in items)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _extract_rendered(field_obj: object) -> str:
    if isinstance(field_obj, dict):
        return str(field_obj.get("rendered") or "").strip()
    return str(field_obj or "").strip()


@dataclass
class WpPagedAdapter:
    """Two-mode WordPress adapter supporting REST (`/wp-json/wp/v2/posts`) and `?paged=N` feeds."""

    source_key: str
    site_url: str
    publisher: str
    source_type: SourceType = "official"
    lang: str = "ja"
    source_tz: str | None = "Asia/Tokyo"
    mode: Literal["rest", "paged_rss", "auto"] = "auto"
    per_page: int = 50
    max_pages: int = 5
    raise_on_duplicate_page: bool = False
    http: PoliteHttpClient = field(default_factory=PoliteHttpClient)
    cache_pages: bool = True
    _cached_items: list[RawIngestItem] | None = field(init=False, default=None, repr=False)
    last_skipped_entries: int = field(init=False, default=0, repr=False)
    last_not_modified: bool = field(init=False, default=False, repr=False)

    def _rest_url(self, page: int) -> str:
        base = self.site_url.rstrip("/")
        endpoint = base if base.endswith("/wp-json/wp/v2/posts") else f"{base}/wp-json/wp/v2/posts"
        sep = "&" if "?" in endpoint else "?"
        return f"{endpoint}{sep}per_page={self.per_page}&page={page}"

    def _rss_url(self, page: int) -> str:
        base = self.site_url.rstrip("/")
        feed = base if base.endswith(("/feed", "/feed/", ".xml", ".rdf")) else f"{base}/feed/"
        if page <= 1:
            return feed
        sep = "&" if "?" in feed else "?"
        return f"{feed}{sep}paged={page}"

    def _parse_rest_page(self, raw_json: str, url: str) -> list[RawIngestItem]:
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise SelectorBrokenError(
                f"[{self.source_key}] Invalid WP REST JSON from {url!r}: {exc}"
            ) from exc

        if not isinstance(payload, list):
            raise SelectorBrokenError(
                f"[{self.source_key}] WP REST response at {url!r} is not a JSON list"
            )
        if not payload:
            return []

        items: list[RawIngestItem] = []
        for post in payload:
            if not isinstance(post, dict):
                self.last_skipped_entries += 1
                continue
            link = str(post.get("link") or "").strip()
            raw_title = _extract_rendered(post.get("title"))
            raw_content = _extract_rendered(post.get("content"))
            raw_excerpt = _extract_rendered(post.get("excerpt"))
            date_gmt = post.get("date_gmt")
            date_local = post.get("date")
            if not link or not raw_title or not (date_gmt or date_local):
                self.last_skipped_entries += 1
                continue

            try:
                if date_gmt:
                    pub_utc = parse_utc_timestamp(str(date_gmt), source_tz="UTC")
                else:
                    pub_utc = parse_utc_timestamp(str(date_local), source_tz=self.source_tz)
                canonical = normalize_url(link)
                title = clean_title(raw_title)
            except Exception:
                self.last_skipped_entries += 1
                continue

            content_clean = strip_html_boilerplate(raw_content) if raw_content else ""
            excerpt_clean = strip_html_boilerplate(raw_excerpt) if raw_excerpt else ""
            body = content_clean or excerpt_clean or title

            ingest_cat = (
                "corporate"
                if self.source_type == "official" and is_corporate_ir_text(f"{title}\n{body}")
                else None
            )
            items.append(
                RawIngestItem(
                    url=canonical,
                    title=title,
                    body=body,
                    published_at=pub_utc,
                    publisher=self.publisher,
                    source_type=self.source_type,
                    source_key=self.source_key,
                    lang=self.lang,
                    source_tz=self.source_tz,
                    ingest_category=ingest_cat,
                )
            )
        return items

    def _parse_rss_page(self, raw_xml: str, url: str) -> list[RawIngestItem]:
        parsed = feedparser.parse(raw_xml)
        entries: list[object] = list(getattr(parsed, "entries", []) or [])
        if not entries:
            entries = list(_fallback_parse_xml_items(raw_xml))
        if not entries:
            return []

        items: list[RawIngestItem] = []
        for entry in entries:
            getter = (
                entry.get if isinstance(entry, dict) else lambda k, e=entry: getattr(e, k, None)
            )
            link = str(getter("link") or "").strip()
            raw_title = str(getter("title") or "").strip()
            raw_summary = str(getter("summary") or getter("description") or "").strip()
            pub_utc = _entry_published_utc(entry, source_tz=self.source_tz)
            if not link or not raw_title or not pub_utc:
                self.last_skipped_entries += 1
                continue
            try:
                canonical = normalize_url(link)
                title = clean_title(raw_title)
            except Exception:
                self.last_skipped_entries += 1
                continue
            cleaned_summary = strip_html_boilerplate(raw_summary) if raw_summary else ""
            body = cleaned_summary if cleaned_summary else title

            ingest_cat = (
                "corporate"
                if self.source_type == "official" and is_corporate_ir_text(f"{title}\n{body}")
                else None
            )
            items.append(
                RawIngestItem(
                    url=canonical,
                    title=title,
                    body=body,
                    published_at=pub_utc,
                    publisher=self.publisher,
                    source_type=self.source_type,
                    source_key=self.source_key,
                    lang=self.lang,
                    source_tz=self.source_tz,
                    ingest_category=ingest_cat,
                )
            )
        return items

    def _crawl(
        self,
        *,
        max_pages: int,
        stop_before_date: str | None = None,
        use_conditional_first_page: bool = False,
    ) -> list[RawIngestItem]:
        self.last_not_modified = False
        if (
            self.cache_pages
            and not use_conditional_first_page
            and stop_before_date is None
            and self._cached_items is not None
        ):
            return list(self._cached_items)

        self.last_skipped_entries = 0
        active_mode: Literal["rest", "paged_rss"] = (
            "paged_rss" if self.mode == "paged_rss" else "rest"
        )
        collected: dict[str, RawIngestItem] = {}
        seen_fps: set[str] = set()

        for page in range(1, max_pages + 1):
            url = self._rest_url(page) if active_mode == "rest" else self._rss_url(page)
            try:
                res = self.http.get(
                    url,
                    use_conditional=(use_conditional_first_page and page == 1),
                )
            except IngestHttpError as exc:
                # WordPress REST returns 400 when `page` exceeds total pages
                if page > 1 and exc.status_code in (400, 404):
                    break
                if page == 1 and self.mode == "auto" and active_mode == "rest":
                    active_mode = "paged_rss"
                    url = self._rss_url(1)
                    res = self.http.get(url, use_conditional=use_conditional_first_page)
                else:
                    raise

            if page == 1 and res.not_modified:
                self.last_not_modified = True
                return []

            if active_mode == "rest":
                try:
                    page_items = self._parse_rest_page(res.text, url)
                except SelectorBrokenError:
                    if page == 1 and self.mode == "auto":
                        active_mode = "paged_rss"
                        rss_url = self._rss_url(1)
                        rss_res = self.http.get(rss_url, use_conditional=use_conditional_first_page)
                        if rss_res.not_modified:
                            self.last_not_modified = True
                            return []
                        page_items = self._parse_rss_page(rss_res.text, rss_url)
                    else:
                        raise
            else:
                page_items = self._parse_rss_page(res.text, url)

            if not page_items:
                if page == 1:
                    raise SelectorBrokenError(
                        f"[{self.source_key}] WordPress page 1 ({url!r}) yielded 0 items"
                    )
                break

            fp = _fingerprint(page_items)
            if fp in seen_fps:
                if self.raise_on_duplicate_page:
                    raise PaginationLoopError(
                        f"[{self.source_key}] Page {page} ({url!r}) repeated fingerprint {fp[:12]}"
                    )
                break
            seen_fps.add(fp)

            new_count = 0
            oldest_on_page = "9999-99-99"
            for item in page_items:
                oldest_on_page = min(oldest_on_page, item.published_at[:10])
                if item.url not in collected:
                    collected[item.url] = item
                    new_count += 1

            if page > 1 and new_count == 0:
                if self.raise_on_duplicate_page:
                    raise PaginationLoopError(
                        f"[{self.source_key}] Page {page} ({url!r}) added 0 new items"
                    )
                break

            if stop_before_date is not None and oldest_on_page < stop_before_date:
                break

        items = list(collected.values())
        if (
            self.cache_pages
            and not use_conditional_first_page
            and stop_before_date is None
            and max_pages >= self.max_pages
        ):
            self._cached_items = list(items)
        return items

    def discover(
        self,
        since: date | datetime | str | None = None,
        *,
        use_conditional: bool = True,
    ) -> list[RawIngestItem]:
        """Incremental discovery from page 1 of the WordPress site."""
        since_ts: str | None = None
        if since is not None:
            since_ts = parse_utc_timestamp(since, source_tz=self.source_tz)

        items = self._crawl(max_pages=1, use_conditional_first_page=use_conditional)
        if not items:
            return []
        if since_ts is None:
            return items
        return [item for item in items if utc_iso_at_or_after(item.published_at, since_ts)]

    def backfill(
        self,
        date_range: tuple[date | datetime | str | None, date | datetime | str | None] | None = None,
    ) -> list[RawIngestItem]:
        """Historical backfill across up to `max_pages` within `[start_date, end_date]`."""
        start_d, end_d = validate_backfill_window(date_range)
        start_iso = start_d.isoformat()
        end_iso = end_d.isoformat()

        all_items = self._crawl(
            max_pages=self.max_pages,
            stop_before_date=None if self.cache_pages else start_iso,
            use_conditional_first_page=False,
        )
        window_items = [
            item for item in all_items if start_iso <= item.published_at[:10] <= end_iso
        ]
        if not window_items:
            raise EmptyBackfillWindowError(
                f"[{self.source_key}] 0 WordPress items fell inside window {start_iso}..{end_iso}"
            )
        return window_items


__all__ = [
    "WpPagedAdapter",
]
