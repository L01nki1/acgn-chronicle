"""WordPress paged adapter (`WpPagedAdapter`) supporting WP REST API (`/wp-json/wp/v2/posts`)
and paged RSS feeds (`/feed/?paged=N`) for anime studios (MAPPA, BONES, P.A.WORKS).

Owner: A (WP1) | Spec: docs/SOURCES.md §3 & §7, Issue #9
"""

from __future__ import annotations

import hashlib
import json
import urllib.parse
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
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
    per_page: int = 100
    max_pages: int = 300
    categories: tuple[int, ...] = ()
    canonical_host: str | None = None
    use_rest_date_filter: bool = True
    disk_cache_dir: Path | None = None
    raise_on_duplicate_page: bool = False
    http: PoliteHttpClient = field(default_factory=PoliteHttpClient)
    cache_pages: bool = True
    _cached_items: list[RawIngestItem] | None = field(init=False, default=None, repr=False)
    _cached_window: tuple[str | None, str | None] | None = field(
        init=False, default=None, repr=False
    )
    last_skipped_entries: int = field(init=False, default=0, repr=False)
    last_not_modified: bool = field(init=False, default=False, repr=False)
    last_page_cap_hit: bool = field(init=False, default=False, repr=False)
    last_pages_fetched: int = field(init=False, default=0, repr=False)
    last_reached_window_start: bool = field(init=False, default=False, repr=False)
    last_earliest_observed_date: str | None = field(init=False, default=None, repr=False)
    last_effective_per_page: int = field(init=False, default=100, repr=False)

    def _apply_canonical_host(self, raw_link: str) -> str:
        if not self.canonical_host:
            return raw_link
        parsed = urllib.parse.urlsplit(raw_link)
        if not parsed.netloc or parsed.netloc.lower() == self.canonical_host.lower():
            return raw_link
        return urllib.parse.urlunsplit(
            (
                parsed.scheme or "https",
                self.canonical_host,
                parsed.path,
                parsed.query,
                parsed.fragment,
            )
        )

    def _rest_url(
        self,
        page: int,
        *,
        per_page: int | None = None,
        after_iso: str | None = None,
        before_iso: str | None = None,
    ) -> str:
        base = self.site_url.rstrip("/")
        endpoint = base if base.endswith("/wp-json/wp/v2/posts") else f"{base}/wp-json/wp/v2/posts"
        sep = "&" if "?" in endpoint else "?"
        eff_per_page = per_page if per_page is not None else self.per_page
        url = f"{endpoint}{sep}per_page={eff_per_page}&page={page}"
        if self.categories:
            cats = ",".join(str(c) for c in self.categories)
            url += f"&categories={cats}"
        if after_iso:
            after_ts = after_iso if "T" in after_iso else f"{after_iso}T00:00:00"
            url += f"&after={after_ts}"
        if before_iso:
            before_ts = before_iso if "T" in before_iso else f"{before_iso}T23:59:59"
            url += f"&before={before_ts}"
        return url

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
                canonical = normalize_url(self._apply_canonical_host(link))
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
            content_parts: list[str] = []
            content_list = getter("content")
            if isinstance(content_list, list):
                for content_item in content_list:
                    if isinstance(content_item, dict):
                        val = str(content_item.get("value") or "").strip()
                        if val:
                            content_parts.append(val)
            elif isinstance(content_list, str) and content_list.strip():
                content_parts.append(content_list.strip())
            for field_name in ("summary", "description"):
                val = str(getter(field_name) or "").strip()
                if val and val not in content_parts:
                    content_parts.append(val)
            raw_summary = "\n\n".join(content_parts).strip()
            pub_utc = _entry_published_utc(entry, source_tz=self.source_tz)
            if not link or not raw_title or not pub_utc:
                self.last_skipped_entries += 1
                continue
            try:
                canonical = normalize_url(self._apply_canonical_host(link))
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

    def _load_disk_cached_window(self, start_iso: str, end_iso: str) -> list[RawIngestItem] | None:
        if self.disk_cache_dir is None:
            return None
        cache_file = self.disk_cache_dir / f"{self.source_key}.json"
        if not cache_file.exists():
            return None
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("v") != 2:
                return None
            if payload.get("page_cap_hit", True):
                return None
            cov_start = str(payload.get("covered_start") or "")
            cov_end = str(payload.get("covered_end") or "")
            if not (cov_start and cov_end and cov_start <= start_iso and end_iso <= cov_end):
                return None
            raw_rows = payload.get("items")
            if not isinstance(raw_rows, list):
                return None
            items = [
                RawIngestItem(
                    url=str(r["url"]),
                    title=str(r["title"]),
                    body=str(r["body"]),
                    published_at=str(r["published_at"]),
                    publisher=str(r["publisher"]),
                    source_type=r["source_type"],
                    source_key=str(r["source_key"]),
                    lang=str(r.get("lang", self.lang)),
                    source_tz=r.get("source_tz"),
                    ingest_category=r.get("ingest_category"),
                )
                for r in raw_rows
                if isinstance(r, dict)
            ]
            self.last_page_cap_hit = False
            self.last_pages_fetched = int(payload.get("pages_fetched") or 1)
            earliest = payload.get("earliest_observed_date")
            self.last_earliest_observed_date = str(earliest) if earliest else None
            self.last_reached_window_start = bool(
                self.last_earliest_observed_date
                and self.last_earliest_observed_date
                <= (date.fromisoformat(start_iso) + timedelta(days=14)).isoformat()
            )
            return items
        except Exception:
            return None

    def _save_disk_cached_window(
        self, start_iso: str, end_iso: str, items: list[RawIngestItem]
    ) -> None:
        if self.disk_cache_dir is None or self.last_page_cap_hit:
            return
        try:
            self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.disk_cache_dir / f"{self.source_key}.json"
            payload = {
                "v": 2,
                "source_key": self.source_key,
                "covered_start": start_iso,
                "covered_end": end_iso,
                "page_cap_hit": self.last_page_cap_hit,
                "pages_fetched": self.last_pages_fetched,
                "earliest_observed_date": self.last_earliest_observed_date,
                "items": [
                    {
                        "url": it.url,
                        "title": it.title,
                        "body": it.body,
                        "published_at": it.published_at,
                        "publisher": it.publisher,
                        "source_type": it.source_type,
                        "source_key": it.source_key,
                        "lang": it.lang,
                        "source_tz": it.source_tz,
                        "ingest_category": it.ingest_category,
                    }
                    for it in items
                ],
            }
            cache_file.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
        except Exception:
            return

    def _crawl(
        self,
        *,
        max_pages: int,
        stop_before_date: str | None = None,
        after_iso: str | None = None,
        before_iso: str | None = None,
        use_conditional_first_page: bool = False,
    ) -> list[RawIngestItem]:
        self.last_not_modified = False
        self.last_page_cap_hit = False
        self.last_pages_fetched = 0
        self.last_reached_window_start = False

        if (
            self.cache_pages
            and not use_conditional_first_page
            and self._cached_items is not None
            and self._cached_window is not None
        ):
            cached_start, cached_end = self._cached_window
            if (
                cached_start is None
                or (stop_before_date is not None and cached_start <= stop_before_date)
            ) and (cached_end is None or (before_iso is not None and before_iso <= cached_end)):
                if stop_before_date is not None and self.last_earliest_observed_date is not None:
                    self.last_reached_window_start = (
                        self.last_earliest_observed_date
                        <= (date.fromisoformat(stop_before_date) + timedelta(days=14)).isoformat()
                    )
                return list(self._cached_items)

        self.last_skipped_entries = 0
        self.last_earliest_observed_date = None
        active_mode: Literal["rest", "paged_rss"] = (
            "paged_rss" if self.mode == "paged_rss" else "rest"
        )
        effective_per_page = self.per_page
        self.last_effective_per_page = effective_per_page
        collected: dict[str, RawIngestItem] = {}
        seen_fps: set[str] = set()
        terminated_before_cap = False

        for page in range(1, max_pages + 1):
            url = (
                self._rest_url(
                    page,
                    per_page=effective_per_page,
                    after_iso=after_iso if self.use_rest_date_filter else None,
                    before_iso=before_iso if self.use_rest_date_filter else None,
                )
                if active_mode == "rest"
                else self._rss_url(page)
            )
            try:
                res = self.http.get(
                    url,
                    use_conditional=(use_conditional_first_page and page == 1),
                )
            except IngestHttpError as exc:
                # WordPress REST returns 400 when `page` exceeds total pages
                if page > 1 and exc.status_code in (400, 404):
                    terminated_before_cap = True
                    break
                # Some WordPress installs reject per_page=100 with HTTP 400; retry with per_page=50
                if (
                    page == 1
                    and active_mode == "rest"
                    and exc.status_code == 400
                    and effective_per_page > 50
                ):
                    effective_per_page = 50
                    self.last_effective_per_page = effective_per_page
                    url = self._rest_url(
                        1,
                        per_page=effective_per_page,
                        after_iso=after_iso if self.use_rest_date_filter else None,
                        before_iso=before_iso if self.use_rest_date_filter else None,
                    )
                    try:
                        res = self.http.get(url, use_conditional=use_conditional_first_page)
                    except IngestHttpError:
                        if self.mode == "auto":
                            active_mode = "paged_rss"
                            url = self._rss_url(1)
                            res = self.http.get(url, use_conditional=use_conditional_first_page)
                        else:
                            raise
                elif page == 1 and self.mode == "auto" and active_mode == "rest":
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
                terminated_before_cap = True
                break

            self.last_pages_fetched = page

            fp = _fingerprint(page_items)
            if fp in seen_fps:
                if self.raise_on_duplicate_page:
                    raise PaginationLoopError(
                        f"[{self.source_key}] Page {page} ({url!r}) repeated fingerprint {fp[:12]}"
                    )
                terminated_before_cap = True
                break
            seen_fps.add(fp)

            new_count = 0
            oldest_on_page = "9999-99-99"
            for item in page_items:
                item_date = item.published_at[:10]
                oldest_on_page = min(oldest_on_page, item_date)
                if (
                    self.last_earliest_observed_date is None
                    or item_date < self.last_earliest_observed_date
                ):
                    self.last_earliest_observed_date = item_date
                if item.url not in collected:
                    collected[item.url] = item
                    new_count += 1

            if max_pages > 1 and page % 20 == 0:
                import sys

                print(
                    f"[{self.source_key}] progress: page {page}/{max_pages}, "
                    f"oldest_on_page={oldest_on_page}, accumulated={len(collected)}",
                    file=sys.stderr,
                    flush=True,
                )

            if page > 1 and new_count == 0:
                if self.raise_on_duplicate_page:
                    raise PaginationLoopError(
                        f"[{self.source_key}] Page {page} ({url!r}) added 0 new items"
                    )
                terminated_before_cap = True
                break

            if stop_before_date is not None and oldest_on_page < stop_before_date:
                terminated_before_cap = True
                self.last_reached_window_start = True
                break

        if (
            not terminated_before_cap
            and stop_before_date is not None
            and self.last_earliest_observed_date is not None
            and self.last_earliest_observed_date >= stop_before_date
        ):
            self.last_page_cap_hit = True
            self.last_reached_window_start = False
        elif stop_before_date is not None and self.last_earliest_observed_date is not None:
            if not self.last_page_cap_hit and (
                self.last_earliest_observed_date
                <= (date.fromisoformat(stop_before_date) + timedelta(days=14)).isoformat()
            ):
                self.last_reached_window_start = True

        items = list(collected.values())
        if (
            self.cache_pages
            and not use_conditional_first_page
            and not self.last_page_cap_hit
            and max_pages >= self.max_pages
        ):
            self._cached_items = list(items)
            self._cached_window = (stop_before_date, before_iso)
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

        disk_cached = self._load_disk_cached_window(start_iso, end_iso)
        if disk_cached is not None:
            all_items = disk_cached
        else:
            all_items = self._crawl(
                max_pages=self.max_pages,
                stop_before_date=start_iso,
                after_iso=start_iso if self.use_rest_date_filter else None,
                before_iso=end_iso if self.use_rest_date_filter else None,
                use_conditional_first_page=False,
            )
            self._save_disk_cached_window(start_iso, end_iso, all_items)

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
