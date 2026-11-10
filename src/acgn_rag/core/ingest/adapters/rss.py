"""RSS / Atom / RDF feed adapter (`RssAdapter`) for full-history feeds (Capcom) and
rolling media/press feeds (PR TIMES with ACGN keyword filter, Gcores, Chuapp, Yystv).

Owner: A (WP1) | Spec: docs/SOURCES.md §3 & §7, Issue #9
"""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass, field
from datetime import date, datetime

import feedparser

from acgn_rag.core.ingest.ann_archive import (
    BackfillNotSupportedError,
    EmptyBackfillWindowError,
    SelectorBrokenError,
    validate_backfill_window,
)
from acgn_rag.core.ingest.http import PoliteHttpClient
from acgn_rag.core.ingest.normalize import (
    RawIngestItem,
    clean_title,
    is_corporate_ir_text,
    is_fiscal_pr_boilerplate_title,
    normalize_text,
    normalize_url,
    parse_utc_timestamp,
    promote_boilerplate_title,
    strip_html_boilerplate,
    utc_iso_at_or_after,
)
from acgn_rag.core.ports import SourceType

PRTIMES_ACGN_KEYWORDS: tuple[str, ...] = (
    "アニメ",
    "マンガ",
    "漫画",
    "ゲーム",
    "声優",
    "配信",
    "ライトノベル",
    "小説",
    "劇場版",
    "コラボ",
    "VTuber",
)

_ITEM_BLOCK_RE = re.compile(r"<item\b[^>]*>(.*?)</item>", re.I | re.S)
_TAG_EXTRACT_RE = {
    "title": re.compile(r"<title\b[^>]*>(.*?)</title>", re.I | re.S),
    "link": re.compile(r"<link\b[^>]*>(.*?)</link>", re.I | re.S),
    "description": re.compile(
        r"<(?:description|content:encoded|summary)\b[^>]*>(.*?)</(?:description|content:encoded|summary)>",
        re.I | re.S,
    ),
    "pubDate": re.compile(
        r"<(?:pubDate|dc:date|published|updated)\b[^>]*>(.*?)</(?:pubDate|dc:date|published|updated)>",
        re.I | re.S,
    ),
}
_CDATA_RE = re.compile(r"^\s*<!\[CDATA\[(.*?)\]\]>\s*$", re.S)


def _strip_cdata(text: str) -> str:
    match = _CDATA_RE.match(text)
    return match.group(1).strip() if match else text.strip()


def _fallback_parse_xml_items(xml_text: str) -> list[dict[str, str]]:
    """Fallback regex parser for RSS/RDF `<item>` blocks if feedparser yields no entries."""
    items: list[dict[str, str]] = []
    for match in _ITEM_BLOCK_RE.finditer(xml_text):
        block = match.group(1)
        row: dict[str, str] = {}
        for key, pattern in _TAG_EXTRACT_RE.items():
            found = pattern.search(block)
            if found:
                row[key] = _strip_cdata(found.group(1))
        if row.get("link") and row.get("title"):
            items.append(row)
    return items


def _entry_published_utc(entry: object, *, source_tz: str | None) -> str | None:
    """Extract and normalize an entry's publication timestamp to UTC `Z`.

    Note: Never uses feedparser's `published_parsed`/`updated_parsed` structs because
    feedparser silently assumes UTC for naive timestamps, violating INGEST_CONTRACT §1.
    """
    getter = entry.get if isinstance(entry, dict) else lambda k: getattr(entry, k, None)
    for field_name in ("published", "updated", "pubDate", "dc_date", "date"):
        raw_val = getter(field_name)
        if raw_val:
            try:
                return parse_utc_timestamp(str(raw_val), source_tz=source_tz)
            except Exception:
                continue
    return None


@dataclass
class RssAdapter:
    """Two-mode RSS/Atom/RDF adapter with keyword filtering and corporate/IR tagging."""

    source_key: str
    feed_url: str
    publisher: str
    source_type: SourceType
    lang: str
    source_tz: str | None = None
    http: PoliteHttpClient = field(default_factory=PoliteHttpClient)
    keyword_filter: tuple[str, ...] = ()
    supports_backfill: bool = True
    tag_corporate_ir: bool = True
    cache_feed: bool = True
    _cached_items: list[RawIngestItem] | None = field(init=False, default=None, repr=False)
    last_raw_entry_count: int = field(init=False, default=0, repr=False)
    last_skipped_entries: int = field(init=False, default=0, repr=False)
    last_keyword_filtered_entries: int = field(init=False, default=0, repr=False)
    last_duplicate_url_entries: int = field(init=False, default=0, repr=False)
    last_earliest_observed_date: str | None = field(init=False, default=None, repr=False)
    last_not_modified: bool = field(init=False, default=False, repr=False)

    def _fetch_all_items(self, *, use_conditional: bool) -> list[RawIngestItem]:
        self.last_not_modified = False
        if self.cache_feed and not use_conditional and self._cached_items is not None:
            return list(self._cached_items)

        res = self.http.get(self.feed_url, use_conditional=use_conditional)
        if res.not_modified:
            self.last_not_modified = True
            return []

        parsed = feedparser.parse(res.text, sanitize_html=False, resolve_relative_uris=False)
        entries: list[object] = list(getattr(parsed, "entries", []) or [])
        if not entries:
            entries = list(_fallback_parse_xml_items(res.text))

        if not entries:
            raise SelectorBrokenError(
                f"[{self.source_key}] RSS/RDF feed {self.feed_url!r} returned 0 parseable entries"
            )

        self.last_raw_entry_count = len(entries)
        self.last_skipped_entries = 0
        self.last_keyword_filtered_entries = 0
        self.last_duplicate_url_entries = 0
        self.last_earliest_observed_date = None

        items: list[RawIngestItem] = []
        seen_url_index: dict[str, int] = {}
        for entry in entries:
            getter = (
                entry.get if isinstance(entry, dict) else lambda k, e=entry: getattr(e, k, None)
            )
            raw_link = str(getter("link") or "").strip()
            raw_title = str(getter("title") or "").strip()
            if not raw_link or not raw_title:
                self.last_skipped_entries += 1
                continue

            pub_utc = _entry_published_utc(entry, source_tz=self.source_tz)
            if pub_utc is None:
                self.last_skipped_entries += 1
                continue

            item_date = pub_utc[:10]
            if (
                self.last_earliest_observed_date is None
                or item_date < self.last_earliest_observed_date
            ):
                self.last_earliest_observed_date = item_date

            content_parts: list[str] = []
            content_list = getter("content")
            if isinstance(content_list, list):
                for content_item in content_list:
                    if isinstance(content_item, dict):
                        val = str(content_item.get("value") or "").strip()
                        if val:
                            content_parts.append(val)
            for field_name in ("summary", "description"):
                val = str(getter(field_name) or "").strip()
                if val and val not in content_parts:
                    content_parts.append(val)

            raw_summary = "\n\n".join(content_parts).strip()

            tags_text = ""
            tags_list = getter("tags")
            if isinstance(tags_list, list):
                tags_text = " ".join(
                    str(t.get("term") or "") for t in tags_list if isinstance(t, dict)
                )

            raw_cleaned_title = clean_title(raw_title)
            cleaned_summary = strip_html_boilerplate(raw_summary) if raw_summary else ""
            body = cleaned_summary if cleaned_summary else raw_cleaned_title
            title = promote_boilerplate_title(raw_cleaned_title, body)

            if self.keyword_filter:
                haystack = normalize_text(f"{title}\n{body}\n{tags_text}")
                if not any(kw in haystack for kw in self.keyword_filter):
                    self.last_keyword_filtered_entries += 1
                    continue

            resolved_link = urllib.parse.urljoin(self.feed_url, raw_link)
            try:
                canonical_url = normalize_url(resolved_link)
            except Exception:
                self.last_skipped_entries += 1
                continue

            ingest_cat: str | None = None
            if (
                self.tag_corporate_ir
                and self.source_type == "official"
                and is_corporate_ir_text(f"{title}\n{body}")
            ):
                ingest_cat = "corporate"

            candidate = RawIngestItem(
                url=canonical_url,
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

            if canonical_url in seen_url_index:
                self.last_duplicate_url_entries += 1
                existing_idx = seen_url_index[canonical_url]
                if is_fiscal_pr_boilerplate_title(
                    items[existing_idx].title
                ) and not is_fiscal_pr_boilerplate_title(candidate.title):
                    items[existing_idx] = candidate
                continue

            seen_url_index[canonical_url] = len(items)
            items.append(candidate)

        if not items:
            raise SelectorBrokenError(
                f"[{self.source_key}] Feed {self.feed_url!r} yielded 0 valid dated items "
                f"(raw_entries={len(entries)}, skipped={self.last_skipped_entries}, "
                f"keyword_filtered={self.last_keyword_filtered_entries})"
            )

        if self.cache_feed:
            self._cached_items = list(items)
        return items

    def discover(
        self,
        since: date | datetime | str | None = None,
        *,
        use_conditional: bool = True,
    ) -> list[RawIngestItem]:
        """Incremental discovery from the RSS/Atom/RDF feed."""
        since_ts: str | None = None
        if since is not None:
            since_ts = parse_utc_timestamp(since, source_tz=self.source_tz)

        items = self._fetch_all_items(use_conditional=use_conditional)
        if not items:
            return []
        if since_ts is None:
            return items
        return [item for item in items if utc_iso_at_or_after(item.published_at, since_ts)]

    def backfill(
        self,
        date_range: tuple[date | datetime | str | None, date | datetime | str | None] | None = None,
    ) -> list[RawIngestItem]:
        """Historical backfill over `[start_date, end_date]`."""
        start_d, end_d = validate_backfill_window(date_range)
        if not self.supports_backfill:
            raise BackfillNotSupportedError(
                f"[{self.source_key}] Source {self.feed_url!r} is a rolling feed and cannot page history"
            )

        all_items = self._fetch_all_items(use_conditional=False)
        start_iso = start_d.isoformat()
        end_iso = end_d.isoformat()
        window_items = [
            item for item in all_items if start_iso <= item.published_at[:10] <= end_iso
        ]
        if not window_items:
            raise EmptyBackfillWindowError(
                f"[{self.source_key}] 0 RSS items fell inside window {start_iso}..{end_iso}"
            )
        return window_items


__all__ = [
    "PRTIMES_ACGN_KEYWORDS",
    "RssAdapter",
]
