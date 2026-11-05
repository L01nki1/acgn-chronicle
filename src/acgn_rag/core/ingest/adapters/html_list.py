"""HTML list-page adapter (`HtmlListAdapter`) for studio/publisher news lists (Toei Animation,
P.A.WORKS, BONES) with content-fingerprint pagination loop detection.

Owner: A (WP1) | Spec: docs/SOURCES.md §3 & §7, Issue #9
"""

from __future__ import annotations

import hashlib
import re
import urllib.parse
from dataclasses import dataclass, field
from datetime import date, datetime

from selectolax.lexbor import LexborHTMLParser as HTMLParser
from selectolax.lexbor import LexborNode as Node

from acgn_rag.core.ingest.ann_archive import (
    BackfillNotSupportedError,
    EmptyBackfillWindowError,
    PaginationLoopError,
    SelectorBrokenError,
    validate_backfill_window,
)
from acgn_rag.core.ingest.http import PoliteHttpClient
from acgn_rag.core.ingest.normalize import (
    RawIngestItem,
    clean_title,
    is_corporate_ir_text,
    normalize_text,
    normalize_url,
    parse_utc_timestamp,
    strip_html_boilerplate,
    utc_iso_at_or_after,
)
from acgn_rag.core.ports import SourceType

_EMBEDDED_DATE_RE = re.compile(
    r"(?<!\d)(20\d{2})\s*(?:[-./年])\s*(\d{1,2})\s*(?:[-./月])\s*(\d{1,2})\s*日?(?!\d)"
)


def _extract_date_from_text(text: str) -> str | None:
    match = _EMBEDDED_DATE_RE.search(text)
    if not match:
        return None
    y, m, d = int(match.group(1)), int(match.group(2)), int(match.group(3))
    try:
        return date(y, m, d).isoformat()
    except ValueError:
        return None


def _extract_date_from_url(url: str) -> str | None:
    match = re.search(r"/(20\d{2})[-/]?(\d{2})[-/]?(\d{2})(?:[/_.-]|$)", url)
    if not match:
        return None
    y, m, d = int(match.group(1)), int(match.group(2)), int(match.group(3))
    try:
        return date(y, m, d).isoformat()
    except ValueError:
        return None


def _clean_entry_title(raw_text: str, date_str: str) -> str:
    """Remove leading date tokens and category badges from a list item's text."""
    cleaned = normalize_text(raw_text)
    cleaned = _EMBEDDED_DATE_RE.sub(" ", cleaned, count=1)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -|\uff5c:\uff1a/\u30fb")
    return cleaned or raw_text.strip()


def parse_html_list_page(
    html_text: str,
    *,
    base_url: str,
    source_key: str,
    publisher: str,
    source_type: SourceType,
    lang: str,
    source_tz: str | None = "Asia/Tokyo",
    item_selector: str = "li, article, tr, dl, .news-item, .c-news__item, .p-news__item",
    link_pattern: str | None = None,
) -> tuple[list[RawIngestItem], str]:
    """Parse an HTML list page into `(items, page_fingerprint)`."""
    if not html_text or not html_text.strip():
        raise SelectorBrokenError(f"[{source_key}] Empty HTML response from {base_url!r}")

    tree = HTMLParser(html_text)
    for tag in tree.css("script, style, nav, footer, header, noscript"):
        tag.decompose()

    link_regex = re.compile(link_pattern, re.I) if link_pattern else None
    candidates: list[Node] = list(tree.css(item_selector))
    if not candidates:
        candidates = list(tree.css("a[href]"))

    items_by_url: dict[str, RawIngestItem] = {}

    def _consider_anchor(anchor: Node, container_text: str) -> None:
        href = (anchor.attributes.get("href") or "").strip()
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            return
        full_url = urllib.parse.urljoin(base_url, href)
        if link_regex and not link_regex.search(full_url):
            return

        anchor_text = normalize_text(anchor.text(separator=" ", strip=True) or "").strip()
        combined_text = normalize_text(container_text or anchor_text).strip()
        date_iso = (
            _extract_date_from_text(anchor_text)
            or _extract_date_from_text(combined_text)
            or _extract_date_from_url(full_url)
        )
        if not date_iso:
            return

        raw_title = _clean_entry_title(anchor_text or combined_text, date_iso)
        if len(raw_title) < 2:
            raw_title = _clean_entry_title(combined_text, date_iso)
        if len(raw_title) < 2:
            return

        try:
            canonical = normalize_url(full_url)
            title = clean_title(raw_title)
        except Exception:
            return

        cleaned_container = (
            strip_html_boilerplate(_clean_entry_title(combined_text, date_iso))
            if combined_text
            else ""
        )
        if (
            cleaned_container
            and cleaned_container != title
            and len(cleaned_container) > len(title) + 20
        ):
            body = cleaned_container
        else:
            body = title
        pub_utc = f"{date_iso}T00:00:00Z"
        ingest_cat = (
            "corporate"
            if source_type == "official" and is_corporate_ir_text(f"{title}\n{body}")
            else None
        )
        items_by_url.setdefault(
            canonical,
            RawIngestItem(
                url=canonical,
                title=title,
                body=body,
                published_at=pub_utc,
                publisher=publisher,
                source_type=source_type,
                source_key=source_key,
                lang=lang,
                source_tz=source_tz,
                ingest_category=ingest_cat,
            ),
        )

    for node in candidates:
        if node.tag == "a":
            parent_text = node.parent.text(separator=" ", strip=True) if node.parent else ""
            _consider_anchor(node, parent_text)
        else:
            anchors = list(node.css("a[href]"))
            if len(anchors) > 3:
                # Wrapper container (e.g. outer <article> or <dl> wrapping the entire list):
                # do not smear the first item's date across all child anchors.
                continue
            node_text = node.text(separator=" ", strip=True) or ""
            for anchor in anchors:
                _consider_anchor(anchor, node_text)

    if not items_by_url:
        # Fallback: scan all anchors on page looking for dated parent/sibling blocks (max 3 links)
        for anchor in tree.css("a[href]"):
            parent = anchor.parent
            context = ""
            for _ in range(3):
                if parent is None:
                    break
                if len(parent.css("a[href]")) > 3:
                    break
                txt = parent.text(separator=" ", strip=True) or ""
                if not _extract_date_from_text(txt) and parent.prev is not None:
                    prev_txt = parent.prev.text(separator=" ", strip=True) or ""
                    txt = f"{prev_txt} {txt}".strip()
                if _extract_date_from_text(txt):
                    context = txt
                    break
                parent = parent.parent
            _consider_anchor(anchor, context)

    items = list(items_by_url.values())
    if not items:
        raise SelectorBrokenError(
            f"[{source_key}] HTML list selector matched 0 dated items on {base_url!r}"
        )

    fp_bytes = "\n".join(sorted(item.url for item in items)).encode("utf-8")
    fingerprint = hashlib.sha256(fp_bytes).hexdigest()
    return items, fingerprint


@dataclass
class HtmlListAdapter:
    """Two-mode HTML list adapter with content-fingerprint pagination loop detection."""

    source_key: str
    list_url: str
    publisher: str
    source_type: SourceType
    lang: str = "ja"
    source_tz: str | None = "Asia/Tokyo"
    http: PoliteHttpClient = field(default_factory=PoliteHttpClient)
    item_selector: str = "li, article, tr, dl, .news-item, .c-news__item, .p-news__item"
    link_pattern: str | None = None
    max_pages: int = 1
    page_url_template: str | None = None
    raise_on_duplicate_page: bool = False
    supports_backfill: bool = True
    cache_pages: bool = True
    last_not_modified: bool = field(init=False, default=False, repr=False)
    _cached_items: list[RawIngestItem] | None = field(init=False, default=None, repr=False)

    def _page_url(self, page: int) -> str:
        if page <= 1:
            return self.list_url
        if self.page_url_template:
            return self.page_url_template.format(page=page)
        sep = "&" if "?" in self.list_url else "?"
        return f"{self.list_url}{sep}page={page}"

    def _crawl_pages(
        self, *, use_conditional_first_page: bool, max_pages: int
    ) -> list[RawIngestItem]:
        self.last_not_modified = False
        if self.cache_pages and not use_conditional_first_page and self._cached_items is not None:
            return list(self._cached_items)

        collected: dict[str, RawIngestItem] = {}
        seen_fingerprints: set[str] = set()

        for page in range(1, max_pages + 1):
            url = self._page_url(page)
            res = self.http.get(
                url,
                use_conditional=(use_conditional_first_page and page == 1),
            )
            if page == 1 and res.not_modified:
                self.last_not_modified = True
                return []

            page_items, fp = parse_html_list_page(
                res.text,
                base_url=url,
                source_key=self.source_key,
                publisher=self.publisher,
                source_type=self.source_type,
                lang=self.lang,
                source_tz=self.source_tz,
                item_selector=self.item_selector,
                link_pattern=self.link_pattern,
            )

            if fp in seen_fingerprints:
                if self.raise_on_duplicate_page:
                    raise PaginationLoopError(
                        f"[{self.source_key}] Page {page} ({url!r}) returned duplicate content fingerprint {fp[:12]}"
                    )
                break

            seen_fingerprints.add(fp)
            new_on_page = 0
            for item in page_items:
                if item.url not in collected:
                    collected[item.url] = item
                    new_on_page += 1

            if page > 1 and new_on_page == 0:
                if self.raise_on_duplicate_page:
                    raise PaginationLoopError(
                        f"[{self.source_key}] Page {page} ({url!r}) added 0 new URLs"
                    )
                break

        items = list(collected.values())
        if self.cache_pages and max_pages >= self.max_pages and not use_conditional_first_page:
            self._cached_items = list(items)
        return items

    def discover(
        self,
        since: date | datetime | str | None = None,
        *,
        use_conditional: bool = True,
    ) -> list[RawIngestItem]:
        """Incremental discovery from page 1 of the HTML news list."""
        since_ts: str | None = None
        if since is not None:
            since_ts = parse_utc_timestamp(since, source_tz=self.source_tz)

        items = self._crawl_pages(use_conditional_first_page=use_conditional, max_pages=1)
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
        if not self.supports_backfill:
            raise BackfillNotSupportedError(
                f"[{self.source_key}] HTML source {self.list_url!r} does not support historical backfill"
            )

        all_items = self._crawl_pages(
            use_conditional_first_page=False,
            max_pages=self.max_pages,
        )
        start_iso = start_d.isoformat()
        end_iso = end_d.isoformat()
        window_items = [
            item for item in all_items if start_iso <= item.published_at[:10] <= end_iso
        ]
        if not window_items:
            raise EmptyBackfillWindowError(
                f"[{self.source_key}] 0 HTML list items fell inside window {start_iso}..{end_iso}"
            )
        return window_items


__all__ = [
    "HtmlListAdapter",
    "parse_html_list_page",
]
