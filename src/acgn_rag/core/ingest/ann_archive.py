"""Anime News Network (ANN) two-mode adapter (`discover` via RSS + `backfill` via per-day HTML archive).

Enforces `docs/SOURCES.md` §3.2 & §7, `docs/COMPLIANCE.md` C1, and Issue #14 Finding V1:
  - Uses slash-form `/news/YYYY/MM/DD` archive URLs directly (avoiding the 301 redirect from
    hyphenated `/news/YYYY-MM-DD`).
  - Filters `<a href>` links strictly by `/news/YYYY-MM-DD/<slug>/.<id>` where the date prefix in
    the article path matches the requested archive day (excluding sidebar "latest" links).
  - Handles multi-page single-day archives with duplicate-page loop detection (`PaginationLoopError`),
    ignoring self-links to `?page=1`.
  - Raises `ValueError` if `backfill` is called without an explicit `(start_date, end_date)` window.

Owner: A (WP1) | Spec: docs/SOURCES.md §3.2 & §7, docs/COMPLIANCE.md C1, Issue #9, Issue #14 V1
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import urllib.parse
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import feedparser
from selectolax.lexbor import LexborHTMLParser as HTMLParser

from acgn_rag.core.ingest.http import PoliteHttpClient, decode_bytes_checked
from acgn_rag.core.ingest.normalize import (
    PARAGRAPH_SEP,
    RawIngestItem,
    clean_title,
    html_to_paragraphs,
    is_corporate_ir_text,
    normalize_text,
    normalize_url,
    parse_utc_timestamp,
    strip_html_boilerplate,
    utc_iso_at_or_after,
)

ANN_BASE_URL = "https://www.animenewsnetwork.com"
ANN_RSS_URL = "https://www.animenewsnetwork.com/all/rss.xml"
ANN_SOURCE_KEY = "ann-news"
ANN_PUBLISHER = "Anime News Network"
ANN_ARCHIVE_TZ = "America/New_York"

# Match ANN article paths: /news/YYYY-MM-DD/<slug>/.<numeric_id> (and /daily-briefs/...)
_ANN_ARTICLE_HREF_RE = re.compile(
    r"^(?:https?://(?:www\.)?animenewsnetwork\.com)?(/(?:news|daily-briefs)/(\d{4}-\d{2}-\d{2})/[a-z0-9\-_.~@+]+/\.\d+)(?:[?#].*)?$",
    re.IGNORECASE,
)
_STRICT_DATE_STR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_PAGE_ONE_RE = re.compile(r"(?:^|[?&])page=1(?:&|$)", re.IGNORECASE)
_ANN_LISTING_TIME_RE = re.compile(r"\(\s*([01]?\d|2[0-3]):([0-5]\d)\s*\)")


class AdapterError(RuntimeError):
    """Base error for source adapter failures (INGEST_CONTRACT.md §5)."""


class SelectorBrokenError(AdapterError):
    """Raised when an HTML/XML/JSON selector matches 0 elements on a non-empty response."""


class PaginationLoopError(AdapterError):
    """Raised when a paged archive returns duplicate page content or repeats a visited page."""


class BackfillNotSupportedError(AdapterError):
    """Raised when `backfill` is called on a rolling-only source that cannot page history."""


class EmptyBackfillWindowError(AdapterError):
    """Raised when a historical backfill over a valid window unexpectedly yields 0 items."""


def _coerce_bound(value: date | datetime | str | None, *, name: str) -> date:
    if value is None:
        raise ValueError(
            f"Backfill requires an explicit ({name}) date bound; windowless crawl is forbidden (COMPLIANCE C1)"
        )
    if isinstance(value, datetime):
        if value.tzinfo is not None and value.utcoffset() not in (None, timedelta(0)):
            raise ValueError(
                f"{name} datetime bound {value!r} must be UTC or a calendar date YYYY-MM-DD (INGEST_CONTRACT §3)"
            )
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(f"Backfill {name} bound must not be empty (COMPLIANCE C1)")
        if _STRICT_DATE_STR_RE.match(cleaned):
            try:
                return date.fromisoformat(cleaned)
            except ValueError as exc:
                raise ValueError(f"Invalid calendar date for {name}: {value!r}") from exc
        # Allow UTC midnight ISO string YYYY-MM-DDT00:00:00Z, reject non-UTC timezone shifts
        if cleaned.endswith("T00:00:00Z") and _STRICT_DATE_STR_RE.match(cleaned[:10]):
            return date.fromisoformat(cleaned[:10])
        raise ValueError(
            f"Invalid {name} bound {value!r}: non-UTC or non-date bound is not allowed; "
            f"window bounds must be UTC calendar dates YYYY-MM-DD (INGEST_CONTRACT §3)"
        )
    raise ValueError(f"Unsupported type for {name}: {type(value).__name__}")


def validate_backfill_window(
    date_range: tuple[date | datetime | str | None, date | datetime | str | None] | None,
) -> tuple[date, date]:
    """Validate `(start_date, end_date)` for `backfill(date_range)` (COMPLIANCE C1).

    Raises `ValueError` if `date_range` is None, not a 2-tuple, missing either bound, or has `start > end`.
    """
    if date_range is None or not isinstance(date_range, tuple) or len(date_range) != 2:
        raise ValueError(
            "backfill(date_range) requires a 2-tuple (start_date, end_date); windowless crawl is forbidden (COMPLIANCE C1)"
        )
    start_raw, end_raw = date_range
    start_d = _coerce_bound(start_raw, name="start_date")
    end_d = _coerce_bound(end_raw, name="end_date")
    if start_d > end_d:
        raise ValueError(
            f"Invalid backfill window: start_date ({start_d.isoformat()}) must not exceed end_date ({end_d.isoformat()})"
        )
    return start_d, end_d


def ann_day_url(target_date: date, *, base_url: str = ANN_BASE_URL, page: int = 1) -> str:
    """Build the canonical slash-form ANN day archive URL `/news/YYYY/MM/DD` (Issue #14 V1)."""
    path = f"/news/{target_date.year:04d}/{target_date.month:02d}/{target_date.day:02d}"
    url = f"{base_url.rstrip('/')}{path}"
    if page > 1:
        return f"{url}?page={page}"
    return url


@dataclass(frozen=True)
class AnnDayParseResult:
    """Result of parsing a single ANN day archive HTML page."""

    target_date: date
    items: list[RawIngestItem]
    dropped_other_day_links: int
    next_page_urls: list[str]
    fingerprint: str
    has_archive_container: bool = True


def parse_ann_day_html(
    html_text: str,
    target_date: date,
    *,
    base_url: str = ANN_BASE_URL,
    source_key: str = ANN_SOURCE_KEY,
    publisher: str = ANN_PUBLISHER,
) -> AnnDayParseResult:
    """Parse an ANN `/news/YYYY/MM/DD` HTML document and extract same-day `RawIngestItem`s.

    Only links matching `/news/YYYY-MM-DD/<slug>/.<id>` where `YYYY-MM-DD == target_date`
    are kept; sidebar links from other dates are counted in `dropped_other_day_links` and
    discarded (Issue #14 Finding V1).
    """
    if not html_text or not html_text.strip():
        raise SelectorBrokenError(
            f"[{source_key}] Empty HTML body on ANN day page for {target_date.isoformat()}"
        )

    tree = HTMLParser(html_text)
    anchors = tree.css("a[href]")
    if not anchors:
        raise SelectorBrokenError(
            f"[{source_key}] No <a href> elements found on ANN day page for {target_date.isoformat()}"
        )

    has_archive_container = bool(
        tree.css("#content-zone, ul.article-list, div.herald.box, div.mainfeed-day")
    )

    target_iso = target_date.isoformat()
    target_slash = target_date.strftime("%Y/%m/%d")
    default_published_at = f"{target_iso}T00:00:00Z"

    # Pre-index any herald card blurbs keyed by canonical_path so articles with cards get full summaries
    herald_blurbs: dict[str, str] = {}
    for box in tree.css("div.herald.box"):
        box_links = [
            _ANN_ARTICLE_HREF_RE.match((a.attributes.get("href") or "").strip())
            for a in box.css("a[href]")
        ]
        valid_paths = {m.group(1) for m in box_links if m is not None and m.group(2) == target_iso}
        if len(valid_paths) == 1:
            only_path = next(iter(valid_paths))
            box_txt = normalize_text(box.text(separator=" ", strip=True) or "").strip()
            box_txt = re.sub(r"\s+", " ", box_txt)
            if len(box_txt) > 40:
                herald_blurbs[only_path] = box_txt

    seen_paths: dict[str, tuple[str, str, str]] = {}
    dropped_other_day = 0
    next_pages: list[str] = []

    for node in anchors:
        href = (node.attributes.get("href") or "").strip()
        if not href:
            continue

        # Check for same-day pagination markers (e.g. /news/YYYY/MM/DD?page=2 or rel="next"),
        # excluding self-links to ?page=1
        rel_attr = (node.attributes.get("rel") or "").lower()
        if not _PAGE_ONE_RE.search(href) and (
            (f"/news/{target_slash}" in href and "page=" in href)
            or (
                "next" in rel_attr
                and (f"/news/{target_slash}" in href or href.startswith("?page="))
            )
        ):
            full_next = urllib.parse.urljoin(f"{base_url.rstrip('/')}/news/{target_slash}", href)
            if not _PAGE_ONE_RE.search(full_next) and full_next not in next_pages:
                next_pages.append(full_next)

        match = _ANN_ARTICLE_HREF_RE.match(href)
        if not match:
            continue

        canonical_path, link_date = match.group(1), match.group(2)
        if link_date != target_iso:
            dropped_other_day += 1
            continue

        raw_title = normalize_text(node.text(deep=True) or "").strip()
        raw_title = re.sub(r"\s+", " ", raw_title)
        if len(raw_title) < 3:
            continue

        # Extract optional listing time `( HH:MM )` from the parent `<li>` (in America/New_York)
        item_pub_at = default_published_at
        if node.parent is not None:
            parent_text = normalize_text(node.parent.text(separator=" ", strip=True) or "").strip()
            time_match = _ANN_LISTING_TIME_RE.search(parent_text)
            if time_match is not None:
                hh, mm = int(time_match.group(1)), int(time_match.group(2))
                item_pub_at = parse_utc_timestamp(
                    f"{target_iso}T{hh:02d}:{mm:02d}:00",
                    source_tz=ANN_ARCHIVE_TZ,
                )

        # Only use genuine herald card blurbs; never synthesize text from the `( HH:MM )` listing row
        snippet = herald_blurbs.get(canonical_path, "")

        prev = seen_paths.get(canonical_path)
        if prev is None or len(raw_title) > len(prev[0]):
            seen_paths[canonical_path] = (
                raw_title,
                snippet,
                prev[2] if (prev and prev[2] != default_published_at) else item_pub_at,
            )

    items: list[RawIngestItem] = []
    for path in sorted(seen_paths):
        raw_title, snippet, item_pub_at = seen_paths[path]
        title = clean_title(raw_title)
        cleaned_snippet = strip_html_boilerplate(snippet) if snippet else ""
        if cleaned_snippet and cleaned_snippet != title and len(cleaned_snippet) > len(title) + 30:
            body = (
                f"{title}{PARAGRAPH_SEP}{cleaned_snippet}"
                if title not in cleaned_snippet
                else cleaned_snippet
            )
        else:
            body = title
        full_url = normalize_url(f"{base_url.rstrip('/')}{path}")
        items.append(
            RawIngestItem(
                url=full_url,
                title=title,
                body=body,
                published_at=item_pub_at,
                publisher=publisher,
                source_type="media",
                source_key=source_key,
                lang="en",
                source_tz=ANN_ARCHIVE_TZ,
                ingest_category=None,
            )
        )

    if not items and dropped_other_day == 0 and not has_archive_container:
        raise SelectorBrokenError(
            f"[{source_key}] ANN day page for {target_date.isoformat()} matched 0 article links and lacks archive DOM"
        )

    fp_input = "\n".join(item.url for item in items).encode("utf-8")
    fingerprint = hashlib.sha256(fp_input).hexdigest()

    return AnnDayParseResult(
        target_date=target_date,
        items=items,
        dropped_other_day_links=dropped_other_day,
        next_page_urls=next_pages,
        fingerprint=fingerprint,
        has_archive_container=has_archive_container,
    )


def _extract_block_paragraphs(body_node: object) -> list[str]:
    """Extract block-level paragraphs from a `selectolax` node while keeping inline markup
    (`<em>`, `<i>`, `<b>`, `<strong>`, `<a>`, `<cite>`, `<span>`, etc.) inline within its
    enclosing paragraph and `<br>` as a soft line break (`\\n`) within its paragraph.
    """
    for bad in body_node.css("script, style, noscript, instaread-player, iframe"):  # type: ignore[attr-defined]
        bad.decompose()
    raw_html = getattr(body_node, "html", None) or ""
    return html_to_paragraphs(raw_html)


@dataclass(frozen=True)
class AnnArticleParseResult:
    """Result of parsing an individual ANN article HTML page (`div.KonaBody`)."""

    title: str
    body: str
    published_at: str | None
    paragraph_count: int = 0


def parse_ann_article_html(
    html_text: str,
    *,
    url: str = "",
    source_key: str = ANN_SOURCE_KEY,
) -> AnnArticleParseResult:
    """Parse a full ANN article page (`#page_header`, `#page-title small time[datetime]`,
    `div.KonaBody`) and return `(title, body, published_at, paragraph_count)`.

    Raises `SelectorBrokenError` if the HTML is empty or `div.KonaBody` / `div.meat` yields an
    empty article body (never falls back to the title on an article page).
    """
    if not html_text or not html_text.strip():
        raise SelectorBrokenError(f"[{source_key}] Empty HTML body on ANN article page {url!r}")

    tree = HTMLParser(html_text)

    raw_title = ""
    h1 = tree.css_first("#page_header")
    if h1 is not None:
        for sub in h1.css(".sub-title, small"):
            sub.decompose()
        raw_title = normalize_text(h1.text(separator=" ", strip=True) or "").strip()
        raw_title = re.sub(r"\s+", " ", raw_title)
    if not raw_title:
        og = tree.css_first('meta[property="og:title"]')
        if og is not None:
            raw_title = normalize_text(og.attributes.get("content") or "").strip()
            raw_title = re.sub(r"\s+", " ", raw_title)

    title = clean_title(raw_title) if raw_title else ""

    pub_utc: str | None = None
    time_node = tree.css_first("#page-title small time[datetime]")
    if time_node is not None:
        dt_attr = (time_node.attributes.get("datetime") or "").strip()
        if dt_attr:
            try:
                pub_utc = parse_utc_timestamp(dt_attr, source_tz=ANN_ARCHIVE_TZ)
            except ValueError:
                pub_utc = None

    body_node = tree.css_first("div.KonaBody") or tree.css_first("div.meat")
    if body_node is None:
        raise SelectorBrokenError(
            f"[{source_key}] Missing div.KonaBody on ANN article page {url!r}"
        )
    paragraphs = _extract_block_paragraphs(body_node)
    cleaned_body = "\n\n".join(paragraphs).strip()
    if not cleaned_body:
        raise SelectorBrokenError(
            f"[{source_key}] Empty div.KonaBody text on ANN article page {url!r}"
        )

    return AnnArticleParseResult(
        title=title,
        body=cleaned_body,
        published_at=pub_utc,
        paragraph_count=len(paragraphs),
    )


def _iter_selected_days(start_date: date, end_date: date, max_days: int | None) -> list[date]:
    """Return every calendar day in `[start_date, end_date]` (newest first) when `max_days` is None,
    or a stratified sample of `max_days` when explicitly requested.
    """
    total_days = (end_date - start_date).days + 1
    if max_days is None or total_days <= max_days:
        return [end_date - timedelta(days=offset) for offset in range(total_days)]
    if max_days <= 1:
        return [end_date]
    selected: list[date] = []
    for idx in range(max_days):
        offset = round(idx * (total_days - 1) / (max_days - 1))
        d = end_date - timedelta(days=offset)
        if d not in selected:
            selected.append(d)
    return selected


@dataclass
class AnnArchiveAdapter:
    """Two-mode adapter for Anime News Network:
    - `discover(since)` uses the rolling RSS feed (`all/rss.xml`).
    - `backfill(date_range)` crawls the slash-form per-day archive `/news/YYYY/MM/DD`.
    """

    http: PoliteHttpClient = field(default_factory=PoliteHttpClient)
    source_key: str = ANN_SOURCE_KEY
    publisher: str = ANN_PUBLISHER
    rss_url: str = ANN_RSS_URL
    base_url: str = ANN_BASE_URL
    max_pages_per_day: int = 5
    max_days_per_call: int | None = None
    cache_dir: Path | None = None
    article_cache_path: Path | None = None
    cache_recent_days: bool = False
    empty_days_count: int = field(init=False, default=0)
    dropped_other_day_links: int = field(init=False, default=0)
    last_requested_days: int = field(init=False, default=0)
    last_sampled_days: int = field(init=False, default=0)
    last_days_with_items: int = field(init=False, default=0)
    last_empty_days: int = field(init=False, default=0)
    last_total_pages_fetched: int = field(init=False, default=0)
    last_truncated_days: int = field(init=False, default=0)
    last_cached_article_bodies_applied: int = field(init=False, default=0)
    last_raw_entry_count: int = field(init=False, default=0)
    last_skipped_entries: int = field(init=False, default=0)
    last_skip_reasons: dict[str, int] = field(init=False, default_factory=dict)
    last_earliest_observed_date: str | None = field(init=False, default=None)
    last_not_modified: bool = field(init=False, default=False)
    _day_cache: dict[date, list[RawIngestItem]] = field(
        init=False, default_factory=dict, repr=False
    )
    _article_cache_loaded: bool = field(init=False, default=False, repr=False)
    _article_by_url: dict[str, tuple[str, str, str, str | None]] = field(
        init=False, default_factory=dict, repr=False
    )
    _article_by_id: dict[str, tuple[str, str, str, str | None]] = field(
        init=False, default_factory=dict, repr=False
    )
    _articles_by_date: dict[str, list[tuple[str, str, str, str | None]]] = field(
        init=False, default_factory=dict, repr=False
    )

    @staticmethod
    def _extract_article_id(url: str) -> str:
        m = re.search(r"/\.(\d+)$", url)
        return m.group(1) if m is not None else ""

    def _ensure_article_cache_loaded(self) -> None:
        if self._article_cache_loaded:
            return
        self._article_cache_loaded = True
        if self.article_cache_path is None or not self.article_cache_path.exists():
            return
        url_date_re = re.compile(r"/(\d{4}-\d{2}-\d{2})/")
        path = self.article_cache_path
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            articles_map = payload.get("articles", {}) if isinstance(payload, dict) else {}
            if isinstance(articles_map, dict):
                for raw_u, entry in sorted(articles_map.items()):
                    if not isinstance(entry, dict):
                        continue
                    norm_u = normalize_url(str(raw_u))
                    t = str(entry.get("title") or "").strip()
                    b = str(entry.get("body") or "").strip()
                    d = str(entry.get("date") or "").strip()
                    pub = str(entry.get("published_at") or "").strip() or None
                    if not d:
                        m = url_date_re.search(norm_u)
                        d = m.group(1) if m else ""
                    if norm_u and b:
                        self._article_by_url[norm_u] = (t, b, d, pub)
                        aid = self._extract_article_id(norm_u)
                        if aid:
                            self._article_by_id[aid] = (t, b, d, pub)
                        if d:
                            self._articles_by_date.setdefault(d, []).append((norm_u, t, b, pub))
            return

        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as zf:
                csv_text = zf.read("articles_index.csv").decode("utf-8")
                rows = list(csv.DictReader(io.StringIO(csv_text)))
                for r in rows:
                    raw_u = str(r.get("url") or "").strip()
                    if not raw_u:
                        continue
                    norm_u = normalize_url(raw_u)
                    cache_basename = (
                        str(r.get("cache_file") or "").replace("\\", "/").split("/")[-1]
                    )
                    html_bytes = zf.read(f"html/{cache_basename}")
                    html_text, _, _ = decode_bytes_checked(
                        html_bytes,
                        url=norm_u,
                        content_type="text/html; charset=utf-8",
                    )
                    parsed_art = parse_ann_article_html(
                        html_text,
                        url=norm_u,
                        source_key=self.source_key,
                    )
                    d = str(r.get("published_date") or "").strip()
                    if not d:
                        m = url_date_re.search(norm_u)
                        d = m.group(1) if m else ""
                    self._article_by_url[norm_u] = (
                        parsed_art.title,
                        parsed_art.body,
                        d,
                        parsed_art.published_at,
                    )
                    aid = self._extract_article_id(norm_u)
                    if aid:
                        self._article_by_id[aid] = (
                            parsed_art.title,
                            parsed_art.body,
                            d,
                            parsed_art.published_at,
                        )
                    if d:
                        self._articles_by_date.setdefault(d, []).append(
                            (norm_u, parsed_art.title, parsed_art.body, parsed_art.published_at)
                        )

    def _apply_article_cache_to_day(
        self, target_date: date, items: list[RawIngestItem]
    ) -> list[RawIngestItem]:
        if self.article_cache_path is None or not self.article_cache_path.exists():
            return items
        self._ensure_article_cache_loaded()
        if not self._article_by_url:
            return items

        target_iso = target_date.isoformat()
        enriched: list[RawIngestItem] = []
        seen_urls: set[str] = set()
        seen_ids: set[str] = set()
        for it in items:
            seen_urls.add(it.url)
            aid = self._extract_article_id(it.url)
            if aid:
                seen_ids.add(aid)
            cached = self._article_by_url.get(it.url) or (
                self._article_by_id.get(aid) if aid else None
            )
            if cached is None:
                enriched.append(it)
                continue
            cached_title, cached_body, _, _ = cached
            eff_title = it.title or cached_title
            eff_body = cached_body
            ingest_cat = (
                "corporate"
                if is_corporate_ir_text(f"{eff_title}\n{eff_body}")
                else it.ingest_category
            )
            self.last_cached_article_bodies_applied += 1
            enriched.append(
                RawIngestItem(
                    url=it.url,
                    title=eff_title,
                    body=eff_body,
                    published_at=it.published_at,
                    publisher=it.publisher,
                    source_type=it.source_type,
                    source_key=it.source_key,
                    lang=it.lang,
                    source_tz=it.source_tz,
                    ingest_category=ingest_cat,
                )
            )

        extra_for_day = self._articles_by_date.get(target_iso, [])
        for cached_url, cached_title, cached_body, cached_pub in sorted(
            extra_for_day, key=lambda x: x[0]
        ):
            aid = self._extract_article_id(cached_url)
            if cached_url in seen_urls or (aid and aid in seen_ids):
                continue
            seen_urls.add(cached_url)
            if aid:
                seen_ids.add(aid)
            eff_title = clean_title(cached_title)
            if not eff_title:
                continue
            eff_body = cached_body
            ingest_cat = "corporate" if is_corporate_ir_text(f"{eff_title}\n{eff_body}") else None
            self.last_cached_article_bodies_applied += 1
            enriched.append(
                RawIngestItem(
                    url=cached_url,
                    title=eff_title,
                    body=eff_body,
                    published_at=cached_pub or f"{target_iso}T00:00:00Z",
                    publisher=self.publisher,
                    source_type="media",
                    source_key=self.source_key,
                    lang="en",
                    source_tz=ANN_ARCHIVE_TZ,
                    ingest_category=ingest_cat,
                )
            )

        return enriched

    def discover(
        self,
        since: date | datetime | str | None = None,
        *,
        use_conditional: bool = True,
    ) -> list[RawIngestItem]:
        """Incremental discovery via ANN's RSS feed (`all/rss.xml`)."""
        self.last_not_modified = False
        since_ts: str | None = None
        if since is not None:
            since_ts = parse_utc_timestamp(since, source_tz="UTC")

        self.last_raw_entry_count = 0
        self.last_skipped_entries = 0
        self.last_skip_reasons = {}

        res = self.http.get(self.rss_url, use_conditional=use_conditional)
        if res.not_modified:
            self.last_not_modified = True
            return []

        parsed = feedparser.parse(res.text, sanitize_html=False, resolve_relative_uris=False)
        entries = list(getattr(parsed, "entries", []) or [])
        if not entries:
            raise SelectorBrokenError(
                f"[{self.source_key}] ANN RSS feed at {self.rss_url!r} returned 0 entries"
            )
        self.last_raw_entry_count = len(entries)

        out: list[RawIngestItem] = []
        for entry in entries:
            link = str(entry.get("link") or "").strip()
            raw_title = str(entry.get("title") or "").strip()
            raw_summary = str(entry.get("summary") or entry.get("description") or raw_title).strip()
            raw_pub = entry.get("published") or entry.get("updated") or entry.get("pubDate")
            if not link or not raw_title or not raw_pub:
                self.last_skipped_entries += 1
                self.last_skip_reasons["missing_required_field"] = (
                    self.last_skip_reasons.get("missing_required_field", 0) + 1
                )
                continue

            try:
                pub_utc = parse_utc_timestamp(str(raw_pub), source_tz="UTC")
            except ValueError:
                self.last_skipped_entries += 1
                self.last_skip_reasons["malformed_published_at"] = (
                    self.last_skip_reasons.get("malformed_published_at", 0) + 1
                )
                continue

            if since_ts is not None and not utc_iso_at_or_after(pub_utc, since_ts):
                continue

            try:
                title = clean_title(raw_title)
                canonical_url = normalize_url(link)
            except ValueError:
                self.last_skipped_entries += 1
                self.last_skip_reasons["invalid_url_or_title"] = (
                    self.last_skip_reasons.get("invalid_url_or_title", 0) + 1
                )
                continue

            summary_clean = strip_html_boilerplate(raw_summary) or title
            body = summary_clean if summary_clean and summary_clean != title else title
            ingest_cat = "corporate" if is_corporate_ir_text(f"{title}\n{body}") else None
            out.append(
                RawIngestItem(
                    url=canonical_url,
                    title=title,
                    body=body,
                    published_at=pub_utc,
                    publisher=self.publisher,
                    source_type="media",
                    source_key=self.source_key,
                    lang="en",
                    source_tz="UTC",
                    ingest_category=ingest_cat,
                )
            )
        if not out and self.last_skipped_entries == len(entries):
            raise SelectorBrokenError(
                f"[{self.source_key}] ANN RSS feed at {self.rss_url!r} yielded 0 valid entries "
                f"out of {len(entries)} (skipped={self.last_skipped_entries}, reasons={self.last_skip_reasons})"
            )
        return out

    def _load_disk_cached_day(self, target_date: date) -> list[RawIngestItem] | None:
        if self.cache_dir is None:
            return None
        cache_file = self.cache_dir / f"{target_date.isoformat()}.json"
        if not cache_file.exists():
            return None
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("v") != 3:
                return None
            raw_rows = payload.get("items")
            if not isinstance(raw_rows, list):
                return None
            return [
                RawIngestItem(
                    url=str(r["url"]),
                    title=str(r["title"]),
                    body=str(r["body"]),
                    published_at=str(r["published_at"]),
                    publisher=str(r["publisher"]),
                    source_type=r["source_type"],
                    source_key=str(r["source_key"]),
                    lang=str(r.get("lang", "en")),
                    source_tz=r.get("source_tz"),
                    ingest_category=r.get("ingest_category"),
                )
                for r in raw_rows
                if isinstance(r, dict)
            ]
        except Exception:
            return None

    def _save_disk_cached_day(self, target_date: date, items: Sequence[RawIngestItem]) -> None:
        if self.cache_dir is None:
            return
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / f"{target_date.isoformat()}.json"
            payload = {
                "v": 3,
                "date": target_date.isoformat(),
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

    def fetch_day(self, target_date: date) -> list[RawIngestItem]:
        """Fetch and parse one ANN day archive page (plus any verified pagination pages)."""
        if target_date in self._day_cache:
            cached_mem = list(self._day_cache[target_date])
            self.last_total_pages_fetched += 1
            return cached_mem

        today_utc = datetime.now(tz=UTC).date()
        disk_cache_cutoff = today_utc if self.cache_recent_days else (today_utc - timedelta(days=1))
        if target_date < disk_cache_cutoff:
            disk_cached = self._load_disk_cached_day(target_date)
            if disk_cached is not None:
                enriched_disk = self._apply_article_cache_to_day(target_date, disk_cached)
                self._day_cache[target_date] = enriched_disk
                self.last_total_pages_fetched += 1
                return list(enriched_disk)

        first_url = ann_day_url(target_date, base_url=self.base_url, page=1)
        res = self.http.get(first_url, use_conditional=False)
        self.last_total_pages_fetched += 1
        page1 = parse_ann_day_html(
            res.text,
            target_date,
            base_url=self.base_url,
            source_key=self.source_key,
            publisher=self.publisher,
        )
        self.dropped_other_day_links += page1.dropped_other_day_links

        # If a past historical day returns 0 items and 0 sidebar links, treat as selector failure
        if not page1.items and target_date < today_utc and not page1.has_archive_container:
            raise SelectorBrokenError(
                f"[{self.source_key}] ANN archive day {target_date.isoformat()} returned 0 items"
            )
        if not page1.items:
            self.empty_days_count += 1

        collected: dict[str, RawIngestItem] = {item.url: item for item in page1.items}
        seen_fingerprints: set[str] = {page1.fingerprint}

        if len(page1.next_page_urls) > max(0, self.max_pages_per_day - 1):
            self.last_truncated_days += 1

        queue: Sequence[str] = page1.next_page_urls[: self.max_pages_per_day - 1]
        for next_url in queue:
            next_res = self.http.get(next_url, use_conditional=False)
            self.last_total_pages_fetched += 1
            parsed_next = parse_ann_day_html(
                next_res.text,
                target_date,
                base_url=self.base_url,
                source_key=self.source_key,
                publisher=self.publisher,
            )
            self.dropped_other_day_links += parsed_next.dropped_other_day_links
            if not parsed_next.items or parsed_next.fingerprint in seen_fingerprints:
                raise PaginationLoopError(
                    f"[{self.source_key}] Duplicate or empty pagination page on {next_url!r}"
                )
            seen_fingerprints.add(parsed_next.fingerprint)
            for item in parsed_next.items:
                collected.setdefault(item.url, item)

        raw_day_items = list(collected.values())
        if target_date < disk_cache_cutoff:
            self._save_disk_cached_day(target_date, raw_day_items)
        day_items = self._apply_article_cache_to_day(target_date, raw_day_items)
        self._day_cache[target_date] = day_items
        return list(day_items)

    def backfill(
        self,
        date_range: tuple[date | datetime | str | None, date | datetime | str | None] | None = None,
    ) -> list[RawIngestItem]:
        """Historical backfill across `[start_date, end_date]` via `/news/YYYY/MM/DD`."""
        start_d, end_d = validate_backfill_window(date_range)
        days = _iter_selected_days(start_d, end_d, self.max_days_per_call)
        self.last_requested_days = (end_d - start_d).days + 1
        self.last_sampled_days = len(days)
        self.last_days_with_items = 0
        self.last_empty_days = 0
        self.last_total_pages_fetched = 0
        self.last_truncated_days = 0
        self.last_cached_article_bodies_applied = 0
        self.last_earliest_observed_date = None

        collected: dict[str, RawIngestItem] = {}
        total_days = len(days)
        for idx, day in enumerate(days, start=1):
            day_items = self.fetch_day(day)
            if day_items:
                self.last_days_with_items += 1
            else:
                self.last_empty_days += 1
            for item in day_items:
                item_d = item.published_at[:10]
                if (
                    self.last_earliest_observed_date is None
                    or item_d < self.last_earliest_observed_date
                ):
                    self.last_earliest_observed_date = item_d
                collected.setdefault(item.url, item)
            if total_days > 100 and (idx % 50 == 0 or idx == total_days):
                import sys

                print(
                    f"[ann-news] progress: {idx}/{total_days} days ({day.isoformat()}), "
                    f"accumulated={len(collected)} unique items",
                    file=sys.stderr,
                    flush=True,
                )

        if not collected:
            raise EmptyBackfillWindowError(
                f"[{self.source_key}] ANN archive backfill returned 0 items in window "
                f"{start_d.isoformat()}..{end_d.isoformat()}"
            )
        return list(collected.values())


__all__ = [
    "ANN_BASE_URL",
    "ANN_PUBLISHER",
    "ANN_RSS_URL",
    "ANN_SOURCE_KEY",
    "AdapterError",
    "AnnArchiveAdapter",
    "AnnArticleParseResult",
    "AnnDayParseResult",
    "BackfillNotSupportedError",
    "EmptyBackfillWindowError",
    "PaginationLoopError",
    "SelectorBrokenError",
    "ann_day_url",
    "parse_ann_article_html",
    "parse_ann_day_html",
    "validate_backfill_window",
]
