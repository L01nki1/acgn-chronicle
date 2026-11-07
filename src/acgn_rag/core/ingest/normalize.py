"""Normalization of raw ingested articles into canonical NormalizedItem records.

Enforces NFC + LF text normalization, contract-exact URL canonicalization, deterministic
`doc_` + SHA-256(url) minting, explicit UTC `Z` timestamp conversion (retaining nonzero
fractional seconds without rounding, quarantining unqualified naive datetimes), IR/corporate
pre-filter tagging, and full-field same-URL conflict detection.

Owner: A (WP1) | Spec: docs/INGEST_CONTRACT.md §1 & §3, docs/COMPLIANCE.md C2 & C5
"""

from __future__ import annotations

import email.utils
import hashlib
import html
import json
import re
import unicodedata
import urllib.parse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from typing import Any, Literal
from zoneinfo import ZoneInfo

from acgn_rag.core.ports import SourceType

NORMALIZATION_VERSION = "1"
PARAGRAPH_SEP = "\n\n"
VALID_SOURCE_TYPES: frozenset[str] = frozenset({"official", "media", "metadata"})
KNOWN_LANGS: frozenset[str] = frozenset({"ja", "en", "zh", "und"})

# Same corporate/IR pre-filter rules as WP2 (src/acgn_rag/core/facts/extract.py)
IR_KEYWORDS: tuple[str, ...] = ("決算", "株主", "決算短信", "事業報告")
IR_TOKEN_PATTERN = re.compile(r"(?<![A-Za-z])[Ii][Rr](?![A-Za-z])")

# Event-type keyword heuristics used as internal clustering input (INGEST_CONTRACT.md §1-§2)
_EVENT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "corporate",
        ("決算", "株主", "決算短信", "事業報告", "配当", "業績予想", "financial results"),
    ),
    (
        "collaboration",
        ("コラボ", "collaboration", "crossover", "联动", "聯動", "タイアップ"),
    ),
    (
        "adaptation",
        ("アニメ化", "TVアニメ化", "映画化", "実写化", "anime adaptation", "动画化", "動畫化"),
    ),
    (
        "broadcast",
        (
            "放送開始",
            "放送決定",
            "配信開始",
            "放送日時",
            "premiere",
            "airing",
            "开播",
            "定档",
            "放送情報",
        ),
    ),
    (
        "release",
        (
            "発売",
            "リリース",
            "配信決定",
            "launch",
            "release date",
            "发售",
            "上线",
            "公测",
            "Steam",
        ),
    ),
    (
        "update",
        ("アップデート", "更新", "patch", "ver.", "DLC", "追加コンテンツ", "版本更新"),
    ),
    (
        "event",
        ("イベント", "開催", "出展", "TGS", "AnimeJapan", "コミケ", "展会", "活动"),
    ),
)

_DATE_ONLY_RE = re.compile(
    r"^\s*(\d{4})\s*(?:[-./年])\s*(\d{1,2})\s*(?:[-./月])\s*(\d{1,2})\s*日?\s*$"
)
_EXPLICIT_UTC_RFC_RE = re.compile(r"(?:[+-]00:?00|\b(?:UTC|GMT|UT|Z)\b)\s*$", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[A-Za-z!/][^>]*>")
_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style|nav|footer|noscript|iframe)\b[^>]*>.*?</\1>", re.I | re.S
)
_BLOCK_TAGS = (
    r"p|div|li|tr|h[1-6]|blockquote|pre|section|article|header|footer|table|ul|ol|dl|dt|dd|figure"
)
HTML_BLOCK_BOUNDARY_RE = re.compile(rf"</?(?:{_BLOCK_TAGS})\b[^>]*>|<hr\s*/?>", re.I)
_BR_RE = re.compile(r"<br\s*/?>", re.I)
_CELL_RE = re.compile(r"</?(?:td|th)\b[^>]*>", re.I)
_HSPACE_RE = re.compile(r"[ \t\f\v\u00a0\u3000]+")
_ESCAPED_FROALA_RE = re.compile(r"<span\b[^>]*\bfr-mk\b[^>]*>.*?</span>", re.I | re.S)
_MARKUP_ONLY_RE = re.compile(r"^(?:\s|&nbsp;|<[^>]*>)*$", re.I)
_INDEX_PAGE_URL_RE = re.compile(r"/ja/press\.html$", re.I)
_TITLE_SUFFIX_RE = re.compile(
    r"\s*(?:[|\-\u2013\u2014\uff5c]\s*(?:Anime News Network|東映アニメーション\s*Toei Animation|東映アニメーション|"
    r"株式会社カプコン|CAPCOM|机核\s*GCORES|GCORES|触乐|游研社|MAPPA|BONES|P\.A\.WORKS|"
    r"にじめん|Real Sound\uff5cリアルサウンド\s*ブック|Real Sound\uff5cリアルサウンド\s*テック|Real Sound\uff5cリアルサウンド|"
    r"Real Sound|リアルサウンド))\s*$",
    re.IGNORECASE,
)


class NormalizationError(ValueError):
    """Raised when a raw item fails mandatory normalization or validation rules."""


class TimestampParseError(NormalizationError):
    """Raised when `published_at` is missing, malformed, or lacks a required timezone."""


class ConflictingObservationError(NormalizationError):
    """Raised when two observations of the same canonical URL have conflicting content/metadata."""

    def __init__(self, doc_id: str, url: str, detail: str) -> None:
        super().__init__(f"Conflicting observations for {doc_id} ({url!r}): {detail}")
        self.doc_id = doc_id
        self.url = url


def canonical_json(value: Any) -> bytes:
    """Serialize `value` to canonical UTF-8 JSON bytes followed by a single LF (INGEST_CONTRACT §3)."""
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def normalize_text(text: str) -> str:
    """Apply Unicode NFC normalization and convert CRLF/CR line endings to LF."""
    if not isinstance(text, str):
        raise NormalizationError(f"Expected string text, got {type(text).__name__}")
    nfc = unicodedata.normalize("NFC", text)
    return nfc.replace("\r\n", "\n").replace("\r", "\n")


def html_to_paragraphs(fragment: str) -> list[str]:
    """Convert an HTML fragment (or plain-text fallback) into block-level paragraphs.

    - Block boundaries (`<p>`, `<div>`, `<li>`, `<tr>`, `<h1-6>`, `<blockquote>`, `<pre>`,
      `<section>`, `<article>`, `<header>`, `<footer>`, `<table>`, `<ul>`, `<ol>`, `<dl>`, `<dt>`,
      `<dd>`, `<figure>`, `<hr>`) separate paragraphs.
    - `<br>` becomes a soft line break `'\\n'` inside its enclosing paragraph (not a paragraph
      boundary).
    - When HTML tags are present, source newlines inside a block are collapsed to spaces (per HTML
      whitespace rules), while `<br>` soft line breaks are preserved as `'\\n'`.
    - When no HTML tags are present, blank lines (`\\n\\s*\\n`) split paragraphs and single `'\\n'`
      is preserved as a soft line break.
    - Escaped Froala marker spans (`fr-mk`) and markup-only/whitespace-only paragraphs are removed.
    """
    if not fragment or not fragment.strip():
        return []
    has_html = bool(_HTML_TAG_RE.search(fragment))
    s = _SCRIPT_STYLE_RE.sub(" ", fragment)
    s = _CELL_RE.sub(" ", s)
    if has_html:
        s = _BR_RE.sub("\x01", s)
        s = HTML_BLOCK_BOUNDARY_RE.sub("\x00", s)
        s = _HTML_TAG_RE.sub("", s)
        s = normalize_text(html.unescape(s))
        s = _ESCAPED_FROALA_RE.sub(" ", s)
        # Collapse source newlines inside HTML blocks to spaces, then restore <br> (\x01) to \n
        s = s.replace("\n", " ").replace("\x01", "\n")
        raw_chunks = s.split("\x00")
    else:
        s = normalize_text(html.unescape(s))
        s = _ESCAPED_FROALA_RE.sub(" ", s)
        raw_chunks = re.split(r"\n[^\S\n]*\n", s)

    paras: list[str] = []
    for chunk in raw_chunks:
        lines = [_HSPACE_RE.sub(" ", ln).strip() for ln in chunk.split("\n")]
        text = "\n".join(ln for ln in lines if ln)
        if text and not _MARKUP_ONLY_RE.match(text):
            paras.append(text)
    return paras


def split_plain_paragraphs(body: str) -> list[str]:
    """Split normalized plain-text `body` into paragraphs on blank lines (`\\n[^\\S\\n]*\\n`).

    Single `'\\n'` soft line breaks (e.g. from `<br>`) remain within their paragraph.
    """
    if not body or not body.strip():
        return []
    return [p.strip() for p in re.split(r"\n[^\S\n]*\n", body) if p.strip()]


def strip_html_boilerplate(raw_html_or_text: str) -> str:
    """Strip script/style/nav/footer blocks and HTML tags before unescaping HTML entities,
    joining block-level paragraphs with `PARAGRAPH_SEP` (`'\\n\\n'`) and preserving `<br>` as
    single soft line breaks (`'\\n'`).
    """
    return PARAGRAPH_SEP.join(html_to_paragraphs(raw_html_or_text))


def clean_title(title: str) -> str:
    """Normalize title text: strip real HTML tags first, then unescape entities and remove suffix boilerplate."""
    raw = normalize_text(title or "")
    no_tags = _HTML_TAG_RE.sub("", raw)
    unescaped = normalize_text(html.unescape(no_tags))
    text = re.sub(r"\s+", " ", unescaped).strip()
    stripped_suffix = _TITLE_SUFFIX_RE.sub("", text).strip()
    if stripped_suffix:
        text = stripped_suffix
    if not text:
        raise NormalizationError("Article title must not be empty after normalization")
    return text


def clean_body(body: str) -> str:
    """Normalize body text (NFC + LF) without collapsing internal whitespace after hashing."""
    text = normalize_text(body or "").strip()
    if not text:
        raise NormalizationError("Article body must not be empty after normalization")
    return text


def normalize_url(url: str) -> str:
    """Canonicalize an article URL strictly per INGEST_CONTRACT.md §1 L37-39:

    - NFC normalize and strip surrounding whitespace
    - Remove `#fragment`
    - Lowercase scheme and hostname
    - Remove default ports (`:443` for https, `:80` for http)
    - Preserve path case, raw query parameters byte-for-byte, and trailing slash.
    """
    raw = normalize_text(url or "").strip()
    if not raw:
        raise NormalizationError("URL must not be empty")
    parsed = urllib.parse.urlsplit(raw)
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise NormalizationError(f"URL scheme must be http or https, got {raw!r}")
    if not parsed.hostname:
        raise NormalizationError(f"URL is missing a hostname: {raw!r}")

    host = parsed.hostname.lower()
    try:
        port = parsed.port
    except ValueError as exc:
        raise NormalizationError(f"Invalid port in URL {raw!r}: {exc}") from exc

    if (scheme == "https" and port == 443) or (scheme == "http" and port == 80) or port is None:
        netloc = host
    else:
        netloc = f"{host}:{port}"

    path = parsed.path or "/"
    query = parsed.query
    return urllib.parse.urlunsplit((scheme, netloc, path, query, ""))


def compute_doc_id(canonical_url: str) -> str:
    """Compute `doc_` + full lowercase SHA-256 of canonical UTF-8 URL (INGEST_CONTRACT §1)."""
    normalized = normalize_url(canonical_url)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"doc_{digest}"


def _resolve_tz(source_tz: str | ZoneInfo | None) -> ZoneInfo | None:
    if source_tz is None:
        return None
    if isinstance(source_tz, ZoneInfo):
        return source_tz
    cleaned = source_tz.strip()
    if not cleaned:
        return None
    if cleaned.upper() == "UTC":
        return ZoneInfo("UTC")
    return ZoneInfo(cleaned)


def _format_utc_iso(dt: datetime) -> str:
    """Format an aware datetime as UTC ISO-8601 `Z`, retaining nonzero fractional seconds
    without rounding up across second/day boundaries (INGEST_CONTRACT.md §1 L23).
    """
    utc_dt = dt.astimezone(UTC)
    if utc_dt.microsecond:
        frac = f"{utc_dt.microsecond:06d}".rstrip("0")
        return utc_dt.strftime("%Y-%m-%dT%H:%M:%S") + f".{frac}Z"
    return utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc_timestamp(
    value: str | datetime | date | None,
    *,
    source_tz: str | ZoneInfo | None = None,
) -> str:
    """Parse a timestamp into canonical UTC ISO-8601 `Z` format (retaining nonzero fractional seconds).

    Boundary rules (INGEST_CONTRACT.md §1 & §3):
      - Date-only inputs (`YYYY-MM-DD`, `YYYY.MM.DD`, `YYYY/MM/DD`, `YYYY年MM月DD日`) map to
        `YYYY-MM-DDT00:00:00Z` on that UTC calendar day.
      - Datetime inputs with an explicit offset are converted to UTC.
      - Naive datetime inputs (containing a time-of-day without an offset) require an explicit
        `source_tz`; if `source_tz` is None, raises `TimestampParseError` (never host-local time).
      - Nonzero fractional seconds are preserved without rounding up.
    """
    if value is None:
        raise TimestampParseError("published_at is None")

    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
            tz = _resolve_tz(source_tz)
            if tz is None:
                raise TimestampParseError(
                    f"Naive datetime {value!r} requires a documented source_tz (never host-local)"
                )
            dt = dt.replace(tzinfo=tz)
        return _format_utc_iso(dt)

    if isinstance(value, date):
        return f"{value.isoformat()}T00:00:00Z"

    if not isinstance(value, str):
        raise TimestampParseError(f"Unsupported published_at type: {type(value).__name__}")

    raw = normalize_text(value).strip()
    if not raw:
        raise TimestampParseError("published_at is empty")

    # 1. Check date-only patterns (e.g., 2026-10-28, 2026.10.28, 2026/10/28, 2026年10月28日)
    date_match = _DATE_ONLY_RE.match(raw)
    if date_match:
        year, month, day = (
            int(date_match.group(1)),
            int(date_match.group(2)),
            int(date_match.group(3)),
        )
        try:
            d = date(year, month, day)
        except ValueError as exc:
            raise TimestampParseError(f"Invalid calendar date {raw!r}: {exc}") from exc
        return f"{d.isoformat()}T00:00:00Z"

    # 2. Try ISO-8601 datetime
    iso_candidate = raw[:-1] + "+00:00" if raw.endswith(("Z", "z")) else raw
    try:
        dt = datetime.fromisoformat(iso_candidate)
        if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
            tz = _resolve_tz(source_tz)
            if tz is None:
                raise TimestampParseError(
                    f"Naive ISO datetime {raw!r} lacks timezone offset and no source_tz was provided"
                )
            dt = dt.replace(tzinfo=tz)
        return _format_utc_iso(dt)
    except TimestampParseError:
        raise
    except ValueError:
        pass

    # 3. Try RFC-2822 (common in RSS <pubDate>)
    try:
        dt = email.utils.parsedate_to_datetime(raw)
        if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
            # RFC-2822 `-0000` is parsed by stdlib as tzinfo=None even though it denotes UTC (00:00)
            if _EXPLICIT_UTC_RFC_RE.search(raw):
                dt = dt.replace(tzinfo=UTC)
            else:
                tz = _resolve_tz(source_tz)
                if tz is None:
                    raise TimestampParseError(
                        f"Naive RFC-2822 datetime {raw!r} lacks timezone offset and no source_tz was provided"
                    )
                dt = dt.replace(tzinfo=tz)
        return _format_utc_iso(dt)
    except TimestampParseError:
        raise
    except (TypeError, ValueError, IndexError):
        pass

    # 4. Try embedded Japanese/dot/slash date with time (e.g. "2026.10.28 15:00" or "2026年10月28日 12:30")
    dt_match = re.match(
        r"^(\d{4})[-./年](\d{1,2})[-./月](\d{1,2})日?\s+(\d{1,2}):(\d{2})(?::(\d{2}))?$",
        raw,
    )
    if dt_match:
        tz = _resolve_tz(source_tz)
        if tz is None:
            raise TimestampParseError(
                f"Naive datetime {raw!r} lacks timezone offset and no source_tz was provided"
            )
        y, m, d_num = int(dt_match.group(1)), int(dt_match.group(2)), int(dt_match.group(3))
        hh, mm, ss = int(dt_match.group(4)), int(dt_match.group(5)), int(dt_match.group(6) or 0)
        try:
            dt = datetime(y, m, d_num, hh, mm, ss, tzinfo=tz)
        except ValueError as exc:
            raise TimestampParseError(f"Invalid datetime components in {raw!r}: {exc}") from exc
        return _format_utc_iso(dt)

    raise TimestampParseError(f"Cannot parse published_at timestamp: {raw!r}")


_FISCAL_PR_TITLE_RE = re.compile(r"^プレスリリース\s*\d{4}年\d{1,2}月期$")


def is_fiscal_pr_boilerplate_title(title: str) -> bool:
    """Return True when `title` matches a generic fiscal-period boilerplate header."""
    return bool(_FISCAL_PR_TITLE_RE.match((title or "").strip()))


def promote_boilerplate_title(title: str, body: str) -> str:
    """When a feed item's `<title>` is a generic fiscal-period boilerplate label
    (e.g. Capcom's `'プレスリリース 2027年3月期'`), promote the leading headline line
    from `body` into `title` (INGEST_CONTRACT.md §1).
    """
    if not _FISCAL_PR_TITLE_RE.match(title) or not body.strip():
        return title
    lines = [
        ln.strip()
        for ln in body.strip().splitlines()
        if ln.strip() and not ln.strip().startswith(("\uff5e", "~"))
    ]
    if not lines:
        return title
    head = lines[0]
    if head.endswith("、") and len(lines) > 1:
        head = head + lines[1]
    return clean_title(head)


def utc_iso_at_or_after(item_iso: str, since_iso: str) -> bool:
    """Compare two canonical UTC ISO-8601 `Z` timestamps chronologically (preserving fractional
    seconds so `'2026-10-28T00:00:00.5Z' >= '2026-10-28T00:00:00Z'` is True).
    """
    dt_item = datetime.fromisoformat(item_iso.replace("Z", "+00:00")).astimezone(UTC)
    dt_since = datetime.fromisoformat(since_iso.replace("Z", "+00:00")).astimezone(UTC)
    return dt_item >= dt_since


def is_corporate_ir_text(text: str) -> bool:
    """True when text matches the corporate/IR pre-filter (RUNBOOK §3.6 & issue #17)."""
    return any(kw in text for kw in IR_KEYWORDS) or bool(IR_TOKEN_PATTERN.search(text))


def infer_event_type(
    title: str, body: str = "", *, ingest_category: str | None = None
) -> str | None:
    """Infer an internal clustering event_type tag from article title/body keywords."""
    if ingest_category == "corporate":
        return "corporate"
    combined = f"{title}\n{body[:400]}"
    lowered = combined.lower()
    for event_label, keywords in _EVENT_RULES:
        for kw in keywords:
            if kw in combined or kw.lower() in lowered:
                return event_label
    return None


def classify_content_category(
    title: str,
    body: str = "",
    *,
    source_key: str = "",
    default_category: str = "anime",
) -> Literal["anime", "manga", "game", "light_novel"]:
    """Classify an item into one of the four G1a domain categories for corpus balance reporting:
    `anime`, `manga`, `game`, or `light_novel`.
    """
    key_lower = source_key.lower()
    if any(tok in key_lower for tok in ("syosetu", "novel")):
        return "light_novel"
    if any(tok in key_lower for tok in ("capcom", "gcores", "chuapp", "gameres", "yystv")):
        return "game"
    if any(tok in key_lower for tok in ("toei", "mappa", "bones", "paworks", "bilibili")):
        return "anime"

    text = f"{title}\n{body[:600]}"
    text_lower = text.lower()
    if any(
        kw in text or kw in text_lower
        for kw in (
            "light novel",
            "ラノベ",
            "ライトノベル",
            "小説",
            "文庫",
            "轻小说",
            "輕小說",
            "novel",
        )
    ):
        return "light_novel"
    if any(
        kw in text or kw in text_lower
        for kw in ("manga", "マンガ", "漫画", "コミックス", "連載", "単行本", "漫畫")
    ):
        return "manga"
    if any(
        kw in text or kw in text_lower
        for kw in (
            "game",
            "ゲーム",
            "steam",
            "playstation",
            "nintendo",
            "switch",
            "xbox",
            "游戏",
            "遊戲",
            "手游",
        )
    ):
        return "game"
    if default_category in ("anime", "manga", "game", "light_novel"):
        return default_category  # type: ignore[return-value]
    return "anime"


@dataclass(frozen=True)
class RawIngestItem:
    """Raw item produced by a `SourceAdapter` prior to canonical normalization & entity linking.

    `source_tz` defaults to `None` so naive timestamps without an explicit documented source
    timezone fail normalization rather than silently assuming UTC (INGEST_CONTRACT.md §3 L104-105).
    """

    url: str
    title: str
    body: str
    published_at: str
    publisher: str
    source_type: SourceType
    source_key: str
    lang: str = "und"
    source_tz: str | None = None
    ingest_category: str | None = None
    event_type: str | None = None
    media_ids: tuple[int, ...] = ()
    external_media_refs: tuple[tuple[str, str], ...] = ()
    fetched_via: Literal["direct", "wayback"] = "direct"


@dataclass(frozen=True)
class NormalizedItem:
    """Canonical 13-field normalized article record (docs/INGEST_CONTRACT.md §1)."""

    doc_id: str
    title: str
    body: str
    url: str
    publisher: str
    source_type: SourceType
    published_at: str
    lang: str
    media_ids: list[int] = field(default_factory=list)
    ingest_category: str | None = None
    cluster_id: str | None = None
    event_type: str | None = None
    source_key: str = ""

    def __post_init__(self) -> None:
        if not self.doc_id.startswith("doc_") or len(self.doc_id) != 68:
            raise NormalizationError(f"Invalid doc_id format: {self.doc_id!r}")
        if self.source_type not in VALID_SOURCE_TYPES:
            raise NormalizationError(f"Invalid source_type: {self.source_type!r}")
        if not self.published_at.endswith("Z"):
            raise NormalizationError(
                f"NormalizedItem.published_at must be UTC Z ISO-8601, got {self.published_at!r}"
            )
        if not self.source_key:
            raise NormalizationError("NormalizedItem.source_key must not be empty")
        sorted_ids = sorted(set(int(mid) for mid in self.media_ids))
        if any(mid <= 0 for mid in sorted_ids):
            raise NormalizationError(f"media_ids must be positive integers, got {sorted_ids}")
        object.__setattr__(self, "media_ids", sorted_ids)

    @property
    def body_sha256(self) -> str:
        """Lowercase hex SHA-256 of the normalized UTF-8 body."""
        return hashlib.sha256(self.body.encode("utf-8")).hexdigest()

    def with_media_ids(self, media_ids: Iterable[int]) -> NormalizedItem:
        return replace(self, media_ids=sorted(set(int(m) for m in media_ids)))

    def with_cluster_id(self, cluster_id: str) -> NormalizedItem:
        if not cluster_id.startswith("cluster_") or len(cluster_id) != 24:
            raise NormalizationError(f"Invalid cluster_id format: {cluster_id!r}")
        return replace(self, cluster_id=cluster_id)

    def to_dict(self) -> dict[str, Any]:
        """Full 13-field dictionary for internal `items.jsonl` storage."""
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "body": self.body,
            "url": self.url,
            "publisher": self.publisher,
            "source_type": self.source_type,
            "published_at": self.published_at,
            "lang": self.lang,
            "media_ids": list(self.media_ids),
            "ingest_category": self.ingest_category,
            "cluster_id": self.cluster_id,
            "event_type": self.event_type,
            "source_key": self.source_key,
        }

    def to_manifest_item(self) -> dict[str, Any]:
        """Canonical manifest item dictionary (all fields except `body`, plus `body_sha256`)."""
        if self.cluster_id is None:
            raise NormalizationError(
                f"Cannot serialize {self.doc_id} to manifest before assigning cluster_id"
            )
        return {
            "body_sha256": self.body_sha256,
            "cluster_id": self.cluster_id,
            "doc_id": self.doc_id,
            "event_type": self.event_type,
            "ingest_category": self.ingest_category,
            "lang": self.lang,
            "media_ids": list(self.media_ids),
            "published_at": self.published_at,
            "publisher": self.publisher,
            "source_key": self.source_key,
            "source_type": self.source_type,
            "title": self.title,
            "url": self.url,
        }

    def to_payload_fields(self) -> dict[str, Any]:
        """Return the WP2 input seam dictionary consumed by `IndexDoc` and `payload_from_fields`."""
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "body": self.body,
            "url": self.url,
            "publisher": self.publisher,
            "source_type": self.source_type,
            "published_at": self.published_at,
            "lang": self.lang,
            "media_ids": list(self.media_ids),
            "ingest_category": self.ingest_category,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> NormalizedItem:
        return cls(
            doc_id=str(data["doc_id"]),
            title=str(data["title"]),
            body=str(data["body"]),
            url=str(data["url"]),
            publisher=str(data["publisher"]),
            source_type=data["source_type"],
            published_at=str(data["published_at"]),
            lang=str(data["lang"]),
            media_ids=[int(x) for x in data.get("media_ids", [])],
            ingest_category=data.get("ingest_category"),
            cluster_id=data.get("cluster_id"),
            event_type=data.get("event_type"),
            source_key=str(data["source_key"]),
        )


@dataclass(frozen=True)
class QuarantinedItem:
    """Record of an item rejected during normalization (e.g. missing timezone or empty body)."""

    source_key: str
    url: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "reason": self.reason,
            "source_key": self.source_key,
            "url": self.url,
        }


@dataclass(frozen=True)
class NormalizationReport:
    """Output of `normalize_batch`: accepted sorted items plus quarantined and drop counters."""

    items: list[NormalizedItem]
    quarantined: list[QuarantinedItem]
    out_of_window_count: int = 0
    exact_duplicates_collapsed: int = 0


def normalize_item(raw: RawIngestItem) -> NormalizedItem:
    """Convert one `RawIngestItem` into a validated `NormalizedItem`."""
    canonical_url = normalize_url(raw.url)
    if _INDEX_PAGE_URL_RE.search(canonical_url):
        raise NormalizationError(
            f"index_page: listing index URL is not an article ({canonical_url!r})"
        )
    doc_id = compute_doc_id(canonical_url)
    raw_cleaned_title = clean_title(raw.title)
    body = clean_body(raw.body)
    title = promote_boilerplate_title(raw_cleaned_title, body)
    was_promoted = title != raw_cleaned_title
    publisher = normalize_text(raw.publisher or "").strip()
    if not publisher:
        raise NormalizationError(f"Publisher display name is empty for {canonical_url!r}")
    if raw.source_type not in VALID_SOURCE_TYPES:
        raise NormalizationError(
            f"Invalid source_type {raw.source_type!r} for {canonical_url!r} (C5)"
        )
    source_key = normalize_text(raw.source_key or "").strip()
    if not source_key:
        raise NormalizationError(f"source_key is empty for {canonical_url!r}")

    published_at = parse_utc_timestamp(raw.published_at, source_tz=raw.source_tz)
    lang_raw = normalize_text(raw.lang or "und").strip().lower()
    lang = lang_raw[:2] if lang_raw[:2] in ("ja", "en", "zh") else "und"

    ingest_category = None if was_promoted else raw.ingest_category
    if (
        ingest_category is None
        and raw.source_type == "official"
        and is_corporate_ir_text(f"{title}\n{body}")
    ):
        ingest_category = "corporate"

    raw_event = None if (was_promoted and raw.event_type == "corporate") else raw.event_type
    event_type = raw_event or infer_event_type(title, body, ingest_category=ingest_category)

    return NormalizedItem(
        doc_id=doc_id,
        title=title,
        body=body,
        url=canonical_url,
        publisher=publisher,
        source_type=raw.source_type,
        published_at=published_at,
        lang=lang,
        media_ids=sorted(set(int(m) for m in raw.media_ids if int(m) > 0)),
        ingest_category=ingest_category,
        cluster_id=None,
        event_type=event_type,
        source_key=source_key,
    )


def _window_bound_to_date_str(bound: date | datetime | str | None) -> str:
    if bound is None:
        return ""
    if isinstance(bound, datetime):
        return bound.date().isoformat()
    if isinstance(bound, date):
        return bound.isoformat()
    s = str(bound).strip()
    return s[:10] if len(s) >= 10 else s


def _reconcile_key(item: NormalizedItem) -> tuple[Any, ...]:
    """All manifest-participating fields (except media_ids) that must agree for same-URL duplicates."""
    return (
        item.title,
        item.body,
        item.published_at,
        item.publisher,
        item.source_type,
        item.lang,
        item.ingest_category,
        item.event_type,
        item.source_key,
    )


def normalize_batch(
    raw_items: Sequence[RawIngestItem],
    *,
    window_start: date | datetime | str | None = None,
    window_end: date | datetime | str | None = None,
    quarantine_on_error: bool = False,
) -> NormalizationReport:
    """Normalize a sequence of raw items, enforce window bounds, and reconcile same-URL duplicates.

    - Identical same-URL observations collapse into one `NormalizedItem`.
    - Conflicting same-URL observations (differing in any manifest field: title, body,
      published_at, publisher, source_type, lang, ingest_category, event_type, or source_key)
      always raise `ConflictingObservationError` (INGEST_CONTRACT.md §1 L39-40), never picking
      an arbitrary order-dependent winner.
    - Out-of-window items are filtered and counted when `window_start` / `window_end` are supplied.
    """
    start_str = _window_bound_to_date_str(window_start)
    end_str = _window_bound_to_date_str(window_end)

    by_doc_id: dict[str, NormalizedItem] = {}
    quarantined: list[QuarantinedItem] = []
    out_of_window_count = 0
    exact_duplicates_collapsed = 0

    for raw in raw_items:
        try:
            item = normalize_item(raw)
        except NormalizationError as exc:
            if quarantine_on_error:
                quarantined.append(
                    QuarantinedItem(
                        source_key=getattr(raw, "source_key", ""),
                        url=getattr(raw, "url", ""),
                        reason=str(exc),
                    )
                )
                continue
            raise

        pub_day = item.published_at[:10]
        if start_str and pub_day < start_str:
            out_of_window_count += 1
            continue
        if end_str and pub_day > end_str:
            out_of_window_count += 1
            continue

        existing = by_doc_id.get(item.doc_id)
        if existing is None:
            by_doc_id[item.doc_id] = item
            continue

        if _reconcile_key(existing) != _reconcile_key(item):
            raise ConflictingObservationError(
                doc_id=item.doc_id,
                url=item.url,
                detail=(
                    f"existing=({existing.title!r}, {existing.publisher!r}, {existing.source_key!r}, "
                    f"{existing.published_at!r}, body_sha256={existing.body_sha256[:12]}) "
                    f"vs new=({item.title!r}, {item.publisher!r}, {item.source_key!r}, "
                    f"{item.published_at!r}, body_sha256={item.body_sha256[:12]})"
                ),
            )
        exact_duplicates_collapsed += 1
        merged_media = sorted(set(existing.media_ids) | set(item.media_ids))
        by_doc_id[item.doc_id] = existing.with_media_ids(merged_media)

    sorted_items = [by_doc_id[k] for k in sorted(by_doc_id)]
    return NormalizationReport(
        items=sorted_items,
        quarantined=quarantined,
        out_of_window_count=out_of_window_count,
        exact_duplicates_collapsed=exact_duplicates_collapsed,
    )


__all__ = [
    "HTML_BLOCK_BOUNDARY_RE",
    "IR_KEYWORDS",
    "IR_TOKEN_PATTERN",
    "NORMALIZATION_VERSION",
    "PARAGRAPH_SEP",
    "ConflictingObservationError",
    "NormalizationError",
    "NormalizationReport",
    "NormalizedItem",
    "QuarantinedItem",
    "RawIngestItem",
    "TimestampParseError",
    "canonical_json",
    "classify_content_category",
    "clean_body",
    "clean_title",
    "compute_doc_id",
    "html_to_paragraphs",
    "infer_event_type",
    "is_corporate_ir_text",
    "is_fiscal_pr_boilerplate_title",
    "normalize_batch",
    "normalize_item",
    "normalize_text",
    "normalize_url",
    "parse_utc_timestamp",
    "split_plain_paragraphs",
    "strip_html_boilerplate",
]
