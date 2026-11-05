"""Polite HTTP client with per-host rate limiting, robots.txt enforcement, conditional GET,
exponential retry with Retry-After, per-host circuit breaker, and charset decode quality guard.

Owner: A (WP1) | Spec: docs/SOURCES.md §7-§8, docs/INGEST_CONTRACT.md §5, docs/COMPLIANCE.md C1
"""

from __future__ import annotations

import email.utils
import json
import logging
import random
import re
import time
import urllib.parse
import urllib.robotparser
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path

import httpx
from charset_normalizer import from_bytes as cn_from_bytes

from acgn_rag.config import HttpConfig, load_settings

logger = logging.getLogger(__name__)

# Retry schedule per docs/SOURCES.md §8: 3 retry sleeps of 1s -> 2s -> 4s (4 total attempts)
DEFAULT_BACKOFF_SCHEDULE: tuple[float, ...] = (1.0, 2.0, 4.0)
DEFAULT_MAX_ATTEMPTS: int = 4
DEFAULT_BREAKER_THRESHOLD: int = 3
DEFAULT_BREAKER_COOLDOWN_S: float = 1800.0  # 30 minutes per docs/SOURCES.md §8
DEFAULT_ROBOTS_TTL_S: float = 86400.0  # 24 hours per docs/SOURCES.md §7
DEFAULT_MAX_REPLACEMENT_RATIO: float = 0.001  # 0.1% max U+FFFD ratio
DEFAULT_MAX_REDIRECTS: int = 5

_META_CHARSET_RE = re.compile(
    rb"""<meta[^>]+charset\s*=\s*["']?\s*([A-Za-z0-9._\-]+)""",
    re.IGNORECASE,
)
_XML_ENCODING_RE = re.compile(
    rb"""<\?xml[^>]+encoding\s*=\s*["']([A-Za-z0-9._\-]+)["']""",
    re.IGNORECASE,
)
_CONTENT_TYPE_CHARSET_RE = re.compile(r"charset\s*=\s*([^\s;\"']+)", re.IGNORECASE)

_SINGLE_BYTE_LATIN_ENCODINGS: frozenset[str] = frozenset(
    {
        "ascii",
        "us-ascii",
        "iso-8859-1",
        "iso8859-1",
        "latin1",
        "latin-1",
        "cp1252",
        "windows-1252",
    }
)


class IngestHttpError(RuntimeError):
    """Base class for deterministic, visible HTTP/ingestion failures (INGEST_CONTRACT.md §5)."""

    def __init__(self, message: str, *, url: str = "", status_code: int | None = None) -> None:
        super().__init__(message)
        self.url = url
        self.status_code = status_code


class RobotsDisallowedError(IngestHttpError):
    """Raised when `robots.txt` disallows fetching a URL or returns 5xx/429/unreachable (C1)."""

    def __init__(
        self,
        message: str,
        *,
        url: str = "",
        status_code: int | None = None,
        explicit_disallow: bool = False,
    ) -> None:
        super().__init__(message, url=url, status_code=status_code)
        self.explicit_disallow = explicit_disallow


class CircuitBreakerOpenError(IngestHttpError):
    """Raised when a host's circuit breaker is open after repeated consecutive failures."""


class DecodeQualityError(IngestHttpError):
    """Raised when response text fails charset decoding or exceeds the U+FFFD replacement limit."""


@dataclass(frozen=True)
class ConditionalCacheEntry:
    """Cached conditional-GET validators and decoded payload for a URL."""

    url: str
    etag: str | None = None
    last_modified: str | None = None
    body_text: str = ""
    encoding: str = "utf-8"
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class FetchResult:
    """Normalized HTTP response returned by `PoliteHttpClient`."""

    url: str
    status_code: int
    text: str
    content: bytes
    encoding: str
    headers: dict[str, str]
    not_modified: bool = False
    from_cache: bool = False


@dataclass
class BreakerEvent:
    """Structured record of a circuit-breaker state transition for logging/auditing."""

    host: str
    consecutive_failures: int
    opened_at: float
    open_until: float
    last_error: str


@dataclass
class _HostRobotsPolicy:
    fetched_at: float
    parser: urllib.robotparser.RobotFileParser | None
    crawl_delay: float | None
    allow_all: bool = False
    disallow_all: bool = False

    def can_fetch(self, user_agent: str, url: str) -> bool:
        if self.disallow_all:
            return False
        if self.allow_all or self.parser is None:
            return True
        bot_token = user_agent.split("/", 1)[0].strip() or "*"
        return bool(
            self.parser.can_fetch(user_agent, url) and self.parser.can_fetch(bot_token, url)
        )


@dataclass
class _HostBreakerState:
    consecutive_failures: int = 0
    open_until: float = 0.0
    last_error: str = ""


def validate_user_agent(user_agent: str) -> str:
    """Validate that `user_agent` is non-empty, identifies the bot, and is not a spoofed browser UA."""
    cleaned = (user_agent or "").strip()
    if not cleaned:
        raise ValueError("HTTP User-Agent must not be empty (docs/COMPLIANCE.md C1)")
    lower = cleaned.lower()
    if lower.startswith("mozilla/") or (
        "acgnragbot" not in lower and "acgn-chronicle" not in lower and "cs5481" not in lower
    ):
        raise ValueError(
            f"HTTP User-Agent {cleaned!r} must identify AcgnRagBot/project and not spoof a browser (C1)"
        )
    return cleaned


def _extract_header_charset(content_type: str | None) -> str | None:
    if not content_type:
        return None
    match = _CONTENT_TYPE_CHARSET_RE.search(content_type)
    if not match:
        return None
    return match.group(1).strip().strip("\"'").lower()


def _extract_declared_encoding(content: bytes, content_type: str | None = None) -> list[str]:
    """Return candidate encodings in priority order: HTTP header -> XML declaration -> HTML <meta>."""
    candidates: list[str] = []
    header_enc = _extract_header_charset(content_type)
    if header_enc:
        candidates.append(header_enc)

    head_slice = content[:4096]
    xml_match = _XML_ENCODING_RE.search(head_slice)
    if xml_match:
        xml_enc = xml_match.group(1).decode("ascii", errors="ignore").lower()
        if xml_enc and xml_enc not in candidates:
            candidates.append(xml_enc)

    meta_match = _META_CHARSET_RE.search(head_slice)
    if meta_match:
        meta_enc = meta_match.group(1).decode("ascii", errors="ignore").lower()
        if meta_enc and meta_enc not in candidates:
            candidates.append(meta_enc)

    return candidates


def _has_c1_controls(text: str) -> bool:
    """True when decoded text contains C1 control characters U+0080..U+009F (classic UTF-8-as-Latin-1 mojibake)."""
    return any(0x80 <= ord(ch) <= 0x9F for ch in text)


def _check_decoded_quality(
    text: str,
    *,
    url: str,
    encoding: str,
    max_replacement_ratio: float,
) -> float:
    if not text:
        return 0.0
    if "\x00" in text:
        raise DecodeQualityError(
            f"Decoded text from {url!r} (encoding={encoding}) contains NUL bytes",
            url=url,
        )
    if _has_c1_controls(text):
        raise DecodeQualityError(
            f"Decoded text from {url!r} (encoding={encoding}) contains C1 control characters "
            f"(U+0080..U+009F mojibake)",
            url=url,
        )
    replacement_count = text.count("\ufffd")
    ratio = replacement_count / len(text)
    if ratio > max_replacement_ratio:
        raise DecodeQualityError(
            f"Decoded text from {url!r} (encoding={encoding}) exceeded decode replacement ratio "
            f"{ratio:.4f} > {max_replacement_ratio:.4f} ({replacement_count}/{len(text)})",
            url=url,
        )
    return ratio


def decode_bytes_checked(
    content: bytes,
    *,
    url: str = "",
    content_type: str | None = None,
    max_replacement_ratio: float = DEFAULT_MAX_REPLACEMENT_RATIO,
) -> tuple[str, str, float]:
    """Decode `content` into `(text, encoding, replacement_ratio)` and enforce the decode quality guard.

    Never trusts single-byte Latin labels (`iso-8859-1`, `cp1252`) when `content` contains
    non-ASCII bytes, preventing silent UTF-8 Japanese/Chinese mojibake (`RUNBOOK.md` §3).
    """
    if not content:
        return "", "utf-8", 0.0

    has_non_ascii = any(b >= 0x80 for b in content)
    declared = _extract_declared_encoding(content, content_type)

    # 1. If non-ASCII bytes are present, try strict UTF-8 / UTF-8-SIG first when declared
    # encoding is absent, UTF-8, or a suspicious single-byte Latin label.
    has_utf8_bom = content.startswith(b"\xef\xbb\xbf")
    utf_candidates = ("utf-8-sig", "utf-8") if has_utf8_bom else ("utf-8", "utf-8-sig")
    if has_non_ascii and (
        not declared
        or declared[0] in ("utf-8", "utf8", "utf-8-sig")
        or declared[0] in _SINGLE_BYTE_LATIN_ENCODINGS
    ):
        for utf_enc in utf_candidates:
            try:
                text = content.decode(utf_enc, errors="strict")
                ratio = _check_decoded_quality(
                    text,
                    url=url,
                    encoding=utf_enc,
                    max_replacement_ratio=max_replacement_ratio,
                )
                return text, ("utf-8-sig" if has_utf8_bom else "utf-8"), ratio
            except (UnicodeDecodeError, DecodeQualityError):
                continue

    # 2. Try non-Latin declared encodings (e.g. Shift_JIS / EUC-JP / GB18030) strictly
    for enc in declared:
        if has_non_ascii and enc in _SINGLE_BYTE_LATIN_ENCODINGS:
            continue
        try:
            text = content.decode(enc, errors="strict")
            ratio = _check_decoded_quality(
                text,
                url=url,
                encoding=enc,
                max_replacement_ratio=max_replacement_ratio,
            )
            return text, enc, ratio
        except (LookupError, UnicodeDecodeError, DecodeQualityError):
            continue

    # 3. Try strict UTF-8 if not tried above
    for utf_enc in utf_candidates:
        try:
            text = content.decode(utf_enc, errors="strict")
            ratio = _check_decoded_quality(
                text,
                url=url,
                encoding=utf_enc,
                max_replacement_ratio=max_replacement_ratio,
            )
            return text, ("utf-8-sig" if has_utf8_bom else "utf-8"), ratio
        except (UnicodeDecodeError, DecodeQualityError):
            continue

    # 4. Try charset-normalizer detection (excluding single-byte Latin on multibyte streams)
    matches = cn_from_bytes(content)
    best = matches.best()
    if best is not None and best.encoding:
        detected_enc = best.encoding.lower()
        if not (has_non_ascii and detected_enc in _SINGLE_BYTE_LATIN_ENCODINGS):
            try:
                text = content.decode(detected_enc, errors="replace")
                ratio = _check_decoded_quality(
                    text,
                    url=url,
                    encoding=detected_enc,
                    max_replacement_ratio=max_replacement_ratio,
                )
                return text, detected_enc, ratio
            except (LookupError, DecodeQualityError):
                pass

    # 5. Final fallback: decode with replacement using primary non-Latin declared encoding or utf-8
    fallback_enc = (
        declared[0]
        if (declared and not (has_non_ascii and declared[0] in _SINGLE_BYTE_LATIN_ENCODINGS))
        else "utf-8"
    )
    try:
        text = content.decode(fallback_enc, errors="replace")
    except LookupError:
        fallback_enc = "utf-8"
        text = content.decode("utf-8", errors="replace")

    ratio = _check_decoded_quality(
        text,
        url=url,
        encoding=fallback_enc,
        max_replacement_ratio=max_replacement_ratio,
    )
    return text, fallback_enc, ratio


def _parse_retry_after(header_value: str | None, *, now: float) -> float | None:
    """Parse an HTTP `Retry-After` header (delta seconds or HTTP-date) into non-negative seconds."""
    if not header_value:
        return None
    raw = header_value.strip()
    if not raw:
        return None
    try:
        seconds = float(raw)
        return max(0.0, seconds)
    except ValueError:
        pass
    try:
        dt = email.utils.parsedate_to_datetime(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return max(0.0, dt.timestamp() - now)
    except (TypeError, ValueError, OverflowError):
        return None


def _extract_crawl_delay(robots_txt: str, user_agent: str) -> float | None:
    """Extract `Crawl-delay` from `robots.txt` for our bot token or `*`."""
    bot_token = user_agent.split("/", 1)[0].strip().lower()
    current_agents: list[str] = []
    specific_delay: float | None = None
    wildcard_delay: float | None = None

    for raw_line in robots_txt.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            if not line:
                current_agents = []
            continue
        key, val = (part.strip() for part in line.split(":", 1))
        key_lower = key.lower()
        if key_lower == "user-agent":
            current_agents.append(val.lower())
        elif key_lower == "crawl-delay":
            try:
                delay = float(val)
            except ValueError:
                continue
            if delay < 0:
                continue
            if any(bot_token and a in (bot_token, "*") for a in current_agents):
                if bot_token and bot_token in current_agents:
                    specific_delay = delay
                elif "*" in current_agents and wildcard_delay is None:
                    wildcard_delay = delay

    return specific_delay if specific_delay is not None else wildcard_delay


class PoliteHttpClient:
    """Polite synchronous HTTP client enforcing COMPLIANCE C1 and SOURCES §7-§8."""

    def __init__(
        self,
        config: HttpConfig | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        jitter_range: tuple[float, float] = (0.05, 0.25),
        jitter_fn: Callable[[float, float], float] = random.uniform,
        backoff_schedule: tuple[float, ...] = DEFAULT_BACKOFF_SCHEDULE,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        breaker_threshold: int = DEFAULT_BREAKER_THRESHOLD,
        breaker_cooldown_s: float = DEFAULT_BREAKER_COOLDOWN_S,
        robots_ttl_s: float = DEFAULT_ROBOTS_TTL_S,
        max_replacement_ratio: float = DEFAULT_MAX_REPLACEMENT_RATIO,
        conditional_cache_path: Path | None = None,
    ) -> None:
        resolved_cfg = config if config is not None else load_settings().http
        if getattr(resolved_cfg, "respect_robots", True) is False:
            raise ValueError(
                "Disabling robots.txt (respect_robots=False) is forbidden under docs/COMPLIANCE.md C1"
            )
        self.user_agent = validate_user_agent(resolved_cfg.user_agent)
        self.timeout_s = float(resolved_cfg.timeout_s)
        self.default_min_interval_s = max(0.0, float(resolved_cfg.min_interval_s))
        self._clock = clock
        self._wall_clock = wall_clock
        self._sleep = sleep
        self._jitter_range = jitter_range
        self._jitter_fn = jitter_fn
        self.backoff_schedule = backoff_schedule
        self.max_attempts = max(1, int(max_attempts))
        self.breaker_threshold = max(1, int(breaker_threshold))
        self.breaker_cooldown_s = float(breaker_cooldown_s)
        self.robots_ttl_s = float(robots_ttl_s)
        self.max_replacement_ratio = float(max_replacement_ratio)
        self.conditional_cache_path = conditional_cache_path

        self._host_last_request_at: dict[str, float] = {}
        self._robots_cache: dict[str, _HostRobotsPolicy] = {}
        self._breakers: dict[str, _HostBreakerState] = {}
        self._conditional_cache: dict[str, ConditionalCacheEntry] = {}
        self.breaker_events: list[BreakerEvent] = []
        self.sleep_history: list[float] = []

        self._transport = transport
        self._client: httpx.Client | None = None

        if self.conditional_cache_path is not None and self.conditional_cache_path.exists():
            self._load_conditional_cache(self.conditional_cache_path)

    def _ensure_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=httpx.Timeout(self.timeout_s),
                follow_redirects=False,
                headers={
                    "User-Agent": self.user_agent,
                    "Accept": (
                        "text/html,application/xhtml+xml,application/xml;q=0.9,"
                        "application/rss+xml,application/rdf+xml,application/json;q=0.9,*/*;q=0.8"
                    ),
                },
                transport=self._transport,
            )
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> PoliteHttpClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _load_conditional_cache(self, path: Path) -> None:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                for url, row in raw.items():
                    if isinstance(row, dict):
                        self._conditional_cache[str(url)] = ConditionalCacheEntry(
                            url=str(url),
                            etag=row.get("etag"),
                            last_modified=row.get("last_modified"),
                            body_text=str(row.get("body_text") or ""),
                            encoding=str(row.get("encoding") or "utf-8"),
                            headers=dict(row.get("headers") or {}),
                        )
        except Exception:
            pass

    def save_conditional_cache(self, path: Path | None = None) -> None:
        target = path or self.conditional_cache_path
        if target is None:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            url: {
                "body_text": entry.body_text,
                "encoding": entry.encoding,
                "etag": entry.etag,
                "headers": entry.headers,
                "last_modified": entry.last_modified,
                "url": entry.url,
            }
            for url, entry in sorted(self._conditional_cache.items())
            if entry.etag or entry.last_modified
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _parse_host(url: str) -> tuple[str, str]:
        parsed = urllib.parse.urlsplit(url)
        scheme = (parsed.scheme or "https").lower()
        host = (parsed.hostname or "").lower()
        if not host:
            raise ValueError(f"URL is missing a valid hostname: {url!r}")
        port = parsed.port
        netloc = f"{host}:{port}" if port and port not in (80, 443) else host
        return scheme, netloc

    def _sleep_for(self, seconds: float) -> None:
        if seconds <= 0:
            return
        self.sleep_history.append(seconds)
        self._sleep(seconds)

    def _pace_host(self, host: str, *, extra_crawl_delay: float | None = None) -> None:
        min_interval = self.default_min_interval_s
        if extra_crawl_delay is not None and extra_crawl_delay > min_interval:
            min_interval = extra_crawl_delay

        now = self._clock()
        last_at = self._host_last_request_at.get(host)
        if last_at is not None and min_interval > 0:
            elapsed = now - last_at
            low, high = self._jitter_range
            jitter = self._jitter_fn(low, high) if high > 0 else 0.0
            required_wait = (min_interval - elapsed) + jitter
            if required_wait > 0:
                self._sleep_for(required_wait)
        self._host_last_request_at[host] = self._clock()

    def _check_breaker(self, host: str, url: str) -> None:
        state = self._breakers.get(host)
        if state is None or state.open_until <= 0:
            return
        now = self._clock()
        if now < state.open_until:
            remaining = state.open_until - now
            raise CircuitBreakerOpenError(
                f"circuit breaker open for host {host!r} ({remaining:.1f}s remaining; "
                f"last_error={state.last_error}) while requesting {url!r}",
                url=url,
            )
        state.consecutive_failures = 0
        state.open_until = 0.0

    def _record_success(self, host: str) -> None:
        state = self._breakers.get(host)
        if state is not None:
            state.consecutive_failures = 0
            state.open_until = 0.0
            state.last_error = ""

    def _record_failure(self, host: str, error_summary: str) -> None:
        state = self._breakers.setdefault(host, _HostBreakerState())
        state.consecutive_failures += 1
        state.last_error = error_summary
        if state.consecutive_failures >= self.breaker_threshold:
            now = self._clock()
            state.open_until = now + self.breaker_cooldown_s
            event = BreakerEvent(
                host=host,
                consecutive_failures=state.consecutive_failures,
                opened_at=now,
                open_until=state.open_until,
                last_error=error_summary,
            )
            self.breaker_events.append(event)
            logger.warning(
                "Circuit breaker opened for host=%s after %d consecutive failures (cooldown=%.0fs): %s",
                host,
                state.consecutive_failures,
                self.breaker_cooldown_s,
                error_summary,
            )

    def _get_robots_policy(self, scheme: str, host: str) -> _HostRobotsPolicy:
        now = self._clock()
        cached = self._robots_cache.get(host)
        if cached is not None and (now - cached.fetched_at) < self.robots_ttl_s:
            return cached

        robots_url = f"{scheme}://{host}/robots.txt"
        self._check_breaker(host, robots_url)
        self._pace_host(host)

        client = self._ensure_client()
        try:
            resp = client.get(robots_url)
        except httpx.HTTPError as exc:
            # RFC 9309 §2.3.1.4: unreachable robots.txt must be treated as temporary disallow
            raise RobotsDisallowedError(
                f"robots.txt unreachable on host {host!r} ({exc}); assuming disallow per RFC 9309",
                url=robots_url,
            ) from exc

        status = resp.status_code
        if status in (401, 403):
            policy = _HostRobotsPolicy(
                fetched_at=now,
                parser=None,
                crawl_delay=None,
                allow_all=False,
                disallow_all=True,
            )
            self._robots_cache[host] = policy
            return policy

        if status == 429 or status >= 500:
            # RFC 9309 §2.3.1.4: server errors (5xx) or 429 on robots.txt must assume disallow
            # and must NOT be cached for 24h as allow-all.
            raise RobotsDisallowedError(
                f"robots.txt returned HTTP {status} on host {host!r}; assuming disallow per RFC 9309",
                url=robots_url,
                status_code=status,
            )

        if 400 <= status < 500:
            # RFC 9309 §2.3.1.3: 4xx (other than 401/403, e.g. 404/410) means allow-all
            policy = _HostRobotsPolicy(
                fetched_at=now,
                parser=None,
                crawl_delay=None,
                allow_all=True,
                disallow_all=False,
            )
            self._robots_cache[host] = policy
            return policy

        text = resp.content.decode("utf-8", errors="replace")
        rp = urllib.robotparser.RobotFileParser()
        rp.parse(text.splitlines())
        crawl_delay = _extract_crawl_delay(text, self.user_agent)
        policy = _HostRobotsPolicy(
            fetched_at=now,
            parser=rp,
            crawl_delay=crawl_delay,
            allow_all=False,
            disallow_all=False,
        )
        self._robots_cache[host] = policy
        return policy

    def _check_robots(self, url: str) -> float | None:
        scheme, host = self._parse_host(url)
        policy = self._get_robots_policy(scheme, host)
        if not policy.can_fetch(self.user_agent, url):
            reason_suffix = (
                " (robots.txt returned 401/403 disallow-all)" if policy.disallow_all else ""
            )
            raise RobotsDisallowedError(
                f"robots.txt disallowed {url!r} on host {host!r} for User-Agent {self.user_agent!r}{reason_suffix}",
                url=url,
                status_code=403 if policy.disallow_all else None,
                explicit_disallow=not policy.disallow_all,
            )
        return policy.crawl_delay

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        use_conditional: bool = True,
    ) -> FetchResult:
        """Perform a polite HTTP GET request with unconditional robots.txt check, rate limiting,
        conditional caching, retry/backoff, circuit breaking, and decode quality checks.
        """
        return self._request(
            "GET",
            url,
            headers=headers,
            json_body=None,
            use_conditional=use_conditional,
        )

    def post(
        self,
        url: str,
        *,
        json_body: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> FetchResult:
        """Perform a polite HTTP POST request (e.g. AniList GraphQL) with unconditional robots.txt check."""
        return self._request(
            "POST",
            url,
            headers=headers,
            json_body=json_body,
            use_conditional=False,
        )

    def _request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None,
        json_body: Mapping[str, object] | None,
        use_conditional: bool,
    ) -> FetchResult:
        current_url = url
        current_method = method

        for _redirect_hop in range(DEFAULT_MAX_REDIRECTS + 1):
            _, host = self._parse_host(current_url)
            self._check_breaker(host, current_url)
            crawl_delay = self._check_robots(current_url)

            req_headers: dict[str, str] = {"User-Agent": self.user_agent}
            if headers:
                req_headers.update(headers)

            cached_entry = (
                self._conditional_cache.get(current_url)
                if (use_conditional and current_method == "GET")
                else None
            )
            if cached_entry is not None:
                if cached_entry.etag and "If-None-Match" not in req_headers:
                    req_headers["If-None-Match"] = cached_entry.etag
                if cached_entry.last_modified and "If-Modified-Since" not in req_headers:
                    req_headers["If-Modified-Since"] = cached_entry.last_modified

            last_exc: Exception | None = None
            last_status: int | None = None
            resp: httpx.Response | None = None

            client = self._ensure_client()
            for attempt in range(self.max_attempts):
                self._check_breaker(host, current_url)
                self._pace_host(host, extra_crawl_delay=crawl_delay)

                try:
                    if current_method == "POST":
                        resp = client.post(current_url, headers=req_headers, json=json_body)
                    else:
                        resp = client.get(current_url, headers=req_headers)
                except httpx.HTTPError as exc:
                    last_exc = exc
                    last_status = None
                    if attempt + 1 < self.max_attempts:
                        backoff = self._backoff_for_attempt(attempt, retry_after_header=None)
                        self._sleep_for(backoff)
                        continue
                    self._record_failure(host, f"{type(exc).__name__}: {exc}")
                    raise IngestHttpError(
                        f"HTTP {current_method} {current_url!r} failed after {self.max_attempts} attempts: {exc}",
                        url=current_url,
                    ) from exc

                status = resp.status_code
                last_status = status

                # Handle redirects manually so every redirect target is checked against robots.txt and paced
                if status in (301, 302, 303, 307, 308):
                    location = resp.headers.get("location") or resp.headers.get("Location")
                    if not location:
                        raise IngestHttpError(
                            f"HTTP {status} redirect from {current_url!r} missing Location header",
                            url=current_url,
                            status_code=status,
                        )
                    self._record_success(host)
                    current_url = urllib.parse.urljoin(current_url, location.strip())
                    if status == 303:
                        current_method = "GET"
                        json_body = None
                    break

                if status == 304:
                    self._record_success(host)
                    if cached_entry is not None:
                        return FetchResult(
                            url=current_url,
                            status_code=304,
                            text=cached_entry.body_text,
                            content=cached_entry.body_text.encode(
                                cached_entry.encoding, errors="replace"
                            ),
                            encoding=cached_entry.encoding,
                            headers=dict(resp.headers),
                            not_modified=True,
                            from_cache=True,
                        )
                    return FetchResult(
                        url=current_url,
                        status_code=304,
                        text="",
                        content=b"",
                        encoding="utf-8",
                        headers=dict(resp.headers),
                        not_modified=True,
                        from_cache=False,
                    )

                if 200 <= status < 300:
                    content_type = resp.headers.get("content-type")
                    text, encoding, _ratio = decode_bytes_checked(
                        resp.content,
                        url=current_url,
                        content_type=content_type,
                        max_replacement_ratio=self.max_replacement_ratio,
                    )
                    self._record_success(host)
                    resp_headers = {k.lower(): v for k, v in resp.headers.items()}
                    etag = resp_headers.get("etag")
                    last_mod = resp_headers.get("last-modified")
                    if current_method == "GET" and (etag or last_mod):
                        self._conditional_cache[current_url] = ConditionalCacheEntry(
                            url=current_url,
                            etag=etag,
                            last_modified=last_mod,
                            body_text=text,
                            encoding=encoding,
                            headers=resp_headers,
                        )
                        if self.conditional_cache_path is not None:
                            self.save_conditional_cache()
                    return FetchResult(
                        url=current_url,
                        status_code=status,
                        text=text,
                        content=resp.content,
                        encoding=encoding,
                        headers=resp_headers,
                        not_modified=False,
                        from_cache=False,
                    )

                if status in (429, 503) or status >= 500:
                    retry_after = resp.headers.get("retry-after")
                    if attempt + 1 < self.max_attempts:
                        backoff = self._backoff_for_attempt(attempt, retry_after_header=retry_after)
                        self._sleep_for(backoff)
                        continue
                    self._record_failure(host, f"HTTP {status} from {current_url}")
                    raise IngestHttpError(
                        f"HTTP {current_method} {current_url!r} exhausted {self.max_attempts} attempts "
                        f"with status {status}",
                        url=current_url,
                        status_code=status,
                    )

                # Non-retryable 4xx (e.g. 400, 401, 403, 404):
                # Only count 401/403 (access block) toward host circuit breaker, not ordinary 404/410.
                if status in (401, 403):
                    self._record_failure(host, f"HTTP {status} from {current_url}")
                raise IngestHttpError(
                    f"HTTP {current_method} {current_url!r} failed with non-retryable status {status}",
                    url=current_url,
                    status_code=status,
                )
            else:
                raise IngestHttpError(
                    f"HTTP {current_method} {current_url!r} failed (last_status={last_status}, last_exc={last_exc})",
                    url=current_url,
                    status_code=last_status,
                )
            # If we broke out of the attempt loop due to a 3xx redirect, continue outer loop
            continue

        raise IngestHttpError(
            f"HTTP {method} {url!r} exceeded maximum redirects ({DEFAULT_MAX_REDIRECTS})",
            url=url,
        )

    def _backoff_for_attempt(self, attempt: int, *, retry_after_header: str | None) -> float:
        idx = min(attempt, len(self.backoff_schedule) - 1)
        scheduled = self.backoff_schedule[idx]
        parsed_ra = _parse_retry_after(retry_after_header, now=self._wall_clock())
        if parsed_ra is not None and parsed_ra > scheduled:
            return parsed_ra
        return scheduled


__all__ = [
    "DEFAULT_BACKOFF_SCHEDULE",
    "BreakerEvent",
    "CircuitBreakerOpenError",
    "ConditionalCacheEntry",
    "DecodeQualityError",
    "FetchResult",
    "IngestHttpError",
    "PoliteHttpClient",
    "RobotsDisallowedError",
    "decode_bytes_checked",
    "validate_user_agent",
]
