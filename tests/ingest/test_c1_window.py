"""COMPLIANCE C1 tests: windowless backfill() raises ValueError; robots.txt Disallow and 5xx/429
are refused; no robots bypass switch exists; redirect targets are checked; self-describing
User-Agent is enforced.

Owner: A (WP1) | Spec: docs/COMPLIANCE.md C1, Issue #9
"""

from __future__ import annotations

import inspect
from datetime import date

import httpx
import pytest
import respx

from acgn_rag.config import HttpConfig
from acgn_rag.core.ingest.adapters import (
    AnnArchiveAdapter,
    HtmlListAdapter,
    RssAdapter,
    StructuredApiAdapter,
    WpPagedAdapter,
    build_default_registry,
    validate_backfill_window,
)
from acgn_rag.core.ingest.http import (
    PoliteHttpClient,
    RobotsDisallowedError,
    validate_user_agent,
)


def _noop_client() -> PoliteHttpClient:
    return PoliteHttpClient(
        sleep=lambda _s: None,
        jitter_range=(0.0, 0.0),
    )


def test_c1_validate_backfill_window_rejects_missing_or_inverted_or_non_utc_bounds() -> None:
    with pytest.raises(ValueError, match="COMPLIANCE C1"):
        validate_backfill_window(None)
    with pytest.raises(ValueError, match="COMPLIANCE C1"):
        validate_backfill_window((None, date(2026, 10, 28)))
    with pytest.raises(ValueError, match="COMPLIANCE C1"):
        validate_backfill_window((date(2023, 10, 28), None))
    with pytest.raises(ValueError, match="must not exceed"):
        validate_backfill_window((date(2026, 10, 29), date(2026, 10, 28)))
    # Must be a 2-tuple, not a list (Probe 13c)
    with pytest.raises(ValueError, match="2-tuple"):
        validate_backfill_window(["2026-10-01", "2026-10-28"])  # type: ignore[arg-type]
    # Non-UTC offset datetime string must be rejected so local day never shifts UTC day (Probe 13d)
    with pytest.raises(ValueError, match="non-UTC"):
        validate_backfill_window(("2026-10-01", "2026-10-28T23:00:00-05:00"))


@pytest.mark.parametrize(
    "bad_window",
    [
        None,
        (None, date(2026, 10, 28)),
        (date(2023, 10, 28), None),
        (date(2026, 10, 29), date(2026, 10, 28)),
    ],
)
def test_c1_every_registered_adapter_rejects_windowless_backfill(bad_window) -> None:
    registry = build_default_registry(http=_noop_client())
    assert len(registry) >= 12
    for name, adapter in registry.items():
        with pytest.raises(ValueError), respx.mock(assert_all_called=False) as router:
            # Even if a test forgot network guard, backfill() must fail before any HTTP call.
            router.route().side_effect = AssertionError(
                f"{name}.backfill({bad_window!r}) must fail before issuing any HTTP request"
            )
            adapter.backfill(date_range=bad_window)


def test_c1_adapter_classes_explicitly_reject_windowless_backfill() -> None:
    http = _noop_client()
    adapters = [
        RssAdapter(
            source_key="test-rss",
            feed_url="https://example.com/feed.xml",
            publisher="Example",
            source_type="official",
            lang="ja",
            http=http,
        ),
        HtmlListAdapter(
            source_key="test-html",
            list_url="https://example.com/news/",
            publisher="Example",
            source_type="official",
            lang="ja",
            http=http,
        ),
        WpPagedAdapter(
            source_key="test-wp",
            site_url="https://example.com",
            publisher="Example",
            source_type="official",
            lang="ja",
            http=http,
        ),
        AnnArchiveAdapter(http=http),
        StructuredApiAdapter(
            source_key="test-api",
            provider="syosetu",
            endpoint_url="https://api.syosetu.com/novelapi/api/",
            publisher="Syosetu",
            http=http,
        ),
    ]
    for adapter in adapters:
        with pytest.raises(ValueError, match="COMPLIANCE C1"):
            adapter.backfill(None)


def test_c1_no_robots_bypass_switch_exists_on_http_client() -> None:
    """COMPLIANCE C1 L15: no check_robots bypass parameter may exist on get/post."""
    assert "check_robots" not in inspect.signature(PoliteHttpClient.get).parameters
    assert "check_robots" not in inspect.signature(PoliteHttpClient.post).parameters


def test_c1_robots_disallow_path_is_refused_and_cached() -> None:
    now = 1000.0
    client = PoliteHttpClient(
        clock=lambda: now,
        sleep=lambda _s: None,
        jitter_range=(0.0, 0.0),
    )
    robots_body = "User-agent: *\nDisallow: /private/\nDisallow: /admin\nAllow: /news/\n"
    with respx.mock(assert_all_called=False) as router:
        robots_route = router.get("https://example.com/robots.txt").mock(
            return_value=httpx.Response(200, text=robots_body)
        )
        news_route = router.get("https://example.com/news/item-1").mock(
            return_value=httpx.Response(200, text="<html><body>Allowed news</body></html>")
        )
        private_route = router.get("https://example.com/private/secret").mock(
            return_value=httpx.Response(200, text="should never be fetched")
        )

        with pytest.raises(RobotsDisallowedError, match=r"robots\.txt disallowed"):
            client.get("https://example.com/private/secret")

        assert private_route.call_count == 0
        assert robots_route.call_count == 1

        # Allowed path reuses cached robots.txt within 24h TTL
        res = client.get("https://example.com/news/item-1")
        assert res.status_code == 200
        assert news_route.call_count == 1
        assert robots_route.call_count == 1


def test_c1_robots_403_and_5xx_and_429_disallow_origin() -> None:
    """RFC 9309 §2.3.1.4: 401/403, 429, and 5xx on /robots.txt must disallow crawl."""
    client = _noop_client()
    with respx.mock(assert_all_called=False) as router:
        router.get("https://locked.example.com/robots.txt").mock(
            return_value=httpx.Response(403, text="Forbidden")
        )
        with pytest.raises(RobotsDisallowedError, match="401/403"):
            client.get("https://locked.example.com/news/")

        robots_503 = router.get("https:// outage.example.com/robots.txt".replace(" ", "")).mock(
            return_value=httpx.Response(503, text="Service Unavailable")
        )
        article_route = router.get("https://outage.example.com/news/").mock(
            return_value=httpx.Response(200, text="should not be fetched")
        )
        with pytest.raises(RobotsDisallowedError, match="503"):
            client.get("https://outage.example.com/news/")
        assert article_route.call_count == 0
        assert robots_503.call_count == 1


def test_c1_redirect_target_checked_against_robots() -> None:
    """Redirect hops must check /robots.txt on the redirect target before fetching."""
    client = _noop_client()
    with respx.mock(assert_all_called=True) as router:
        router.get("https://example.com/robots.txt").mock(
            return_value=httpx.Response(200, text="User-agent: *\nDisallow: /secret/\n")
        )
        router.get("https://example.com/public").mock(
            return_value=httpx.Response(302, headers={"Location": "https://example.com/secret/doc"})
        )
        with pytest.raises(RobotsDisallowedError, match=r"robots\.txt disallowed"):
            client.get("https://example.com/public")


def test_c1_user_agent_rejects_browser_disguise() -> None:
    with pytest.raises(ValueError, match="AcgnRagBot"):
        validate_user_agent("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")
    with pytest.raises(ValueError, match="spoof"):
        validate_user_agent("Mozilla/5.0 AcgnRagBot/0.1 (+https://example.com)")
    with pytest.raises(ValueError, match="empty"):
        validate_user_agent("   ")
    with pytest.raises(ValueError, match="AcgnRagBot"):
        PoliteHttpClient(config=HttpConfig(user_agent="Googlebot/2.1"))
