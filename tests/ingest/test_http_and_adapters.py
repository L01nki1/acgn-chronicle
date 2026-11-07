"""Offline unit tests for `core/ingest/http.py`, `ann_archive.py`, and the four adapter families.

Uses `respx` exclusively; zero network calls.
Owner: A (WP1) | Spec: Issue #9, Issue #14 V1, docs/INGEST_CONTRACT.md §5
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import httpx
import pytest
import respx

from acgn_rag.core.ingest.adapters import (
    BackfillNotSupportedError,
    EmptyBackfillWindowError,
    HtmlListAdapter,
    PaginationLoopError,
    RssAdapter,
    SelectorBrokenError,
    StructuredApiAdapter,
    WpPagedAdapter,
    ann_day_url,
    build_default_registry,
    parse_ann_day_html,
)
from acgn_rag.core.ingest.adapters.html_list import parse_html_list_page
from acgn_rag.core.ingest.http import (
    CircuitBreakerOpenError,
    DecodeQualityError,
    IngestHttpError,
    PoliteHttpClient,
    decode_bytes_checked,
)


class FakeClock:
    def __init__(self, initial: float = 100.0) -> None:
        self.now = initial
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def test_http_per_host_token_bucket_pacing_and_jitter() -> None:
    clock = FakeClock()
    client = PoliteHttpClient(
        clock=clock,
        sleep=clock.sleep,
        jitter_range=(0.1, 0.1),
        jitter_fn=lambda low, _high: low,
    )
    with respx.mock(assert_all_called=True) as router:
        router.get("https://example.org/robots.txt").mock(return_value=httpx.Response(404))
        router.get("https://example.org/a").mock(return_value=httpx.Response(200, text="ok-a"))
        router.get("https://example.org/b").mock(return_value=httpx.Response(200, text="ok-b"))

        client.get("https://example.org/a")
        client.get("https://example.org/b")

    # Request 1: robots.txt (0s wait), Request 2: /a (waits 2.5 + 0.1 = 2.6s), Request 3: /b (waits 2.6s)
    assert [round(s, 2) for s in clock.slept] == [2.6, 2.6]


def test_http_conditional_etag_and_304_not_modified() -> None:
    clock = FakeClock()
    client = PoliteHttpClient(clock=clock, sleep=clock.sleep, jitter_range=(0.0, 0.0))
    seen_if_none_match: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        inm = request.headers.get("If-None-Match")
        seen_if_none_match.append(inm)
        if inm == '"v1-etag"':
            return httpx.Response(304, headers={"ETag": '"v1-etag"'})
        return httpx.Response(
            200,
            text="<rss><channel><item><title>First</title><link>https://example.org/1</link>"
            "<pubDate>Wed, 28 Oct 2026 00:00:00 GMT</pubDate></item></channel></rss>",
            headers={"ETag": '"v1-etag"', "Last-Modified": "Wed, 28 Oct 2026 00:00:00 GMT"},
        )

    with respx.mock(assert_all_called=True) as router:
        router.get("https://example.org/robots.txt").mock(return_value=httpx.Response(404))
        router.get("https://example.org/feed.xml").mock(side_effect=handler)

        first = client.get("https://example.org/feed.xml")
        assert first.status_code == 200
        assert first.not_modified is False

        second = client.get("https://example.org/feed.xml")
        assert second.status_code == 304
        assert second.not_modified is True
        assert "First" in second.text
        assert seen_if_none_match == [None, '"v1-etag"']


def test_http_retry_backoff_1_2_4_and_retry_after_header() -> None:
    clock = FakeClock()
    client = PoliteHttpClient(clock=clock, sleep=clock.sleep, jitter_range=(0.0, 0.0))
    attempts = 0

    def flaky_handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "5"}, text="rate limited")
        if attempts == 2:
            return httpx.Response(503, text="service unavailable")
        if attempts == 3:
            return httpx.Response(502, text="bad gateway")
        return httpx.Response(200, text="recovered")

    with respx.mock(assert_all_called=True) as router:
        router.get("https://retry.example.org/robots.txt").mock(return_value=httpx.Response(404))
        router.get("https://retry.example.org/data").mock(side_effect=flaky_handler)

        res = client.get("https://retry.example.org/data")
        assert res.status_code == 200
        assert res.text == "recovered"
        assert attempts == 4
        # Retry 1 honours Retry-After=5 (> scheduled 1.0s); Retry 2 uses 2.0s; Retry 3 uses 4.0s
        assert 5.0 in clock.slept
        assert 2.0 in clock.slept
        assert 4.0 in clock.slept


def test_http_circuit_breaker_opens_after_three_failures_for_30_minutes_and_ignores_404() -> None:
    clock = FakeClock()
    client = PoliteHttpClient(
        clock=clock,
        sleep=clock.sleep,
        jitter_range=(0.0, 0.0),
        max_attempts=1,
        breaker_threshold=3,
        breaker_cooldown_s=1800.0,
    )
    with respx.mock(assert_all_called=True) as router:
        router.get("https://broken.example.org/robots.txt").mock(return_value=httpx.Response(404))
        missing = router.get("https://broken.example.org/missing").mock(
            return_value=httpx.Response(404, text="Not Found")
        )
        route = router.get("https://broken.example.org/news").mock(
            return_value=httpx.Response(403, text="Forbidden")
        )

        # 404 Not Found on individual URLs must not trip the host-wide circuit breaker
        for _ in range(4):
            with pytest.raises(IngestHttpError, match="404"):
                client.get("https://broken.example.org/missing")
        assert len(client.breaker_events) == 0
        assert missing.call_count == 4

        for _ in range(3):
            with pytest.raises(IngestHttpError, match="403"):
                client.get("https://broken.example.org/news")

        assert len(client.breaker_events) == 1
        assert client.breaker_events[0].host == "broken.example.org"
        assert client.breaker_events[0].consecutive_failures == 3
        assert route.call_count == 3

        # 4th call during cooldown fails immediately via CircuitBreakerOpenError without network
        with pytest.raises(CircuitBreakerOpenError, match="circuit breaker open"):
            client.get("https://broken.example.org/news")
        assert route.call_count == 3

        # Advance clock past 30-minute cooldown (1800s); half-open probe succeeds and resets breaker
        clock.now += 1801.0
        route.mock(return_value=httpx.Response(200, text="back online"))
        recovered = client.get("https://broken.example.org/news")
        assert recovered.status_code == 200
        assert route.call_count == 4


def test_http_charset_detection_mislabeled_latin1_and_decode_error_rate_guard() -> None:
    clock = FakeClock()
    client = PoliteHttpClient(clock=clock, sleep=clock.sleep, jitter_range=(0.0, 0.0))
    shift_jis_html = (
        '<html><head><meta charset="Shift_JIS"></head>'
        "<body>東映アニメーションの新作発表ニュース</body></html>"
    ).encode("shift_jis")
    corrupt_bytes = b"\xff\xfe\xfd" * 20 + b"abc"
    utf8_ja_bytes = "東映アニメーション最新ニュース".encode()

    # Probe 5: UTF-8 Japanese mislabeled as iso-8859-1 must decode as UTF-8, never mojibake
    decoded_text, encoding_used, ratio = decode_bytes_checked(
        utf8_ja_bytes,
        url="https://jp.example.org/mislabeled",
        content_type="text/html; charset=iso-8859-1",
    )
    assert decoded_text == "東映アニメーション最新ニュース"
    assert encoding_used == "utf-8"
    assert ratio == 0.0

    with respx.mock(assert_all_called=True) as router:
        router.get("https://jp.example.org/robots.txt").mock(return_value=httpx.Response(404))
        router.get("https://jp.example.org/sjis").mock(
            return_value=httpx.Response(
                200,
                content=shift_jis_html,
                headers={"Content-Type": "text/html"},
            )
        )
        router.get("https://jp.example.org/corrupt").mock(
            return_value=httpx.Response(
                200,
                content=corrupt_bytes,
                headers={"Content-Type": "text/html; charset=utf-8"},
            )
        )

        ok = client.get("https://jp.example.org/sjis")
        assert "東映アニメーション" in ok.text

        with pytest.raises(DecodeQualityError, match="decode replacement ratio"):
            client.get("https://jp.example.org/corrupt")


def test_ann_archive_v1_slash_url_same_day_filtering_and_self_pager_skip() -> None:
    target = date(2026, 10, 20)
    assert ann_day_url(target) == "https://www.animenewsnetwork.com/news/2026/10/20"

    sample_html = """
    <html><body>
      <div class="mainfeed">
        <div class="herald box">
          <h3><a href="/news/2026-10-20/frieren-season-2-reveals-key-visual/.216901">
            Frieren Season 2 Reveals Key Visual and Premiere Date
          </a></h3>
          <p>The official website announced that the broadcast starts in January.</p>
        </div>
        <div class="herald box">
          <h3><a href="/news/2026-10-20/capcom-reports-record-q2-sales/.216902">
            Capcom Reports Record Q2 Sales Across PC Catalog
          </a></h3>
        </div>
        <div class="pager">
          <a href="/news/2026/10/20?page=1">1</a>
          <a href="/news/2026/10/20?page=2">2</a>
        </div>
      </div>
      <aside class="sidebar-latest">
        <a href="/news/2026-10-21/sidebar-tomorrow-item/.216950">Tomorrow Sidebar Link</a>
        <a href="/news/2026-10-19/sidebar-yesterday-item/.216850">Yesterday Sidebar Link</a>
      </aside>
    </body></html>
    """
    parsed = parse_ann_day_html(sample_html, target)
    assert len(parsed.items) == 2
    assert parsed.dropped_other_day_links == 2
    # Self-link to ?page=1 must be ignored so page 1 is never re-queued (Probe 12)
    assert list(parsed.next_page_urls) == [
        "https://www.animenewsnetwork.com/news/2026/10/20?page=2"
    ]
    urls = [item.url for item in parsed.items]
    assert urls == [
        "https://www.animenewsnetwork.com/news/2026-10-20/capcom-reports-record-q2-sales/.216902",
        "https://www.animenewsnetwork.com/news/2026-10-20/frieren-season-2-reveals-key-visual/.216901",
    ]
    assert all(item.published_at == "2026-10-20T00:00:00Z" for item in parsed.items)
    # Herald blurb is included in body
    assert "broadcast starts in January" in parsed.items[1].body


def test_html_list_and_wp_paged_detect_duplicate_page_loops_and_broken_selectors() -> None:
    clock = FakeClock()
    http = PoliteHttpClient(clock=clock, sleep=clock.sleep, jitter_range=(0.0, 0.0))
    toei_like_html = """
    <html><body>
      <ul class="news-list">
        <li><span class="date">2026.10.28</span>
            <a href="/news/detail/2026102801">『ワンピース』特別編放送決定のお知らせ</a></li>
        <li><span class="date">2025.04.15</span>
            <a href="/news/detail/2025041501">新作劇場版アニメ制作決定</a></li>
      </ul>
    </body></html>
    """
    adapter = HtmlListAdapter(
        source_key="toei-news",
        list_url="https://www.toei-anim.co.jp/news/",
        publisher="Toei Animation",
        source_type="official",
        lang="ja",
        max_pages=2,
        raise_on_duplicate_page=True,
        cache_pages=False,
        http=http,
    )
    with respx.mock(assert_all_called=True) as router:
        router.get("https://www.toei-anim.co.jp/robots.txt").mock(return_value=httpx.Response(404))
        router.get(url__eq="https://www.toei-anim.co.jp/news/").mock(
            return_value=httpx.Response(200, text=toei_like_html)
        )
        router.get(url__eq="https://www.toei-anim.co.jp/news/?page=2").mock(
            return_value=httpx.Response(200, text=toei_like_html)
        )
        with pytest.raises(PaginationLoopError, match="duplicate content fingerprint"):
            adapter.backfill((date(2024, 1, 1), date(2026, 10, 28)))

    with respx.mock(assert_all_called=True) as router:
        router.get("https://www.toei-anim.co.jp/news/").mock(
            return_value=httpx.Response(
                200, text="<html><body><div>No links here</div></body></html>"
            )
        )
        with pytest.raises(SelectorBrokenError, match="matched 0 dated items"):
            adapter.discover()


def test_rss_adapter_corporate_tagging_and_rolling_feed_backfill_refusal() -> None:
    clock = FakeClock()
    http = PoliteHttpClient(clock=clock, sleep=clock.sleep, jitter_range=(0.0, 0.0))
    capcom_rss = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel>
      <item>
        <title>2026年3月期 第2四半期決算短信を発表</title>
        <link>https://www.capcom.co.jp/ir/news/20261027.html</link>
        <pubDate>Tue, 27 Oct 2026 06:00:00 +0000</pubDate>
        <description>連結業績および決算の概要をお知らせいたします。</description>
      </item>
      <item>
        <title>『モンスターハンターワイルズ』最新アップデート配信決定</title>
        <link>https://www.capcom.co.jp/game/news/20261028.html</link>
        <pubDate>Wed, 28 Oct 2026 06:00:00 +0000</pubDate>
        <description>新モンスターとイベントクエストを追加。</description>
      </item>
    </channel></rss>
    """
    capcom = RssAdapter(
        source_key="capcom-news",
        feed_url="https://www.capcom.co.jp/rss/news.xml",
        publisher="Capcom",
        source_type="official",
        lang="ja",
        source_tz="Asia/Tokyo",
        http=http,
    )
    with respx.mock(assert_all_called=True) as router:
        router.get("https://www.capcom.co.jp/robots.txt").mock(return_value=httpx.Response(404))
        router.get("https://www.capcom.co.jp/rss/news.xml").mock(
            return_value=httpx.Response(200, text=capcom_rss)
        )
        items = capcom.backfill((date(2026, 10, 1), date(2026, 10, 30)))
        assert len(items) == 2
        by_url = {item.url: item for item in items}
        assert (
            by_url["https://www.capcom.co.jp/ir/news/20261027.html"].ingest_category == "corporate"
        )
        assert by_url["https://www.capcom.co.jp/game/news/20261028.html"].ingest_category is None

        with pytest.raises(EmptyBackfillWindowError):
            capcom.backfill((date(2010, 1, 1), date(2010, 12, 31)))

    # Rolling RSS feeds in REGISTRY must have supports_backfill=False (SOURCES.md §3 L48-49, P1-9)
    reg = build_default_registry(http=http)
    assert reg["ann-news"].max_days_per_call is None  # type: ignore[attr-defined]
    for rolling_key in ("prtimes-acgn", "gcores-news", "chuapp-news", "yystv-news"):
        with pytest.raises(BackfillNotSupportedError, match="rolling feed"):
            reg[rolling_key].backfill((date(2024, 1, 1), date(2026, 10, 28)))


def test_structured_api_syosetu_metadata_only_and_bilibili_backfill_guard() -> None:
    clock = FakeClock()
    http = PoliteHttpClient(clock=clock, sleep=clock.sleep, jitter_range=(0.0, 0.0))
    syosetu_json = (
        '[{"allcount": 1}, '
        '{"title": "転生したらスライムだった件", "ncode": "N6316BN", "writer": "伏瀬", '
        '"general_firstup": "2025-02-20 01:15:39", "novelupdated_at": "2026-10-15 18:00:00", '
        '"keyword": "異世界 転生 ファンタジー", "general_all_no": 308}]'
    )
    syosetu = StructuredApiAdapter(
        source_key="syosetu-novelapi",
        provider="syosetu",
        endpoint_url="https://api.syosetu.com/novelapi/api/?out=json",
        publisher="Syosetu",
        source_type="metadata",
        lang="ja",
        source_tz="Asia/Tokyo",
        http=http,
    )
    bilibili_json = (
        '{"code": 0, "result": [{"date": "2026-10-28", "episodes": ['
        '{"episode_id": 8801, "season_id": 4401, "title": "葬送的芙莉莲", '
        '"pub_index": "第1话", "pub_time": "23:00", "pub_ts": 1793145600}]}]}'
    )
    bilibili = StructuredApiAdapter(
        source_key="bilibili-timeline",
        provider="bilibili",
        endpoint_url="https://api.bilibili.com/pgc/web/timeline",
        publisher="Bilibili",
        source_type="metadata",
        lang="zh",
        source_tz="Asia/Shanghai",
        http=http,
    )
    with respx.mock(assert_all_called=True) as router:
        router.get("https://api.syosetu.com/robots.txt").mock(return_value=httpx.Response(404))
        router.get("https://api.syosetu.com/novelapi/api/?out=json").mock(
            return_value=httpx.Response(200, text=syosetu_json)
        )
        router.get("https://api.bilibili.com/robots.txt").mock(return_value=httpx.Response(404))
        router.get("https://api.bilibili.com/pgc/web/timeline").mock(
            return_value=httpx.Response(200, text=bilibili_json)
        )

        novels = syosetu.backfill((date(2025, 1, 1), date(2026, 10, 30)))
        assert len(novels) == 1
        assert novels[0].url == "https://ncode.syosetu.com/n6316bn/"
        assert novels[0].external_media_refs == (("syosetu", "n6316bn"),)
        # Uses general_firstup (2025-02-20 JST -> 2025-02-19T16:15:39Z) rather than drifting novelupdated_at
        assert novels[0].published_at == "2025-02-19T16:15:39Z"

        bili_items = bilibili.discover()
        assert len(bili_items) == 1
        assert bili_items[0].url == "https://www.bilibili.com/bangumi/play/ep8801"

        with pytest.raises(BackfillNotSupportedError, match="rolling ±7d endpoint"):
            bilibili.backfill((date(2024, 1, 1), date(2024, 12, 31)))


def test_wp_paged_rest_fallback_to_rss() -> None:
    clock = FakeClock()
    http = PoliteHttpClient(clock=clock, sleep=clock.sleep, jitter_range=(0.0, 0.0))
    studio_rss = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel>
      <item>
        <title>TVアニメ新シリーズ制作決定</title>
        <link>https://www.mappa.co.jp/news/101</link>
        <pubDate>Sun, 25 Oct 2026 03:00:00 +0000</pubDate>
        <description>MAPPAによる最新アニメーション制作発表。</description>
      </item>
    </channel></rss>
    """
    wp = WpPagedAdapter(
        source_key="mappa-news",
        site_url="https://www.mappa.co.jp",
        publisher="MAPPA",
        source_type="official",
        lang="ja",
        mode="auto",
        per_page=50,
        max_pages=1,
        use_rest_date_filter=False,
        http=http,
    )
    with respx.mock(assert_all_called=True) as router:
        router.get("https://www.mappa.co.jp/robots.txt").mock(return_value=httpx.Response(404))
        router.get("https://www.mappa.co.jp/wp-json/wp/v2/posts?per_page=50&page=1").mock(
            return_value=httpx.Response(404, text="REST disabled")
        )
        router.get("https://www.mappa.co.jp/feed/").mock(
            return_value=httpx.Response(200, text=studio_rss)
        )
        items = wp.backfill((date(2026, 10, 1), date(2026, 10, 30)))
        assert len(items) == 1
        assert items[0].url == "https://www.mappa.co.jp/news/101"
        assert "MAPPAによる最新アニメーション制作発表" in items[0].body


def test_no_synthetic_body_padding_ann_time_extraction_and_fractional_since() -> None:
    """R1, P2-2, P2-8: ANN listing time `( 21:24 )` is parsed into `published_at` (EDT -> UTC)
    and never injected into `body`; HTML list / WP REST keep `body == title` when no excerpt
    exists; `discover(since=...)` handles fractional-second timestamps chronologically.
    """
    ann_html = """
    <html><body><div id="content-zone"><ul class="article-list">
      <li><a href="/news/2026-10-28/oshi-no-ko-season-3/.230001">Oshi no Ko Season 3 Announced</a> ( 21:24 )</li>
    </ul></div></body></html>
    """
    parsed_ann = parse_ann_day_html(ann_html, date(2026, 10, 28))
    assert len(parsed_ann.items) == 1
    assert parsed_ann.items[0].title == "Oshi no Ko Season 3 Announced"
    # R1: body must not contain synthetic "( 21:24 )"
    assert parsed_ann.items[0].body == "Oshi no Ko Season 3 Announced"
    # P2-8: 2026-10-28 21:24 America/New_York (EDT, -04:00) -> 2026-10-29T01:24:00Z
    assert parsed_ann.items[0].published_at == "2026-10-29T01:24:00Z"

    # R1: HTML list row with only date + title keeps body == title
    toei_html = """
    <html><body><ul>
      <li>2026.10.28 <a href="https://www.toei-anim.co.jp/news/2026102801.html">新作アニメ発表のお知らせ</a></li>
    </ul></body></html>
    """
    list_items, _fp = parse_html_list_page(
        toei_html,
        base_url="https://www.toei-anim.co.jp/news/",
        source_key="toei-news",
        publisher="Toei Animation",
        source_type="official",
        lang="ja",
    )
    assert len(list_items) == 1
    assert list_items[0].body == list_items[0].title

    # P2-2: fractional-second timestamp at 00:00:00.5Z must sort after 00:00:00Z in discover(since)
    clock = FakeClock()
    http = PoliteHttpClient(clock=clock, sleep=clock.sleep, jitter_range=(0.0, 0.0))
    frac_rss = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel>
      <item>
        <title>Fractional Second Item</title>
        <link>https://example.org/news/frac-1</link>
        <pubDate>2026-10-28T00:00:00.5Z</pubDate>
        <description>Fractional second body</description>
      </item>
    </channel></rss>
    """
    rss = RssAdapter(
        source_key="frac-rss",
        feed_url="https://example.org/feed.xml",
        publisher="Example",
        source_type="media",
        lang="en",
        source_tz="UTC",
        http=http,
    )
    with respx.mock(assert_all_called=True) as router:
        router.get("https://example.org/robots.txt").mock(return_value=httpx.Response(404))
        router.get("https://example.org/feed.xml").mock(
            return_value=httpx.Response(200, text=frac_rss)
        )
        found = rss.discover(since="2026-10-28T00:00:00Z")
        assert len(found) == 1
        assert found[0].published_at == "2026-10-28T00:00:00.5Z"


def test_round3_n1_warm_304_cache_preserves_backfill_rolling_tails_and_n2_capcom_title_promotion(
    tmp_path,
) -> None:
    """Round-3 N1 & N2 regressions:
    - N1: A second backfill run with a warm persistent ETag cache must still return rolling-feed
      rows (via `use_conditional=False`), while incremental `discover()` reports `last_not_modified=True`.
    - N2: Capcom's `プレスリリース YYYY年M月期` boilerplate title is replaced by the first headline line
      of `<description>`, and game news without IR terms stays `ingest_category=None`.
    """
    import importlib.util
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("backfill", repo_root / "scripts" / "backfill.py")
    assert spec is not None and spec.loader is not None
    backfill_mod = importlib.util.module_from_spec(spec)
    sys.modules["backfill"] = backfill_mod
    spec.loader.exec_module(backfill_mod)

    cache_file = tmp_path / "http_conditional_cache.json"
    gcores_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel>
      <item>
        <title>Gcores Game News 1</title>
        <link>https://www.gcores.com/articles/1001</link>
        <pubDate>2026-11-04T10:00:00Z</pubDate>
        <description>Game news article body</description>
      </item>
    </channel></rss>
    """

    def _gcores_handler(request: httpx.Request) -> httpx.Response:
        if request.headers.get("If-None-Match") == '"gcores-etag-1"':
            return httpx.Response(304, headers={"ETag": '"gcores-etag-1"'})
        return httpx.Response(200, text=gcores_xml, headers={"ETag": '"gcores-etag-1"'})

    with respx.mock(assert_all_called=False) as router:
        router.get("https://www.gcores.com/robots.txt").mock(return_value=httpx.Response(404))
        router.get("https://www.gcores.com/rss").mock(side_effect=_gcores_handler)

        # Run 1: warm the ETag cache via incremental discover()
        clock1 = FakeClock()
        http1 = PoliteHttpClient(
            clock=clock1,
            sleep=clock1.sleep,
            jitter_range=(0.0, 0.0),
            conditional_cache_path=cache_file,
        )
        rss1 = RssAdapter(
            source_key="gcores-news",
            feed_url="https://www.gcores.com/rss",
            publisher="Gcores",
            source_type="media",
            lang="zh",
            source_tz="Asia/Shanghai",
            http=http1,
            supports_backfill=False,
        )
        disc1 = rss1.discover(since="2026-11-01")
        assert len(disc1) == 1
        assert rss1.last_not_modified is False
        assert cache_file.exists()

        # Incremental discover() with warm cache receives 304 and sets last_not_modified=True
        http2 = PoliteHttpClient(
            clock=clock1,
            sleep=clock1.sleep,
            jitter_range=(0.0, 0.0),
            conditional_cache_path=cache_file,
        )
        rss2 = RssAdapter(
            source_key="gcores-news",
            feed_url="https://www.gcores.com/rss",
            publisher="Gcores",
            source_type="media",
            lang="zh",
            source_tz="Asia/Shanghai",
            http=http2,
            supports_backfill=False,
            cache_feed=False,
        )
        disc2 = rss2.discover(since="2026-11-01")
        assert disc2 == []
        assert rss2.last_not_modified is True

        # Backfill run with warm cache must fetch unconditionally and return the same 1 row!
        items_bf, summary_bf = backfill_mod.crawl_backfill_window(
            "2023-11-05",
            "2026-11-05",
            registry={"gcores-news": rss2},
        )
        assert len(items_bf) == 1
        assert summary_bf["sources"]["gcores-news"]["status"] == "ok"
        assert summary_bf["sources"]["gcores-news"]["count"] == 1

    # N2: Capcom RSS boilerplate title promotion + corporate vs game classification
    capcom_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel>
      <item>
        <title>プレスリリース 2027年3月期</title>
        <link>https://www.capcom.co.jp/ir/news/html/261028.html?utm_source=capcom_top</link>
        <pubDate>2026-10-28T09:00:00+09:00</pubDate>
        <description>映画『ストリートファイター/ザ・ムービー』が、全米週間興行収入ランキング1位を獲得!
~ 映画化を通じ、IPのグローバルな認知拡大と価値向上に寄与 ~</description>
      </item>
      <item>
        <title>プレスリリース 2027年3月期</title>
        <link>https://www.capcom.co.jp/ir/news/html/261027.html?utm_source=capcom_top</link>
        <pubDate>2026-10-27T09:00:00+09:00</pubDate>
        <description>2027年3月期第2四半期(中間期)決算の連結業績は、
前年同期比で増収増益、通期計画に対し順調に進捗</description>
      </item>
    </channel></rss>
    """
    clock_c = FakeClock()
    http_c = PoliteHttpClient(clock=clock_c, sleep=clock_c.sleep, jitter_range=(0.0, 0.0))
    capcom_rss = RssAdapter(
        source_key="capcom-news",
        feed_url="https://www.capcom.co.jp/rss/news.xml",
        publisher="Capcom",
        source_type="official",
        lang="ja",
        source_tz="Asia/Tokyo",
        http=http_c,
        supports_backfill=True,
    )
    with respx.mock(assert_all_called=True) as router:
        router.get("https://www.capcom.co.jp/robots.txt").mock(return_value=httpx.Response(404))
        router.get("https://www.capcom.co.jp/rss/news.xml").mock(
            return_value=httpx.Response(200, text=capcom_xml)
        )
        cap_items = capcom_rss.backfill(("2026-10-01", "2026-10-28"))
        assert len(cap_items) == 2
        sf_item = next(it for it in cap_items if "261028" in it.url)
        ir_item = next(it for it in cap_items if "261027" in it.url)
        assert (
            sf_item.title
            == "映画『ストリートファイター/ザ・ムービー』が、全米週間興行収入ランキング1位を獲得!"
        )
        assert sf_item.ingest_category is None
        assert (
            ir_item.title
            == "2027年3月期第2四半期(中間期)決算の連結業績は、前年同期比で増収増益、通期計画に対し順調に進捗"
        )
        assert ir_item.ingest_category == "corporate"


def test_respect_robots_false_rejected_and_conditional_cache_body_preserved(
    tmp_path: Path,
) -> None:
    """Enforce HttpConfig.respect_robots=True (COMPLIANCE C1) and verify 304 after restart
    preserves cached body_text (E P3).
    """
    from acgn_rag.config import HttpConfig

    with pytest.raises(ValueError, match=r"respect_robots=False.*forbidden"):
        PoliteHttpClient(config=HttpConfig(respect_robots=False))

    cond_path = tmp_path / "cond_cache.json"
    clock = FakeClock()
    http1 = PoliteHttpClient(
        clock=clock,
        sleep=clock.sleep,
        jitter_range=(0.0, 0.0),
        conditional_cache_path=cond_path,
    )
    with respx.mock(assert_all_called=True) as router:
        router.get("https://example.org/robots.txt").mock(return_value=httpx.Response(404))
        router.get("https://example.org/feed.xml").mock(
            side_effect=[
                httpx.Response(
                    200,
                    text="<rss>cached body</rss>",
                    headers={"ETag": '"etag-v9"', "Last-Modified": "Wed, 05 Nov 2026 00:00:00 GMT"},
                ),
                httpx.Response(304),
            ]
        )
        res1 = http1.get("https://example.org/feed.xml", use_conditional=True)
        assert res1.not_modified is False
        assert res1.text == "<rss>cached body</rss>"

        http2 = PoliteHttpClient(
            clock=clock,
            sleep=clock.sleep,
            jitter_range=(0.0, 0.0),
            conditional_cache_path=cond_path,
        )
        res2 = http2.get("https://example.org/feed.xml", use_conditional=True)
        assert res2.not_modified is True
        assert res2.text == "<rss>cached body</rss>"


def test_wp_paged_rest_server_bounds_100_to_50_fallback_canonical_host_and_page_cap() -> None:
    """Verify WpPagedAdapter server-side after/before query params, categories filter,
    HTTP 400 per_page=100 -> 50 fallback, canonical_host rewrite, content:encoded RSS priority,
    and last_page_cap_hit detection (A2, A10, D P2-6).
    """
    import json

    clock = FakeClock()
    http = PoliteHttpClient(clock=clock, sleep=clock.sleep, jitter_range=(0.0, 0.0))
    wp = WpPagedAdapter(
        source_key="nijimen-news",
        site_url="https://nijimen.kusuguru.co.jp",
        publisher="にじめん",
        source_type="media",
        lang="ja",
        mode="rest",
        per_page=100,
        max_pages=2,
        categories=(116583, 116584),
        canonical_host="nijimen.kusuguru.co.jp",
        http=http,
    )
    page1_json = json.dumps(
        [
            {
                "link": "https://nijimen.net/topics/649001",
                "title": {"rendered": "アニメ新情報公開 - にじめん"},
                "content": {"rendered": "<p>" + ("詳報テキスト" * 60) + "</p>"},
                "excerpt": {"rendered": "<p>抜粋</p>"},
                "date_gmt": "2026-10-20T05:00:00",
            }
        ]
    )
    page2_json = json.dumps(
        [
            {
                "link": "https://nijimen.net/topics/649002",
                "title": {"rendered": "アニメ続報公開"},
                "content": {"rendered": "<p>" + ("続報テキスト" * 60) + "</p>"},
                "excerpt": {"rendered": "<p>抜粋2</p>"},
                "date_gmt": "2026-10-15T05:00:00",
            }
        ]
    )
    with respx.mock(assert_all_called=True) as router:
        router.get("https://nijimen.kusuguru.co.jp/robots.txt").mock(
            return_value=httpx.Response(404)
        )
        router.get(
            "https://nijimen.kusuguru.co.jp/wp-json/wp/v2/posts"
            "?per_page=100&page=1&categories=116583,116584"
            "&after=2023-10-28T00:00:00&before=2026-10-28T23:59:59"
        ).mock(return_value=httpx.Response(400, text="per_page max 50"))
        router.get(
            "https://nijimen.kusuguru.co.jp/wp-json/wp/v2/posts"
            "?per_page=50&page=1&categories=116583,116584"
            "&after=2023-10-28T00:00:00&before=2026-10-28T23:59:59"
        ).mock(return_value=httpx.Response(200, text=page1_json))
        router.get(
            "https://nijimen.kusuguru.co.jp/wp-json/wp/v2/posts"
            "?per_page=50&page=2&categories=116583,116584"
            "&after=2023-10-28T00:00:00&before=2026-10-28T23:59:59"
        ).mock(return_value=httpx.Response(200, text=page2_json))

        items = wp.backfill((date(2023, 10, 28), date(2026, 10, 28)))
        assert len(items) == 2
        assert wp.last_effective_per_page == 50
        assert wp.last_page_cap_hit is True
        assert wp.last_reached_window_start is False
        assert items[0].url == "https://nijimen.kusuguru.co.jp/topics/649001"
        assert items[0].title == "アニメ新情報公開"

    # Also verify _parse_rss_page prefers content:encoded over short description (D P2-6)
    rss_with_content = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/"><channel>
      <item>
        <title>スタジオ新作発表</title>
        <link>https://www.mappa.co.jp/news/999</link>
        <pubDate>Sun, 25 Oct 2026 03:00:00 +0000</pubDate>
        <description>短い要約</description>
        <content:encoded><![CDATA[<p>これは全文のプレスリリース本文です。制作スタッフとキャストの詳細情報を含みます。</p>]]></content:encoded>
      </item>
    </channel></rss>
    """
    parsed_rss_items = wp._parse_rss_page(rss_with_content, "https://www.mappa.co.jp/feed/")
    assert len(parsed_rss_items) == 1
    assert "これは全文のプレスリリース本文です" in parsed_rss_items[0].body


def test_rss_duplicate_url_replaces_boilerplate_and_ann_recent_day_cache(tmp_path: Path) -> None:
    """Verify RssAdapter replaces a raw fiscal boilerplate title with an informative duplicate entry
    (E P3, C4), and AnnArchiveAdapter skips disk caching for today/yesterday UTC and tracks
    last_truncated_days (B1, E P1-2).
    """
    clock = FakeClock()
    http = PoliteHttpClient(clock=clock, sleep=clock.sleep, jitter_range=(0.0, 0.0))
    dup_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel>
      <item>
        <title>プレスリリース 2027年3月期</title>
        <link>https://www.capcom.co.jp/ir/news/html/261101.html</link>
        <pubDate>2026-11-01T09:00:00+09:00</pubDate>
        <description>~ サブタイトルのみ ~</description>
      </item>
      <item>
        <title>『モンスターハンターワイルズ』全世界販売本数1000万本突破</title>
        <link>https://www.capcom.co.jp/ir/news/html/261101.html</link>
        <pubDate>2026-11-01T09:00:00+09:00</pubDate>
        <description>カプコンの最新作が大ヒットを記録しました。</description>
      </item>
    </channel></rss>
    """
    rss = RssAdapter(
        source_key="capcom-news",
        feed_url="https://www.capcom.co.jp/rss/news.xml",
        publisher="Capcom",
        source_type="official",
        lang="ja",
        source_tz="Asia/Tokyo",
        http=http,
    )
    with respx.mock(assert_all_called=True) as router:
        router.get("https://www.capcom.co.jp/robots.txt").mock(return_value=httpx.Response(404))
        router.get("https://www.capcom.co.jp/rss/news.xml").mock(
            return_value=httpx.Response(200, text=dup_xml)
        )
        items = rss.backfill(("2026-10-01", "2026-11-05"))
        assert len(items) == 1
        assert rss.last_duplicate_url_entries == 1
        assert items[0].title == "『モンスターハンターワイルズ』全世界販売本数1000万本突破"

    # AnnArchiveAdapter: yesterday UTC is not written to disk cache when cache_recent_days=False,
    # and last_truncated_days increments when next_page_urls exceeds max_pages_per_day - 1
    from datetime import UTC, datetime, timedelta

    from acgn_rag.core.ingest.ann_archive import AnnArchiveAdapter

    today_utc = datetime.now(tz=UTC).date()
    yesterday_utc = today_utc - timedelta(days=1)
    y_slash = yesterday_utc.strftime("%Y/%m/%d")
    y_iso = yesterday_utc.isoformat()
    ann_html = f"""
    <html><body><div id="content-zone"><ul class="article-list">
      <li><a href="/news/{y_iso}/test-story/.300001">Test Story Headline</a> ( 10:00 )</li>
    </ul>
    <a href="/news/{y_slash}?page=2" rel="next">Page 2</a>
    <a href="/news/{y_slash}?page=3" rel="next">Page 3</a>
    </div></body></html>
    """
    ann_cache_dir = tmp_path / "ann_cache"
    ann = AnnArchiveAdapter(
        http=http,
        cache_dir=ann_cache_dir,
        max_pages_per_day=1,
        cache_recent_days=False,
    )
    with respx.mock(assert_all_called=True) as router:
        router.get("https://www.animenewsnetwork.com/robots.txt").mock(
            return_value=httpx.Response(404)
        )
        router.get(f"https://www.animenewsnetwork.com/news/{y_slash}").mock(
            return_value=httpx.Response(200, text=ann_html)
        )
        ann_items = ann.backfill((yesterday_utc, yesterday_utc))
        assert len(ann_items) == 1
        assert ann.last_truncated_days == 1
        assert ann.last_days_with_items == 1
        assert ann.last_empty_days == 0
        assert not (ann_cache_dir / f"{y_iso}.json").exists()


def test_parse_ann_article_html_and_offline_article_cache_enrichment(tmp_path: Path) -> None:
    """Verify parse_ann_article_html extracts #page_header, #page-title small time[datetime],
    and div.KonaBody (raising SelectorBrokenError on empty body), and AnnArchiveAdapter enriches
    archive days from an offline article cache (.zip or .json).
    """
    import zipfile

    from acgn_rag.core.ingest.ann_archive import (
        AnnArchiveAdapter,
        SelectorBrokenError,
        parse_ann_article_html,
    )

    article_html = """
    <html><body>
      <div id="page-title">
        <h1 id="page_header"><div class="sub-title">News</div>Frieren Season 2 Premieres in January 2026</h1>
        <small>posted on <time datetime="2025-10-20T10:30:00-04:00">2025-10-20</time></small>
      </div>
      <div class="KonaBody">
        <p>The official website for the Frieren: Beyond Journey's End television anime announced on Monday that the second season will premiere in January 2026 on Nippon TV's Friday Anime Night block.</p>
        <p>Madhouse returns to animate the new season with Keiichiro Saito supervising the production and Evan Call composing the music.</p>
      </div>
    </body></html>
    """
    parsed = parse_ann_article_html(
        article_html,
        url="https://www.animenewsnetwork.com/news/2025-10-20/frieren-s2/.200001",
    )
    assert parsed.title == "Frieren Season 2 Premieres in January 2026"
    assert parsed.published_at == "2025-10-20T14:30:00Z"
    assert "Madhouse returns to animate" in parsed.body
    assert "\n" in parsed.body

    with pytest.raises(SelectorBrokenError, match=r"Empty div\.KonaBody"):
        parse_ann_article_html(
            "<html><body><h1 id='page_header'>Title</h1><div class='KonaBody'>   </div></body></html>"
        )

    # Create a synthetic ANN article cache zip and verify AnnArchiveAdapter enriches day items
    zip_path = tmp_path / "ann-cache.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(
            "articles_index.csv",
            "url,published_date,cache_file\n"
            "https://www.animenewsnetwork.com/news/2025-10-20/frieren-s2/.200001,2025-10-20,article-1.html\n"
            "https://www.animenewsnetwork.com/daily-briefs/2025-10-20/extra-brief/.200002,2025-10-20,article-2.html\n",
        )
        zf.writestr("html/article-1.html", article_html)
        zf.writestr(
            "html/article-2.html",
            "<html><body><h1 id='page_header'><div class='sub-title'>News</div>Extra Daily Brief</h1>"
            "<div class='KonaBody'><p>Paragraph one of extra daily brief body text.</p>"
            "<p>Paragraph two of extra daily brief body text.</p></div></body></html>",
        )

    day_html = """
    <html><body><div id="content-zone"><ul class="article-list">
      <li><a href="/news/2025-10-20/frieren-s2/.200001">Frieren Season 2 Premieres in January 2026</a> ( 10:30 )</li>
    </ul></div></body></html>
    """
    clock = FakeClock()
    http = PoliteHttpClient(clock=clock, sleep=clock.sleep, jitter_range=(0.0, 0.0))
    ann = AnnArchiveAdapter(http=http, article_cache_path=zip_path)
    with respx.mock(assert_all_called=True) as router:
        router.get("https://www.animenewsnetwork.com/robots.txt").mock(
            return_value=httpx.Response(404)
        )
        router.get("https://www.animenewsnetwork.com/news/2025/10/20").mock(
            return_value=httpx.Response(200, text=day_html)
        )
        items = ann.backfill(("2025-10-20", "2025-10-20"))
        assert len(items) == 2
        assert ann.last_cached_article_bodies_applied == 2
        by_url = {it.url: it for it in items}
        assert (
            "Madhouse returns to animate"
            in by_url["https://www.animenewsnetwork.com/news/2025-10-20/frieren-s2/.200001"].body
        )
        assert (
            "Paragraph two of extra daily brief"
            in by_url[
                "https://www.animenewsnetwork.com/daily-briefs/2025-10-20/extra-brief/.200002"
            ].body
        )
