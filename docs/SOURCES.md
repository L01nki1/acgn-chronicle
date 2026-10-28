# SOURCES — what we crawl, how, and what "enough" means

> Owner: A. Everyone else only needs §1 and §6.
> This is the operational subset of the source inventory: the endpoints that matter for W1/W2, the ban list so
> nobody re-walks a dead end, and the acceptance thresholds. Every endpoint here was hit for real on
> 2026-10-27/28 (50+ endpoints, bot UA and browser UA both tested, 10 s timeout).

## 1. Four layers, three of which enter the corpus

| Layer | Content | Use | In corpus? |
|---|---|---|---|
| **L0** first-party | publishers, studios, official press releases | fact fields, ground truth, strongest citations | yes |
| **L1** structured | AniList / Bangumi / Bilibili / syosetu / ANN Encyclopedia | entity linking, fact checks, type-B QA lookups | metadata only, not news flow |
| **L2** media | ANN, Gcores, Chuapp, GameRes, AUTOMATON, 4Gamer | long-form reporting, cross-source synthesis | yes |
| **L3** community | Reddit, Mastodon | heat signal for the daily board only | **no** — `SourceType` has no `community` value yet |

## 2. Four access modes (RSS is no longer the norm, they are not interchangeable)

| Mode | Reality | Backfill |
|---|---|---|
| RSS / Atom | the feed is a rolling window | usually 7–8 days only |
| WordPress REST probe | `/wp-json/wp/v2/posts` first, else feed + `?paged=N` | deep, years |
| HTML list page | one selector config per site | depth = that page, no generic paging |
| Structured API | JSON / XML | by query |

Three facts learned the hard way:
1. **RSS is a rolling window.** ANN `all/rss.xml` covers 7–8 days. History only comes from per-day archives or paging, which is why every adapter has two modes.
2. **WordPress sites are three-state, not one.** REST open (MAPPA, Animate, AUTOMATON) → full-text feed (Dengeki, 209 KB) → title-only feed (Animate, 1.7 KB, needs the detail page).
3. **Blocking is per endpoint, not per site.** ufotable's homepage is 200 while `/news/` is 403; Dengeki's RSS is 200 while `wp-json` is 403.

## 3. W1 sources (critical path — scope frozen)

| Source | Endpoint | Mode | Reality | Role |
|---|---|---|---|---|
| **Toei Animation** ⭐ | `https://www.toei-anim.co.jp/news/` | HTML list | one page carries 322 dated entries back to 2008-10 — **one request ≈ 18 years** | highest yield per request |
| **Capcom** | `https://www.capcom.co.jp/rss/news.xml` | RSS | full archive, 7,293 items since 2014-03; titles are templated ⇒ must LLM-extract fields | game side, bulk volume |
| **MAPPA / BONES / P.A.WORKS** | `https://mappa.co.jp/feed/` etc. + `?paged=N` | WP / RSS | 5 pages ≈ 1.5–2.5 years; MAPPA's REST works, prefer it | studio-side first-party |
| **PR TIMES** | `https://prtimes.jp/index.rdf` | RSS | no usable category channel (all 404) ⇒ keyword filter on アニメ / マンガ / ゲーム / 声優 / 配信 | press releases; noisy without the filter |
| **ANN** | RSS `all/rss.xml` + `https://www.animenewsnetwork.com/news/YYYY-MM-DD` | RSS + per-day | ~31 items/day; per-day URL is addressable ⇒ **90 days with no gaps is the W1 hard target** | main L2 corpus |
| **Gcores 机核** | `https://www.gcores.com/rss` | RSS | ~6.7/day, median 565 chars | Chinese media |
| **Chuapp 触乐** | `http://www.chuapp.com/feed` | RSS | ~2.1/day, median 3,846 chars — the long-document workhorse | Chinese media |
| **GameRes 游研社** | `https://www.yystv.cn/rss/feed` | RSS | ~2.4/day, full text | Chinese media |
| **AniList** | `https://graphql.anilist.co` (POST) | API | 90 req/min, 30 when degraded | entity IDs, type-B QA |
| **Bangumi 番组** | `https://api.bgm.tv/v0/` | API | loose but undocumented ⇒ treat as 1 req/2 s | Chinese-language metadata |
| **Bilibili schedule** | `https://api.bilibili.com/pgc/web/timeline?types=1` | API | no key; 1 req/5 s | episode table for the daily digest |
| **syosetu** | `https://api.syosetu.com/novelapi/api/?out=json` | API | 1.26M novels, **metadata only** | light-novel coverage |

Only ANN supports both `discover()` and `backfill()`. Capcom's backfill is "parse the whole 7,293-item feed";
Toei's is "one request". **Do not invent a backfill for sources that cannot support it.**

## 4. W2 expansion pool (does not block W1)

Dengeki (`dengeki.com/feed`, full-text, the key light-novel increment) · Animate (`wp-json`, 100/page) ·
SEGA (`/topics/`) · Atlus (`/feed/`) · PlayStation / Xbox / Steam news feeds · Bandai Namco (`rss/news.xml`, Atom
syntax) · Horipro (`/feed`, REST explicitly disabled) · CloverWorks (`/news/`) · Koei Tecmo (`/news/YYYY/`, year
paths = natural deep backfill) · Bushiroad (`/news/`) · Falcom (`/news`) · Marvelous (`/news/`) · Kyoto Animation
(`/information/`) · Ichijinsha / Shogakukan / Akita Shoten (`/news/`) · KADOKAWA (`/topics/`) · Shuueisha / WIT
(feed, 1 item/page — weak) · AUTOMATON (`/feed/` + REST) · 4Gamer (`rss/index.xml`) · GamesIndustry.biz (`/feed`)
· Anime!Anime! (server-rendered, no RSS) · Mantan Web · natalie.mu (405 on both UAs — **one attack attempt only,
drop it if it fails**) · ANN Encyclopedia API · Kitsu · VNDB kana · MangaUpdates.

W2 priority follows the coverage gap: light novel and manga first (they are the thinnest categories).

## 5. Ban list — do not re-walk these

| Source | Reason |
|---|---|
| `api.syosetu.com/allnovel.Atom` | 100-item cap, minute-level amateur chapter updates: floods the corpus, breaks dedup, distorts hit rate. Use `novelapi` metadata only |
| niconico | every path 403/404, no unauthenticated channel |
| X / Twitter | 429 / 451 / API paywall |
| 游民星空, indienova, ACG批评, pixivision, Famitsu, Anime Times | 404 / returns HTML not a feed / no feed |
| RSSHub | needs cookies, keep it out of the main path |
| Jikan | timeouts; Kitsu covers the need |
| Kodansha, Hakusensha, Houbunsha, Futabasha, Jump+, Madhouse, Production I.G, Square Enix, Nintendo, FromSoftware, Cygames, Kakuyomu, Level-5, bookwalker, Lantis | measured 403 / 523 / connection failure / path mismatch |
| Bandai Namco Filmworks, Studio Trigger | two retries, connection dies at TLS handshake — not solvable with `requests`, and should not be |
| ufotable | homepage 200 but `/news/` 403; WPGraphQL headless app |
| Dengeki Online | Next.js SPA, old feed 404 |

## 6. Acceptance thresholds (W1 hard gate, judged on 11/2)

| Criterion | Threshold |
|---|---|
| Corpus size | ≥8,000 items; span ≥90 days; ≥6,000 after dedup |
| Category balance | ≥300 each: anime / manga / game / light novel |
| Source mix | `official` share 60–75% |
| Idempotency | second consecutive run carries only increments |
| Reproducibility | the same corpus rebuilds the same `snapshot_id` |
| Failure visibility | broken selectors raise; each crawl asserts `items > 0` |

Expected scale: ~65–70 new items/day (39 media + 25–30 first-party); 90-day backfill ≈ 3,500 media + 9,500
first-party ≈ 13–14k total. If 11/2 lands short of 8,000, the decision is made **before W3**: downgrade the A/B
comparison to qualitative + small-sample and state it in the report.

## 7. Adapter contract

```python
class SourceAdapter:
    def discover(self, cursor) -> list[RawItem]: ...        # daily increment
    def backfill(self, date_range) -> list[RawItem]: ...    # history, raises without a range
```

Four families with different failure modes — never merge them: `RssAdapter`, `WpPagedAdapter`,
`HtmlListAdapter` (one selector config per site), `StructuredApiAdapter` (rate limits live in class constants).

Rules that prevent silent shrinkage:
- A broken selector **must raise**, never return `[]`. Every crawl asserts `items > 0`.
- Paging is validated by *content change*, not status code: `?page=2` on Toei / Aniplex / KADOKAWA returns the same
  items as page 1, so a 200 proves nothing.
- Charset fallback via `charset-normalizer`, then assert the UTF-8 decode error rate.
- Rescue ladder when blocked: official equivalent endpoint → sibling domain → Wayback CDX for history gaps →
  mark `blocked_permanent` and add a line to §5. Skipping a rung breaks C1.

## 8. Politeness defaults (all sources)

Self-describing UA `AcgnRagBot/0.1 (CS5481 course project; +repo url)` · per-host token bucket 1 req/2.5 s with
jitter · global concurrency 8–14 · `robots.txt` read once and cached 24 h · conditional requests with
`ETag` / `If-Modified-Since` (a 304 is the cheapest possible politeness) · exponential backoff 1→2→4 s honoring
`Retry-After`, 3 attempts max · breaker: three consecutive failures on one source ⇒ 30-minute cooldown + run log.
