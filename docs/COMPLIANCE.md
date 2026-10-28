# COMPLIANCE — implementation spec for C1–C6

> Scope: the **engineering** side of compliance. The red lines themselves are in spec 06 §3; this file records
> how they land in code and how they are asserted. If anything here contradicts spec 06, spec 06 wins and
> this file gets corrected.

## C1 — Crawl official, public channels only

| Requirement | Where it lands |
|---|---|
| Default self-describing UA: `AcgnRagBot/0.1 (CS5481 course project; +repo url)` | `config.HttpConfig.user_agent` |
| Per-host token bucket, ~1 req/2.5 s with jitter | `core/ingest/http.py` |
| Read `robots.txt` before crawling, cache 24 h, skip `Disallow` paths | `core/ingest/http.py` |
| Backfill must carry a date range; a windowless request raises `ValueError` | every `backfill(date_range)` |
| Forbidden: bypassing logins, IP pools, faking search-engine UAs, headless browsers to beat a gate, throwaway accounts | no such switch exists in the code |

**Test**: `tests/ingest/test_c1_window.py` — a windowless request must raise; a `Disallow` path must be refused.

## C2 — Two-zone storage

| Zone | Contents | Exposure |
|---|---|---|
| Fact zone `data/facts/` | title / release date / platform / event type, each with `chunkId` + `url` | publishable |
| Raw zone `data/raw/` | full crawled bodies | internal retrieval only, never serialized |

**Assertion**: no outbound path (`to_dict()` / `model_dump()`) may carry raw-zone text. Clipping happens only in
`core/ground/serialize.py`; other modules must not build responses on their own.

## C3 — Four-part outbound payload (<=200 characters)

```
summary + source URL + <=200 character excerpt + mandatory attribution
```

- Constant: `MAX_QUOTE_CHARS = 200` in `core/ground/serialize.py` (single source).
- **Unit: Unicode characters** (`len(str)`). Counting bytes would leave Japanese with roughly 60 characters
  per excerpt while English gets 200 — a per-language display gap that "quote a passage" does not intend.
- Clipping: `text[:199] + "…"`, so the ellipsis is charged against the budget and the total stays <=200.
- Whitespace collapsing is allowed; **re-wording, reordering or dropping sentences is not** — that is rewriting.
- Two lines of defense: backend `serialize.py` plus a second clip before rendering in `web/static/` (owned by C).

**Test**: `tests/compliance/test_serialize_limits.py` (11 cases, in place).

## C4 — `media` commentary must not be paraphrased

Judgement rule used by both the prompt and the writers:

| Allowed | Forbidden |
|---|---|
| Extract verifiable facts from `media` ("the report states X ships on 14 March") | Reproduce its opinions, reviews or speculation ("called the best of the season") |
| Quote <=200 characters verbatim with attribution | Rewrite or summarize its editorial stance |

**Prompt clause** (owned by B, stored as a constant in `core/ground/`):
> If a source's `source_type` is `media`, extract only verifiable facts and figures; never paraphrase its
> commentary, reviews or speculation. If the source is `official`, official statements and facts may be restated.

**Test**: `tests/compliance/test_media_no_paraphrase.py` — snapshot the prompt so the clause cannot disappear silently.

## C5 — Every item is labeled

`SourceType = Literal["official", "media", "metadata"]`, labeled **at ingestion**, never inferred later.
Display strength differs per type (official strongest, metadata weakest). `DailyDigest.source_coverage` reuses the
same field for crawl monitoring.

## C6 — No training, no fine-tuning

The repository must contain no training script, no `train*.py`, no LoRA / fine-tune config. CI cannot check this
exhaustively; review enforces it.

## Quick reference

Never: serve full text | paraphrase media commentary | train models | commercialize

## Test matrix (`pytest tests/compliance -q` must be green)

| Case | Assertion |
|---|---|
| `test_serialize_limits.py` | <=200 chars (CJK and ASCII), ellipsis inside the budget, whitespace collapsed, payload carries a url |
| `test_media_no_paraphrase.py` | the no-paraphrase clause exists in the prompt |
| `test_no_raw_export.py` | no API response contains full text (pending WP4) |
