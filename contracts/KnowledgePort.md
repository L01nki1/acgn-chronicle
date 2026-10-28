# KnowledgePort — contract reference

> **Status: skeleton (2026-10-28).** Structure and field semantics are final enough to code against; the
> example payloads are illustrative and must be verified with a real `curl` before the freeze.
> **Owner: B.** Frozen at **G1a (2026-11-02)**; after that, changes follow §7.
>
> Source of truth is code, not this file: [`src/acgn_rag/core/ports.py`](../src/acgn_rag/core/ports.py).
> The schema is exported to [`knowledge_port.schema.json`](knowledge_port.schema.json); CI fails if the two drift.

## 1. Boundary

```
src/acgn_rag/core/          ← implements the pipeline; nothing else may import it directly
        │
        │  KnowledgePort (JSON over HTTP; same process may import)
        ▼
┌───────────────┬────────────────┬──────────────────────────┐
│ web/ (WP6)    │ daily/ (later) │ Isla (December, TypeScript)│
└───────────────┴────────────────┴──────────────────────────┘
```

A consumer depends on **this document plus the JSON Schema**, never on Python code. That is what lets Isla consume
the port from TypeScript without sharing a codebase.

## 2. Design rules

1. **Everything is bound to a `snapshot_id`.** A query names the snapshot it runs against; the server refuses a
   request whose snapshot no longer matches the current corpus (`409`). This is what makes results reproducible.
2. **The port carries excerpts, never full text.** Any `RetrievedChunk.text` that leaves the process is clipped to
   ≤200 characters by `core/ground/serialize.py`. A consumer that bypasses clipping breaks compliance C3.
3. **Additions are Optional with defaults.** A new required field would break every consumer at once.
4. **Types are structural.** No consumer may depend on field ordering, or on fields documented here but absent
   from the schema.

## 3. Types

### 3.1 `CorpusSnapshot`

| Field | Type | Req | Semantics |
|---|---|---|---|
| `snapshot_id` | str | yes | `sha256(content manifest)[:12]` + date; rebuilding the same corpus must produce the same value |
| `created_at` | str | yes | ISO-8601 timestamp of the freeze |
| `item_count` | int | yes | raw items in the snapshot |
| `deduped_count` | int | yes | items after clustering — **the denominator of every metric** |
| `source_count` | int | yes | distinct publishers contributing |
| `window_start` / `window_end` | str | yes | corpus time span; required for reproducible evaluation |

### 3.2 `RetrievalFilters`

| Field | Type | Req | Semantics |
|---|---|---|---|
| `source_types` | `("official" \| "media" \| "metadata")[]` | no | display strength and compliance treatment follow this |
| `since` / `until` | str | no | inclusive date bounds; **the daily view uses these, not a time-bucketed index** |
| `media_ids` | int[] | no | restrict to specific works |

### 3.3 `RetrievalQuery`

| Field | Type | Req | Semantics |
|---|---|---|---|
| `text` | str | yes | natural-language question |
| `top_k` | int | no (5) | desired result count; the server may return fewer if filters exclude most candidates |
| `snapshot_id` | str | yes | binds the query to a frozen corpus |
| `filters` | `RetrievalFilters` | no | |

### 3.4 `Citation`

| Field | Type | Req | Semantics |
|---|---|---|---|
| `source_type` | enum | yes | `official` / `media` / `metadata` |
| `publisher` | str | yes | e.g. `Capcom`, `ANN`, `AniList` |
| `title` | str | yes | headline as published |
| `url` | str | yes | canonical original URL — **never** an archive link |
| `published_at` | str | yes | publication date |
| `chunk_id` | str | yes | the citation must be able to locate its own chunk |

### 3.5 `RetrievedChunk`

| Field | Type | Req | Semantics |
|---|---|---|---|
| `chunk_id` | str | yes | stable within a snapshot |
| `doc_id` | str | yes | parent document |
| `snapshot_id` | str | yes | echoed for client-side verification |
| `text` | str | yes | **internal field**; clipped to ≤200 chars before leaving the process |
| `score` | float | yes | similarity; comparable only within one query, not across configurations |
| `citation` | `Citation` | yes | |

### 3.6 `GroundedAnswer`

| Field | Type | Req | Semantics |
|---|---|---|---|
| `answer` | str | yes | answer text; when `abstained` is true this holds the refusal explanation |
| `citations` | `Citation[]` | yes | deduplicated, ordered by first use |
| `cited_chunk_ids` | str[] | yes | **must be a subset of the chunks actually retrieved** for this query |
| `abstained` | bool | yes | true when the knowledge base does not support an answer |
| `abstain_reason` | str \| null | no | why it refused |

### 3.7 Daily digest types (later stage)

`DailyReportPort`, `CategoryDigest`, `ClusterCard`, `FactUpdate`, `DailyDigest` are defined in `ports.py` and already
exported to the schema, but their semantics freeze later: `DailyDigest` fields freeze at **W3 end**, and the daily
pipeline itself is conditional (see `docs/ROADMAP.md` P3).

| Type | Purpose |
|---|---|
| `CategoryDigest` | per-category counts for the day: `category`, `new_count`, `cluster_count` |
| `ClusterCard` | one deduplicated hot cluster: `cluster_id`, `size`, `media_ids`, `representative_title`, `first_seen`, `citations` |
| `FactUpdate` | one structured fact extracted that day: `media`, `field`, `value`, `chunk_id`, `url`, `publisher` |
| `DailyDigest` | the whole day: `by_category`, `top_clusters`, `fact_updates`, `episode_table`, `source_coverage`, plus `snapshot_id` and `date` |

## 4. Endpoints

| Method | Path | Request | Response |
|---|---|---|---|
| GET | `/snapshot?id=` | — | `CorpusSnapshot` (current when `id` is omitted) |
| POST | `/retrieve` | `RetrievalQuery` | `RetrievedChunk[]` (text already clipped) |
| POST | `/ask` | `RetrievalQuery` | `GroundedAnswer` |
| GET | `/daily?snapshot_id=&date=` | — | `DailyDigest` (later stage) |

Illustrative exchange — **verify before freeze**:

```http
POST /ask
Content-Type: application/json

{
  "text": "When does the second season start airing?",
  "top_k": 5,
  "snapshot_id": "20261028abcdef",
  "filters": {"source_types": ["official"], "since": "2023-10-28"}
}
```

```json
{
  "answer": "The second season starts on 2026-10-05, per the studio announcement.",
  "citations": [
    {
      "source_type": "official",
      "publisher": "Toei Animation",
      "title": "Second season announcement",
      "url": "https://www.toei-anim.co.jp/news/2026/...",
      "published_at": "2026-09-12",
      "chunk_id": "c_8f21"
    }
  ],
  "cited_chunk_ids": ["c_8f21"],
  "abstained": false,
  "abstain_reason": null
}
```

## 5. Error codes

| Code | Meaning | Client action |
|---|---|---|
| 400 | malformed request body | fix the payload; do not retry |
| 404 | snapshot or resource not found | re-read `/snapshot` and retry once with the current id |
| 409 | requested snapshot no longer matches the current corpus | fetch the current snapshot, surface the change to the user |
| 422 | invalid filter combination (e.g. `since` after `until`) | fix the filter; do not retry |

## 6. Client obligations

1. **Never render a citation without its `url`.** Attribution is mandatory (C3).
2. **Display the source type.** `official` / `media` / `metadata` must be visually distinguishable (C5).
3. **Never re-expand an excerpt.** What the port returns is already the maximum allowed quote.
4. **Echo `snapshot_id`.** A cached result may only be reused for the same snapshot.

## 7. Change process (freeze policy)

| Stage | Rule |
|---|---|
| Until 11/02 (G1a) | fields may still change; every change updates `ports.py`, re-exports the schema, and edits this document in the same PR |
| 11/02 – W2 | no breaking changes; additions must be Optional |
| From W3 | fully frozen, including additions |

Steps for any semantic change: PR states the reason → `python scripts/export_schema.py` → update this document →
notify C (web) and F (Isla contact).

## 8. TODO before the G1a freeze (owner: B)

- [ ] Run one real `curl` per endpoint and paste the actual responses over the illustrative ones in §4
- [ ] Pin down `episode_table`'s element shape (currently `list[dict]`) or drop it from the first release
- [ ] Document how `media_ids` filtering is implemented (Chroma metadata holds scalars only, so a reverse lookup
      table maps media ids to doc ids — consumers need to know the constraint, not the mechanism)
- [ ] State whether `score` is cosine similarity or a normalized distance, and whether it is comparable across configs
- [ ] Add one worked example of an abstention response
