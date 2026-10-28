# KnowledgePort — contract reference

> **Status: 0.2.0 (2026-10-28).** Structure and semantics are now consistent with the code; the
> example payloads are still illustrative and must be replaced with real `curl` output before the freeze.
> **Owner: B.** Frozen at **G1a (2026-11-02)**; after that, changes follow §7.

## 0. Who owns what (read this before arguing with the document)

Three artefacts describe this port, and they answer different questions. They are not alternatives.

| Artefact | Answers | Machine-checked |
|---|---|---|
| `src/acgn_rag/core/ports.py` | **What the structure is** — fields, types, required-ness | yes, by CI |
| `contracts/knowledge_port.schema.json` | The same, rendered as JSON Schema — **the only thing a non-Python consumer can read** | yes, CI fails on drift from `ports.py` |
| this file | **What it means** — semantics, error behaviour, client obligations | no |

So the answer to *"the doc says `ports.py` is the source of truth, but I depend on the schema — who
decides?"* is:

1. **For structure: `ports.py` decides, and the schema is its shadow.** They cannot disagree, because
   `python scripts/export_schema.py --check` fails the build. A consumer reading the schema is reading
   `ports.py`, one export later.
2. **For meaning: this file decides**, and `ports.py` repeats the important parts in
   `Field(description=...)` so the meaning also travels to consumers who never open a Markdown file.
3. **If the two ever disagree about structure, the document is wrong** — fix the document. If a field has
   no meaning written down here, this document is wrong — write it down. Neither wins by recency.
4. **B arbitrates** (`docs/ROADMAP.md` §4). A semantic change is a PR that touches all three artefacts:
   `ports.py` → re-export → this file → notify C (web) and F (Isla contact).

Only `ports.WIRE_MODELS` is exported. Internal types (§3.9) are deliberately absent from the schema: a
shape that never crosses the boundary must not be findable by a client, or it will be coded against.

## 1. Boundary

```
src/acgn_rag/core/          ← implements the pipeline; nothing else may import it directly
        │
        │  KnowledgePort (JSON over HTTP; same process may import)
        ▼
┌───────────────┬────────────────┬────────────────────────────┐
│ web/ (WP6)    │ daily/ (later) │ Isla (December, TypeScript) │
└───────────────┴────────────────┴────────────────────────────┘
```

A consumer depends on **this document plus the JSON Schema**, never on Python code. That is what lets Isla
consume the port from TypeScript without sharing a codebase.

## 2. Design rules

1. **Snapshots are immutable and addressable.** A crawl never mutates an existing snapshot: it writes a new
   one and then **flips a pointer atomically** (`os.replace`). "Current" means *what the pointer names at
   the moment of the request* — that is the whole definition, there is no second one.
2. **A read never observes a partial build.** A build writes under a non-served name and is renamed only on
   success; the pointer flips last. So there is no "rebuilding" state for a client to wait on, and a long
   query can never be interrupted by a build.
3. **Availability is governed by retention, not by currency.** The current and the previous snapshot are
   kept (`ACGN_KEEP_SNAPSHOTS=2`). An older retained snapshot **answers normally**. This is what makes
   invariant 2 possible: an evaluation run binds a `snapshot_id` and must still be re-runnable after the next
   crawl lands. Refusing an old snapshot because "the corpus moved" would make every re-run impossible and
   would kill a three-hour evaluation mid-flight, so it is not allowed.
4. **The port carries excerpts, never full text.** The wire types have exactly one text field, `quote`,
   clipped to ≤200 Unicode characters by `core/ground/serialize.py`. A consumer that bypasses clipping
   breaks compliance C3.
5. **Additions are Optional with defaults.** A new required field would break every consumer at once.
6. **Types are structural.** No consumer may depend on field ordering, or on fields documented here but
   absent from the schema.

## 3. Types

The tables below are a reading aid. **Normative shape = the schema**; if a table here disagrees with the
schema, the table is a bug.

### 3.1 `CorpusSnapshot`

| Field | Type | Req | Semantics |
|---|---|---|---|
| `snapshot_id` | str | yes | `sha256(content manifest)[:12]` + date; rebuilding the same corpus must produce the same value |
| `created_at` | str | yes | ISO-8601 timestamp of the freeze |
| `item_count` | int | yes | raw items in the snapshot |
| `deduped_count` | int | yes | items after clustering — **the denominator of every metric** |
| `source_count` | int | yes | distinct publishers contributing |
| `window_start` / `window_end` | str | yes | corpus time span; required for reproducible evaluation |

### 3.2 `SnapshotInventory`

Response of `GET /snapshots`. A client must not invent an id, and must not assume yesterday's id is still
retained.

| Field | Type | Req | Semantics |
|---|---|---|---|
| `current` | str | yes | the snapshot the read path serves |
| `retained` | str[] | yes | ids still queryable, most recent first; anything else answers `410` |

### 3.3 `RetrievalFilters`

| Field | Type | Req | Semantics |
|---|---|---|---|
| `source_types` | `("official" \| "media" \| "metadata")[]` | no | display strength and compliance treatment follow this |
| `since` / `until` | str | no | inclusive date bounds; **the daily view uses these, not a time-bucketed index** |
| `media_ids` | int[] | no | restrict to specific works |

### 3.4 `RetrievalQuery`

| Field | Type | Req | Semantics |
|---|---|---|---|
| `text` | str | yes | natural-language question |
| `top_k` | int | no (5) | desired result count; the server may return fewer if filters exclude most candidates |
| `snapshot_id` | str | yes | binds the query to a frozen corpus; unknown → `404`, retired → `410` |
| `filters` | `RetrievalFilters` | no | |

### 3.5 `Citation`

| Field | Type | Req | Semantics |
|---|---|---|---|
| `source_type` | enum | yes | `official` / `media` / `metadata` |
| `publisher` | str | yes | e.g. `Capcom`, `ANN`, `AniList` |
| `title` | str | yes | headline as published |
| `url` | str | yes | canonical original URL — **never** an archive link. Required, so attribution is structural rather than a convention |
| `published_at` | str | yes | publication date |
| `chunk_id` | str | yes | the citation must be able to locate its own chunk |

### 3.6 `ChunkExcerpt` — the response unit of `/retrieve`

| Field | Type | Req | Semantics |
|---|---|---|---|
| `chunk_id` | str | yes | stable within a snapshot |
| `score` | float | yes | cosine similarity in `[-1, 1]`, higher is better — see below |
| `quote` | str | yes | **≤200 Unicode characters.** The complete text content of the response |
| `citation` | `Citation` | yes | |

There is **no field carrying more of the article**. If a client wants the whole text it must fetch the `url`
itself — that is the compliance boundary, not an implementation detail.

**`score` semantics** (was open; decided 2026-10-28):

- It is **cosine similarity**, `1 - distance`, computed on L2-normalised vectors. The collection must be
  created with `hnsw:space="cosine"`; with the default `l2` the number returned would be a distance.
- **Higher is better.** Never expose a raw distance.
- It is **ordinal within one response only**. Not comparable across queries, and **never** across chunk
  configurations: A(512/64), A'(256/0) and A₀(512/0) put different amounts of text in a chunk, so the
  distributions differ even for the same document. A fused (RRF) score is not a similarity at all.
- Consequence: **never average `score` as a quality metric**, and never put it in a comparison table.
  Cross-config comparisons use `retrieval_hit_rate` / `answer_correctness` (`docs/EVAL_PROTOCOL.md`).

### 3.7 `GroundedAnswer`

| Field | Type | Req | Semantics |
|---|---|---|---|
| `answer` | str | yes | answer text; when `abstained` is true this holds the refusal explanation |
| `citations` | `Citation[]` | yes | deduplicated, ordered by first use |
| `cited_chunk_ids` | str[] | yes | **must be a subset of the chunks actually retrieved** for this query |
| `abstained` | bool | yes | true when the knowledge base does not support an answer |
| `abstain_reason` | str \| null | no | why it refused |

### 3.8 Daily digest types — **provisional**

These are in the schema, so CI keeps them valid, but **they are not stable**: `DailyDigest` fields freeze at
the end of W3, and the whole daily pipeline is conditional (`docs/ROADMAP.md` G3). "In the schema" means
*typed and validated*, not *promised*.

**What a consumer should do now: implement, but behind one adapter.** Build against the shape (otherwise C
idles for three weeks) and keep every field access in a single module that maps `DailyDigest` → view model.
Then a W3 change is one file, not a hunt. Nothing persists these fields — no database, no cache file keyed
by their names.

On the server side, a pre-freeze change to these types must be Optional-with-default, or announced in the
owning issue before merge.

| Type | Purpose |
|---|---|
| `CategoryDigest` | per-category counts for the day: `category`, `new_count`, `cluster_count` |
| `ClusterCard` | one deduplicated hot cluster: `cluster_id`, `size`, `media_ids`, `representative_title`, `first_seen`, `citations` |
| `FactUpdate` | one structured fact extracted that day: `media`, `field`, `value`, `chunk_id`, `url`, `publisher` |
| `DailyDigest` | the whole day: `by_category`, `top_clusters`, `fact_updates`, `source_coverage`, plus `snapshot_id` and `date` |

`episode_table` was **removed in 0.2.0**. `Sequence[dict]` has no shape, so it could neither be validated nor
typed by a consumer, and its only user (the digest) has not been crawled yet — deciding its shape now would
be guessing with the least possible information. It returns when the digest is green-lit, as
`Sequence[EpisodeRow] | None = None` with `EpisodeRow` typed from the real Bilibili timeline response.

### 3.9 Internal types (not exported)

| Type | Why it is internal |
|---|---|
| `RetrievedChunk` | carries the full body (`text`), so it must never be serialized; `core/ground/serialize.to_excerpt()` is the only door out |

## 4. Endpoints

| Method | Path | Request | Response |
|---|---|---|---|
| GET | `/snapshots` | — | `SnapshotInventory` |
| GET | `/snapshot?id=` | — | `CorpusSnapshot` (current when `id` is omitted; any retained id also answers) |
| POST | `/retrieve` | `RetrievalQuery` | `ChunkExcerpt[]` |
| POST | `/ask` | `RetrievalQuery` | `GroundedAnswer` |
| GET | `/daily?snapshot_id=&date=` | — | `DailyDigest` (later stage) |

Illustrative exchange — **verify before freeze**:

```http
POST /retrieve
Content-Type: application/json

{
  "text": "When does the second season start airing?",
  "top_k": 5,
  "snapshot_id": "20261028abcdef",
  "filters": {"source_types": ["official"], "since": "2023-10-28"}
}
```

```json
[
  {
    "chunk_id": "c_8f21",
    "score": 0.84,
    "quote": "The second season will begin airing on 2026-10-05, the studio announced on ...",
    "citation": {
      "source_type": "official",
      "publisher": "Toei Animation",
      "title": "Second season announcement",
      "url": "https://www.toei-anim.co.jp/news/2026/...",
      "published_at": "2026-09-12",
      "chunk_id": "c_8f21"
    }
  }
]
```

## 5. Error codes

| Code | Meaning | Client action |
|---|---|---|
| 400 | malformed request body | fix the payload; do not retry |
| 404 | snapshot or resource not found | re-read `/snapshots` and retry once with a valid id |
| **410** | the snapshot was retired (beyond retention) | re-read `/snapshots`; do **not** silently re-run against the new one (§6.6) |
| 422 | invalid filter combination (e.g. `since` after `until`) | fix the filter; do not retry |
| 503 | host is not serving yet (no snapshot built, or shutting down) | retry with backoff, honour `Retry-After` |

**409 was removed from every read path in 0.2.0.** It used to mean "the requested snapshot no longer matches
the current corpus", which — see §2.3 — makes a retained snapshot unqueryable and therefore makes the
evaluation irreproducible. `/retrieve` and `/ask` are reads; there is no conflict for them to signal. 409 is
reserved for a future write endpoint (e.g. `POST /build` while a build is already running).

## 6. Client obligations

1. **Never render a citation without its `url`.** Attribution is mandatory (C3).
2. **Display the source type.** `official` / `media` / `metadata` must be visually distinguishable (C5).
3. **Never re-expand an excerpt.** `quote` is already the maximum allowed quote; do not fetch the page and
   splice more text in.
4. **Echo the `snapshot_id`** the result was computed against, wherever the result is shown.
5. **Cache key = `(snapshot_id, text, filters, top_k, config)`.** A cache keyed on the question alone will
   serve an answer built from a different corpus, and nothing in the payload will reveal it.
6. **On `410`, do not silently fall back to the current snapshot.** That would attach an old answer to a new
   corpus. Show that the corpus was retired and offer a re-run.
7. **Do not retry `400` / `422`**, and do not retry `410` against the same id.

## 7. Change process (freeze policy)

| Stage | Rule |
|---|---|
| Until 11/02 (G1a) | fields may still change; every change updates `ports.py`, re-exports the schema, and edits this document in the same PR |
| 11/02 – W2 | no breaking changes; additions must be Optional |
| From W3 | fully frozen, including additions |

Steps for any semantic change: PR states the reason → `python scripts/export_schema.py` → update this
document → notify C (web) and F (Isla contact). A change that alters the wire shape must also bump
`SCHEMA_VERSION`.

## 8. TODO before the G1a freeze (owner: B)

- [x] Decide `score`'s meaning — cosine similarity, higher is better, never comparable across configs (§3.6)
- [x] `episode_table` — dropped from 0.2.0 rather than guessed (§3.8)
- [x] Keep internal types out of the schema via `ports.WIRE_MODELS` (§0)
- [x] Replace the "409 when the corpus moves" rule with immutability + retention (§2, §5)
- [ ] Run one real `curl` per endpoint and paste the actual responses over the illustrative ones in §4
- [ ] Document how `media_ids` filtering is implemented (Chroma metadata holds scalars only, so a reverse
      lookup table maps media ids to doc ids — consumers need to know the constraint, not the mechanism)
- [ ] Add one worked example of an abstention response
- [ ] Verify `GET /snapshots` and a `410` against a real retired snapshot
