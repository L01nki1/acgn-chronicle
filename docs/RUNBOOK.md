# RUNBOOK — commands, failures, SOPs

> Scope: operational manual. Personal workarounds do not belong here; only things that still hold on another machine.

## 1. Commands

When CI is red, reproduce it locally before touching the code. `uv run python scripts/check.py` runs the
four gates in CI's order and stops at the first failure (details in `ONBOARDING.md` §4.1; inside an
activated `.venv`, plain `python scripts/check.py` is the same thing). Running them before every push
is optional — CI covers that.

```bash
ruff check .                                  # 1 lint
ruff format --check .                         # 2 format   (fix: ruff format .)
python scripts/export_schema.py --check       # 3 contract (fix: export + commit the JSON)
pytest -q                                     # 4 tests
```

| Goal | Command | Status |
|---|---|---|
| Install the commit/push gates (once per clone) | `python scripts/install_hooks.py` | ready |
| Are the gates active in this clone? | `python scripts/install_hooks.py --check` | ready |
| All four CI gates, in CI order | `python scripts/check.py` (`--fix` to auto-fix) | ready |
| Install deps (no heavy extras) | `uv sync --extra dev` | ready |
| Install embedding + store | `uv sync --extra dev --extra embed --extra store` | ready |
| Contract consistency | `python scripts/export_schema.py --check` | ready |
| All tests (zero network) | `pytest -q` | ready |
| Network-marked tests | `pytest -m network -q` | ready (local only) |
| Historical backfill | `python scripts/backfill.py --since ... --until ...` | pending A |
| Daily increment | `python scripts/discover_daily.py` | pending A |
| Build index | `python scripts/build_index.py --snapshot <id> --input <docs.jsonl> [--config A] [--fresh]` | ready (`--input` until A's reader lands) |
| Fact extraction | `python scripts/extract_facts.py --snapshot <id> --input <chunks.jsonl>` | ready |
| Evaluation | `python scripts/run_eval.py --config A --snapshot <id>` | pending E |
| Serve | `uvicorn acgn_rag.web.api:app --reload` | pending C |

## 2. Snapshot rebuild SOP

1. Back up `data/` before any backfill.
2. `python scripts/backfill.py --since <start> --until <end>`; check `items > 0` per source.
3. `python scripts/build_index.py --config A` (run the three configs sequentially — running them in parallel
   oversubscribes the CPU and is slower overall).
4. Verify idempotency: rebuild once and compare `snapshot_id`.
5. Post the `snapshot_id` + window in the group; every later run must use that id.

> When idempotency fails, the first suspects are always **manifest ordering / time zone / timestamp format**,
> not the hash function.

> Rebuilds **upsert**: a build over a *smaller* input than an earlier run leaves the extra rows in place
> (stale entries a `--limit` trial run commonly produces). Pass `--fresh` to replace the selected
> snapshot/config collections instead — each collection is deleted only **after** the input has parsed and
> produced chunks, so an invalid or empty input never wipes a valid index, and sibling collections are
> never touched. A build over zero documents, or whose documents yield zero chunks, is rejected with exit
> code 2 and writes nothing (no sidecar, no collection change). Recovery limit: `--fresh` replaces the
> whole selected collection, not individual rows — rebuilding a subset into it means the subset is all it
> will contain; the full input must be rebuilt again to restore the rest.

## 3. Common failures

| Symptom | Check first | Action |
|---|---|---|
| Item count drops to 0 | Did the site change its HTML or selectors? | Update selectors + fixtures; **never** add `try: return []` |
| Item count collapses | 403 / 429 / paging stopped working | Follow the response ladder in 01 §5.4; never rotate IPs |
| Garbled Japanese body | Header charset disagrees with the real encoding | Detect with `charset-normalizer`; assert the decode error rate |
| `top_k=5` returns 2 rows | Chroma's where-filter removed candidates | Raise over-fetch (`ACGN_OVERFETCH`) and assert the row count |
| Two `snapshot_id`s for one corpus | CRLF / ordering / time zone | See the note in §2 |
| Chunk count changes after rebuild | Chunk params edited, or tokenizer mismatch | Check `chunk/variants.py` params and the tokenizer |
| Model invents `chunk_id`s | The prompt passes ids instead of indices | Pass indices, map back in code, assert `cited ⊆ retrieved` |
| CI raises `NetworkBlocked` | A test made a real HTTP call | Mock with `respx`; mark genuinely online cases `@pytest.mark.network` |
| Green on macOS, red in CI | Filename casing | All module names lowercase |
| Green on Windows, red in CI | CRLF or path joins | `.gitattributes` enforces LF; use `pathlib` |
| `ruff check` green but `ruff format --check` red | An edit was never formatted; lint and format are separate gates | Run `ruff format .` and commit the result |
| Committing shows no `[pre-commit]` line at all | `core.hooksPath` is not set in this clone (it is a local setting) | `python scripts/install_hooks.py` |
| `hooks/pre-commit: bad interpreter` | The hook file picked up CRLF | `.gitattributes` forces LF for `.githooks/*`; re-checkout the file (`git checkout -- .githooks`) |

### 3.1 Chroma specifics — three pitfalls to verify before building the index (chromadb >=1.0,<2)

1. **Metadata holds scalars or flat typed arrays.** Since 1.0 Chroma accepts `str / int /
   float / bool` and one-type, one-dimensional arrays (`int[]` works, nesting is still
   forbidden). `media_ids` is stored natively as `int[]` and filtered with
   `{"$or": [{"media_ids": {"$contains": id}}, ...]}` — measured adequate for the small
   any-of lists `RetrievalFilters.media_ids` implies. Long any-of lists may still justify a
   SQLite reverse table `doc_id ↔ media_id` translated into `{"doc_id": {"$in": [...]}}`;
   that stays a recorded future option, not the current path.
2. **Store time as an integer epoch** (`published_ts`), not an ISO string. Range filters then compare
   numbers, which avoids format and time-zone ambiguity.
3. **Filtering is pre-KNN in 1.x** — the old "the filter trims the top-k after the search"
   story is gone. But HNSW can still under-return when `ef`/`M` are small relative to the
   filtered candidate set, so the defense stays: request `top_k × ACGN_OVERFETCH`, truncate
   afterwards, and assert `returned >= min(top_k, available_after_filter)` (symptom row
   "`top_k=5` returns 2 rows" in §3).

**The collection must be created with `hnsw:space="cosine"`** — in 1.x via
`create_collection(configuration={"hnsw": {"space": "cosine"}})`. The default space is still
`l2`, and the default would make `score` a distance while the contract promises a similarity
(`KnowledgePort.md` §3.6). With `cosine`, Chroma returns `1 - cosine_similarity` as the
distance, so the score is `1 - distance`.

### 3.2 Snapshot pointer and retention

- A build writes to a directory that is **not** served, and renames it only on success. The pointer
  `data/snapshots/CURRENT` is replaced with `os.replace`, which is atomic, so a reader either sees the old
  snapshot or the new one and never a half-built one.
- `GET /snapshot` with no id reads the pointer. That is the single definition of "current".
- Retention is `ACGN_KEEP_SNAPSHOTS=2` (current + previous). A retained old snapshot keeps answering
  normally — that is what makes an evaluation re-runnable. Beyond retention it answers `410`, not `409`.
- **Never delete the previous snapshot while an evaluation is running**: the run is pinned to it.

The read-host definition of known/retired ids is in `contracts/KnowledgePort.md` §3.2: validate matching
completed `build.json` and `identity.json` sidecars for the served configuration, then check collection
availability **before** any build-side `bind` can create a collection. Incomplete/inconsistent catalog
state is a service failure, not evidence that an id was never known. Active run pins must survive prune;
the current store's pruning alone does not implement that lease. Full snapshot ids are date-first
`YYYYMMDD-<12 hex>` (§3.1); a shortened collection-name prefix must not alias another full id (§6 of
`contracts/G1aReview.md`). These host-lifecycle checks are W2 work under the PM's #4 resolution.

### 3.3 Dedup auxiliary-channel calibration (issue #14 P2)

Measured on real ACGN headlines with e5 (F pilot, 2026-10-27): unrelated headline pairs average
**~0.70** cosine, same-story paraphrases **~0.90** — the usable window is narrow. Single-linkage
merging collapsed 149 items into **1 cluster** at threshold 0.62 (chain collapse); complete-linkage
at **0.86** gave 82 clusters with a correct cross-paraphrase merge. If the "text similarity within a
language" auxiliary is embedding-based: use **complete-linkage, start at 0.86**, treat the threshold as
a calibration item, and add a **cluster human spot-check to the G1b acceptance for config B** — B's
quality is bounded by cluster quality, so the G1b line "chunk counts stable and explainable" must
include cluster counts, not just chunk counts.

The start value's margins are asymmetric — 0.04 below the measured paraphrase mean, 0.16 above the
unrelated mean — which biases toward *splitting*, the failure that costs config B its clusters on a
corpus full of syndicated re-publication. **G1b calibration item: sweep 0.82 / 0.84 / 0.86 / 0.88**
on the frozen corpus, report cluster counts and 20 hand-checked pairs per value, and state in one
line what the threshold optimizes before it freezes.

### 3.4 Encoding lanes and encoder identity

- **Encoder identity** = model + backend (`torch` / `openvino` / `onnx`) + device + precision + library
  versions + batch size. Every index build records it in its metadata, every eval run record copies it
  (`EVAL_PROTOCOL` §7) — a build whose metadata lacks the full identity is invalid. Vectors built under
  different identities are never mixed in one index and never compared as if they were the same.
- **Cache key** = `hash(header_version + header + body) + model + prefix + encoder identity`. The
  encode-time header template (`title | publisher | date | body`, §3.5) is versioned and hashed
  *together with* the body: bump `header_version` when the template changes or stale vectors are
  silently reused, and hash both because two syndicated copies with identical bodies must not serve
  each other's header.
- **Content budget (truncation trap).** e5 models have `max_seq_length = 512`, and the chunker's
  content budget must be
  `min(config_tokens, max_seq_length − len(tokenizer(prefix).input_ids))`. The tokenizer call includes
  the model's special tokens, so they are counted exactly once; the declared config binds and the model
  only caps — an 8k-context lane must not turn config A into an 8k chunker, or the A/A′ comparison's
  independent variable becomes encoder-dependent. Measured for e5-base + `passage: `:
  `len(tokenizer("passage: ").input_ids) == 4` (2 prefix tokens + `<s>`/`</s>`), so the budget is
  **508** content tokens — 508 encodes to exactly 512 ids on jp/zh/en text, while 512 content tokens
  encode to 516 and silently lose their tail (nothing crashes; the A/A′ comparison quietly measures
  truncated text). Chunk-PR acceptance asserts it: encode `prefix + one budget-sized chunk` per lane
  and require `len(input_ids) == max_seq_length` with no truncation warning.
- **Throughput is reported in tokens/s, never bare items/s** — items/s varies ~3× with chunk length
  and reads as noise. Measured CPU baseline (i5-13400TEF, 16 threads, torch 2.15 CPU): e5-base ≈
  **1,661 tokens/s** on the pilot corpus (re-measured 1,644 inside the A380 run) ⇒ a 40k-chunk ×
  512-token corpus ≈ **3.4 h per config** — plan on it. Config-A-shaped 145/431-token batches
  (batch 32) measured 1.9–2.0k tokens/s; the ~15 % gap tracks batch size and passage-length mix and
  is not isolated, so the lower figure plans the schedule. Peak RSS ≈ 1.4 GB. Lane tables and
  per-run provenance: `notebooks/embedding-bakeoff/`.
- GPU lanes are permitted (DECISIONS 2026-10-29); `.env.example` keeps `cpu` as the default lane.
- **Extraction-vendor quirk:** if extraction runs on the CSLab DSV4-flash deployment, set `max_tokens ≥ 1024`
  — it is a reasoning model and returns empty content below that, because reasoning tokens consume the budget.

### 3.5 Embedding-model policy (pre-registered)

The measured evidence — bake-off tables, GPU-lane numbers, fp16 agreement, community-survey
validation — lives in `notebooks/embedding-bakeoff/` (reproducible from there per `PROJECT.md` §9);
this section is only the decision.

- **GPU canonical lane: two candidates — Qwen3-Embedding-0.6B and bge-m3 (dense)** — decided by the G1a
  bake-off on the real frozen corpus; **CPU lane: e5-base** stays the any-member fallback. Encoder
  identity keeps lanes separable; a G2 run binds to the identity that built its index.
- **Bake-off menu amendment (2026-10-29): EmbeddingGemma joins the GPU-lane candidates** — that is
  **`google/embeddinggemma-300m`** (308M params / 768-d MRL / Apache-2.0 / 2048-token context, strong
  in the 2026 sub-1B reviews; not the newer multimodal `embeddinggemma-2`, see rejected) — under
  the same pre-registered displacement rule (zh-subset hit@5 ≥ +10 pts or MRR ≥ +0.05 vs e5-base).
  Its usage recipe (query task prompt) is verified against the model card at bake-off time, not
  from memory.
- **Pre-registered single-lane rule:** if bge-m3 ties or wins the G1a bake-off under the existing
  rule **and** keeps the top zh-subset MRR, bge-m3 is adopted for **both** lanes, e5-base is
  demoted to emergency fallback, and G1b scheduling books CPU rebuilds at ~15.6 h/config
  (365 tokens/s measured).
- **GPU lane priority (2026-10-29): the canonical GPU runtime is Colab** (T4 measured; every member
  can reproduce it). The local A380/OpenVINO lane is a cross-check / evaluation fallback recorded
  via encoder identity, not a canonical build lane — members cannot reproduce that hardware.
- Usage differences to respect: Qwen3 takes an **instruct-style query prompt** (`prompt_name="query"` in
  sentence-transformers), documents carry no prefix. **bge-m3 needs no prefixes** and mean-pools —
  with 8k context it is the named late-chunking-capable fallback for config B (the EmbeddingGemma
  family also mean-pools: 300m's 2k context is too short for it; `embeddinggemma-2`'s 128k would
  qualify technically but is menu-excluded, see rejected). We use bge-m3's **dense channel only**;
  the sparse channel stays untouched so the hand-built BM25 stretch keeps its point.
- **Bake-off protocol before the G1b freeze**: rerun the mini-eval on the real frozen corpus with
  30+ hand-checked queries, both candidates; pick by zh-subset MRR and overall MRR with the spread
  reported; a candidate must beat e5-base by ≥10 points hit@5 or ≥0.05 MRR on the zh subset to
  displace it. Decision and identity recorded in DECISIONS.
- Considered and rejected: **Ruri v3** (JP-specialist, tops JMTEB but cannot serve zh/en queries — we
  need one trilingual retriever) · **8B+ embedders** (Qwen3-8B, Octen-8B, harrier-oss-27b: out of our
  compute class for 4×40k chunks) · **`embeddinggemma-2`** (multimodal 740M — 270M text tower + vision/audio
  encoders — 128k ctx, mean-pooling, Apache-2.0): public multilingual MTEB is flat vs 300m
  (61.15 → 61.36) with the gains concentrated in code retrieval and modalities we do not use, so it
  stays out of the bake-off menu. Recorded as a **second late-chunking-capable option after bge-m3**
  should config B ever need one.
- **Embed-time context headers (adopted, zero cost):** what gets encoded is not the raw chunk but
  `title | publisher | published date | chunk body`. RSS bodies routinely lose their parent context
  (who announced, when); a chunk that says "will begin airing on 2026-10-05" is unretrievable without it.
  The metadata is already in NormalizedItem — prepend at encode time, keep storing it as metadata too.
  The template is versioned and hashed into the cache key (§3.4).
- **Reranker stretch (proposed, pairs with the Qwen3 lane):** Qwen3-Reranker-0.6B beats
  bge-reranker-v2-m3 on the published multilingual benches, but community reports show the default
  instruction can underperform — it needs a task-specific instruction tuned on our data. Retrieve
  top-50 with the embedder, rerank to top-5; only viable on the GPU lane within the p95 budget. Owner:
  E decides latency, F owns the instruction. Not in the minimum pass condition.
- **Contingencies, not defaults:** *late chunking* (Jina) needs a long-context **mean-pooling** encoder —
  applies to bge-m3 (8k), not to e5 (512) nor to Qwen3 (last-token pooling); if config B's aggregated
  chunks underperform, late-chunking-over-bge-m3 is the documented fallback experiment, not a default.
  *Anthropic-style contextual retrieval* (one LLM call per chunk) is ~40k LLM calls here — out of
  course budget. *LanceDB* has OSS full-text search + hybrid while Chroma OSS does not (that is Chroma
  Cloud only) — if Chroma's filters or hnsw tuning underdeliver at G1b, LanceDB is the documented
  contingency, but switching is a D4 revision and needs gate evidence, not taste. *model2vec/static
  embeddings* are a dev-smoke option only, quality far below any canonical lane.

### 3.6 L0 facts-extraction SOP

- **Batch 10 announcements per LLM call** with numbered output; measured pattern from the pilot cuts
  cost 3–5× vs one-per-call, and JSON per numbered slot is easier to validate. ~9.5k L0 items ⇒ ~950
  calls, not 9,500.
- **Flat schema, validate, retry**: one flat Pydantic model (title / release_date / platform list /
  event_type / relevance), no deep nesting or Union types (a known structured-output failure mode);
  prefer the vendor's native JSON-schema `response_format` when available, else strict-parse + **2
  retries** on validation failure; after retries degrade to a plain-text chunk (never drop the doc).
- **Pre-filter before the LLM**: Capcom's feed mixes 決算/IR corporate releases with game content — a
  keyword classifier (決算｜株主｜IR｜決算短信 …) tags relevance *before* extraction so (a) the category
  balance at G1a counts what we intended and (b) ~30–40 % of the Capcom volume skips LLM calls
  entirely. Tagged items keep `source_type=official` plus an **ingest-side** category tag (e.g.
  `ingest_category=corporate`): `official/corporate` is **not** a `SourceType` (`ports.py`
  fixes the three labels) and `FactUpdate` has fixed fields. B/F resolved
  [interface #17](https://github.com/L01nki1/acgn-chronicle/issues/17) with option 1: the category stays
  ingest-side through G1a/G1b, no wire field is added, and source-mix counts use the three schema labels.
  Any later query-time consumer needs an explicit Optional-with-default contract addition (§7 of
  `contracts/KnowledgePort.md`), rather than turning `official/corporate` into a fourth source type.
- `max_tokens ≥ 1024` per call (see the DSV4-flash quirk above), and the system prompt is identical
  across a batch — keep it stable so any provider-side prompt caching can act on it.
- **Read the run summary before trusting a run**: `extract_facts.py` prints per-run counters —
  `calls` (LLM calls, initial attempts included) vs `retries` (whole-batch re-attempts),
  `extracted_chunks` vs `extracted_rows` (a parsed result that states no known value is a valid
  empty extraction: it adds to `empty_extractions` and produces no rows — not a parse failure),
  `degraded_batches`/`degraded`, `ir_skipped`. The buckets always reconcile:
  `extracted_chunks + degraded + ir_skipped == chunks`.

## 4. Emergency degradation


- **Corpus below 8,000 items**: decide before W3; downgrade the A/B comparison to "qualitative + small-sample
  quantitative" and say so in the report (04 R6).
- **LLM API unavailable**: `/ask` returns a stub answer (`abstained=true` with an explanation) so the site and
  citation jumps remain demonstrable.
- **No network during the demo**: fall back to the recording; the local index and frontend run offline.
