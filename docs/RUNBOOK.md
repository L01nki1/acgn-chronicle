# RUNBOOK — commands, failures, SOPs

> Scope: operational manual. Personal workarounds do not belong here; only things that still hold on another machine.

## 1. Commands

When CI is red, reproduce it locally before touching the code. `python scripts/check.py` runs the four gates
in CI's order and stops at the first failure (details in `ONBOARDING.md` §4.1). Running them before every push
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
| Install deps (no heavy extras) | `pip install -e ".[dev]"` | ready |
| Install embedding + store | `pip install -e ".[dev,embed,store]"` | ready |
| Contract consistency | `python scripts/export_schema.py --check` | ready |
| All tests (zero network) | `pytest -q` | ready |
| Network-marked tests | `pytest -m network -q` | ready (local only) |
| Historical backfill | `python scripts/backfill.py --since ... --until ...` | pending A |
| Daily increment | `python scripts/discover_daily.py` | pending A |
| Build index | `python scripts/build_index.py --config A` | pending F |
| Fact extraction | `python scripts/extract_facts.py --snapshot <id>` | pending F |
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

### 3.1 Chroma specifics — three pitfalls to verify before building the index

1. **Metadata holds scalars only** (`str / int / float / bool`), never arrays. `media_ids` is
   `Sequence[int]`, so writing it directly fails. Keep a SQLite reverse table `doc_id ↔ media_id`,
   translate the condition into `doc_id in (...)`, and pass `{"doc_id": {"$in": [...]}}` to Chroma.
2. **Store time as an integer epoch** (`published_ts`), not an ISO string. Range filters then compare
   numbers, which avoids format and time-zone ambiguity.
3. **A filter can return fewer than `top_k`.** Chroma trims a candidate set, so narrow conditions return
   little. Request `top_k × ACGN_OVERFETCH`, truncate afterwards, and assert
   `returned >= min(top_k, available_after_filter)`.

**The collection must be created with `hnsw:space="cosine"`.** Chroma defaults to `l2`, and the default
would make `score` a distance while the contract promises a similarity (`KnowledgePort.md` §3.6). With
`cosine`, Chroma returns `1 - cosine_similarity` as the distance, so the score is `1 - distance`.

### 3.2 Snapshot pointer and retention

- A build writes to a directory that is **not** served, and renames it only on success. The pointer
  `data/snapshots/CURRENT` is replaced with `os.replace`, which is atomic, so a reader either sees the old
  snapshot or the new one and never a half-built one.
- `GET /snapshot` with no id reads the pointer. That is the single definition of "current".
- Retention is `ACGN_KEEP_SNAPSHOTS=2` (current + previous). A retained old snapshot keeps answering
  normally — that is what makes an evaluation re-runnable. Beyond retention it answers `410`, not `409`.
- **Never delete the previous snapshot while an evaluation is running**: the run is pinned to it.

### 3.2a Dedup auxiliary-channel calibration (issue #14 P2)

Measured on real ACGN headlines with e5 (F pilot, 2026-10-27): unrelated headline pairs average
**~0.70** cosine, same-story paraphrases **~0.90** — the usable window is narrow. Single-linkage
merging collapsed 149 items into **1 cluster** at threshold 0.62 (chain collapse); complete-linkage
at **0.86** gave 82 clusters with a correct cross-paraphrase merge. If the "text similarity within a
language" auxiliary is embedding-based: use **complete-linkage, start at 0.86**, treat the threshold as
a calibration item, and add a **cluster human spot-check to the G1b acceptance for config B** — B's
quality is bounded by cluster quality, so the G1b line "chunk counts stable and explainable" must
include cluster counts, not just chunk counts.

### 3.3 Encoding lanes and encoder identity

- **Encoder identity** = model + backend (`torch` / `openvino` / `onnx`) + device + precision + library
  versions + batch size. Every index build records it in its metadata, every eval run record copies it
  (`EVAL_PROTOCOL` §7). Vectors built under different identities are never mixed in one index and never
  compared as if they were the same.
- The embed cache is keyed by `content hash + model + prefix + encoder identity`, so switching lanes never
  serves vectors from another lane, and a rebuild after a chunk change re-encodes only what changed.
- **Measured CPU baseline (2026-10-29, F's machine: i5-13400TEF, 16 threads, torch 2.15 CPU, batch 32):**
  e5-base encodes at ≈ **1.9–2.0k tokens/s**, stable across passage lengths (13.8 items/s at ~145 tokens,
  4.4 items/s at ~431 tokens; peak RSS ≈ 1.4 GB). At the config-A budget (512 tokens/chunk) that is
  ≈ 3.9 items/s ⇒ a 40k-chunk corpus ≈ 2.9 h **per config**; plan the four configs sequentially.
  Report throughput in **tokens/s**, never bare items/s — items/s silently varies ~3× with chunk length.
- GPU lanes are permitted (DECISIONS 2026-10-29). An OpenVINO lane measured ≈ 8.8× CPU on e5-small on the
  same machine; e5-base GPU numbers will be added once the canonical lane is chosen (F, G1a).
- **512-token budget must include the prefix (truncation trap).** e5 models have `max_seq_length = 512`;
  a 512-content-token config-A chunk plus the `passage: ` prefix silently truncates the chunk tail at
  encode time — every config-A chunk loses its last few tokens, uniformly, so nothing crashes and the A/B
  comparison quietly measures truncated text. The chunker's content budget must be
  `max_seq_length − len(tokenizer(passage_prefix).input_ids)` (≈ 508), or the encoder must be configured
  with the prefix inside the counted budget. Same cap applies to config-B aggregation chunks.
- **Extraction-vendor quirk:** if extraction runs on the CSLab DSV4-flash deployment, set `max_tokens ≥ 1024`
  — it is a reasoning model and returns empty content below that, because reasoning tokens consume the budget.

### 3.3a Embedding-model bake-off — measured (issue #15 review, 2026-10-29)

Pilot bake-off on real trilingual ACGN text (135 passages: Capcom JP press + ANN EN news + Gcores ZH
media; 16 queries over 8 gold entities, half cross-lingual). Same machine and lane (torch CPU), same
texts; only the encoder varies. Numbers are a *selection* sample (n=16), not the G2 evaluation:

| model | dim | CPU tok/s | hit@5 | MRR@5 | zh-query hit | zh-query MRR |
|---|---|---|---|---|---|---|
| multilingual-e5-small | 384 | 5090 | 0.69 | 0.559 | 0.50 | 0.344 |
| multilingual-e5-base (current default) | 768 | 1661 | 0.81 | 0.771 | 0.75 | 0.667 |
| multilingual-e5-large | 1024 | 421 | 0.81 | 0.771 | 0.62 | 0.625 |
| **Qwen3-Embedding-0.6B** | 1024 | 53 (bf16) / 206 (fp32) | **1.00** | **0.880** | **1.00** | **0.844** |

All three e5 misses were cross-lingual queries on JP-source items; Qwen3 answered every zh and en query
on JP sources directly, with no translation lane. This matches the public picture (Qwen3-Embedding small
variants lead the current MMTEB multilingual boards; e5 is a 2023/24-generation baseline), but the deciding
numbers here are ours.

**Policy (pre-registered):**
- **GPU canonical lane: Qwen3-Embedding-0.6B** is the candidate to confirm at G1a; **CPU lane: e5-base**
  stays the any-member fallback. Encoder identity keeps the two lanes separable; a G2 run binds to the
  identity that built its index.
- Usage differences to respect: Qwen3 takes an **instruct-style query prompt** (`prompt_name="query"` in
  sentence-transformers), documents carry no prefix; its context is 32k so the 512-budget truncation trap
  does not apply — but config A stays 512/64 for the course A/B comparison, and the chunker's budget must
  be derived from the model (`max_seq_length − prefix`), not hard-coded. Qwen3 supports MRL output dims
  (32–1024) if storage wants a smaller vector.
- **Bake-off protocol before the G1b freeze**: rerun the same mini-eval on the real frozen corpus with
  30+ hand-checked queries; switch the GPU lane to Qwen3 only if it keeps a clear margin (≥10 points
  hit@5 or ≥0.05 MRR on the zh subset). Decision and identity recorded in DECISIONS.
- Considered and rejected: **bge-m3** (its sparse channel would make the BM25 stretch moot; dense quality
  below Qwen3 on current boards; heavier) · **Ruri v3** (JP-specialist, tops JMTEB but cannot serve zh/en
  queries — we need one trilingual retriever) · **8B+ embedders** (Qwen3-8B, Octen-8B, harrier-oss-27b:
  out of our compute class for 4×40k chunks).
- **Embed-time context headers (adopted, zero cost):** what gets encoded is not the raw chunk but
  `title | publisher | published date | chunk body`. RSS bodies routinely lose their parent context
  (who announced, when); a chunk that says "will begin airing on 2026-10-05" is unretrievable without it.
  The metadata is already in NormalizedItem — prepend at encode time, keep storing it as metadata too.
- **Reranker stretch (proposed, pairs with the Qwen3 lane):** Qwen3-Reranker-0.6B beats
  bge-reranker-v2-m3 on the published multilingual benches, but community reports show the default
  instruction can underperform — it needs a task-specific instruction tuned on our data. Retrieve
  top-50 with the embedder, rerank to top-5; only viable on the GPU lane within the p95 budget. Owner:
  E decides latency, F owns the instruction. Not in the minimum pass condition.
- **Contingencies, not defaults:** *late chunking* (Jina) needs a long-context **mean-pooling** encoder —
  applies to bge-m3 (8k), not to e5 (512) nor to Qwen3 (last-token pooling); if config B's aggregated
  chunks underperform, late-chunking-over-bge-m3 is the documented fallback experiment, not a default.
  *Anthropic-style contextual retrieval* (one LLM call per chunk) beats both on quality in published
  comparisons and is ~40k LLM calls here — out of course budget. *LanceDB* has OSS full-text search +
  hybrid while Chroma OSS does not (that is Chroma Cloud only) — if Chroma's filters or hnsw tuning
  underdeliver at G1b, LanceDB is the documented contingency, but switching is a D4 revision and needs
  gate evidence, not taste. *model2vec/static embeddings* are a dev-smoke option only, quality far below
  any canonical lane.

### 3.4 L0 facts-extraction SOP

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
  entirely. Tagged items stay in the corpus as `official/corporate` unless A/D decide otherwise.
- `max_tokens ≥ 1024` per call (see the DSV4-flash quirk above), and the system prompt is identical
  across a batch — keep it stable so any provider-side prompt caching can act on it.

## 4. Emergency degradation


- **Corpus below 8,000 items**: decide before W3; downgrade the A/B comparison to "qualitative + small-sample
  quantitative" and say so in the report (04 R6).
- **LLM API unavailable**: `/ask` returns a stub answer (`abstained=true` with an explanation) so the site and
  citation jumps remain demonstrable.
- **No network during the demo**: fall back to the recording; the local index and frontend run offline.
