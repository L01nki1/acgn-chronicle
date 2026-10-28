# PROJECT — what we are building and why

> Read this once, end to end, before touching code. It is the map; the detail lives in the linked docs.
> Spec numbers (e.g. `04 §1.3`) point at the planning library; everything a member needs day to day is in this repo.

## 1. One sentence

**An ACGN cross-time archive QA system built on RAG, where every claim in an answer is traceable to a source chunk and the system refuses instead of guessing.**

Course: CS5481 Data Engineering, Topic 5 — *Knowledge Base Q&A System with Verifiable Citations*.

## 2. Scope: what counts, what does not

| Item | Treatment |
|---|---|
| RAG QA + citations + abstention + evaluation (the five Topic-5 requirements) | **Core artifact.** All of it goes into the report |
| Retrieval website with citation jumps | Build it (WP6); one paragraph in the report |
| Daily digest | Sample only (3–5 days, human-checked); full pipeline is a later stage |
| Daily board (visual) | Stretch: starts only after other workstreams land; ships as a demo easter egg, not in the report |
| Isla integration | **Not this stage.** Only freeze the contract; F validates it from the consumer side |
| Model training / fine-tuning | **Never** (compliance C6) |

The grading weight sits on the backend pipeline and the faithfulness mechanisms, not on the chat UI. The brief says so explicitly: *"the main challenge is not the chatbot interface, but the backend data pipeline ... as well as the mechanisms that keep the answers faithful to the sources."*

## 3. Why RAG is necessary here (the argument we will reuse in §1 of the report)

- A single day of ACGN news fits in a context window (~10⁵ tokens), so **the daily view is not the argument**.
- What we actually index: a 90-day backfill plus daily increments ≈ 13–14k articles ≈ 2.5 × 10⁷ tokens ≈ **100× a 200K context window**.
- Therefore the product is a *traceable archive QA*, not "what happened today" — and that decision shapes the QA set: it cannot be all "what is new today".

## 4. How Topic 5 grades us

| # | Requirement (brief) | Our implementation | Where | Status |
|---|---|---|---|---|
| 1 | Collect documents; load / chunk / embed; **compare ≥2 chunking strategies** | 50+ sources → ≥8,000 items over ≥90 days; configs A(512/64), A'(256/0), A₀(512/0), B(event aggregation); e5-base | `core/ingest`, `core/chunk`, `core/embed` | W1–W2 |
| 2 | Store chunks + embeddings in **a vector database** | Chroma (embedded persistent), two-zone storage | `core/store` | W1–W2 |
| 3 | Query → embedding → **semantic retrieval** | Vector top-k + filters (`source_types` / `since`–`until` / `media_ids`); stretch BM25 + RRF | `core/retrieve` | W3 |
| 4 | **Grounded generation**, cite source per claim, **refuse when unknown** | Citation with `chunk_id` + URL, `abstained` + reason, compliance clipping | `core/ground` | W2–W3 |
| 5 | **≥20 QA pairs**, report hit rate + answer correctness | 30+ items in four types, three rule-based metrics, every run bound to a snapshot | `core/eval`, `eval/` | W3–W4 |

Report structure: §1 Introduction, §2–§5 Methodology, §6 Experiments & Discussions. Required references include Lewis et al. 2020 and Gao et al. 2023.

## 5. How data flows

```
[50+ sources · four adapter families · W1/W2 batches]
   │ discover() daily increment / backfill(range) history
   ▼
RawItem → NormalizedItem → entity linking (mediaIds / event type)
   ▼
SimHash + 72h window clustering (key: mediaIds + event type)
   ▼
Snapshot freeze (snapshot_id = content hash + date; window_start/end)
   ▼
┌──────────────────────┬────────────────────────┐
│ Fact zone (publishable)│ Raw zone (internal only)│  ← L0 fact extraction (chunkId + URL)
└──────────────────────┴────────────────────────┘
   ▼
Chunk A / A' / A₀ / B  →  e5-base encode + cache  →  vector index (Chroma)
   ▼
Retrieve: vector + filters  |  stretch BM25 + RRF
   ▼
Grounded generation: per-claim citation + abstention + compliance clipping (≤200 chars)
   ▼
KnowledgePort (JSON over HTTP) →  web  |  daily digest  |  (December) Isla
```

## 6. Six invariants (write them in comments, enforce them in tests)

| # | Invariant | Why |
|---|---|---|
| 1 | The raw zone is never serialized to clients | Copyright: full text leaking out is the one unrecoverable failure |
| 2 | Every experiment is bound to a `snapshot_id` | Without it, an A/B comparison is not reproducible |
| 3 | Every metric uses `deduped_count` as denominator | Deduplicated corpus only; otherwise hit rate is inflated by syndicated copies |
| 4 | Per-day views use `filters.since/until`, never a time-bucketed index | A bucketed index would contaminate the chunking comparison |
| 5 | `ports.py` and `contracts/*.json` are kept in sync by CI | The contract is the only thing web / daily / Isla depend on |
| 6 | `data/` never enters git | Size and copyright; corpus is rebuildable by scripts |

## 7. Compliance in one screen

| Rule | In practice |
|---|---|
| C1 official public channels only | self-describing UA, robots.txt, per-host pacing, no login bypass / IP pools / fake browser UAs |
| C2 two zones | facts are publishable, full text is internal-only |
| C3 outbound payload = summary + URL + ≤200 chars + attribution | one clipping function, two layers of defense |
| C4 never paraphrase `media` commentary | extract verifiable facts only; `official` may be restated |
| C5 label every item | `official` / `media` / `metadata`, at ingestion |
| C6 no training | retrieval-only; no `train*.py` in the repo |

Details and tests: [`COMPLIANCE.md`](COMPLIANCE.md).

## 8. Vocabulary (use these exact terms in code, issues and the report)

| Term | Meaning |
|---|---|
| `snapshot_id` | content hash of the corpus + date; the binding key for every experiment |
| `deduped_count` | items after clustering; the denominator of all metrics |
| `NormalizedItem` | the unified record every adapter emits |
| Config `A` / `A'` / `A₀` / `B` | 512/64, 256/0, 512/0, event aggregation; `A+hybrid` = BM25+vector+RRF |
| QA type `A`/`B`/`C`/`D` | official facts / structured lookup / event clustering / mixed + abstention |
| `abstained` | explicit "the knowledge base does not contain this" instead of guessing |
| dual mode | `discover()` daily increment vs `backfill(range)` history |
| fact zone / raw zone | publishable structured facts vs internal-only full text |
| source types | `official` / `media` / `metadata` (display strength differs) |

## 9. Definition of done (course deliverables)

- Report PDF: **6 pages** of main content (intro / methodology / experiments) + references and appendix unlimited
- **A separate page** with the contribution statement, individual percentages and **every member's signature**, placed after the body and before the slides. Missing page or missing signature = **desk reject**, not a late penalty
- Slides appended at the end of the report; 15-minute in-class demo (script + recording as fallback)
- Source code: Python / Jupyter; every number in the report must be reproducible from `notebooks/`
- Evaluation assets: `eval/qa_set.jsonl` + `eval/runs/` (tracked, they are the reproducibility evidence)
- Deadline: **2026-11-24 23:59**, Canvas

## 10. Where to go next

| You want | Read |
|---|---|
| What to build this week, who owns what | [`ROADMAP.md`](ROADMAP.md) |
| Which sources to crawl, and the acceptance thresholds | [`SOURCES.md`](SOURCES.md) |
| Environment, branches, PR rules, "CI does not use the network" | [`ONBOARDING.md`](ONBOARDING.md) |
| Copyright red lines and their tests | [`COMPLIANCE.md`](COMPLIANCE.md) |
| How to annotate and judge QA, how to log a run | [`EVAL_PROTOCOL.md`](EVAL_PROTOCOL.md) |
| Commands and failure triage | [`RUNBOOK.md`](RUNBOOK.md) |
| Why a decision was made | [`DECISIONS.md`](DECISIONS.md) |
