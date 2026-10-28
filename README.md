# acgn-rag

CS5481 Data Engineering · Topic 5 — **an ACGN cross-time archive QA system built on RAG, with verifiable citations**.

Three names, one project: repository `acgn-chronicle` · Python package `acgn_rag` · distribution `acgn-rag`.

Scope in one line: the only production artifact is `src/acgn_rag/core/`; the website, the daily digest and Isla are merely KnowledgePort consumers.
Model training / fine-tuning is **never** done here (C6, spec 06).

- Submission: 2026-11-24 23:59 (Canvas) — 6 pages of main content + a separate contribution page + appended slides + a 15-minute demo
- Planning and collaboration live in the library space `GroupProjctMartial` (documents 00–07); this repo holds **engineering artifacts only**
- Work breakdown and acceptance gates: `PLAN.md` (internal, not committed)

## Layout

```
src/acgn_rag/
├── core/          ★ the only production artifact
│   ├── ports.py   # contract source of truth (Pydantic) -> contracts/*.json is exported from it
│   ├── ingest/    # WP1 · A: four adapter families / normalization / dedup clustering / snapshots
│   ├── facts/     # WP2 · F: L0 fact-field extraction
│   ├── chunk/     # WP2 · F: A(512/64) / A'(256/0) / B(event aggregation)
│   ├── embed/     # WP2 · F: multilingual-e5-base + cache
│   ├── store/     # WP2 · F: fact zone / raw zone (two-zone design)
│   ├── retrieve/  # WP3 · E: vector-only + filters; stretch BM25 + RRF
│   ├── ground/    # WP4 · B: grounded generation / abstention / compliance clipping
│   └── eval/      # WP5 · D+E: QA set / three metrics / runner / report tables
├── daily/         # WP6 · C: daily digest consumer (later stage)
└── web/           # WP6 · C: FastAPI + web/static
scripts/           # repeatable entry points (backfill / build_index / run_eval / export_schema ...)
contracts/         # cross-language contract: knowledge_port.schema.json (+ KnowledgePort.md)
eval/              # tracked: qa_set.jsonl + runs/ (small, reproducible evidence)
data/              # never tracked: raw / facts / index / snapshots
tests/             # includes tests/compliance/ (no full text out, <=200 chars, no media paraphrase)
docs/              # RUNBOOK / DECISIONS / EVAL_PROTOCOL / COMPLIANCE / ONBOARDING
```

## Read first (10 minutes, everyone)

| Order | Document | Why |
|---|---|---|
| 1 | [`docs/PROJECT.md`](docs/PROJECT.md) | What we are building, how Topic 5 grades us, the six invariants, the vocabulary |
| 2 | [`docs/ROADMAP.md`](docs/ROADMAP.md) | Three acceptance phases, the weekly timeline, who owns what, this week's first action |
| 3 | [`docs/ONBOARDING.md`](docs/ONBOARDING.md) | Environment, branches, PR checklist, and the "CI does not use the network" rule |
| 4 | your WP's doc: [`SOURCES.md`](docs/SOURCES.md) (A) · [`EVAL_PROTOCOL.md`](docs/EVAL_PROTOCOL.md) (D/E) · [`COMPLIANCE.md`](docs/COMPLIANCE.md) (B/C) · [`RUNBOOK.md`](docs/RUNBOOK.md) (everyone, as needed) | The specifics of your area |

Task-level work is tracked as GitHub issues, one per unit, each labeled `WPn` and linked to its gate in
`ROADMAP.md`. Decisions go to [`docs/DECISIONS.md`](docs/DECISIONS.md) — verbal agreements do not count.

## Quick start

Python 3.11+ is required (CI pins 3.11; 3.13 torch wheels are unverified, so avoid it locally).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # dev only: no model download, no torch
cp .env.example .env               # fill in keys only when you need the LLM
```

> `[embed]` (sentence-transformers + torch) and `[store]` (chromadb) are heavy extras; install on demand:
> `pip install -e ".[dev,embed,store]"`. CI never installs them.

## Three commands (skeleton ready, implementations land per WP)

```bash
python scripts/export_schema.py --check     # contract consistency (a CI gate)
pytest -q                                   # all tests, zero network by default
ruff check . && ruff format --check .       # style
```

Once the pipeline lands, also:

```bash
python scripts/backfill.py --since 2026-07-30 --until 2026-10-28   # WP1 backfill
python scripts/build_index.py --config A                           # WP2 index build
python scripts/run_eval.py --config A --snapshot <snapshot_id>     # WP5 evaluation
```

## Two hard rules

1. **Tests do not use the network.** `tests/conftest.py` intercepts sockets; cases that genuinely need it are
   marked `@pytest.mark.network` and excluded by default. CI is allowed to go online only while installing
   dependencies — during tests there is zero outbound traffic (no live crawling, no model download, no LLM call).
2. **Any contract change must be re-exported.** Edit `ports.py` without running
   `python scripts/export_schema.py` and CI fails.

See [`docs/ONBOARDING.md`](docs/ONBOARDING.md) (macOS / Windows differences and commit conventions) and
[`docs/COMPLIANCE.md`](docs/COMPLIANCE.md) (the implementation spec for C1–C6).
