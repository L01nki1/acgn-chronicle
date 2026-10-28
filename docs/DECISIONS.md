# DECISIONS — decision log

> Rule (06 §2): every change of scope, interface or metric definition gets one line here
> (date, decision, reason, who it affects). **Verbal decisions do not count.**
> This file is the only place for change history; the spec documents describe the current truth only.

## Baseline decisions (04 §7)

| # | Decision | Content | Status |
|---|---|---|---|
| D1 | Core language | Python (permitted by the brief / ecosystem fit / reproducible evaluation; Isla consumes a JSON contract and is unaffected) | fixed |
| D2 | Evaluation window | **≥3 years**, widened year by year until 8,000 items is reached or the available history is exhausted; the window is sized by `scripts/probe_sources.py`, not extrapolated | revised 10-28 |
| D3 | Daily report | Sample only; the full implementation waits for the Isla stage; not used to justify RAG | fixed |
| D4 | Vector store | Chroma (embedded persistent); see the 2026-10-28 entry for the baseline rationale | fixed (10-28) |
| D5 | LLM API | OpenAI-compatible endpoints; extraction and answer generation may use different vendors | fixed (10-28) |

## Change log

| Date | Decision | Reason | Affects |
|---|---|---|---|
| 2026-10-28 | Repository name is `acgn-daily` | Matches the product direction; the remote was renamed and the local remote updated | everyone |
| 2026-10-28 | Repository renamed again to **`acgn-chronicle`** | Owner's naming call; local remote and the crawler user agent now point at the new location. The Python package (`acgn_rag`) and distribution name (`acgn-rag`) are unchanged | everyone |
| 2026-10-28 | Planning notes (`PLAN.md`) removed from the workspace | Its content now lives in `docs/PROJECT.md` + `docs/ROADMAP.md`; keeping both invites two versions of the truth | everyone |
| 2026-10-28 | Package path is `src/acgn_rag/core/` | `src/acgn_rag/` has three sibling packages (`core/`, `daily/`, `web/`), so the simplified `src/core/` does not apply; the boundary diagram in 05 §1 needs correcting | B, C, F |
| 2026-10-28 | Vector store baseline = Chroma | The only option that is literally a vector database while offering built-in metadata filters, a persistent directory and zero services; the three pitfalls and their workarounds are in `docs/RUNBOOK.md` §3.1 | F, E |
| 2026-10-28 | Extraction and answer generation may use different vendors | The two lines have opposite needs: extraction wants cheap, strict JSON; generation wants instruction following and no invented citations. A Protocol plus `.env` keeps the vendor swappable without code changes | B, F |
| 2026-10-28 | Tests use no network; CI goes online only to install dependencies | Site availability and 403s produce false failures, and a false-failing CI gets ignored. Online cases carry the `network` marker and run locally | everyone |
| 2026-10-28 | Isla contact moves from B to F; metric definitions move from F to E | F produces the knowledge base and `DailyDigest` and knows what retrieval output looks like; B remains the sole author of `ports.py`, forming an "F outward, B inward" pair. Metrics belong with the runner | B, E, F |
| 2026-10-28 | Daily board becomes conditionally triggered | It starts only after other members' modules land; if the schedule is tight, only the D2 sample ships | C, D |
| 2026-10-28 | Evaluation pins `ACGN_EMBED_DEVICE=cpu` | Floating-point results differ per device, so cross-machine metrics would not be comparable | E, F |
| 2026-10-28 | `ruff format` skips Markdown; ruff is pinned to `>=0.17,<0.18` | Ruff formats fenced `python` blocks in `docs/` but does not lint them, so a harmless snippet could redden CI for a reason members cannot see; pinning stops a future ruff release from reformatting the whole repo at once | everyone |
| 2026-10-28 | CI runs on every branch push, not just `main` and `wp*/**` | Members push their own branch before opening a PR and expect feedback then, but a branch named `wp1-ingest` or `fix/typo` matched neither filter, so the first check only appeared at PR time. Two runs per commit (`push` + `pull_request`) is the accepted cost | everyone |
| 2026-10-28 | CI matrix covers ubuntu + macOS + **Windows**, Python 3.11 and 3.13; local hooks are optional | The owner is the only Windows user, so without a Windows job their platform was never tested; 3.13 is what a fresh laptop ships, so the versions we allow must be the versions we test. Hooks are demoted to an optimisation: the requirement is 'commit and push', and CI plus the ruleset guarantee correctness | everyone |
| 2026-10-28 | Evaluation window widened from 90 days to **≥3 years**, widened further until 8,000 items | 90 days cannot reach 8,000 with these twelve sources, and the old sizing note was self-contradictory (65–70 items/day × 90 ≈ 6k, not the 13–14k it claimed). A rolling feed describes *new* items, while a backfill is bounded by how far back each source can be paged. The window is now set by a measurement (`scripts/probe_sources.py`), starting at 3 years and widening year by year until 8,000 is met or history runs out | A, D, F |
| 2026-10-28 | `official` share quota (60–75%) **removed** | At 3-year depth the items that survive are concentrated in the sources that actually keep archives; requiring 4,800+ official items would force either invented sources or a fake mix. The achieved mix is reported instead of being targeted | everyone |
| 2026-10-28 | **Two corpora, never mixed**: a frozen evaluation snapshot and a rolling daily-digest window | Evaluation needs a corpus that does not move (invariant 2); the product needs a corpus that moves every day. One corpus cannot do both, and keeping them separate is what resolves the conflict instead of trading one requirement against the other | everyone |
| 2026-10-28 | Daily digest dedup window = **14 days**, and it keeps no archive beyond that | Fourteen days is what deduplication needs to recognise the same announcement on its third re-publication, and what makes 'this story is developing' meaningful. History belongs to the evaluation corpus, not to the digest | C, D |
