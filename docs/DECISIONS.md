# DECISIONS — decision log

> Rule (06 §2): every change of scope, interface or metric definition gets one line here
> (date, decision, reason, who it affects). **Verbal decisions do not count.**
> This file is the only place for change history; the spec documents describe the current truth only.

## Baseline decisions (04 §7)

| # | Decision | Content | Status |
|---|---|---|---|
| D1 | Core language | Python (permitted by the brief / ecosystem fit / reproducible evaluation; Isla consumes a JSON contract and is unaffected) | fixed |
| D2 | Evaluation window | 90 days (~8,000–12,000 items) | fixed |
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
| 2026-10-28 | Vector store baseline = Chroma | The only option that is literally a vector database while offering built-in metadata filters, a persistent directory and zero services; workarounds for the three pitfalls are in PLAN §D4 | F, E |
| 2026-10-28 | Extraction and answer generation may use different vendors | The two lines have opposite needs: extraction wants cheap, strict JSON; generation wants instruction following and no invented citations. A Protocol plus `.env` keeps the vendor swappable without code changes | B, F |
| 2026-10-28 | Tests use no network; CI goes online only to install dependencies | Site availability and 403s produce false failures, and a false-failing CI gets ignored. Online cases carry the `network` marker and run locally | everyone |
| 2026-10-28 | Isla contact moves from B to F; metric definitions move from F to E | F produces the knowledge base and `DailyDigest` and knows what retrieval output looks like; B remains the sole author of `ports.py`, forming an "F outward, B inward" pair. Metrics belong with the runner | B, E, F |
| 2026-10-28 | Daily board becomes conditionally triggered | It starts only after other members' modules land; if the schedule is tight, only the D2 sample ships | C, D |
| 2026-10-28 | Evaluation pins `ACGN_EMBED_DEVICE=cpu` | Floating-point results differ per device, so cross-machine metrics would not be comparable | E, F |
