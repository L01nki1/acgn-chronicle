# ROADMAP — phases, gates and ownership

> This is the execution map. **Task-level detail lives in GitHub issues**; each issue links back to the gate
> it belongs to. If an issue and this document disagree about dates or ownership, this document wins and we fix
> the issue.
> Today is 2026-10-28. W1 closes on 11/2 — the corpus gate is five days out.

## 1. Three acceptance phases

| Phase | Scope | Exit gate | Owner | Window |
|---|---|---|---|---|
| **P1 Core pipeline** | skeleton + contract, ingestion, knowledge base, fact extraction | G1a / G1b | A, F | W1–W2 |
| **P2 Topic-5 requirements** | retrieval, chunking comparison, grounded generation, evaluation, report | G2 | B, E, D | W2–W4 |
| **P3 Daily pipeline** | digest contract → `/daily` endpoint → sample → board | G3 | C | W3–W4 (conditional) |

Dependency rule: **P1 is the only critical path.** P2's evaluation consumes P1's snapshot; P3 starts only after
P2 is green and evaluation visualization is done. If the schedule slips, P3 is cut first — its budget is ≤1.5 person-days.

## 2. Gates and what "passing" means

### G1a · 2026-11-02 — corpus and contract

| Check | Threshold |
|---|---|
| Corpus size | ≥8,000 items; span ≥90 days; ≥6,000 after dedup |
| Category balance | ≥300 items each for anime / manga / game / light novel |
| Source mix | `official` share 60–75% |
| Idempotency | two consecutive runs, second contains only increments; same corpus rebuilds the same `snapshot_id` |
| Failure visibility | selector breakage raises; every crawl asserts `items > 0` |
| Contract | `ports.py` v1 + exported schema + `contracts/KnowledgePort.md`; CI contract check green |

Run: `python scripts/backfill.py --since … --until …`, then compare `snapshot_id` on a rebuild.

### G1b · 2026-11-09 — indexes and facts

- A / A' / B index builds reproducible; chunk counts explainable and stable across rebuilds
- e5-base encoding with a working cache; compliance tests green (`pytest tests/compliance -q`)
- L0 fact extraction on a 30-item spot check ≥90% field accuracy, every row carries `chunkId` + `url`

### G2 · 2026-11-21 — Topic 5 complete

| Requirement | Pass condition |
|---|---|
| ① data prep + ≥2 chunking strategies | A vs A' (optionally A₀) differ, and the difference is explained without confounded variables |
| ② vector database | three collections rebuild independently; filters correct |
| ③ retrieval | filter combination tests; p95 well under 500 ms locally |
| ④ grounded generation | `cited_chunk_ids ⊆ retrieved`; unanswerable questions refuse; clipping tests green |
| ⑤ evaluation | ≥30 QA items (course minimum is 20), three metrics, reruns on the same snapshot give the same numbers |
| Delivery | 6-page body drafted, contribution page signed, slides ready, 15-minute demo rehearsed |

### G3 · conditional — daily pipeline

`DailyDigest` fields frozen (W3 end) → `GET /daily?snapshot_id=&date=` → 3–5 day human-checked sample →
board page (≤1.5 person-days) → later stage: scheduler and alerting.
Pass condition: same `(snapshot, date)` reproduces the same digest; every citation clipped and jumpable.

## 3. Weekly timeline

| Week | Dates | Exit criteria | Everyone's focus |
|---|---|---|---|
| W1 | 10/27–11/2 | G1a | A: adapters + backfill; B: contract frozen; D: repo + CI + QA type A; E: snapshot review + runner skeleton; F: chunk A + encoding budget; C: query page skeleton |
| W2 | 11/3–11/9 | G1b | A: 9 more HTML sources + filters; F: three indexes + fact extraction; B: grounded generation; C: retrieval page wiring; D: QA B/C annotation; E: runner usable |
| W3 | 11/10–11/15 | end-to-end QA works | B: abstention + compliance tests; E: tuning + latency; C: citation jumps; D: experiment scheduling; F: metric rules frozen |
| W4 | 11/16–11/21 | G2 | D: evaluation tables + report §6; B: report §1/§5; F: report §2/§3; E: report §4; C: site polish + evaluation visualization (+ board if time) |
| Sprint | 11/20–11/23 | submit | Everyone proofreads; C+D slides; D assembles signature page; recording as fallback |
| Submit | **11/24 23:59** | — | Canvas: report + code + slides |

## 4. Work packages and ownership

| WP | Scope | Primary | Cross-review | Directory |
|---|---|---|---|---|
| WP1 | Ingestion, dedup clustering, snapshots | **A** | WP2 | `core/ingest/`, `scripts/backfill.py` |
| WP2 | Chunking, embedding, storage, L0 facts | **F** | WP3 | `core/{facts,chunk,embed,store}/` |
| WP3 | Retrieval and filters | **E** | WP1 | `core/retrieve/` |
| WP4 | Grounded generation, citations, abstention, contract | **B** | WP6 | `core/ground/`, `contracts/` |
| WP5 | QA set, metrics, runner, report tables | **D** (runner: E) | with F | `core/eval/`, `eval/`, `notebooks/03` |
| WP6 | Website, demo, daily digest | **C** | WP4 | `web/`, `daily/` |
| Compliance | Two lines of defense | **B + C** | everyone | `tests/compliance/` |

Secondary responsibilities (revised 2026-10-28):

| Member | Primary | Secondary |
|---|---|---|
| A | WP1 | entity-extraction support for WP2 |
| B | WP4 + **contract author** | WP1 collector plumbing; contract arbitration |
| C | WP6 | WP4 clipping (front-end last line); WP5 visualization |
| D | WP5 + PM | demo script; QA organization; contribution log |
| E | WP3 | WP5 runner + **metric definitions**; WP1 snapshot review |
| F | WP2 | **Isla contact**; extraction prompts |

## 5. This week, per member (W1, ends 11/2)

| Member | First action |
|---|---|
| A | `SourceAdapter` two-mode interface + Toei HTML-list adapter (one request ≈ 18 years) + ANN per-day archive + Capcom full backfill |
| B | `ports.py` v1 + export schema + `contracts/KnowledgePort.md`; research grounded-prompt patterns |
| C | Query page skeleton (citation jump left blank); agree the front-end clipping contract with B |
| D | Repo + CI done; start QA type-A construction; set up the contribution log |
| E | Design snapshot idempotency and storage schema, review A's interface; `run_eval.py` skeleton |
| F | Get config A chunking + e5 encoding running, measure throughput to derive the indexing budget |

## 6. Top risks

| # | Risk | Mitigation | Owner |
|---|---|---|---|
| R1 | Corpus short of 8,000 → no A/B separation | Decide before W3; downgrade to qualitative + small-sample and say so | A, D |
| R2 | Copyright breach (full text out, media paraphrase) | C2/C3/C4 + two lines of defense + tests | B, C |
| R3 | Irreproducible evaluation | snapshot binding + run archive | D, E |
| R4 | Scope creep (site, board) | Report stays five sections; board is conditional; D holds the cut list | D |
| R5 | Missing signature page = desk reject | Checked daily from 11/20 | D |
| R6 | Silent corpus shrinkage | selector errors raise; `items > 0` assertion; coverage monitoring | A |
| R7 | Encoding over budget | e5-base first, cache mandatory, measure before scaling | F |
| R8 | QA set degenerates into lookups | type A ≤ 1/3; C/D must include cross-source synthesis | D |

## 7. How we track work

- **Gate status** = this document. Update it when a gate passes or slips; do not keep private progress notes.
- **Task detail** = GitHub issues, one per actionable unit, each labeled `WPn` and linked to its gate.
- **Decisions** = `docs/DECISIONS.md`, one line per decision. Verbal agreements do not count.
- **Weekly** = D updates the contribution log; E archives one `run_eval.py` result.
