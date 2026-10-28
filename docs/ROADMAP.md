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
| Corpus size | ≥8,000 items; ≥6,000 after dedup |
| Corpus span | **starts at 3 years**; widen year by year until 8,000 is reached or the available history is exhausted |
| Source depth probe | per-source × per-depth counts recorded before the window is frozen (`docs/SOURCES.md` §6) |
| Category balance | ≥300 items each for anime / manga / game / light novel |
| Source mix | **no quota** — report the achieved `official` / `media` / `metadata` mix |
| Idempotency | two consecutive runs, second contains only increments; same corpus rebuilds the same `snapshot_id` |
| Failure visibility | selector breakage raises; every crawl asserts `items > 0` |
| Contract | `ports.py` v1 + exported schema + `contracts/KnowledgePort.md`; CI contract check green |

Run: `python scripts/probe_sources.py` to size the window, then `python scripts/backfill.py --since … --until …`,
then compare `snapshot_id` on a rebuild. The chosen window is frozen into `docs/DECISIONS.md` and into every
run's `meta.json`.

### G1b · 2026-11-09 — indexes and facts

- A / A' / B index builds reproducible; chunk counts explainable and stable across rebuilds
- encoding on the recorded lane with an identity-keyed cache (model policy: RUNBOOK §3.5); compliance tests green (`pytest tests/compliance -q`)
- L0 fact extraction on a 30-item spot check ≥90% field accuracy, every row carries `chunkId` + `url`

### G2 · 2026-11-21 — Topic 5 complete (official brief, hard gate)

> Topic 5 is a grounded QA system over a custom knowledge base, not a chatbot demo. The gate below reflects the course brief exactly: data prep + retrieval + grounded answer + evaluation, with traceable citations and explicit abstention.

| Requirement | Minimum pass condition | Evidence required |
|---|---|---|
| ① Data prep + ≥2 chunking strategies | A vs A' (optionally A₀) differ, and the difference is explained without confounded variables | chunk-count table + config comparison + snapshot binding |
| ② Knowledge base storage | vector database stores chunk embeddings; collections rebuild independently; metadata filters work | one successful build for each collection + filter tests |
| ③ Retrieval mechanism | question is embedded; relevant chunks are retrieved by semantic similarity; filter combinations work | retrieval logs / tests; p95 under 500 ms locally |
| ④ Grounded generation | `cited_chunk_ids ⊆ retrieved`; answer is generated only from retrieved chunks; unsupported questions refuse instead of guessing; clipping tests green | answer payloads + citation audit + abstention test cases |
| ⑤ Evaluation | ≥30 QA items (course minimum is 20); the run archive includes `snapshot_id`, `window`, `run_id` and metrics; three metrics reported; reruns on the same snapshot stay within the documented spread | `eval/runs/<run_id>/meta.json` + QA pack + repeat-run evidence |
| Delivery | 6-page body drafted, contribution page signed, slides ready, 15-minute demo rehearsed | final report bundle + signature page |

#### G2 hard checks (must be green before G2 closes)

1. `snapshot_id` is bound to the corpus and every run uses a frozen snapshot.
2. At least two chunking strategies are implemented and compared; the comparison is not confounded by a different corpus or window.
3. Retrieval is vector-based and uses the same `snapshot_id` as the question.
4. Answers are grounded in retrieved chunks; citations are reproducible and every cited chunk is actually in the retrieved set.
5. Unanswerable questions produce `abstained=true` and a refusal reason instead of a guessed answer.
6. Evaluation uses a QA set with at least 30 items, balanced across the required QA types, and reports the metric definitions and rerun spread.
7. The final report contains the required task-5 narrative: method, experiments, discussion, references, and the separate contribution/signature page.

#### G2 stretch items (nice-to-have, cut first if schedule slips)

- BM25 + RRF hybrid retrieval
- multi-source daily digest sample
- board / visual demo page
- more aggressive query-time filtering and latency tuning

These are optional. The minimum success condition remains the grounded QA system, not the extra UI or board work.

### G3 · conditional — daily pipeline

`DailyDigest` fields frozen (W3 end) → `GET /daily?snapshot_id=&date=` → 3–5 day human-checked sample →
board page (≤1.5 person-days) → later stage: scheduler and alerting.
Pass condition: same `(snapshot, date)` reproduces the same digest; every citation clipped and jumpable.

The digest reads **a rolling 14-day window**, never the evaluation snapshot: 14 days is what deduplication needs
to recognise the same announcement on its third re-publication, and what "this story is developing" needs to mean
anything. It keeps no archive beyond that window — history belongs to the evaluation corpus, which is separate
and frozen (`docs/DECISIONS.md`, two-corpora entry).

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

| Member | Focus direction |
|---|---|
| A | Build the ingest backbone: adapters, backfill, dedup pipeline, and a reproducible snapshot freeze. First concrete target is a working source-to-snapshot path for ANN + Toei + Capcom. |
| B | Freeze the contract and the grounded answer path: `ports.py`, exported schema, citation contract, and the abstention / refusal mechanism. |
| C | Stand up the retrieval-facing interface and the front-end citation contract: query page, citation display, and clipping-compatible rendering. |
| D | Set up evaluation-first work: QA type A/B/C/D construction, runner skeleton, run records, and the contribution/signature plan for the final report. |
| E | Own retrieval and evaluation backend: filters, retrieval correctness, run_id and metric rules, and the measurement of latency / rerun spread. |
| F | Own the knowledge-base core: chunk strategies A / A', embed/cache, vector store, and L0 fact extraction quality checks. |

### 5.1 Member milestone targets

| Member | Milestone target |
|---|---|
| A | A working source-to-snapshot flow with real data and reproducible `snapshot_id` |
| B | Contract + answer format + abstention logic in place |
| C | Query experience and citation rendering are wired to the contract |
| D | QA pack and unambiguous evaluation rules are ready |
| E | Retrieval path and metric runner are working on a frozen snapshot |
| F | At least two chunking strategies produce indexable outputs and pass quality checks |

## 6. Top risks

| # | Risk | Mitigation | Owner |
|---|---|---|---|
| R1 | Corpus short of 8,000 → no A/B separation | Decide before W3; downgrade to qualitative + small-sample and say so | A, D |
| R2 | Copyright breach (full text out, media paraphrase) | C2/C3/C4 + two lines of defense + tests | B, C |
| R3 | Irreproducible evaluation | snapshot binding + run archive | D, E |
| R4 | Scope creep (site, board) | Report stays five sections; board is conditional; D holds the cut list | D |
| R5 | Missing signature page = desk reject | Checked daily from 11/20 | D |
| R6 | Silent corpus shrinkage | selector errors raise; `items > 0` assertion; coverage monitoring | A |
| R7 | Encoding over budget | content budget derived from the model (RUNBOOK §3.4), cache mandatory, measured lane numbers before scaling | F |
| R8 | QA set degenerates into lookups | type A ≤ 1/3; C/D must include cross-source synthesis | D |

## 7. How we track work

- **Gate status** = this document. Update it when a gate passes or slips; do not keep private progress notes.
- **Member issue** = **one persistent issue per member** (`[A]`, `[B]`…) labeled `WPn`, with no milestone
  because it spans every gate. It is a workbench, not a ticket for one task. Do not open a task issue for
  each unit of work.
- **Stage report** = a comment on your own member issue at the end of each stage, in this fixed shape. A
  report missing the evidence line does not count:

  ```
  Stage:     G1a
  Delivered: <file path / command / artifact>
  Evidence:  <the number, or the command output pasted>
  Blocked:   <none / what is blocked, and on whom>
  Next:      <first thing in the next stage>
  ```

  Progress is judged by that evidence line, never by "basically done".
- **Interface issue** = opened only when someone must hand something to someone else, titled
  `[interface] <needer> -> <provider>: <what you need>`. It is temporary: once both sides agree, the
  conclusion is written into `docs/DECISIONS.md` (one line) or into `contracts/`, and only then is the issue
  deleted. **Deleting it before the conclusion is in the repository loses the decision, not just the chat.**
- **Decisions** = `docs/DECISIONS.md`, one line per decision. Verbal agreements do not count.
- **Weekly** = D updates the contribution log; E archives one `run_eval.py` result.
