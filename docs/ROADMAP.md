# ROADMAP — phases, gates and ownership

> This is the execution map. **Task-level detail lives in GitHub issues**; each issue links back to the gate
> it belongs to. If an issue and this document disagree about dates or ownership, this document wins and we fix
> the issue.
> W1 runs 2026-10-27–2026-11-02; the corpus and contract gate closes on 2026-11-02.

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

**G1a outcome — recorded 2026-11-10: PASSED.** All eight checks above are green on `main` at `84c052e`
([PR #47](https://github.com/L01nki1/acgn-chronicle/pull/47) merged 2026-11-10T04:29Z), closing the
2026-11-07 partial record and its carry-over to the 2026-11-09 corpus re-checkpoint. Evidence:
`eval/snapshot_evidence.json` and `eval/source_depth_probe.csv` — size `36,167` items / `36,121` deduped,
span `2023-11-05..2026-11-05`, probe 81 rows across `90d`/`1y`/`2y`/`3y`/`all` recorded before the window
freeze, category balance `anime=22,428` / `manga=6,781` / `game=4,507` / `light_novel=2,149`, idempotency
proven by a forward-vs-reversed rebuild with `byte_diff_count=0` across all seven snapshot files, and
failure visibility via `EmptyBackfillWindowError` + unknown-adapter `KeyError` + per-source `status`.
The contract row passed at the gate (#26 `22d404e`, schema 0.2.3; post-gate #40 `fb02d3a`).
Two recorded non-blockers are accepted explicitly, not silently: `coverage_complete=false` with the eight
structural `coverage_gaps`, and `dedup_calibration_status=provisional_pending_g1b_sweep` — neither is a G1a
threshold and both carry into G1b. #47's E/F review is waived by team agreement and is not a gate condition.
See `docs/DECISIONS.md` 2026-11-10.

**WP4 / B status — 0.2.3 merged; 0.2.4 candidate (2026-11-01).**
[PR #26](https://github.com/L01nki1/acgn-chronicle/pull/26) merged as `22d404e`. C's
[0.2.3 approval](https://github.com/L01nki1/acgn-chronicle/pull/26#pullrequestreview-5724246386)
is bound to `aac6598`; F [acknowledged that delta](https://github.com/L01nki1/acgn-chronicle/pull/26#issuecomment-6552818840).
The actual C/F/E questions, responses and older approvals remain in
[`contracts/G1aReview.md`](../contracts/G1aReview.md). Historical Windows / Python 3.13.5 validation
at `aac6598`: all four gates passed, **165 offline tests passed / 2 opt-in real-e5 skips**. F independently
reproduced 165/2 on Linux. Actual retrieved-set and final-payload/refusal enforcement remain B's G1b
outbound guard, not the model-validation equality check.

Candidate **0.2.4**, rebased onto `main@03a115f`, aligns `source_count` with A's producer design: distinct
contributing `source_key` (feed/adapter) values among accepted manifest-member rows, not publisher or
organization count and not unused registry keys. Field structure is unchanged; the Field/schema
description and reference are synchronized. This semantic delta requires C/E/F review; neither C's
0.2.3 approval nor F's 0.2.3 acknowledgement is reused as acceptance of 0.2.4.
The rebased Windows / Python 3.13.5 frozen dev/store/llm/web four-gate run passed **209 tests / 2 opt-in
real-e5 skips**, including the merged web/Node checks. The earlier 165/2 belongs to `24f6600` before
the WP6 integration; it is not relabelled as this larger suite.

A's [PR #38](https://github.com/L01nki1/acgn-chronicle/pull/38) merged as `9b17396`: its JSONL/member
seam, local entity-registry namespace and canonical full-id/manifest design agree with the contract.
Publisher remains display text; no `publisher_id` is adopted without a separate real organization/alias
registry and propagation/default proposal. B reproduced the synthetic **1,348-byte** manifest and id
`20261028-7576782837e7`, including order independence and an id change after body-hash mutation.
This is design/format evidence as recorded on 2026-11-01: real corpus counts, source-depth probe and
two actual rebuilds were pending at that date. **G1a has since passed** — see the 2026-11-10 outcome
record above and `docs/DECISIONS.md`.

The [PM's existing resolution](https://github.com/L01nki1/acgn-chronicle/issues/4#issuecomment-6502751345)
keeps real curl/filter/refusal/retirement checks with W2's running service. It does not defer the full-id
write-safety prerequisite: real multi-snapshot writes, `--fresh`, drops and reads need the reviewed guard.
Unsafe replacement transitions remain `503` until completion is proven; old sidecars do not establish
that proof. Catalog/completion transitions and active pins/leases are still W2 lifecycle work.

F's [PR #32](https://github.com/L01nki1/acgn-chronicle/pull/32) remains unmerged at `c9dc0e9`.
It retains no-create `bind_existing`, bounds the availability probe and rejects empty bound collections
while allowing a valid restrictive no-match to return empty. E
[approved this head](https://github.com/L01nki1/acgn-chronicle/pull/32#pullrequestreview-5726871154);
B [approved this head](https://github.com/L01nki1/acgn-chronicle/pull/32#pullrequestreview-5727688562)
after **66 identity/store/build tests passed**. The earlier **61-test** B approval belongs to `430aafc`,
not this new head. These checks do not resolve #31 or prove a live host.

C repaired PR #35 at `119a1c4`: the in-flight action coordination, request-count rejection and citation/id
alignment passed B's independent check (**41 web tests / 16 Node cases**); B
[approved this head](https://github.com/L01nki1/acgn-chronicle/pull/35#pullrequestreview-5727688908).
PR #35 merged as `62c26da`; its HTTP mock is a consumer implementation, not a live W2 backend.
PR #37 at `75c3bc1` fixes the `source_type` closed-set example and production-value wording; B
[approved the correction](https://github.com/L01nki1/acgn-chronicle/pull/37#pullrequestreview-5727689267).
PR #37 merged as `03a115f`, so its invariant checklist is now adopted process, with implementation
evidence still required at the corresponding gates.
PR #36 is **closed without merge**; its duplicate #17
record is withdrawn, while the effective ingest-only conclusion is on main through #26.
[Issue #17 is closed as completed](https://github.com/L01nki1/acgn-chronicle/issues/17#issuecomment-6557062553)
with its discussion retained; that interface closure does not pass the corpus or runtime gates.
B's spacing/candidate correction is applied verbatim in PR #33 at `128ab6f`. B may refresh its own
cell after these state changes; other members' entries and the remaining merge preconditions stay with
their owners.

### G1b · 2026-11-09 — indexes and facts

- A / A' / B index builds reproducible; chunk counts explainable and stable across rebuilds
- **no silent truncation**: for every config and a non-empty context header,
  `len(ids(prefix + header + "\n" + chunk)) <= max_seq_length` on every chunk
  (`test_encoded_length_never_exceeds_context`); a full-budget chunk that overflows raises at build time
- encoding on the recorded lane with an identity-keyed cache (model policy: RUNBOOK §3.5); compliance tests green (`pytest tests/compliance -q`)
- L0 fact extraction on a 30-item spot check ≥90% field accuracy, every row carries `chunkId` + `url`

**G1b outbound guard (`KnowledgePort` §3.7):** code candidate in
[#45](https://github.com/L01nki1/acgn-chronicle/pull/45) — not gate evidence; no real corpus, index or
live LLM run yet.

### G2 · 2026-11-21 — Topic 5 complete (official brief, hard gate)

> Topic 5 is a grounded QA system over a custom knowledge base, not a chatbot demo. The gate below reflects the course brief exactly: data prep + retrieval + grounded answer + evaluation, with traceable citations and explicit abstention.

| Requirement | Minimum pass condition | Evidence required |
|---|---|---|
| ① Data prep + ≥2 chunking strategies | A vs A' (optionally A₀) differ, and the difference is explained without confounded variables; **each reported config's retrieval quality is measured on the same QA set and snapshot** | chunk-count table + **per-config retrieval metrics (hit@k / MRR / evidence recall)** + config comparison + snapshot binding |
| ② Knowledge base storage | vector database stores chunk embeddings; collections rebuild independently; metadata filters work | one successful build for each collection + filter tests |
| ③ Retrieval mechanism | question is embedded; relevant chunks are retrieved by semantic similarity; filter combinations work | retrieval logs / tests; p95 under 500 ms locally |
| ④ Grounded generation | `cited_chunk_ids ⊆ retrieved`; answer is generated only from retrieved chunks; unsupported questions refuse instead of guessing; clipping tests green | answer payloads + citation audit + abstention test cases **+ at least one archived real-LLM run, with a second rerun for the tier-2 spread (EVAL_PROTOCOL §8)** |
| ⑤ Evaluation | ≥30 QA items (course minimum is 20); the run archive includes `snapshot_id`, `window`, `run_id` and metrics; three metrics reported; reruns on the same snapshot stay within the documented spread | `eval/runs/<run_id>/meta.json` + QA pack + repeat-run evidence |
| Delivery | 6-page body drafted; contribution statement with **all** signatures on its own page — a separate page after the 6-page body and before the appended slides, not counted in the 6-page limit (course instruction 5: without it the submission is desk rejected); slides attached at the end of the report; individual contributions and ratios (%) stated; 15-minute demo rehearsed; source-code bundle contains no `data/` (`PROJECT.md` invariant 6) | final report bundle + signature page (checked daily from 11/20, R5) |

#### ① Chunking comparison — how it is measured

The course brief asks for at least two chunking strategies **and their impact on retrieval quality**.
Chunk counts alone show that two configs differ; they do not show what that difference costs or buys.
So the comparison is a retrieval experiment, not a counting exercise:

- **Configs.** `CHUNK_VARIANTS` (`core/chunk/variants.py`) is the single table for the A-family chunking
  parameters: **A** (512/64), **A'** (256/0), **A0** (512/0). **A vs A' is mandatory** (it is the pair
  requirement ① names); **A0 is optional** — if it is built, report A vs A0 as well, because that is the cleaner
  overlap-only contrast (same size, 64 vs 0, so only the overlap variable moves). Every reported config must be
  built from this table so the comparison stays on one code path. Config **B** is a separate event-aggregation
  strategy (`core/chunk/strategy_b.py`, still a placeholder), not a `CHUNK_VARIANTS` parameter variant; it is
  **out of scope for G2** and is not part of this comparison.
- **Held fixed.** Same frozen `snapshot_id`, same QA set, same `top_k`, same encoder identity and lane. A run
  that changes any of these is not a chunking comparison (hard check 2).
- **Reported.** Per config: chunk count, and **hit@k / MRR / evidence recall** over the same QA set
  (`core/eval/metrics.py`; run via `scripts/run_eval.py --config "A'"`). `hit@k` is `hit_at_k` at the run's
  recorded `top_k` (default 5, overridable with `--top-k`); the report states the `top_k` it used. The metric
  deltas are explained in the report — e.g. smaller chunks may raise evidence recall while lowering hit@k.
- **Owner split.** F builds the reported indexes (A and A', plus A0 if built) on one snapshot; E guarantees the
  runs are comparable (only `config` varies); D produces the side-by-side table and the impact analysis
  (`core/eval/report.py`).

#### ④ The real-LLM run — what counts and who owns it

Requirement ④'s evidence now includes an archived run that actually called a model. Today
`scripts/run_eval.py` hardcodes `NullAnswerer()` and `eval/runs/` holds only `.gitkeep`, so no such run exists
and none can be produced yet. `meta.llm` is populated from settings even under `NullAnswerer` while
`answers.jsonl` records `status: "not_run"`, so a run can look LLM-configured without being one.

- **Definition.** A run counts as a real-LLM run only when every `answers.jsonl` row has `status: "answered"`,
  a non-empty `answer`, `meta.json` records the actual provider, model, version and temperature, **and the run
  shows at least one returned provider completion**. The seed is
  recorded as unset when the request carries none — E must populate `meta.llm` from the live answerer rather than
  from settings, because `scripts/run_eval.py` currently writes `"seed": 0` from settings even under
  `NullAnswerer()`, which is the same "looks LLM-configured without being one" problem described above.
- **A refusal is not a completion.** WP4 returns a non-empty, `answered`-shaped refusal *before*
  `client.complete()` in two cases: empty retrieval (`ABSTAIN_NO_EVIDENCE`) and a top score below
  `min_retrieval_score` (`ABSTAIN_RETRIEVAL_BELOW_THRESHOLD`), both in
  `core/ground/generate.py::generate_grounded_answer_run`. Those refusals are correct behaviour and stay in the
  archive, but a run made only of them called no model. The archive therefore records, per answer row, whether a
  provider completion was actually returned, and the run-level validator requires at least one `true`: a run with
  zero completions is not a real-LLM run even when every row says `answered`.
- **Owner.** E wires the live answerer into `scripts/run_eval.py`; B owns the run-record contract side (which
  fields must be archived).
- **Fallback.** If the provider is unavailable before the G2 window closes, the gate records the run as
  **blocked with the reason**, and ④ is reported as partially met rather than passed on design alone. The
  retrieval half of the evaluation is unaffected and still reportable.

#### G2 hard checks (must be green before G2 closes)

1. `snapshot_id` is bound to the corpus and every run uses a frozen snapshot.
2. At least two chunking strategies are implemented and compared, and the comparison reports **retrieval quality per config** (hit@k / MRR / evidence recall on the same QA set and snapshot), not chunk counts alone; it is not confounded by a different corpus, window, encoder or `top_k`.
3. Retrieval is vector-based and uses the same `snapshot_id` as the question.
4. Answers are grounded in retrieved chunks; citations are reproducible and every cited chunk is actually in the retrieved set.
5. Unanswerable questions produce `abstained=true` and a refusal reason instead of a guessed answer.
6. Evaluation uses a QA set with at least 30 items, balanced across the required QA types, and reports the metric definitions and rerun spread.
7. The final report contains the required task-5 narrative: method, experiments, discussion, references, and the separate contribution/signature page.
8. `cited_chunk_ids` equals the chunk ids of `citations` (a validator, not a convention), and every citation's `chunk_id` is in the retrieved set.
9. Filter conversion is single-point: `RetrievalFilters` → store `where` happens in one function, covered by a test for each filter family and for `[]` (means "no filter", not "match nothing").
10. Every per-item failure or quality claim in the report's discussion section traces to a specific `qa_id` in the archived `answers.jsonl` of the run it cites; aggregate metric claims trace to that run's `metrics.json`. No new archive file is required: `answers.jsonl` already records `question`, `retrieved_chunk_ids`, `retrieved_urls`, `answer`, `citations`, `cited_chunk_ids` and `abstained` per item, and the frozen QA set carries the `gold_evidence` urls needed to judge hit/miss. Failure classification is an analysis dimension of the report, not a per-run artifact. D produces this traceability in `core/eval/report.py`, which is still a scaffold.

#### G2 stretch items (nice-to-have, cut first if schedule slips)

- BM25 + RRF hybrid retrieval — judged on **MRR headroom** as pre-registered (issue #14 P3), never on hit@k alone
- multi-source daily digest sample
- more aggressive query-time filtering and latency tuning
- board / visual demo page — **cut first** if anything slips

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
| B | Freeze `ports.py` and the exported contract, including answer/citation/refusal response semantics and the grounded-prompt design. Runtime generation follows in W2, as in the weekly timeline and Issue #4. |
| C | Stand up the retrieval-facing interface and the front-end citation contract: query page, citation display, and clipping-compatible rendering. |
| D | Set up evaluation-first work: QA type A/B/C/D construction, runner skeleton, run records, and the contribution/signature plan for the final report. |
| E | Own retrieval and evaluation backend: filters, retrieval correctness, run_id and metric rules, and the measurement of latency / rerun spread. |
| F | Own the knowledge-base core: chunk strategies A / A', embed/cache, vector store, and L0 fact extraction quality checks. |

### 5.1 Member milestone targets

| Member | Milestone target |
|---|---|
| A | A working source-to-snapshot flow with real data and reproducible `snapshot_id` |
| B | W1: contract, answer format and abstention design in place; W2: implement the grounded answer and abstention path |
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
- **A finding about code that is already written** goes to the owner's member issue — unless the file sits
  on an open branch, in which case it goes on that PR as a line comment (ONBOARDING §3 item 8). It never
  becomes a task issue: a fix proposed against `main` for a file someone is rewriting is a conflict plus a
  second source of truth.
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
