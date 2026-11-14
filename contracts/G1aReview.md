# G1a contract review preparation

Owner: B. Candidate: schema **0.2.5**, rebased onto `main@53d611b` (Python 3.13 / locked uv),
including the merged WP6 consumer and process-checklist corrections.
PR [#26](https://github.com/L01nki1/acgn-chronicle/pull/26) merged as `22d404e`; A's design handoff
[#38](https://github.com/L01nki1/acgn-chronicle/pull/38) merged as `9b17396`. This candidate aligns
`source_count` semantics with that producer design (§5.5), without adding fields. The contract reference is
[KnowledgePort.md](KnowledgePort.md).
Gate status stays in [`docs/ROADMAP.md`](../docs/ROADMAP.md); member progress stays in
[Issue #4](https://github.com/L01nki1/acgn-chronicle/issues/4).

## 1. Consumer review agenda

The following agenda entries are **B-authored prompts for review**, not questions or approvals supplied by C/F/E.
Issue #4 requires each of C, F and E to raise at least three consumer-side questions and for the actual
questions and resolutions to be recorded in the PR. C's C1–C6, F's A1/A2/Q1–Q3 and E's E1–E3 now have
actual source records and B's answers in §5. C approved 0.2.2 at `c8c3e88` and separately
[approved the 0.2.3 delta at `aac6598`](https://github.com/L01nki1/acgn-chronicle/pull/26#pullrequestreview-5724246386).
F separately acknowledged 0.2.3 (§5.4). Those records do not approve the new 0.2.4 `source_count`
semantics; C/E/F review of that delta is pending.
The agenda itself is not evidence that a consumer asked or accepted a question.

| Consumer | Topics to check against the candidate | Actual questions and resolution |
|---|---|---|
| C — web / WP6 ([#5](https://github.com/L01nki1/acgn-chronicle/issues/5)) | (1) Can the refusal fixture render separately from `503`? (2) Are empty multi-selects sent as no filter, and is `410` shown without silently changing snapshots? (3) Can `ChunkExcerpt` render a clipped quote and attribution while the request's snapshot is carried in the view and cache key? | C1–C6 accepted; 0.2.2/0.2.3 approved at their recorded heads (§5.3–§5.4); 0.2.4/0.2.5 review pending |
| F — KB / Isla consumer ([#8](https://github.com/L01nki1/acgn-chronicle/issues/8)) | (1) Can media-id membership and the index resolve against the same retained snapshot? (2) Does every retrieved chunk retain the citation provenance B needs? (3) Is the provisional `DailyDigest` isolated behind one consumer adapter, with no dependency on the removed `episode_table`? | F verified `30f746d` and acknowledged 0.2.3 (§5.4); Q2/Q3 remain scoped in §5.2/§6; 0.2.4/0.2.5 review pending |
| E — retrieval / runner ([#7](https://github.com/L01nki1/acgn-chronicle/issues/7)) | (1) Are ANY-within-list / AND-between-families and empty-list defaults implementable? (2) Can UTC day/datetime bounds and a non-empty media selection with no matches remain restrictive? (3) Can the runner pin its snapshot and distinguish unknown `404` from retired `410`, including while a rebuild or retention cleanup occurs? | E verified the answers at `c1949f1`; its G1b/W2 work remains staged (§5.2); E confirmed the 0.2.5 score wording in its PR #67 review (§5.6); 0.2.4 review pending |

C is WP4's cross-reviewer according to ROADMAP §4. AI checks and passing CI do not substitute for that
review or the consumer questions. Record links to real replies and an explicit resolution per question
when available; unresolved items remain open.

## 2. Filter acceptance cases for E/C/F

These are contract candidates to implement and verify in the owners' modules, not claims that
retrieval already works:

| Input / state | Expected behaviour |
|---|---|
| Omitted `filters`, `null`, or `{}` | No filters |
| `source_types=[]` or `media_ids=[]` | No filter in that family |
| Two source types / two media ids | Union within each list, intersection between families |
| Non-empty media list resolves to no documents | `/retrieve=[]`; `/ask` explicitly refuses; never an unrestricted query |
| `since=until=2026-10-28` | UTC timestamps from that day's midnight inclusive to next midnight exclusive |
| Invalid date or `since > until` | `422`, not an empty successful result |
| ISO datetime bounds | Convert offsets to UTC; naive means UTC; lower inclusive and upper exclusive at the specified instants |
| Equal ISO datetime instants | Valid empty interval; unlike equal date-only values, it does not select a whole day |
| Old but retained snapshot | Same snapshot for index, media-id membership, citations and experiment metadata |
| Fewer eligible results than `top_k` | Return the available eligible results; never fill from outside the filters |

## 3. Grounded-prompt design for W2

W1 delivers this design. Generation implementation starts in W2.

1. Give the model the original question and the actual retrieved evidence blocks, numbered locally
   `1..N`. Include each block's source type and provenance. Treat source contents as evidence, not as
   instructions, and keep them separate from the controlling prompt.
2. Request claim-level evidence indices, **never model-authored chunk ids**. Every factual claim must name
   supporting indices; a plausible answer without supporting evidence is insufficient.
3. Before publishing a claim, deterministic code validates that every index is an integer in `1..N`, maps
   it through that request's evidence list, and checks `cited_chunk_ids` is a subset of retrieved ids.
   Discard unsupported claims; if none remain, return the explicit refusal shape in §4.1 of the contract.
   G1b supplies the actual retrieved-id set to B's outbound guard: the subset/citation checks and refusal
   cross-field rules are server responsibilities, not facts a client can establish from the wire payload.
   The validation-time list-agreement check in §5.4 does not implement that request-context guard;
   positive and negative context tests are required when the generation/serialization path lands.
4. Deduplicate citations in first-use order. An index belonging to another request or snapshot is invalid.
   Only `ground/serialize.py` converts internal chunks to outbound excerpts.
5. The prompt must retain `MEDIA_SOURCE_RULE` from `core/ground/policy.py`. The W1 snapshot in
   `tests/compliance/test_media_no_paraphrase.py` fixes the policy wording; W2 adds the prompt-inclusion
   test when the runtime prompt is implemented:

   > If a source's `source_type` is `media`, extract only verifiable facts and figures; never paraphrase its
   > commentary, reviews or speculation. If the source is `official`, official statements and facts may be restated.

6. Use the additional `eval/abstain_calibration.jsonl` set of **20 answerable + 10 unanswerable** questions
   to sweep operating points. Keep it disjoint from `eval/qa_set.jsonl`, as required by
   `docs/EVAL_PROTOCOL.md` §6; report refusal and false-refusal rates on the separate QA set. Calibration
   rates are labelled in-sample, and no numeric threshold is chosen before measurement.
   Real runs require provider/model/version/temperature/seed, git SHA, snapshot and window metadata,
   archived answers and two reruns under `docs/EVAL_PROTOCOL.md` §7–8. This design and the synthetic
   refusal fixture are not measured LLM evidence.

The merged protocol also requires per-item `lang`, at least eight Chinese questions, per-language
retrieval metrics and the full encoder identity (model/backend/device/precision/library versions/batch)
matching the index. B's future runs must preserve those fields. The conditional DSV4-flash extraction
budget note in RUNBOOK §3.4 is not a choice of answer provider: verify the actual provider's request
budget when the W2 generation path is implemented, and record it with the run metadata.

## 4. W2 runtime evidence dependencies

At `main@03a115f`, F's chunk/embed/store/facts/build pipeline, A's design handoff and C's HTTP mock
consumer are merged. `web/api.py` serves contract-shaped mock data; `core/ingest/snapshot.py` and
`core/retrieve/vector.py` remain scaffolds on main. The mock is not a live retrieval/generation backend,
snapshot catalog or pin/lease coordinator. Real backend endpoint captures therefore remain W2
work under the [PM resolution](https://github.com/L01nki1/acgn-chronicle/issues/4#issuecomment-6502751345),
not missing G1a HTTP receipts. G1a still requires the schema/doc/CI artifacts and actual consumer review.

| Required evidence | Dependency / owner | What B will verify once supplied |
|---|---|---|
| `GET /snapshots`, `GET /snapshot`, `POST /retrieve` | C's HTTP host; A's snapshot metadata; E/F's queryable index | Capture real status, headers and body; validate successful bodies against the wire types |
| `POST /ask` answer and refusal | C's host and B's W2 generation path | Per-claim citations use retrieved chunks; refusal carries no invented support |
| Unknown `404`, retired `410`, unserved `503` | A/C/F's read-side catalog/host guard | Validate matching completed `build.json` + `identity.json`; check availability before `bind`; never create a collection to determine whether an id exists (KnowledgePort §3.2) |
| Rebuild / retention during a long request | A/E/F snapshot pinning and cleanup | A request keeps its selected snapshot until completion; consumers do not silently substitute current |
| `GET /daily` | Conditional G3, C | Capture only when that stage is active; do not fabricate a W1 response |

When those implementations exist, replace illustrative examples with actual captures. The PM explicitly
moved the four former §8 runtime checks to W2; this is recorded in DECISIONS with the source comment.
Fixtures remain design evidence and are not relabelled as HTTP or LLM receipts.

## 5. Actual consumer questions and B's responses

Sources: [F's A1/A2/Q1–Q3](https://github.com/L01nki1/acgn-chronicle/pull/26#issuecomment-6502273614),
[E's E1–E3](https://github.com/L01nki1/acgn-chronicle/pull/26#issuecomment-6517534109).
B's responses below distinguish implemented contract validation from G1b/W2 design commitments.
They are not an approval on behalf of C or a claim that live verification has run.

| Item | B's response encoded in this revision | Evidence / remaining implementation |
|---|---|---|
| F A1 | Current backend is Chroma 1.x with native `media_ids: int[]`, `$contains` / `$or`; reverse lookup is a long-list fallback | KnowledgePort §3.3 matches merged `media_filter` |
| F A2 | Never-known id → `404`; known completed snapshot retired from service → `410`; an absent id alone does not prove retirement | `SnapshotInventory.retained` Field description and exported schema updated with both codes |
| F Q1 | Quote construction, serialization and normal attribute assignment enforce ≤200 Unicode code points now; the request-context citation/refusal guard is a mandatory G1b delivery in `ground/` | Three assignment regressions reproduced before the fix; G1b must test actual retrieved-set containment and refusal consistency |
| F Q2 | Matching successful `build.json` + `identity.json` are initial known-build evidence for the served configuration; validate them before any build-side `bind` | KnowledgePort §3.2 defines `404` / `410` / `503`; read guard/catalog/pins are W2 lifecycle work |
| F Q3 | Pin `YYYYMMDD-<12 lowercase SHA-256 hex>`, with date from frozen `window_end` UTC day, not rebuild time | KnowledgePort §3.1; A's canonical minting and F's full-id namespace guard remain implementation handoffs (§6) |
| E1 | Accepted: omitted/null/empty means no filter in that family; OR within, AND between; restrictive no-match → empty; insufficient eligible results are never padded from outside filters | KnowledgePort §3.3–§3.4; G1b where-builder tests belong to E |
| E2 | Accepted with explicit edges: UTC whole days; ISO datetimes accepted with offset conversion / naive UTC; half-open bounds; invalid or reversed range → `422`; equal datetime instants select an empty interval | KnowledgePort §3.3; reversed date-only days are checked before expanding the upper day |
| E3 | Accepted as the target contract: one full snapshot/config/build identity per run, `404` unknown / `410` retired / `503` not serving, no fallback, cache tuple includes the snapshot; active pins survive prune | KnowledgePort §3.2/§6; current `prune` does not implement a run lease, so A/C/F must coordinate it before the W2 service relies on it |

### 5.1 Corporate ingest tag (#17)

Question raised for F: does the Capcom corporate/IR tag reach the fact zone, or stay ingest-side?
F explicitly endorsed **option 1** in
[this reply](https://github.com/L01nki1/acgn-chronicle/issues/17#issuecomment-6486809283): the tag is used
only by the pre-extraction filter, and there is no query-time consumer through G1b.

B's resolution is **option 1**: keep `source_type=official` and `ingest_category=corporate` internal;
add no `SourceType` label and no `FactUpdate` field. A later wire consumer needs an explicit
Optional-with-default addition and C/F notice. The conclusion is recorded in DECISIONS and
KnowledgePort §3.8 and is now on main through PR #26. [Issue #17 is closed as completed](https://github.com/L01nki1/acgn-chronicle/issues/17#issuecomment-6557062553),
with discussion retained. This closes the interface arbitration, not the separate corpus/runtime gate.

PR #36 at `ce5b788` was closed **without merge**, withdrawing its duplicate record. B's historical
[category/source-label finding](https://github.com/L01nki1/acgn-chronicle/pull/36#pullrequestreview-5722388244)
is not reported as a merged correction. The effective option-1 decision is the row adopted through #26.

This historical resolution is preserved separately from F's newer multi-question review above. F's and
E's verification and C's version-bound 0.2.2 and 0.2.3 approvals are recorded below.
W2 live checks are separately scoped.

### 5.2 Follow-up verification (2026-10-31)

E [verified the answers at `c1949f1`](https://github.com/L01nki1/acgn-chronicle/pull/26#issuecomment-6524709359):
E1 is confirmed, E2 is confirmed in principle, and E3 is confirmed as the target contract. E owns the
remaining date-boundary tests and exact timestamp comparisons in the where-builder. Integer `published_ts`
storage does not reject otherwise valid fractional-second ISO bounds: preserve the boundary when comparing
it with stored timestamps, as §3.3 requires. E's runner will check the selected build identity
and collection against `identity.json`. Those implementation commitments are not verified remote code.
The [#31](https://github.com/L01nki1/acgn-chronicle/issues/31) namespace guard is a prerequisite for
multi-snapshot runs; pin/lease coordination remains W2. E has marked B3/B4 resolved in #30.

D's [compliance review of `c1949f1`](https://github.com/L01nki1/acgn-chronicle/pull/26#pullrequestreview-5715924995)
reported no blocking findings in that scope and 20 compliance tests, eight contract tests and schema
sync passing. Its fixture-format suggestion is addressed by using `20261028-abcdef012345` in the
serializer compliance fixture. C4 wording changes must keep the policy constant, its snapshot test and
the COMPLIANCE quote in sync. D's review does not cover contract semantics or replace C's cross-review.

F [verified `30f746d`](https://github.com/L01nki1/acgn-chronicle/pull/26#issuecomment-6537520129),
confirming A1/A2/Q1 and the fractional-second clarification. Its Q2 probe left old successful sidecars
beside an empty replacement collection after an interrupted fresh build. KnowledgePort §3.2 now requires
completion evidence for the present collection generation and `503` throughout an unsafe transition;
the host must open existing collections without creating them. This is the W2 lifecycle contract, not
a claim that its transition/pin mechanism exists. Q3's candidate implementation remains under review (§6).

D's [WP5 consumer-computability check](https://github.com/L01nki1/acgn-chronicle/pull/26#issuecomment-6538104805)
found no metric-input gap. F/E/D's scoped verification does not replace C's approval of the revised
candidate or pass the separate corpus gate.

### 5.3 C's actual questions and B's resolutions

Sources: C's [C1–C5 COMMENT cross-review at `2832e0b`](https://github.com/L01nki1/acgn-chronicle/pull/26#pullrequestreview-5721850600),
the [C6 addendum](https://github.com/L01nki1/acgn-chronicle/pull/26#issuecomment-6542128712), and
[D's C4 consumer position](https://github.com/L01nki1/acgn-chronicle/pull/26#issuecomment-6540790847).
B accepted all six below. C supplied the required own consumer questions and found no blocking
structural issue. C's [detailed verification](https://github.com/L01nki1/acgn-chronicle/pull/26#pullrequestreview-5723365234)
and [APPROVE review](https://github.com/L01nki1/acgn-chronicle/pull/26#pullrequestreview-5723391766)
apply to the 0.2.2 candidate at `c8c3e88`; they do not approve later semantic changes.

| Item | Accepted resolution | Remaining owner / verification |
|---|---|---|
| C1 — quote unit | The 200-unit limit counts Unicode code points, not UTF-16 code units; JS clients count code points rather than use `.length` as the unit | B updates the Field/schema descriptions and reference; C retains client clipping |
| C2 — error shape | Every non-2xx contract response has `{"detail": string}` with extra fields allowed; `503` supplies `Retry-After` as integer delay-seconds | C's host enforces the envelope and header; W2 tests use the real backend |
| C3 — validation boundary | Invalid body shape/type/enum is `400`; structurally valid requests with invalid date/datetime syntax or reversed filter bounds are `422` | C maps HTTP errors; E implements semantic filter validation |
| C4 — serving identity | G1a/G2 use one active config/encoder identity per deployment, supplied through deployment/session context (CLI/env), outside the wire. Restart or identity/config change invalidates the client cache namespace | C/E configure and invalidate consumers. If W2 needs simultaneous configs, agree an Optional field or equivalent header before the 11/02 freeze; this revision adds neither |
| C5 — inventory membership | A ready host's `current` is queryable and included in `retained`; an unsafe transition returns `503` rather than an inconsistent successful inventory | A/C/F enforce lifecycle readiness; C may retain its defensive union |
| C6 — `top_k` minimum | `top_k` is a positive integer, minimum 1 (default 5); non-positive values are request errors (`400`), not empty success | B enforces the minimum in the model/exported schema; C's request-validation handler maps failure to `400` |

The 0.2.2 candidate passed all four local gates on Windows / Python 3.13.5: **158 passed**, with two
explicitly opt-in real-e5 checks skipped. Three C6 cases failed before the fix: zero/negative counts
were accepted and the exported minimum was absent. A separate combination check of C's PR #35
`e30f6d4` plus B's 0.2.2 port model passed seven HTTP-adapter cases for C2/C3/C6. That combination is
not reported as the unmodified PR #35 result or as a running W2 backend.

B also reviewed [WP6 PR #35](https://github.com/L01nki1/acgn-chronicle/pull/35) at `e30f6d4`.
Its [independent review](https://github.com/L01nki1/acgn-chronicle/pull/35#pullrequestreview-5722387980)
reproduced a P2 finding: an example-question action is dropped while a request
is busy. C's [repair at `119a1c4`](https://github.com/L01nki1/acgn-chronicle/pull/35#pullrequestreview-5727065603)
coordinates queued actions and adopts the merged `top_k` rule; citation/id lists are aligned after
deduplication. B's independent recheck passed **41 web tests and 16 Node cases**;
[B approved that head](https://github.com/L01nki1/acgn-chronicle/pull/35#pullrequestreview-5727688908).
PR #35 then merged as `62c26da`; this mock consumer check does not establish W2 retrieval or generation.

PR #37 at `75c3bc1` corrects B's closed-set example to `source_type` and the limit wording to
the actual production value/payload. [B approved those corrections](https://github.com/L01nki1/acgn-chronicle/pull/37#pullrequestreview-5727689267).
PR #37 then merged as `03a115f`; its adopted G1b/G2 criteria are now the main-branch process baseline.
That adoption does not assert that the corresponding runtime implementations or real-corpus evidence exist.

### 5.4 D's follow-up proposals and the 0.2.3 boundary

Source: D's [two contract proposals](https://github.com/L01nki1/acgn-chronicle/pull/26#issuecomment-6544652493).
B accepts validation-time agreement between `GroundedAnswer`'s two citation lists: construction or
fresh model validation checks `sorted(c.chunk_id for c in citations) == sorted(cited_chunk_ids)`.
This is multiset equality, including duplicate counts, not an ordering or deduplication rule. It does
not guarantee consistency after arbitrary later mutation. G1b must freshly validate the final payload
and check it against the actual retrieved set, enforce refusal consistency, and deduplicate in first-use
order. The model check alone establishes none of those request-context guarantees.

B deferred the proposed Optional `publisher_id` pending a real organization registry, aliases and field
propagation through [#23](https://github.com/L01nki1/acgn-chronicle/issues/23). A's merged
[INGEST_CONTRACT.md](../docs/INGEST_CONTRACT.md) confirms that acquisition-channel `source_key` is not
publisher identity, and no organization registry is supplied. `publisher` remains open display text;
its examples are not an enum. No `publisher_id` is adopted. A later field needs its own versioned
organization/alias definition, propagation/default rules and Optional contract review.

D also [ratified the metric denominator rule](https://github.com/L01nki1/acgn-chronicle/issues/30#issuecomment-6542499030):
`deduped_count` records corpus provenance, not the denominator of any QA metric. Each metric reports
its own eligible-item denominator and exclusions. B's contract description follows that ruling; D/E
retain ownership of metric definitions and output.
The corresponding old PROJECT invariant/glossary wording is D's pending PR #28 sync, not a claim that
this contract-only correction has updated every project document.

The 0.2.3 revision passed all four local gates on Windows / Python 3.13.5: **165 passed**, two opt-in
real-e5 checks skipped. Five list-mismatch cases failed before the validator and pass after it.

Those 0.2.3 changes were adopted through PR #26 (`22d404e`). C
[approved the delta at exact `aac6598`](https://github.com/L01nki1/acgn-chronicle/pull/26#pullrequestreview-5724246386),
after separately approving 0.2.2 at `c8c3e88`; neither review covers 0.2.4.

The [0.2.3 publication reply](https://github.com/L01nki1/acgn-chronicle/pull/26#issuecomment-6546162341)
already notified C and F (`@inscripoem`) and described the validation/denominator/publisher decisions.
F subsequently [acknowledged and verified 0.2.3](https://github.com/L01nki1/acgn-chronicle/pull/26#issuecomment-6552818840),
testing `aac6598` and inspecting the record-only `053ebc4` on Linux: **165 passed / 2 opt-in skips**.
This acknowledgement is not transferred to 0.2.4.

B's [single-cell correction](https://github.com/L01nki1/acgn-chronicle/pull/33#discussion_r4433017516)
is applied verbatim in PR #33 at `128ab6f`; it no longer awaits first application. Later state changes
are for B to refresh in its own cell, preserving other members' entries.

### 5.5 Candidate 0.2.4 — contributing acquisition-source count

A's merged producer design defines `source_key` as a stable adapter/feed key and counts distinct keys
with accepted rows. That conflicts with the earlier B description of `source_count` as publishers.
This candidate defines `source_count` as the number of distinct contributing `source_key` values among
accepted manifest-member rows. It is neither a publisher/organization count nor the size of a registry
that includes unused keys. Field structure is unchanged; the Field/schema description and reference
must carry the same definition. C/E/F semantic review is pending; earlier approvals remain version-bound.

The original 0.2.4 tree at `24f6600` passed **165 offline tests / 2 opt-in real-e5 skips**. After PR #35
and #37 merged, B rebased onto `03a115f` and reran lint, format, schema sync and the expanded suite:
**209 passed / 2 opt-in real-e5 skips**, Windows / Python 3.13.5, frozen dev/store/llm/web. The suite also
executes both Node contract/queue test files. The count clarification adds no field or counting
implementation: A's runtime producer will derive the integer from the frozen member rows.

B's check of A's design confirms the JSONL/member seam, canonical manifest/full-id format and local
entity-registry namespace are consistent with the contract. Its independent synthetic check reproduced
**1,348 canonical bytes**, id `20261028-7576782837e7`, order independence and an id change after body-hash
mutation. Those checks establish format agreement only: no real corpus, source-depth probe or two actual
rebuilds have been produced or verified by this work. A/D's possible partial-gate plan remains undecided;
the existing gate date, thresholds and evidence requirements remain in force.

### 5.6 Candidate 0.2.5 — score comparability scoped to one identity

The merged calibration utility (`core/ground/calibration.py`, merged as `614e746`) selects one
abstention threshold from labelled per-item top retrieval scores. Applying that threshold compares one
request's best score against a value derived from other requests' scores, which the previous wording
forbade outright ("ordinal within one response only: never compare across queries"). E/WP3 confirmed the
replacement wording in its review of PR #67 and asked B to draft the contract sentence while D records
the run-record keys.

This candidate states that scores are comparable across queries only when the snapshot, chunk
configuration, retrieval mode, encoder identity, top-k and over-fetch are identical (per-query filters
may differ), and that the comparability exists solely for abstention-threshold calibration: scores are
never comparable across identities, snapshots, chunk configurations or retrieval modes, and are never
averaged as a quality metric. The sanctioned use is `core/ground/calibration.py` with the strict
`score < threshold` rule; the threshold is recorded with the identity that produced it and re-calibrated
when any of those fields changes (`docs/RUNBOOK.md` §3.7).

Field structure is unchanged: no field is added, renamed, retyped, made required or optional, or given a
default. The exported schema diff is one `description` line plus the version string, and a structural
comparison with every `description` removed differs only in `schema_version`.

`SCHEMA_VERSION` moves 0.2.4 to 0.2.5 following the 0.2.4 precedent (`fb02d3a`), where a
description-only semantic clarification was versioned because C/E/F approvals are version-bound (§8.1).
E confirmed the wording in the PR #67 review. **C (web) and F (Isla contact) notification and re-review
are pending**; this entry does not claim their approval, and D ratifies the `EVAL_PROTOCOL` §7
run-record keys separately.

Local gate result on the candidate tree: `python scripts/check.py` four gates green (ruff check, ruff
format, contract schema in sync, offline pytest) on Windows / CPython 3.13.5 with the locked
`dev,store,llm,web` extras (the web extra is installed so `tests/web` runs); direct pytest run
**701 passed / 2 skipped** (the two opt-in real-e5 skips;
`check.py` runs pytest at `-qq`, so its own summary line is suppressed).

## 6. Snapshot identity handoff to A/F

The canonical full id is date-first and hyphenated (§3.1). A's merged PR #38 defines the canonical
manifest and minting design; its real producer implementation and rebuild evidence remain pending.

The merged store currently names a collection from `snapshot_id[:12]`. A deterministic probe with two
valid full ids, `20261028-abc000000000` and `20261028-abc111111111`, returns the same
`snap_20261028-abc_A`. The shortened name therefore cannot itself prove a snapshot's identity.
Before any real multi-snapshot build, `--fresh`, `drop_collection` or read binding touches storage,
F must use an injective full-id mapping or reject prefix aliases against retained build records. The
guard must run before deletion as well as before `get_or_create`/upsert: `--fresh` currently drops by
the same shortened key, so an alias can erase another snapshot even without an HTTP host. A retained
read must also match the selected build/config/
encoder identity; it must never silently open another full id's collection. The store implementation is
F-owned and is handed off explicitly in the PR response rather than edited here.

Date-first makes the id readable but does not make its hash a clock. Retention's publication order must
come from validated build/catalog timestamps, especially when several snapshots share a `window_end`
day; a missing age record must not be replaced by an assumed ordering of those hashes.

F published [PR #32](https://github.com/L01nki1/acgn-chronicle/pull/32) for #31. B's historical check of
`fbc609d` on Windows / Python 3.13.5 ran `test_snapshot_identity_safety.py`, `test_store_chroma.py` and
`test_build_index.py`: **52 passed**. That candidate covered full-id isolation, pre-mutation ownership,
verified legacy ownership, same-owner full/legacy ambiguity and publication-time pruning; its missing
no-create read binding was recorded in E's [requested change](https://github.com/L01nki1/acgn-chronicle/pull/32#pullrequestreview-5720797378).

F's [update at `430aafc`](https://github.com/L01nki1/acgn-chronicle/pull/32#issuecomment-6539237186)
adds `bind_existing` and the missing-storage/collection, legacy-resolution and binding-preservation
regressions. B reran those three suites on Windows / Python 3.13.5: **61 passed, no skips**. Build-side
`bind` may create; read-side `bind_existing` must not. B's [consumer acceptance review](https://github.com/L01nki1/acgn-chronicle/pull/32#pullrequestreview-5722387709)
approves that historical head. F's [new `c9dc0e9` candidate](https://github.com/L01nki1/acgn-chronicle/pull/32#issuecomment-6552818589)
bounds the availability probe and rejects an empty bound collection while preserving valid filtered
no-match results. E [approved `c9dc0e9`](https://github.com/L01nki1/acgn-chronicle/pull/32#pullrequestreview-5726871154);
B [approved that exact head](https://github.com/L01nki1/acgn-chronicle/pull/32#pullrequestreview-5727688562)
after its three identity/store/build suites passed **66 tests, no skips**, on Windows / Python 3.13.5.
The prior 52/61 results remain tied to their older heads. PR #32 remains unmerged.
These offline results do not establish the W2 completion/catalog/lease mechanism or resolve #31.
