# G1a contract review preparation

Owner: B. Candidate: schema **0.2.1**, based on `main` at `e58b21d`, including the contract from merged
PR [#11](https://github.com/L01nki1/acgn-chronicle/pull/11) and the evaluation/encoder-identity protocol
through [#15](https://github.com/L01nki1/acgn-chronicle/pull/15). The contract reference is
[KnowledgePort.md](KnowledgePort.md).
Gate status stays in [`docs/ROADMAP.md`](../docs/ROADMAP.md); member progress stays in
[Issue #4](https://github.com/L01nki1/acgn-chronicle/issues/4).

## 1. Consumer review agenda

The following agenda entries are **B-authored prompts for review**, not questions or approvals supplied by C/F/E.
Issue #4 requires each of C, F and E to raise at least three consumer-side questions and for the actual
questions and resolutions to be recorded in the PR. That requirement remains open. F's actual #17 input
and the agreed option are recorded separately in §5.

| Consumer | Topics to check against the candidate | Actual questions and resolution |
|---|---|---|
| C — web / WP6 ([#5](https://github.com/L01nki1/acgn-chronicle/issues/5)) | (1) Can the refusal fixture render separately from `503`? (2) Are empty multi-selects sent as no filter, and is `410` shown without silently changing snapshots? (3) Can `ChunkExcerpt` render a clipped quote and attribution while the request's snapshot is carried in the view and cache key? | Pending C's own questions and reply |
| F — KB / Isla consumer ([#8](https://github.com/L01nki1/acgn-chronicle/issues/8)) | (1) Can media-id membership and the index resolve against the same retained snapshot? (2) Does every retrieved chunk retain the citation provenance B needs? (3) Is the provisional `DailyDigest` isolated behind one consumer adapter, with no dependency on the removed `episode_table`? | One actual input (#17) recorded in §5; full consumer review remains open |
| E — retrieval / runner ([#7](https://github.com/L01nki1/acgn-chronicle/issues/7)) | (1) Are ANY-within-list / AND-between-families and empty-list defaults implementable? (2) Can UTC inclusive day bounds and a non-empty media selection with no matches remain restrictive? (3) Can the runner pin its snapshot and distinguish unknown `404` from retired `410`, including while a rebuild or retention cleanup occurs? | Pending E's own questions and reply |

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
| Old but retained snapshot | Same snapshot for index, reverse lookup, citations and experiment metadata |
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

## 4. Real endpoint evidence dependencies

At `main@e58b21d`, the following files are scaffold placeholders: `web/api.py`,
`core/ingest/snapshot.py`, `core/store/__init__.py`, and `core/retrieve/vector.py`. There is no application
entry point or implemented snapshot lifecycle from which to capture the requested HTTP responses.
F's implementation chain (#18–#24) is still under review at this inspection; it is not a merged HTTP host.

| Required evidence | Dependency / owner | What B will verify once supplied |
|---|---|---|
| `GET /snapshots`, `GET /snapshot`, `POST /retrieve` | C's HTTP host; A's snapshot metadata; E/F's queryable index | Capture real status, headers and body; validate successful bodies against the wire types |
| `POST /ask` answer and refusal | C's host and B's W2 generation path | Per-claim citations use retrieved chunks; refusal carries no invented support |
| Unknown `404`, retired `410`, unserved `503` | A/E/F's snapshot lifecycle exposed by C | Distinguish never-known from retired ids; verify retained snapshots remain queryable; record `Retry-After` for `503` |
| Rebuild / retention during a long request | A/E/F snapshot pinning and cleanup | A request keeps its selected snapshot until completion; consumers do not silently substitute current |
| `GET /daily` | Conditional G3, C | Capture only when that stage is active; do not fabricate a W1 response |

When those implementations exist, replace illustrative examples with actual captures. Do not create a
dummy HTTP service or relabel fixture JSON as a real endpoint receipt to close G1a. Whether the stage can
freeze before W2's `/ask` exists needs an explicit owner/consumer resolution in Issue #4; B cannot silently
waive its current real-curl requirement.

## 5. Actual consumer input: corporate ingest tag (#17)

Question raised for F: does the Capcom corporate/IR tag reach the fact zone, or stay ingest-side?
F explicitly endorsed **option 1** in
[this reply](https://github.com/L01nki1/acgn-chronicle/issues/17#issuecomment-6486809283): the tag is used
only by the pre-extraction filter, and there is no query-time consumer through G1b.

B's resolution is **option 1**: keep `source_type=official` and `ingest_category=corporate` internal;
add no `SourceType` label and no `FactUpdate` field. A later wire consumer needs an explicit
Optional-with-default addition and C/F notice. The conclusion is recorded in DECISIONS and
KnowledgePort §3.8; the interface stays traceable while that record is reviewed and merged.

This is one real consumer input, not evidence that all three consumers have completed their required
questions or approved the freeze. Their remaining review and the live endpoint evidence remain open.
