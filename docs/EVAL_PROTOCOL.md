# EVAL_PROTOCOL — how evaluation is executed

> Scope: the **execution rules**. Design rationale and the config matrix are in spec 04 §4; this file only says
> who annotates what, how a hit is judged, and how a run is recorded.
> Goal: the same QA set against the same corpus snapshot must produce the same numbers for anyone who reruns it —
> and when it cannot, the protocol says by how much.

## 1. Who does what, and when

| Role | Who | Responsibility |
|---|---|---|
| QA set owner | **D** | Wording, gold answers, quotas, adjudication, freeze. A set written by six people has six standards, so one person owns the wording |
| Rubric + runner owner | **E** | Applies the rubric, computes the metrics, keeps `run_eval.py` reproducible |
| Evidence verifier | **F** | Checks that each `gold_evidence` url actually contains the claim it is cited for |
| Candidates | anyone | Post a candidate question as a comment on the owning issue; D decides whether it enters the set |

**It is not "everyone annotates ten items."** D labels all 30; E independently labels a fixed 10-item subset so
that inter-annotator agreement can be measured. Thirty items is roughly 5–6 hours for the primary annotator, not
30 minutes for each of six people.

| Date | Milestone |
|---|---|
| 10/29 – 11/05 | D drafts all 30 items from first-party announcements. No corpus needed — the facts are on the live pages |
| 11/09 | A's index exists (G1b), so items can be resolved against a real snapshot |
| 11/09 – 11/12 | D resolves every item against the snapshot; F verifies each gold url contains the claim |
| 11/12 – 11/14 | E labels the 10-item overlap independently; agreement is computed |
| **11/14** | **QA set v1 frozen** (a git commit). After this date: additions only, never an edit to an existing gold answer |
| 11/15 – 11/21 | Runs, then the report tables |

The freeze date exists because a set edited after seeing the results is not a measurement.

## 2. QA set format (`eval/qa_set.jsonl`, one item per line)

```json
{
  "qa_id": "A-007",
  "type": "A",
  "answerable": true,
  "question": "When does season 2 of <title> start airing?",
  "gold_answer": "2026-10-05",
  "gold_evidence": {
    "urls": ["https://www.toei-anim.co.jp/news/..."],
    "publisher": "Toei Animation",
    "spans": ["will begin airing on 2026-10-05"]
  },
  "key_entities": ["2026-10-05"],
  "window": ["2023-10-28", "2026-10-28"],
  "annotator": "D",
  "verified_by": "F",
  "overlap": true
}
```

| Field | Rule |
|---|---|
| `qa_id` | `<type>-<nnn>`, unique, never reused |
| `type` | `A` time-sensitive · `B` detail · `C` cross-source synthesis · `D` out-of-scope / unanswerable |
| `answerable` | `false` for items the corpus cannot answer. **Explicit, never inferred from `type`** |
| `gold_answer` | the shortest string that is either right or wrong; no hedging |
| `gold_evidence.urls` | **the authoritative field.** 1–3 canonical source urls |
| `gold_evidence.spans` | optional quotes that let a human grader find the sentence quickly. Never used by the automatic hit rule |
| `key_entities` | the facts that must appear in a correct answer. They grade **answers**, not retrieval (§5) |
| `window` | validity window; evidence outside it does not count |
| `overlap` | `true` on the fixed 10 items E double-labels |

**Chunk ids are deliberately not stored here.** A chunk id is valid only inside one `(snapshot, chunk_config)`
pair — A(512/64) and A'(256/0) split the same sentence into different chunks with different ids, so a stored id
would silently point at nothing the moment either changes. Evidence is therefore keyed on **urls**, which survive
a new snapshot, a new chunking config and a rebuild. Per-run chunk ids are resolved at run time into
`eval/runs/<run_id>/resolved_evidence.json`, for debugging only.

## 3. Quotas

| Constraint | Value |
|---|---|
| Total | ≥30 (the course minimum is 20; we target 30) |
| Per type | A 10–12 · B 8–10 · C 6–8 · D ≥8 |
| Type A share | ≤ one third of the total |
| Unanswerable | ≥5 items with `answerable: false` (all inside type D) |
| Language | **≥8 of 30 in Chinese**, several of them targeting JP-language official sources (issue #14 P1; the corpus is trilingual — JP official, EN media, ZH media — and an all-English proxy would not measure that) |
| Evidence | every answerable item has ≥1 `urls` entry |
| Independent labels | exactly the 10 `overlap: true` items, labelled twice |

`python scripts/check_qa.py` enforces these rules; a set that fails it is not frozen.

### 3.1 Cross-lingual queries (pre-registered, issue #14 P1)

Chinese questions on JP-source evidence are part of the measured reality, not a stress case:
in the F pilot on a frozen bilingual corpus, zh questions hit gold doc@5 at **37.5 % direct vs
93.8 % with LLM query translation**, and translation itself cost one terminology miss (棋
rendered as "chess" instead of "Go"). Two consequences, registered before the freeze:

- Runs carry a **`query_lang` field** (`en` / `zh`), and metrics are reported per language beside
  the totals — a single blended number would hide exactly the interesting failure.
- The translate-then-retrieve variant (LLM translates the zh question, then retrieval runs as
  usual) is **pre-registered as the cross-lingual lane**. It is tried before the QA freeze so its
  status — adopted lane or reported-only ablation — is itself a frozen decision, not a post-hoc pick.

## 4. Retrieval quality — `retrieval_hit_rate`

Two numbers, always reported together, never one alone.

| Metric | Definition |
|---|---|
| `hit@k` (primary) | 1 if **any** returned chunk has `citation.url` in `gold_evidence.urls`, else 0. `retrieval_hit_rate@k` is the mean over answerable items |
| `mrr@k` (stretch gate) | mean reciprocal rank of the first gold-url hit over answerable items. **Pre-registered (issue #14 P3): the BM25+RRF stretch is judged on this, not on hit@k.** In the F pilot hybrid's gain lived in ranking (MRR@5 0.841 → 0.953/0.964) while hit@5 was saturated at 1.0 — judging a ranking improvement on a hit metric can only ever read as "no effect". If our hit@k has headroom, the stretch may lift both; that is a finding, not an assumption. |
| `evidence_recall@k` (secondary) | mean of `len(distinct gold urls retrieved) / len(gold urls)` — did we cover the *sources*, not just one of them |

- **Any gold source, not all of them.** A system that surfaces one authoritative source has done its job;
  requiring every url punishes correct behaviour when two outlets report the same fact differently.
- URL comparison normalises the fragment, a trailing slash and host case. Nothing else.
- An item whose gold url is **absent from the snapshot** is marked `gold_unavailable`, excluded from both
  denominators, and **counted in the report**. It is never scored 0, and never silently dropped.
- `key_entities` plays no part here. Grading retrieval on answer strings measures the generator's luck.

## 5. Answer correctness — human judgement, code only flags

| Level | Rule | Example |
|---|---|---|
| `fully_correct` | all key entities/figures correct, nothing contradicts `gold_evidence` | date, platform, episode count all right |
| `partially_correct` | core correct but one item missing, or correct with one unsupported extra claim | date right, platform missing |
| `incorrect` | key entity wrong, off-topic, or citations contradict the answer | says 10-05 but the source says 10-12 |

- **A human assigns the level. Code never does.** `key_entities` presence and citation containment are computed as
  *flags*, and a flag is not a label: "10-05 appears in the text" cannot tell whether the answer asserts it or
  denies it. No LLM self-grading, ever.
- **Read the citations before the answer.** An answer whose citations do not reach the source drops one level even
  when the text is right — verifiability is the point of this topic.
- **When a flag and a human disagree, the human wins and the disagreement is logged** in
  `eval/runs/<run_id>/disputes.md`. It is usually informative: a right answer with wrong citations is a lucky
  guess that will not survive the next question.
- Agreement: measured on the 10 overlap items. Below **85%** (i.e. ≥2 of 10 disagree), the rubric is ambiguous —
  fix the rubric and relabel, rather than carrying the disagreement into the report.
- Unresolved dispute between D and E: **drop the item, log it, and count the drop against the quota.** Forcing a
  label to protect a count is worse than a smaller honest set.

## 6. Abstention — and why the calibration set is extra

| Set | Size | Used for |
|---|---|---|
| `eval/abstain_calibration.jsonl` | 20 answerable + 10 unanswerable | sweeping the abstention threshold |
| `eval/qa_set.jsonl`, type D | ≥8, of which ≥5 unanswerable | **reporting** `abstention_accuracy` |

**The calibration set is additional, not part of the 30.** Choosing the threshold on the same items you then
report on is leakage: the reported number would be optimistic by construction, and nothing in the report would
reveal it. The two sets are disjoint, which `check_qa.py` enforces.

Report both: refusal rate on the QA set's unanswerable items, and false-refusal rate on its answerable items. One
number alone invites the obvious objection. The calibration set's own rates may be quoted afterwards, but
**labelled in-sample**, so nobody mistakes them for the result.

## 7. Run record and traceability

`eval/runs/<run_id>/` is tracked in git and contains:

| File | Content |
|---|---|
| `meta.json` | snapshot, window, config, retrieval settings, embed model/device, llm provider/model/version/temperature/seed, git sha, metrics |
| `answers.jsonl` | the exact generated outputs that were graded — frozen, so grading can be replayed without the model |
| `labels.jsonl` | the human labels, per item, with annotator |
| `metrics.json` | the numbers the report quotes |
| `resolved_evidence.json` | gold urls resolved to chunk ids **for this snapshot and config** |
| `disputes.md` | adjudications and dropped items |

```json
{
  "run_id": "20261115-A-20261028abcdef",
  "snapshot_id": "20261028abcdef",
  "window_start": "2023-10-28",
  "window_end": "2026-10-28",
  "chunk_config": "A",
  "retrieval": {"mode": "vector", "top_k": 5, "over_fetch": 5},
  "embed": {"model": "intfloat/multilingual-e5-base", "backend": "torch", "device": "cpu", "precision": "fp32", "batch": 32},
  "llm": {"provider": "...", "model": "...", "version": "...", "temperature": 0.0, "seed": 0},
  "git_sha": "...",
  "metrics": {"retrieval_hit_rate": 0.0, "evidence_recall": 0.0, "answer_correctness": {}, "abstention_accuracy": {}}
}
```

- **A run without `snapshot_id` or `window` is invalid** (invariant 2, spec 02).
- **Every number in the report carries a `run_id`.** A number produced in an interactive session, or in a
  notebook cell that was not archived as a run, is not reportable. If it is worth printing, it is worth
  archiving.
- Runs read the **evaluation corpus only**. The daily digest keeps its own rolling 14-day view; it never feeds a
  run, and a run never reads it (`docs/DECISIONS.md`).

## 8. Reproducibility — the two-tier acceptance

"Reruns give the same numbers" and "LLMs jitter" are both true, so acceptance is split by stage.

| Tier | What | Acceptance |
|---|---|---|
| **1 deterministic** | `retrieval_hit_rate`, `evidence_recall`, the retrieved chunk id lists and their order, abstention decisions at a fixed threshold | **identical across two reruns on the same snapshot and config.** Any difference is a bug: index rebuilt differently, cache serving stale vectors, or an unordered set in the code |
| **2 generative** | generated answer text, `answer_correctness` levels, abstention when the threshold is re-tuned | two reruns; level counts may differ by **≤1 item out of 30**, abstention decisions by ≤1. The observed spread is reported; above that, investigate |

- `temperature=0` **is not determinism**: MoE routing and batching still introduce jitter. Record provider, model,
  version, date and seed, and state in the report that reruns differ slightly, with the measured spread.
- **Two reruns are archived, not one.** A spread claim with a single run behind it is unsupported.
- Grading reproducibility is a separate question: **re-grade the same `answers.jsonl`** and the labels must be
  identical. A difference means the rubric is ambiguous, not that the model moved.

## 9. Reporting rules

1. Besides totals, **group by `sourceType`**: on short official notices A and B barely differ; only long documents
   separate them.
2. Conclusions must be conditional: "A'/A win on detail questions, B wins on cross-source synthesis" — never
   "A is better".
3. With ~30 items the confidence interval is wide (order of ±10 points); attach it or state the sample-size limit.
4. State the `run_id`, the two-rerun spread, and the count of `gold_unavailable` / dropped items next to the table
   they affect. A table whose exclusions are invisible is a misleading table.
