# EVAL_PROTOCOL — how evaluation is executed

> Scope: the **execution rules**. Design rationale and the config matrix are in spec 04 §4; this file only says
> how to annotate, how to judge, and how to record.
> Goal: the same QA set against the same corpus snapshot must produce the same numbers for anyone who reruns it.

## 1. QA set format (`eval/qa_set.jsonl`, one item per line)

```json
{
  "qa_id": "A-007",
  "type": "A",
  "question": "When does season 2 of <title> start airing?",
  "gold_answer": "2026-10-05",
  "gold_evidence": {
    "chunk_ids": ["c_8f21"],
    "urls": ["https://www.toei-anim.co.jp/news/..."],
    "publisher": "Toei Animation"
  },
  "key_entities": ["2026-10-05"],
  "source_type": "official",
  "window": ["2023-10-28", "2026-10-28"],
  "annotator": "D",
  "verified_by": "F"
}
```

- `key_entities`: the judgement basis for rule-based hit rate; each must appear in `gold_answer`.
- `window`: the validity window for this item; corpus outside it does not count.

The dates above are illustrative. The evaluated window is **sized by measurement** (3 years, widened until 8,000 items) and then **frozen**: it is recorded in `meta.json` and in `docs/DECISIONS.md`, and a later run with a different window is a different experiment. Do not copy these numbers into a run.
- Quotas: A 10–12 / B 8–10 / C 6–8 / D >=8 (>=5 abstention), total >=30, **type A <= one third of the total**.

## 2. Three-level judgement (`answer_correctness`)

| Level | Rule | Example |
|---|---|---|
| `fully_correct` | all key entities/figures correct, no statement contradicting `gold_evidence` | date, platform, episode count all right |
| `partially_correct` | core correct but one item missing, or correct with one unsupported extra claim | date right, platform missing |
| `incorrect` | key entity wrong, off-topic, or citations contradict the answer | says 10-05 but the source says 10-12 |

Discipline:
1. **Read the citations before the answer.** An answer whose citations do not reach the source drops one level
   even if the text is right — verifiability is the point of this topic.
2. Two people annotate 10 items independently; if agreement is below 85%, revise these rules instead of
   leaving the disagreement to the end.
3. Disputed items are adjudicated by D and logged in `eval/runs/<run_id>/disputes.md`.

## 3. Abstention (`abstention_accuracy`)

- Abstention counts only when the knowledge base truly lacks the answer *and* the response says so explicitly
  rather than guessing.
- Calibration set: 20 answerable + 10 unanswerable items; sweep the threshold and pick an operating point from
  the false-refusal / missed-refusal curve.
- Report both numbers: refusal rate on the unanswerable set and false-refusal rate on the answerable set.
  Reporting one alone invites the obvious objection.

## 4. Run record (`eval/runs/<run_id>/meta.json`)

```json
{
  "run_id": "20261115-A-20261028abcdef",
  "snapshot_id": "20261028abcdef",
  "window_start": "2023-10-28",
  "window_end": "2026-10-28",
  "chunk_config": "A",
  "retrieval": {"mode": "vector", "top_k": 5, "over_fetch": 5},
  "embed": {"model": "intfloat/multilingual-e5-base", "device": "cpu"},
  "llm": {"provider": "...", "model": "...", "version": "...", "temperature": 0.0},
  "git_sha": "...",
  "metrics": {"retrieval_hit_rate": 0.0, "answer_correctness": {}, "abstention_accuracy": {}}
}
```

**A run without `snapshot_id` or `window` is invalid** (invariant 2, spec 02).

Runs read the **evaluation corpus only**. The daily digest keeps its own rolling 14-day view for
deduplication and trend detection; it never feeds a run, and a run never reads it (`docs/DECISIONS.md`).

## 5. Reproducibility

- `temperature=0` **is not determinism**: MoE routing and batching still introduce jitter. The report must state
  that reruns differ slightly and give the observed spread across two reruns.
- Changing the device (cpu <-> mps), the model version or the chunk config produces a new `run_id`.
- Metric denominators are always `deduped_count` (invariant 3, spec 02).

## 6. Reporting rules (three lines that go into §6)

1. Besides totals, **group by `sourceType`**: on short official notices A and B barely differ; only long documents separate them.
2. Conclusions must be conditional: "A'/A win on detail questions, B wins on cross-source synthesis" — never "A is better".
3. With ~30 items the confidence interval is wide (order of ±10 points); attach it or state the sample-size limit.
