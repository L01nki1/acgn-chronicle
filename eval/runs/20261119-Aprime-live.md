# ④ live runs on A′ — 2026-11-19

Two Tier-2 live runs on the adopted e5 Config A′ lane, for D's ④ adjudication.
Both are `reportable=true`, `qa.role=report`, with the A′ calibration artifact.

| Item | Value |
|---|---|
| Runs | `20261119-Aprime-live-1-e5`, `20261119-Aprime-live-2-e5` |
| Calibration | `20261119-cal-Aprime-e5` — threshold **0.8579685091972351**, `balanced_accuracy`, gate set 30 = 20/10 |
| Calibration identity | `chunk_config=A'`, e5, `top_k=5`, over-fetch 5 |
| QA set | `eval/qa_set.jsonl`, 32 items (26 answerable / 6 unanswerable), fingerprint `dd0bbe…dacc` |
| Encoder | `multilingual-e5-base-torch-cuda-fp32-d2fa8669` |
| GPU | Tesla T4 (both runs) |
| Code | `main@7fef7e6` |
| `meta.llm` | `group/deepseek-flash` · version `deepseek-flash` · temperature 0.0 · seed null |
| `meta.llm_probe` | `status=200`, `model_present=true`, `model_count=1` |
| `ACGN_ANSWER_MAX_TOKENS` | 8192 (2048 produced `finish_reason=length` with empty content) |

## Retrieval (headline rates match the 11/17 A′ archive)

| Metric | Value |
|---|---|
| hit@5 | 0.347826 (n=23) |
| MRR@5 | 0.264493 |
| evidence recall | 0.326087 |

The headline rates are identical to the 11/17 A′ archive. The row-level lists are not:
the same URL set appears on 4/30 items, and no row has identical `retrieved_urls` /
`retrieved_chunk_ids` (D's #114 review).

## Generation

| | run 1 | run 2 |
|---|---:|---:|
| rows `answered` | 32/32 | 32/32 |
| abstained | 19 | 19 |
| `retrieval_below_threshold` | 16 | 16 |
| `model_returned_no_claims` | 3 | 3 |
| provider completions | 16 | 16 |
| rows with citations | 13 | 13 |
| total citations | 14 | 13 |
| citation containment | 1.0 (n=13) | 1.0 (n=13) |
| citation violations | 0 | 0 |

- **Unanswerable:** 5 of 6 refused.
- **Answerable:** 14 of 26 refused; `false_refusal_rate` 0.478 (n=23 after 3 `gold_unavailable`).
- **Abstention spread:** none. A-003 and A-006 both answered in both runs.
- **B-005 host:** run 1 cites Real Sound (`realsound.jp/2026/09/post-2522771.html`); run 2 cites the gold Nijimen page (`nijimen.kusuguru.co.jp/topics/642080`). Both are grounded; the report must record the per-row host.
- **A-008:** 2 citations in run 1, 1 in run 2.

## Before/after — the 11/17 pilot vs these runs (same 32-item QA set)

| | 11/17 pilot (A, 0.825) | today (A′, 0.858) |
|---|---:|---:|
| cited rows | 1 | 13 |
| abstentions | 31 | 19 |
| provider completions | 27 | 16 |
| false-refusal rate (n=23) | 0.9565 | 0.478 |

The gate is **stricter** now, so fewer rows reach the model (27 → 16 completions). The gain is
in the claim path (B's #106 media/official citation rules) plus A→A′ retrieval (6/23 → 8/23 gold
hits), not in a looser threshold.

## The one item D has to adjudicate

**D-004** (`answerable: false`) was answered in both runs:

> 開催時間：10:00〜18:00（最終入場17:00） [1]

The cited chunk (Nijimen, `https://nijimen.kusuguru.co.jp/topics/645376`, `…::A'::5`)
contains that sentence verbatim:
`開催時間：10:00〜18:00（最終入場17:00）`.

So this is not a hallucination and not a citation violation — the retrieval found the
exhibition hours. Either the item's label is too strict (the corpus answers the question),
or the question is read as asking for the 進撃の巨人 zone specifically rather than the
exhibition. D's call; ④ is judged on groundedness/refusal, and this answer is grounded.

## Known retrieval degradation

17 of 30 rows return duplicate URLs in `retrieved_urls` (A: 12/30). One document's chunks can
occupy several of the five slots, so `top_k=5` is sometimes effectively ~3 distinct pages. This
is a query-side assembly question (collapse sibling chunks vs return whole documents), recorded
by D in the #114 review; it is not a live-run defect and does not change the citations or the
threshold.

## Notes

- `metrics.abstention.threshold` is `null`; the applied threshold is recorded in
  `meta.abstention.threshold` and in every row's `diagnostics.min_retrieval_score`.
- The 0.858 A′ threshold is stricter than A's 0.825: calibration false refusals were
  8/20 (0.40) versus 1/20, which is why the live false-refusal rate is high.
