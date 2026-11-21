# Quote-discipline experiment — result and decision

Run: 2026-11-20 16:00 – 17:26 (Asia/Shanghai) · branch `answer-line-fixes` @ `fb4dddf` + uncommitted work-tree changes
Instrument: `eval/qa_set_delivery.jsonl` (34 items: 27 answerable / 7 unanswerable)
Model: `group/deepseek-flash` @ cs-project-ai, temperature 0.0, max_tokens 2048
Index: bge-m3 Config A, RTX 5060 (identity `bge-m3-torch-cuda-fp32-6174cb54` matches the build sidecar)
Fixed knobs in both arms: `--translate-queries zh --official-bonus 1.08`, top_k 5, no abstention threshold

## The one manipulated variable

`ACGN_ANSWER_QUOTE_DISCIPLINE` — unset (shipped `SYSTEM_PROMPT`) versus `1` (same prompt + 807 appended characters).
Verified append-only: `OFF == SYSTEM_PROMPT` → True, `ON.startswith(SYSTEM_PROMPT)` → True,
`SYSTEM_PROMPT.endswith(MEDIA_SOURCE_RULE)` → True.

## Headline

| arm | runs | answered | false-refused (mean) | mean rate | unanswerable correct | false answers |
|---|---:|---:|---:|---:|---:|---:|
| **off** | 4 | 19,16,26,25 | 5.50 / 27 | **20.4%** | 28/28 | 0 |
| **on** | 3 | 26,27,26 | 0.67 / 27 | **2.5%** | 21/21 | 0 |

Every on-run (3.7%, 0.0%, 3.7%) is at or below the **best** off-run (3.7%). No on-run is worse
than any off-run, and no run in either arm produced a false answer or a wrong refusal.

## The paired result, which is what actually decides it

The off arm was not a stable baseline — it ranged 3.7%–40.7%. Comparing means alone would be
misleading, so each question's refusal probability was compared across arms:

| | count |
|---|---:|
| improved (refused less often under `on`) | **12** |
| unchanged | 14 |
| worsened | 1 (`C-008`, 0% → 33%) |

A-001 75%→0%, C-002 75%→0%, B-008/C-001/C-004/C-005/C-006/D-004 50%→0%. The rule fixes
exactly the questions that were flapping, not a random subset.

## The mechanism, confirmed

`all_claims_rejected` is the only cause of answerable refusal in both arms — `model_returned_no_claims`
never refuses an answerable item. The rule reduces the failure it was written for:

| arm | claims generated | claims dropped by C4 | drop rate |
|---|---:|---:|---:|
| off | 145 | 48 | **33.1%** |
| on | 109 | 15 | **13.8%** |

## Decision (against the criteria fixed in NEXT_STEPS §2 before the run)

The criteria were: adopt if false refusal ≤ 20% **and** unanswerable 7/7 **and** false answers 0.

- false refusal 2.5% ≤ 20% → **met**
- unanswerable 7/7 in all three runs (21/21) → **met**
- false answers 0 in all three runs → **met**

**→ 采纳 (adopt).** The rule should be enabled, not left behind a flag.

## Two things this experiment changed about our own understanding

1. **The reported "base line" was a lucky draw, not a baseline.** NEXT_STEPS recorded
   `20261120-bge-A-live-v2` at 11/27 (40.7%). Four identical off-runs in one afternoon gave
   8, 11, 1, 2. Quoting any single run — including that 40.7% — is unsafe. The off arm's
   honest description is "5.5 false-refusals ± a very wide spread".
2. **Refusal is largely a generation-time coin flip, not a retrieval or validation defect.**
   Across four identical runs, 12 of 27 answerable questions (44%) flipped between answered and
   refused, and **zero** questions were refused every time. 15 were answered every time.
   Temperature is already 0.0 and `finish_reason` was `stop` in every refused row, so this is
   provider-side nondeterminism we do not control. Any future claim about the answer line needs
   repeats, and a single-run A/B between two arms would have been noise.

## What this does NOT settle

- **Not reportable.** All runs are `NOT REPORTABLE: window_unverified` (same as every earlier
  delivery-set run). These are product-instrument numbers, not gate evidence.
- **One provider, one model, one index.** The 44% flip rate is specific to this endpoint; a
  different provider may be far more stable and the effect size would then need re-measuring.
- **C-008 got worse** (0% → 33%, 1 of 3 runs). Its claim text probably needs a quote the rule's
  wording does not encourage. Worth one look, not worth blocking on.
- **The A-vs-A′ and e5-vs-bge comparisons are untouched** — retrieval was already at ceiling
  (hit@5 1.0 in the clean runs), confirming the bottleneck is the answer line, as NEXT_STEPS §3.3 said.

## Reproducing

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\experiments\run_arm.ps1 -Arm off -RunId <id>     # shipped prompt
.\experiments\run_arm.ps1 -Arm on  -RunId <id>     # + quote discipline
.\.evalvenv\Scripts\python.exe experiments\summarize_arms.py 20261120-qd
```

Each run takes ~9–15 min. Exit code `1` is expected: it is the archive's own
`NOT REPORTABLE` verdict, not a crash. Archives land in `evalout/<run-id>/`.

## Before pushing (NEXT_STEPS §4–5 still open)

1. Decide whether to flip the default: adopting means `quote_discipline_enabled()` should
   return True by default, with the flag kept only as an escape hatch.
2. Update the PR #116 body — it currently reports 19.2% and does not mention this rule at all.
3. Push the `qa_set_delivery.jsonl` label fix (D-004 → answerable), which changes the
   answerable denominator from 26 to 27 and therefore every rate previously quoted.
4. Clean the untracked scratch files listed in NEXT_STEPS §5, or add them to `.gitignore`,
   before any `git add -A`.
