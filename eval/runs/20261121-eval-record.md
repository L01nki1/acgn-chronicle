# E evaluation record — delivery A/B and four-index replay (2026-11-21)

Single E-owned record for the archives added alongside #115. Raw runs are in this directory; this
document is the interpretation. It is aligned with the PR #116 follow-up
([`7057543594`](https://github.com/L01nki1/acgn-chronicle/pull/116#issuecomment-7057543594)) and
the #111 correction
([`7060286198`](https://github.com/L01nki1/acgn-chronicle/issues/111#issuecomment-7060286198)).

## Three rounds, not pooled

| round | instrument | items | reportable | result |
|---|---|---|---:|---|
| E4 bge | `abstain_calibration.jsonl`, fp `66384069…` | 30 (20 answerable, 10 unanswerable); 7 `gold_unavailable` → **n=13** | false, `calibration_instrument` | bge A 8/13, A′ 7/13; e5 0/13 |
| Delivery A/B | `qa_set_delivery.jsonl` @ `fb4dddf`, fp `c77c842d…` | 34; **26 answerable / 8 unanswerable** (pre-relabel) | true, window verified | bge A′ 25/26, 25 answered, 1 refusal, 7/8 unanswerable, **1 false answer (D-004)** |
| Index replay | `qa_set_delivery.jsonl` @ `e152902`, fp `97b7ca7e…` | 34; **27 answerable / 7 unanswerable** (post-relabel) | true, window verified | bge 26/27, e5 21/27; 0 false answers |

The 25/26 and 26/27 figures are the same delivery archives under two labels: the D-004 relabel
turns one false answer into an answerable hit. Quote 26/27 for the current instrument.

## Delivery A/B — 2026-11-20

Runs `20261120-delivery-bge-Aprime-off` and `-on`; bge-m3 A′, identity
`bge-m3-torch-cuda-fp32-6174cb54`; OpenCode Go `deepseek-v4.1-flash`; no threshold; code
`fb4dddf`. The arms differ only in `ACGN_ANSWER_MAX_TOKENS` (8192 / 32768) and the prompt knobs;
every count is identical.

| | hit@5 | MRR | answered | false refused | unanswerable | false answers |
|---|---:|---:|---:|---:|---:|---:|
| off | 25/26 | 0.8667 | 25 | 1 (A-009) | 7/8 | 1 (D-004) |
| on | 25/26 | 0.8667 | 25 | 1 (A-009) | 7/8 | 1 (D-004) |

Citation containment 1.0 (n=26), 0 violations. A-009 is also the only retrieval miss; D-004 is
the label defect the relabel resolves. These are exploratory product-instrument numbers, not the
④ gate.

## Index replay — 2026-11-21

Twelve archives: bge-m3 A/A′ and e5 A/A′, each with single-pool retrieval, official-bonus
retrieval, and a live run (`translate zh`, `bonus 1.08`, both prompt rules, no threshold). Code
`e152902+dirty:1` (the local session-header injection); QA fingerprint `97b7ca7e…`; 27/7.

| index | hit@5 | MRR | gold at rank 1 | answered | false refusals | unanswerable | false answers |
|---|---:|---:|---:|---:|---:|---:|---:|
| bge-m3 A | 26/27 | 0.7802 | 18/27 | 27 | 0 | 7/7 | 0 |
| bge-m3 A′ | 26/27 | 0.8469 | 21/27 | 26 | 1 | 7/7 | 0 |
| e5 A | 21/27 | 0.5877 | 12/27 | 24 | 3 | 7/7 | 0 |
| e5 A′ | 21/27 | 0.6543 | 15/27 | 22 | 5 | 7/7 | 0 |

- **A′ reproduces the reference counts**: 26/27, 26 answered, one refusal; the refusal is
  `delivery-A-009` and the miss is also `delivery-A-009`.
- **Translation lane did not fire**: all 20 zh rows record `query_translation.used=false` while
  `meta.query_transform="translate-zh"`. These are direct-retrieval numbers; do not compare the zh
  column to a translated-lane run.
- **Official bonus is rank-neutral, not evidence-neutral**: zero gold-rank changes across the 108
  answerable (index × item) comparisons; official-tier chunks 1 / 3 / 36 / 28 for bge A, A′, e5 A,
  e5 A′ (single-pool: 0). `source_mix` counts chunks, not distinct pages.
- **Duplicate pages**: 28/34 rows on both bge configs, 18/34 (e5 A), 23/34 (e5 A′); different
  chunks of the same article, zero duplicate chunk ids.
- **27 items resolve to 16 unique gold pages** (nine shared), so counts are item-level.

## Verification

`20261121-archive-verification.txt` — 96/96 expected files, recompiled hit/MRR from
`answers.jsonl` equals `metrics.json` for every archive, abstention counts match, fingerprints and
`window_verified` match, no secret-like strings, no zips tracked. Result: **PASS**.

## Artifacts

- E4 ledger: `20261120-E4-bge-ledger.json` (tarball sha `7c0ee14b…`).
- Replay bundle outside git: `eval/index-eval-runs.zip`, sha256
  `6ad832719522fe8c50f521eecb914e3c26f23189925ca19517c49b9a3171c6c2`, 197,704 bytes.
- PR #115 carries the e5 A′ live runs; this branch adds the E4 bge, delivery A/B and replay
  archives so D/F can review the measurement set in one place.

## Do not

- Pool these rounds with the frozen ④ instrument or with each other.
- Quote 25/26 as the current bge number; it is the pre-relabel instrument.
- Treat the E4 13-item subset as a rate for all 20 calibration items.
- Do not pair the 11/19 calibration (`20261119-cal-Aprime-e5`) with post-#116 code; it predates
  the three identity fields.
