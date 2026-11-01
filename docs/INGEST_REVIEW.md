# A's response to E's 15-item G1a review

2026-11-01; #9 review 6517500655, tracked by #30.
Producer commitment: [INGEST_CONTRACT.md](INGEST_CONTRACT.md).
Status: design response for review; real corpus and runtime acceptance pending.

| # | Verdict | A's response / evidence boundary |
|---|---|---|
| 1 | Reject old spelling; accept canonical identity | Full wire ID is YYYYMMDD-12hex, date from frozen window_end UTC day, per #26/#31. Contract section 3 defines stable sorting, exact bytes and hash inputs. Rebuild-day dating is rejected. |
| 2 | Accept | Full row metadata plus normalized body SHA-256, registries and dedup policy are hashed; body edits change identity. Synthetic example demonstrates this, production verification pending. |
| 3 | Accept | Aware UTC Z timestamps, preserve fractions. Missing timezone requires documented source timezone or quarantine, never host-local inference. |
| 4 | Accept counts; reject denominator wording | Counts are derived from manifest members/clusters/source keys; frozen bounds remain configured bounds, with observed coverage separately reported. D's #30 ruling makes QA denominators eligible question sets, not deduped_count. |
| 5 | Accept, runtime evidence pending | Synthetic two-build identity proof is committed. Live acquisition idempotency and two real snapshot rebuilds are not yet demonstrated. Same-input rebuild and no-new-data incremental ingest are separate checks. |
| 6 | Reject short physical names | Follow F's #31/PR #32 safe full-ID ownership mapping; read actual collection from validated sidecars. Labels A/A-prime/A0/B accepted. No claim that PR32 is merged. |
| 7 | Accept target; W2 implementation pending | Current + previous, plus every active pin. Order by validated publication time, never hash order. Retirement requires catalog evidence. |
| 8 | Accept target; W2 implementation pending | Stage a non-served generation; establish authoritative completion before atomic CURRENT switch. Old sidecars after interrupted --fresh do not prove readiness. A/C/F coordinate lifecycle. |
| 9 | Accept, adapters pending | Explicit selector/403/decode failures and counters; no silent empty historical result. Legitimate zero increments/304 are distinguishable successes. |
| 10 | Accept | Raw bodies remain internal under ignored data/. Only sanctioned excerpts/facts may leave. This change includes only synthetic format examples, no third-party bodies. |
| 11 | Accept | Nine requested fields plus optional ingest_category; internal cluster_id/event_type/source_key are explicit. No publisher_id or wire change adopted. See #23 response and contract section 1. |
| 12 | Accept target; registry/linker pending | A owns versioned local integer entity IDs with external aliases; no mixing bare external namespaces. Link before clustering; unresolved [] does not authorize broad merging. |
| 13 | Accept, probe/window pending | Source x depth table precedes production window freeze. A/D choose the window from measured history; no 8000-item or three-year corpus claim is made here. |
| 14 | Accept target | Existing-only reads; 404 unknown, 410 known retired, 503 unserved/incomplete; no read 409/fallback. Host/catalog implementation remains W2. |
| 15 | Accept | This table provides individual verdicts and replacements, with implementation/evidence boundaries. E must review; A does not tick E's acceptance checklist. |

## A stage report (design slice only)

Stage: G1a in progress, not an acceptance claim.

Delivered: #23 producer fields/cluster identity and publisher-display decision;
canonical snapshot manifest/minting design; E's 15-item response; synthetic identity
example and independent two-build byte/hash evidence, all in this PR.

Evidence: linked committed docs/examples and PR diff. No real-source count,
source-depth CSV, live backfill or production idempotency result is supplied.

Blocked: live ingestion/registry/normalization/dedup implementation remains A work;
F's storage fix adoption and A/C/F publication/pin protocol are distinct dependencies.
The actual evaluation window needs A/D coordination with #28. Other owners' review
items on #30 are not resolved by this response.

Next: implement the minimum HTTP/adapter registry; run source-depth probes including
ANN pagination/date filtering; collect and freeze actual data; provide the reader
to F and real two-rebuild evidence to E/D.
