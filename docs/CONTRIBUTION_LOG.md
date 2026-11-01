# CONTRIBUTION LOG — per-member weekly record

> Feeds the signed contribution page the brief requires: a missing signature page is a desk reject
> (`docs/ROADMAP.md` §6, risk R5), so this log is not optional bookkeeping. The shape is fixed by
> issue #6 ("Contribution log"): six members × weeks — **what was delivered, what blocked it, what
> is next** — and `docs/DECISIONS.md` keeps a pointer to it.
>
> Rules of use:
>
> - **Each cell is written by that member.** D only keeps the table current (`docs/ROADMAP.md` §4
>   secondary responsibilities; §7 "Weekly = D updates the contribution log") and never ghost-writes
>   an entry for someone else.
> - **Cell format** — `Delivered: … / Blocked: … / Next: …`, citing real artifacts only (PR numbers,
>   `DECISIONS.md` entries, run ids). A sub-field is dropped only when it is genuinely empty.
> - An empty cell means the member has not filed that entry yet — per issue #6 Dependencies the
>   first entry per member was due at G1a — not that the member did nothing.

| Week | A (shallight) | B (Pioneer141) | C (Warpshlczy) | D (L01nki1) | E (MichaelLsh) | F (inscripoem) |
|---|---|---|---|---|---|---|
| W1 (10/27–11/2) | | **Delivered:** #26 (`aac6598`), schema 0.2.3 **candidate**, including citation/id model-validation agreement, QA-count provenance and publisher display semantics; C approved this head after the earlier C1–C6 review at `c8c3e88`; four local gates: **165 passed / 2 opt-in real-e5 skips**. #32 consumer approval at `430aafc` / **61 tests**; #35/#36/#37 cross-reviews with corrections requested; #14 P4 conditional acknowledgement. **Blocked:** A corpus/minting/registry evidence, E's final #32 review and C/D consumer/process corrections. F was notified of 0.2.3; no acknowledgement is claimed. **Next:** recheck owner repairs and implement the G1b actual-retrieved-set/final-payload/refusal guard in W2. | | **Delivered:** #12 merged 10/28 — protocol made executable (`scripts/check_qa.py` + `core/eval/qa.py`: hit rule, calibration disjointness, two-tier acceptance); PM calls recorded in `DECISIONS.md` 10/29 — GPU-lane permit (owner's call) and #14 P1 language-quota adoption, both landed via F's #15; #28 open — QA-set draft 32 items (A10/B8/C6/D8, zh=11, overlap=10) + abstention calibration 20+10; 10 out-of-scope gold urls replaced at `c8cf3ee`, then 9 out-of-window rows replaced and the `check_qa.py` window check rebuilt as UTC-instant with ok/manual/invalid buckets at `9cc02dc` — `check_qa.py --check-urls` exit 0 (0 invalid; 23 urls listed for manual verification, pages stating no machine-readable time); F's 2026-10-31 span findings fixed at `9a8fe76` (A-004/C-001 opening quote mark, B-908/B-909 outer brackets). **Blocked:** the overlap-composition call is D's to adjudicate (all ten overlap items are answerable); item resolution awaits A's ingest path (G1b, 11/09); F's frozen-set evidence pass runs 11/09–11/12. **Next:** obtain F's evidence re-review, decide the overlap composition, then merge #28; G1b item resolution 11/09–11/12. | | **Delivered:** encoder identity/model-lane documentation (#15); WP2 chunking, embedding, storage, fact extraction and builds (#19–#22, #24, #27); published snapshot isolation/no-create binding #32 (`430aafc`, B approved); reviewed #26/#28/#29/#33/#34. Prepared local follow-ups for #8/#32: complete-input limits, batch-independent cache identity, per-item fact recovery and bounded/empty-store queries. Local checks: store 178 passed/2 skipped; input/extraction 189/2; with #26 `aac6598`, 218/2; two additional real-e5 checks passed offline. **Blocked:** A's NormalizedItem/cluster-id/registry definition and frozen snapshot; E's final #32 review; follow-up candidates await consumer review/adoption. **Next:** complete the reviewed-code handoff, follow D's QA replacements and finish actual-corpus checks/config B after A's handoff. |
| W2 | | | | | | |
| W3 | | | | | | |
| W4 | | | | | | |
| Sprint | | | | | | |

The **Sprint** row aggregates the whole milestone for the signature page; it is filled after G2 from
the weekly rows, not from memory.
