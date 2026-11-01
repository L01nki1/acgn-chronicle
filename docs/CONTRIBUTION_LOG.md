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
| W1 (10/27–11/2) | | **Delivered:** #26 merged as `22d404e` (schema 0.2.3, C approval and F acknowledgement); #40 (`24f6600`), schema **0.2.4 candidate**, aligns source_count with the merged #38 feed-key handoff. Four local gates: **165 passed / 2 opt-in real-e5 skips**. B approved #32 at `c9dc0e9` / **66 tests**, #35 at `119a1c4` / **41 web + 16 Node cases**, and #37 at `75c3bc1`; checked A's synthetic manifest identity separately; #17 interface decision is closed with history retained. **Blocked:** A's real pipeline/corpus/probe/window/rebuild evidence, #40 C/E/F semantic review, and adoption of the reviewed implementation candidates. **Next:** recheck changes after integration and implement the G1b actual-retrieved-set/final-payload/refusal guard in W2. | | **Delivered:** #12 merged 10/28 — protocol made executable (`scripts/check_qa.py` + `core/eval/qa.py`: hit rule, calibration disjointness, two-tier acceptance); PM calls recorded in `DECISIONS.md` 10/29 — GPU-lane permit (owner's call) and #14 P1 language-quota adoption, both landed via F's #15; #28 open — QA-set draft 32 items (A10/B8/C6/D8, zh=11, overlap=10) + abstention calibration 20+10; 10 out-of-scope gold urls replaced at `c8cf3ee`, then 9 out-of-window rows replaced and the `check_qa.py` window check rebuilt as UTC-instant with ok/manual/invalid buckets at `9cc02dc` — `check_qa.py --check-urls` exit 0 (0 invalid; 23 urls listed for manual verification, pages stating no machine-readable time); F's 2026-10-31 span findings fixed at `9a8fe76` (A-004/C-001 opening quote mark, B-908/B-909 outer brackets); overlap-composition adjudicated 11/01 — composition stays as-is (ten answerable overlap items; the protocol requires exactly ten, not an unanswerable one), revisited only if F re-raises with a protocol ground. **Blocked:** item resolution awaits A's ingest path (G1b, 11/09); F's frozen-set evidence pass runs 11/09–11/12. **Next:** obtain F's evidence re-review, then merge #28; G1b item resolution 11/09–11/12. | | **Delivered:** encoder identity/model-lane documentation (#15); WP2 chunking, embedding, storage, fact extraction and builds (#19–#22, #24, #27); published snapshot isolation/no-create binding #32 (`430aafc`, B approved); reviewed #26/#28/#29/#33/#34. Prepared local follow-ups for #8/#32: complete-input limits, batch-independent cache identity, per-item fact recovery and bounded/empty-store queries. Local checks: store 178 passed/2 skipped; input/extraction 189/2; with #26 `aac6598`, 218/2; two additional real-e5 checks passed offline. **Blocked:** A's NormalizedItem/cluster-id/registry definition and frozen snapshot; E's final #32 review; follow-up candidates await consumer review/adoption. **Next:** complete the reviewed-code handoff, follow D's QA replacements and finish actual-corpus checks/config B after A's handoff. |
| W2 | | | | | | |
| W3 | | | | | | |
| W4 | | | | | | |
| Sprint | | | | | | |

The **Sprint** row aggregates the whole milestone for the signature page; it is filled after G2 from
the weekly rows, not from memory.
