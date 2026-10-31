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
| W1 (10/27–11/2) | | | | **Delivered:** #12 merged 10/28 — protocol made executable (`scripts/check_qa.py` + `core/eval/qa.py`: hit rule, calibration disjointness, two-tier acceptance); PM calls recorded in `DECISIONS.md` 10/29 — GPU-lane permit (owner's call) and #14 P1 language-quota adoption, both landed via F's #15; #28 open — QA-set draft 32 items (A10/B8/C6/D8, zh=11, overlap=10) + abstention calibration 20+10, `check_qa.py` exit 0. **Blocked:** resolving items against the corpus awaits A's ingest path (G1b, 11/09). **Next:** merge #28 after F's re-review, then G1b item resolution 11/09–11/12. | | |
| W2 | | | | | | |
| W3 | | | | | | |
| W4 | | | | | | |
| Sprint | | | | | | |

The **Sprint** row aggregates the whole milestone for the signature page; it is filled after G2 from
the weekly rows, not from memory.
