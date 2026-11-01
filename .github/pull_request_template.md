## What and why

<!-- Two or three lines. What changed, and what made it necessary. -->

## Where it sits

- **Work package**: WP1 / WP2 / WP3 / WP4 / WP5 / WP6
- **Acceptance gate**: G1a / G1b / G2 / G3 / not tied to a gate (`docs/ROADMAP.md` §2)
- **Spec**: `docs/...` §…

## Author checklist

- [ ] I ran `python scripts/check.py` (all four gates), not just `pytest`
- [ ] `git status` shows nothing that must not be committed: `data/`, `.env`, model files
- [ ] Files are in the directory the ownership map gives me (spec 02 §2)
- [ ] If I touched an invariant (spec 02 §3), I added a test for it
- [ ] Every limit I introduced or changed has a test asserting the **composed** production input against it
      (not a subset of it), and no semantic rule lives only in a `Field(description=…)`

## Contract changes (only if `src/acgn_rag/core/ports.py` is touched)

- [ ] The reason is explained above, and the reviewer is B (sole author of the contract)
- [ ] `python scripts/export_schema.py` was run and `contracts/knowledge_port.schema.json` is in this PR
- [ ] Affected consumers are listed: web (C) / daily (C) / Isla (F)

## Reviewer checklist (`docs/ONBOARDING.md` §3)

- [ ] CI is green on all four gates, on both platforms
- [ ] No secrets, no full article bodies, no model files in the diff
- [ ] This reviewer is the cross-reviewer for this WP, not the author
