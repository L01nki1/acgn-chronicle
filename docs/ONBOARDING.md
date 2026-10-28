# ONBOARDING — read this first

> Scope: engineering collaboration details. Spec 06 §1 states the rules; this file turns them into a checklist.
> When a rule changes, change it here first, then ping everyone in the PR.

## 1. Environment (5 members on macOS, 1 on Windows)

| Item | macOS (most of us) | Windows (1 member) |
|---|---|---|
| Python | 3.11 (`brew install python@3.11` or `uv python install 3.11`) | 3.11 (official installer, tick "Add to PATH") |
| Shell | zsh (system bash is 3.2 — do not use bash 4+ syntax) | Git Bash (not cmd / PowerShell) |
| Virtualenv | `source .venv/bin/activate` | `.venv\Scripts\activate` |
| torch wheel | native arm64, installs cleanly | fine |

**Do not** commit `.sh` / `.ps1` / `.bat` files as entry points. Every repeatable action goes through
`python scripts/xxx.py` or `python -m acgn_rag...`, so both platforms share one command set.

## 2. Branches and commits (06 §1)

- `main` only moves through PRs.
- One long-lived branch per WP: `wp1-ingest` / `wp2-kb` / `wp3-retrieve` / `wp4-ground` / `wp5-eval` / `wp6-web`.
  Personal work uses short-lived branches on top: `wp1-ingest/a-toei-adapter`.
- Commit message: `[WP1] add Toei html list adapter`; contract changes get the `[contract]` prefix.
- Contract changes (`ports.py`) must explain the reason in the PR description and re-export the schema (§4).

## 3. PR checklist (the reviewer ticks each line)

1. CI is green on all four gates (lint / format / contract check / tests) — the author ran
   `python scripts/check.py` before pushing, not just `pytest` (§4.1)
2. Nothing leaked into the commit: `data/`, `.env`, model files (`git status` first)
3. New directories match the ownership map in spec 02 §2; cross-WP edits @ the owner in the description
4. Any change touching an invariant (spec 02 §3) comes with a test
5. The reviewer is that WP's cross-reviewer, not yourself

## 4. "CI does not use the network" — exact meaning

**Meaning**: CI may go online only while installing dependencies; while tests run, **no external request is allowed**.

| Allowed | Forbidden |
|---|---|
| `pip install -e ".[dev]"` from PyPI | `httpx.get("https://...")` against a live site inside a test |
| Reading committed sample pages from fixtures | Downloading the e5 model or calling an LLM API in a test |
| — | An "integration test" that only passes online |

**Why**: site availability, rate limits and Japanese sites returning 403 would make CI red for reasons
unrelated to the code — and a CI that cries wolf gets ignored.

**How to do it**:
- Mock HTTP with `respx`; return samples from `tests/ingest/fixtures/`.
- Use the `fake_embedder` fixture (`tests/conftest.py`) for deterministic fake vectors.
- Genuinely online checks get `@pytest.mark.network` and run **locally only**: `pytest -m network -q`.
  The default `addopts` already contains `-m 'not network'`.
- `tests/conftest.py` intercepts sockets, so breaking the rule raises `NetworkBlocked` — by design.

**Crawler exception**: `scripts/backfill.py` / `discover_daily.py` need the network by nature, but they never
run in CI and carry no unit tests (selector logic is tested offline against fixtures, see 01 §5.3).

### 4.1 Run the same four steps CI runs, before you push

One command runs all four in CI's order and stops at the first red step:

```bash
python scripts/check.py          # gate only
python scripts/check.py --fix    # apply ruff fixes first, then gate
```

The four gates, in order (CI stops at the first failure, so a later gate never explains an earlier red):

| # | Gate | Command | Catches | Fix |
|---|---|---|---|---|
| 1 | Lint | `ruff check .` | unused imports, shadowed names, import order | by hand |
| 2 | Format | `ruff format --check .` | edited-but-never-formatted file | `ruff format .` |
| 3 | Contract | `python scripts/export_schema.py --check` | `ports.py` edited without re-export | export, commit the JSON in the same commit |
| 4 | Tests | `pytest -q` | real failures | read the test |

`pytest` alone is not enough: a commit that only touches docs or comments can still fail lint or the
contract gate. If you edited `src/acgn_rag/core/ports.py`, regenerate the schema with
`python scripts/export_schema.py` and commit the JSON in the same commit.

**Markdown is not formatted by ruff** (`*.md` is in `extend-exclude`). Ruff would format fenced
`python` blocks but does not lint them, which turns a harmless snippet in `docs/` into a red build for a
reason nobody can see. Docs may therefore contain illustrative pseudo-code and column-aligned comments.
Inside `.py` files the formatter is strict — do not fight it, run `ruff format .`.

## 5. Cross-platform differences that change results

| Difference | Consequence | Handling |
|---|---|---|
| Case sensitivity: macOS is case-insensitive, Linux is not | `RawItem.py` vs `rawitem.py` passes locally, fails in CI | All module names lowercase; the ubuntu job catches case bugs |
| Line endings: Windows CRLF | **Shifts the snapshot content hash**, so one corpus yields two `snapshotId`s | `.gitattributes` enforces LF; read files with explicit `encoding` |
| Path separators | Broken joins | Use `pathlib.Path` everywhere; never concatenate `"/"` or `"\\"` |
| Default encoding: UTF-8 on macOS, cp936 on Windows | Garbled sample pages | Always pass `encoding=`; detect Japanese encodings with `charset-normalizer` |
| Unicode normalisation: macOS uses NFD for filenames | Japanese titles split into two files | **Never use a title as a filename** — use media_id / hash |
| Time zone | Corrupted snapshot windows | Store UTC internally, convert for display only |
| Embedding device: macOS has MPS | Floating-point results differ slightly per device | Pin `ACGN_EMBED_DEVICE=cpu` for indexing and evaluation; any device change must be recorded in the run |

## 6. Division of labor

The authoritative ownership table (primary / secondary / cross-review) lives in [`ROADMAP.md`](ROADMAP.md) §4.
This section keeps only the part that is a collaboration rule rather than a schedule.

**The 2026-10-28 revision moved two responsibilities**, which matters for who you ask:
1. **Isla contact: B → F.** F produces the knowledge base and `DailyDigest`, so F knows what retrieval output
   looks like; B remains the sole author of `ports.py`. That gives an "F outward, B inward" pairing and satisfies
   the bus-factor rule.
2. **Metric definitions: F → E.** E implements the runner, so metric rules and runner stay with one person.

**F's stage-1 boundary (important)**: this stage does *not* integrate Isla — it only freezes the contract. So
before 11/24 F does exactly three things and never touches Isla repository code:

- Read `contracts/KnowledgePort.md` + schema and raise consumer-side questions (end of W1, after B's G1a output)
- Write a one-page integration note plus one curl example (lives in `contracts/`, tracked)
- Only after December: the merge work (`context.ts` third bucket, `provenance.ts` fourth source kind)

## 7. Rhythm

- Daily: whoever changed the contract or touched an invariant posts one line in the group.
- Weekly: D updates the contribution log; E runs `run_eval.py` once and archives the result.
- Stuck for 30 minutes? Ask. Crawling selectors and rate limits especially are not worth solo grinding.
