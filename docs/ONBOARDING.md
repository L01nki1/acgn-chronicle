# ONBOARDING — read this first

> Scope: engineering collaboration details. Spec 06 §1 states the rules; this file turns them into a checklist.
> When a rule changes, change it here first, then ping everyone in the PR.

## 1. Environment (5 members on macOS, 1 on Windows)

| Item | macOS (most of us) | Windows (1 member) |
|---|---|---|
| Python | 3.13 (`uv python install 3.13`; `uv sync` provisions it from `.python-version` automatically) | 3.13 (same uv path; the standalone uv installer works on Windows) |
| Shell | zsh (system bash is 3.2 — do not use bash 4+ syntax) | Git Bash (not cmd / PowerShell) |
| Virtualenv | `uv sync --extra dev` creates `.venv` from `uv.lock`; activation optional — `uv run <cmd>` uses it | same (`uv run` finds `.venv\Scripts` on its own) |
| torch wheel | native arm64, installs cleanly | fine |

On **Linux**, `pyproject.toml` resolves torch from the PyTorch CPU index automatically (the PyPI linux
wheel bundles CUDA and is ~3 GB bigger) — no flags needed. Plain `pip install -e ".[dev]"` still works
if you maintain the venv by hand; `uv.lock` is the canonical resolution either way.

**Do not** commit `.sh` / `.ps1` / `.bat` files as entry points. Every repeatable action goes through
`python scripts/xxx.py` or `python -m acgn_rag...`, so both platforms share one command set.

## 2. Branches and commits (06 §1)

- `main` only moves through PRs. This is enforced, not just agreed: a direct `git push origin main` is
  rejected by the repository ruleset with `GH013 ... Changes must be made through a pull request`, and the
  two required status checks must be green before the merge button enables.
- One long-lived branch per WP: `wp1-ingest` / `wp2-kb` / `wp3-retrieve` / `wp4-ground` / `wp5-eval` / `wp6-web`.
  Personal work uses short-lived branches on top: `wp1-ingest/a-toei-adapter`.
- Commit message: `[WP1] add Toei html list adapter`; contract changes get the `[contract]` prefix.
- Contract changes (`ports.py`) must explain the reason in the PR description and re-export the schema (§4).

### 2.1 Where the checks fire

**You only ever do two things: commit and push.** Do not run the checks by hand — the table below is what
runs for you.

| Moment | What runs | Mechanism | Bypassable |
|---|---|---|---|
| `git commit` on your branch | lint + format on the **staged `.py` files only**; contract check if `ports.py` is staged | local hook `.githooks/pre-commit` | yes, `--no-verify` |
| `git push` your branch | the four gates | local hook `.githooks/pre-push`, then CI on the pushed commit | locally yes — the CI run still happens |
| open or update a PR | the four gates on ubuntu, macOS **and** Windows | CI, `pull_request` event | no |
| merge into `main` | the four gates again on `main` | CI, `push` event | no, once the ruleset below is on |

Two CI runs for one commit (`push` and `pull_request`) is intentional: the push run tells you before you ask
anyone to review, the PR run is the one a reviewer reads and the one a ruleset can require.

**Correctness is guaranteed by CI, not by the hooks.** The local hooks are optional: they only move the
feedback earlier, so you find out in one second instead of waiting for a CI run. They exist per clone
(`python scripts/install_hooks.py`) because `core.hooksPath` is local configuration that git cannot commit —
if you skip that command nothing breaks, the PR checks still catch everything. See §4.2 for what each hook
deliberately does *not* do.

**The only layer nobody can bypass is server-side.** To block merging while a check is red, the repository
owner sets it once: `Settings → Rules → New branch ruleset` → target `main` → **Require status checks to
pass** → select `ubuntu-latest / py3.13`, `macos-latest / py3.13` and `windows-latest / py3.13`. Until that
exists, a red PR can still be merged by whoever presses the button.

## 3. PR checklist (the reviewer ticks each line)

The same list ships as `.github/pull_request_template.md`, so it appears in the description of every PR you
open — you do not have to remember it.

1. CI is green on all four gates (lint / format / contract check / tests) on every platform you can see.
   The author does not have to have run anything locally — "green in CI" is the requirement, not
   "green on my laptop"
2. Nothing leaked into the commit: `data/`, `.env`, model files (`git status` first)
3. New directories match the ownership map in spec 02 §2; cross-WP edits @ the owner in the description
4. Any change touching an invariant (spec 02 §3) comes with a test
5. The reviewer is that WP's cross-reviewer, not yourself

## 4. "CI does not use the network" — exact meaning

**Meaning**: CI may go online only while installing dependencies; while tests run, **no external request is allowed**.

CI installs the `dev`, `store` and `llm` extras, so Chroma tests run on ubuntu, macOS and Windows.
The `embed` extra and torch stay out; encoder tests use deterministic fakes and numpy from `dev`.
Installing the LLM client does not permit live API calls in tests.

| Allowed | Forbidden |
|---|---|
| `uv sync --frozen --extra dev --extra store --extra llm` from the package indexes | `httpx.get("https://...")` against a live site inside a test |
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

### 4.1 The four gates, and how to reproduce a red CI run locally

You are not expected to run these before pushing — CI does it. This section exists so that when CI
*is* red, you can reproduce it in one command instead of guessing:

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

### 4.2 Optional: run the gates locally on commit and push

Installing these is **not required** — CI already gates every push and every PR. It only buys you faster
feedback (one second locally instead of a CI round trip). Skip it if you prefer.

```bash
python scripts/install_hooks.py            # install (sets core.hooksPath = .githooks)
python scripts/install_hooks.py --check    # is it active in this clone?
```

What runs when:

| Hook | Runs | Time | Purpose |
|---|---|---|---|
| `pre-commit` | `ruff check` + `ruff format --check` on **staged `.py` files only**; the contract check when `ports.py` is staged | well under a second | catch the two failures that are cheap to fix now and annoying to explain later |
| `pre-push` | all four gates via `python scripts/check.py` | a few seconds | nothing leaves the machine unless CI would be green |

Deliberate design choices, so nobody "fixes" them the wrong way:

- **Staged files only on commit.** Checking the whole tree on every commit gets slow, and slow hooks teach
  people to type `--no-verify`, which skips the real problems too.
- **Tests are not in `pre-commit`.** They belong to the push gate; a 3-second commit is a commit you make.
- **Missing tooling warns and lets the commit through; a real violation blocks it.** A member who has not run
  `uv sync --extra dev` yet gets a clear message, not a dead end — an environment problem must not push
  people toward `--no-verify`.
- **Hooks live in `.githooks/`, tracked by git.** They are not copied into `.git/hooks`, because `.git/` is
  never updated by `git pull`, so a copied hook silently rots. `core.hooksPath` is a per-clone setting, which
  is why every new clone needs the one command above.

`git commit --no-verify` and `git push --no-verify` exist, but they only move the failure to a **public CI
run** that the whole group sees. If a hook is wrong, fix the hook in `.githooks/` and send a PR.

## 5. Cross-platform differences that change results

CI runs the same four gates on **ubuntu, macOS and Windows**, so a difference that only appears on one
platform is caught before a PR is merged. What CI cannot tell you is *why* it broke on one platform only —
that is what this table is for.

| Difference | Consequence | Handling |
|---|---|---|
| Case sensitivity: macOS is case-insensitive, Linux is not | `RawItem.py` vs `rawitem.py` passes locally, fails in CI | All module names lowercase; the ubuntu job catches case bugs |
| Line endings: Windows CRLF | **Shifts the snapshot content hash**, so one corpus yields two `snapshotId`s | `.gitattributes` enforces LF; read files with explicit `encoding` |
| Path separators | Broken joins | Use `pathlib.Path` everywhere; never concatenate `"/"` or `"\\"` |
| Default encoding: UTF-8 on macOS, cp936 on Windows | Garbled sample pages | Always pass `encoding=`; detect Japanese encodings with `charset-normalizer` |
| Unicode normalisation: macOS uses NFD for filenames | Japanese titles split into two files | **Never use a title as a filename** — use media_id / hash |
| Time zone | Corrupted snapshot windows | Store UTC internally, convert for display only |
| Embedding device: macOS has MPS | Floating-point results differ slightly per device | Never mix vectors across devices: every build records its **encoder identity** and the cache is keyed by it (RUNBOOK §3.4); `.env.example` defaults to `cpu`, the lane every member can run |

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
