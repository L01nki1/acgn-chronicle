# RUNBOOK — commands, failures, SOPs

> Scope: operational manual. Personal workarounds do not belong here; only things that still hold on another machine.

## 1. Commands

| Goal | Command | Status |
|---|---|---|
| Install deps (no heavy extras) | `pip install -e ".[dev]"` | ready |
| Install embedding + store | `pip install -e ".[dev,embed,store]"` | ready |
| Contract consistency | `python scripts/export_schema.py --check` | ready |
| All tests (zero network) | `pytest -q` | ready |
| Network-marked tests | `pytest -m network -q` | ready (local only) |
| Historical backfill | `python scripts/backfill.py --since ... --until ...` | pending A |
| Daily increment | `python scripts/discover_daily.py` | pending A |
| Build index | `python scripts/build_index.py --config A` | pending F |
| Fact extraction | `python scripts/extract_facts.py --snapshot <id>` | pending F |
| Evaluation | `python scripts/run_eval.py --config A --snapshot <id>` | pending E |
| Serve | `uvicorn acgn_rag.web.api:app --reload` | pending C |

## 2. Snapshot rebuild SOP

1. Back up `data/` before any backfill.
2. `python scripts/backfill.py --since <start> --until <end>`; check `items > 0` per source.
3. `python scripts/build_index.py --config A` (run the three configs sequentially — running them in parallel
   oversubscribes the CPU and is slower overall).
4. Verify idempotency: rebuild once and compare `snapshot_id`.
5. Post the `snapshot_id` + window in the group; every later run must use that id.

> When idempotency fails, the first suspects are always **manifest ordering / time zone / timestamp format**,
> not the hash function.

## 3. Common failures

| Symptom | Check first | Action |
|---|---|---|
| Item count drops to 0 | Did the site change its HTML or selectors? | Update selectors + fixtures; **never** add `try: return []` |
| Item count collapses | 403 / 429 / paging stopped working | Follow the response ladder in 01 §5.4; never rotate IPs |
| Garbled Japanese body | Header charset disagrees with the real encoding | Detect with `charset-normalizer`; assert the decode error rate |
| `top_k=5` returns 2 rows | Chroma's where-filter removed candidates | Raise over-fetch (`ACGN_OVERFETCH`) and assert the row count |
| Two `snapshot_id`s for one corpus | CRLF / ordering / time zone | See the note in §2 |
| Chunk count changes after rebuild | Chunk params edited, or tokenizer mismatch | Check `chunk/variants.py` params and the tokenizer |
| Model invents `chunk_id`s | The prompt passes ids instead of indices | Pass indices, map back in code, assert `cited ⊆ retrieved` |
| CI raises `NetworkBlocked` | A test made a real HTTP call | Mock with `respx`; mark genuinely online cases `@pytest.mark.network` |
| Green on macOS, red in CI | Filename casing | All module names lowercase |
| Green on Windows, red in CI | CRLF or path joins | `.gitattributes` enforces LF; use `pathlib` |
| `ruff check` green but `ruff format --check` red | An edit was never formatted; lint and format are separate gates | Run `ruff format .` and commit the result |

### 3.1 Chroma specifics — three pitfalls to verify before building the index

1. **Metadata holds scalars only** (`str / int / float / bool`), never arrays. `media_ids` is
   `Sequence[int]`, so writing it directly fails. Keep a SQLite reverse table `doc_id ↔ media_id`,
   translate the condition into `doc_id in (...)`, and pass `{"doc_id": {"$in": [...]}}` to Chroma.
2. **Store time as an integer epoch** (`published_ts`), not an ISO string. Range filters then compare
   numbers, which avoids format and time-zone ambiguity.
3. **A filter can return fewer than `top_k`.** Chroma trims a candidate set, so narrow conditions return
   little. Request `top_k × ACGN_OVERFETCH`, truncate afterwards, and assert
   `returned >= min(top_k, available_after_filter)`.

## 4. Emergency degradation

- **Corpus below 8,000 items**: decide before W3; downgrade the A/B comparison to "qualitative + small-sample
  quantitative" and say so in the report (04 R6).
- **LLM API unavailable**: `/ask` returns a stub answer (`abstained=true` with an explanation) so the site and
  citation jumps remain demonstrable.
- **No network during the demo**: fall back to the recording; the local index and frontend run offline.
