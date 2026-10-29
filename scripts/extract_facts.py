"""CLI: L0 fact extraction over announcement chunks, following the RUNBOOK §3.6 SOP.

Usage:
    python scripts/extract_facts.py --snapshot <id> --input chunks.jsonl --dry-run
    python scripts/extract_facts.py --snapshot <id> --input chunks.jsonl [--limit 100]

Input rows (JSONL): chunk_id / text / url / publisher. Output goes to
``data/facts/<snapshot>/facts.jsonl`` (FactUpdate rows, written through a temp file and an
atomic rename, RUNBOOK §3.2) plus ``ir_skipped.jsonl``, the ingest-side corporate tag for
IR-prefiltered chunks (interface issue #17, option 1 - never a FactUpdate field).

TODO(snapshot reader): A owns the snapshot/chunk reader. Until that interface lands this script
reads a raw chunks.jsonl via ``--input`` (default ``data/chunks/<snapshot>/chunks.jsonl``);
replace ``_load_chunks`` with the reader and drop ``--input`` when it exists. The RUNBOOK §1
row keeps ``--input`` until then.

Exit codes: 0 = done, 2 = missing input or unusable extraction config. Zero network at import
time: the vendor client is built only for a real (non-dry) run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pydantic import ValidationError

from acgn_rag.config import load_settings
from acgn_rag.core.facts import (
    BATCH_SIZE,
    AnnouncementChunk,
    LLMClient,
    expand_to_fact_updates,
    extract_facts,
    is_ir_release,
    make_openai_client,
)
from acgn_rag.core.ports import FactUpdate


def _load_chunks(path: Path, limit: int | None) -> list[AnnouncementChunk]:
    """Read chunk rows from JSONL, stopping at ``limit``. A malformed line aborts the run."""
    chunks: list[AnnouncementChunk] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                chunks.append(AnnouncementChunk.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise ValueError(f"{path}:{lineno}: invalid chunk row ({exc})") from exc
            if limit is not None and len(chunks) >= limit:
                break
    return chunks


def _atomic_write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
    """Write JSONL through a temp file + ``os.replace`` (RUNBOOK §3.2).

    A reader either sees the previous file or the complete new one, never a half-written one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    os.replace(tmp, path)


def main(argv: list[str] | None = None, *, client: LLMClient | None = None) -> int:
    """Run the extraction. ``client`` is injectable so tests run fully offline."""
    parser = argparse.ArgumentParser(description="L0 fact extraction (RUNBOOK §3.6)")
    parser.add_argument("--snapshot", required=True, help="snapshot id the chunks belong to")
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "chunks.jsonl with chunk_id/text/url/publisher rows; default "
            "data/chunks/<snapshot>/chunks.jsonl"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process at most N chunks (applied before the IR prefilter)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="count batches and IR-skipped items; no LLM call, nothing written",
    )
    args = parser.parse_args(argv)

    input_path = (
        Path(args.input) if args.input else Path("data/chunks") / args.snapshot / "chunks.jsonl"
    )
    if not input_path.exists():
        print(f"NOT FOUND: {input_path} - pass --input <chunks.jsonl>", file=sys.stderr)
        return 2
    try:
        chunks = _load_chunks(input_path, args.limit)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    settings = load_settings()
    out_dir = settings.paths.facts / args.snapshot
    facts_path = out_dir / "facts.jsonl"
    ir_path = out_dir / "ir_skipped.jsonl"

    if args.dry_run:
        batches = planned_calls = ir_skipped = 0
        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start : start + BATCH_SIZE]
            batches += 1
            skipped = sum(1 for chunk in batch if is_ir_release(chunk.text))
            ir_skipped += skipped
            if skipped < len(batch):  # at least one candidate -> exactly one planned call
                planned_calls += 1
        print(
            f"dry-run: snapshot={args.snapshot} chunks={len(chunks)} batches={batches} "
            f"planned_calls={planned_calls} extracted_rows=0 degraded=0 ir_skipped={ir_skipped}"
        )
        print("dry-run: no LLM call was made and nothing was written")
        return 0

    if client is None:
        if not settings.llm.extract_model:
            print(
                "no extraction model configured: set ACGN_EXTRACT_MODEL (docs/RUNBOOK.md §3.6)",
                file=sys.stderr,
            )
            return 2
        try:
            client = make_openai_client(settings.llm)
        except Exception as exc:  # missing [llm] extra or credentials: report, do not traceback
            print(f"cannot build the extraction client: {exc}", file=sys.stderr)
            return 2

    batches = calls = 0
    rows: list[FactUpdate] = []
    degraded: list[str] = []
    ir_rows: list[dict[str, str]] = []
    for start in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[start : start + BATCH_SIZE]
        batch = extract_facts(batch_chunks, client)
        batches += 1
        calls += batch.calls
        degraded.extend(batch.degraded_chunk_ids)
        skipped_ids = set(batch.ir_skipped_chunk_ids)
        for chunk in batch_chunks:
            if chunk.chunk_id in skipped_ids:
                ir_rows.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "url": chunk.url,
                        "publisher": chunk.publisher,
                        "ingest_category": "corporate",
                    }
                )
        dropped = skipped_ids | set(batch.degraded_chunk_ids)
        candidates = [chunk for chunk in batch_chunks if chunk.chunk_id not in dropped]
        # extract_facts aligns extracted[i] with the i-th candidate chunk of the batch
        assert len(batch.extracted) == len(candidates)
        rows.extend(expand_to_fact_updates(batch.extracted, candidates))

    _atomic_write_jsonl(facts_path, [row.model_dump() for row in rows])
    _atomic_write_jsonl(ir_path, ir_rows)
    print(
        f"snapshot={args.snapshot} chunks={len(chunks)} batches={batches} calls={calls} "
        f"extracted_rows={len(rows)} degraded={len(degraded)} ir_skipped={len(ir_rows)}"
    )
    print(f"wrote {facts_path}")
    if ir_rows:
        print(f"wrote {ir_path} (ingest-side corporate tag, interface issue #17 option 1)")
    if degraded:
        print(f"note: {len(degraded)} chunk(s) degraded to plain text after retries - not dropped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
