"""CLI: run one evaluation configuration against one frozen snapshot.

Usage:
    python scripts/run_eval.py --config A --snapshot <id>
    python scripts/run_eval.py --config A --snapshot <id> --qa eval/qa_set.jsonl --out eval/runs

The runner writes `eval/runs/<run_id>/` with `meta.json`, `answers.jsonl`, `labels.jsonl`,
`metrics.json`, `resolved_evidence.json` and `disputes.md` (EVAL_PROTOCOL §7).
`NullAnswerer` is used until B's `/ask` lands; retrieval metrics are still measured.

Owner: E | Spec: docs/EVAL_PROTOCOL.md §7-§8, docs/RUNBOOK.md §1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acgn_rag.config import load_settings
from acgn_rag.core.embed import Embedder
from acgn_rag.core.eval.qa import load_jsonl, validate
from acgn_rag.core.eval.runner import (
    NullAnswerer,
    RunSpec,
    load_build_records,
    run,
    verify_build_identity,
)
from acgn_rag.core.store import ChromaStore, collection_name

CONFIG_CHOICES = ("A", "A'", "A0", "B")
COLLECTION_LABELS = {"A'": "A-prime"}  # Chroma names reject "'"
ROOT = Path(__file__).resolve().parents[1]


def _git_sha() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one evaluation configuration")
    parser.add_argument("--config", required=True, choices=CONFIG_CHOICES)
    parser.add_argument("--snapshot", required=True, help="frozen snapshot id")
    parser.add_argument("--qa", default="eval/qa_set.jsonl", help="QA JSONL path")
    parser.add_argument("--out", default="eval/runs", help="run archive root")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--run-id", default=None, help="override the generated run id")
    parser.add_argument("--window-start", default=None)
    parser.add_argument("--window-end", default=None)
    parser.add_argument(
        "--allow-invalid-qa",
        action="store_true",
        help="development only: run against a draft QA set that fails the protocol quotas",
    )
    args = parser.parse_args(argv)

    qa_path = Path(args.qa)
    if not qa_path.exists():
        print(f"NOT FOUND: {qa_path}", file=sys.stderr)
        return 2
    items = load_jsonl(qa_path)
    if not items:
        print(f"EMPTY: {qa_path}", file=sys.stderr)
        return 2

    if not args.allow_invalid_qa:
        problems = validate(items)
        if problems:
            print(f"INVALID QA SET ({len(problems)} problem(s)):", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            print("Use --allow-invalid-qa only for development runs.", file=sys.stderr)
            return 1

    window_start = args.window_start or min(str(item["window"][0]) for item in items)
    window_end = args.window_end or max(str(item["window"][1]) for item in items)

    settings = load_settings()
    label = COLLECTION_LABELS.get(args.config, args.config)
    try:
        build_identity, collection = load_build_records(
            settings.paths.index,
            snapshot_id=args.snapshot,
            config=args.config,
            label=label,
        )
    except OSError as exc:
        print(f"BUILD RECORD NOT FOUND: {exc}", file=sys.stderr)
        return 2
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"INVALID BUILD RECORD: {exc}", file=sys.stderr)
        return 1

    # TODO(#31): after F's injective full-id mapping lands, bind by the recorded collection
    # instead of recomputing the short name. Until then this equality is the fail-safe.
    expected_collection = collection_name(args.snapshot, label)
    if collection != expected_collection:
        print(
            f"COLLECTION MISMATCH: build record={collection!r}, "
            f"current naming={expected_collection!r}",
            file=sys.stderr,
        )
        return 1

    embed_config = replace(
        settings.embed,
        model_name=str(build_identity["model"]),
        device=str(build_identity["device"]),
        batch_size=int(build_identity["batch_size"]),
    )
    embedder = Embedder(embed_config)
    identity_problems = verify_build_identity(
        build_identity,
        embedder.identity().to_metadata(),
    )
    if identity_problems:
        print("BUILD IDENTITY MISMATCH:", file=sys.stderr)
        for problem in identity_problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    store = ChromaStore(settings.paths.index / "chroma", settings.store)
    store.bind(args.snapshot, label)

    spec = RunSpec(
        snapshot_id=args.snapshot,
        window_start=window_start,
        window_end=window_end,
        config=args.config,
        out_root=Path(args.out),
        qa_path=qa_path,
        top_k=args.top_k,
        over_fetch=settings.store.over_fetch_factor,
        run_id=args.run_id,
        git_sha=_git_sha(),
        build_identity=build_identity,
        collection=collection,
        llm={
            "provider": settings.llm.base_url or "none",
            "model": settings.llm.model or "none",
            "version": settings.llm.model_version or "none",
            "temperature": settings.llm.temperature,
            "seed": 0,
        },
    )
    result = run(
        spec,
        store=store,
        embedder=embedder,
        answerer=NullAnswerer(),
        qa_items=items,
    )
    print(f"run_id: {result.run_id}")
    print(f"out:    {result.out_dir}")
    print(f"hit@k:  {result.metrics['retrieval_hit_rate']}")
    print(f"recall: {result.metrics['evidence_recall']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
