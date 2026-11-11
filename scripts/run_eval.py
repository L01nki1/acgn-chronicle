"""CLI: run one evaluation configuration against one frozen snapshot.

Usage:
    python scripts/run_eval.py --config A --snapshot <id>
    python scripts/run_eval.py --config A --snapshot <id> --qa eval/qa_set.jsonl --out eval/runs
    python scripts/run_eval.py --config A --snapshot <id> --answerer live

The runner writes `eval/runs/<run_id>/` with `meta.json`, `answers.jsonl`, `labels.jsonl`,
`metrics.json`, `resolved_evidence.json` and `disputes.md` (EVAL_PROTOCOL §7).
`--answerer null` (the default) keeps retrieval-only measurement; `--answerer live` requires the
answer-line provider/model/key and archives real generated answers.

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

from acgn_rag.config import LLMConfig, load_settings
from acgn_rag.core.embed import Embedder
from acgn_rag.core.eval.qa import load_jsonl, validate
from acgn_rag.core.eval.runner import (
    Answerer,
    NullAnswerer,
    RunSpec,
    bind_existing_collection,
    build_evidence_resolver,
    load_build_records,
    load_snapshot_member_urls,
    resolve_window,
    run,
    verify_build_identity,
)
from acgn_rag.core.store import ChromaStore

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


def _configure_determinism(seed: int = 0) -> dict[str, object]:
    """Set and record the torch determinism knobs for a reportable run."""
    try:
        import torch
    except Exception:
        return {
            "torch_available": False,
            "torch_seed": seed,
            "use_deterministic_algorithms": False,
            "warn_only": None,
        }
    torch.manual_seed(seed)
    warn_only = True
    enabled = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        enabled = True
    except Exception:
        enabled = False
    return {
        "torch_available": True,
        "torch_seed": seed,
        "use_deterministic_algorithms": enabled,
        "warn_only": warn_only if enabled else None,
    }


def _build_answerer(mode: str, config: LLMConfig) -> Answerer:
    """Return the retrieval-only null answerer or the configured live answerer.

    The live path is deliberately lazy: the ground package imports the optional `llm` extra
    only when a live run is requested, so retrieval-only CI does not need it.
    """
    if mode == "null":
        return NullAnswerer()
    answer_api_key = str(getattr(config, "answer_api_key", "") or "").strip()
    answer_model = str(getattr(config, "answer_model", "") or "").strip()
    if not answer_api_key or not answer_model:
        raise ValueError("--answerer live requires ACGN_ANSWER_API_KEY and ACGN_ANSWER_MODEL")
    from acgn_rag.core.ground.provider import make_grounded_answerer

    return make_grounded_answerer(config)


def _llm_metadata(config: LLMConfig, answerer_mode: str) -> dict[str, object]:
    """Record the answer-line identity for a live run; keep the prior shape for null runs."""
    if answerer_mode == "live":
        return {
            "provider": str(getattr(config, "answer_base_url", "") or "none"),
            "model": str(getattr(config, "answer_model", "") or "none"),
            "version": str(
                getattr(config, "answer_model_version", "")
                or getattr(config, "model_version", "")
                or "none"
            ),
            "temperature": float(getattr(config, "temperature", 0.0)),
            "seed": 0,
        }
    return {
        "provider": str(getattr(config, "base_url", "") or "none"),
        "model": str(getattr(config, "model", "") or "none"),
        "version": str(getattr(config, "model_version", "") or "none"),
        "temperature": float(getattr(config, "temperature", 0.0)),
        "seed": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one evaluation configuration")
    parser.add_argument("--config", required=True, choices=CONFIG_CHOICES)
    parser.add_argument("--snapshot", required=True, help="frozen snapshot id")
    parser.add_argument("--qa", default="eval/qa_set.jsonl", help="QA JSONL path")
    parser.add_argument("--out", default="eval/runs", help="run archive root")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--run-id",
        default=None,
        help="override the generated identity-scoped run id",
    )
    parser.add_argument(
        "--window-start",
        default=None,
        help="frozen corpus start; required when the snapshot manifest is absent",
    )
    parser.add_argument(
        "--window-end",
        default=None,
        help="frozen corpus end; required when the snapshot manifest is absent",
    )
    parser.add_argument(
        "--answerer",
        choices=("null", "live"),
        default="null",
        help="null: retrieval-only archives (default); live: require the answer-line provider",
    )
    parser.add_argument(
        "--allow-invalid-qa",
        action="store_true",
        help="development only: run against a draft QA set that fails the protocol quotas",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="enforce the PR #48 GPU-first policy; CPU runs are then non-reportable",
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

    settings = load_settings()
    try:
        answerer = _build_answerer(args.answerer, settings.llm)
    except (ImportError, ValueError) as exc:
        print(f"ANSWERER CONFIG: {exc}", file=sys.stderr)
        return 2
    try:
        window_start, window_end, window_verified = resolve_window(
            settings.paths.snapshots / args.snapshot,
            args.snapshot,
            args.window_start,
            args.window_end,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"WINDOW RESOLUTION FAILED: {exc}", file=sys.stderr)
        return 1
    member_urls = load_snapshot_member_urls(settings.paths.snapshots / args.snapshot)
    label = COLLECTION_LABELS.get(args.config, args.config)
    try:
        build_identity, collection, counts, chunker = load_build_records(
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

    # The sidecar is the authority: use its actual collection name (legacy or full-id).
    # Never reconstruct the physical name from the snapshot id.

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
    try:
        bound = bind_existing_collection(
            store,
            snapshot_id=args.snapshot,
            chunk_config=label,
            collection=collection,
        )
    except (FileNotFoundError, LookupError) as exc:
        print(f"COLLECTION NOT FOUND: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"COLLECTION BINDING FAILED: {exc}", file=sys.stderr)
        return 1

    try:
        evidence_resolver = build_evidence_resolver(bound)
    except RuntimeError as exc:
        print(f"EVIDENCE RESOLVER FAILED: {exc}", file=sys.stderr)
        return 1

    determinism = _configure_determinism()
    spec = RunSpec(
        snapshot_id=args.snapshot,
        window_start=window_start,
        window_end=window_end,
        window_verified=window_verified,
        config=args.config,
        out_root=Path(args.out),
        qa_path=qa_path,
        top_k=args.top_k,
        over_fetch=settings.store.over_fetch_factor,
        run_id=args.run_id,
        git_sha=_git_sha(),
        build_identity=build_identity,
        collection=collection,
        counts=counts,
        chunker=chunker,
        llm=_llm_metadata(settings.llm, args.answerer),
        determinism=determinism,
        require_gpu=args.require_gpu,
        qa_validated=not args.allow_invalid_qa,
        qa_source="cli",
    )
    result = run(
        spec,
        store=store,
        embedder=embedder,
        answerer=answerer,
        qa_items=items,
        evidence_resolver=evidence_resolver,
        snapshot_member_urls=member_urls,
    )
    print(f"run_id: {result.run_id}")
    print(f"out:    {result.out_dir}")
    if result.metrics.get("provisional"):
        print(
            "PROVISIONAL: no evidence resolver configured; unresolved gold urls are excluded "
            "from retrieval metrics and this run is not reportable.",
            file=sys.stderr,
        )
    if not result.meta.get("reportable", False):
        reason = result.meta.get("non_reportable_reason") or "unknown"
        suffix = " (PR #48 GPU-first policy, pending)" if reason == "non_gpu_device" else ""
        print(f"NOT REPORTABLE: {reason}{suffix}", file=sys.stderr)
    print(f"hit@k:  {result.metrics['retrieval']['hit']}")
    print(f"mrr:    {result.metrics['retrieval']['mrr']}")
    print(f"recall: {result.metrics['retrieval']['evidence_recall']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
