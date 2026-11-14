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
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acgn_rag.config import LLMConfig, load_settings
from acgn_rag.core.embed import Embedder
from acgn_rag.core.eval.qa import load_jsonl, validate
from acgn_rag.core.eval.runner import (
    Answerer,
    NullAnswerer,
    RunResult,
    RunSpec,
    bind_existing_collection,
    build_evidence_resolver,
    load_build_records,
    load_snapshot_member_urls,
    read_jsonl_rows,
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


_CURRENT_CUDA_DEVICE = -1


def _cuda_index(device: str | None) -> int | None:
    """Return the CUDA index named by the encoder device.

    An unnumbered `cuda` means the current CUDA device in PyTorch, not device 0; it returns the
    sentinel `_CURRENT_CUDA_DEVICE` for the caller to resolve. A malformed CUDA string raises:
    silently recording "no GPU" would hide a bad build sidecar.
    """
    value = str(device or "").strip().lower()
    if value == "cuda":
        return _CURRENT_CUDA_DEVICE
    if value.startswith("cuda:"):
        suffix = value.split(":", 1)[1]
        try:
            index = int(suffix)
        except ValueError as exc:
            raise ValueError(f"invalid CUDA device {device!r}") from exc
        if index < 0:
            raise ValueError(f"invalid CUDA device {device!r}")
        return index
    return None


def _configure_determinism(seed: int = 0, *, device: str | None = None) -> dict[str, object]:
    """Set and record the torch determinism knobs for a reportable run.

    `device` is the encoder's actual device from the build sidecar. A CUDA run must record the
    name of that device, not device 0, or a two-device host collapses to one physical model.
    """
    index = _cuda_index(device)
    try:
        import torch
    except Exception:
        return {
            "torch_available": False,
            "torch_seed": seed,
            "use_deterministic_algorithms": False,
            "warn_only": None,
            "gpu_model": None,
        }
    gpu_model: str | None = None
    try:
        if index == _CURRENT_CUDA_DEVICE:
            index = int(torch.cuda.current_device()) if torch.cuda.is_available() else None
        if index is not None and torch.cuda.is_available():
            gpu_model = str(torch.cuda.get_device_name(index))
    except Exception:
        gpu_model = None
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
        "gpu_model": gpu_model,
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
        answer_model = str(getattr(config, "answer_model", "") or "")
        answer_version = str(getattr(config, "answer_model_version", "") or "")
        if not answer_version and answer_model == str(getattr(config, "model", "") or ""):
            # Backward compatibility only when extraction and answer use the same model. When the
            # models differ, the extraction version would misidentify the answer model.
            answer_version = str(getattr(config, "model_version", "") or "")
        return {
            "provider": str(getattr(config, "answer_base_url", "") or "none"),
            "model": answer_model or "none",
            "version": answer_version or "none",
            "temperature": float(getattr(config, "temperature", 0.0)),
            "seed": None,
        }
    return {
        "provider": str(getattr(config, "base_url", "") or "none"),
        "model": str(getattr(config, "model", "") or "none"),
        "version": str(getattr(config, "model_version", "") or "none"),
        "temperature": float(getattr(config, "temperature", 0.0)),
        "seed": None,
    }


def _llm_metadata_problems(metadata: Mapping[str, object]) -> list[str]:
    """Validate the archived LLM identity; shared by preflight and post-run validation."""
    problems: list[str] = []
    for field in ("provider", "model", "version"):
        value = metadata.get(field)
        if not isinstance(value, str) or not value or value == "none":
            problems.append(f"meta.llm.{field} is unset")
    if not isinstance(metadata.get("temperature"), (int, float)):
        problems.append("meta.llm.temperature is unset")
    if metadata.get("seed") is not None:
        problems.append("meta.llm.seed must be null when the request carries none")
    return problems


def _live_metadata_problems(config: LLMConfig, metadata: Mapping[str, object]) -> list[str]:
    """Preflight checks for a live run, before any index or answer work."""
    problems = _llm_metadata_problems(metadata)
    answer_model = str(getattr(config, "answer_model", "") or "")
    extraction_model = str(getattr(config, "model", "") or "")
    answer_version = str(getattr(config, "answer_model_version", "") or "").strip()
    if (
        answer_model
        and extraction_model
        and answer_model != extraction_model
        and not answer_version
    ):
        problems.append(
            "ACGN_ANSWER_MODEL_VERSION is required when the answer model differs from the extraction model"
        )
    return problems


def _validate_live_run(result: RunResult) -> list[str]:
    """Enforce the ROADMAP ④ definition of a real-LLM archive.

    A live run counts only when every row is answered and `meta.llm` records the actual provider,
    model, version and temperature. `seed` stays `None` because the request carries none.
    """
    problems: list[str] = _llm_metadata_problems(result.meta.get("llm") or {})

    build_identity = result.meta.get("build_identity") or {}
    device = str(build_identity.get("device") or "").lower()
    gpu_model = (result.meta.get("determinism") or {}).get("gpu_model")
    if device.startswith("cuda") and not (isinstance(gpu_model, str) and gpu_model):
        problems.append("meta.determinism.gpu_model is unset for a CUDA encoder")

    path = result.out_dir / "answers.jsonl"
    try:
        rows = read_jsonl_rows(path)
    except ValueError as exc:
        # The archive was just written; a parse failure is an archive defect, not a crash.
        problems.append(f"answers.jsonl is unreadable: {exc}")
        return problems
    if not rows:
        problems.append("answers.jsonl is empty")
        return problems
    unanswered = [
        str(row.get("qa_id", "?"))
        for row in rows
        if row.get("status") != "answered" or not str(row.get("answer") or "").strip()
    ]
    if unanswered:
        problems.append(
            f"{len(unanswered)}/{len(rows)} item(s) not answered: {', '.join(unanswered[:5])}"
        )
    if not any(row.get("provider_completion") is True for row in rows):
        problems.append("no returned provider completion in answers.jsonl")
    return problems


def _nonreportable_exit(result: RunResult) -> int:
    """Return 1 for a non-reportable archive; print the reason for the operator."""
    if result.meta.get("reportable", False):
        return 0
    reason = result.meta.get("non_reportable_reason") or "unknown"
    suffix = " (PR #48 GPU-first policy, pending)" if reason == "non_gpu_device" else ""
    print(f"NOT REPORTABLE: {reason}{suffix}", file=sys.stderr)
    return 1


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
    llm_metadata = _llm_metadata(settings.llm, args.answerer)
    if args.answerer == "live":
        metadata_problems = _live_metadata_problems(settings.llm, llm_metadata)
        if metadata_problems:
            print("ANSWERER CONFIG: " + "; ".join(metadata_problems), file=sys.stderr)
            return 2
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

    try:
        determinism = _configure_determinism(device=str(build_identity.get("device") or ""))
    except ValueError as exc:
        print(f"DETERMINISM CONFIG: {exc}", file=sys.stderr)
        return 1
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
        llm=llm_metadata,
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
    if args.answerer == "live":
        problems = _validate_live_run(result)
        if problems:
            print("NOT A REAL-LLM RUN: " + "; ".join(problems), file=sys.stderr)
            return 1
    if result.metrics.get("provisional"):
        print(
            "PROVISIONAL: no evidence resolver configured; unresolved gold urls are excluded "
            "from retrieval metrics and this run is not reportable.",
            file=sys.stderr,
        )
    if _nonreportable_exit(result):
        return 1
    print(f"hit@k:  {result.metrics['retrieval']['hit']}")
    print(f"mrr:    {result.metrics['retrieval']['mrr']}")
    print(f"recall: {result.metrics['retrieval']['evidence_recall']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
