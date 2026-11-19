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
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

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
    qa_inputs_fingerprint,
    read_jsonl_rows,
    resolve_window,
    run,
    validate_abstention_metadata,
    validate_run_spec,
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


def _abstention_block(
    path: str | None, settings: object, answerer_mode: str
) -> dict[str, object] | None:
    """Load the calibration artifact that authorises a pre-generation threshold.

    A threshold without provenance is refused, and provenance without a threshold is refused:
    the archive must say what was applied and where it was calibrated.
    """
    llm = settings.llm
    configured = getattr(llm, "answer_min_retrieval_score", None)
    if answerer_mode != "live":
        if path is not None:
            raise ValueError("--abstention-calibration requires --answerer live")
        # Retrieval-only runs cannot apply a pre-generation score gate; record nulls even when
        # the environment carries a threshold set for a live answerer.
        return None
    if path is None:
        if configured is not None:
            raise ValueError(
                "ACGN_ANSWER_MIN_RETRIEVAL_SCORE is set but --abstention-calibration is missing"
            )
        return None
    artifact_path = Path(path)
    if artifact_path.name != "calibration.json":
        raise ValueError("--abstention-calibration must point at a calibration.json artifact")
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("calibration artifact must be a JSON object")
    block = payload.get("abstention", payload)
    if not isinstance(block, Mapping):
        raise ValueError("calibration artifact must be a JSON object")
    threshold = block.get("threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
    ):
        raise ValueError("calibration artifact threshold must be a finite number")
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("calibration artifact does not record its run_id")
    calibration = payload.get("calibration")
    if not isinstance(calibration, Mapping) or calibration.get("in_sample") is not True:
        raise ValueError("calibration artifact does not carry CalibrationResult.to_metadata()")
    if calibration.get("threshold") != threshold:
        raise ValueError("calibration artifact threshold does not match its calibration record")
    calibration_set = payload.get("set")
    if not isinstance(calibration_set, Mapping) or calibration_set.get("gate_set") is not True:
        raise ValueError(
            "calibration artifact is exploration-only and cannot author a reportable run"
        )
    if configured is None:
        raise ValueError(
            "calibration artifact provided but ACGN_ANSWER_MIN_RETRIEVAL_SCORE is unset"
        )
    if float(configured) != float(threshold):
        raise ValueError(
            f"ACGN_ANSWER_MIN_RETRIEVAL_SCORE ({configured}) "
            f"does not equal the calibration threshold ({threshold})"
        )
    identity = block.get("calibration_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("calibration artifact has no calibration_identity object")
    return {
        "threshold": threshold,
        "source": str(path),
        "calibration_identity": identity,
    }


def _preflight_device_problems(
    build_identity: Mapping[str, object],
    determinism: Mapping[str, object],
    *,
    require_gpu: bool,
) -> list[str]:
    """CUDA/GPU checks that must pass before a reportable run starts."""
    device = str(build_identity.get("device") or "").strip().lower()
    problems: list[str] = []
    if device.startswith("cuda"):
        gpu_model = determinism.get("gpu_model")
        if not isinstance(gpu_model, str) or not gpu_model:
            problems.append("CUDA encoder has no determinism.gpu_model")
    if require_gpu and not device.startswith("cuda"):
        problems.append("--require-gpu is set but the encoder device is not CUDA")
    return problems


def _calibration_set_problems(
    items: Sequence[Mapping[str, Any]], reference: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Validate the 30-item abstention instrument without the 32-item QA quota rules.

    The instrument is a different set with a different purpose: it has to keep the
    pre-registered 20 answerable / 10 unanswerable shape and stay disjoint from the QA set,
    so a threshold selected on it cannot leak into the reported QA numbers.
    """
    problems: list[str] = []
    if len(items) != 30:
        problems.append(f"calibration set has {len(items)} items, need exactly 30")
    answerable = sum(1 for item in items if item.get("answerable") is True)
    unanswerable = sum(1 for item in items if item.get("answerable") is False)
    if answerable != 20:
        problems.append(f"calibration set has {answerable} answerable items, need exactly 20")
    if unanswerable != 10:
        problems.append(f"calibration set has {unanswerable} unanswerable items, need exactly 10")
    ids = [str(item.get("qa_id") or "") for item in items]
    if any(not value for value in ids):
        problems.append("calibration set has an item without qa_id")
    if len(set(ids)) != len(ids):
        problems.append("calibration set has duplicate qa_id values")
    shared = sorted({value for value in ids} & {str(item.get("qa_id") or "") for item in reference})
    if shared:
        problems.append("calibration set shares qa_id with the QA set: " + ", ".join(shared[:5]))
    for item in items:
        if not str(item.get("question") or "").strip():
            problems.append(f"{item.get('qa_id')}: empty question")
        if item.get("answerable") is not True:
            continue
        if not str(item.get("gold_answer") or "").strip():
            problems.append(f"{item.get('qa_id')}: answerable item without gold_answer")
        evidence = item.get("gold_evidence")
        urls = evidence.get("urls") if isinstance(evidence, Mapping) else None
        if not urls:
            problems.append(f"{item.get('qa_id')}: answerable item without gold_evidence.urls")
    return problems


def _preflight_expectation_problems(
    *,
    counts: Mapping[str, object],
    qa_fingerprint: str,
    expect_docs: int | None,
    expect_chunks: int | None,
    expect_qa_fingerprint: str | None,
) -> list[str]:
    """Compare the transferred build/QA against the values agreed in the handoff."""
    problems: list[str] = []
    if expect_docs is not None and counts.get("docs") != expect_docs:
        problems.append(f"docs {counts.get('docs')} != expected {expect_docs}")
    if expect_chunks is not None and counts.get("chunks") != expect_chunks:
        problems.append(f"chunks {counts.get('chunks')} != expected {expect_chunks}")
    if expect_qa_fingerprint and qa_fingerprint != expect_qa_fingerprint:
        problems.append(f"qa fingerprint {qa_fingerprint} != expected {expect_qa_fingerprint}")
    return problems


def _validate_live_run(result: RunResult) -> list[str]:
    """Enforce the ROADMAP ④ definition of a real-LLM archive.

    A live run counts only when every row is answered and `meta.llm` records the actual provider,
    model, version and temperature. `seed` stays `None` because the request carries none.
    """
    problems: list[str] = _llm_metadata_problems(result.meta.get("llm") or {})
    problems.extend(validate_abstention_metadata(result.meta))

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
    if reason == "calibration_instrument":
        # Decision input: the archive is complete and readable, it just cannot be cited as
        # QA-set evidence. Exiting 0 keeps a baseline sweep from looking like a failed run.
        print(
            "DECISION INPUT: calibration instrument; not gate-reportable",
            file=sys.stderr,
        )
        return 0
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
        "--qa-role",
        choices=("report", "calibration"),
        default="report",
        help=(
            "report: the frozen QA set, gate-reportable (default); calibration: the 30-item "
            "abstention instrument, decision input only and never gate-reportable"
        ),
    )
    parser.add_argument(
        "--qa-reference",
        default="eval/qa_set.jsonl",
        help="QA set the calibration instrument must stay disjoint from (calibration role only)",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="enforce the PR #48 GPU-first policy; CPU runs are then non-reportable",
    )
    parser.add_argument(
        "--abstention-calibration",
        default=None,
        help="archived calibration.json authorising ACGN_ANSWER_MIN_RETRIEVAL_SCORE; live runs only",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate QA, build records, identity, GPU, threshold and binding; write no archive",
    )
    parser.add_argument("--expect-docs", type=int, default=None)
    parser.add_argument("--expect-chunks", type=int, default=None)
    parser.add_argument("--expect-qa-fingerprint", default=None)
    args = parser.parse_args(argv)

    qa_path = Path(args.qa)
    if not qa_path.exists():
        print(f"NOT FOUND: {qa_path}", file=sys.stderr)
        return 2
    items = load_jsonl(qa_path)
    if not items:
        print(f"EMPTY: {qa_path}", file=sys.stderr)
        return 2

    if args.qa_role == "calibration":
        if args.allow_invalid_qa:
            print(
                "--qa-role calibration cannot be combined with --allow-invalid-qa",
                file=sys.stderr,
            )
            return 1
        reference_path = Path(args.qa_reference)
        if not reference_path.exists():
            print(f"NOT FOUND: {reference_path}", file=sys.stderr)
            return 2
        problems = _calibration_set_problems(items, load_jsonl(reference_path))
        if problems:
            print(f"INVALID CALIBRATION SET ({len(problems)} problem(s)):", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1
        qa_validated = True
    else:
        if not args.allow_invalid_qa:
            problems = validate(items)
            if problems:
                print(f"INVALID QA SET ({len(problems)} problem(s)):", file=sys.stderr)
                for problem in problems:
                    print(f"  - {problem}", file=sys.stderr)
                print("Use --allow-invalid-qa only for development runs.", file=sys.stderr)
                return 1
        qa_validated = not args.allow_invalid_qa

    settings = load_settings()
    try:
        abstention = _abstention_block(args.abstention_calibration, settings, args.answerer)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"ABSTENTION CONFIG: {exc}", file=sys.stderr)
        return 2
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
        qa_validated=qa_validated,
        qa_source="cli",
        qa_role=args.qa_role,
        abstention=abstention,
    )
    if args.preflight_only:
        problems: list[str] = []
        try:
            validate_run_spec(spec)
        except ValueError as exc:
            problems.append(str(exc))
        problems.extend(
            _preflight_device_problems(
                build_identity,
                determinism,
                require_gpu=args.require_gpu,
            )
        )
        qa_fingerprint = qa_inputs_fingerprint(items)
        problems.extend(
            _preflight_expectation_problems(
                counts=counts,
                qa_fingerprint=qa_fingerprint,
                expect_docs=args.expect_docs,
                expect_chunks=args.expect_chunks,
                expect_qa_fingerprint=args.expect_qa_fingerprint,
            )
        )
        if problems:
            print("PREFLIGHT FAILED:", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1
        print("PREFLIGHT OK")
        print(f"  snapshot:   {args.snapshot}")
        print(f"  config:     {args.config}")
        print(f"  collection: {collection}")
        print(f"  counts:     {dict(counts)}")
        print(f"  identity:   {build_identity.get('identity_slug')}")
        print(f"  device:     {build_identity.get('device')}")
        print(f"  gpu_model:  {determinism.get('gpu_model')}")
        print(f"  qa:         {len(items)} items")
        print(f"  qa_fp:      {qa_fingerprint}")
        print(f"  threshold:  {abstention['threshold'] if abstention else None}")
        return 0
    try:
        result = run(
            spec,
            store=store,
            embedder=embedder,
            answerer=answerer,
            qa_items=items,
            evidence_resolver=evidence_resolver,
            snapshot_member_urls=member_urls,
        )
    except ValueError as exc:
        print(f"RUN REFUSED: {exc}", file=sys.stderr)
        return 1
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
