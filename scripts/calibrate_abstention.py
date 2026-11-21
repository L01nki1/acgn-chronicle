"""Calibrate the pre-generation abstention threshold on a frozen evaluation build.

Owner: E (WP5 runner) | Spec: `docs/EVAL_PROTOCOL.md` §6/§7, `docs/RUNBOOK.md` §3.7

The calibration set is separate from the QA set (`eval/abstain_calibration.jsonl`, 20
answerable + 10 unanswerable). For every item this script records the best retrieval score
(``None`` when nothing is returned), calls B's `choose_threshold()`, and archives the selected
operating point together with the identity it was calibrated at:

    eval/runs/<run_id>/calibration.json

A run may apply that threshold only when its own snapshot / chunking / retrieval / encoder
identity equals the identity recorded here; `scripts/run_eval.py --abstention-calibration`
enforces that. In-sample calibration rates are calibration evidence, not QA-set metrics.

    python scripts/calibrate_abstention.py --snapshot 20261028abcdef --config A
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acgn_rag.config import load_settings
from acgn_rag.core.embed import Embedder
from acgn_rag.core.eval import qa
from acgn_rag.core.eval.runner import (
    bind_existing_collection,
    load_build_records,
    run_identity,
    verify_build_identity,
)
from acgn_rag.core.ground.calibration import CalibrationSample, choose_threshold
from acgn_rag.core.ground.generate import answerer_version
from acgn_rag.core.ports import RetrievalFilters, RetrievalQuery
from acgn_rag.core.retrieve.official_first import search_with_official_bonus
from acgn_rag.core.retrieve.vector import search
from acgn_rag.core.store import ChromaStore

CONFIG_CHOICES = ("A", "A'", "A0", "B")
COLLECTION_LABELS = {"A'": "A-prime"}
_RUN_ID = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")


def _filters(row: Mapping[str, Any]) -> RetrievalFilters:
    """The item's frozen window, matching what `runner.run()` applies to a QA item."""
    window = row.get("window")
    if not isinstance(window, (list, tuple)) or len(window) != 2:
        raise ValueError(f"{row.get('qa_id', '?')}: window must be [start, end]")
    return RetrievalFilters(since=str(window[0]), until=str(window[1]))


def _set_problems(rows: Sequence[Mapping[str, Any]], *, allow_any_set: bool) -> list[str]:
    """Validate the calibration set's shape before it can author a gate threshold."""
    problems: list[str] = []
    seen: set[str] = set()
    for row in rows:
        qa_id = row.get("qa_id")
        if not isinstance(qa_id, str) or not qa_id.strip():
            problems.append("an item has no qa_id")
        elif qa_id in seen:
            problems.append(f"{qa_id}: duplicate qa_id")
        else:
            seen.add(qa_id)
        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            problems.append(f"{qa_id or '?'}: question is missing")
        if not isinstance(row.get("answerable"), bool):
            problems.append(f"{qa_id or '?'}: answerable must be a bool")
        try:
            _filters(row)
        except ValueError as exc:
            problems.append(str(exc))
    answerable = sum(1 for row in rows if row.get("answerable") is True)
    unanswerable = sum(1 for row in rows if row.get("answerable") is False)
    if not answerable or not unanswerable:
        problems.append("the set needs at least one answerable and one unanswerable item")
    if not allow_any_set:
        if len(rows) != 30:
            problems.append(f"gate calibration set must have 30 items, got {len(rows)}")
        if answerable != 20:
            problems.append(f"gate calibration set must have 20 answerable items, got {answerable}")
        if unanswerable != 10:
            problems.append(
                f"gate calibration set must have 10 unanswerable items, got {unanswerable}"
            )
    return problems


def _samples(
    rows: Sequence[Mapping[str, Any]], scores: Sequence[float | None]
) -> list[CalibrationSample]:
    """Pair every labelled calibration row with the retrieval score seen for it."""
    if len(rows) != len(scores):
        raise ValueError(f"{len(rows)} rows but {len(scores)} scores")
    samples: list[CalibrationSample] = []
    for row, score in zip(rows, scores, strict=True):
        answerable = row.get("answerable")
        if not isinstance(answerable, bool):
            raise ValueError(f"{row.get('qa_id', '?')}: answerable must be a bool")
        if score is not None and not math.isfinite(float(score)):
            raise ValueError(f"{row.get('qa_id', '?')}: non-finite retrieval score")
        samples.append(CalibrationSample(retrieval_top_score=score, answerable=answerable))
    return samples


def _artifact(
    *,
    run_id: str,
    snapshot_id: str,
    config: str,
    identity: Mapping[str, Any],
    result: Any,
) -> dict[str, Any]:
    """Build the archived calibration record that `meta.abstention.source` will name."""
    if not identity.get("encoder_identity_slug"):
        raise ValueError("calibration identity has no encoder_identity_slug")
    metadata = result.to_metadata()
    return {
        "run_id": run_id,
        "snapshot_id": snapshot_id,
        "config": config,
        "threshold": metadata["threshold"],
        "calibration": metadata,
        "calibration_identity": dict(identity),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _run_id(config: str, slug: str) -> str:
    token = COLLECTION_LABELS.get(config, config)
    return f"{date.today():%Y%m%d}-cal-{token}-{slug}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate the abstention threshold")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--config", choices=CONFIG_CHOICES, default="A")
    parser.add_argument("--qa", default="eval/abstain_calibration.jsonl")
    parser.add_argument("--out", default="eval/runs")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--strategy",
        choices=("balanced_accuracy", "target_missed_refusal"),
        default="balanced_accuracy",
    )
    parser.add_argument("--target-missed-refusal-rate", type=float, default=0.2)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--official-bonus",
        type=float,
        default=1.0,
        help="official-chunk score multiplier used when scoring (default 1.0 = single pool); "
        "recorded in the calibration identity so a run on the same lane can use it",
    )
    parser.add_argument(
        "--translate-queries",
        default="",
        metavar="ZH",
        help="translate questions in this language before scoring, e.g. 'zh' (default: off); "
        "requires --answerer live and is recorded in the calibration identity",
    )
    parser.add_argument(
        "--allow-any-set",
        action="store_true",
        help="exploration only: accept a calibration set other than 30 items (20/10); not gate evidence",
    )
    args = parser.parse_args(argv)

    if args.top_k <= 0:
        print("--top-k must be positive", file=sys.stderr)
        return 2
    rows = qa.load_jsonl(Path(args.qa))
    if not rows:
        print(f"EMPTY: {args.qa}", file=sys.stderr)
        return 2
    problems = _set_problems(rows, allow_any_set=args.allow_any_set)
    if problems:
        print("INVALID CALIBRATION SET:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2

    settings = load_settings()
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

    embed_config = replace(
        settings.embed,
        model_name=str(build_identity["model"]),
        device=str(build_identity["device"]),
        batch_size=int(build_identity["batch_size"]),
    )
    embedder = Embedder(embed_config)
    problems = verify_build_identity(build_identity, embedder.identity().to_metadata())
    if problems:
        print("BUILD IDENTITY MISMATCH:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    store = ChromaStore(settings.paths.index / "chroma", settings.store)
    try:
        bind_existing_collection(
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

    scores: list[float | None] = []
    for row in rows:
        query = RetrievalQuery(
            text=str(row["question"]),
            top_k=args.top_k,
            snapshot_id=args.snapshot,
            filters=_filters(row),
        )
        if args.translate_queries:
            raise SystemExit(
                "TRANSLATION UNSUPPORTED: --translate-queries needs an answer-line client and is "
                "not wired into calibration yet; omit it to calibrate the single-pool lane"
            )
        if args.official_bonus == 1.0:
            chunks = search(query, store=store, embedder=embedder)
        else:
            # The bonus reranks official chunks, so the top score that selects the threshold must
            # come from the same reranked pool a live run will compare against.
            chunks, _mix = search_with_official_bonus(
                query,
                store=store,
                embedder=embedder,
                official_weight=args.official_bonus,
            )
        scores.append(max((chunk.score for chunk in chunks), default=None))
    try:
        samples = _samples(rows, scores)
        result = choose_threshold(
            samples,
            strategy=args.strategy,
            target_missed_refusal_rate=args.target_missed_refusal_rate,
        )
    except ValueError as exc:
        print(f"CALIBRATION FAILED: {exc}", file=sys.stderr)
        return 1

    identity = run_identity(
        snapshot_id=args.snapshot,
        config=args.config,
        chunker=chunker,
        top_k=args.top_k,
        over_fetch=settings.store.over_fetch_factor,
        build_identity=build_identity,
        # Recorded from the flags actually used, not hard-coded: a threshold is bound to the
        # lane it was measured on, so a calibration produced for a bonus/translation lane has to
        # name that lane or a live run on it can never pair with the artifact.
        query_transform=f"translate-{args.translate_queries}" if args.translate_queries else "none",
        official_score_weight=args.official_bonus,
        answerer_version=answerer_version(),
    )
    run_id = args.run_id or _run_id(args.config, str(build_identity["identity_slug"]))
    if not _RUN_ID.fullmatch(run_id):
        print(f"invalid --run-id {run_id!r}", file=sys.stderr)
        return 2
    out_dir = Path(args.out) / run_id
    if out_dir.exists():
        print(f"calibration archive already exists: {out_dir}", file=sys.stderr)
        return 2
    artifact = _artifact(
        run_id=run_id,
        snapshot_id=args.snapshot,
        config=args.config,
        identity=identity,
        result=result,
    )
    answerable_count = sum(1 for row in rows if row.get("answerable") is True)
    artifact["set"] = {
        "total": len(rows),
        "answerable": answerable_count,
        "unanswerable": len(rows) - answerable_count,
        "gate_set": not args.allow_any_set,
    }
    out_dir.mkdir(parents=True)
    (out_dir / "calibration.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "path": str(out_dir / "calibration.json"),
                "threshold": artifact["threshold"],
                "strategy": result.strategy,
                "counts": counts,
                "collection": collection,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
