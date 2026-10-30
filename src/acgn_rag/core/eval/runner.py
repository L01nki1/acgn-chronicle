"""WP5 evaluation runner: one run bound to one snapshot and one chunk config.

Owner: E | Spec: `docs/EVAL_PROTOCOL.md` §1/§7/§8, issue #7

The runner is deliberately injectable: `search` consumes the merged store/embedder, and the
answerer is a protocol. Until B's `/ask` lands, `NullAnswerer` records every answer as
`not_run`; the retrieval half (hit rate, evidence recall, MRR) is still measured and archived.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from acgn_rag.core.eval import metrics, qa
from acgn_rag.core.ports import (
    GroundedAnswer,
    RetrievalFilters,
    RetrievalQuery,
    RetrievedChunk,
)
from acgn_rag.core.retrieve.vector import search

CONFIG_TOKENS = {"A'": "A-prime"}


class Answerer(Protocol):
    """Generation adapter; B's `/ask` or a test double."""

    def answer(
        self,
        query: RetrievalQuery,
        chunks: Sequence[RetrievedChunk],
    ) -> GroundedAnswer | None: ...


class NullAnswerer:
    """Records that generation has not run yet; retrieval metrics stay valid."""

    def answer(
        self,
        query: RetrievalQuery,
        chunks: Sequence[RetrievedChunk],
    ) -> GroundedAnswer | None:
        return None


@dataclass(frozen=True)
class RunSpec:
    snapshot_id: str
    window_start: str
    window_end: str
    config: str
    out_root: Path = Path("eval/runs")
    qa_path: Path = Path("eval/qa_set.jsonl")
    top_k: int = 5
    over_fetch: int = 5
    run_id: str | None = None
    run_date: str | None = None
    git_sha: str = "unknown"
    embed_identity: Mapping[str, Any] = field(default_factory=dict)
    llm: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunResult:
    run_id: str
    out_dir: Path
    meta: dict[str, Any]
    metrics: dict[str, Any]


def _config_token(config: str) -> str:
    return CONFIG_TOKENS.get(config, config)


def _run_id(spec: RunSpec) -> str:
    if spec.run_id:
        return spec.run_id
    day = spec.run_date or datetime.now(UTC).strftime("%Y%m%d")
    return f"{day}-{_config_token(spec.config)}-{spec.snapshot_id}"


def _validate_spec(spec: RunSpec) -> None:
    missing = [
        name
        for name in ("snapshot_id", "window_start", "window_end", "config")
        if not getattr(spec, name)
    ]
    if missing:
        raise ValueError(f"run spec is missing {', '.join(missing)}")


def _load_items(
    spec: RunSpec, qa_items: Sequence[Mapping[str, Any]] | None
) -> list[dict[str, Any]]:
    if qa_items is None:
        items = qa.load_jsonl(spec.qa_path)
        problems = qa.validate(items)
        if problems:
            raise ValueError(
                "invalid QA set:\n" + "\n".join(f"- {problem}" for problem in problems)
            )
    else:
        items = [dict(item) for item in qa_items]
    if not items:
        raise ValueError("QA set is empty")
    return items


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _item_urls(item: Mapping[str, Any]) -> tuple[str, ...]:
    evidence = item.get("gold_evidence") or {}
    urls = evidence.get("urls") or []
    return tuple(str(url) for url in urls)


def _answer_record(
    item: Mapping[str, Any],
    chunks: Sequence[RetrievedChunk],
    answer: GroundedAnswer | None,
) -> dict[str, Any]:
    return {
        "qa_id": str(item["qa_id"]),
        "question": str(item["question"]),
        "retrieved_chunk_ids": [chunk.chunk_id for chunk in chunks],
        "retrieved_urls": [chunk.citation.url for chunk in chunks],
        "answer": answer.answer if answer else None,
        "citations": [citation.model_dump() for citation in answer.citations] if answer else [],
        "cited_chunk_ids": list(answer.cited_chunk_ids) if answer else [],
        "abstained": answer.abstained if answer else None,
        "status": "answered" if answer else "not_run",
    }


def run(
    spec: RunSpec,
    *,
    store: Any,
    embedder: Any,
    answerer: Answerer | None = None,
    qa_items: Sequence[Mapping[str, Any]] | None = None,
    evidence_resolver: Callable[[Sequence[str]], Mapping[str, Sequence[str]]] | None = None,
) -> RunResult:
    """Execute one run and write the archive under `out_root/<run_id>/`."""
    _validate_spec(spec)
    items = _load_items(spec, qa_items)
    run_id = _run_id(spec)
    out_dir = spec.out_root / run_id
    if out_dir.exists():
        raise FileExistsError(f"run directory already exists: {out_dir}")

    count = getattr(store, "count", None)
    if callable(count) and count() == 0:
        raise ValueError(
            f"index for snapshot {spec.snapshot_id!r} / config {spec.config!r} is empty"
        )

    resolved: dict[str, list[str]] = {}
    all_gold = sorted({url for item in items for url in _item_urls(item)})
    if evidence_resolver is not None:
        resolved = {
            str(url): [str(chunk_id) for chunk_id in chunk_ids]
            for url, chunk_ids in evidence_resolver(all_gold).items()
        }
    unresolved = [url for url in all_gold if not resolved.get(url)]

    cases: list[metrics.CaseResult] = []
    answers: list[dict[str, Any]] = []
    for item in items:
        gold_urls = _item_urls(item)
        window = item["window"]
        query = RetrievalQuery(
            text=str(item["question"]),
            top_k=spec.top_k,
            snapshot_id=spec.snapshot_id,
            filters=RetrievalFilters(since=str(window[0]), until=str(window[1])),
        )
        chunks = search(query, store=store, embedder=embedder)
        answer = answerer.answer(query, chunks) if answerer is not None else None
        answers.append(_answer_record(item, chunks, answer))
        cases.append(
            metrics.CaseResult(
                qa_id=str(item["qa_id"]),
                lang=str(item["lang"]),
                answerable=bool(item["answerable"]),
                gold_urls=gold_urls,
                retrieved_urls=tuple(chunk.citation.url for chunk in chunks),
                retrieved_chunk_ids=tuple(chunk.chunk_id for chunk in chunks),
                cited_chunk_ids=tuple(answer.cited_chunk_ids) if answer else (),
                gold_unavailable=bool(gold_urls)
                and evidence_resolver is not None
                and all(not resolved.get(url) for url in gold_urls),
                answer_label=None,
                abstained=answer.abstained if answer else None,
            )
        )

    metrics_payload = metrics.aggregate(cases)
    meta = {
        "run_id": run_id,
        "snapshot_id": spec.snapshot_id,
        "window_start": spec.window_start,
        "window_end": spec.window_end,
        "chunk_config": spec.config,
        "retrieval": {
            "mode": "vector",
            "top_k": spec.top_k,
            "over_fetch": spec.over_fetch,
        },
        "embed": dict(spec.embed_identity),
        "llm": dict(spec.llm),
        "git_sha": spec.git_sha,
        "qa": {"path": str(spec.qa_path), "count": len(items)},
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "metrics": metrics_payload,
    }

    out_dir.mkdir(parents=True)
    _write_json(out_dir / "meta.json", meta)
    _write_json(out_dir / "metrics.json", metrics_payload)
    _write_jsonl(out_dir / "answers.jsonl", answers)
    _write_jsonl(out_dir / "labels.jsonl", [])
    _write_json(
        out_dir / "resolved_evidence.json",
        {
            "snapshot_id": spec.snapshot_id,
            "config": spec.config,
            "resolved": resolved,
            "unresolved": unresolved,
            "status": "resolved" if evidence_resolver is not None else "not_configured",
        },
    )
    (out_dir / "disputes.md").write_text(
        "# Disputes\n\nNo adjudications recorded for this run.\n",
        encoding="utf-8",
    )
    return RunResult(run_id=run_id, out_dir=out_dir, meta=meta, metrics=metrics_payload)


__all__ = [
    "CONFIG_TOKENS",
    "Answerer",
    "NullAnswerer",
    "RunResult",
    "RunSpec",
    "run",
]
