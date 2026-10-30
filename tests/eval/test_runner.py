"""Tests for the evaluation runner's archive and metric wiring."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from acgn_rag.core.eval.runner import NullAnswerer, RunSpec, run
from acgn_rag.core.store.chroma_store import ScoredPayload
from acgn_rag.core.store.payload import ChunkPayload, iso_to_epoch

SNAPSHOT = "20261028abcdef"
GOLD = "https://example.com/news/a"
DAY = 86_400


def _hit() -> ScoredPayload:
    return ScoredPayload(
        payload=ChunkPayload(
            chunk_id="c1",
            doc_id="d1",
            snapshot_id=SNAPSHOT,
            title="Title",
            url=GOLD,
            publisher="Publisher",
            source_type="official",
            published_ts=iso_to_epoch("2026-10-28"),
            lang="en",
            body="The body.",
            media_ids=(7,),
        ),
        score=0.9,
    )


@dataclass
class FakeStore:
    hits: list[ScoredPayload]
    calls: list[tuple[int, dict[str, Any] | None]] = field(default_factory=list)

    def query(
        self,
        embedding: Any,
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[ScoredPayload]:
        self.calls.append((top_k, where))
        return self.hits[:top_k]


class FakeEmbedder:
    def encode_queries(self, texts: Any) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]


def _qa_item(
    qa_id: str = "A-001",
    *,
    lang: str = "en",
    answerable: bool = True,
    urls: tuple[str, ...] = (GOLD,),
    window: tuple[str, str] = ("2023-10-28", "2026-10-28"),
) -> dict[str, Any]:
    return {
        "qa_id": qa_id,
        "type": "A" if answerable else "D",
        "lang": lang,
        "answerable": answerable,
        "question": "When does it air?",
        "gold_answer": "2026-10-05",
        "gold_evidence": {"urls": list(urls), "publisher": "Publisher"},
        "key_entities": ["2026-10-05"],
        "window": list(window),
        "annotator": "D",
        "verified_by": "F",
    }


def _spec(tmp_path: Path, **overrides: object) -> RunSpec:
    values: dict[str, object] = {
        "snapshot_id": SNAPSHOT,
        "window_start": "2023-10-28",
        "window_end": "2026-10-28",
        "config": "A",
        "out_root": tmp_path / "runs",
        "qa_path": tmp_path / "qa.jsonl",
        "run_id": "20261030-A-20261028abcdef",
        "run_date": "20261030",
        "git_sha": "test-sha",
        "embed_identity": {"model": "fake", "device": "cpu"},
        "llm": {"provider": "none", "model": "none"},
    }
    values.update(overrides)
    return RunSpec(**values)  # type: ignore[arg-type]


def test_run_writes_complete_archive(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    result = run(
        spec,
        store=FakeStore([_hit()]),
        embedder=FakeEmbedder(),
        answerer=NullAnswerer(),
        qa_items=[_qa_item()],
    )

    assert result.run_id == spec.run_id
    for name in (
        "meta.json",
        "answers.jsonl",
        "labels.jsonl",
        "metrics.json",
        "resolved_evidence.json",
        "disputes.md",
    ):
        assert (result.out_dir / name).exists(), name

    meta = json.loads((result.out_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["snapshot_id"] == SNAPSHOT
    assert meta["chunk_config"] == "A"
    assert meta["retrieval"]["mode"] == "vector"
    assert meta["git_sha"] == "test-sha"
    assert meta["qa"]["count"] == 1

    assert result.metrics["retrieval_hit_rate"]["all"] == 1.0
    assert result.metrics["retrieval_hit_rate"]["en"] == 1.0
    assert result.metrics["retrieval_hit_rate"]["zh"] is None


def test_null_answerer_marks_not_run(tmp_path: Path) -> None:
    result = run(
        _spec(tmp_path),
        store=FakeStore([_hit()]),
        embedder=FakeEmbedder(),
        answerer=NullAnswerer(),
        qa_items=[_qa_item()],
    )

    answer = json.loads((result.out_dir / "answers.jsonl").read_text(encoding="utf-8").strip())
    assert answer["status"] == "not_run"
    assert answer["answer"] is None
    assert answer["abstained"] is None
    assert result.metrics["answer_correctness"]["labeled"] == 0
    assert result.metrics["abstention_accuracy"]["false_refusal_rate"] is None


def test_run_uses_item_window_as_filters(tmp_path: Path) -> None:
    store = FakeStore([_hit()])
    run(
        _spec(tmp_path),
        store=store,
        embedder=FakeEmbedder(),
        qa_items=[_qa_item(window=("2023-10-28", "2026-10-28"))],
    )

    assert store.calls == [
        (
            5,
            {
                "$and": [
                    {"published_ts": {"$gte": iso_to_epoch("2023-10-28")}},
                    {"published_ts": {"$lt": iso_to_epoch("2026-10-28") + DAY}},
                ]
            },
        )
    ]


def test_run_id_uses_a_prime_token(tmp_path: Path) -> None:
    result = run(
        _spec(tmp_path, config="A'", run_id=None),
        store=FakeStore([_hit()]),
        embedder=FakeEmbedder(),
        qa_items=[_qa_item()],
    )
    assert result.run_id == "20261030-A-prime-20261028abcdef"


def test_run_requires_snapshot_and_window(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="snapshot_id"):
        run(
            _spec(tmp_path, snapshot_id=""),
            store=FakeStore([_hit()]),
            embedder=FakeEmbedder(),
            qa_items=[_qa_item()],
        )
    with pytest.raises(ValueError, match="window_start"):
        run(
            _spec(tmp_path, window_start=""),
            store=FakeStore([_hit()]),
            embedder=FakeEmbedder(),
            qa_items=[_qa_item()],
        )


def test_run_rejects_existing_directory(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    run(
        spec,
        store=FakeStore([_hit()]),
        embedder=FakeEmbedder(),
        qa_items=[_qa_item()],
    )
    with pytest.raises(FileExistsError):
        run(
            spec,
            store=FakeStore([_hit()]),
            embedder=FakeEmbedder(),
            qa_items=[_qa_item()],
        )


def test_rerun_metrics_are_identical(tmp_path: Path) -> None:
    first = run(
        _spec(tmp_path, run_id="r1"),
        store=FakeStore([_hit()]),
        embedder=FakeEmbedder(),
        qa_items=[_qa_item()],
    )
    second = run(
        _spec(tmp_path, run_id="r2"),
        store=FakeStore([_hit()]),
        embedder=FakeEmbedder(),
        qa_items=[_qa_item()],
    )
    assert first.metrics == second.metrics


def test_evidence_resolver_marks_gold_unavailable(tmp_path: Path) -> None:
    result = run(
        _spec(tmp_path),
        store=FakeStore([_hit()]),
        embedder=FakeEmbedder(),
        qa_items=[_qa_item()],
        evidence_resolver=lambda urls: {GOLD: []},
    )
    assert result.metrics["excluded"]["gold_unavailable"] == 1
    assert result.metrics["retrieval_hit_rate"]["all"] is None
    resolved = json.loads((result.out_dir / "resolved_evidence.json").read_text(encoding="utf-8"))
    assert resolved["status"] == "resolved"
    assert resolved["unresolved"] == [GOLD]
