"""Tests for the evaluation runner's archive and metric wiring."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pytest

from acgn_rag.core.eval import qa
from acgn_rag.core.eval.runner import (
    CHUNKER_FIELDS,
    VECTOR_IDENTITY_KEYS,
    NullAnswerer,
    RunSpec,
    _latency_summary,
    _run_id,
    bind_existing_collection,
    build_evidence_resolver,
    compare_tier1_runs,
    load_build_records,
    load_snapshot_member_urls,
    resolve_window,
    run,
    verify_build_identity,
)
from acgn_rag.core.ports import GroundedAnswer
from acgn_rag.core.store.chroma_store import ScoredPayload
from acgn_rag.core.store.payload import ChunkPayload, iso_to_epoch

SNAPSHOT = "20261028abcdef"
GOLD = "https://example.com/news/a"
OTHER = "https://example.com/news/b"
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


@dataclass(frozen=True)
class NamedCollection:
    name: str


class BindStore:
    def __init__(self, bound_name: str, error: Exception | None = None) -> None:
        self._bound_name = bound_name
        self._error = error
        self.calls: list[tuple[str, str]] = []
        self.bound = False

    def bind_existing(self, snapshot_id: str, chunk_config: str) -> NamedCollection:
        self.calls.append((snapshot_id, chunk_config))
        if self._error is not None:
            raise self._error
        self.bound = True
        return NamedCollection(self._bound_name)


class BuildOnlyStore:
    """A store with only the build-side bind; a read must not fall back to it."""

    def __init__(self) -> None:
        self.bind_called = False

    def bind(self, snapshot_id: str, chunk_config: str) -> NamedCollection:
        self.bind_called = True
        return NamedCollection("snap_legacy_A")


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
        "window_verified": True,
        "git_sha": "test-sha",
        "build_identity": {
            "model": "fake",
            "backend": "torch",
            "device": "cuda",
            "precision": "fp32",
            "batch_size": 32,
            "identity_slug": "fake-torch-cuda-fp32-b32-12345678",
            "lib_torch": "2.15.0",
        },
        "collection": "snap_20261028abc_A",
        "counts": {"docs": 6402, "chunks": 6500, "stored": 9120},
        "chunker": {
            "version": "composed-input-v2",
            "header_version": "h1",
            "config": "A",
            "max_tokens": 512,
            "overlap": 64,
            "max_seq_length": 512,
        },
        "llm": {"provider": "none", "model": "none"},
        "determinism": {"torch_seed": 0, "use_deterministic_algorithms": True},
        "require_gpu": False,
        # Test fixtures explicitly attest QA validity so the reportability-composition
        # tests can isolate resolver/window/GPU/metrics. The production default is False.
        "qa_validated": True,
        "qa_source": "caller",
    }
    values.update(overrides)
    return RunSpec(**values)  # type: ignore[arg-type]


def _build_identity(model: str = "fake") -> dict[str, Any]:
    return {
        "model": model,
        "backend": "torch",
        "device": "cpu",
        "precision": "fp32",
        "batch_size": 32,
        "identity_slug": "fake-torch-cpu-fp32-b32-12345678",
        "lib_torch": "2.15.0",
    }


def _write_build_records(
    root: Path,
    *,
    snapshot_id: str = SNAPSHOT,
    config: str = "A",
    label: str = "A",
    collection: str = "snap_20261028abc_A",
    identity: dict[str, Any] | None = None,
    build_identity: dict[str, Any] | None = None,
    chunks: int = 1,
    stored: int = 1,
    docs: int = 1,
    chunker: dict[str, Any] | None = None,
) -> Path:
    base = root / snapshot_id / label
    base.mkdir(parents=True)
    identity = dict(identity or _build_identity())
    build_identity = dict(build_identity or identity)
    chunker = dict(
        chunker
        or {
            "version": "composed-input-v2",
            "header_version": "h1",
            "config": config,
            "max_tokens": 512,
            "overlap": 64,
            "max_seq_length": 512,
        }
    )
    common = {
        "snapshot_id": snapshot_id,
        "config": config,
        "label": label,
        "collection": collection,
    }
    (base / "identity.json").write_text(
        json.dumps({**common, "identity": identity, "chunker": chunker}),
        encoding="utf-8",
    )
    (base / "build.json").write_text(
        json.dumps(
            {
                **common,
                "docs": docs,
                "chunks": chunks,
                "stored": stored,
                "identity": build_identity,
                "chunker": chunker,
            }
        ),
        encoding="utf-8",
    )
    return base


def test_run_writes_complete_archive(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    result = run(
        spec,
        store=FakeStore([_hit()]),
        embedder=FakeEmbedder(),
        answerer=NullAnswerer(),
        qa_items=[_qa_item()],
        evidence_resolver=lambda urls: {url: ["doc::A::1"] for url in urls},
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
    assert meta["qa"]["validated"] is True
    assert meta["qa"]["validation"] == "validated"
    assert meta["qa"]["source"] == "caller"
    assert meta["build_identity"]["model"] == "fake"
    assert meta["embed"] == {
        "model": "fake",
        "backend": "torch",
        "device": "cuda",
        "precision": "fp32",
        "library_versions": {"torch": "2.15.0"},
        "batch": 32,
    }
    assert meta["determinism"] == {"torch_seed": 0, "use_deterministic_algorithms": True}
    assert meta["non_reportable_reason"] is None
    assert meta["collection"] == "snap_20261028abc_A"
    assert meta["chunker"]["max_tokens"] == 512
    assert meta["reportable"] is True

    assert result.metrics["retrieval"]["hit"]["all"] == {"value": 1.0, "n": 1}
    assert result.metrics["retrieval"]["hit"]["en"] == {"value": 1.0, "n": 1}
    assert result.metrics["retrieval"]["hit"]["zh"] == {"value": None, "n": 0}
    assert result.metrics["counts"]["deduped_count"] == 6402
    assert result.metrics["counts"]["indexed_chunks"] == 9120


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
    assert answer["provider_completion"] is False
    assert result.metrics["correctness"]["rates"]["all"] == {"value": None, "n": 0}
    assert result.metrics["abstention"]["false_refusal_rate"] == {"value": None, "n": 0}
    assert result.metrics["provisional"] is True


@dataclass(frozen=True)
class FakeAnswerDiagnostics:
    max_tokens: int
    finish_reason: str | None = None
    usage: dict[str, int] | None = None
    dropped_claims: int = 0
    abstain_code: str | None = None
    provider_completion: bool = True


@dataclass(frozen=True)
class FakeAnswerRun:
    answer: GroundedAnswer
    diagnostics: FakeAnswerDiagnostics


class DiagnosticsAnswerer:
    def answer(self, query: Any, chunks: Any) -> GroundedAnswer:
        return self.answer_with_diagnostics(query, chunks).answer

    def answer_with_diagnostics(self, query: Any, chunks: Any) -> FakeAnswerRun:
        chunk = chunks[0]
        answer = GroundedAnswer(
            answer="The answer. [1]",
            citations=[chunk.citation],
            cited_chunk_ids=[chunk.chunk_id],
            abstained=False,
            abstain_reason=None,
        )
        return FakeAnswerRun(
            answer=answer,
            diagnostics=FakeAnswerDiagnostics(
                max_tokens=2048,
                finish_reason="stop",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                dropped_claims=2,
                abstain_code=None,
            ),
        )


class MetadataFreeCompletionAnswerer(DiagnosticsAnswerer):
    def answer_with_diagnostics(self, query: Any, chunks: Any) -> FakeAnswerRun:
        run = super().answer_with_diagnostics(query, chunks)
        return FakeAnswerRun(
            answer=run.answer,
            diagnostics=FakeAnswerDiagnostics(max_tokens=1, provider_completion=True),
        )


def test_diagnostics_answerer_is_archived(tmp_path: Path) -> None:
    result = run(
        _spec(tmp_path),
        store=FakeStore([_hit()]),
        embedder=FakeEmbedder(),
        answerer=DiagnosticsAnswerer(),
        qa_items=[_qa_item()],
    )

    answer = json.loads((result.out_dir / "answers.jsonl").read_text(encoding="utf-8").strip())
    assert answer["status"] == "answered"
    assert answer["provider_completion"] is True
    assert answer["diagnostics"] == {
        "max_tokens": 2048,
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "dropped_claims": 2,
        "abstain_code": None,
        "provider_completion": True,
    }


def test_completion_without_optional_metadata_is_archived(tmp_path: Path) -> None:
    result = run(
        _spec(tmp_path),
        store=FakeStore([_hit()]),
        embedder=FakeEmbedder(),
        answerer=MetadataFreeCompletionAnswerer(),
        qa_items=[_qa_item()],
    )

    answer = json.loads((result.out_dir / "answers.jsonl").read_text(encoding="utf-8").strip())
    assert answer["provider_completion"] is True
    assert answer["diagnostics"]["finish_reason"] is None
    assert answer["diagnostics"]["usage"] is None


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
    chunker = {
        "version": "composed-input-v2",
        "header_version": "h1",
        "config": "A'",
        "max_tokens": 512,
        "overlap": 64,
        "max_seq_length": 512,
    }
    result = run(
        _spec(tmp_path, config="A'", run_id=None, chunker=chunker),
        store=FakeStore([_hit()]),
        embedder=FakeEmbedder(),
        qa_items=[_qa_item()],
    )
    assert result.run_id == ("20261030-A-prime-20261028abcdef-fake-torch-cuda-fp32-b32-12345678")


def test_run_id_is_stable_for_same_identity(tmp_path: Path) -> None:
    spec = _spec(tmp_path, run_id=None)
    assert _run_id(spec) == _run_id(spec)


def test_run_id_changes_with_identity_slug(tmp_path: Path) -> None:
    base = _spec(tmp_path, run_id=None)
    identity = dict(base.build_identity)
    identity["identity_slug"] = "fake-torch-cpu-fp32-12345678"
    other = _spec(tmp_path, run_id=None, build_identity=identity)

    assert _run_id(base) != _run_id(other)
    assert _run_id(other).endswith("fake-torch-cpu-fp32-12345678")


def test_run_id_changes_with_config(tmp_path: Path) -> None:
    assert _run_id(_spec(tmp_path, run_id=None, config="A")) != _run_id(
        _spec(tmp_path, run_id=None, config="A'")
    )


def test_explicit_run_id_overrides_generated(tmp_path: Path) -> None:
    assert _run_id(_spec(tmp_path, run_id="manual-run")) == "manual-run"


def test_run_id_requires_identity_slug(tmp_path: Path) -> None:
    spec = _spec(tmp_path, run_id=None)
    identity = dict(spec.build_identity)
    identity["identity_slug"] = ""
    with pytest.raises(ValueError, match="identity_slug"):
        _run_id(_spec(tmp_path, run_id=None, build_identity=identity))


def test_run_id_rejects_unsafe_segments(tmp_path: Path) -> None:
    for unsafe in ("../escape", "manual.", "manual/"):
        with pytest.raises(ValueError, match="run_id"):
            _run_id(_spec(tmp_path, run_id=unsafe))

    spec = _spec(tmp_path, run_id=None)
    identity = dict(spec.build_identity)
    identity["identity_slug"] = "../escape"
    with pytest.raises(ValueError, match="identity_slug"):
        _run_id(_spec(tmp_path, run_id=None, build_identity=identity))


def test_run_id_matches_meta_identity_slug(tmp_path: Path) -> None:
    result = run(
        _spec(tmp_path, run_id=None),
        store=FakeStore([_hit()]),
        embedder=FakeEmbedder(),
        answerer=NullAnswerer(),
        qa_items=[_qa_item()],
        evidence_resolver=lambda urls: {url: ["doc::A::1"] for url in urls},
    )
    meta = json.loads((result.out_dir / "meta.json").read_text(encoding="utf-8"))

    assert result.run_id.endswith(meta["build_identity"]["identity_slug"])
    assert meta["build_identity"]["identity_slug"] in result.run_id


def test_run_id_ignores_batch_size_change(tmp_path: Path) -> None:
    base = _spec(tmp_path, run_id=None)
    identity = dict(base.build_identity)
    identity["batch_size"] = 64

    assert _run_id(base) == _run_id(_spec(tmp_path, run_id=None, build_identity=identity))


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
    assert result.metrics["counts"]["gold_unavailable"] == 1
    assert result.metrics["retrieval"]["hit"]["all"] == {"value": None, "n": 0}
    resolved = json.loads((result.out_dir / "resolved_evidence.json").read_text(encoding="utf-8"))
    assert resolved["status"] == "resolved"
    assert resolved["unresolved"] == [GOLD]
    assert resolved["collection"] == "snap_20261028abc_A"
    assert resolved["build_identity"]["model"] == "fake"


def test_run_requires_build_identity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="build_identity"):
        run(
            _spec(tmp_path, build_identity={}),
            store=FakeStore([_hit()]),
            embedder=FakeEmbedder(),
            qa_items=[_qa_item()],
        )


def test_load_build_records_reads_both(tmp_path: Path) -> None:
    root = tmp_path / "index"
    _write_build_records(root)
    identity, collection, counts, chunker = load_build_records(
        root, snapshot_id=SNAPSHOT, config="A", label="A"
    )
    assert identity["model"] == "fake"
    assert collection == "snap_20261028abc_A"
    assert counts == {"docs": 1, "chunks": 1, "stored": 1}
    assert chunker["max_tokens"] == 512
    assert chunker["header_version"] == "h1"


def test_load_build_records_rejects_an_incomplete_build(tmp_path: Path) -> None:
    root = tmp_path / "index"
    base = _write_build_records(root)
    (base / "build.incomplete").write_text("interrupted\n", encoding="utf-8")

    with pytest.raises(ValueError, match="incomplete"):
        load_build_records(root, snapshot_id=SNAPSHOT, config="A", label="A")


def test_load_build_records_rechecks_build_state_after_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from acgn_rag.core.eval import runner

    root = tmp_path / "index"
    base = _write_build_records(root)
    read_json = runner._read_json

    def interrupt_during_read(path: Path) -> dict[str, Any]:
        result = read_json(path)
        if path.name == "build.json":
            (base / "build.incomplete").write_text("interrupted\n", encoding="utf-8")
        return result

    monkeypatch.setattr(runner, "_read_json", interrupt_during_read)
    with pytest.raises(ValueError, match="incomplete"):
        load_build_records(root, snapshot_id=SNAPSHOT, config="A", label="A")


def test_load_build_records_requires_build_json(tmp_path: Path) -> None:
    root = tmp_path / "index"
    base = _write_build_records(root)
    (base / "build.json").unlink()
    with pytest.raises(FileNotFoundError):
        load_build_records(root, snapshot_id=SNAPSHOT, config="A", label="A")


def test_load_build_records_rejects_mismatched_identity(tmp_path: Path) -> None:
    root = tmp_path / "index"
    _write_build_records(
        root,
        build_identity={**_build_identity(), "model": "other"},
    )
    with pytest.raises(ValueError, match="encoder identity"):
        load_build_records(root, snapshot_id=SNAPSHOT, config="A", label="A")


def test_load_build_records_rejects_matching_but_incomplete_chunker(
    tmp_path: Path,
) -> None:
    root = tmp_path / "index"
    _write_build_records(root, chunker={"version": "composed-input-v2"})
    with pytest.raises(ValueError, match="chunker identity is missing"):
        load_build_records(root, snapshot_id=SNAPSHOT, config="A", label="A")


def test_load_build_records_rejects_a_chunker_for_another_config(tmp_path: Path) -> None:
    root = tmp_path / "index"
    _write_build_records(
        root,
        chunker={
            "version": "composed-input-v2",
            "header_version": "h1",
            "config": "A0",
            "max_tokens": 512,
            "overlap": 64,
            "max_seq_length": 512,
        },
    )
    with pytest.raises(ValueError, match="!= selected config"):
        load_build_records(root, snapshot_id=SNAPSHOT, config="A", label="A")


def test_load_build_records_requires_identity_chunker(tmp_path: Path) -> None:
    root = tmp_path / "index"
    base = _write_build_records(root)
    payload = json.loads((base / "identity.json").read_text(encoding="utf-8"))
    payload.pop("chunker")
    (base / "identity.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=r"identity\.json: missing or invalid `chunker`"):
        load_build_records(root, snapshot_id=SNAPSHOT, config="A", label="A")


def test_load_build_records_rejects_chunker_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "index"
    base = _write_build_records(root)
    payload = json.loads((base / "identity.json").read_text(encoding="utf-8"))
    payload["chunker"]["max_tokens"] = 256
    (base / "identity.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="chunker identity"):
        load_build_records(root, snapshot_id=SNAPSHOT, config="A", label="A")


def test_load_build_records_requires_chunker(tmp_path: Path) -> None:
    root = tmp_path / "index"
    base = _write_build_records(root)
    payload = json.loads((base / "build.json").read_text(encoding="utf-8"))
    payload.pop("chunker")
    (base / "build.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="missing or invalid `chunker`"):
        load_build_records(root, snapshot_id=SNAPSHOT, config="A", label="A")


def test_load_build_records_rejects_zero_chunks(tmp_path: Path) -> None:
    root = tmp_path / "index"
    _write_build_records(root, chunks=0, stored=0)
    with pytest.raises(ValueError, match="no successful chunks"):
        load_build_records(root, snapshot_id=SNAPSHOT, config="A", label="A")


def test_load_build_records_rejects_wrong_snapshot(tmp_path: Path) -> None:
    root = tmp_path / "index"
    base = _write_build_records(root)
    identity_path = base / "identity.json"
    payload = json.loads(identity_path.read_text(encoding="utf-8"))
    payload["snapshot_id"] = "other"
    identity_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="snapshot_id"):
        load_build_records(root, snapshot_id=SNAPSHOT, config="A", label="A")


def test_verify_build_identity_matches_and_reports_mismatch() -> None:
    build = _build_identity()
    assert verify_build_identity(build, build) == []
    assert verify_build_identity(build, {**build, "device": "cuda"}) == [
        "device: build='cpu', current='cuda'"
    ]


def test_verify_build_identity_checks_library_versions_and_slug() -> None:
    build = _build_identity()
    lib = verify_build_identity(build, {**build, "lib_torch": "9.9"})
    assert lib == ["lib_torch: build='2.15.0', current='9.9'"]
    slug = verify_build_identity(build, {**build, "identity_slug": "other-slug"})
    assert slug == ["identity_slug: build='fake-torch-cpu-fp32-b32-12345678', current='other-slug'"]


def test_verify_build_identity_ignores_batch_size() -> None:
    build = _build_identity()
    assert verify_build_identity(build, {**build, "batch_size": 8}) == []


def test_verify_build_identity_requires_every_library_key() -> None:
    build = _build_identity()
    current = {key: value for key, value in build.items() if key != "lib_torch"}
    assert verify_build_identity(build, current) == ["lib_torch: build='2.15.0', current=None"]


def test_bind_existing_collection_uses_actual_name() -> None:
    store = BindStore("snap_legacy_A")
    bound = bind_existing_collection(
        store,
        snapshot_id=SNAPSHOT,
        chunk_config="A",
        collection="snap_legacy_A",
    )
    assert bound.name == "snap_legacy_A"
    assert store.bound
    assert store.calls == [(SNAPSHOT, "A")]


def test_bind_existing_collection_missing_does_not_bind() -> None:
    store = BindStore("snap_full_A", error=LookupError("no existing collection"))
    with pytest.raises(LookupError):
        bind_existing_collection(
            store,
            snapshot_id=SNAPSHOT,
            chunk_config="A",
            collection="snap_full_A",
        )
    assert not store.bound


def test_bind_existing_collection_rejects_name_mismatch() -> None:
    store = BindStore("snap_full_A")
    with pytest.raises(RuntimeError, match="does not match sidecar"):
        bind_existing_collection(
            store,
            snapshot_id=SNAPSHOT,
            chunk_config="A",
            collection="snap_legacy_A",
        )


def test_bind_existing_collection_propagates_ambiguity() -> None:
    store = BindStore("snap_legacy_A", error=RuntimeError("ambiguous snapshot"))
    with pytest.raises(RuntimeError, match="ambiguous"):
        bind_existing_collection(
            store,
            snapshot_id=SNAPSHOT,
            chunk_config="A",
            collection="snap_legacy_A",
        )


def test_bind_existing_collection_requires_no_create_api() -> None:
    store = BuildOnlyStore()
    with pytest.raises(RuntimeError, match="bind_existing"):
        bind_existing_collection(
            store,
            snapshot_id=SNAPSHOT,
            chunk_config="A",
            collection="snap_legacy_A",
        )
    assert not store.bind_called


def _manifest(window_end: str = "2026-11-05") -> dict[str, Any]:
    return {
        "manifest_version": "1",
        "window_start": "2021-01-01",
        "window_end": window_end,
        "items": [
            {"doc_id": "doc_b", "title": "B"},
            {"doc_id": "doc_a", "title": "A"},
        ],
    }


def _manifest_id(manifest: dict[str, Any]) -> str:
    payload = dict(manifest)
    payload["items"] = sorted(payload["items"], key=lambda row: row["doc_id"])
    data = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return manifest["window_end"].replace("-", "") + "-" + hashlib.sha256(data).hexdigest()[:12]


def test_resolve_window_uses_and_verifies_the_manifest(tmp_path: Path) -> None:
    manifest = _manifest()
    snapshot_id = _manifest_id(manifest)
    snap = tmp_path / "snapshots" / snapshot_id
    snap.mkdir(parents=True)
    (snap / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert resolve_window(snap, snapshot_id, None, None) == (
        "2021-01-01",
        "2026-11-05",
        True,
    )
    assert resolve_window(snap, snapshot_id, "2021-01-01", "2026-11-05") == (
        "2021-01-01",
        "2026-11-05",
        True,
    )


def test_resolve_window_rejects_a_manifest_under_another_id(tmp_path: Path) -> None:
    manifest = _manifest()
    correct = _manifest_id(manifest)
    wrong = "20261105-" + "0" * 12
    assert wrong != correct
    snap = tmp_path / "snapshots" / wrong
    snap.mkdir(parents=True)
    (snap / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="content does not match snapshot"):
        resolve_window(snap, wrong, None, None)


def test_resolve_window_rejects_a_manifest_with_a_different_date(tmp_path: Path) -> None:
    manifest = _manifest()
    snapshot_id = _manifest_id(manifest)
    wrong = "20200101-" + snapshot_id.split("-")[1]
    snap = tmp_path / "snapshots" / wrong
    snap.mkdir(parents=True)
    (snap / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match its date"):
        resolve_window(snap, wrong, None, None)


def test_resolve_window_leaves_an_unhashable_manifest_unverified(tmp_path: Path) -> None:
    snapshot_id = "20261105-" + "a" * 12
    snap = tmp_path / "snapshots" / snapshot_id
    snap.mkdir(parents=True)
    (snap / "manifest.json").write_text(
        json.dumps({"window_start": "2021-01-01", "window_end": "2026-11-05"}),
        encoding="utf-8",
    )
    assert resolve_window(snap, snapshot_id, None, None) == (
        "2021-01-01",
        "2026-11-05",
        False,
    )


def test_resolve_window_rejects_conflicting_explicit_bounds(tmp_path: Path) -> None:
    snap = tmp_path / "snapshots" / SNAPSHOT
    snap.mkdir(parents=True)
    (snap / "manifest.json").write_text(
        json.dumps({"window_start": "2021-01-01", "window_end": "2026-11-05"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="conflicts with manifest"):
        resolve_window(snap, SNAPSHOT, "2023-10-28", None)
    with pytest.raises(ValueError, match="conflicts with manifest"):
        resolve_window(snap, SNAPSHOT, None, "2026-10-28")


def test_resolve_window_requires_both_explicit_bounds_without_manifest(
    tmp_path: Path,
) -> None:
    snap = tmp_path / "snapshots" / SNAPSHOT
    with pytest.raises(ValueError, match="supply both"):
        resolve_window(snap, SNAPSHOT, None, None)
    with pytest.raises(ValueError, match="supply both"):
        resolve_window(snap, SNAPSHOT, "2023-10-28", None)


def test_resolve_window_marks_explicit_bounds_unverified(tmp_path: Path) -> None:
    snap = tmp_path / "snapshots" / SNAPSHOT
    assert resolve_window(snap, SNAPSHOT, "2023-10-28", "2026-11-05") == (
        "2023-10-28",
        "2026-11-05",
        False,
    )


def test_resolve_window_rejects_a_manifest_without_bounds(tmp_path: Path) -> None:
    snap = tmp_path / "snapshots" / SNAPSHOT
    snap.mkdir(parents=True)
    (snap / "manifest.json").write_text(json.dumps({"window": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="no window_start/window_end"):
        resolve_window(snap, SNAPSHOT, None, None)


def test_run_without_resolver_is_provisional_and_excludes_unresolved_gold(
    tmp_path: Path,
) -> None:
    result = run(
        _spec(tmp_path),
        store=FakeStore([_hit()]),
        embedder=FakeEmbedder(),
        qa_items=[_qa_item()],
    )
    assert result.metrics["provisional"] is True
    assert result.metrics["counts"]["gold_unresolved"] == 1
    assert result.metrics["retrieval"]["hit"]["all"] == {"value": None, "n": 0}
    resolved = json.loads((result.out_dir / "resolved_evidence.json").read_text(encoding="utf-8"))
    assert resolved["status"] == "not_configured"
    meta = json.loads((result.out_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["reportable"] is False
    assert meta["non_reportable_reason"] == "no_evidence_resolver"


def test_bind_existing_collection_does_not_create_real_storage(tmp_path: Path) -> None:
    pytest.importorskip("chromadb")
    from acgn_rag.config import StoreConfig
    from acgn_rag.core.store import ChromaStore

    store = ChromaStore(tmp_path / "chroma", StoreConfig())
    with pytest.raises(LookupError):
        bind_existing_collection(
            store,
            snapshot_id=SNAPSHOT,
            chunk_config="A",
            collection="snap_full_20261028abcdef1234_A",
        )
    assert not (tmp_path / "chroma" / "chroma.sqlite3").exists()


def test_bind_existing_collection_mismatch_does_not_create_the_sidecar_name(
    tmp_path: Path,
) -> None:
    pytest.importorskip("chromadb")
    from acgn_rag.config import StoreConfig
    from acgn_rag.core.store import ChromaStore

    store = ChromaStore(tmp_path / "chroma", StoreConfig())
    store.bind(SNAPSHOT, "A")  # build-side setup creates only the real collection
    with pytest.raises(RuntimeError, match="does not match sidecar"):
        bind_existing_collection(
            store,
            snapshot_id=SNAPSHOT,
            chunk_config="A",
            collection="snap_full_other_A",
        )
    names = {collection.name for collection in store.client.list_collections()}
    assert "snap_full_other_A" not in names


def test_verify_build_identity_reports_a_current_library_missing_from_build() -> None:
    build = _build_identity()
    current = {**build, "lib_newdep": "1.0"}
    assert verify_build_identity(build, current) == ["build identity missing 'lib_newdep'"]


def test_run_with_resolver_but_unverified_window_is_not_reportable(
    tmp_path: Path,
) -> None:
    result = run(
        _spec(tmp_path, window_verified=False),
        store=FakeStore([_hit()]),
        embedder=FakeEmbedder(),
        qa_items=[_qa_item()],
        evidence_resolver=lambda urls: {url: ["doc::A::1"] for url in urls},
    )
    assert result.metrics["provisional"] is False
    meta = json.loads((result.out_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["window_verified"] is False
    assert meta["reportable"] is False
    assert meta["non_reportable_reason"] == "window_unverified"


@pytest.mark.parametrize("missing", VECTOR_IDENTITY_KEYS)
def test_verify_build_identity_requires_each_vector_field(missing: str) -> None:
    current = _build_identity()
    build = {key: value for key, value in current.items() if key != missing}
    assert verify_build_identity(build, current) == [f"build identity missing '{missing}'"]


@pytest.mark.parametrize("missing", CHUNKER_FIELDS)
def test_load_build_records_requires_each_chunker_field(tmp_path: Path, missing: str) -> None:
    root = tmp_path / "index"
    chunker = {
        "version": "composed-input-v2",
        "header_version": "h1",
        "config": "A",
        "max_tokens": 512,
        "overlap": 64,
        "max_seq_length": 512,
    }
    chunker.pop(missing)
    _write_build_records(root, chunker=chunker)
    with pytest.raises(ValueError, match="chunker identity is missing"):
        load_build_records(root, snapshot_id=SNAPSHOT, config="A", label="A")


@pytest.mark.parametrize(
    ("resolver", "verified", "reportable"),
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, False),
    ],
)
def test_reportable_is_the_composition_of_resolver_and_verified_window(
    tmp_path: Path,
    resolver: bool,
    verified: bool,
    reportable: bool,
) -> None:
    result = run(
        _spec(tmp_path, window_verified=verified),
        store=FakeStore([_hit()]),
        embedder=FakeEmbedder(),
        qa_items=[_qa_item()],
        evidence_resolver=(
            (lambda urls: {url: ["doc::A::1"] for url in urls}) if resolver else None
        ),
    )
    meta = json.loads((result.out_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["reportable"] is reportable


@pytest.mark.parametrize(
    ("device", "reportable", "reason"),
    [
        ("cuda", True, None),
        ("cuda:0", True, None),
        ("cpu", False, "non_gpu_device"),
        ("mps", False, "non_gpu_device"),
        ("cudafoo", False, "non_gpu_device"),
        ("", False, "non_gpu_device"),
    ],
)
def test_reportable_requires_a_gpu_device(
    tmp_path: Path,
    device: str,
    reportable: bool,
    reason: str | None,
) -> None:
    identity = {
        **_build_identity(),
        "device": device,
        "identity_slug": f"fake-torch-{device}-fp32-b32-12345678",
    }
    result = run(
        _spec(tmp_path, build_identity=identity, require_gpu=True),
        store=FakeStore([_hit()]),
        embedder=FakeEmbedder(),
        qa_items=[_qa_item()],
        evidence_resolver=lambda urls: {url: ["doc::A::1"] for url in urls},
    )
    meta = json.loads((result.out_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["reportable"] is reportable
    assert meta["non_reportable_reason"] == reason


def test_reportable_requires_an_explicit_device(tmp_path: Path) -> None:
    identity = _build_identity()
    identity.pop("device")
    result = run(
        _spec(tmp_path, build_identity=identity, require_gpu=True),
        store=FakeStore([_hit()]),
        embedder=FakeEmbedder(),
        qa_items=[_qa_item()],
        evidence_resolver=lambda urls: {url: ["doc::A::1"] for url in urls},
    )
    meta = json.loads((result.out_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["reportable"] is False
    assert meta["non_reportable_reason"] == "non_gpu_device"


def test_cpu_is_reportable_when_the_gpu_policy_is_not_enforced(tmp_path: Path) -> None:
    result = run(
        _spec(tmp_path, build_identity=_build_identity(), require_gpu=False),
        store=FakeStore([_hit()]),
        embedder=FakeEmbedder(),
        qa_items=[_qa_item()],
        evidence_resolver=lambda urls: {url: ["doc::A::1"] for url in urls},
    )
    meta = json.loads((result.out_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["reportable"] is True
    assert meta["require_gpu"] is False
    assert meta["gpu_policy"] == "not_enforced"


def test_gold_member_not_indexed_is_distinct_from_absent(tmp_path: Path) -> None:
    result = run(
        _spec(tmp_path),
        store=FakeStore([_hit()]),
        embedder=FakeEmbedder(),
        qa_items=[_qa_item()],
        evidence_resolver=lambda urls: {url: [] for url in urls},
        snapshot_member_urls=(GOLD,),
    )
    meta = json.loads((result.out_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["metrics"]["counts"]["gold_member_not_indexed"] == 1
    assert meta["metrics"]["counts"]["gold_unavailable"] == 0
    assert meta["metrics"]["retrieval"]["hit"]["all"]["n"] == 0
    resolved = json.loads((result.out_dir / "resolved_evidence.json").read_text(encoding="utf-8"))
    assert resolved["member_not_indexed"] == [GOLD]
    assert resolved["unresolved"] == []


def test_gold_absent_from_snapshot_is_unavailable(tmp_path: Path) -> None:
    result = run(
        _spec(tmp_path),
        store=FakeStore([_hit()]),
        embedder=FakeEmbedder(),
        qa_items=[_qa_item()],
        evidence_resolver=lambda urls: {url: [] for url in urls},
        snapshot_member_urls=(),
    )
    meta = json.loads((result.out_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["metrics"]["counts"]["gold_unavailable"] == 1
    assert meta["metrics"]["counts"]["gold_member_not_indexed"] == 0
    resolved = json.loads((result.out_dir / "resolved_evidence.json").read_text(encoding="utf-8"))
    assert resolved["unresolved"] == [GOLD]
    assert resolved["member_not_indexed"] == []


def test_load_snapshot_member_urls_reads_manifest(tmp_path: Path) -> None:
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "manifest.json").write_text(
        json.dumps(
            {
                "items": [
                    {"url": GOLD},
                    {"url": "https://example.com/news/b/"},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert load_snapshot_member_urls(snap) == frozenset({GOLD, "https://example.com/news/b"})


def test_load_snapshot_member_urls_missing_manifest(tmp_path: Path) -> None:
    assert load_snapshot_member_urls(tmp_path / "absent") == frozenset()


def test_no_resolver_keeps_member_urls_unresolved(tmp_path: Path) -> None:
    result = run(
        _spec(tmp_path),
        store=FakeStore([_hit()]),
        embedder=FakeEmbedder(),
        qa_items=[_qa_item()],
        snapshot_member_urls=(GOLD,),
    )
    resolved = json.loads((result.out_dir / "resolved_evidence.json").read_text(encoding="utf-8"))
    assert resolved["status"] == "not_configured"
    assert resolved["member_not_indexed"] == []
    assert resolved["unresolved"] == [GOLD]


def test_partial_gold_member_not_indexed_keeps_item_eligible(tmp_path: Path) -> None:
    result = run(
        _spec(tmp_path),
        store=FakeStore([_hit()]),
        embedder=FakeEmbedder(),
        qa_items=[_qa_item(urls=(GOLD, OTHER))],
        evidence_resolver=lambda urls: {GOLD: ["doc::A::1"], OTHER: []},
        snapshot_member_urls=(OTHER,),
    )
    meta = json.loads((result.out_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["metrics"]["counts"]["gold_member_not_indexed"] == 0
    assert meta["metrics"]["retrieval"]["hit"]["all"] == {"value": 1.0, "n": 1}
    resolved = json.loads((result.out_dir / "resolved_evidence.json").read_text(encoding="utf-8"))
    assert resolved["member_not_indexed"] == [OTHER]


def test_no_eligible_retrieval_is_non_reportable(tmp_path: Path) -> None:
    result = run(
        _spec(tmp_path),
        store=FakeStore([_hit()]),
        embedder=FakeEmbedder(),
        qa_items=[_qa_item()],
        evidence_resolver=lambda urls: {url: [] for url in urls},
    )
    meta = json.loads((result.out_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["reportable"] is False
    assert meta["non_reportable_reason"] == "no_reportable_metrics"
    assert meta["metrics"]["reporting"]["retrieval"] == {"available": False, "n": 0}


def test_bypassed_qa_validation_is_recorded_and_non_reportable(tmp_path: Path) -> None:
    item = _qa_item()
    assert qa.validate([item]), "fixture must be invalid under the full QA rules"
    values = asdict(_spec(tmp_path))
    values.pop("qa_validated")
    values.pop("qa_source")
    spec = RunSpec(**values)  # type: ignore[arg-type]
    result = run(
        spec,
        store=FakeStore([_hit()]),
        embedder=FakeEmbedder(),
        qa_items=[item],
        evidence_resolver=lambda urls: {url: ["doc::A::1"] for url in urls},
    )
    meta = json.loads((result.out_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["reportable"] is False
    assert meta["non_reportable_reason"] == "qa_validation_bypassed"
    assert meta["qa"]["validated"] is False
    assert meta["qa"]["validation"] == "bypassed"
    assert meta["qa"]["source"] == "caller"
    assert meta["metrics"]["reporting"]["retrieval"] == {"available": True, "n": 1}


def test_file_path_qa_is_auto_validated_and_reportable(tmp_path: Path) -> None:
    values = asdict(_spec(tmp_path))
    values["qa_path"] = Path("eval/qa_set.jsonl")
    values.pop("qa_validated")
    values.pop("qa_source")
    spec = RunSpec(**values)  # type: ignore[arg-type]
    result = run(
        spec,
        store=FakeStore([_hit()]),
        embedder=FakeEmbedder(),
        qa_items=None,
        evidence_resolver=lambda urls: {url: ["doc::A::1"] for url in urls},
    )
    meta = json.loads((result.out_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["qa"]["count"] == 32
    assert meta["qa"]["validated"] is True
    assert meta["qa"]["validation"] == "validated"
    assert meta["qa"]["source"] == "runner"
    assert meta["reportable"] is True
    assert meta["metrics"]["reporting"]["retrieval"]["available"] is True


class _FakeCollection:
    def __init__(self, rows: list[dict[str, Any]], *, omit: set[str] | None = None) -> None:
        self.rows = rows
        self.omit = omit or set()
        self.calls: list[tuple[int | None, int]] = []

    def get(
        self, *, include: list[str], limit: int | None = None, offset: int = 0
    ) -> dict[str, Any]:
        assert include == ["metadatas"]
        self.calls.append((limit, offset))
        rows = [row for row in self.rows if row["id"] not in self.omit]
        page = rows[offset:] if limit is None else rows[offset : offset + limit]
        return {
            "ids": [row["id"] for row in page],
            "metadatas": [{"url": row["url"]} for row in page],
        }

    def count(self) -> int:
        return len(self.rows)


def test_build_evidence_resolver_maps_normalized_urls_to_chunk_ids() -> None:
    collection = _FakeCollection(
        [
            {"id": "c1", "url": "https://Example.com/news/a#frag"},
            {"id": "c2", "url": "https://example.com/news/a/"},
            {"id": "c3", "url": "https://example.com/news/b"},
        ]
    )
    resolve = build_evidence_resolver(collection)

    assert resolve(["https://EXAMPLE.com/news/a"]) == {"https://EXAMPLE.com/news/a": ["c1", "c2"]}
    assert resolve(["https://example.com/news/b"]) == {"https://example.com/news/b": ["c3"]}
    assert resolve(["https://example.com/news/missing"]) == {"https://example.com/news/missing": []}
    assert collection.calls == [(500, 0)]


def test_build_evidence_resolver_pages_metadata() -> None:
    rows = [
        {"id": f"c{index}", "url": f"https://example.com/news/{index}"} for index in range(1201)
    ]
    collection = _FakeCollection(rows)

    resolve = build_evidence_resolver(collection, page_size=500)

    assert collection.calls == [(500, 0), (500, 500), (500, 1000)]
    assert resolve(["https://example.com/news/0", "https://example.com/news/1200"]) == {
        "https://example.com/news/0": ["c0"],
        "https://example.com/news/1200": ["c1200"],
    }


def test_build_evidence_resolver_reads_an_empty_collection() -> None:
    collection = _FakeCollection([])

    resolve = build_evidence_resolver(collection)

    assert resolve(["https://example.com/missing"]) == {"https://example.com/missing": []}
    assert collection.calls == [(500, 0)]


def test_build_evidence_resolver_rejects_an_incomplete_scan() -> None:
    rows = [
        {"id": "c1", "url": "https://example.com/a"},
        {"id": "c2", "url": "https://example.com/b"},
        {"id": "c3", "url": "https://example.com/c"},
    ]
    collection = _FakeCollection(rows, omit={"c2"})

    with pytest.raises(RuntimeError, match="scan incomplete"):
        build_evidence_resolver(collection)


class _StuckCollection:
    def get(self, *, include: list[str], limit: int, offset: int) -> dict[str, Any]:
        assert include == ["metadatas"]
        return {"ids": ["c1"], "metadatas": [{"url": "https://example.com/a"}]}

    def count(self) -> int:
        return 2


def test_build_evidence_resolver_rejects_duplicate_pages() -> None:
    with pytest.raises(RuntimeError, match="duplicate chunk ids"):
        build_evidence_resolver(_StuckCollection(), page_size=1)


def _tier1_archive(tmp_path: Path, name: str, **overrides: object) -> Path:
    result = run(
        _spec(tmp_path, run_id=name, **overrides),
        store=FakeStore([_hit()]),
        embedder=FakeEmbedder(),
        answerer=NullAnswerer(),
        qa_items=[_qa_item()],
        evidence_resolver=lambda urls: {url: ["doc::A::1"] for url in urls},
    )
    return result.out_dir


def test_compare_tier1_runs_identical_archives(tmp_path: Path) -> None:
    left = _tier1_archive(tmp_path, "left")
    right = _tier1_archive(tmp_path, "right")

    result = compare_tier1_runs(left, right)

    assert result["identical"] is True
    assert result["comparable"] is True
    assert result["checked_items"] == 1
    assert result["metric_mismatches"] == []
    assert result["item_mismatches"] == []


def test_compare_tier1_runs_reports_metric_drift(tmp_path: Path) -> None:
    left = _tier1_archive(tmp_path, "left")
    right = _tier1_archive(tmp_path, "right")
    metrics_path = right / "metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["retrieval"]["hit"]["all"]["value"] = 0.123
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")

    result = compare_tier1_runs(left, right)

    assert result["identical"] is False
    assert result["comparable"] is True
    assert result["metric_mismatches"][0]["field"] == "retrieval"


def test_compare_tier1_runs_reports_item_order_drift(tmp_path: Path) -> None:
    left = _tier1_archive(tmp_path, "left")
    right = _tier1_archive(tmp_path, "right")
    answers_path = right / "answers.jsonl"
    rows = [json.loads(line) for line in answers_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["retrieved_chunk_ids"] = [*list(reversed(rows[0]["retrieved_chunk_ids"])), "c0"]
    answers_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    result = compare_tier1_runs(left, right)

    assert result["identical"] is False
    assert result["item_mismatches"][0]["field"] == "retrieved_chunk_ids"


def test_compare_tier1_runs_handles_unicode_line_separators(tmp_path: Path) -> None:
    left = _tier1_archive(tmp_path, "left")
    right = _tier1_archive(tmp_path, "right")
    answers_path = right / "answers.jsonl"
    rows = [
        json.loads(line)
        for line in answers_path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]
    rows[0]["answer"] = "line\u2028separator"
    answers_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    result = compare_tier1_runs(left, right)

    assert result["identical"] is True
    assert result["checked_items"] == 1


def test_compare_tier1_runs_ignores_generation_fields(tmp_path: Path) -> None:
    left = _tier1_archive(tmp_path, "left")
    right = _tier1_archive(tmp_path, "right")
    answers_path = right / "answers.jsonl"
    rows = [
        json.loads(line)
        for line in answers_path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]
    rows[0]["cited_chunk_ids"] = ["c1"]
    rows[0]["status"] = "answered"
    answers_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    result = compare_tier1_runs(left, right)

    assert result["identical"] is True
    assert result["item_mismatches"] == []


def test_compare_tier1_runs_warns_on_git_sha_drift(tmp_path: Path) -> None:
    left = _tier1_archive(tmp_path, "left")
    right = _tier1_archive(tmp_path, "right", git_sha="other-sha")

    result = compare_tier1_runs(left, right)

    assert result["identical"] is True
    assert result["comparable"] is True
    assert result["warnings"] == [{"field": "git_sha", "left": "test-sha", "right": "other-sha"}]


def test_compare_tier1_runs_rejects_identity_mismatch(tmp_path: Path) -> None:
    left = _tier1_archive(tmp_path, "left")
    identity = dict(_spec(tmp_path, run_id=None).build_identity)
    identity["identity_slug"] = "other-torch-cpu-fp32-12345678"
    right = _tier1_archive(tmp_path, "right", build_identity=identity)

    result = compare_tier1_runs(left, right)

    assert result["identical"] is False
    assert result["comparable"] is False
    assert result["reason"] == "identity mismatch"
    assert result["identity_mismatches"][0]["field"] == "identity_slug"


def test_compare_tier1_runs_rejects_different_snapshot(tmp_path: Path) -> None:
    left = _tier1_archive(tmp_path, "left")
    right = _tier1_archive(tmp_path, "right")
    meta_path = right / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["snapshot_id"] = "20261029abcdef"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    result = compare_tier1_runs(left, right)

    assert result["comparable"] is False
    assert result["identity_mismatches"][0]["field"] == "snapshot_id"


def test_compare_tier1_runs_rejects_missing_archive_file(tmp_path: Path) -> None:
    left = _tier1_archive(tmp_path, "left")
    right = _tier1_archive(tmp_path, "right")
    (right / "answers.jsonl").unlink()

    with pytest.raises(ValueError, match="invalid run archive"):
        compare_tier1_runs(left, right)


def _archive_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_archive_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _archive_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


def _write_archive_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_compare_tier1_runs_detects_changed_question(tmp_path: Path) -> None:
    left = _tier1_archive(tmp_path, "left")
    right = _tier1_archive(tmp_path, "right")
    answers_path = right / "answers.jsonl"
    rows = _archive_rows(answers_path)
    rows[0]["question"] = "A different question with the same qa_id"
    _write_archive_rows(answers_path, rows)

    result = compare_tier1_runs(left, right)

    assert result["identical"] is False
    assert result["comparable"] is True
    assert result["item_mismatches"][0]["field"] == "question"


def test_compare_tier1_runs_rejects_chunker_drift(tmp_path: Path) -> None:
    left = _tier1_archive(tmp_path, "left")
    right = _tier1_archive(tmp_path, "right")
    meta_path = right / "meta.json"
    meta = _archive_json(meta_path)
    meta["chunker"]["header_version"] = "h2"
    _write_archive_json(meta_path, meta)

    result = compare_tier1_runs(left, right)

    assert result["identical"] is False
    assert result["comparable"] is False
    assert result["reason"] == "identity mismatch"
    assert result["identity_mismatches"][0]["field"] == "chunker"


def test_compare_tier1_runs_rejects_retrieval_mode_drift(tmp_path: Path) -> None:
    left = _tier1_archive(tmp_path, "left")
    right = _tier1_archive(tmp_path, "right")
    meta_path = right / "meta.json"
    meta = _archive_json(meta_path)
    meta["retrieval"]["mode"] = "bm25"
    _write_archive_json(meta_path, meta)

    result = compare_tier1_runs(left, right)

    assert result["comparable"] is False
    assert result["reason"] == "identity mismatch"
    assert result["identity_mismatches"][0]["field"] == "retrieval_mode"


def test_compare_tier1_runs_rejects_two_empty_answer_files(tmp_path: Path) -> None:
    left = _tier1_archive(tmp_path, "left")
    right = _tier1_archive(tmp_path, "right")
    (left / "answers.jsonl").write_text("", encoding="utf-8")
    (right / "answers.jsonl").write_text("", encoding="utf-8")

    result = compare_tier1_runs(left, right)

    assert result["identical"] is False
    assert result["comparable"] is False
    assert result["reason"].startswith("answer count mismatch")


def test_compare_tier1_runs_rejects_equally_truncated_answer_files(tmp_path: Path) -> None:
    left = _tier1_archive(tmp_path, "left")
    right = _tier1_archive(tmp_path, "right")
    for archive in (left, right):
        meta_path = archive / "meta.json"
        meta = _archive_json(meta_path)
        meta["qa"]["count"] = 2
        _write_archive_json(meta_path, meta)

    result = compare_tier1_runs(left, right)

    assert result["comparable"] is False
    assert result["reason"].startswith("answer count mismatch")


def test_compare_tier1_runs_rejects_qa_fingerprint_drift(tmp_path: Path) -> None:
    left = _tier1_archive(tmp_path, "left")
    right = _tier1_archive(tmp_path, "right")
    meta_path = right / "meta.json"
    meta = _archive_json(meta_path)
    meta["qa"]["inputs_fingerprint"] = "0" * 64
    _write_archive_json(meta_path, meta)

    result = compare_tier1_runs(left, right)

    assert result["comparable"] is False
    assert result["reason"] == "identity mismatch"
    assert result["identity_mismatches"][0]["field"] == "qa_inputs_fingerprint"


def test_compare_tier1_runs_requires_a_qa_fingerprint(tmp_path: Path) -> None:
    left = _tier1_archive(tmp_path, "left")
    right = _tier1_archive(tmp_path, "right")
    for archive in (left, right):
        meta_path = archive / "meta.json"
        meta = _archive_json(meta_path)
        del meta["qa"]["inputs_fingerprint"]
        _write_archive_json(meta_path, meta)

    result = compare_tier1_runs(left, right)

    assert result["comparable"] is False
    assert result["reason"] == "identity unavailable: qa_inputs_fingerprint"


def test_compare_tier1_runs_rejects_one_sided_truncation(tmp_path: Path) -> None:
    left = _tier1_archive(tmp_path, "left")
    right = _tier1_archive(tmp_path, "right")
    (right / "answers.jsonl").write_text("", encoding="utf-8")

    result = compare_tier1_runs(left, right)

    assert result["comparable"] is False
    assert result["reason"].startswith("answer count mismatch")


def test_compare_tier1_runs_requires_identity_slug(tmp_path: Path) -> None:
    left = _tier1_archive(tmp_path, "left")
    right = _tier1_archive(tmp_path, "right")
    for archive in (left, right):
        meta_path = archive / "meta.json"
        meta = _archive_json(meta_path)
        meta["build_identity"]["identity_slug"] = ""
        _write_archive_json(meta_path, meta)

    result = compare_tier1_runs(left, right)

    assert result["comparable"] is False
    assert result["reason"] == "identity unavailable: identity_slug"


def test_compare_tier1_runs_requires_chunker_identity(tmp_path: Path) -> None:
    left = _tier1_archive(tmp_path, "left")
    right = _tier1_archive(tmp_path, "right")
    for archive in (left, right):
        meta_path = archive / "meta.json"
        meta = _archive_json(meta_path)
        del meta["chunker"]
        _write_archive_json(meta_path, meta)

    result = compare_tier1_runs(left, right)

    assert result["comparable"] is False
    assert result["reason"] == "identity unavailable: chunker"


@pytest.mark.parametrize(
    "field", ["question", "retrieved_chunk_ids", "retrieved_urls", "abstained"]
)
def test_compare_tier1_runs_rejects_a_required_answer_field_missing_from_both(
    tmp_path: Path, field: str
) -> None:
    left = _tier1_archive(tmp_path, "left")
    right = _tier1_archive(tmp_path, "right")
    for archive in (left, right):
        answers_path = archive / "answers.jsonl"
        rows = _archive_rows(answers_path)
        del rows[0][field]
        _write_archive_rows(answers_path, rows)

    with pytest.raises(ValueError, match=field):
        compare_tier1_runs(left, right)


def test_compare_tier1_runs_rejects_a_wrong_typed_retrieval_field(tmp_path: Path) -> None:
    left = _tier1_archive(tmp_path, "left")
    right = _tier1_archive(tmp_path, "right")
    for archive in (left, right):
        answers_path = archive / "answers.jsonl"
        rows = _archive_rows(answers_path)
        rows[0]["retrieved_chunk_ids"] = "c1"
        _write_archive_rows(answers_path, rows)

    with pytest.raises(ValueError, match="retrieved_chunk_ids"):
        compare_tier1_runs(left, right)


def test_compare_tier1_runs_allows_a_null_abstained_in_both(tmp_path: Path) -> None:
    left = _tier1_archive(tmp_path, "left")
    right = _tier1_archive(tmp_path, "right")
    for archive in (left, right):
        answers_path = archive / "answers.jsonl"
        rows = _archive_rows(answers_path)
        rows[0]["abstained"] = None
        _write_archive_rows(answers_path, rows)

    result = compare_tier1_runs(left, right)

    assert result["identical"] is True
    assert result["comparable"] is True


def test_run_id_rejects_path_shaped_generated_segments(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run_date"):
        _run_id(_spec(tmp_path, run_id=None, run_date="../20261030"))
    with pytest.raises(ValueError, match="snapshot_id"):
        _run_id(_spec(tmp_path, run_id=None, snapshot_id="../escape"))
    with pytest.raises(ValueError, match="config"):
        _run_id(_spec(tmp_path, run_id=None, config="A/../B"))


@pytest.mark.parametrize(
    ("path", "name"),
    [
        (("window_verified",), "window_verified"),
        (("qa", "validated"), "qa_validated"),
        (("reportable",), "reportable"),
        (("retrieval", "mode"), "retrieval_mode"),
    ],
    ids=["window-verified", "qa-validated", "reportable", "retrieval-mode"],
)
def test_compare_tier1_runs_requires_identity_flags(
    tmp_path: Path, path: tuple[str, ...], name: str
) -> None:
    left = _tier1_archive(tmp_path, "left")
    right = _tier1_archive(tmp_path, "right")
    for archive in (left, right):
        meta_path = archive / "meta.json"
        meta = _archive_json(meta_path)
        node = meta
        for part in path[:-1]:
            node = node[part]
        del node[path[-1]]
        _write_archive_json(meta_path, meta)

    result = compare_tier1_runs(left, right)

    assert result["comparable"] is False
    assert result["reason"] == f"identity unavailable: {name}"


@pytest.mark.parametrize(
    ("path", "name", "value"),
    [
        (("qa", "inputs_fingerprint"), "qa_inputs_fingerprint", "not-a-fingerprint"),
        (("retrieval", "top_k"), "top_k", "5"),
    ],
    ids=["fingerprint", "top-k"],
)
def test_compare_tier1_runs_rejects_invalid_identity_types(
    tmp_path: Path, path: tuple[str, str], name: str, value: object
) -> None:
    left = _tier1_archive(tmp_path, "left")
    right = _tier1_archive(tmp_path, "right")
    for archive in (left, right):
        meta_path = archive / "meta.json"
        meta = _archive_json(meta_path)
        meta[path[0]][path[1]] = value
        _write_archive_json(meta_path, meta)

    result = compare_tier1_runs(left, right)

    assert result["comparable"] is False
    assert result["reason"] == f"identity unavailable: {name}"


@pytest.mark.parametrize(
    ("path", "name"),
    [
        (("retrieval",), "retrieval"),
        (("counts",), "counts"),
        (("provisional",), "provisional"),
        (("reporting",), "reporting"),
        (("reporting", "retrieval"), "reporting.retrieval"),
    ],
    ids=["retrieval", "counts", "provisional", "reporting", "reporting-retrieval"],
)
def test_compare_tier1_runs_requires_metric_blocks(
    tmp_path: Path, path: tuple[str, ...], name: str
) -> None:
    left = _tier1_archive(tmp_path, "left")
    right = _tier1_archive(tmp_path, "right")
    for archive in (left, right):
        metrics_path = archive / "metrics.json"
        metrics = _archive_json(metrics_path)
        node = metrics
        for part in path[:-1]:
            node = node[part]
        del node[path[-1]]
        _write_archive_json(metrics_path, metrics)

    result = compare_tier1_runs(left, right)

    assert result["comparable"] is False
    assert result["reason"] == f"metrics unavailable: {name}"


def test_run_rejects_incomplete_chunker_identity(tmp_path: Path) -> None:
    spec = _spec(tmp_path, chunker={"version": "composed-input-v2"})

    with pytest.raises(ValueError, match="chunker identity is missing"):
        run(spec, store=FakeStore([_hit()]), embedder=FakeEmbedder(), qa_items=[_qa_item()])


def test_compare_tier1_runs_rejects_incomplete_chunker_identity(tmp_path: Path) -> None:
    left = _tier1_archive(tmp_path, "left")
    right = _tier1_archive(tmp_path, "right")
    for archive in (left, right):
        meta_path = archive / "meta.json"
        meta = _archive_json(meta_path)
        meta["chunker"] = {"version": "composed-input-v2"}
        _write_archive_json(meta_path, meta)

    result = compare_tier1_runs(left, right)

    assert result["comparable"] is False
    assert result["reason"] == "identity unavailable: chunker"


@pytest.mark.parametrize("name", ["hit", "evidence_recall"])
def test_compare_tier1_runs_rejects_missing_tier1_metric(tmp_path: Path, name: str) -> None:
    left = _tier1_archive(tmp_path, "left")
    right = _tier1_archive(tmp_path, "right")
    for archive in (left, right):
        metrics_path = archive / "metrics.json"
        metrics = _archive_json(metrics_path)
        del metrics["retrieval"][name]
        _write_archive_json(metrics_path, metrics)

    result = compare_tier1_runs(left, right)

    assert result["comparable"] is False
    assert result["reason"] == f"metrics unavailable: retrieval.{name}"


def test_compare_tier1_runs_rejects_a_missing_language_block(tmp_path: Path) -> None:
    left = _tier1_archive(tmp_path, "left")
    right = _tier1_archive(tmp_path, "right")
    for archive in (left, right):
        metrics_path = archive / "metrics.json"
        metrics = _archive_json(metrics_path)
        del metrics["retrieval"]["hit"]["en"]
        _write_archive_json(metrics_path, metrics)

    result = compare_tier1_runs(left, right)

    assert result["comparable"] is False
    assert result["reason"] == "metrics unavailable: retrieval.hit.en"


def test_compare_tier1_runs_allows_null_metric_values(tmp_path: Path) -> None:
    left = _tier1_archive(tmp_path, "left")
    right = _tier1_archive(tmp_path, "right")
    empty = {"value": None, "n": 0}
    for archive in (left, right):
        metrics_path = archive / "metrics.json"
        metrics = _archive_json(metrics_path)
        metrics["retrieval"]["hit"] = {"all": empty, "en": empty, "zh": empty}
        metrics["retrieval"]["evidence_recall"] = {"all": empty, "en": empty, "zh": empty}
        _write_archive_json(metrics_path, metrics)

    result = compare_tier1_runs(left, right)

    assert result["identical"] is True
    assert result["comparable"] is True


def test_compare_tier1_runs_rejects_a_null_rate_with_nonzero_n(tmp_path: Path) -> None:
    left = _tier1_archive(tmp_path, "left")
    right = _tier1_archive(tmp_path, "right")
    for archive in (left, right):
        metrics_path = archive / "metrics.json"
        metrics = _archive_json(metrics_path)
        metrics["retrieval"]["hit"]["all"] = {"value": None, "n": 1}
        _write_archive_json(metrics_path, metrics)

    result = compare_tier1_runs(left, right)

    assert result["comparable"] is False
    assert result["reason"] == "metrics unavailable: retrieval.hit.all"


def test_run_records_retrieval_latency(tmp_path: Path) -> None:
    result = run(
        _spec(tmp_path, run_id="latency"),
        store=FakeStore([_hit()]),
        embedder=FakeEmbedder(),
        qa_items=[_qa_item("A-001"), _qa_item("A-002")],
    )

    latency = result.meta["retrieval"]["latency_ms"]

    assert latency["n"] == 2
    assert latency["p50"] >= 0
    assert latency["p95"] >= latency["p50"]


def test_latency_summary_handles_no_values() -> None:
    assert _latency_summary([]) == {"n": 0, "p50": None, "p95": None}


class _UnboundedRejectingCollection(_FakeCollection):
    def get(
        self, *, include: list[str], limit: int | None = None, offset: int = 0
    ) -> dict[str, Any]:
        if limit is None:
            raise AssertionError("collection.get() must be paged")
        return super().get(include=include, limit=limit, offset=offset)


def test_build_evidence_resolver_never_calls_get_without_limit() -> None:
    collection = _UnboundedRejectingCollection([{"id": "c1", "url": "https://example.com/news/a"}])

    resolve = build_evidence_resolver(collection)

    assert resolve(["https://example.com/news/a"]) == {"https://example.com/news/a": ["c1"]}


def test_a_prime_run_changes_only_config_identity(tmp_path: Path) -> None:
    def _run(config: str, run_id: str, chunker: dict[str, object]) -> dict[str, Any]:
        return run(
            _spec(tmp_path, config=config, run_id=run_id, chunker=chunker),
            store=FakeStore([_hit()]),
            embedder=FakeEmbedder(),
            qa_items=[_qa_item()],
            evidence_resolver=lambda urls: {url: ["c1"] for url in urls},
        ).meta

    a = _run(
        "A",
        "matrix-a",
        {
            "version": "composed-input-v2",
            "header_version": "h1",
            "config": "A",
            "max_tokens": 512,
            "overlap": 64,
            "max_seq_length": 512,
        },
    )
    a_prime = _run(
        "A'",
        "matrix-aprime",
        {
            "version": "composed-input-v2",
            "header_version": "h1",
            "config": "A'",
            "max_tokens": 256,
            "overlap": 0,
            "max_seq_length": 512,
        },
    )

    for name in ("snapshot_id", "window_start", "window_end"):
        assert a[name] == a_prime[name], name
    for name in ("mode", "top_k", "over_fetch"):
        assert a["retrieval"][name] == a_prime["retrieval"][name], name
    assert a["qa"]["inputs_fingerprint"] == a_prime["qa"]["inputs_fingerprint"]
    assert a["build_identity"] == a_prime["build_identity"]
    assert a["embed"] == a_prime["embed"]
    for name in ("version", "header_version", "max_seq_length"):
        assert a["chunker"][name] == a_prime["chunker"][name], name

    assert a["chunk_config"] != a_prime["chunk_config"]
    assert a["chunker"]["config"] != a_prime["chunker"]["config"]
    assert a["chunker"]["max_tokens"] != a_prime["chunker"]["max_tokens"]
    assert a["chunker"]["overlap"] != a_prime["chunker"]["overlap"]
