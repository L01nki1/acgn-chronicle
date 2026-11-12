"""The offline end-to-end smoke runs on a synthetic snapshot: build -> export -> retrieve -> /ask.

Owner: F (WP2) | Spec: docs/ROADMAP.md G1b, docs/RUNBOOK.md §1-§3

These tests pin the *wiring* the smoke exists to prove, not any quality number: with the hash
encoder and the stub answer client injected, a frozen snapshot must reach a bound collection,
an exported chunk file, a retrieval result and a grounded `/ask` payload whose citations are a
subset of what was retrieved. Chroma is required for the store half; the snapshot half runs
without it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")  # Included in [dev]; the pipeline uses a fake encoder.

from acgn_rag.core.ingest.normalize import NormalizedItem

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import make_smoke_snapshot  # noqa: E402
import smoke_e2e  # noqa: E402


@pytest.fixture()
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "data"
    monkeypatch.setenv("ACGN_DATA_DIR", str(root))
    monkeypatch.delenv("ACGN_EMBED_MODEL", raising=False)
    return root


def _publish(root: Path, count: int = 6) -> str:
    assert make_smoke_snapshot.main(["--data-root", str(root), "--count", str(count)]) == 0
    # The publisher also writes the catalog beside the snapshots; only directories are snapshots.
    snapshots = sorted(path for path in (root / "snapshots").iterdir() if path.is_dir())
    assert len(snapshots) == 1
    return snapshots[0].name


def test_fixture_writes_a_verifiable_frozen_snapshot(data_root: Path) -> None:
    snapshot = _publish(data_root, count=6)
    snap_dir = data_root / "snapshots" / snapshot
    for name in ("manifest.json", "items.jsonl", "representatives.jsonl", "clusters.jsonl"):
        assert (snap_dir / name).is_file(), f"{name} missing from the published snapshot"
    manifest = json.loads((snap_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["window_end"] == "2026-11-05"
    assert snapshot.startswith("20261105-")


def test_fixture_rejects_a_nonpositive_count(data_root: Path) -> None:
    assert make_smoke_snapshot.main(["--data-root", str(data_root), "--count", "0"]) == 2
    assert not (data_root / "snapshots").exists()


def test_fixture_documents_are_synthetic_and_stable() -> None:
    first = make_smoke_snapshot._doc(0, make_smoke_snapshot.SEEDS[0])
    again = make_smoke_snapshot._doc(0, make_smoke_snapshot.SEEDS[0])
    assert isinstance(first, NormalizedItem)
    assert first == again, "the fixture must be deterministic"
    assert first.url.startswith("https://example.invalid/")
    assert first.doc_id.startswith("doc_") and len(first.doc_id) == 68


def test_smoke_reports_a_missing_snapshot_without_touching_data(data_root: Path) -> None:
    code = smoke_e2e.main(["--snapshot", "20261105-000000000000", "--data-root", str(data_root)])
    assert code == 2
    assert not (data_root / "index").exists()


def test_smoke_rejects_a_nonpositive_limit(data_root: Path) -> None:
    assert smoke_e2e.main(["--snapshot", "x", "--limit", "0"]) == 2


def test_smoke_end_to_end_on_the_synthetic_snapshot(data_root: Path, tmp_path: Path) -> None:
    pytest.importorskip("chromadb")
    pytest.importorskip("fastapi")

    snapshot = _publish(data_root, count=6)
    out = tmp_path / "report.json"
    code = smoke_e2e.main(
        [
            "--snapshot",
            snapshot,
            "--data-root",
            str(data_root),
            "--top-k",
            "3",
            "--out",
            str(out),
        ]
    )
    report = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0, report.get("error")
    assert report["steps"]["build"]["exit_code"] == 0
    assert report["steps"]["export"]["exit_code"] == 0

    retrieval = report["steps"]["retrieval"]
    assert retrieval["collection"] == f"snap_full_{snapshot}_A"
    assert retrieval["returned"] == 3
    assert retrieval["counts"]["stored"] == retrieval["counts"]["chunks"]

    facts = report["steps"]["facts"]
    assert facts["exit_code"] == 0
    assert facts["rows"] > 0
    # G1b: every fact row carries its chunkId and url.
    assert facts["rows_with_provenance"] == facts["rows"]

    ask = report["steps"]["ask"]
    assert ask["status_code"] == 200
    assert ask["model_calls"] == 1
    assert ask["abstained"] is False
    assert ask["cited_subset_of_retrieved"] is True
    assert set(ask["cited_chunk_ids"]) <= set(retrieval["top_chunk_ids"])


def test_first_evidence_text_reads_the_numbered_block() -> None:
    user = (
        "Question:\ntest\n\nNumbered evidence for this request:\n"
        '1. {"text":"first body","url":"u"}\n'
        '2. {"text":"second body","url":"u"}\n'
    )
    assert smoke_e2e._first_evidence_text(user) == "first body"
    assert smoke_e2e._first_evidence_text("no evidence here") == ""


def test_stub_claims_a_verbatim_window_of_the_first_evidence() -> None:
    stub = smoke_e2e.StubAnswerClient()
    user = 'Question:\nq\n\nNumbered evidence for this request:\n1. {"text":"abcdefghij"}\n'
    completion = stub.complete("system", user, max_tokens=1024)
    payload = json.loads(completion.text)
    assert payload["claims"][0]["evidence_indices"] == [1]
    assert "abcdefghij" in payload["claims"][0]["text"]
    assert stub.calls == 1


def test_extraction_stub_answers_one_object_per_numbered_slot() -> None:
    stub = smoke_e2e.StubExtractionClient()
    user = "Extract the facts for the 3 numbered announcements below.\n\n1. a\n\n2. b\n\n3. c\n\n"
    rows = json.loads(stub.complete("system", user, max_tokens=1024))
    assert len(rows) == 3
    assert set(rows[0]) == {"title", "release_date", "platforms", "event_type", "relevance"}
    assert stub.calls == 1


def test_smoke_never_touches_an_existing_collection_or_fact_file(
    data_root: Path, tmp_path: Path
) -> None:
    """F's #75 review, P1: a smoke run must not replace real indexes or fact files."""
    pytest.importorskip("chromadb")
    pytest.importorskip("fastapi")

    snapshot = _publish(data_root, count=6)

    # Seed the source root the way a previous full run would have left it: a complete
    # collection with its build record, and a fact file at the ordinary facts path.
    docs, _ = smoke_e2e._docs_path(snapshot, data_root, data_root, None)
    assert smoke_e2e._build(snapshot, docs, data_root)["exit_code"] == 0
    facts_file = data_root / "facts" / snapshot / "facts.jsonl"
    facts_file.parent.mkdir(parents=True, exist_ok=True)
    facts_file.write_bytes(b'{"seed": true}\n')

    from acgn_rag.config import load_settings
    from acgn_rag.core.store.chroma_store import ChromaStore

    store = ChromaStore(data_root / "index" / "chroma", load_settings().store)
    store.bind(snapshot, smoke_e2e.LABEL)
    before_count = store.count()
    assert before_count == 6
    build_record = (data_root / "index" / snapshot / smoke_e2e.LABEL / "build.json").read_bytes()

    # A limited smoke run must not touch any of it.
    assert (
        smoke_e2e.main(["--snapshot", snapshot, "--data-root", str(data_root), "--limit", "2"]) == 0
    )
    assert store.count() == before_count
    assert (
        data_root / "index" / snapshot / smoke_e2e.LABEL / "build.json"
    ).read_bytes() == build_record
    assert facts_file.read_bytes() == b'{"seed": true}\n'

    # Neither must an unlimited run — the ordinary facts path was the old overwrite vector.
    assert (
        smoke_e2e.main(
            [
                "--snapshot",
                snapshot,
                "--data-root",
                str(data_root),
                "--out",
                str(tmp_path / "unlimited.json"),
            ]
        )
        == 0
    )
    assert store.count() == before_count
    assert (
        data_root / "index" / snapshot / smoke_e2e.LABEL / "build.json"
    ).read_bytes() == build_record
    assert facts_file.read_bytes() == b'{"seed": true}\n'

    report = json.loads((tmp_path / "unlimited.json").read_text(encoding="utf-8"))
    workspace = Path(report["workspace"])
    assert (workspace / "index" / "chroma").is_dir()
