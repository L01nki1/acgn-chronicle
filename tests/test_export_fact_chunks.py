"""Synthetic index-to-fact handoff through a temporary Chroma collection."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("chromadb")
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import export_fact_chunks  # noqa: E402
import extract_facts  # noqa: E402

from acgn_rag.config import StoreConfig  # noqa: E402
from acgn_rag.core.store import ChromaStore  # noqa: E402
from acgn_rag.core.store.payload import payload_from_fields  # noqa: E402

SNAPSHOT = "20261028-abcdef123456"
CONFIG = "A0"


@pytest.fixture()
def index(tmp_path: Path) -> tuple[Path, Any, dict[str, str]]:
    root = tmp_path / "data"
    store = ChromaStore(root / "index" / "chroma", StoreConfig())
    collection = store.bind(SNAPSHOT, CONFIG)
    payloads = []
    urls = {}
    for number in range(4):
        doc_id = f"doc{number}"
        chunk_id = f"{doc_id}::{CONFIG}::0"
        url = f"https://example.com/news/{number}"
        urls[chunk_id] = url
        payloads.append(
            payload_from_fields(
                chunk_id=chunk_id,
                doc_id=doc_id,
                snapshot_id=SNAPSHOT,
                title=f"Title {number}",
                url=url,
                publisher=f"Publisher {number}",
                source_type="official",
                published_at="2026-10-28T00:00:00+00:00",
                lang="en",
                body="Shared announcement text" if number < 2 else f"Announcement {number}",
                media_ids=[],
            )
        )
    store.upsert_chunks(payloads, [[1.0, float(number)] for number in range(4)])
    identity = {
        "model": "fake",
        "backend": "torch",
        "device": "cpu",
        "precision": "fp32",
        "batch_size": 2,
        "identity_slug": "fake-cpu",
    }
    chunker = {
        "version": "composed-input-v2",
        "header_version": "h1",
        "config": CONFIG,
        "max_tokens": 512,
        "overlap": 0,
        "max_seq_length": 512,
    }
    common = {
        "snapshot_id": SNAPSHOT,
        "config": CONFIG,
        "label": CONFIG,
        "collection": collection.name,
        "identity": identity,
        "chunker": chunker,
    }
    records = root / "index" / SNAPSHOT / CONFIG
    records.mkdir(parents=True)
    (records / "identity.json").write_text(json.dumps(common), encoding="utf-8")
    (records / "build.json").write_text(
        json.dumps(common | {"docs": 4, "chunks": 4, "stored": 4}), encoding="utf-8"
    )
    return root, collection, urls


def _args(root: Path, *extra: str) -> list[str]:
    return [
        "--data-root",
        str(root),
        "--snapshot",
        SNAPSHOT,
        "--config",
        CONFIG,
        "--output",
        "export/chunks.jsonl",
        "--page-size",
        "2",
        *extra,
    ]


class FakeLLM:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, system: str, user: str, *, max_tokens: int) -> str:
        self.calls += 1
        return json.dumps(
            [
                {
                    "title": f"Work {i}",
                    "release_date": "2026-10-05",
                    "platforms": [],
                    "event_type": "release",
                    "relevance": 0.9,
                }
                for i in range(4)
            ]
        )


def test_paged_export_preserves_provenance_and_drives_fact_extraction(
    index: tuple[Path, Any, dict[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _collection, urls = index
    assert export_fact_chunks.main(_args(root)) == 0
    output = root / "chunks" / "export" / "chunks.jsonl"
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4
    assert {row["chunk_id"]: row["url"] for row in rows} == urls
    shared = [row for row in rows if row["text"] == "Shared announcement text"]
    assert len(shared) == 2 and shared[0]["url"] != shared[1]["url"]
    receipt = json.loads(output.with_name("chunks.jsonl.receipt.json").read_text(encoding="utf-8"))
    assert receipt["snapshot_id"] == SNAPSHOT and receipt["config"] == CONFIG
    assert receipt["rows"] == 4
    assert receipt["jsonl_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()

    monkeypatch.setenv("ACGN_DATA_DIR", str(root))
    client = FakeLLM()
    assert extract_facts.main(["--snapshot", SNAPSHOT, "--input", str(output)], client=client) == 0
    facts = [
        json.loads(line)
        for line in (root / "facts" / SNAPSHOT / "facts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert client.calls == 1 and facts
    assert {fact["chunk_id"]: fact["url"] for fact in facts} == urls


def test_explicit_a_prime_uses_the_a_prime_collection_label(
    index: tuple[Path, Any, dict[str, str]],
) -> None:
    root, _collection, _urls = index
    store = ChromaStore(root / "index" / "chroma", StoreConfig())
    collection = store.bind(SNAPSHOT, "A-prime")
    payload = payload_from_fields(
        chunk_id="doc-prime::A'::0",
        doc_id="doc-prime",
        snapshot_id=SNAPSHOT,
        title="Title",
        url="https://example.com/prime",
        publisher="Publisher",
        source_type="official",
        published_at="2026-10-28T00:00:00+00:00",
        lang="en",
        body="A prime synthetic article",
        media_ids=[],
    )
    store.upsert_chunks([payload], [[1.0, 0.0]])
    old_records = root / "index" / SNAPSHOT / CONFIG
    new_records = root / "index" / SNAPSHOT / "A-prime"
    new_records.mkdir()
    for name in ("identity.json", "build.json"):
        record = json.loads((old_records / name).read_text(encoding="utf-8"))
        record.update({"config": "A'", "label": "A-prime", "collection": collection.name})
        record["chunker"]["config"] = "A'"
        if name == "build.json":
            record.update({"docs": 1, "chunks": 1, "stored": 1})
        (new_records / name).write_text(json.dumps(record), encoding="utf-8")
    assert (
        export_fact_chunks.main(
            [
                "--data-root",
                str(root),
                "--snapshot",
                SNAPSHOT,
                "--config",
                "A'",
                "--output",
                "prime/chunks.jsonl",
            ]
        )
        == 0
    )
    output = root / "chunks" / "prime" / "chunks.jsonl"
    row = json.loads(output.read_text(encoding="utf-8"))
    receipt = json.loads(output.with_name("chunks.jsonl.receipt.json").read_text(encoding="utf-8"))
    assert row["chunk_id"] == "doc-prime::A'::0"
    assert receipt["config"] == "A'" and receipt["label"] == "A-prime"


def test_incomplete_marker_and_stale_build_preserve_prior_output(
    index: tuple[Path, Any, dict[str, str]], capsys: pytest.CaptureFixture[str]
) -> None:
    root, _collection, _urls = index
    output = root / "chunks" / "export" / "chunks.jsonl"
    output.parent.mkdir(parents=True)
    output.write_text("previous output\n", encoding="utf-8")
    records = root / "index" / SNAPSHOT / CONFIG
    (records / "build.incomplete").write_text("interrupted", encoding="utf-8")
    assert export_fact_chunks.main(_args(root)) == 2
    assert "incomplete" in capsys.readouterr().err
    assert output.read_text(encoding="utf-8") == "previous output\n"
    (records / "build.incomplete").unlink()
    build_path = records / "build.json"
    build = json.loads(build_path.read_text(encoding="utf-8"))
    build["chunks"] = 3
    build_path.write_text(json.dumps(build), encoding="utf-8")
    assert export_fact_chunks.main(_args(root)) == 2
    assert output.read_text(encoding="utf-8") == "previous output\n"


@pytest.mark.parametrize(
    "field,value", [("snapshot_id", "other"), ("chunk_id", "wrong"), ("url", None)]
)
def test_invalid_row_identity_or_source_preserves_prior_output(
    index: tuple[Path, Any, dict[str, str]], field: str, value: Any
) -> None:
    root, collection, _urls = index
    collection.update(ids=[f"doc0::{CONFIG}::0"], metadatas=[{field: value}])
    output = root / "chunks" / "export" / "chunks.jsonl"
    output.parent.mkdir(parents=True)
    output.write_text("previous output\n", encoding="utf-8")
    assert export_fact_chunks.main(_args(root)) == 2
    assert output.read_text(encoding="utf-8") == "previous output\n"


@pytest.mark.parametrize("failure", ["second-page", "duplicate", "changed-record"])
def test_interrupted_or_inconsistent_paged_export_preserves_prior_output(
    index: tuple[Path, Any, dict[str, str]], monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    root, _collection, _urls = index
    output = root / "chunks" / "export" / "chunks.jsonl"
    output.parent.mkdir(parents=True)
    output.write_text("previous output\n", encoding="utf-8")
    receipt = output.with_name("chunks.jsonl.receipt.json")
    receipt.write_text("previous receipt\n", encoding="utf-8")
    original_bind = export_fact_chunks.bind_existing_collection

    class AlteredCollection:
        def __init__(self, collection: Any) -> None:
            self.collection = collection

        def count(self) -> int:
            return self.collection.count()

        def get(self, **kwargs: Any) -> Any:
            if kwargs["offset"] >= 2:
                if failure == "second-page":
                    raise RuntimeError("simulated page failure")
                if failure == "duplicate":
                    kwargs["offset"] = 0
                if failure == "changed-record":
                    path = root / "index" / SNAPSHOT / CONFIG / "build.json"
                    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            return self.collection.get(**kwargs)

    def altered_bind(*args: Any, **kwargs: Any) -> AlteredCollection:
        return AlteredCollection(original_bind(*args, **kwargs))

    monkeypatch.setattr(export_fact_chunks, "bind_existing_collection", altered_bind)
    assert export_fact_chunks.main(_args(root)) == 2
    assert output.read_text(encoding="utf-8") == "previous output\n"
    assert receipt.read_text(encoding="utf-8") == "previous receipt\n"
    assert not list(output.parent.glob("*.tmp"))


@pytest.mark.parametrize(
    "output", ["../escape.jsonl", "/tmp/escape.jsonl", "C:\\escape.jsonl", "bad.txt"]
)
def test_output_path_cannot_escape_chunks_directory(
    index: tuple[Path, Any, dict[str, str]], output: str
) -> None:
    root, _collection, _urls = index
    assert (
        export_fact_chunks.main(
            [
                "--data-root",
                str(root),
                "--snapshot",
                SNAPSHOT,
                "--config",
                CONFIG,
                "--output",
                output,
            ]
        )
        == 2
    )
    assert not (root / "chunks").exists()


def test_output_rejects_symlinks_and_hardlinks(
    index: tuple[Path, Any, dict[str, str]], tmp_path: Path
) -> None:
    root, _collection, _urls = index
    output = root / "chunks" / "export" / "chunks.jsonl"
    output.parent.mkdir(parents=True)
    target = tmp_path / "do-not-replace.jsonl"
    target.write_text("original\n", encoding="utf-8")
    try:
        output.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symbolic links are unavailable: {exc}")
    assert export_fact_chunks.main(_args(root)) == 2
    assert target.read_text(encoding="utf-8") == "original\n"
    output.unlink()
    try:
        output.hardlink_to(target)
    except OSError as exc:
        pytest.skip(f"hard links are unavailable: {exc}")
    assert export_fact_chunks.main(_args(root)) == 2
    assert target.read_text(encoding="utf-8") == "original\n"


def test_output_rejects_a_linked_parent_directory(
    index: tuple[Path, Any, dict[str, str]], tmp_path: Path
) -> None:
    root, _collection, _urls = index
    chunks = root / "chunks"
    chunks.mkdir()
    target = tmp_path / "outside"
    target.mkdir()
    try:
        (chunks / "export").symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symbolic links are unavailable: {exc}")
    assert export_fact_chunks.main(_args(root)) == 2
    assert not (target / "chunks.jsonl").exists()


def test_failed_jsonl_replacement_preserves_old_output_and_invalidates_receipt(
    index: tuple[Path, Any, dict[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _collection, _urls = index
    output = root / "chunks" / "export" / "chunks.jsonl"
    output.parent.mkdir(parents=True)
    output.write_text("previous output\n", encoding="utf-8")
    receipt = output.with_name("chunks.jsonl.receipt.json")
    receipt.write_text("previous receipt\n", encoding="utf-8")
    original_replace = export_fact_chunks.os.replace

    def fail_final_replace(source: Any, destination: Any) -> None:
        if Path(destination) == output:
            raise OSError("simulated JSONL replacement failure")
        original_replace(source, destination)

    monkeypatch.setattr(export_fact_chunks.os, "replace", fail_final_replace)
    assert export_fact_chunks.main(_args(root)) == 2
    assert output.read_text(encoding="utf-8") == "previous output\n"
    new_receipt = json.loads(receipt.read_text(encoding="utf-8"))
    assert new_receipt["jsonl_sha256"] != hashlib.sha256(output.read_bytes()).hexdigest()
    assert not list(output.parent.glob("*.tmp"))


def test_mismatched_config_missing_records_and_empty_collection_are_rejected(
    index: tuple[Path, Any, dict[str, str]],
) -> None:
    root, collection, urls = index
    assert (
        export_fact_chunks.main(
            [
                "--data-root",
                str(root),
                "--snapshot",
                SNAPSHOT,
                "--config",
                "A",
                "--output",
                "out.jsonl",
            ]
        )
        == 2
    )
    collection.delete(ids=list(urls))
    assert export_fact_chunks.main(_args(root)) == 2
    assert not (root / "chunks").exists()


def test_preview_only_records_cannot_be_exported_as_formal_index(tmp_path: Path) -> None:
    root = tmp_path / "data" / "previews" / "trial"
    records = root / "index" / SNAPSHOT / CONFIG
    records.mkdir(parents=True)
    (records / "preview-build.json").write_text("{}", encoding="utf-8")
    (records / "preview-identity.json").write_text("{}", encoding="utf-8")
    assert export_fact_chunks.main(_args(root)) == 2
    assert not (root / "chunks").exists()


@pytest.mark.parametrize("page_size", ["0", "-1"])
def test_nonpositive_page_size_is_rejected_before_access(
    tmp_path: Path, page_size: str, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "missing-data"
    assert (
        export_fact_chunks.main(
            [
                "--data-root",
                str(root),
                "--snapshot",
                SNAPSHOT,
                "--config",
                CONFIG,
                "--output",
                "out.jsonl",
                "--page-size",
                page_size,
            ]
        )
        == 2
    )
    assert "--page-size must be positive" in capsys.readouterr().err
    assert not root.exists()
