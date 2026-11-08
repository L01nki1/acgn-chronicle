"""CLI contract for scripts/build_index.py: end-to-end builds on fakes, fully offline.

numpy gates acgn_rag.core.embed (as in tests/test_embed_encoder.py), chromadb gates the store
(as in tests/test_store_chroma.py), and the FakeTokenizer/FakeModel pair mirrors those two
files. A 3-doc corpus (50/40/12 sentences of 20 whitespace tokens) drives the whole pipeline -
chunk -> embed with context headers -> identity-keyed cache -> Chroma upsert - for all three
configs sequentially (RUNBOOK §2), asserting per-config chunk counts (A' > A > A0 on this
corpus), the sidecar identity/build records, warm-cache behavior, idempotent rebuilds and the
--limit / --config / --data-root / --embed-cache-dir CLI surface. Nothing loads a model.
Rejection and replacement semantics (PR #24 review) get their own tests: zero-document and
zero-chunk inputs exit 2 without writing or deleting anything, and --fresh replaces exactly
the selected collection (stale rows of a larger earlier build) while siblings stay intact.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")  # Included in [dev]; the pipeline uses a fake encoder.
pytest.importorskip("chromadb")  # [store] extra; same gate as tests/test_store_chroma.py

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import build_index  # noqa: E402

from acgn_rag.config import StoreConfig  # noqa: E402
from acgn_rag.core.chunk import chunk_text  # noqa: E402
from acgn_rag.core.embed import build_header  # noqa: E402
from acgn_rag.core.store.chroma_store import ChromaStore, collection_name, media_filter  # noqa: E402

SNAP = "20261028abcdef1234"
LABELS = {"A": "A", "A'": "A-prime", "A0": "A0"}
DIM = 32
MODEL_E5 = "intfloat/multilingual-e5-base"


class FakeTokenizer:
    """Whitespace tokenizer shaped like e5 (copy of tests/test_chunk_a.py's fake).

    ":" splits off as its own token so count("passage: ") == 2 content tokens, mirroring the
    real model; ids() wraps the content ids in 2 specials, so the A/A0 budget is exactly the
    e5 shape (508 at max_seq_length 512) without a model. Ids are assigned by first sight
    (insertion order), so they are stable across runs without depending on Python's randomized
    hashing.
    """

    BOS = 0
    EOS = 2

    def __init__(self) -> None:
        self._id_of: dict[str, int] = {}
        self._word_of: dict[int, str] = {}

    def _word_ids(self, text: str) -> list[int]:
        ids: list[int] = []
        for word in text.split():
            for part in self._split_colon(word):
                if part not in self._id_of:
                    self._id_of[part] = 10 + len(self._id_of)
                    self._word_of[self._id_of[part]] = part
                ids.append(self._id_of[part])
        return ids

    @staticmethod
    def _split_colon(word: str) -> list[str]:
        parts: list[str] = []
        for index, piece in enumerate(word.split(":")):
            if index:
                parts.append(":")
            if piece:
                parts.append(piece)
        return parts

    def count(self, text: str) -> int:
        return len(self._word_ids(text))

    def ids(self, text: str) -> list[int]:
        return [self.BOS, *self._word_ids(text), self.EOS]

    def decode(self, ids: Sequence[int]) -> str:
        return " ".join(self._words_of_ids(ids))

    def _words_of_ids(self, ids: Sequence[int]) -> list[str]:
        return [self._word_of[i] for i in ids if i in self._word_of]


class FakeModel:
    """Deterministic vectors without a model download (copy of tests/test_embed_encoder.py's).

    seed = sha256(text) -> 32 standard-normal floats: the same text gives the identical
    vector, different texts give different vectors. ``calls`` records exactly what encode()
    received, so tests can assert what reached the model and how often.
    """

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim
        self.calls: list[dict[str, Any]] = []

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int | None = None,
        prompt_name: str | None = None,
        **_kwargs: Any,
    ) -> np.ndarray:
        self.calls.append(
            {"texts": list(texts), "batch_size": batch_size, "prompt_name": prompt_name}
        )
        return np.array([self.vector(text) for text in texts], dtype=np.float32)

    def vector(self, text: str) -> np.ndarray:
        seed = int.from_bytes(sha256(text.encode("utf-8")).digest()[:8], "big")
        return np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)


def _sentences(count: int, words: int, tag: str) -> str:
    """`count` sentences of `words` whitespace tokens each (the fake tokenizer's unit)."""
    return " ".join(" ".join(f"{tag}{i}w{j}" for j in range(words)) + "." for i in range(count))


def _doc(index: int, sentences: int) -> dict[str, Any]:
    return {
        "doc_id": f"doc{index:02d}",
        "title": f"Title {index}",
        "publisher": "Example Publisher",
        "published_at": "2026-10-28T00:00:00+00:00",
        "url": f"https://example.com/news/{index}",
        "source_type": "official",
        "lang": "ja",
        "media_ids": [index],
        "body": _sentences(sentences, 20, f"d{index}s"),
    }


# 50/40/12 sentences of 20 tokens: A' (budget 256) needs 10 chunks, A (508/64 overlap) 6,
# A0 (508/0) 5 - the overlap eats capacity in A, the smaller budget eats more in A'.
DOC_SENTENCES = (50, 40, 12)


def _rows() -> list[dict[str, Any]]:
    return [
        _doc(index, sentences) for index, sentences in zip(range(1, 4), DOC_SENTENCES, strict=True)
    ]


def _expected_chunks(config: str, rows: list[dict[str, Any]] | None = None) -> int:
    """Independent chunk count: run the chunker directly, never through the CLI."""
    rows = _rows() if rows is None else rows
    return sum(
        len(
            chunk_text(
                row["doc_id"],
                row["body"],
                config,
                FakeTokenizer(),
                header=build_header(row["title"], row["publisher"], row["published_at"]),
            )
        )
        for row in rows
    )


def _write_docs(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


@pytest.fixture()
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "data"
    monkeypatch.setenv("ACGN_DATA_DIR", str(root))
    # the e5 default keeps the model-spec registry honest; a stray local override would fail
    # model_spec() for a family nothing here knows about
    monkeypatch.delenv("ACGN_EMBED_MODEL", raising=False)
    return root


@pytest.fixture()
def docs_path(tmp_path: Path) -> Path:
    return _write_docs(tmp_path / "docs.jsonl", _rows())


def _run(argv: list[str], *, model: FakeModel | None = None) -> int:
    return build_index.main(argv, tokenizer=FakeTokenizer(), model=model or FakeModel())


def _store_at(data_root: Path) -> ChromaStore:
    return ChromaStore(data_root / "index" / "chroma", StoreConfig())


def _collection_ids(store: ChromaStore, snapshot: str, label: str) -> list[str]:
    collection = store.client.get_collection(collection_name(snapshot, label))
    result = collection.get(include=[])  # 1.x returns models; older wheels returned dicts
    return list(result["ids"] if isinstance(result, dict) else result.ids)


def _summary(out: str, config: str) -> re.Match[str]:
    line = re.search(
        rf"config={re.escape(config)} label={re.escape(LABELS[config])} "
        r"docs=(\d+) chunks=(\d+) cache_hits=(\d+) cache_misses=(\d+) "
        r"encodes=(\d+) stored=(\d+) seconds=\d+\.\d{3} "
        rf"collection={re.escape(collection_name(SNAP, LABELS[config]))} "
        r"mode=(?P<mode>replace|upsert)",
        out,
    )
    assert line, f"no summary line for config {config!r} in:\n{out}"
    return line


def test_end_to_end_builds_all_three_configs(
    data_root: Path, docs_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = _run(["--snapshot", SNAP, "--input", str(docs_path)])
    assert code == 0
    out = capsys.readouterr().out

    store = _store_at(data_root)
    counts: dict[str, int] = {}
    for config, label in LABELS.items():
        expected = _expected_chunks(config)
        counts[config] = expected
        line = _summary(out, config)
        # groups 1-6 are the numeric counters; group 7 is the named mode (upsert/replace)
        docs, chunks, hits, misses, encodes, stored = (int(g) for g in line.groups()[:6])
        assert docs == 3
        assert chunks == expected  # the store holds exactly the chunker's output
        assert hits + misses == chunks  # one cache lookup per chunk
        assert encodes == misses  # only misses are encoded
        assert stored == expected

        ids = _collection_ids(store, SNAP, label)
        assert len(ids) == expected
        assert ids[0].startswith(f"doc01::{config}::")  # chunk ids keep the variant name

        build = json.loads(
            (data_root / "index" / SNAP / label / "build.json").read_text(encoding="utf-8")
        )
        assert build["snapshot_id"] == SNAP and build["config"] == config
        assert build["label"] == label
        assert build["chunks"] == expected and build["docs"] == 3
        assert build["collection"] == collection_name(SNAP, label)
        assert build["identity"]["model"] == MODEL_E5  # the Settings default, honestly recorded
        assert build["identity"]["identity_slug"]

        identity = json.loads(
            (data_root / "index" / SNAP / label / "identity.json").read_text(encoding="utf-8")
        )
        assert identity["snapshot_id"] == SNAP and identity["config"] == config
        assert identity["identity"] == build["identity"]
        assert identity["chunker"] == build["chunker"]
        assert build["chunker"]["version"] == "composed-input-v2"
    # The smaller budget produces more chunks. Header-dependent packing can leave the
    # overlap and no-overlap variants with equal counts but different boundaries.
    assert counts["A'"] > counts["A"] >= counts["A0"]
    # the first config on a cold cache encodes every chunk
    first = _summary(out, "A")
    assert first.group(3) == "0" and first.group(4) == first.group(2)


def test_encoded_texts_carry_prefix_and_context_header(data_root: Path, docs_path: Path) -> None:
    model = FakeModel()
    code = _run(
        [
            "--snapshot",
            SNAP,
            "--input",
            str(docs_path),
            "--config",
            "A0",
            "--limit",
            "1",
            "--preview",
            "--preview-id",
            "header-test",
        ],
        model=model,
    )
    assert code == 0
    row = _rows()[0]
    first_body = chunk_text(
        "doc01",
        row["body"],
        "A0",
        FakeTokenizer(),
        header=build_header(row["title"], row["publisher"], row["published_at"]),
    )[0].text
    assert all(
        len(FakeTokenizer().ids(text)) <= 512 for call in model.calls for text in call["texts"]
    )
    # RUNBOOK §3.5: what is encoded is prefix + "title | publisher | date" + newline + body
    assert model.calls[0]["texts"][0] == (
        "passage: Title 1 | Example Publisher | 2026-10-28T00:00:00+00:00\n" + first_body
    )
    assert model.calls[0]["prompt_name"] is None


def test_second_build_with_warm_cache_encodes_nothing(
    data_root: Path, docs_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["--snapshot", SNAP, "--input", str(docs_path), "--config", "A"]
    first = FakeModel()
    assert _run(argv, model=first) == 0
    cold = _summary(capsys.readouterr().out, "A")  # flush: the next readouterr is run 2 only
    assert cold.group(3) == "0"  # cache_hits: a cold cache misses every chunk
    assert first.calls  # the cold build did encode
    cache_files = list((data_root / "cache" / "embed").rglob("*.npy"))
    assert len(cache_files) == _expected_chunks("A")  # one vector per miss, under the default dir

    second = FakeModel()
    assert _run(argv, model=second) == 0
    line = _summary(capsys.readouterr().out, "A")
    chunks, hits, misses, encodes = (int(line.group(i)) for i in (2, 3, 4, 5))
    assert (misses, encodes) == (0, 0)  # every vector served from the cache
    assert hits == chunks
    assert second.calls == []  # zero model calls on the warm rebuild


def test_rebuild_is_idempotent_upsert_not_duplicate(data_root: Path, docs_path: Path) -> None:
    argv = ["--snapshot", SNAP, "--input", str(docs_path), "--config", "A"]
    assert _run(argv) == 0
    first_ids = _collection_ids(_store_at(data_root), SNAP, "A")
    assert _run(argv) == 0  # same input, same snapshot: rebuild
    second_ids = _collection_ids(_store_at(data_root), SNAP, "A")
    assert len(second_ids) == len(first_ids)  # upsert semantics: no duplicate ids
    assert len(set(second_ids)) == len(second_ids)
    assert set(second_ids) == set(first_ids)


def test_empty_input_is_rejected_and_writes_nothing(
    data_root: Path,
    docs_path: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Zero document rows: exit 2, no sidecar rewrite, existing collections preserved."""
    assert _run(["--snapshot", SNAP, "--input", str(docs_path), "--config", "A0"]) == 0
    ids_before = _collection_ids(_store_at(data_root), SNAP, "A0")
    build_before = (data_root / "index" / SNAP / "A0" / "build.json").read_text(encoding="utf-8")

    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    code = _run(["--snapshot", SNAP, "--input", str(empty), "--config", "A0", "--fresh"])
    assert code == 2
    assert "holds no document rows" in capsys.readouterr().err
    # the earlier valid build is untouched: same sidecar bytes, same collection rows
    assert (data_root / "index" / SNAP / "A0" / "build.json").read_text(encoding="utf-8") == (
        build_before
    )
    assert _collection_ids(_store_at(data_root), SNAP, "A0") == ids_before


def test_blank_bodies_produce_no_chunks_and_reject_the_build(
    data_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Non-empty rows whose bodies are blank produce zero chunks: rejected, nothing written."""
    blank = _doc(1, 3) | {"body": "   \n\n  "}
    path = _write_docs(tmp_path / "blank.jsonl", [blank])
    code = _run(["--snapshot", SNAP, "--input", str(path), "--config", "A0"])
    assert code == 2
    assert "produced no chunks" in capsys.readouterr().err
    assert not (data_root / "index" / SNAP / "A0").exists()  # no sidecar dir
    assert _store_at(data_root).client.list_collections() == []  # no collection created


def test_fresh_rebuild_replaces_stale_rows_only_in_the_target_collection(
    data_root: Path,
    docs_path: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A smaller rebuild upserts (stale rows survive) unless --fresh replaces the collection;
    a --fresh rebuild touches only the selected snapshot/config, never its siblings."""
    assert _run(["--snapshot", SNAP, "--input", str(docs_path), "--config", "A0,A"]) == 0
    capsys.readouterr()  # flush the full build's output; the next readouterr is the trial run
    store = _store_at(data_root)
    full_a0 = _collection_ids(store, SNAP, "A0")
    sibling_a = _collection_ids(store, SNAP, "A")

    subset_rows = _rows()[:1]
    subset = _write_docs(tmp_path / "subset.jsonl", subset_rows)
    expected_subset = _expected_chunks("A0", subset_rows)
    assert expected_subset < len(full_a0)  # the trial really is smaller

    # default behavior: upsert keeps the stale rows of the larger earlier build
    assert _run(["--snapshot", SNAP, "--input", str(subset), "--config", "A0"]) == 0
    line = _summary(capsys.readouterr().out, "A0")
    assert line.group("mode") == "upsert"
    assert int(line.group(2)) == expected_subset  # chunks this build produced
    assert int(line.group(6)) == len(full_a0)  # stored: stale rows silently survive
    assert _collection_ids(store, SNAP, "A0") == full_a0

    # --fresh: exactly the selected collection is rebuilt
    assert _run(["--snapshot", SNAP, "--input", str(subset), "--config", "A0", "--fresh"]) == 0
    line = _summary(capsys.readouterr().out, "A0")
    assert line.group("mode") == "replace"
    assert int(line.group(6)) == expected_subset  # metadata/count consistency
    expected_ids = [
        chunk.chunk_id
        for row in subset_rows
        for chunk in chunk_text(
            row["doc_id"],
            row["body"],
            "A0",
            FakeTokenizer(),
            header=build_header(row["title"], row["publisher"], row["published_at"]),
        )
    ]
    assert _collection_ids(store, SNAP, "A0") == expected_ids
    assert _collection_ids(store, SNAP, "A") == sibling_a  # the sibling is untouched
    build = json.loads(
        (data_root / "index" / SNAP / "A0" / "build.json").read_text(encoding="utf-8")
    )
    assert build["mode"] == "replace"
    assert build["chunks"] == build["stored"] == expected_subset


@pytest.mark.parametrize("omit_media_ids", [False, True], ids=["empty", "missing"])
def test_fresh_rebuild_accepts_an_unresolved_item(
    data_root: Path, tmp_path: Path, omit_media_ids: bool
) -> None:
    original = _write_docs(tmp_path / "original.jsonl", [_doc(1, 3) | {"media_ids": [7]}])
    assert _run(["--snapshot", SNAP, "--input", str(original), "--config", "A0"]) == 0

    unresolved = _doc(1, 3)
    if omit_media_ids:
        del unresolved["media_ids"]
    else:
        unresolved["media_ids"] = []
    replacement = _write_docs(tmp_path / "replacement.jsonl", [unresolved])
    assert _run(["--snapshot", SNAP, "--input", str(replacement), "--config", "A0", "--fresh"]) == 0

    store = _store_at(data_root)
    store.bind(SNAP, "A0")
    assert store.count() == 1
    assert store.query([1.0] * DIM, top_k=1)[0].payload.media_ids == ()
    assert store.query([1.0] * DIM, top_k=1, where=media_filter([7])) == []
    sidecars = data_root / "index" / SNAP / "A0"
    build = json.loads((sidecars / "build.json").read_text(encoding="utf-8"))
    identity = json.loads((sidecars / "identity.json").read_text(encoding="utf-8"))
    assert build["mode"] == "replace"
    assert build["chunks"] == build["stored"] == 1
    assert build["snapshot_id"] == identity["snapshot_id"] == SNAP
    assert build["config"] == identity["config"] == "A0"
    assert build["collection"] == identity["collection"] == collection_name(SNAP, "A0")


@pytest.mark.parametrize("omit_category", [False, True], ids=["null", "missing"])
def test_rebuild_preserves_and_clears_ingest_category(
    data_root: Path, tmp_path: Path, omit_category: bool
) -> None:
    original = _doc(1, 3) | {"media_ids": [7], "ingest_category": "corporate"}
    original_path = _write_docs(tmp_path / "tagged.jsonl", [original])
    assert _run(["--snapshot", SNAP, "--input", str(original_path), "--config", "A0"]) == 0

    store = _store_at(data_root)
    store.bind(SNAP, "A0")
    assert store.query([1.0] * DIM, top_k=1)[0].payload.ingest_category == "corporate"

    updated = _doc(1, 3) | {"media_ids": [7]}
    if not omit_category:
        updated["ingest_category"] = None
    updated_path = _write_docs(tmp_path / "untagged.jsonl", [updated])
    assert _run(["--snapshot", SNAP, "--input", str(updated_path), "--config", "A0"]) == 0

    stored = store.query([1.0] * DIM, top_k=1)[0].payload
    assert stored.ingest_category is None
    assert stored.media_ids == (7,)
    assert store.query([1.0] * DIM, top_k=1, where={"ingest_category": {"$eq": "corporate"}}) == []
    assert store.query([1.0] * DIM, top_k=1, where=media_filter([7]))[0].payload == stored


def test_fresh_rejects_invalid_input_without_deleting_the_collection(
    data_root: Path,
    docs_path: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--fresh validates the input BEFORE the destructive step: a rejected build never wipes."""
    assert _run(["--snapshot", SNAP, "--input", str(docs_path), "--config", "A0"]) == 0
    ids_before = _collection_ids(_store_at(data_root), SNAP, "A0")

    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    code = _run(["--snapshot", SNAP, "--input", str(empty), "--config", "A0", "--fresh"])
    assert code == 2
    assert _collection_ids(_store_at(data_root), SNAP, "A0") == ids_before

    bad = _doc(1, 3) | {"source_type": "blog"}  # invalid doc row (not a SourceType)
    path = _write_docs(tmp_path / "bad.jsonl", [bad])
    code = _run(["--snapshot", SNAP, "--input", str(path), "--config", "A0", "--fresh"])
    assert code == 2
    assert _collection_ids(_store_at(data_root), SNAP, "A0") == ids_before


def test_fresh_keeps_a_different_snapshot_with_the_same_legacy_prefix(
    data_root: Path, docs_path: Path, tmp_path: Path
) -> None:
    snapshot_a = "20261028-abc000000000"
    snapshot_b = "20261028-abc111111111"
    docs_b = _write_docs(tmp_path / "snapshot-b.jsonl", [_doc(7, 3)])
    assert _run(["--snapshot", snapshot_a, "--input", str(docs_path), "--config", "A0"]) == 0
    assert _run(["--snapshot", snapshot_b, "--input", str(docs_b), "--config", "A0"]) == 0

    store = _store_at(data_root)
    store.bind(snapshot_b, "A0")
    before_ids = _collection_ids(store, snapshot_b, "A0")
    before_build = (data_root / "index" / snapshot_b / "A0" / "build.json").read_bytes()
    before_identity = (data_root / "index" / snapshot_b / "A0" / "identity.json").read_bytes()
    subset_a = _write_docs(tmp_path / "snapshot-a-subset.jsonl", _rows()[:1])
    assert (
        _run(["--snapshot", snapshot_a, "--input", str(subset_a), "--config", "A0", "--fresh"]) == 0
    )

    assert store.count() == len(before_ids)
    assert _collection_ids(store, snapshot_b, "A0") == before_ids
    assert store.query([1.0] * DIM, top_k=1)[0].payload.snapshot_id == snapshot_b
    assert (data_root / "index" / snapshot_b / "A0" / "build.json").read_bytes() == before_build
    assert (data_root / "index" / snapshot_b / "A0" / "identity.json").read_bytes() == (
        before_identity
    )


def test_ambiguous_legacy_is_rejected_before_encoding_or_cache_writes(
    data_root: Path, docs_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot = "20261028-abc000000000"
    store = _store_at(data_root)
    legacy_name = "snap_20261028-abc_A0"
    legacy = store.client.create_collection(
        name=legacy_name,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"created_ts": 1},
    )
    model = FakeModel()
    assert (
        _run(
            ["--snapshot", snapshot, "--input", str(docs_path), "--config", "A0", "--fresh"],
            model=model,
        )
        == 2
    )
    assert "snapshot ownership check failed" in capsys.readouterr().err
    assert model.calls == []
    assert not (data_root / "cache" / "embed").exists()
    assert legacy.count() == 0
    assert {collection.name for collection in store.client.list_collections()} == {legacy_name}
    assert not (data_root / "index" / snapshot).exists()


def test_legacy_build_records_its_actual_collection_and_fresh_uses_full_name(
    data_root: Path, docs_path: Path
) -> None:
    snapshot = "20261028-abc000000000"
    store = _store_at(data_root)
    legacy_name = "snap_20261028-abc_A0"
    store.client.create_collection(
        name=legacy_name,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"snapshot_id": snapshot},
    )
    argv = ["--snapshot", snapshot, "--input", str(docs_path), "--config", "A0"]
    assert _run(argv) == 0
    sidecar = data_root / "index" / snapshot / "A0"
    assert json.loads((sidecar / "identity.json").read_text(encoding="utf-8"))["collection"] == (
        legacy_name
    )
    assert json.loads((sidecar / "build.json").read_text(encoding="utf-8"))["collection"] == (
        legacy_name
    )
    assert store.bind(snapshot, "A0").name == legacy_name

    assert _run([*argv, "--fresh"]) == 0
    assert store.bind(snapshot, "A0").name == collection_name(snapshot, "A0")
    assert json.loads((sidecar / "build.json").read_text(encoding="utf-8"))["collection"] == (
        collection_name(snapshot, "A0")
    )


def test_limit_caps_documents(
    data_root: Path, docs_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = _run(
        [
            "--snapshot",
            SNAP,
            "--input",
            str(docs_path),
            "--config",
            "A0",
            "--limit",
            "2",
            "--preview",
            "--preview-id",
            "limit-test",
        ]
    )
    assert code == 0
    line = _summary(capsys.readouterr().out, "A0")
    assert line.group(1) == "2"  # docs
    expected = _expected_chunks("A0", _rows()[:2])
    assert line.group(2) == str(expected)
    preview_root = data_root / "previews" / "limit-test"
    assert len(_collection_ids(_store_at(preview_root), SNAP, "A0")) == expected


def test_limited_fresh_without_preview_preserves_formal_index(
    data_root: Path, docs_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["--snapshot", SNAP, "--input", str(docs_path), "--config", "A0"]
    assert _run(argv) == 0
    before_ids = _collection_ids(_store_at(data_root), SNAP, "A0")
    build_path = data_root / "index" / SNAP / "A0" / "build.json"
    before_build = build_path.read_bytes()
    model = FakeModel()
    capsys.readouterr()

    assert _run([*argv, "--limit", "1", "--fresh"], model=model) == 2
    assert "--preview" in capsys.readouterr().err
    assert model.calls == []
    assert _collection_ids(_store_at(data_root), SNAP, "A0") == before_ids
    assert build_path.read_bytes() == before_build
    assert not (data_root / "previews").exists()


def test_preview_builds_are_separate_and_not_formal_evaluation_records(
    data_root: Path, docs_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from acgn_rag.core.eval.runner import load_build_records

    argv = ["--snapshot", SNAP, "--input", str(docs_path), "--config", "A0"]
    assert _run(argv) == 0
    before_ids = _collection_ids(_store_at(data_root), SNAP, "A0")
    build_path = data_root / "index" / SNAP / "A0" / "build.json"
    before_build = build_path.read_bytes()
    capsys.readouterr()
    for preview_id in ("trial-one", "trial-two"):
        assert (
            _run([*argv, "--limit", "1", "--fresh", "--preview", "--preview-id", preview_id]) == 0
        )
        preview_root = data_root / "previews" / preview_id
        preview_records = preview_root / "index" / SNAP / "A0"
        assert (preview_records / "preview-build.json").is_file()
        assert (preview_records / "preview-identity.json").is_file()
        assert not (preview_records / "build.json").exists()
        assert not (preview_records / "identity.json").exists()
        assert (preview_root / "cache" / "embed").is_dir()
        assert len(_collection_ids(_store_at(preview_root), SNAP, "A0")) == _expected_chunks(
            "A0", _rows()[:1]
        )
        with pytest.raises(FileNotFoundError):
            load_build_records(preview_root / "index", snapshot_id=SNAP, config="A0", label="A0")

    assert _collection_ids(_store_at(data_root), SNAP, "A0") == before_ids
    assert build_path.read_bytes() == before_build


@pytest.mark.parametrize("preview_id", ["../escape", "C:escape", ".", "CON", "trial."])
def test_preview_rejects_unsafe_ids_before_writes(
    data_root: Path, docs_path: Path, capsys: pytest.CaptureFixture[str], preview_id: str
) -> None:
    model = FakeModel()
    argv = ["--snapshot", SNAP, "--input", str(docs_path), "--config", "A0", "--limit", "1"]
    assert _run([*argv, "--preview", "--preview-id", preview_id], model=model) == 2
    assert "preview" in capsys.readouterr().err.lower()
    assert model.calls == []
    assert not (data_root / "index").exists()


@pytest.mark.parametrize("flags", [["--preview"], ["--preview-id", "trial"]])
def test_preview_requires_flag_and_id_together(
    data_root: Path, docs_path: Path, flags: list[str]
) -> None:
    model = FakeModel()
    argv = ["--snapshot", SNAP, "--input", str(docs_path), "--config", "A0", "--limit", "1"]
    assert _run([*argv, *flags], model=model) == 2
    assert model.calls == []
    assert not data_root.exists()


def test_preview_rejects_existing_id_and_external_cache_override(
    data_root: Path, docs_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["--snapshot", SNAP, "--input", str(docs_path), "--config", "A0", "--limit", "1"]
    preview_args = ["--preview", "--preview-id", "already-used"]
    assert _run([*argv, *preview_args]) == 0
    prior = data_root / "previews" / "already-used" / "index" / SNAP / "A0" / "preview-build.json"
    before = prior.read_bytes()
    model = FakeModel()
    assert _run([*argv, *preview_args, "--fresh"], model=model) == 2
    assert model.calls == [] and prior.read_bytes() == before
    capsys.readouterr()
    external = data_root / "cache" / "embed"
    assert (
        _run(
            [*argv, "--preview", "--preview-id", "other", "--embed-cache-dir", str(external)],
            model=model,
        )
        == 2
    )
    assert model.calls == []
    assert not external.exists()


def test_preview_rejects_symlinked_parent_without_touching_formal_data(
    data_root: Path, docs_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["--snapshot", SNAP, "--input", str(docs_path), "--config", "A0"]
    assert _run(argv) == 0
    before = (data_root / "index" / SNAP / "A0" / "build.json").read_bytes()
    try:
        (data_root / "previews").symlink_to(data_root / "index", target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symbolic links are unavailable: {exc}")
    model = FakeModel()
    assert _run([*argv, "--limit", "1", "--preview", "--preview-id", "trial"], model=model) == 2
    assert model.calls == []
    assert (data_root / "index" / SNAP / "A0" / "build.json").read_bytes() == before


def test_existing_preview_with_hard_link_is_never_reused(data_root: Path, docs_path: Path) -> None:
    argv = ["--snapshot", SNAP, "--input", str(docs_path), "--config", "A0"]
    assert _run(argv) == 0
    formal = data_root / "index" / SNAP / "A0" / "build.json"
    before = formal.read_bytes()
    preview = data_root / "previews" / "already-used"
    preview.mkdir(parents=True)
    try:
        (preview / "linked-record").hardlink_to(formal)
    except OSError as exc:
        pytest.skip(f"hard links are unavailable: {exc}")
    model = FakeModel()
    assert (
        _run(
            [*argv, "--limit", "1", "--fresh", "--preview", "--preview-id", "already-used"],
            model=model,
        )
        == 2
    )
    assert model.calls == [] and formal.read_bytes() == before


def test_preview_root_rejects_unsafe_id_even_when_called_directly(data_root: Path) -> None:
    from acgn_rag.core.preview import create_preview_root

    with pytest.raises(ValueError, match="preview ID"):
        create_preview_root(data_root, "../outside")
    assert not data_root.exists()


def test_failed_preview_reserves_its_id_and_does_not_create_formal_output(
    data_root: Path, docs_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_identity(_embedder: Any) -> None:
        raise RuntimeError("synthetic model initialization failure")

    monkeypatch.setattr(build_index.Embedder, "identity", fail_identity)
    argv = [
        "--snapshot",
        SNAP,
        "--input",
        str(docs_path),
        "--config",
        "A0",
        "--limit",
        "1",
        "--preview",
        "--preview-id",
        "failed-trial",
    ]
    assert _run(argv) == 2
    assert (data_root / "previews" / "failed-trial").is_dir()
    assert not (data_root / "index").exists()
    assert _run(argv) == 2
    assert not (data_root / "index").exists()


def test_config_flag_accepts_comma_lists_and_dedupes(
    data_root: Path, docs_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = _run(
        ["--snapshot", SNAP, "--input", str(docs_path), "--config", "A,A0", "--config", "A"]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert out.count("config=A label=A docs=") == 1  # the repeated A collapsed
    assert out.count("config=A0 label=A0 docs=") == 1
    assert "config=A'" not in out  # not requested, not built
    assert out.index("config=A label=A ") < out.index("config=A0 label=A0 ")  # order kept


def test_unknown_config_is_rejected(
    data_root: Path, docs_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = _run(["--snapshot", SNAP, "--input", str(docs_path), "--config", "B"])
    assert code == 2
    assert "unknown chunk config" in capsys.readouterr().err


def test_missing_input_is_reported_not_raised(
    data_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = _run(["--snapshot", SNAP, "--input", str(tmp_path / "nope.jsonl")])
    assert code == 2
    assert "NOT FOUND" in capsys.readouterr().err


def test_invalid_doc_row_aborts_with_line_number(
    data_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = _doc(1, 3) | {"source_type": "blog"}  # not a SourceType (ports.py)
    path = _write_docs(tmp_path / "bad.jsonl", [bad])
    code = _run(["--snapshot", SNAP, "--input", str(path)])
    assert code == 2
    error = capsys.readouterr().err
    assert f"{path}:1" in error and "invalid doc row" in error


def test_data_root_and_embed_cache_dir_overrides(
    tmp_path: Path,
    docs_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("ACGN_DATA_DIR", raising=False)  # the flag alone must locate everything
    data_root = tmp_path / "elsewhere"
    cache_dir = tmp_path / "vector-cache"
    code = _run(
        [
            "--snapshot",
            SNAP,
            "--input",
            str(docs_path),
            "--config",
            "A",
            "--data-root",
            str(data_root),
            "--embed-cache-dir",
            str(cache_dir),
        ]
    )
    assert code == 0
    build = json.loads(
        (data_root / "index" / SNAP / "A" / "build.json").read_text(encoding="utf-8")
    )
    assert build["snapshot_id"] == SNAP
    assert (data_root / "index" / "chroma").is_dir()
    assert len(list(cache_dir.rglob("*.npy"))) == _expected_chunks("A")
    assert not (tmp_path / "data").exists()  # nothing fell back to the ./data default
    assert "wrote" in capsys.readouterr().out


@pytest.mark.parametrize("limit", ["0", "-1"])
def test_nonpositive_limit_rejected_before_input_or_settings(
    docs_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    limit: str,
) -> None:
    def unexpected(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("invalid limit reached input or settings")

    monkeypatch.setattr(build_index, "_load_docs", unexpected)
    monkeypatch.setattr(build_index, "load_settings", unexpected)
    code = _run(["--snapshot", SNAP, "--input", str(docs_path), "--limit", limit])
    assert code == 2
    assert "--limit must be a positive integer" in capsys.readouterr().err


@pytest.mark.parametrize("limit", ["0", "-1"])
def test_nonpositive_limit_fresh_preserves_existing_index(
    data_root: Path, docs_path: Path, capsys: pytest.CaptureFixture[str], limit: str
) -> None:
    argv = ["--snapshot", SNAP, "--input", str(docs_path), "--config", "A0"]
    assert _run(argv) == 0
    before_ids = _collection_ids(_store_at(data_root), SNAP, "A0")
    records = data_root / "index" / SNAP / "A0"
    before_build = (records / "build.json").read_bytes()
    before_identity = (records / "identity.json").read_bytes()
    capsys.readouterr()

    model = FakeModel()
    assert _run([*argv, "--fresh", "--limit", limit], model=model) == 2
    assert "--limit must be a positive integer" in capsys.readouterr().err
    assert model.calls == []
    assert _collection_ids(_store_at(data_root), SNAP, "A0") == before_ids
    assert (records / "build.json").read_bytes() == before_build
    assert (records / "identity.json").read_bytes() == before_identity


def test_build_index_writes_multiple_batches_and_records_success(
    data_root: Path, docs_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chromadb.api.client import Client
    from chromadb.api.models.Collection import Collection

    from acgn_rag.core.eval.runner import load_build_records

    monkeypatch.setattr(Client, "get_max_batch_size", lambda _self: 2)
    upsert = Collection.upsert
    batch_lengths: list[int] = []

    def observe_upsert(collection: Any, *args: Any, **kwargs: Any) -> Any:
        batch_lengths.append(len(kwargs["ids"]))
        return upsert(collection, *args, **kwargs)

    monkeypatch.setattr(Collection, "upsert", observe_upsert)
    args = ["--snapshot", SNAP, "--input", str(docs_path), "--config", "A0"]
    assert _run(args) == 0

    assert batch_lengths == [2, 2, 2]
    assert not (data_root / "index" / SNAP / "A0" / "build.incomplete").exists()
    _, _, counts, _ = load_build_records(
        data_root / "index", snapshot_id=SNAP, config="A0", label="A0"
    )
    assert counts["stored"] == 6
    assert _store_at(data_root).bind_existing(SNAP, "A0").count() == 6


def test_interrupted_rebuild_is_unreadable_until_explicit_fresh_recovery(
    data_root: Path, docs_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from acgn_rag.core.eval.runner import load_build_records

    args = ["--snapshot", SNAP, "--input", str(docs_path), "--config", "A0"]
    assert _run(args) == 0
    original_upsert = ChromaStore.upsert_chunks

    def fail_after_one_row(
        store: ChromaStore, payloads: Sequence[Any], embeddings: Sequence[Any]
    ) -> None:
        original_upsert(store, payloads[:1], embeddings[:1])
        raise RuntimeError("synthetic interruption after a partial write")

    with monkeypatch.context() as patcher:
        patcher.setattr(ChromaStore, "upsert_chunks", fail_after_one_row)
        assert _run([*args, "--fresh"]) == 2

    marker = data_root / "index" / SNAP / "A0" / "build.incomplete"
    assert marker.exists()
    with pytest.raises(ValueError, match="incomplete"):
        load_build_records(data_root / "index", snapshot_id=SNAP, config="A0", label="A0")
    with pytest.raises(RuntimeError, match="incomplete"):
        _store_at(data_root).bind_existing(SNAP, "A0")

    assert _run(args) == 2
    assert _run([*args, "--fresh"]) == 2
    assert _run([*args, "--fresh", "--recover-incomplete"]) == 0
    assert not marker.exists()
    _, _, counts, _ = load_build_records(
        data_root / "index", snapshot_id=SNAP, config="A0", label="A0"
    )
    assert counts["stored"] == _expected_chunks("A0")
    assert _store_at(data_root).bind_existing(SNAP, "A0").count() == counts["stored"]


def test_failed_sidecar_write_leaves_the_build_unreadable(
    data_root: Path, docs_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from acgn_rag.core.eval.runner import load_build_records

    args = ["--snapshot", SNAP, "--input", str(docs_path), "--config", "A0"]
    assert _run(args) == 0
    write_json = build_index._atomic_write_json

    def fail_build_record(path: Path, payload: dict[str, object]) -> None:
        if path.name == "build.json":
            raise OSError("synthetic record write failure")
        write_json(path, payload)

    with monkeypatch.context() as patcher:
        patcher.setattr(build_index, "_atomic_write_json", fail_build_record)
        assert _run([*args, "--fresh"]) == 2

    assert (data_root / "index" / SNAP / "A0" / "build.incomplete").exists()
    with pytest.raises(ValueError, match="incomplete"):
        load_build_records(data_root / "index", snapshot_id=SNAP, config="A0", label="A0")
    assert _run([*args, "--fresh", "--recover-incomplete"]) == 0


def test_recovery_requires_fresh_before_any_write(data_root: Path, docs_path: Path) -> None:
    args = ["--snapshot", SNAP, "--input", str(docs_path), "--config", "A0"]
    assert _run([*args, "--recover-incomplete"]) == 2
    assert not (data_root / "index").exists()


@pytest.mark.parametrize("selection", [[], ["--config", "A0"]], ids=["all-configs", "single"])
def test_incomplete_build_is_rejected_before_model_or_store_access(
    data_root: Path,
    docs_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    selection: list[str],
) -> None:
    marker = data_root / "index" / SNAP / "A0" / "build.incomplete"
    marker.parent.mkdir(parents=True)
    marker.write_text("interrupted\n", encoding="utf-8")

    calls: list[str] = []

    def unexpected_identity(_embedder: Any) -> None:
        calls.append("identity")
        raise AssertionError("the encoder must not be loaded")

    monkeypatch.setattr(build_index.Embedder, "identity", unexpected_identity)
    assert _run(["--snapshot", SNAP, "--input", str(docs_path), *selection]) == 2
    assert calls == []
    assert "incomplete" in capsys.readouterr().err
    assert not (data_root / "index" / "chroma").exists()


def test_invalid_snapshot_is_rejected_before_model_or_store_access(
    data_root: Path, docs_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def unexpected_identity(_embedder: Any) -> None:
        calls.append("identity")
        raise AssertionError("the encoder must not be loaded")

    monkeypatch.setattr(build_index.Embedder, "identity", unexpected_identity)
    assert _run(["--snapshot", "../other", "--input", str(docs_path), "--config", "A0"]) == 2
    assert calls == []
    assert not (data_root / "index").exists()


def test_interrupted_preview_stays_unreadable_and_preserves_formal_index(
    data_root: Path, docs_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from acgn_rag.core.eval.runner import load_build_records

    args = ["--snapshot", SNAP, "--input", str(docs_path), "--config", "A0"]
    assert _run(args) == 0
    before_ids = _collection_ids(_store_at(data_root), SNAP, "A0")
    formal_record = data_root / "index" / SNAP / "A0" / "build.json"
    before_record = formal_record.read_bytes()
    original_upsert = ChromaStore.upsert_chunks

    def fail_after_one_row(
        store: ChromaStore, payloads: Sequence[Any], embeddings: Sequence[Any]
    ) -> None:
        original_upsert(store, payloads[:1], embeddings[:1])
        raise RuntimeError("synthetic preview interruption")

    preview_args = [*args, "--limit", "1", "--fresh", "--preview", "--preview-id", "interrupted"]
    with monkeypatch.context() as patcher:
        patcher.setattr(ChromaStore, "upsert_chunks", fail_after_one_row)
        assert _run(preview_args) == 2
    preview_root = data_root / "previews" / "interrupted"
    assert (preview_root / "index" / SNAP / "A0" / "build.incomplete").exists()
    with pytest.raises(ValueError, match="incomplete"):
        load_build_records(preview_root / "index", snapshot_id=SNAP, config="A0", label="A0")
    with pytest.raises(RuntimeError, match="incomplete"):
        _store_at(preview_root).bind_existing(SNAP, "A0")
    assert _run(preview_args) == 2
    assert _collection_ids(_store_at(data_root), SNAP, "A0") == before_ids
    assert formal_record.read_bytes() == before_record
