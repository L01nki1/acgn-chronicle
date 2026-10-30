"""CLI: chunk + embed + build the vector index for one snapshot (RUNBOOK §2 SOP).

Usage:
    python scripts/build_index.py --snapshot <id> --input docs.jsonl
    python scripts/build_index.py --snapshot <id> --input docs.jsonl --config A --limit 100

Per chunk config (A / A' / A0 - arguments of one implementation, issue #8) the pipeline is
``chunk_text`` -> ``Embedder.encode_passages`` behind the identity-keyed ``EmbedCache`` ->
``ChromaStore.upsert_chunks``. The configs run SEQUENTIALLY on purpose (RUNBOOK §2 step 3:
running them in parallel oversubscribes the CPU and is slower overall); do not parallelize
the loop in ``main``.

Encoder identity (RUNBOOK §3.4): ``ChromaStore.bind`` records the full snapshot/config owner,
while a build records its encoder identity in
``<data-root>/index/<snapshot>/<label>/identity.json`` and ``build.json`` next to the store
(the store root is ``<data-root>/index/chroma``), and the output names both paths. Moving the
identity into the Chroma collection metadata needs a store API change and stays out of this
script. Collection names are Chroma-safe labels: ``A'`` is stored as ``A-prime`` (chromadb
names accept ``[a-zA-Z0-9._-]`` only), while chunk ids keep the variant name.

TODO(snapshot reader): A owns the snapshot/doc reader. Until that interface lands this script
reads a raw docs.jsonl via ``--input`` (rows: doc_id / title / publisher / published_at /
url / source_type / lang / media_ids / body, validated by ``IndexDoc``); replace ``_load_docs``
with the reader and drop ``--input`` when it exists. The RUNBOOK §1 row keeps ``--input``
until then (same pattern as scripts/extract_facts.py).

Model loading is the caller's concern - this script downloads nothing itself. Real runs need
the ``[embed]`` extra plus a local HF cache offline, e.g. ``HF_HUB_OFFLINE=1
HF_HOME=<local cache> python scripts/build_index.py ...``. Tests inject a fake tokenizer (the
chunk ``TokenCounter`` protocol) and a fake model and so load nothing; in a real run the
tokenizer is taken from the embedder's own model, because the budget and the packing must
never mix tokenizers (RUNBOOK §3.4).

Rebuilds upsert by default (idempotent for the same input), so rows of a larger earlier
build survive a smaller rebuild as stale entries (PR #24 review). ``--fresh`` replaces the
selected snapshot/config collections instead: each collection is deleted only after the
input has parsed and produced chunks, so an invalid or empty input never wipes a valid
index. A build over zero documents, or one whose documents produce zero chunks, is
rejected with exit code 2 and writes nothing.

Exit codes: 0 = done, 2 = invalid input, no usable chunks, or an unverified snapshot owner.

Owner: F | Spec: 02 §1, RUNBOOK §2 (build SOP), §3.4 (encoder identity), §3.5 (context headers)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pydantic import BaseModel, Field, ValidationError, field_validator

from acgn_rag.config import Paths, Settings, load_settings
from acgn_rag.core.chunk import CHUNK_VARIANTS, TokenCounter, chunk_text
from acgn_rag.core.embed import (
    EmbedCache,
    Embedder,
    EncoderIdentity,
    PassageInput,
    build_header,
    cache_key,
    model_spec,
)
from acgn_rag.core.ports import SourceType
from acgn_rag.core.store import ChromaStore
from acgn_rag.core.store.payload import ChunkPayload, iso_to_epoch, payload_from_fields

#: Default config order: every variant in CHUNK_VARIANTS (dict order = the declared order in
#: variants.py), i.e. A, A', A0.
DEFAULT_CONFIGS: tuple[str, ...] = tuple(CHUNK_VARIANTS)

#: chromadb collection names accept [a-zA-Z0-9._-] only (1.5.9 rejects "A'" outright), so
#: configs get filesystem-safe labels for collection names and sidecar directories; the
#: store reserves underscores as name separators and accepts A / A-prime / A0 / B as labels.
#: Chunk ids keep the variant name itself.
COLLECTION_LABELS: dict[str, str] = {"A'": "A-prime"}

#: e5-base context limit. The budget formula (RUNBOOK §3.4) must never hard-code 508: other
#: lanes differ (bge-m3 8k, EmbeddingGemma 2048) and must pass --max-seq-length.
DEFAULT_MAX_SEQ_LENGTH = 512


class BuildError(RuntimeError):
    """A rejected build: the input or collection ownership cannot safely produce an index.

    Empty input and the initial ownership check run before encoding, cache writes or deletion.
    """


class IndexDoc(BaseModel):
    """One input document for a build: a NormalizedItem-shaped docs.jsonl row (adapter seam).

    Required: doc_id / published_at / url / source_type / lang / body. The header fields
    title and publisher default to empty because ``build_header`` drops empty fields anyway;
    ``media_ids`` defaults to no media binding. ``published_at`` must parse as ISO 8601 - the
    check runs at load time, so a malformed stamp aborts with the input line in hand instead
    of failing mid-upsert inside ``payload_from_fields``.
    """

    doc_id: str = Field(min_length=1)
    title: str = ""
    publisher: str = ""
    published_at: str
    url: str = Field(min_length=1)
    source_type: SourceType
    lang: str = Field(min_length=1)
    media_ids: list[int] = Field(default_factory=list)
    body: str

    @field_validator("published_at")
    @classmethod
    def _check_iso8601(cls, value: str) -> str:
        iso_to_epoch(value)  # raises ValueError on a malformed stamp
        return value


class HFTokenCounter:
    """Adapt the embedder's HF tokenizer to the chunk ``TokenCounter`` protocol.

    Same contract as the test-side adapter in tests/test_chunk_a.py: ``ids`` keeps the
    model's special tokens (what max_seq_length pays for), ``count`` drops them (what the
    packing budget governs), and ``decode`` skips them so id slices decode to content only.
    """

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def count(self, text: str) -> int:
        return len(self._tokenizer(text, add_special_tokens=False).input_ids)

    def ids(self, text: str) -> list[int]:
        return self._tokenizer(text).input_ids

    def decode(self, ids: Sequence[int]) -> str:
        return self._tokenizer.decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )


@dataclass(frozen=True)
class _BuildContext:
    """Everything a per-config build needs; built once in ``main``, shared across configs."""

    snapshot_id: str
    settings: Settings
    embedder: Embedder
    tokenizer: TokenCounter
    identity: EncoderIdentity
    passage_prefix: str  # literal prefix text the model sees ("" unless the family is prefix mode)
    passage_side: str | None  # spec.passage: prefix text or prompt name, for cache_key()
    max_seq_length: int
    cache: EmbedCache
    store: ChromaStore
    fresh: bool  # replace each selected collection instead of upserting into it


def _load_docs(path: Path, limit: int | None) -> list[IndexDoc]:
    """Read doc rows from JSONL, stopping at ``limit``. A malformed line aborts the run."""
    docs: list[IndexDoc] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                docs.append(IndexDoc.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise ValueError(f"{path}:{lineno}: invalid doc row ({exc})") from exc
            if limit is not None and len(docs) >= limit:
                break
    return docs


def _parse_configs(raw: Sequence[str] | None) -> list[str]:
    """Resolve repeated and/or comma-separated --config values into an ordered, deduped list.

    Default: every variant in CHUNK_VARIANTS (A, A', A0). Unknown names raise ValueError -
    the chunker would reject them anyway, but failing before any encode work runs is the
    honest order.
    """
    if not raw:
        return list(DEFAULT_CONFIGS)
    configs: list[str] = []
    for part in raw:
        for name in part.split(","):
            name = name.strip()
            if name and name not in configs:
                configs.append(name)
    unknown = [name for name in configs if name not in CHUNK_VARIANTS]
    if unknown:
        known = ", ".join(repr(name) for name in CHUNK_VARIANTS)
        raise ValueError(f"unknown chunk config(s) {unknown}; known configs: {known}")
    return configs


def _collection_label(config: str) -> str:
    """Chroma-safe label for a config (see COLLECTION_LABELS); chunk ids keep the variant."""
    return COLLECTION_LABELS.get(config, config)


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    """Write JSON through a temp file + ``os.replace`` (RUNBOOK §3.2 atomicity).

    A reader either sees the previous file or the complete new one, never a half-written one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _build_config(ctx: _BuildContext, config: str, docs: Sequence[IndexDoc]) -> dict[str, object]:
    """Chunk, embed (cache-first) and upsert one config's collection; return the build record.

    Every chunk is looked up in the cache exactly once before anything is encoded, so the
    reported hits/misses count chunks; a build over a warm cache encodes nothing. All misses
    go through ONE ``encode_passages`` call (the model batches internally via
    EmbedConfig.batch_size).
    """
    started = time.perf_counter()
    label = _collection_label(config)
    inputs: list[PassageInput] = []
    keys: list[str] = []
    payloads: list[ChunkPayload] = []
    for doc in docs:
        for chunk in chunk_text(
            doc.doc_id, doc.body, config, ctx.tokenizer, ctx.max_seq_length, ctx.passage_prefix
        ):
            header = build_header(doc.title, doc.publisher, doc.published_at)
            inputs.append(
                PassageInput(
                    title=doc.title,
                    publisher=doc.publisher,
                    published_at=doc.published_at,
                    body=chunk.text,
                    doc_id=doc.doc_id,
                    chunk_id=chunk.chunk_id,
                )
            )
            keys.append(
                cache_key(
                    ctx.identity,
                    ctx.settings.embed.model_name,
                    ctx.passage_side,
                    header,
                    chunk.text,
                )
            )
            payloads.append(
                payload_from_fields(
                    chunk_id=chunk.chunk_id,
                    doc_id=doc.doc_id,
                    snapshot_id=ctx.snapshot_id,
                    title=doc.title,
                    url=doc.url,
                    publisher=doc.publisher,
                    source_type=doc.source_type,
                    published_at=doc.published_at,
                    lang=doc.lang,
                    body=chunk.text,
                    media_ids=doc.media_ids,
                )
            )

    if not payloads:
        # Non-empty documents can still yield zero chunks (empty or blank bodies): a build
        # must never report success over nothing (PR #24 review, G1a "items > 0" principle).
        raise BuildError(
            f"the {len(docs)} input document(s) produced no chunks for config {config!r} "
            "(empty or blank bodies?): nothing was encoded, written or deleted"
        )

    try:
        ctx.store.validate_binding(ctx.snapshot_id, label)
    except (RuntimeError, ValueError) as exc:
        raise BuildError(f"snapshot ownership check failed: {exc}") from exc

    before = ctx.cache.stats()
    vectors: list[Any] = []
    misses: list[int] = []
    for index, key in enumerate(keys):
        vector = ctx.cache.get(key)  # exactly one lookup per chunk: stats count chunks
        if vector is None:
            misses.append(index)
        vectors.append(vector)
    if misses:
        encoded = ctx.embedder.encode_passages([inputs[index] for index in misses])
        for position, index in enumerate(misses):
            ctx.cache.put(keys[index], encoded[position])
            vectors[index] = encoded[position]
    after = ctx.cache.stats()

    try:
        if ctx.fresh:
            # Recheck ownership before deletion: preflight cannot prevent an intervening change.
            ctx.store.drop_collection(ctx.snapshot_id, label)
        collection = ctx.store.bind(ctx.snapshot_id, label)
    except (RuntimeError, ValueError) as exc:
        raise BuildError(f"snapshot ownership check failed: {exc}") from exc
    # chromadb 1.5.9 accepts only Python scalars (its type check rejects np.float32), so the
    # numpy rows coming from the encoder or the cache are flattened with .tolist() here;
    # upsert_chunks' own `list(embedding)` keeps numpy scalars and would be rejected.
    ctx.store.upsert_chunks(payloads, [vector.tolist() for vector in vectors])
    stored = ctx.store.count()

    record: dict[str, object] = {
        "snapshot_id": ctx.snapshot_id,
        "config": config,
        "label": label,
        "collection": collection.name,
        "mode": "replace" if ctx.fresh else "upsert",
        "docs": len(docs),
        "chunks": len(payloads),
        "cache_hits": after["hits"] - before["hits"],
        "cache_misses": after["misses"] - before["misses"],
        "encodes": len(misses),
        "stored": stored,
        "max_seq_length": ctx.max_seq_length,
        "passage_prefix": ctx.passage_prefix,
        "identity": ctx.identity.to_metadata(),
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "seconds": round(time.perf_counter() - started, 3),
    }
    out_dir = ctx.settings.paths.index / ctx.snapshot_id / label
    _atomic_write_json(
        out_dir / "identity.json",
        {
            "snapshot_id": ctx.snapshot_id,
            "config": config,
            "label": label,
            "collection": record["collection"],
            "identity": ctx.identity.to_metadata(),
        },
    )
    _atomic_write_json(out_dir / "build.json", record)
    return record


def main(
    argv: list[str] | None = None,
    *,
    tokenizer: TokenCounter | None = None,
    model: Any | None = None,
) -> int:
    """Run the build. ``tokenizer``/``model`` are injectable so tests run fully offline."""
    parser = argparse.ArgumentParser(
        description="Chunk + embed + build the index for one snapshot (RUNBOOK §2)"
    )
    parser.add_argument("--snapshot", required=True, help="snapshot id the docs belong to")
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "docs.jsonl rows (doc_id/title/publisher/published_at/url/source_type/lang/"
            "media_ids/body); required until A's snapshot reader lands (module TODO)"
        ),
    )
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        metavar="CONFIG",
        help=(
            "chunk config A, A' or A0; repeat the flag or comma-separate several "
            f"(default: {','.join(DEFAULT_CONFIGS)}, run sequentially per RUNBOOK §2)"
        ),
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="data root holding index/ and cache/; default ACGN_DATA_DIR or ./data",
    )
    parser.add_argument(
        "--embed-cache-dir",
        default=None,
        help="identity-keyed vector cache dir; default <data-root>/cache/embed",
    )
    parser.add_argument("--limit", type=int, default=None, help="build from at most N documents")
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=DEFAULT_MAX_SEQ_LENGTH,
        help="encoder context limit behind the chunk budget (RUNBOOK §3.4); default 512 (e5)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "replace each selected snapshot/config collection instead of upserting into it "
            "(a rebuild over a smaller input would otherwise leave stale rows); each "
            "collection is deleted only after the input parses and produces chunks"
        ),
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"NOT FOUND: {input_path} - pass --input <docs.jsonl>", file=sys.stderr)
        return 2
    try:
        docs = _load_docs(input_path, args.limit)
        configs = _parse_configs(args.config)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not docs:
        print(
            f"EMPTY INPUT: {input_path} holds no document rows; a build must produce at "
            "least one chunk - nothing was read into the store, written or deleted",
            file=sys.stderr,
        )
        return 2

    settings = load_settings()
    if args.data_root:
        settings = replace(settings, paths=Paths(root=Path(args.data_root)))

    embedder = Embedder(settings.embed, model=model)
    try:
        identity = embedder.identity()  # loads the real model when none is injected
    except ModuleNotFoundError as exc:
        print(
            f"cannot load the encoder ({exc}): real runs need the [embed] extra and a local "
            "HF cache (HF_HUB_OFFLINE=1); tests inject a fake model",
            file=sys.stderr,
        )
        return 2
    except Exception as exc:  # missing cache / offline HF failures: report, do not traceback
        print(
            f"cannot load the encoder ({exc}) - model loading is the caller's concern; see "
            "the module docstring for the offline invocation",
            file=sys.stderr,
        )
        return 2
    if tokenizer is None:
        # The tokenizer must come from the SAME model that encodes: mixing tokenizers between
        # the budget and the packing is the classic silent-truncation bug (RUNBOOK §3.4).
        try:
            tokenizer = HFTokenCounter(embedder.tokenizer)
        except AttributeError as exc:
            print(
                f"cannot derive a tokenizer from the model ({exc}): inject a tokenizer "
                "together with a model, or run on a real [embed] install",
                file=sys.stderr,
            )
            return 2

    spec = model_spec(settings.embed.model_name)
    ctx = _BuildContext(
        snapshot_id=args.snapshot,
        settings=settings,
        embedder=embedder,
        tokenizer=tokenizer,
        identity=identity,
        # prompt names are not literal prefixes: only prefix-mode families see text here
        passage_prefix=spec.passage if spec.mode == "prefix" else "",
        passage_side=spec.passage,
        max_seq_length=args.max_seq_length,
        cache=EmbedCache(
            Path(args.embed_cache_dir)
            if args.embed_cache_dir
            else settings.paths.root / "cache" / "embed"
        ),
        store=ChromaStore(settings.paths.index / "chroma", settings.store),
        fresh=args.fresh,
    )

    # RUNBOOK §2 step 3: the configs run SEQUENTIALLY - running them in parallel
    # oversubscribes the CPU and is slower overall. Do not parallelize this loop.
    try:
        for config in configs:
            record = _build_config(ctx, config, docs)
            print(
                f"config={record['config']} label={record['label']} docs={record['docs']} "
                f"chunks={record['chunks']} cache_hits={record['cache_hits']} "
                f"cache_misses={record['cache_misses']} encodes={record['encodes']} "
                f"stored={record['stored']} seconds={record['seconds']:.3f} "
                f"collection={record['collection']} mode={record['mode']}"
            )
            out_dir = settings.paths.index / args.snapshot / record["label"]
            print(f"wrote {out_dir / 'identity.json'}")
            print(f"wrote {out_dir / 'build.json'}")
    except BuildError as exc:
        print(f"BUILD REJECTED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
