"""WP5 evaluation runner: one run bound to one snapshot and one chunk config.

Owner: E | Spec: `docs/EVAL_PROTOCOL.md` §1/§7/§8, issue #7

The runner is deliberately injectable: `search` consumes the merged store/embedder, and the
answerer is a protocol. Until B's `/ask` lands, `NullAnswerer` records every answer as
`not_run`; the retrieval half (hit rate, evidence recall, MRR) is still measured and archived.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from acgn_rag.core.eval import metrics, qa
from acgn_rag.core.ports import (
    GroundedAnswer,
    RetrievalFilters,
    RetrievalQuery,
    RetrievedChunk,
)
from acgn_rag.core.retrieve.vector import search
from acgn_rag.core.store.build_state import assert_no_incomplete_build

CONFIG_TOKENS = {"A'": "A-prime"}
BUILD_IDENTITY_KEYS = ("model", "backend", "device", "precision", "batch_size", "identity_slug")
VECTOR_IDENTITY_KEYS = ("model", "backend", "device", "precision", "identity_slug")
CHUNKER_FIELDS = ("version", "header_version", "config", "max_tokens", "overlap", "max_seq_length")
_FULL_SNAPSHOT_ID = re.compile(r"(\d{8})-([0-9a-f]{12})")


class Answerer(Protocol):
    """Generation adapter; B's `/ask` or a test double."""

    def answer(
        self,
        query: RetrievalQuery,
        chunks: Sequence[RetrievedChunk],
    ) -> GroundedAnswer | None: ...


class DiagnosticsAnswerer(Protocol):
    """Optional extension: an answerer that also returns non-wire run diagnostics."""

    def answer_with_diagnostics(
        self,
        query: RetrievalQuery,
        chunks: Sequence[RetrievedChunk],
    ) -> Any: ...


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
    window_verified: bool = False
    top_k: int = 5
    over_fetch: int = 5
    run_id: str | None = None
    run_date: str | None = None
    git_sha: str = "unknown"
    build_identity: Mapping[str, Any] = field(default_factory=dict)
    collection: str | None = None
    counts: Mapping[str, int] = field(default_factory=dict)
    chunker: Mapping[str, Any] = field(default_factory=dict)
    llm: Mapping[str, Any] = field(default_factory=dict)
    determinism: Mapping[str, Any] = field(default_factory=dict)
    require_gpu: bool = False
    qa_validated: bool = False
    qa_source: Literal["cli", "caller", "runner"] = "caller"


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


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def load_build_records(
    index_dir: Path,
    *,
    snapshot_id: str,
    config: str,
    label: str,
) -> tuple[dict[str, Any], str, dict[str, int], dict[str, Any]]:
    """Read and cross-validate `identity.json` + `build.json` for one run.

    `build.json` is written last by `build_index.py`, so it is the build-success marker. Both
    records must name the same full snapshot id, config, label, collection and encoder identity;
    a partial build (identity only, zero chunks or zero stored rows) is rejected before `bind`.

    The returned counts (`docs`, `chunks`, `stored`) are provenance for `metrics.json`: the
    deduplicated documents the index covers and the chunks it holds. They are never divisors.
    """
    base = Path(index_dir) / snapshot_id / label
    assert_no_incomplete_build(index_dir, snapshot_id, label)
    identity_payload = _read_json(base / "identity.json")
    build_payload = _read_json(base / "build.json")

    for name, payload in (("identity.json", identity_payload), ("build.json", build_payload)):
        if payload.get("snapshot_id") != snapshot_id:
            raise ValueError(
                f"{name}: snapshot_id {payload.get('snapshot_id')!r} != {snapshot_id!r}"
            )
        if payload.get("config") != config:
            raise ValueError(f"{name}: config {payload.get('config')!r} != {config!r}")
        if payload.get("label") != label:
            raise ValueError(f"{name}: label {payload.get('label')!r} != {label!r}")

    collection = identity_payload.get("collection")
    if not isinstance(collection, str) or not collection:
        raise ValueError("identity.json: missing or invalid `collection`")
    if build_payload.get("collection") != collection:
        raise ValueError(
            "build.json: collection does not match identity.json "
            f"({build_payload.get('collection')!r} != {collection!r})"
        )

    identity = identity_payload.get("identity")
    if not isinstance(identity, dict) or not identity:
        raise ValueError("identity.json: missing or invalid `identity`")
    if build_payload.get("identity") != identity:
        raise ValueError("build.json: encoder identity does not match identity.json")
    missing = [key for key in BUILD_IDENTITY_KEYS if key not in identity]
    if missing:
        raise ValueError(f"identity.json: identity is missing {', '.join(missing)}")

    chunks = build_payload.get("chunks")
    stored = build_payload.get("stored")
    docs = build_payload.get("docs")
    if not isinstance(chunks, int) or chunks <= 0:
        raise ValueError(f"build.json: no successful chunks (chunks={chunks!r})")
    if not isinstance(stored, int) or stored <= 0:
        raise ValueError(f"build.json: no stored rows (stored={stored!r})")
    if not isinstance(docs, int) or docs <= 0:
        raise ValueError(f"build.json: no indexed documents (docs={docs!r})")

    chunker = build_payload.get("chunker")
    identity_chunker = identity_payload.get("chunker")
    if not isinstance(chunker, dict) or not chunker:
        raise ValueError("build.json: missing or invalid `chunker` identity")
    if not isinstance(identity_chunker, dict) or not identity_chunker:
        raise ValueError("identity.json: missing or invalid `chunker` identity")
    if chunker != identity_chunker:
        raise ValueError("build.json: chunker identity does not match identity.json")
    _validate_chunker(chunker, config)

    assert_no_incomplete_build(index_dir, snapshot_id, label)
    counts = {"docs": docs, "chunks": chunks, "stored": stored}
    return dict(identity), collection, counts, dict(chunker)


def _embed_metadata(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the build identity to the `EVAL_PROTOCOL` §7 `embed` shape."""
    versions = {
        key.removeprefix("lib_"): value for key, value in identity.items() if key.startswith("lib_")
    }
    return {
        "model": identity.get("model"),
        "backend": identity.get("backend"),
        "device": identity.get("device"),
        "precision": identity.get("precision"),
        "library_versions": versions,
        "batch": identity.get("batch_size"),
    }


_GPU_DEVICE_PREFIX = "cuda"


def _is_reportable_device(build_identity: Mapping[str, Any]) -> bool:
    """Return True when the recorded encoder device is a GPU lane.

    PR #48 proposes GPU-first encoding; the policy is not merged yet, so this check is only
    enforced when `RunSpec.require_gpu` is set. `mps` is deliberately not accepted until D
    defines whether it is a reportable lane.
    """
    device = str(build_identity.get("device") or "").strip().lower()
    return device == _GPU_DEVICE_PREFIX or device.startswith(f"{_GPU_DEVICE_PREFIX}:")


def _reportability_reason(
    *,
    evidence_resolver: object | None,
    window_verified: bool,
    build_identity: Mapping[str, Any],
    require_gpu: bool,
    qa_validated: bool,
    has_reportable_metrics: bool,
) -> str | None:
    """Return why a run is not reportable, or None when it is reportable."""
    if not qa_validated:
        return "qa_validation_bypassed"
    if evidence_resolver is None:
        return "no_evidence_resolver"
    if not window_verified:
        return "window_unverified"
    if require_gpu and not _is_reportable_device(build_identity):
        return "non_gpu_device"
    if not has_reportable_metrics:
        return "no_reportable_metrics"
    return None


def _library_keys(*identities: Mapping[str, Any]) -> set[str]:
    keys: set[str] = set()
    for identity in identities:
        keys.update(key for key in identity if isinstance(key, str) and key.startswith("lib_"))
    return keys


def verify_build_identity(
    build_identity: Mapping[str, Any],
    current_identity: Mapping[str, Any],
) -> list[str]:
    """Return vector-affecting identity fields that differ from the index.

    Model, backend, device, precision, the encoder slug and every recorded library version
    can change the vector. `batch_size` is run metadata only and is not a mismatch.
    """
    problems: list[str] = []
    for key in VECTOR_IDENTITY_KEYS:
        if key not in build_identity:
            problems.append(f"build identity missing {key!r}")
            continue
        if str(build_identity[key]) != str(current_identity.get(key)):
            problems.append(
                f"{key}: build={build_identity[key]!r}, current={current_identity.get(key)!r}"
            )
    for key in sorted(_library_keys(build_identity, current_identity)):
        if key not in build_identity:
            problems.append(f"build identity missing {key!r}")
            continue
        if str(build_identity[key]) != str(current_identity.get(key)):
            problems.append(
                f"{key}: build={build_identity[key]!r}, current={current_identity.get(key)!r}"
            )
    return problems


def _collection_name(value: Any) -> str:
    return str(getattr(value, "name", value))


def bind_existing_collection(
    store: Any,
    *,
    snapshot_id: str,
    chunk_config: str,
    collection: str,
) -> Any:
    """Bind the sidecar-named collection without creating one.

    `bind_existing` is the store's no-create read API: it raises `LookupError` when no
    verified full or legacy collection exists and never initializes missing storage. The
    sidecar is the authority, so the returned physical name must match it. No fallback to
    the build-side `bind()`, which may create a collection.
    """
    binder = getattr(store, "bind_existing", None)
    if binder is None:
        raise RuntimeError("store does not expose bind_existing for read binding")
    bound = binder(snapshot_id, chunk_config)
    bound_name = _collection_name(bound)
    if bound_name != collection:
        raise RuntimeError(
            f"bound collection {bound_name!r} does not match sidecar collection {collection!r}"
        )
    return bound


def _validate_chunker(chunker: Mapping[str, Any], config: str) -> None:
    """Reject a matching but incomplete chunker block, or one for another config."""
    missing = [name for name in CHUNKER_FIELDS if name not in chunker]
    if missing:
        raise ValueError(f"build.json: chunker identity is missing {', '.join(missing)}")
    for name in ("version", "header_version", "config"):
        if not isinstance(chunker[name], str) or not chunker[name]:
            raise ValueError(f"build.json: invalid chunker {name}")
    for name in ("max_tokens", "max_seq_length"):
        value = chunker[name]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"build.json: invalid chunker {name}")
    overlap = chunker["overlap"]
    if not isinstance(overlap, int) or isinstance(overlap, bool) or overlap < 0:
        raise ValueError("build.json: invalid chunker overlap")
    if chunker["config"] != config:
        raise ValueError(
            f"build.json: chunker config {chunker['config']!r} != selected config {config!r}"
        )


def _canonical_manifest_digest(manifest: Mapping[str, Any]) -> str | None:
    """A's canonical manifest hash, or None when the manifest has no item rows."""
    items = manifest.get("items")
    if not isinstance(items, list):
        return None
    if not items:
        return None
    rows: list[dict[str, Any]] = []
    for row in items:
        if not isinstance(row, dict) or not isinstance(row.get("doc_id"), str):
            raise ValueError("manifest items are not hashable identity rows")
        rows.append(row)
    payload = dict(manifest)
    payload["items"] = sorted(rows, key=lambda row: row["doc_id"])
    try:
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
    except (TypeError, ValueError) as exc:
        raise ValueError("manifest cannot be canonicalized for identity verification") from exc
    return hashlib.sha256(data).hexdigest()


def build_evidence_resolver(
    collection: Any,
) -> Callable[[Sequence[str]], Mapping[str, Sequence[str]]]:
    """Resolve gold URLs to chunk ids from the bound collection's metadata.

    The resolver is scoped to the collection the runner will query: a URL present in a
    different snapshot/config must stay unresolved rather than borrowing evidence.
    """
    rows = collection.get(include=["metadatas"])
    ids = list(rows.get("ids") or [])
    metadatas = list(rows.get("metadatas") or [])
    if len(ids) != len(metadatas):
        raise RuntimeError("collection metadata rows are incomplete")
    by_url: dict[str, list[str]] = {}
    for chunk_id, metadata in zip(ids, metadatas, strict=True):
        if not isinstance(metadata, dict):
            continue
        url = metadata.get("url")
        if isinstance(url, str) and url:
            by_url.setdefault(metrics.normalize_url(url), []).append(str(chunk_id))

    def resolve(urls: Sequence[str]) -> Mapping[str, Sequence[str]]:
        return {str(url): list(by_url.get(metrics.normalize_url(str(url)), [])) for url in urls}

    return resolve


def load_snapshot_member_urls(snapshot_dir: Path) -> frozenset[str]:
    """Return every member URL recorded in the frozen manifest.

    Configs A/A'/A0 index one representative per cluster, so a gold URL can be present in
    the snapshot while absent from the bound collection. The run uses this set to tell
    "absent from the snapshot" apart from "member not indexed".
    """
    manifest_path = Path(snapshot_dir) / "manifest.json"
    if not manifest_path.exists():
        return frozenset()
    manifest = _read_json(manifest_path)
    items = manifest.get("items")
    if not isinstance(items, list):
        return frozenset()
    urls: set[str] = set()
    for row in items:
        if not isinstance(row, dict):
            continue
        url = row.get("url")
        if isinstance(url, str) and url:
            urls.add(metrics.normalize_url(url))
    return frozenset(urls)


def resolve_window(
    snapshot_dir: Path,
    snapshot_id: str,
    window_start: str | None,
    window_end: str | None,
) -> tuple[str, str, bool]:
    """Resolve the corpus window from the frozen manifest, or require explicit bounds.

    The evaluation corpus window is a property of the frozen snapshot (A's manifest), not
    of the QA items' per-question validity windows. If the manifest is absent, both bounds
    must be supplied explicitly and the run is marked unverified.
    """
    manifest_path = Path(snapshot_dir) / "manifest.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        start = manifest.get("window_start")
        end = manifest.get("window_end")
        if not isinstance(start, str) or not isinstance(end, str):
            raise ValueError(f"snapshot {snapshot_id!r} manifest has no window_start/window_end")
        if window_start is not None and window_start != start:
            raise ValueError(f"--window-start {window_start!r} conflicts with manifest {start!r}")
        if window_end is not None and window_end != end:
            raise ValueError(f"--window-end {window_end!r} conflicts with manifest {end!r}")
        if start > end:
            raise ValueError(f"snapshot {snapshot_id!r} manifest window is reversed")

        match = _FULL_SNAPSHOT_ID.fullmatch(snapshot_id)
        if match is None:
            return start, end, False
        date, digest = match.groups()
        if end.replace("-", "") != date:
            raise ValueError(
                f"snapshot {snapshot_id!r} manifest window_end {end!r} does not match its date"
            )
        recorded = manifest.get("snapshot_id")
        if recorded is not None and recorded != snapshot_id:
            raise ValueError(f"manifest snapshot_id {recorded!r} != requested {snapshot_id!r}")
        computed = _canonical_manifest_digest(manifest)
        if computed is None:
            return start, end, False
        if computed[:12] != digest:
            raise ValueError(f"manifest content does not match snapshot {snapshot_id!r}")
        return start, end, True

    if window_start is None or window_end is None:
        raise ValueError(
            f"snapshot {snapshot_id!r} has no frozen manifest; supply both "
            "--window-start and --window-end (the run is marked unverified)"
        )
    if window_start > window_end:
        raise ValueError("--window-start must not be later than --window-end")
    return window_start, window_end, False


def _validate_spec(spec: RunSpec) -> None:
    missing = [
        name
        for name in ("snapshot_id", "window_start", "window_end", "config")
        if not getattr(spec, name)
    ]
    if not spec.build_identity:
        missing.append("build_identity")
    if not spec.chunker:
        missing.append("chunker")
    if missing:
        raise ValueError(f"run spec is missing {', '.join(missing)}")


def _load_items(
    spec: RunSpec, qa_items: Sequence[Mapping[str, Any]] | None
) -> list[dict[str, Any]]:
    """Load QA items from the file path or a caller-supplied sequence.

    The file path is validated here and archived as `source="runner"` with
    `validated=True`. Caller-supplied items are accepted as-is only when
    `RunSpec.qa_validated` is explicitly True, which is the caller's attestation that
    `qa.validate` passed; the default is False and archives the run as bypassed and
    non-reportable. `run_eval.py` sets the flag only after its own validation.
    """
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
    diagnostics: Mapping[str, Any] | None = None,
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
        "diagnostics": diagnostics,
    }


def run(
    spec: RunSpec,
    *,
    store: Any,
    embedder: Any,
    answerer: Answerer | None = None,
    qa_items: Sequence[Mapping[str, Any]] | None = None,
    evidence_resolver: Callable[[Sequence[str]], Mapping[str, Sequence[str]]] | None = None,
    snapshot_member_urls: Collection[str] | None = None,
) -> RunResult:
    """Execute one run and write the archive under `out_root/<run_id>/`.

    `snapshot_member_urls` is the full member URL set from the frozen manifest. A gold URL
    present there but absent from the bound collection is recorded as `member_not_indexed`,
    not `gold_unavailable`; the alias policy is pending the E/F/D rule.
    """
    _validate_spec(spec)
    items = _load_items(spec, qa_items)
    if qa_items is None:
        # The runner loaded and validated the file path itself.
        effective_qa_validated = True
        effective_qa_source: Literal["cli", "caller", "runner"] = "runner"
    else:
        effective_qa_validated = spec.qa_validated
        effective_qa_source = spec.qa_source
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
    member_urls = frozenset(metrics.normalize_url(str(url)) for url in (snapshot_member_urls or ()))
    if evidence_resolver is not None:
        member_not_indexed = [
            url
            for url in all_gold
            if not resolved.get(url) and metrics.normalize_url(url) in member_urls
        ]
        unresolved = [
            url
            for url in all_gold
            if not resolved.get(url) and metrics.normalize_url(url) not in member_urls
        ]
    else:
        member_not_indexed = []
        unresolved = list(all_gold)

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
        answer: GroundedAnswer | None = None
        diagnostics: dict[str, Any] | None = None
        if answerer is not None:
            if hasattr(answerer, "answer_with_diagnostics"):
                answer_run = answerer.answer_with_diagnostics(query, chunks)
                answer = answer_run.answer
                raw_diagnostics = getattr(answer_run, "diagnostics", None)
                if raw_diagnostics is not None:
                    diagnostics = (
                        asdict(raw_diagnostics)
                        if is_dataclass(raw_diagnostics)
                        else dict(raw_diagnostics)
                    )
            else:
                answer = answerer.answer(query, chunks)
        answers.append(_answer_record(item, chunks, answer, diagnostics))
        gold_resolved = any(resolved.get(url) for url in gold_urls)
        gold_member_not_indexed = (
            bool(gold_urls)
            and evidence_resolver is not None
            and not gold_resolved
            and any(metrics.normalize_url(url) in member_urls for url in gold_urls)
        )
        gold_unavailable = (
            bool(gold_urls)
            and evidence_resolver is not None
            and not gold_resolved
            and not gold_member_not_indexed
        )
        cases.append(
            metrics.CaseResult(
                qa_id=str(item["qa_id"]),
                lang=str(item["lang"]),
                answerable=bool(item["answerable"]),
                gold_urls=gold_urls,
                retrieved_urls=tuple(chunk.citation.url for chunk in chunks),
                retrieved_chunk_ids=tuple(chunk.chunk_id for chunk in chunks),
                cited_chunk_ids=tuple(answer.cited_chunk_ids) if answer else (),
                gold_unavailable=gold_unavailable,
                gold_member_not_indexed=gold_member_not_indexed,
                gold_unresolved=bool(gold_urls) and evidence_resolver is None,
                dropped=bool(item.get("dropped", False)),
                calibration=bool(item.get("calibration", False)),
                answer_label=None,
                abstained=answer.abstained if answer else None,
            )
        )

    metrics_payload = metrics.aggregate(
        cases,
        k=spec.top_k,
        deduped_count=spec.counts.get("docs"),
        indexed_chunks=spec.counts.get("stored"),
        provisional=evidence_resolver is None,
    )
    has_reportable_metrics = any(
        bool(block["available"]) for block in metrics_payload["reporting"].values()
    )
    non_reportable_reason = _reportability_reason(
        evidence_resolver=evidence_resolver,
        window_verified=spec.window_verified,
        build_identity=spec.build_identity,
        require_gpu=spec.require_gpu,
        qa_validated=effective_qa_validated,
        has_reportable_metrics=has_reportable_metrics,
    )
    meta = {
        "run_id": run_id,
        "snapshot_id": spec.snapshot_id,
        "window_start": spec.window_start,
        "window_end": spec.window_end,
        "window_verified": spec.window_verified,
        "chunk_config": spec.config,
        "retrieval": {
            "mode": "vector",
            "top_k": spec.top_k,
            "over_fetch": spec.over_fetch,
        },
        "embed": _embed_metadata(spec.build_identity),
        "build_identity": dict(spec.build_identity),
        "chunker": dict(spec.chunker),
        "reportable": non_reportable_reason is None,
        "non_reportable_reason": non_reportable_reason,
        "reportable_scope": "run; per-family availability is in metrics.reporting",
        "determinism": dict(spec.determinism),
        "gpu_policy": "pr48_require_gpu" if spec.require_gpu else "not_enforced",
        "require_gpu": spec.require_gpu,
        "collection": spec.collection,
        "llm": dict(spec.llm),
        "git_sha": spec.git_sha,
        "qa": {
            "path": str(spec.qa_path),
            "count": len(items),
            "validated": effective_qa_validated,
            "validation": "validated" if effective_qa_validated else "bypassed",
            "source": effective_qa_source,
        },
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
            "collection": spec.collection,
            "build_identity": dict(spec.build_identity),
            "resolved": resolved,
            "unresolved": unresolved,
            "member_not_indexed": member_not_indexed,
            "status": "resolved" if evidence_resolver is not None else "not_configured",
        },
    )
    (out_dir / "disputes.md").write_text(
        "# Disputes\n\nNo adjudications recorded for this run.\n",
        encoding="utf-8",
    )
    return RunResult(run_id=run_id, out_dir=out_dir, meta=meta, metrics=metrics_payload)


__all__ = [
    "BUILD_IDENTITY_KEYS",
    "CONFIG_TOKENS",
    "Answerer",
    "DiagnosticsAnswerer",
    "NullAnswerer",
    "RunResult",
    "RunSpec",
    "bind_existing_collection",
    "build_evidence_resolver",
    "load_build_records",
    "load_snapshot_member_urls",
    "resolve_window",
    "run",
    "verify_build_identity",
]
