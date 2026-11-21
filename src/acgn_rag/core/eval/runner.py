"""WP5 evaluation runner: one run bound to one snapshot, config and encoder identity.

Owner: E | Spec: `docs/EVAL_PROTOCOL.md` §1/§7/§8, issue #7

The runner is deliberately injectable: `search` consumes the merged store/embedder, and the
answerer is a protocol. Until B's `/ask` lands, `NullAnswerer` records every answer as
`not_run`; the retrieval half (hit rate, evidence recall, MRR) is still measured and archived.

A generated run id includes the build's `identity_slug`, so a second run on the same
snapshot/config but a different encoder lane cannot collide with the first. An explicit
`--run-id` still overrides the generated value.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from acgn_rag.core.eval import metrics, qa
from acgn_rag.core.ground.generate import answerer_version
from acgn_rag.core.ports import (
    GroundedAnswer,
    RetrievalFilters,
    RetrievalQuery,
    RetrievedChunk,
)
from acgn_rag.core.retrieve.official_first import search_with_official_bonus
from acgn_rag.core.retrieve.vector import normalize_query_text
from acgn_rag.core.store.build_state import assert_no_incomplete_build

CONFIG_TOKENS = {"A'": "A-prime"}
BUILD_IDENTITY_KEYS = ("model", "backend", "device", "precision", "batch_size", "identity_slug")
VECTOR_IDENTITY_KEYS = ("model", "backend", "device", "precision", "identity_slug")
CHUNKER_FIELDS = ("version", "header_version", "config", "max_tokens", "overlap", "max_seq_length")
#: `meta.abstention` shape (EVAL_PROTOCOL §7). The identity is compared field by field; a
#: threshold calibrated at another identity must never be applied to this run.
ABSTENTION_KEYS = ("threshold", "source", "calibration_identity")
ABSTENTION_IDENTITY_KEYS = (
    "snapshot_id",
    "chunk_config",
    "chunker",
    "retrieval",
    "embed",
    "encoder_identity_slug",
    "query_transform",
    "official_score_weight",
    "answerer_version",
)
ABSTENTION_RETRIEVAL_KEYS = ("mode", "top_k", "over_fetch")
ABSTENTION_EMBED_KEYS = ("model", "backend", "device", "precision", "library_versions", "batch")
#: The dedup auxiliary-channel sweep is a clustering threshold, never abstention provenance.
_ABSTENTION_SOURCE_MARKERS = ("threshold_sweep", "snapshot_evidence")
_FULL_SNAPSHOT_ID = re.compile(r"(\d{8})-([0-9a-f]{12})")
_RUN_ID_SEGMENT = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")
# QA fields that change what retrieval is asked or judged. Review metadata (`annotator`,
# `verified_by`) and the agreement marker (`overlap`) are deliberately excluded: they can move
# while labels are being completed without changing the evaluation inputs.
QA_INPUT_KEYS = (
    "qa_id",
    "question",
    "answerable",
    "gold_answer",
    "gold_evidence",
    "key_entities",
    "type",
    "lang",
    "window",
)
# Every field a Tier-1 comparison must establish before "identical" means anything.
REQUIRED_TIER1_IDENTITY = (
    "snapshot_id",
    "window_start",
    "window_end",
    "window_verified",
    "chunk_config",
    "collection",
    "top_k",
    "over_fetch",
    "retrieval_mode",
    "identity_slug",
    "chunker",
    "qa_inputs_fingerprint",
    "qa_validated",
    "reportable",
    "query_transform",
    "official_score_weight",
    "answerer_version",
)
_TIER1_STRING_IDENTITY = (
    "snapshot_id",
    "window_start",
    "window_end",
    "chunk_config",
    "collection",
    "retrieval_mode",
    "identity_slug",
    "qa_inputs_fingerprint",
    "query_transform",
    "answerer_version",
)
_TIER1_BOOL_IDENTITY = ("window_verified", "qa_validated", "reportable")
# Identity fields archives written before the answer-line knobs existed carry none of. Reading
# fills these defaults, which reproduce what those runs actually executed -- single-pool search,
# no query transform, the pre-#116 answer line -- so old-vs-old comparisons stay identical.
# Every defaulted field is surfaced as a warning by `compare_tier1_runs`, so the approximation
# stays visible instead of silently widening what "identical" means.
LEGACY_IDENTITY_DEFAULTS = {
    "query_transform": "none",
    "official_score_weight": 1.0,
    "answerer_version": "pre-titlequote",
}
# Answer-row fields the Tier-1 comparison reads. They must be present in both archives:
# two absent values are not evidence that the retrieval outputs matched.
_TIER1_ANSWER_FIELDS = ("question", "retrieved_chunk_ids", "retrieved_urls", "abstained")


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
    qa_role: Literal["report", "calibration"] = "report"
    llm_probe: Mapping[str, Any] | None = None
    abstention: Mapping[str, Any] | None = None
    #: Optional query-translation lane (REBUILD_PLAN §4.3). When set, a question in
    #: ``translate_langs`` is rewritten into English before retrieval and the rewritten text
    #: is what the store sees. The original question still goes to the answerer, so a bad
    #: translation can move a result but can never reach the prompt as evidence.
    translate_langs: tuple[str, ...] = ()
    #: Optional NFKC + whitespace input lane (REBUILD_PLAN L3). Applied to the retrieval text
    #: before the translation lane sees it; recorded as ``query_transform`` in the identity.
    #: Default off until the frozen-set comparison run passes.
    nfkc_queries: bool = False
    #: Multiplier on an official chunk's score before the single ranking (§1b). 1.0 keeps the
    #: historical single-pool behaviour, so an on/off comparison is one constant.
    official_score_weight: float = 1.0


@dataclass(frozen=True)
class RunResult:
    run_id: str
    out_dir: Path
    meta: dict[str, Any]
    metrics: dict[str, Any]


def _config_token(config: str) -> str:
    return CONFIG_TOKENS.get(config, config)


def _run_id_segment(value: str, *, label: str) -> str:
    """Validate one filesystem-safe run-id segment."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if not _RUN_ID_SEGMENT.fullmatch(value):
        raise ValueError(
            f"{label} must start and end with a letter or digit and contain only letters, "
            f"digits, dots, underscores or hyphens: {value!r}"
        )
    return value


def _identity_token(build_identity: Mapping[str, Any]) -> str:
    """Return the encoder identity segment for a generated run id."""
    slug = build_identity.get("identity_slug")
    if not isinstance(slug, str) or not slug.strip():
        raise ValueError(
            "run_id requires build_identity['identity_slug']; "
            "pass --run-id to override the generated id"
        )
    return _run_id_segment(slug.strip(), label="build_identity['identity_slug']")


def _run_id(spec: RunSpec) -> str:
    if spec.run_id:
        return _run_id_segment(spec.run_id, label="run_id")
    day = _run_id_segment(spec.run_date or datetime.now(UTC).strftime("%Y%m%d"), label="run_date")
    config = _run_id_segment(_config_token(spec.config), label="config")
    snapshot = _run_id_segment(spec.snapshot_id, label="snapshot_id")
    identity = _identity_token(spec.build_identity)
    return f"{day}-{config}-{snapshot}-{identity}"


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


def _null_abstention() -> dict[str, Any]:
    """The record for a run where no pre-generation score gate was applied."""
    return {"threshold": None, "source": None, "calibration_identity": None}


def run_identity(
    *,
    snapshot_id: str,
    config: str,
    chunker: Mapping[str, Any],
    top_k: int,
    over_fetch: int,
    build_identity: Mapping[str, Any],
    query_transform: str,
    official_score_weight: float,
    answerer_version: str,
) -> dict[str, Any]:
    """The canonical identity a threshold must be calibrated at.

    Both the applying run and the calibration artifact use this shape, so equality is a field
    comparison, not a re-derivation. `chunker` is restricted to the six protocol fields; a
    build sidecar's extra fields cannot silently enter the identity. The answer-line fields
    bind a threshold to the query transforms and answer rules it was calibrated under, so a
    pre-#116 calibration artifact cannot silently serve a run with the new answer line.
    """
    missing = [name for name in CHUNKER_FIELDS if name not in chunker]
    if missing:
        raise ValueError(f"run identity: chunker is missing {', '.join(missing)}")
    slug = str(build_identity.get("identity_slug") or "").strip()
    if not slug:
        raise ValueError("run identity: build_identity['identity_slug'] is required")
    if not isinstance(query_transform, str) or not query_transform.strip():
        raise ValueError("run identity: query_transform must be a non-empty string")
    if (
        isinstance(official_score_weight, bool)
        or not isinstance(official_score_weight, (int, float))
        or not math.isfinite(float(official_score_weight))
        or official_score_weight <= 0
    ):
        raise ValueError(
            "run identity: official_score_weight must be a finite number greater than zero"
        )
    if not isinstance(answerer_version, str) or not answerer_version.strip():
        raise ValueError("run identity: answerer_version must be a non-empty string")
    return {
        "snapshot_id": str(snapshot_id),
        "chunk_config": str(config),
        "chunker": {name: chunker[name] for name in CHUNKER_FIELDS},
        "retrieval": {"mode": "vector", "top_k": int(top_k), "over_fetch": int(over_fetch)},
        "embed": _embed_metadata(build_identity),
        "encoder_identity_slug": slug,
        "query_transform": str(query_transform),
        "official_score_weight": float(official_score_weight),
        "answerer_version": str(answerer_version),
    }


def _query_transform_slug(spec: RunSpec) -> str:
    """Deterministic identity slug for the query-side transforms a run applied."""
    parts: list[str] = []
    if spec.nfkc_queries:
        parts.append("nfkc")
    if spec.translate_langs:
        parts.append("translate-" + ",".join(sorted(spec.translate_langs)))
    return "+".join(parts) if parts else "none"


def _exact_keys(value: Any, expected: Sequence[str], label: str) -> None:
    """Require a mapping with exactly `expected` keys, naming anything missing or extra."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    keys = set(value)
    missing = [name for name in expected if name not in keys]
    extra = sorted(str(name) for name in keys if name not in expected)
    if missing or extra:
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if extra:
            detail.append("unknown " + ", ".join(extra))
        raise ValueError(f"{label} has {'; '.join(detail)}")


def _validate_abstention(
    block: Mapping[str, Any] | None,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize one `meta.abstention` record against the run identity.

    `None` is the caller's "no threshold" request and becomes the null triple. A thresholded
    record must name its archived calibration artifact and carry an identity equal to this run's
    on every field that can move the score.
    """
    if block is None:
        return _null_abstention()
    _exact_keys(block, ABSTENTION_KEYS, "meta.abstention")
    threshold = block["threshold"]
    source = block["source"]
    calibration = block["calibration_identity"]
    if threshold is None:
        if source is not None or calibration is not None:
            raise ValueError(
                "meta.abstention.source and calibration_identity must be null when threshold is null"
            )
        return _null_abstention()
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
    ):
        raise ValueError("meta.abstention.threshold must be a finite number or null")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("meta.abstention.source must name the archived calibration artifact")
    lowered = source.lower()
    for marker in _ABSTENTION_SOURCE_MARKERS:
        if marker in lowered:
            raise ValueError(
                f"meta.abstention.source names {marker!r}; that is the dedup sweep, "
                "not abstention calibration"
            )
    if isinstance(calibration, Mapping):
        # Calibration artifacts written before the answer-line identity fields existed carry
        # none of them; fill the documented defaults so the artifact still validates. The
        # defaults reproduce the single-pool, untransformed, pre-#116 answer line that artifact
        # was calibrated under. A run whose own identity differs on any of the three is still
        # refused below, which is what forces recalibration after an answer-line change.
        calibration = dict(calibration)
        for name, default in LEGACY_IDENTITY_DEFAULTS.items():
            if name not in calibration:
                calibration[name] = default
    _exact_keys(calibration, ABSTENTION_IDENTITY_KEYS, "meta.abstention.calibration_identity")
    _exact_keys(
        calibration["chunker"],
        CHUNKER_FIELDS,
        "meta.abstention.calibration_identity.chunker",
    )
    _exact_keys(
        calibration["retrieval"],
        ABSTENTION_RETRIEVAL_KEYS,
        "meta.abstention.calibration_identity.retrieval",
    )
    _exact_keys(
        calibration["embed"],
        ABSTENTION_EMBED_KEYS,
        "meta.abstention.calibration_identity.embed",
    )
    if not isinstance(calibration["embed"]["library_versions"], Mapping):
        raise ValueError(
            "meta.abstention.calibration_identity.embed.library_versions must be an object"
        )
    for name in (
        "snapshot_id",
        "chunk_config",
        "encoder_identity_slug",
        "query_transform",
        "answerer_version",
    ):
        value = calibration[name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"meta.abstention.calibration_identity.{name} must be a non-empty string"
            )
    weight = calibration["official_score_weight"]
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(float(weight))
        or weight <= 0
    ):
        raise ValueError(
            "meta.abstention.calibration_identity.official_score_weight must be a finite "
            "number greater than zero"
        )
    embed = calibration["embed"]
    for name in ("model", "backend", "device", "precision"):
        value = embed[name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"meta.abstention.calibration_identity.embed.{name} must be a non-empty string"
            )
    batch = embed["batch"]
    if isinstance(batch, bool) or not isinstance(batch, int) or batch <= 0:
        raise ValueError(
            "meta.abstention.calibration_identity.embed.batch must be a positive integer"
        )
    if not calibration["embed"]["library_versions"]:
        raise ValueError(
            "meta.abstention.calibration_identity.embed.library_versions must not be empty"
        )

    diffs: list[str] = []
    for name in ("snapshot_id", "chunk_config"):
        if calibration[name] != identity.get(name):
            diffs.append(name)
    for name in ("chunker", "retrieval", "embed"):
        claimed = calibration[name]
        actual = identity.get(name) or {}
        for part in actual:
            if claimed.get(part) != actual.get(part):
                diffs.append(f"{name}.{part}")
    if calibration["encoder_identity_slug"] != identity.get("encoder_identity_slug"):
        diffs.append("encoder_identity_slug")
    for name in ("query_transform", "official_score_weight", "answerer_version"):
        if calibration[name] != identity.get(name):
            diffs.append(name)
    if diffs:
        raise ValueError(
            "meta.abstention.calibration_identity does not match this run: " + ", ".join(diffs)
        )
    return {
        "threshold": threshold,
        "source": source,
        "calibration_identity": {
            "snapshot_id": calibration["snapshot_id"],
            "chunk_config": calibration["chunk_config"],
            "chunker": dict(calibration["chunker"]),
            "retrieval": dict(calibration["retrieval"]),
            "embed": {
                **dict(calibration["embed"]),
                "library_versions": dict(calibration["embed"]["library_versions"]),
            },
            "encoder_identity_slug": calibration["encoder_identity_slug"],
            "query_transform": calibration["query_transform"],
            "official_score_weight": calibration["official_score_weight"],
            "answerer_version": calibration["answerer_version"],
        },
    }


def _spec_identity(spec: RunSpec) -> dict[str, Any]:
    return run_identity(
        snapshot_id=spec.snapshot_id,
        config=spec.config,
        chunker=spec.chunker,
        top_k=spec.top_k,
        over_fetch=spec.over_fetch,
        build_identity=spec.build_identity,
        query_transform=_query_transform_slug(spec),
        official_score_weight=spec.official_score_weight,
        answerer_version=answerer_version(),
    )


def _abstention_metadata(spec: RunSpec) -> dict[str, Any]:
    return _validate_abstention(spec.abstention, _spec_identity(spec))


def _validate_applied_threshold(spec: RunSpec, answerer: Answerer | None) -> None:
    """Bind the recorded threshold to the answerer that will actually apply it.

    A caller can pass a `GroundedAnswerer(min_retrieval_score=...)` without a spec block, or a
    spec block without an answerer. Either direction makes `meta.abstention` false evidence, so
    reject the disagreement before retrieval or generation.
    """
    recorded = _abstention_metadata(spec)["threshold"]
    applied = None if answerer is None else getattr(answerer, "min_retrieval_score", None)
    if applied is not None:
        if (
            isinstance(applied, bool)
            or not isinstance(applied, (int, float))
            or not math.isfinite(float(applied))
        ):
            raise ValueError("answerer min_retrieval_score must be a finite number or None")
        applied = float(applied)
    if recorded is not None:
        recorded = float(recorded)
    if recorded == applied:
        return
    if recorded is None:
        raise ValueError(
            "the answerer applies a threshold but meta.abstention records no threshold"
        )
    if applied is None:
        raise ValueError(
            "meta.abstention records a threshold but the run has no answerer to apply it"
        )
    raise ValueError(
        f"answerer min_retrieval_score ({applied}) does not equal "
        f"meta.abstention.threshold ({recorded})"
    )


def validate_abstention_metadata(meta: Mapping[str, Any]) -> list[str]:
    """Validate an archived `meta.abstention` against the archive's own identity.

    Returns problems instead of raising so the live-run validator can report them after the
    archive exists. A missing block is a problem; explicit nulls are not.
    """
    block = meta.get("abstention")
    if block is None:
        return ["meta.abstention is missing"]
    recorded = block.get("threshold") if isinstance(block, Mapping) else None
    if recorded is None:
        try:
            _validate_abstention(block, {})
        except ValueError as exc:
            return [str(exc)]
        return []
    retrieval = meta.get("retrieval") or {}
    build_identity = meta.get("build_identity") or {}
    chunker = meta.get("chunker") or {}
    missing_chunker = [name for name in CHUNKER_FIELDS if name not in chunker]
    if missing_chunker:
        return [f"meta.chunker is missing {', '.join(missing_chunker)}"]
    identity = {
        "snapshot_id": meta.get("snapshot_id"),
        "chunk_config": meta.get("chunk_config"),
        "chunker": {name: chunker[name] for name in CHUNKER_FIELDS},
        "retrieval": {name: retrieval.get(name) for name in ABSTENTION_RETRIEVAL_KEYS},
        "embed": meta.get("embed") or {},
        "encoder_identity_slug": build_identity.get("identity_slug"),
        # Archives from before the answer-line fields existed lack them on both the block and
        # the run meta; the legacy defaults on both sides keep their post-hoc validation green.
        **{
            name: meta.get(name) if meta.get(name) is not None else default
            for name, default in LEGACY_IDENTITY_DEFAULTS.items()
        },
    }
    try:
        _validate_abstention(block, identity)
    except ValueError as exc:
        return [str(exc)]
    return []


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
    qa_role: str = "report",
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
    if qa_role == "calibration":
        # The calibration instrument is decision input, never gate evidence: it is not the
        # frozen 32-item QA set, so its metrics cannot be reported as QA-set results. This
        # check comes last so a broken calibration run reports its real failure instead of
        # borrowing the role's non-reportable status.
        return "calibration_instrument"
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


def _validate_chunker(
    chunker: Mapping[str, Any], config: str, *, source: str = "build.json"
) -> None:
    """Reject a matching but incomplete chunker block, or one for another config."""
    missing = [name for name in CHUNKER_FIELDS if name not in chunker]
    if missing:
        raise ValueError(f"{source}: chunker identity is missing {', '.join(missing)}")
    for name in ("version", "header_version", "config"):
        if not isinstance(chunker[name], str) or not chunker[name]:
            raise ValueError(f"{source}: invalid chunker {name}")
    for name in ("max_tokens", "max_seq_length"):
        value = chunker[name]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{source}: invalid chunker {name}")
    overlap = chunker["overlap"]
    if not isinstance(overlap, int) or isinstance(overlap, bool) or overlap < 0:
        raise ValueError(f"{source}: invalid chunker overlap")
    if chunker["config"] != config:
        raise ValueError(
            f"{source}: chunker config {chunker['config']!r} != selected config {config!r}"
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


EVIDENCE_PAGE_SIZE = 500


def build_evidence_resolver(
    collection: Any,
    *,
    page_size: int = EVIDENCE_PAGE_SIZE,
) -> Callable[[Sequence[str]], Mapping[str, Sequence[str]]]:
    """Resolve gold URLs to chunk ids from the bound collection's metadata.

    The resolver is scoped to the collection the runner will query: a URL present in a
    different snapshot/config must stay unresolved rather than borrowing evidence.

    Metadata is read in bounded pages: an unbounded `get()` on the full collection can exceed
    SQLite's variable limit. Duplicate/overlapping ids and a final count mismatch raise instead
    of returning a partial map. This is a consistency check, not a lock; the index must not be
    changing while the scan runs.
    """
    if page_size < 1:
        raise ValueError("page_size must be >= 1")

    by_url: dict[str, list[str]] = {}
    seen: set[str] = set()
    offset = 0
    while True:
        rows = collection.get(limit=page_size, offset=offset, include=["metadatas"])
        ids = [str(chunk_id) for chunk_id in (rows.get("ids") or [])]
        metadatas = list(rows.get("metadatas") or [])
        if len(ids) != len(metadatas):
            raise RuntimeError("collection metadata rows are incomplete")
        if not ids:
            break
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise RuntimeError("collection metadata scan returned duplicate chunk ids")
        seen.update(ids)
        for chunk_id, metadata in zip(ids, metadatas, strict=True):
            if not isinstance(metadata, dict):
                continue
            url = metadata.get("url")
            if isinstance(url, str) and url:
                by_url.setdefault(metrics.normalize_url(url), []).append(chunk_id)
        if len(ids) < page_size:
            break
        offset += len(ids)

    count = getattr(collection, "count", None)
    if callable(count):
        total = count()
        if total != len(seen):
            raise RuntimeError(
                f"collection metadata scan incomplete: read {len(seen)} of {total} chunk ids"
            )

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
    _validate_chunker(spec.chunker, spec.config, source="run spec")
    _validate_abstention(spec.abstention, _spec_identity(spec))
    if spec.qa_role not in ("report", "calibration"):
        raise ValueError("run spec qa_role must be 'report' or 'calibration'")


def validate_run_spec(spec: RunSpec) -> None:
    """Public preflight wrapper for `run_eval --preflight-only`."""
    _validate_spec(spec)


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


def _qa_inputs_fingerprint(items: Sequence[Mapping[str, Any]]) -> str:
    """Stable digest of the evaluation inputs, independent of JSON key order and formatting.

    Only the fields that change what is asked or judged are hashed; review metadata can move
    while a QA set is being labelled without making two reruns incomparable.
    """
    canonical = [{key: item.get(key) for key in QA_INPUT_KEYS} for item in items]
    payload = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def qa_inputs_fingerprint(items: Sequence[Mapping[str, Any]]) -> str:
    """Public wrapper for the archived QA `inputs_fingerprint`."""
    return _qa_inputs_fingerprint(items)


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


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    """Nearest-rank percentile; deterministic for the small QA sets used here."""
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return round(ordered[index], 3)


def _latency_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
    }


def _answer_record(
    item: Mapping[str, Any],
    chunks: Sequence[RetrievedChunk],
    answer: GroundedAnswer | None,
    diagnostics: Mapping[str, Any] | None = None,
    translation: Mapping[str, Any] | None = None,
    source_mix: Any = None,
) -> dict[str, Any]:
    diagnostics_row = diagnostics or {}
    provider_completion = diagnostics_row.get("provider_completion") is True
    record = {
        "qa_id": str(item["qa_id"]),
        "question": str(item["question"]),
        "retrieved_chunk_ids": [chunk.chunk_id for chunk in chunks],
        "retrieved_urls": [chunk.citation.url for chunk in chunks],
        "answer": answer.answer if answer else None,
        "citations": [citation.model_dump() for citation in answer.citations] if answer else [],
        "cited_chunk_ids": list(answer.cited_chunk_ids) if answer else [],
        "abstained": answer.abstained if answer else None,
        "status": "answered" if answer else "not_run",
        "provider_completion": provider_completion,
        "diagnostics": diagnostics,
    }
    if translation is not None:
        # Only written when the lane is active, so a run without translation keeps the
        # archived shape earlier runs and the Tier-1 comparison already rely on.
        record["query_translation"] = dict(translation)
    if source_mix is not None and (source_mix.official or source_mix.media):
        # Which tier supplied this row's evidence; advisory, and absent when the bonus is off.
        record["source_mix"] = {
            "official": source_mix.official,
            "media": source_mix.media,
        }
    return record


def run(
    spec: RunSpec,
    *,
    store: Any,
    embedder: Any,
    answerer: Answerer | None = None,
    qa_items: Sequence[Mapping[str, Any]] | None = None,
    evidence_resolver: Callable[[Sequence[str]], Mapping[str, Sequence[str]]] | None = None,
    snapshot_member_urls: Collection[str] | None = None,
    query_translator: Callable[[str], Any] | None = None,
) -> RunResult:
    """Execute one run and write the archive under `out_root/<run_id>/`.

    `snapshot_member_urls` is the full member URL set from the frozen manifest. A gold URL
    present there but absent from the bound collection is recorded as `member_not_indexed`,
    not `gold_unavailable`; the alias policy is pending the E/F/D rule.

    `query_translator` is the optional cross-language retrieval lane: it takes a question and
    returns an object with an `english` attribute (or None when translation is unavailable).
    It is consulted only for items whose `lang` is in `spec.translate_langs`, and only the
    retrieval text changes - the answerer always receives the original question.
    """
    _validate_spec(spec)
    _validate_applied_threshold(spec, answerer)
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
    retrieval_latencies: list[float] = []
    for item in items:
        gold_urls = _item_urls(item)
        window = item["window"]
        question = str(item["question"])
        # The translation lane changes only what the store is asked. The answerer keeps the
        # original question, so the answer is still about what the user actually asked. The
        # NFKC lane runs first, so a translation never has to re-derive the normalized form.
        retrieval_text = question
        if spec.nfkc_queries:
            retrieval_text = normalize_query_text(retrieval_text)
        translation: dict[str, Any] | None = None
        if query_translator is not None and str(item.get("lang", "")) in spec.translate_langs:
            translated = query_translator(retrieval_text)
            # The translator may return a bare result (`.english`) or a `TranslationAttempt`
            # carrying the miss reason. Both are accepted; the attempt form is what makes a
            # broken lane visible instead of showing up as `used: false` on every row.
            attempt = getattr(translated, "translated", translated)
            reason = getattr(translated, "reason", None)
            detail = getattr(translated, "detail", None)
            english = getattr(attempt, "english", None)
            translation = {
                "lang": str(item.get("lang", "")),
                "english": english,
                "used": attempt is not None,
            }
            if reason is not None:
                translation["reason"] = reason
            if detail is not None:
                translation["detail"] = detail
            if attempt is not None:
                retrieval_text = attempt.english
        query = RetrievalQuery(
            text=retrieval_text,
            top_k=spec.top_k,
            snapshot_id=spec.snapshot_id,
            filters=RetrievalFilters(since=str(window[0]), until=str(window[1])),
        )
        answer_query = RetrievalQuery(
            text=question,
            top_k=spec.top_k,
            snapshot_id=spec.snapshot_id,
            filters=RetrievalFilters(since=str(window[0]), until=str(window[1])),
        )
        started = time.perf_counter()
        chunks, source_mix = search_with_official_bonus(
            query,
            store=store,
            embedder=embedder,
            official_weight=spec.official_score_weight,
        )
        retrieval_latencies.append((time.perf_counter() - started) * 1000.0)
        answer: GroundedAnswer | None = None
        diagnostics: dict[str, Any] | None = None
        if answerer is not None:
            if hasattr(answerer, "answer_with_diagnostics"):
                answer_run = answerer.answer_with_diagnostics(answer_query, chunks)
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
        answers.append(_answer_record(item, chunks, answer, diagnostics, translation, source_mix))
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
        qa_role=spec.qa_role,
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
            "latency_ms": _latency_summary(retrieval_latencies),
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
        "query_transform": _query_transform_slug(spec),
        "official_score_weight": spec.official_score_weight,
        "answerer_version": answerer_version(),
        "llm": dict(spec.llm),
        "llm_probe": dict(spec.llm_probe) if spec.llm_probe else None,
        "abstention": _abstention_metadata(spec),
        "git_sha": spec.git_sha,
        "qa": {
            "path": str(spec.qa_path),
            "count": len(items),
            "inputs_fingerprint": _qa_inputs_fingerprint(items),
            "validated": effective_qa_validated,
            "validation": "validated" if effective_qa_validated else "bypassed",
            "source": effective_qa_source,
            "role": spec.qa_role,
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


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL run archive file; every row must be a JSON object.

    Split on LF only: JSON strings written with ``ensure_ascii=False`` may contain
    U+2028/U+2029/U+0085, and ``str.splitlines()`` would split a record mid-JSON.
    """
    try:
        lines = path.read_text(encoding="utf-8").split("\n")
    except OSError as exc:
        raise ValueError(f"cannot read run archive file {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{lineno}: expected a JSON object")
        rows.append(row)
    return rows


def _load_run_archive(
    path: Path | str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    root = Path(path)
    try:
        meta = _read_json(root / "meta.json")
        metrics = _read_json(root / "metrics.json")
        answers = read_jsonl_rows(root / "answers.jsonl")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid run archive {root}: {exc}") from exc
    return meta, metrics, answers


def _archive_identity(meta: Mapping[str, Any]) -> dict[str, Any]:
    retrieval = meta.get("retrieval")
    retrieval = retrieval if isinstance(retrieval, dict) else {}
    build_identity = meta.get("build_identity")
    build_identity = build_identity if isinstance(build_identity, dict) else {}
    qa_block = meta.get("qa")
    qa_block = qa_block if isinstance(qa_block, dict) else {}
    identity = {
        "snapshot_id": meta.get("snapshot_id"),
        "window_start": meta.get("window_start"),
        "window_end": meta.get("window_end"),
        "window_verified": meta.get("window_verified"),
        "chunk_config": meta.get("chunk_config"),
        "collection": meta.get("collection"),
        "top_k": retrieval.get("top_k"),
        "over_fetch": retrieval.get("over_fetch"),
        "retrieval_mode": retrieval.get("mode"),
        "identity_slug": build_identity.get("identity_slug"),
        "chunker": meta.get("chunker"),
        "qa_inputs_fingerprint": qa_block.get("inputs_fingerprint"),
        "qa_validated": qa_block.get("validated"),
        "reportable": meta.get("reportable"),
        "query_transform": meta.get("query_transform"),
        "official_score_weight": meta.get("official_score_weight"),
        "answerer_version": meta.get("answerer_version"),
    }
    for name, default in LEGACY_IDENTITY_DEFAULTS.items():
        if identity[name] is None:
            identity[name] = default
    return identity


def _legacy_identity_fields(meta: Mapping[str, Any]) -> list[str]:
    """Identity fields absent from an archive and therefore filled with legacy defaults."""
    return sorted(name for name in LEGACY_IDENTITY_DEFAULTS if meta.get(name) is None)


def _missing_tier1_identity(identity: Mapping[str, Any]) -> list[str]:
    """Identity fields that are absent, empty or the wrong type on one side."""
    invalid: list[str] = []
    for name in REQUIRED_TIER1_IDENTITY:
        value = identity.get(name)
        if name == "qa_inputs_fingerprint":
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                invalid.append(name)
        elif name in _TIER1_STRING_IDENTITY:
            if not isinstance(value, str) or not value.strip():
                invalid.append(name)
        elif name in _TIER1_BOOL_IDENTITY:
            if not isinstance(value, bool):
                invalid.append(name)
        elif name in ("top_k", "over_fetch"):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                invalid.append(name)
        elif name == "official_score_weight":
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
            ):
                invalid.append(name)
        elif name == "chunker":
            if not isinstance(value, dict) or not value:
                invalid.append(name)
            else:
                config = identity.get("chunk_config")
                try:
                    _validate_chunker(
                        value, config if isinstance(config, str) else "", source="archive"
                    )
                except (TypeError, ValueError):
                    invalid.append(name)
        elif value is None or value == "" or value == {}:
            invalid.append(name)
    return invalid


def _archive_qa_count(meta: Mapping[str, Any]) -> int | None:
    qa_block = meta.get("qa")
    qa_block = qa_block if isinstance(qa_block, dict) else {}
    count = qa_block.get("count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        return None
    return count


def _missing_tier1_metrics(metrics: Mapping[str, Any]) -> list[str]:
    """Metric blocks the comparison reads; absent blocks are not evidence of equality."""
    missing: list[str] = []
    retrieval = metrics.get("retrieval")
    if not isinstance(retrieval, dict):
        missing.append("retrieval")
    else:
        for name in ("hit", "evidence_recall"):
            block = retrieval.get(name)
            if not isinstance(block, dict):
                missing.append(f"retrieval.{name}")
                continue
            for language in ("all", "en", "zh"):
                entry = block.get(language)
                if (
                    not isinstance(entry, dict)
                    or "value" not in entry
                    or not isinstance(entry.get("n"), int)
                    or isinstance(entry.get("n"), bool)
                    or entry["n"] < 0
                ):
                    missing.append(f"retrieval.{name}.{language}")
                    continue
                # A null rate is legitimate only for an empty eligible set.
                value = entry["value"]
                if value is None:
                    if entry["n"] != 0:
                        missing.append(f"retrieval.{name}.{language}")
                elif not isinstance(value, (int, float)) or isinstance(value, bool):
                    missing.append(f"retrieval.{name}.{language}")
    if not isinstance(metrics.get("counts"), dict):
        missing.append("counts")
    if not isinstance(metrics.get("provisional"), bool):
        missing.append("provisional")
    reporting = metrics.get("reporting")
    if not isinstance(reporting, dict):
        missing.append("reporting")
    elif not isinstance(reporting.get("retrieval"), dict):
        missing.append("reporting.retrieval")
    return missing


def _answers_by_qa(rows: Sequence[Mapping[str, Any]], *, side: str) -> dict[str, Mapping[str, Any]]:
    by_qa: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        qa_id = row.get("qa_id")
        if not isinstance(qa_id, str) or not qa_id:
            raise ValueError(f"{side} archive has an answer row without a qa_id")
        if qa_id in by_qa:
            raise ValueError(f"{side} archive has duplicate qa_id {qa_id!r}")
        for name in _TIER1_ANSWER_FIELDS:
            if name not in row:
                raise ValueError(f"{side} archive row {qa_id!r} is missing required field {name!r}")
            value = row[name]
            if name == "question":
                if not isinstance(value, str):
                    raise ValueError(f"{side} archive row {qa_id!r} has a non-string question")
            elif name == "abstained":
                if value is not None and not isinstance(value, bool):
                    raise ValueError(
                        f"{side} archive row {qa_id!r} has an invalid abstained value {value!r}"
                    )
            elif not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ValueError(
                    f"{side} archive row {qa_id!r} has an invalid {name} value; expected a list "
                    "of strings"
                )
        by_qa[qa_id] = row
    return by_qa


def compare_tier1_runs(left: Path | str, right: Path | str) -> dict[str, Any]:
    """Compare two run archives for Tier-1 reproducibility.

    Tier-1 is the deterministic half of `EVAL_PROTOCOL §8`: same snapshot/window/config,
    chunker and QA inputs, encoder identity and collection; identical retrieval metrics,
    retrieved chunk-id/url order and abstention decisions at a fixed threshold. Answer text,
    citations, `diagnostics` and human labels are Tier-2 and are deliberately not compared.

    A comparison is `comparable` only when both archives carry every required identity field,
    every required Tier-1 answer field, the recorded QA-set fingerprint matches, the answer-row
    count equals `meta.qa.count` on both sides and every QA id was compared. A missing answer
    field is rejected rather than treated as "both sides agree"; otherwise two archives could be
    reported identical without recording the retrieval order. A differing `git_sha` is a warning,
    not a comparability failure: a docs-only commit does not change the vectors.
    """
    left_meta, left_metrics, left_answers = _load_run_archive(left)
    right_meta, right_metrics, right_answers = _load_run_archive(right)

    warnings: list[dict[str, Any]] = []
    if left_meta.get("git_sha") != right_meta.get("git_sha"):
        warnings.append(
            {
                "field": "git_sha",
                "left": left_meta.get("git_sha"),
                "right": right_meta.get("git_sha"),
            }
        )
    for side, side_meta in (("left", left_meta), ("right", right_meta)):
        for name in _legacy_identity_fields(side_meta):
            warnings.append(
                {
                    "field": name,
                    "note": f"{name} absent from the {side} archive; legacy default "
                    f"{LEGACY_IDENTITY_DEFAULTS[name]!r} applied",
                }
            )

    left_identity = _archive_identity(left_meta)
    right_identity = _archive_identity(right_meta)

    def _not_comparable(
        reason: str,
        *,
        identity_mismatches: Sequence[Mapping[str, Any]] = (),
        metric_mismatches: Sequence[Mapping[str, Any]] = (),
        item_mismatches: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        return {
            "identical": False,
            "comparable": False,
            "reason": reason,
            "identity_mismatches": list(identity_mismatches),
            "metric_mismatches": list(metric_mismatches),
            "item_mismatches": list(item_mismatches),
            "warnings": warnings,
            "checked_items": 0,
        }

    missing = sorted(
        set(_missing_tier1_identity(left_identity)) | set(_missing_tier1_identity(right_identity))
    )
    if missing:
        return _not_comparable("identity unavailable: " + ", ".join(missing))

    identity_mismatches = [
        {"field": name, "left": left_identity[name], "right": right_identity[name]}
        for name in left_identity
        if left_identity[name] != right_identity[name]
    ]
    if identity_mismatches:
        return _not_comparable("identity mismatch", identity_mismatches=identity_mismatches)

    left_count = _archive_qa_count(left_meta)
    right_count = _archive_qa_count(right_meta)
    if left_count is None or right_count is None:
        return _not_comparable("qa count unavailable")
    if (
        left_count != right_count
        or len(left_answers) != left_count
        or len(right_answers) != right_count
    ):
        return _not_comparable(
            f"answer count mismatch: left {len(left_answers)}/{left_count}, "
            f"right {len(right_answers)}/{right_count}"
        )

    metric_mismatches: list[dict[str, Any]] = []
    for name in ("retrieval", "counts", "provisional"):
        if left_metrics.get(name) != right_metrics.get(name):
            metric_mismatches.append(
                {
                    "field": name,
                    "left": left_metrics.get(name),
                    "right": right_metrics.get(name),
                }
            )
    missing_metrics = sorted(
        set(_missing_tier1_metrics(left_metrics)) | set(_missing_tier1_metrics(right_metrics))
    )
    if missing_metrics:
        return _not_comparable(
            "metrics unavailable: " + ", ".join(missing_metrics),
            metric_mismatches=metric_mismatches,
        )
    left_reporting = left_metrics["reporting"]
    right_reporting = right_metrics["reporting"]
    left_retrieval_reporting = left_reporting["retrieval"]
    right_retrieval_reporting = right_reporting["retrieval"]
    if left_retrieval_reporting != right_retrieval_reporting:
        metric_mismatches.append(
            {
                "field": "reporting.retrieval",
                "left": left_retrieval_reporting,
                "right": right_retrieval_reporting,
            }
        )

    left_by_qa = _answers_by_qa(left_answers, side="left")
    right_by_qa = _answers_by_qa(right_answers, side="right")
    item_mismatches: list[dict[str, Any]] = []
    for qa_id in sorted(set(left_by_qa) | set(right_by_qa)):
        if qa_id not in left_by_qa or qa_id not in right_by_qa:
            item_mismatches.append(
                {
                    "qa_id": qa_id,
                    "field": "presence",
                    "left": qa_id in left_by_qa,
                    "right": qa_id in right_by_qa,
                }
            )
            continue
        for name in ("question", "retrieved_chunk_ids", "retrieved_urls", "abstained"):
            if left_by_qa[qa_id].get(name) != right_by_qa[qa_id].get(name):
                item_mismatches.append(
                    {
                        "qa_id": qa_id,
                        "field": name,
                        "left": left_by_qa[qa_id].get(name),
                        "right": right_by_qa[qa_id].get(name),
                    }
                )

    checked_items = len(set(left_by_qa) & set(right_by_qa))
    if checked_items != left_count:
        return _not_comparable(
            "item coverage mismatch",
            metric_mismatches=metric_mismatches,
            item_mismatches=item_mismatches,
        )

    identical = not metric_mismatches and not item_mismatches
    return {
        "identical": identical,
        "comparable": True,
        "reason": None,
        "identity_mismatches": [],
        "metric_mismatches": metric_mismatches,
        "item_mismatches": item_mismatches,
        "warnings": warnings,
        "checked_items": checked_items,
    }


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
    "compare_tier1_runs",
    "load_build_records",
    "load_snapshot_member_urls",
    "qa_inputs_fingerprint",
    "read_jsonl_rows",
    "resolve_window",
    "run",
    "run_identity",
    "validate_abstention_metadata",
    "validate_run_spec",
    "verify_build_identity",
]
