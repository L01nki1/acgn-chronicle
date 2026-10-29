"""Encoding lane: encoder identity, model-family policy, context headers, cache keys, throughput.

Owner: F | Spec: 04 §7 D4 / RUNBOOK §3.4-§3.5

Design constraints:
  - Every build records its full encoder identity (model, backend, device, precision, library
    versions, batch size); vectors from different identities never mix in one index (§3.4).
  - Prefix-vs-prompt policy is keyed by model family here, so D's `EmbedConfig` keeps only the e5
    defaults and this module is the single place a new bake-off candidate registers.
  - What gets encoded is `prefix + header + "\n" + body` with `header = title | publisher | date`
    (§3.5). The header template is versioned (`HEADER_VERSION`) and hashed into the cache key
    together with the body, so two syndicated copies with identical bodies never serve each
    other's header (§3.4).
  - Throughput is measured in tokens (tokenizer counts, never characters): items/s varies ~3x with
    chunk length and is a forbidden reporting unit (issue #8, §3.4).

Heavy imports (sentence_transformers) stay inside `Embedder._load`, so an environment without the
[embed] extra can still import this module.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from acgn_rag.config import EmbedConfig

# Bump when the header template changes: the version is hashed into every cache key (RUNBOOK §3.4),
# so stale vectors are rebuilt instead of silently reused.
HEADER_VERSION = "h1"


def _tag(value: str, separator: str = "-") -> str:
    """Lowercase, keep only [0-9a-zA-Z], join runs of anything else with `separator`."""
    return re.sub(r"[^0-9a-zA-Z]+", separator, value).strip(separator).lower()


@dataclass(frozen=True)
class ModelSpec:
    """How one model family turns queries/passages into model input.

    mode="prefix": `query`/`passage` are literal text prefixes (e5: "query: " / "passage: ").
    mode="prompt": they are sentence-transformers `prompt_name`s (Qwen3, EmbeddingGemma); None on
        a side means that side sends no prompt (Qwen3 documents carry none).
    mode="none": neither prefixes nor prompts (bge-m3 dense).
    """

    mode: Literal["prefix", "prompt", "none"]
    query: str | None = None
    passage: str | None = None


# The e5 prefixes are taken from EmbedConfig's defaults (no env hook exists for them, so this is
# deterministic at import). The prompt names for the other families are the ones their
# sentence-transformers configs ship with.
_EMBED_DEFAULTS = EmbedConfig()

# Registry keyed by lowercase substring of the model name; first match wins. Registering a new
# bake-off candidate here is the whole integration - unknown families fail loudly (model_spec).
MODEL_SPECS: Mapping[str, ModelSpec] = {
    "multilingual-e5": ModelSpec(
        mode="prefix", query=_EMBED_DEFAULTS.query_prefix, passage=_EMBED_DEFAULTS.passage_prefix
    ),
    "qwen3-embedding": ModelSpec(mode="prompt", query="query", passage=None),
    "bge-m3": ModelSpec(mode="none"),
    "embeddinggemma": ModelSpec(mode="prompt", query="query", passage="document"),
}


def model_spec(model_name: str) -> ModelSpec:
    """Family policy for a model name (substring match, case-insensitive); unknown names raise."""
    lowered = model_name.lower()
    for substring, spec in MODEL_SPECS.items():
        if substring in lowered:
            return spec
    raise ValueError(
        f"no registered model spec for {model_name!r}; add the family to MODEL_SPECS "
        f"(registered: {', '.join(sorted(MODEL_SPECS))})"
    )


@dataclass(frozen=True)
class PassageInput:
    """One chunk to encode: the three context-header fields plus the body (RUNBOOK §3.5).

    `doc_id`/`chunk_id` ride along for the caller's bookkeeping; the encoded text uses only the
    header fields and the body.
    """

    title: str = ""
    publisher: str = ""
    published_at: str = ""
    body: str = ""
    doc_id: str = ""
    chunk_id: str = ""


def build_header(title: str, publisher: str, published_at: str) -> str:
    """`title | publisher | published_at`; empty fields are dropped, not left as separators."""
    return " | ".join(part.strip() for part in (title, publisher, published_at) if part.strip())


@dataclass(frozen=True)
class EncoderIdentity:
    """Everything about an encoder that can change the vector (RUNBOOK §3.4).

    Recorded in index metadata (`to_metadata`) and hashed into the cache key (`slug`), so vectors
    from different identities never mix; `backend` covers lanes this module does not build itself
    (openvino/onnx construct the identity directly).
    """

    model: str
    backend: Literal["torch", "openvino", "onnx"] = "torch"
    device: str = "cpu"
    precision: Literal["fp32", "fp16"] = "fp32"
    batch_size: int = 32
    lib_versions: tuple[tuple[str, str], ...] = ()

    def slug(self) -> str:
        """Filesystem-safe short id: readable fields plus an 8-hex digest of the full identity.

        sha256 (not `hash`), so the slug is stable across processes and PYTHONHASHSEED settings.
        """
        payload = {
            "model": self.model,
            "backend": self.backend,
            "device": self.device,
            "precision": self.precision,
            "batch_size": self.batch_size,
            "lib_versions": [list(pair) for pair in self.lib_versions],
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:8]
        return "-".join(
            (
                _tag(self.model.split("/")[-1]),
                self.backend,
                _tag(self.device),
                self.precision,
                f"b{self.batch_size}",
                digest,
            )
        )

    def to_metadata(self) -> dict[str, str | int]:
        """Flat scalars only - Chroma metadata rejects nested values (RUNBOOK §3.1 item 1)."""
        meta: dict[str, str | int] = {
            "model": self.model,
            "backend": self.backend,
            "device": self.device,
            "precision": self.precision,
            "batch_size": self.batch_size,
            "identity_slug": self.slug(),
        }
        for name, version in self.lib_versions:
            meta[f"lib_{_tag(name, '_')}"] = version
        return meta


def cache_key(
    identity: EncoderIdentity,
    model: str,
    prefix_or_prompt: str | None,
    header: str,
    body: str,
) -> str:
    """sha256 over every input that can change the vector (RUNBOOK §3.4): header template
    version, model, family mode, prefix/prompt, header, body, and the encoder identity slug.

    The canonical string joins fields with the ASCII unit separator (0x1f), which cannot occur in
    the fields themselves, so no header/body pair can shift a field boundary.
    """
    canonical = "\x1f".join(
        (
            HEADER_VERSION,
            model,
            model_spec(model).mode,
            prefix_or_prompt or "",
            header,
            body,
            identity.slug(),
        )
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _join(header: str, body: str) -> str:
    return f"{header}\n{body}" if header else body


def _coerce_passage(item: PassageInput | tuple[str, str] | dict[str, str]) -> tuple[str, str]:
    """Accept PassageInput, a ready-made (header, body) pair, or a dict with the header fields."""
    if isinstance(item, PassageInput):
        return build_header(item.title, item.publisher, item.published_at), item.body
    if isinstance(item, dict):
        header = build_header(
            str(item.get("title", "")),
            str(item.get("publisher", "")),
            str(item.get("published_at", "")),
        )
        return header, str(item.get("body", ""))
    header, body = item
    return header, body


def _l2_normalized(vectors: Any) -> np.ndarray:
    """float32 with unit rows; a zero row passes through unchanged (it cannot be normalized)."""
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return arr / norms


def _param_precision(model: Any) -> Literal["fp32", "fp16"]:
    """fp16 when the first parameter says so (covers float16 and bfloat16); fp32 otherwise -
    including injected fakes that have no parameters."""
    parameters = getattr(model, "parameters", None)
    if parameters is None:
        return "fp32"
    for parameter in parameters():
        return "fp16" if "16" in str(getattr(parameter, "dtype", "")) else "fp32"
    return "fp32"


def _library_versions() -> tuple[tuple[str, str], ...]:
    """(dist, version) pairs for the libraries that can change encode results.

    sys.modules only: a library nobody imported cannot have influenced the vectors.
    """
    out: list[tuple[str, str]] = []
    for dist, module_name in (
        ("sentence-transformers", "sentence_transformers"),
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("numpy", "numpy"),
    ):
        version = getattr(sys.modules.get(module_name), "__version__", None)
        if version is not None:
            out.append((dist, version))
    return tuple(out)


def _peak_rss_mb() -> float | None:
    """Process peak RSS in MB, or None on Windows (no `resource` module there).

    Linux reports ru_maxrss in KiB, macOS in bytes. This is the process-lifetime maximum to
    date, not the delta of any single call.
    """
    try:
        import resource
    except ImportError:
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


class Embedder:
    """Encode passages/queries into unit-length float32 vectors.

    `model`/`tokenizer` are injectable so CI runs on a fake with zero downloads; when `model` is
    None the real sentence-transformers model is built lazily (the import stays inside `_load`,
    so environments without the [embed] extra never load torch). The backend is torch: an
    OpenVINO/ONNX lane constructs `EncoderIdentity` directly and injects its own model.
    """

    def __init__(
        self,
        config: EmbedConfig,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        self.config = config
        self.spec = model_spec(config.model_name)
        self._model = model
        self._tokenizer = tokenizer

    @property
    def tokenizer(self) -> Any:
        """The tokenizer whose counts define budgets and throughput (RUNBOOK §3.4)."""
        if self._tokenizer is None:
            self._tokenizer = self._load().tokenizer
        return self._tokenizer

    def identity(self) -> EncoderIdentity:
        """The encoder identity of this embedder, for index metadata and run records (§3.4)."""
        model = self._load()  # ensure sentence_transformers/torch appear in lib_versions
        return EncoderIdentity(
            model=self.config.model_name,
            backend="torch",
            device=self.config.device,
            precision=_param_precision(model),
            batch_size=self.config.batch_size,
            lib_versions=_library_versions(),
        )

    def encode_passages(
        self,
        items: Sequence[PassageInput | tuple[str, str] | dict[str, str]],
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Encode chunks as `prefix + header + "\n" + body` (RUNBOOK §3.5): a chunk that lost its
        parent context is unretrievable. `batch_size` overrides the config batch for one call."""
        texts = [_join(header, body) for header, body in map(_coerce_passage, items)]
        return self._encode(texts, "passage", batch_size)

    def encode_queries(self, texts: Sequence[str], batch_size: int | None = None) -> np.ndarray:
        """Encode raw query strings: a query has no parent document, so it gets no header."""
        return self._encode(list(texts), "query", batch_size)

    def _encode(
        self, texts: list[str], side: Literal["query", "passage"], batch_size: int | None
    ) -> np.ndarray:
        chosen = self.spec.query if side == "query" else self.spec.passage
        prompt_name: str | None = None
        if self.spec.mode == "prefix":
            if chosen is None:
                raise ValueError(
                    f"prefix-mode spec for {self.config.model_name!r} has no {side} prefix"
                )
            texts = [chosen + text for text in texts]
        elif self.spec.mode == "prompt":
            prompt_name = chosen
        model = self._load()
        batch = batch_size if batch_size is not None else self.config.batch_size
        if prompt_name is None:
            vectors = model.encode(texts, batch_size=batch)
        else:
            vectors = model.encode(texts, batch_size=batch, prompt_name=prompt_name)
        return _l2_normalized(vectors)

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # heavy: [embed] extra only

            self._model = SentenceTransformer(self.config.model_name, device=self.config.device)
        return self._model


def measure_throughput(
    embedder: Embedder,
    tokenizer: Any,
    texts: Sequence[str],
    batch_size: int | None = None,
) -> dict[str, float | int | None]:
    """Wall-clock encode throughput in tokens/s plus peak RSS, for the RUNBOOK §3.4 lane tables.

    Tokens are `len(tokenizer(text).input_ids)` summed over texts - tokenizer counts, never
    characters, and items/s is a forbidden reporting unit (issue #8). The measured numbers belong
    in docs/RUNBOOK.md, not in code. `texts` are encoded as passages with an empty header (the
    model cost is the measurement, not the header); `batch_size` overrides the config batch for
    the measured call only. `peak_rss_mb` is None where `resource` is unavailable (Windows).
    """
    total_tokens = sum(len(tokenizer(text).input_ids) for text in texts)
    start = time.perf_counter()
    embedder.encode_passages([("", text) for text in texts], batch_size=batch_size)
    elapsed = time.perf_counter() - start
    return {
        "texts": len(texts),
        "tokens": total_tokens,
        "seconds": elapsed,
        "tokens_per_s": total_tokens / elapsed if elapsed else float("inf"),
        "batch_size": batch_size if batch_size is not None else embedder.config.batch_size,
        "peak_rss_mb": _peak_rss_mb(),
    }


__all__ = [
    "HEADER_VERSION",
    "MODEL_SPECS",
    "Embedder",
    "EncoderIdentity",
    "ModelSpec",
    "PassageInput",
    "build_header",
    "cache_key",
    "measure_throughput",
    "model_spec",
]
