"""Runtime configuration: read from environment / .env only, never hard-coded.

Owner: D | Spec: 04 §7 D4/D5, 06 §1

Design constraints:
  - Swapping the vector store or the LLM vendor changes `.env` only, never code
    (the internal LLMClient / Embedder interfaces stay stable).
  - Secrets never enter the repo: `.env` is gitignored, only `.env.example` is committed.
  - Every setting that can change an experimental result (model name, version, device)
    must be recordable in an eval run.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_UA = "AcgnRagBot/0.1 (CS5481 course project; +https://github.com/L01nki1/acgn-chronicle)"


def _env(key: str, default: str | None = None) -> str | None:
    value = os.environ.get(key)
    return value if value not in (None, "") else default


def _env_int(key: str, default: int) -> int:
    raw = _env(key)
    return int(raw) if raw is not None else default


def _env_float(key: str, default: float | None = None) -> float | None:
    raw = _env(key)
    # A whitespace-only value means "unset" for a float setting: otherwise a stray blank in
    # .env turns the documented "unset disables the threshold" behaviour into a ValueError.
    # Handled here, not in _env, so string settings keep their exact raw value.
    if raw is None or not raw.strip():
        return default
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"{key} must be a finite number, got {raw!r}")
    return value


@dataclass(frozen=True)
class Paths:
    root: Path = field(default_factory=lambda: Path(_env("ACGN_DATA_DIR", "data") or "data"))

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def facts(self) -> Path:
        return self.root / "facts"

    @property
    def index(self) -> Path:
        return self.root / "index"

    @property
    def snapshots(self) -> Path:
        return self.root / "snapshots"


@dataclass(frozen=True)
class EmbedConfig:
    model_name: str = field(
        default_factory=lambda: _env("ACGN_EMBED_MODEL", "intfloat/multilingual-e5-base") or ""
    )
    # Indexing and evaluation both run on this device; identical device is what makes
    # numbers comparable across machines.
    device: str = field(default_factory=lambda: _env("ACGN_EMBED_DEVICE", "cpu") or "cpu")
    batch_size: int = field(default_factory=lambda: _env_int("ACGN_EMBED_BATCH", 32))
    # e5 models require these prefixes; omitting them costs recall (PLAN §3 item 1).
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "


@dataclass(frozen=True)
class LLMConfig:
    """Extraction and answer generation may use different vendors: the two groups are independent."""

    base_url: str = field(default_factory=lambda: _env("ACGN_LLM_BASE_URL", "") or "")
    api_key: str = field(default_factory=lambda: _env("ACGN_LLM_API_KEY", "") or "")
    model: str = field(default_factory=lambda: _env("ACGN_LLM_MODEL", "") or "")
    model_version: str = field(default_factory=lambda: _env("ACGN_LLM_MODEL_VERSION", "") or "")
    temperature: float = 0.0

    # Extraction line (F, W2 batch: cheap, strict JSON, retryable)
    extract_base_url: str = field(
        default_factory=lambda: _env("ACGN_EXTRACT_BASE_URL", _env("ACGN_LLM_BASE_URL", "")) or ""
    )
    extract_api_key: str = field(
        default_factory=lambda: _env("ACGN_EXTRACT_API_KEY", _env("ACGN_LLM_API_KEY", "")) or ""
    )
    extract_model: str = field(
        default_factory=lambda: _env("ACGN_EXTRACT_MODEL", _env("ACGN_LLM_MODEL", "")) or ""
    )

    # Answer line (B, W3 onwards: instruction following, no invented citations, abstention)
    answer_base_url: str = field(
        default_factory=lambda: _env("ACGN_ANSWER_BASE_URL", _env("ACGN_LLM_BASE_URL", "")) or ""
    )
    answer_api_key: str = field(
        default_factory=lambda: _env("ACGN_ANSWER_API_KEY", _env("ACGN_LLM_API_KEY", "")) or ""
    )
    answer_model: str = field(
        default_factory=lambda: _env("ACGN_ANSWER_MODEL", _env("ACGN_LLM_MODEL", "")) or ""
    )
    # The answer line may use a different model/version from extraction. Do not silently fall
    # back to the extraction version when the models differ; the live-run preflight rejects that.
    answer_model_version: str = field(
        default_factory=lambda: _env("ACGN_ANSWER_MODEL_VERSION", "") or ""
    )
    # RUNBOOK §3.4: 1024 is a reasoning-token floor, not an answer budget; the default
    # leaves room for reasoning plus the structured claims payload.
    answer_max_tokens: int = field(default_factory=lambda: _env_int("ACGN_ANSWER_MAX_TOKENS", 2048))
    # EVAL_PROTOCOL §6/§8: the calibration set exists to sweep this threshold, and abstention
    # decisions must be identical at a fixed threshold. Unset disables the pre-generation
    # check and keeps the existing behavior (abstention only after claim validation).
    answer_min_retrieval_score: float | None = field(
        default_factory=lambda: _env_float("ACGN_ANSWER_MIN_RETRIEVAL_SCORE")
    )


@dataclass(frozen=True)
class HttpConfig:
    user_agent: str = field(default_factory=lambda: _env("ACGN_HTTP_UA", DEFAULT_UA) or DEFAULT_UA)
    timeout_s: float = field(default_factory=lambda: float(_env("ACGN_HTTP_TIMEOUT", "10") or "10"))
    # Per-host pacing and global concurrency (01 §5.2): 1 req/2.5s with jitter, 8-14 workers.
    min_interval_s: float = 2.5
    max_concurrency: int = field(default_factory=lambda: _env_int("ACGN_HTTP_CONCURRENCY", 8))
    respect_robots: bool = True


@dataclass(frozen=True)
class StoreConfig:
    backend: str = field(default_factory=lambda: _env("ACGN_STORE_BACKEND", "chroma") or "chroma")
    # Collection naming uses the complete snapshot id; retention defaults to two snapshots.
    keep_snapshots: int = field(default_factory=lambda: _env_int("ACGN_KEEP_SNAPSHOTS", 2))
    over_fetch_factor: int = field(default_factory=lambda: _env_int("ACGN_OVERFETCH", 5))


@dataclass(frozen=True)
class Settings:
    paths: Paths = field(default_factory=Paths)
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    http: HttpConfig = field(default_factory=HttpConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    env: str = field(default_factory=lambda: _env("ACGN_ENV", "local") or "local")


def load_settings() -> Settings:
    """Re-read the environment on every call so tests cannot leak state into each other."""
    return Settings()


__all__ = [
    "DEFAULT_UA",
    "EmbedConfig",
    "HttpConfig",
    "LLMConfig",
    "Paths",
    "Settings",
    "StoreConfig",
    "load_settings",
]
