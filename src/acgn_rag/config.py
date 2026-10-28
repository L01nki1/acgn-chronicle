"""运行时配置：只从环境变量 / .env 读取，不在代码里写死密钥与厂商。

负责人：D　|　依据：04 §7 D4/D5、06 §1

设计约束：
  - 换向量库或换模型厂商 → 只改 .env，不改代码（内部接口 LLMClient / Embedder 保持不变）。
  - 密钥永不入库：.env 进 .gitignore，仓库只提交 .env.example。
  - 一切影响实验结果的配置（模型名、版本、device）都必须能被 eval run 记录带走。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_UA = "AcgnRagBot/0.1 (CS5481 course project; +https://github.com/L01nki1/acgn-daily)"


def _env(key: str, default: str | None = None) -> str | None:
    value = os.environ.get(key)
    return value if value not in (None, "") else default


def _env_int(key: str, default: int) -> int:
    raw = _env(key)
    return int(raw) if raw is not None else default


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
    # 评测与建索引统一在此 device 上跑；跨机器数字可比的前提是 device 一致
    device: str = field(default_factory=lambda: _env("ACGN_EMBED_DEVICE", "cpu") or "cpu")
    batch_size: int = field(default_factory=lambda: _env_int("ACGN_EMBED_BATCH", 32))
    # e5 系列必须加前缀，否则召回系统性偏低（PLAN §3 第 1 条）
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "


@dataclass(frozen=True)
class LLMConfig:
    """抽取线与生成线允许用不同厂商：两组变量各自独立。"""

    base_url: str = field(default_factory=lambda: _env("ACGN_LLM_BASE_URL", "") or "")
    api_key: str = field(default_factory=lambda: _env("ACGN_LLM_API_KEY", "") or "")
    model: str = field(default_factory=lambda: _env("ACGN_LLM_MODEL", "") or "")
    model_version: str = field(default_factory=lambda: _env("ACGN_LLM_MODEL_VERSION", "") or "")
    temperature: float = 0.0

    # 抽取线（F，W2 批量；要便宜、JSON 稳、可重试）
    extract_base_url: str = field(
        default_factory=lambda: _env("ACGN_EXTRACT_BASE_URL", _env("ACGN_LLM_BASE_URL", "")) or ""
    )
    extract_api_key: str = field(
        default_factory=lambda: _env("ACGN_EXTRACT_API_KEY", _env("ACGN_LLM_API_KEY", "")) or ""
    )
    extract_model: str = field(
        default_factory=lambda: _env("ACGN_EXTRACT_MODEL", _env("ACGN_LLM_MODEL", "")) or ""
    )

    # 生成线（B，W3 起在线；要指令遵循 + 引用编号不编 + 拒答）
    answer_base_url: str = field(
        default_factory=lambda: _env("ACGN_ANSWER_BASE_URL", _env("ACGN_LLM_BASE_URL", "")) or ""
    )
    answer_api_key: str = field(
        default_factory=lambda: _env("ACGN_ANSWER_API_KEY", _env("ACGN_LLM_API_KEY", "")) or ""
    )
    answer_model: str = field(
        default_factory=lambda: _env("ACGN_ANSWER_MODEL", _env("ACGN_LLM_MODEL", "")) or ""
    )


@dataclass(frozen=True)
class HttpConfig:
    user_agent: str = field(default_factory=lambda: _env("ACGN_HTTP_UA", DEFAULT_UA) or DEFAULT_UA)
    timeout_s: float = field(default_factory=lambda: float(_env("ACGN_HTTP_TIMEOUT", "10") or "10"))
    # per-host 限速与全局并发（01 §5.2）：默认 1 req/2.5s + 抖动，全局并发 8–14
    min_interval_s: float = 2.5
    max_concurrency: int = field(default_factory=lambda: _env_int("ACGN_HTTP_CONCURRENCY", 8))
    respect_robots: bool = True


@dataclass(frozen=True)
class StoreConfig:
    backend: str = field(default_factory=lambda: _env("ACGN_STORE_BACKEND", "chroma") or "chroma")
    # collection 命名：snap_<snapshot_id[:12]>_<config>；只保留当前与上一快照
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
    """每次调用重新读环境，避免测试之间互相污染。"""
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
