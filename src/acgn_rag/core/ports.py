"""KnowledgePort 契约类型 —— 全项目唯一真相（05 §2）。

负责人：B（契约变更仲裁）　|　依据：05 §2
冻结纪律（05 §5）：
  1. 本文件是唯一真相；`contracts/knowledge_port.schema.json` 由 `scripts/export_schema.py` 导出，
     CI 校验二者一致，不一致即失败。
  2. 字段语义变更（改名 / 改类型 / 改必填）必须 PR 说明理由 → 重新导出 schema → 通知 C（web）与 F（Isla 对接）。
  3. 新增字段一律 Optional + 默认值，保证旧 consumer 不破。
  4. W2 起禁止破坏性变更；W3 起完全冻结（含新增字段）。

禁止在此文件写任何实现逻辑：这里只有类型。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

SourceType = Literal["official", "media", "metadata"]

SCHEMA_VERSION = "0.1.0"


class CorpusSnapshot(BaseModel):
    """一次冻结的语料状态。语料更新产生新快照，不修改旧快照。"""

    snapshot_id: str = Field(..., description="内容哈希 + 日期；同语料重建必须一致")
    created_at: str
    item_count: int
    deduped_count: int = Field(..., description="去重后条目数 —— 一切指标的口径")
    source_count: int
    window_start: str = Field(..., description="语料时间跨度，评估可复现的必要字段")
    window_end: str


class RetrievalFilters(BaseModel):
    source_types: Sequence[SourceType] | None = None
    since: str | None = Field(None, description="日报“只取今天的”也走这里，禁止新增时间分桶索引")
    until: str | None = None
    media_ids: Sequence[int] | None = None


class RetrievalQuery(BaseModel):
    text: str
    top_k: int = 5
    snapshot_id: str = Field(..., description="绑定快照，保证可复现")
    filters: RetrievalFilters | None = None


class Citation(BaseModel):
    """结构化引用，不是拼接的字符串。"""

    source_type: SourceType = Field(..., description="决定前端展示强度与合规裁剪")
    publisher: str = Field(..., description="'Capcom' | 'ANN' | 'AniList' ...")
    title: str
    url: str
    published_at: str
    chunk_id: str = Field(..., description="引用必须能定位到块")


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    snapshot_id: str
    text: str = Field(
        ...,
        description="内部使用；对外由 ground/serialize 裁剪（≤200 字符片段）",
    )
    score: float
    citation: Citation


class GroundedAnswer(BaseModel):
    answer: str
    citations: Sequence[Citation]
    cited_chunk_ids: Sequence[str]
    abstained: bool
    abstain_reason: str | None = None


# ---- 附加任务端口（接口先行，本阶段只定义不实现完整逻辑）----


class DailyReportPort(BaseModel):
    """日报生成入口（后置实现）"""

    snapshot_id: str
    date: str


class CategoryDigest(BaseModel):
    category: str = Field(..., description="anime / manga / game / novel")
    new_count: int
    cluster_count: int


class ClusterCard(BaseModel):
    """日报热点簇：一张卡片 = 一个去重簇"""

    cluster_id: str
    size: int = Field(..., description="簇大小 = 热度代理")
    media_ids: Sequence[int]
    representative_title: str
    first_seen: str
    citations: Sequence[Citation] = Field(..., description="复用 Citation；前端走合规裁剪")


class FactUpdate(BaseModel):
    """L0 当日新增结构化事实"""

    media: str
    field: str = Field(..., description="发售日 / 平台 / 事件类型 ...")
    value: str
    chunk_id: str
    url: str
    publisher: str


class DailyDigest(BaseModel):
    """日报按日候选集（看板 / 阶段二文字版账号日报共用）。

    按日取数仍用 RetrievalFilters.since/until，禁止新增时间分桶索引（02 不变量 4）。
    """

    snapshot_id: str
    date: str
    by_category: Sequence[CategoryDigest]
    top_clusters: Sequence[ClusterCard]
    fact_updates: Sequence[FactUpdate]
    episode_table: Sequence[dict] = Field(..., description="B站时间表直通")
    source_coverage: dict[str, int] = Field(
        ..., description="per-source 当日条数（01 §5.5 采集监控复用）"
    )


__all__ = [
    "SCHEMA_VERSION",
    "CategoryDigest",
    "Citation",
    "ClusterCard",
    "CorpusSnapshot",
    "DailyDigest",
    "DailyReportPort",
    "FactUpdate",
    "GroundedAnswer",
    "RetrievalFilters",
    "RetrievalQuery",
    "RetrievedChunk",
    "SourceType",
]
