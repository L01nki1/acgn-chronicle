"""Guards on the mock itself: the page's evidence must stay obviously fictional and contract-legal."""

from __future__ import annotations

import re
from urllib.parse import urlparse

import pytest

from acgn_rag.core.ports import ChunkExcerpt
from acgn_rag.web.errors import RetiredSnapshot, UnknownSnapshot
from acgn_rag.web.mock_port import (
    CURRENT_SNAPSHOT_ID,
    MOCK_RECORDS,
    RETAINED_SNAPSHOT_ID,
    RETIRED_SNAPSHOT_ID,
    MockKnowledgePort,
)

SNAPSHOT_ID = re.compile(r"^\d{8}-[0-9a-f]{12}$")


def test_snapshot_ids_follow_the_documented_shape() -> None:
    for snapshot_id in (CURRENT_SNAPSHOT_ID, RETAINED_SNAPSHOT_ID, RETIRED_SNAPSHOT_ID):
        assert SNAPSHOT_ID.match(snapshot_id)


def test_mock_quotes_fit_the_contract_budget() -> None:
    for record in MOCK_RECORDS:
        excerpt = ChunkExcerpt.model_validate(record.excerpt.model_dump())
        assert 0 < len(excerpt.quote) <= 200


def test_mock_urls_are_fictional_and_attributable() -> None:
    for record in MOCK_RECORDS:
        citation = record.excerpt.citation
        parsed = urlparse(citation.url)
        assert parsed.scheme == "https"
        assert parsed.netloc == "example.com"
        assert citation.publisher
        assert citation.published_at
        assert citation.chunk_id == record.excerpt.chunk_id


def test_mock_retired_and_unknown_snapshots_are_distinct() -> None:
    port = MockKnowledgePort()
    with pytest.raises(RetiredSnapshot):
        port.snapshot(RETIRED_SNAPSHOT_ID)
    with pytest.raises(UnknownSnapshot):
        port.snapshot("20260101-deadbeef0000")
