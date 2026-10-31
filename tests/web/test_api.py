"""Wire-level tests for the G1a KnowledgePort host.

Every assertion goes through HTTP shapes, not the mock's internals: the page is a consumer of the
contract, so the tests must fail if a route starts answering a shape the contract does not describe.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from acgn_rag.core.ports import (
    ChunkExcerpt,
    CorpusSnapshot,
    GroundedAnswer,
    RetrievalQuery,
    SnapshotInventory,
)
from acgn_rag.web.api import create_app
from acgn_rag.web.errors import NotServing
from acgn_rag.web.mock_port import (
    CURRENT_SNAPSHOT_ID,
    RETAINED_SNAPSHOT_ID,
    RETIRED_SNAPSHOT_ID,
)


class _NotServingBackend:
    """A port that can honestly answer "not serving" - the only source of a 503 here."""

    def inventory(self) -> SnapshotInventory:
        raise NotServing("No complete snapshot is being served.", retry_after_seconds=12)

    def snapshot(self, snapshot_id: str | None = None) -> CorpusSnapshot:
        raise NotServing("No complete snapshot is being served.", retry_after_seconds=12)

    def retrieve(self, query: RetrievalQuery) -> list[ChunkExcerpt]:
        raise NotServing("No complete snapshot is being served.", retry_after_seconds=12)

    def ask(self, query: RetrievalQuery) -> GroundedAnswer:
        raise NotServing("No complete snapshot is being served.", retry_after_seconds=12)


@pytest.fixture()
def client() -> Iterator[TestClient]:
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture()
def not_serving_client() -> Iterator[TestClient]:
    with TestClient(create_app(_NotServingBackend())) as test_client:
        yield test_client


def _query(text: str = "第二季什么时候开播？", **overrides: Any) -> dict[str, Any]:  # noqa: RUF001
    body: dict[str, Any] = {"text": text, "snapshot_id": CURRENT_SNAPSHOT_ID, "top_k": 5}
    body.update(overrides)
    return body


def test_snapshots_lists_current_first_and_retains_it(client: TestClient) -> None:
    payload = client.get("/snapshots").json()
    assert payload["current"] == CURRENT_SNAPSHOT_ID
    assert payload["retained"] == [CURRENT_SNAPSHOT_ID, RETAINED_SNAPSHOT_ID]
    assert payload["current"] in payload["retained"]


def test_snapshot_defaults_to_current(client: TestClient) -> None:
    payload = client.get("/snapshot").json()
    assert payload["snapshot_id"] == CURRENT_SNAPSHOT_ID
    assert payload["item_count"] == 8123


def test_retained_snapshot_is_served(client: TestClient) -> None:
    payload = client.get("/snapshot", params={"id": RETAINED_SNAPSHOT_ID}).json()
    assert payload["snapshot_id"] == RETAINED_SNAPSHOT_ID


def test_unknown_snapshot_is_404(client: TestClient) -> None:
    response = client.get("/snapshot", params={"id": "20260101-deadbeef0000"})
    assert response.status_code == 404
    assert "not known" in response.json()["detail"]


def test_retired_snapshot_is_410_not_404(client: TestClient) -> None:
    response = client.get("/snapshot", params={"id": RETIRED_SNAPSHOT_ID})
    assert response.status_code == 410
    assert "retired" in response.json()["detail"]


def test_retrieve_returns_contract_shaped_excerpts(client: TestClient) -> None:
    response = client.post("/retrieve", json=_query())
    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 5
    for item in payload:
        excerpt = ChunkExcerpt.model_validate(item)
        assert excerpt.citation.url
        assert len(excerpt.quote) <= 200
    assert payload[0]["chunk_id"] == "doc-90210::A::1"
    assert payload[0]["citation"]["publisher"] == "Aurora Animation"


def test_source_type_filter(client: TestClient) -> None:
    payload = client.post("/retrieve", json=_query(filters={"source_types": ["metadata"]})).json()
    assert [item["citation"]["source_type"] for item in payload] == ["metadata"]


def test_empty_source_type_list_means_no_filter(client: TestClient) -> None:
    payload = client.post("/retrieve", json=_query(filters={"source_types": []})).json()
    assert len(payload) == 5


def test_media_id_filter(client: TestClient) -> None:
    payload = client.post("/retrieve", json=_query(filters={"media_ids": [77123]})).json()
    assert [item["chunk_id"] for item in payload] == ["doc-77123::A::1"]


def test_media_id_filter_with_no_match_is_an_empty_list(client: TestClient) -> None:
    response = client.post("/retrieve", json=_query(filters={"media_ids": [12345]}))
    assert response.status_code == 200
    assert response.json() == []


def test_date_boundaries_are_half_open(client: TestClient) -> None:
    since_payload = client.post("/retrieve", json=_query(filters={"since": "2026-09-20"})).json()
    assert [item["citation"]["published_at"] for item in since_payload] == [
        "2026-09-20",
        "2026-09-25",
    ]

    until_payload = client.post("/retrieve", json=_query(filters={"until": "2026-09-13"})).json()
    assert [item["citation"]["published_at"] for item in until_payload] == [
        "2026-09-12",
        "2026-09-13",
        "2026-09-01",
    ]


def test_same_day_range_keeps_that_day(client: TestClient) -> None:
    payload = client.post(
        "/retrieve",
        json=_query(filters={"since": "2026-09-25", "until": "2026-09-25"}),
    ).json()
    assert [item["citation"]["published_at"] for item in payload] == ["2026-09-25"]


def test_invalid_date_is_422(client: TestClient) -> None:
    response = client.post("/retrieve", json=_query(filters={"since": "not-a-date"}))
    assert response.status_code == 422
    assert "ISO" in response.json()["detail"]


def test_inverted_adjacent_dates_are_422(client: TestClient) -> None:
    response = client.post(
        "/retrieve",
        json=_query(filters={"since": "2026-10-05", "until": "2026-10-04"}),
    )
    assert response.status_code == 422
    assert "since" in response.json()["detail"]


def test_invalid_filter_is_422_even_with_empty_text(client: TestClient) -> None:
    response = client.post(
        "/retrieve",
        json=_query(text="   ", filters={"since": "2026-10-05", "until": "2026-10-04"}),
    )
    assert response.status_code == 422


def test_top_k_truncates(client: TestClient) -> None:
    payload = client.post("/retrieve", json=_query(top_k=1)).json()
    assert len(payload) == 1
    assert payload[0]["score"] == pytest.approx(0.93)


def test_top_k_zero_returns_no_excerpts(client: TestClient) -> None:
    response = client.post("/retrieve", json=_query(top_k=0))
    assert response.status_code == 200
    assert response.json() == []


def test_empty_text_returns_no_excerpts(client: TestClient) -> None:
    response = client.post("/retrieve", json=_query(text="   "))
    assert response.status_code == 200
    assert response.json() == []


def test_malformed_json_body_is_400(client: TestClient) -> None:
    response = client.post(
        "/retrieve",
        content=b"{",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Malformed request body."


def test_missing_required_field_is_400(client: TestClient) -> None:
    response = client.post("/retrieve", json={"snapshot_id": CURRENT_SNAPSHOT_ID})
    assert response.status_code == 400


def test_ask_returns_grounded_answer_with_citations(client: TestClient) -> None:
    payload = client.post("/ask", json=_query()).json()
    answer = GroundedAnswer.model_validate(payload)
    assert answer.abstained is False
    assert len(answer.citations) == 5
    assert set(answer.cited_chunk_ids) == {item.chunk_id for item in answer.citations}


def test_ask_with_no_evidence_abstains_instead_of_returning_empty(client: TestClient) -> None:
    payload = client.post("/ask", json=_query(filters={"media_ids": [12345]})).json()
    answer = GroundedAnswer.model_validate(payload)
    assert answer.abstained is True
    assert answer.citations == []
    assert answer.cited_chunk_ids == []
    assert answer.abstain_reason


def test_ask_private_question_abstains(client: TestClient) -> None:
    payload = client.post("/ask", json=_query(text="导演的私人电话是多少？")).json()  # noqa: RUF001
    answer = GroundedAnswer.model_validate(payload)
    assert answer.abstained is True
    assert answer.citations == []


def test_ask_empty_text_abstains(client: TestClient) -> None:
    payload = client.post("/ask", json=_query(text="   ")).json()
    answer = GroundedAnswer.model_validate(payload)
    assert answer.abstained is True
    assert answer.abstain_reason


def test_ask_invalid_filter_is_422(client: TestClient) -> None:
    response = client.post(
        "/ask", json=_query(filters={"since": "2026-10-05", "until": "2026-10-04"})
    )
    assert response.status_code == 422


def test_daily_is_501_until_g3(client: TestClient) -> None:
    response = client.get(
        "/daily", params={"snapshot_id": CURRENT_SNAPSHOT_ID, "date": "2026-10-30"}
    )
    assert response.status_code == 501


def test_not_serving_is_503_with_retry_after(not_serving_client: TestClient) -> None:
    response = not_serving_client.get("/snapshots")
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "12"
    assert "No complete snapshot" in response.json()["detail"]


def test_api_reads_are_no_store(client: TestClient) -> None:
    for response in (client.get("/snapshots"), client.get("/snapshot")):
        assert response.headers["Cache-Control"] == "no-store"


def test_index_and_module_are_served(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200
    assert 'id="query-form"' in page.text
    assert "example.com" in page.text

    module = client.get("/static/app.mjs")
    assert module.status_code == 200
    assert "javascript" in module.headers["Content-Type"]
    assert "selectRenderableExcerpts" in module.text
