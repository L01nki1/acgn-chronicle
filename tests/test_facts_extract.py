"""L0 extraction rules (RUNBOOK §3.6): batch, flat schema, retry, degrade, IR prefilter."""

from __future__ import annotations

import json
from typing import Any

import pytest

from acgn_rag.config import LLMConfig
from acgn_rag.core.facts.extract import (
    BATCH_SIZE,
    MIN_MAX_TOKENS,
    AnnouncementChunk,
    AnnouncementFacts,
    build_batch_prompt,
    expand_to_fact_updates,
    extract_facts,
    is_ir_release,
    make_openai_client,
)


class FakeLLMClient:
    """Offline LLMClient: pops canned responses and records every call (prompts + max_tokens)."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(self, system: str, user: str, *, max_tokens: int) -> str:
        self.calls.append({"system": system, "user": user, "max_tokens": max_tokens})
        return self.responses.pop(0)


def _chunk(index: int, text: str | None = None) -> AnnouncementChunk:
    return AnnouncementChunk(
        chunk_id=f"doc{index:02d}::config::0",
        text=text if text is not None else f"Announcement {index}: the new season starts soon.",
        url=f"https://example.com/news/{index}",
        publisher="Example Publisher",
    )


def _facts(index: int, **overrides: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "title": f"Work {index}",
        "release_date": "2026-10-05",
        "platforms": ["Switch"],
        "event_type": "release",
        "relevance": 0.9,
    }
    item.update(overrides)
    return item


def _payload(count: int) -> str:
    return json.dumps([_facts(i) for i in range(count)], ensure_ascii=False)


def test_is_ir_release_matches_corporate_keywords_only() -> None:
    for text in (
        "第2四半期決算短信",
        "株主向け説明会の開催について",
        "IRライブラリを更新しました",
        "quarterly ir briefing",
        "(IR)開示情報",
        "事業報告書を掲載",
    ):
        assert is_ir_release(text), text
    for text in (
        "New trailer for the third season",
        "Spirit Cadets gets an anime adaptation",
        "festival event announced for October",
    ):
        assert not is_ir_release(text), text


def test_build_batch_prompt_numbers_slots_and_keeps_system_stable() -> None:
    first = [_chunk(i) for i in range(1, BATCH_SIZE + 1)]
    second = [_chunk(i, text=f"other {i}") for i in range(BATCH_SIZE + 1, 13)]
    system_a, user_a = build_batch_prompt(first)
    system_b, user_b = build_batch_prompt(second)
    assert system_a == system_b  # identical across batches: prompt caching can act on it
    for index in range(1, BATCH_SIZE + 1):
        assert f"{index}. Announcement {index}" in user_a
    assert user_a.index("1. Announcement 1") < user_a.index("10. Announcement 10")
    assert f"exactly {BATCH_SIZE} objects" in user_a
    assert "exactly 2 objects" in user_b
    with pytest.raises(ValueError, match=r"1\.\.10"):
        build_batch_prompt([_chunk(i) for i in range(BATCH_SIZE + 2)])


def test_extract_facts_happy_path_maps_slots_to_chunks() -> None:
    chunks = [_chunk(i) for i in range(1, BATCH_SIZE + 1)]
    client = FakeLLMClient([_payload(len(chunks))])
    batch = extract_facts(chunks, client)
    assert batch.calls == 1
    assert batch.degraded_chunk_ids == []
    assert batch.ir_skipped_chunk_ids == []
    assert [facts.title for facts in batch.extracted] == [f"Work {i}" for i in range(len(chunks))]
    assert all(call["max_tokens"] >= MIN_MAX_TOKENS for call in client.calls)
    # slot i maps back to chunk i (mapping lives in code, never in the model's answer)
    rows = expand_to_fact_updates(batch.extracted, chunks)
    assert len(rows) == len(chunks) * 3  # release_date + one platform + event_type each
    date_rows = rows[0::3]
    for row, chunk in zip(date_rows, chunks, strict=True):
        assert (row.field, row.value) == ("release_date", "2026-10-05")
        assert row.chunk_id == chunk.chunk_id
        assert row.url == chunk.url
        assert row.publisher == chunk.publisher


def test_extract_facts_retries_the_whole_batch_on_bad_json() -> None:
    chunks = [_chunk(i) for i in range(1, 4)]
    client = FakeLLMClient(["this is not JSON at all", _payload(len(chunks))])
    batch = extract_facts(chunks, client)
    assert batch.calls == 2
    assert len(batch.extracted) == 3
    assert batch.degraded_chunk_ids == []


def test_extract_facts_retries_on_schema_violation() -> None:
    chunks = [_chunk(1), _chunk(2)]
    bad = json.dumps([{**_facts(0), "relevance": 5.0}, _facts(1)])  # outside [0, 1]
    client = FakeLLMClient([bad, _payload(2)])
    batch = extract_facts(chunks, client)
    assert batch.calls == 2
    assert len(batch.extracted) == 2
    assert batch.degraded_chunk_ids == []


def test_extract_facts_tolerates_a_fenced_json_block() -> None:
    fenced = "```json\n" + _payload(1) + "\n```"
    batch = extract_facts([_chunk(1)], FakeLLMClient([fenced]))
    assert batch.calls == 1
    assert batch.extracted[0].title == "Work 0"


def test_extract_facts_degrades_after_retries_and_drops_nothing() -> None:
    chunks = [_chunk(i) for i in range(1, 5)]
    chunks.append(_chunk(99, text="決算短信を発表しました"))
    client = FakeLLMClient(["nope", "still nope", "[{broken"])
    batch = extract_facts(chunks, client)
    assert batch.calls == 3  # 1 attempt + max_retries=2
    assert batch.extracted == []
    assert set(batch.degraded_chunk_ids) == {chunk.chunk_id for chunk in chunks[:4]}
    assert batch.ir_skipped_chunk_ids == [chunks[4].chunk_id]
    # every chunk is accounted for: degraded or IR-skipped, none dropped
    assert set(batch.degraded_chunk_ids) | set(batch.ir_skipped_chunk_ids) == {
        chunk.chunk_id for chunk in chunks
    }


def test_ir_items_skip_the_llm_entirely() -> None:
    mixed = [_chunk(1), _chunk(2, text="決算説明会資料を公開"), _chunk(3)]
    client = FakeLLMClient([_payload(2)])
    batch = extract_facts(mixed, client)
    assert len(client.calls) == 1
    # the IR text never reaches the client; the two candidates are numbered 1..2 in order
    assert all("決算" not in str(call["user"]) for call in client.calls)
    assert "1. Announcement 1" in client.calls[0]["user"]
    assert "2. Announcement 3" in client.calls[0]["user"]
    assert batch.ir_skipped_chunk_ids == [mixed[1].chunk_id]
    rows = expand_to_fact_updates(batch.extracted, [mixed[0], mixed[2]])
    assert {row.chunk_id for row in rows} == {mixed[0].chunk_id, mixed[2].chunk_id}


def test_unknown_values_never_reach_fact_rows() -> None:
    raw = json.dumps(
        [
            _facts(
                0,
                release_date="unknown",
                event_type="unknown",
                platforms=["unknown", "PS5", "PS5", " "],
            )
        ]
    )
    batch = extract_facts([_chunk(1)], FakeLLMClient([raw]))
    facts = batch.extracted[0]
    assert facts.release_date is None
    assert facts.event_type is None
    assert facts.platforms == ["PS5"]
    rows = expand_to_fact_updates(batch.extracted, [_chunk(1)])
    assert [(row.field, row.value) for row in rows] == [("platform", "PS5")]


def test_expand_to_fact_updates_pairs_every_row_with_its_chunk() -> None:
    chunks = [_chunk(1), _chunk(2)]
    items = [
        AnnouncementFacts(
            title="Work A", release_date="2026-10-05", platforms=["Switch", "PS5"], relevance=0.8
        ),
        AnnouncementFacts(title="Work B", platforms=[], relevance=0.1),  # no known facts
    ]
    rows = expand_to_fact_updates(items, chunks)
    assert len(rows) == 3
    platform_rows = [row for row in rows if row.field == "platform"]
    assert sorted(row.value for row in platform_rows) == ["PS5", "Switch"]
    for row in platform_rows:
        assert row.chunk_id == chunks[0].chunk_id
        assert row.url == chunks[0].url
        assert row.publisher == chunks[0].publisher
        assert row.media == "Work A"
    assert rows[0].field == "release_date"
    with pytest.raises(ValueError, match="cannot be paired"):
        expand_to_fact_updates(items[:1], chunks)


def test_make_openai_client_sends_extract_model_and_parses_content() -> None:
    pytest.importorskip("openai")
    respx = pytest.importorskip("respx")
    import httpx

    config = LLMConfig(
        extract_base_url="http://acgn-test/v1",
        extract_api_key="test-key",
        extract_model="test-extract-model",
    )
    canned = _payload(2)
    with respx.mock:
        route = respx.post("http://acgn-test/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "chatcmpl-1",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": canned},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        )
        # openai >= 3 vendors its own HTTP stack (httpx2), which respx's global patch does not
        # intercept; route the SDK through an httpx.MockTransport backed by the same respx
        # router. Zero real network, and the vendor request/response path is still exercised.
        client = make_openai_client(
            config,
            http_client=httpx.Client(transport=httpx.MockTransport(respx.mock.handler)),
        )
        out = client.complete("system prompt", "user prompt", max_tokens=MIN_MAX_TOKENS)
        assert [item["title"] for item in json.loads(out)] == ["Work 0", "Work 1"]

        body = json.loads(route.calls.last.request.content)
        assert body["model"] == "test-extract-model"
        assert body["max_tokens"] >= 1024  # DSV4-flash floor, RUNBOOK §3.4
        assert [message["role"] for message in body["messages"]] == ["system", "user"]

        # below the floor the adapter raises instead of clamping or sending a doomed request
        with pytest.raises(ValueError, match="max_tokens"):
            client.complete("system prompt", "user prompt", max_tokens=512)
