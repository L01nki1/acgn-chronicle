"""Production input limits, cache identity and per-item extraction recovery."""

from __future__ import annotations

import json
from typing import ClassVar

import pytest

from acgn_rag.config import EmbedConfig
from acgn_rag.core.chunk.strategy_a import chunk_text
from acgn_rag.core.embed.encoder import Embedder, EncoderIdentity, build_header, cache_key
from acgn_rag.core.facts.extract import AnnouncementChunk, expand_to_fact_updates, extract_facts


class CharacterCounter:
    def count(self, text: str) -> int:
        return len(text)

    def ids(self, text: str) -> list[int]:
        return [1, *map(ord, text), 2]

    def decode(self, ids) -> str:
        return "".join(chr(value) for value in ids if value > 2)


@pytest.mark.parametrize("variant", ["A", "A'", "A0"])
@pytest.mark.parametrize(
    "body",
    [
        "新作の放送を発表しました。" * 100,
        "新作品公布上映日期。" * 100,
        "A new release is announced. " * 100,
    ],
)
def test_composed_input_fits_context(variant: str, body: str) -> None:
    counter = CharacterCounter()
    header = build_header("新作アニメーションの放送開始情報", "公式出版社", "2026-10-28")
    chunks = chunk_text("doc", body, variant, counter, header=header)
    assert chunks
    assert all(
        len(counter.ids("passage: " + header + "\n" + chunk.text)) <= 512 for chunk in chunks
    )


def test_full_chunk_pays_for_the_header() -> None:
    counter = CharacterCounter()
    header = "日本語の題名 | 出版社 | 2026-10-28"
    chunks = chunk_text("doc", "x" * 900, "A0", counter, header=header)
    assert len(counter.ids("passage: " + header + "\n" + chunks[0].text)) == 512


def test_header_that_leaves_no_content_budget_is_rejected() -> None:
    with pytest.raises(ValueError, match="budget"):
        chunk_text("doc", "body", "A0", CharacterCounter(), header="x" * 512)


def test_batch_size_does_not_invalidate_vector_cache() -> None:
    first = EncoderIdentity(model="intfloat/multilingual-e5-base", batch_size=32)
    second = EncoderIdentity(model=first.model, batch_size=64)
    assert first.slug() == second.slug()
    assert cache_key(first, first.model, "passage: ", "header", "body") == cache_key(
        second, second.model, "passage: ", "header", "body"
    )
    assert first.to_metadata()["batch_size"] == 32
    assert second.to_metadata()["batch_size"] == 64


def test_token_boundary_expansion_is_repacked() -> None:
    class BoundaryCounter(CharacterCounter):
        def ids(self, text: str) -> list[int]:
            ids = super().ids(text)
            return [*ids, *([3] * 7 if "\nx" in text else [])]

    counter = BoundaryCounter()
    chunks = chunk_text("doc", "x" * 900, "A0", counter, header="header")
    assert all(len(counter.ids("passage: header\n" + chunk.text)) <= 512 for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == "x" * 900


def test_model_limit_and_named_prompt_are_measured_before_encoding() -> None:
    from types import SimpleNamespace

    class Model:
        max_seq_length = 16
        prompts: ClassVar[dict[str, str]] = {"document": "document context: "}
        tokenizer = staticmethod(lambda text, **_kwargs: SimpleNamespace(input_ids=list(text)))

        def encode(self, *_args, **_kwargs):
            pytest.fail("oversized input must fail before reaching the model")

    embedder = Embedder(EmbedConfig(model_name="google/embeddinggemma-300m"), model=Model())
    assert embedder.context_limit(512) == 16
    assert embedder.passage_context() == "document context: "
    with pytest.raises(ValueError, match="exceeds max_seq_length"):
        embedder.encode_passages([("header", "body")])


def test_id_mapping_rejects_duplicate_or_missing_sources() -> None:
    from acgn_rag.core.facts.extract import AnnouncementFacts

    chunk = AnnouncementChunk(chunk_id="one", text="body", url="https://example.com", publisher="P")
    fact = AnnouncementFacts(title="Title", relevance=1)
    with pytest.raises(ValueError, match="duplicate"):
        expand_to_fact_updates([("one", fact), ("one", fact)], [chunk])
    with pytest.raises(ValueError, match="unknown"):
        expand_to_fact_updates([("other", fact)], [chunk])


def test_per_item_recovery_preserves_other_facts_and_provenance() -> None:
    chunks = [
        AnnouncementChunk(
            chunk_id=f"chunk-{i}",
            text=f"Announcement {i}",
            url=f"https://example.com/{i}",
            publisher="Publisher",
        )
        for i in range(10)
    ]
    responses = ["not json", "not json"]
    responses.extend(
        "bad json"
        if i == 4
        else json.dumps([{"title": f"Work {i}", "release_date": "2026-10-28", "relevance": 0.9}])
        for i in range(10)
    )

    class Client:
        def complete(self, *_args, **_kwargs):
            return responses.pop(0)

    batch = extract_facts(chunks, Client())
    assert batch.calls == 12
    assert batch.retries == 1
    assert batch.single_calls == 10
    assert batch.degraded_chunk_ids == ["chunk-4"]
    assert len(batch.extracted) == 9
    rows = expand_to_fact_updates(batch.extracted, chunks)
    assert {row.chunk_id for row in rows} == {chunk.chunk_id for chunk in chunks} - {"chunk-4"}
    assert all(row.url == f"https://example.com/{row.chunk_id.split('-')[1]}" for row in rows)
    assert not responses
