"""Config-A chunker tests: budget, packing, overlap, ids - all offline except one e5 check.

The fast suite runs on FakeTokenizer: whitespace tokens, two specials around the content
ids, and ":" split off as its own token so count("passage: ") == 2 - the same content
width as the real model, which makes the budget exactly the e5 shape (508 at 512) without
a model. The one real-tokenizer test is opt-in and offline-only; run it with:

    ACGN_EMBED_SLOW=1 HF_HUB_OFFLINE=1 \
    HF_HOME=$HOME/LocalDev/cityu/hf-cache/2026-2027_SemA/cs5481-acgn-chronicle \
    pytest -q tests/test_chunk_a.py -k e5

(the HF_HOME value is the LocalDev cache e5 was downloaded into; any offline cache holding
intfloat/multilingual-e5-base works - CI never downloads models).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from itertools import pairwise
from typing import Any

import pytest

from acgn_rag.core.chunk import CHUNK_VARIANTS, budget_for, chunk_text
from acgn_rag.core.chunk.strategy_a import split_sentences

PASSAGE_PREFIX = "passage: "


class FakeTokenizer:
    """Whitespace tokenizer shaped like e5: ids() wraps the content ids in 2 specials.

    ":" splits off as its own token so count("passage: ") == 2 content tokens, mirroring
    the real model; ids() then measures exactly 4 for the prefix (BOS + 2 + EOS). Ids are
    assigned by first sight (insertion order), so they are stable across runs without
    depending on Python's randomized hashing.
    """

    BOS = 0
    EOS = 2

    def __init__(self) -> None:
        self._id_of: dict[str, int] = {}
        self._word_of: dict[int, str] = {}

    def _word_ids(self, text: str) -> list[int]:
        ids: list[int] = []
        for word in text.split():
            for part in self._split_colon(word):
                if part not in self._id_of:
                    self._id_of[part] = 10 + len(self._id_of)
                    self._word_of[self._id_of[part]] = part
                ids.append(self._id_of[part])
        return ids

    @staticmethod
    def _split_colon(word: str) -> list[str]:
        parts: list[str] = []
        for index, piece in enumerate(word.split(":")):
            if index:
                parts.append(":")
            if piece:
                parts.append(piece)
        return parts

    def count(self, text: str) -> int:
        return len(self._word_ids(text))

    def ids(self, text: str) -> list[int]:
        return [self.BOS, *self._word_ids(text), self.EOS]

    def decode(self, ids: Sequence[int]) -> str:
        return " ".join(self._words_of_ids(ids))

    def _words_of_ids(self, ids: Sequence[int]) -> list[str]:
        return [self._word_of[i] for i in ids if i in self._word_of]


class OpaqueTokenizer:
    """Only count/ids: forces the no-decode fallback paths (halving, word-suffix overlap)."""

    def __init__(self) -> None:
        self._inner = FakeTokenizer()

    def count(self, text: str) -> int:
        return self._inner.count(text)

    def ids(self, text: str) -> list[int]:
        return self._inner.ids(text)


class HFCounter:
    """TokenCounter over a real HF tokenizer: ids() keeps the model specials, count() drops them."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def count(self, text: str) -> int:
        return len(self._tokenizer(text, add_special_tokens=False).input_ids)

    def ids(self, text: str) -> list[int]:
        return self._tokenizer(text).input_ids

    def decode(self, ids: Sequence[int]) -> str:
        return self._tokenizer.decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )


def _paragraphs(count: int, words: int) -> str:
    """`count` sentences of `words` whitespace tokens each, terminated by ASCII periods."""
    return " ".join(" ".join(f"s{i}w{j}" for j in range(words)) + "." for i in range(count))


def _one_long_sentence(words: int) -> str:
    """A single sentence with no terminator: forces the oversize-split path."""
    return " ".join(f"w{i}" for i in range(words))


def test_variant_table_is_the_three_parameter_sets() -> None:
    assert set(CHUNK_VARIANTS) == {"A", "A'", "A0"}
    assert (CHUNK_VARIANTS["A"].max_tokens, CHUNK_VARIANTS["A"].overlap) == (512, 64)
    assert (CHUNK_VARIANTS["A'"].max_tokens, CHUNK_VARIANTS["A'"].overlap) == (256, 0)
    assert (CHUNK_VARIANTS["A0"].max_tokens, CHUNK_VARIANTS["A0"].overlap) == (512, 0)


def test_budget_formula_pays_for_prefix_specials() -> None:
    fake = FakeTokenizer()
    assert fake.count(PASSAGE_PREFIX) == 2
    assert len(fake.ids(PASSAGE_PREFIX)) == 4
    assert budget_for(512, 512, PASSAGE_PREFIX, fake) == 508  # the e5 shape: 512 - 4 ids
    assert budget_for(256, 512, PASSAGE_PREFIX, fake) == 256  # the config binds ...
    assert budget_for(512, 8192, PASSAGE_PREFIX, fake) == 512  # ... the model only caps
    assert budget_for(512, 128, PASSAGE_PREFIX, fake) == 124
    # even an empty prefix pays for the model's specials
    assert budget_for(512, 512, "", fake) == 510


def test_budget_sized_chunk_encodes_to_exactly_max_seq_ids() -> None:
    """CI-runnable form of the RUNBOOK §3.4 acceptance, on the fake tokenizer."""
    fake = FakeTokenizer()
    body = _one_long_sentence(508)  # one sentence, exactly the A/A0 budget
    for variant in ("A", "A0"):
        chunks = chunk_text("doc:acc", body, variant, fake)
        assert len(chunks) == 1
        assert chunks[0].token_count == 508
        assert len(fake.ids(PASSAGE_PREFIX + chunks[0].text)) == 512


def test_packing_respects_budget_for_every_variant() -> None:
    fake = FakeTokenizer()
    text = _paragraphs(60, 25)  # 1500 content tokens
    chunk_counts: dict[str, int] = {}
    for variant in ("A", "A'", "A0"):
        budget = budget_for(CHUNK_VARIANTS[variant].max_tokens, 512, PASSAGE_PREFIX, fake)
        chunks = chunk_text("doc:pack", text, variant, fake)
        assert chunks
        for chunk in chunks:
            assert chunk.text, chunk.chunk_id
            assert chunk.token_count <= budget, chunk.chunk_id
            assert fake.count(chunk.text) == chunk.token_count  # no drift
        chunk_counts[variant] = len(chunks)
    assert chunk_counts["A'"] > chunk_counts["A0"]  # 256 budget needs more chunks than 508


def test_overlap_repeats_the_trailing_64_tokens() -> None:
    fake = FakeTokenizer()
    chunks = chunk_text("doc:ov", _paragraphs(40, 30), "A", fake)
    assert len(chunks) >= 3
    for previous, current in pairwise(chunks):
        head = fake.ids(current.text)[1:-1][:64]
        tail = fake.ids(previous.text)[1:-1][-64:]
        assert head == tail
        assert current.token_count > 64  # carries new content, not just the head


def test_overlap_head_is_dropped_when_it_cannot_share_the_next_chunk() -> None:
    """The documented drop rule (strategy_a ``_next_parts``): when the 64-token head plus the
    next piece exceed the budget, the head is discarded and the next chunk carries only new
    content - never a head-only chunk duplicating the previous chunk's tail verbatim."""
    fake = FakeTokenizer()
    first = " ".join(f"a{i}" for i in range(445)) + "."  # fits alone (445 <= 508)
    second = " ".join(f"b{i}" for i in range(490)) + "."  # fits alone, head + piece would not
    third = "closing sentence here."  # small enough to pack after the second sentence
    chunks = chunk_text("doc:drop", f"{first} {second} {third}", "A", fake)

    assert [chunk.token_count for chunk in chunks] == [445, 493]  # 490 + 3, no head in chunk 2
    assert [chunk.chunk_id for chunk in chunks] == ["doc:drop::A::1", "doc:drop::A::2"]
    head = fake.ids(chunks[0].text)[1:-1][-64:]  # the trailing 64 content tokens of chunk 1
    opening = fake.ids(chunks[1].text)[1:-1][:64]
    assert opening != head  # the head was dropped, not carried into chunk 2
    tail_words = chunks[0].text.split()[-64:]
    assert chunks[1].text.split()[:64] != tail_words
    for chunk in chunks:
        assert chunk.token_count <= 508  # every chunk stays within the budget


def test_oversize_sentence_splits_at_token_boundaries() -> None:
    fake = FakeTokenizer()
    chunks = chunk_text("doc:big", _one_long_sentence(1200), "A0", fake)
    assert [chunk.token_count for chunk in chunks] == [508, 508, 184]
    # A0 has no overlap: the chunks are exactly the original words, in order
    assert " ".join(chunk.text for chunk in chunks).split() == _one_long_sentence(1200).split()


def test_oversize_sentence_keeps_overlap_across_the_token_cut() -> None:
    fake = FakeTokenizer()
    chunks = chunk_text("doc:big", _one_long_sentence(1200), "A", fake)
    assert [chunk.token_count for chunk in chunks] == [508, 508, 312]
    # sum = original content + one 64-token overlap per boundary after the first
    assert sum(chunk.token_count for chunk in chunks) == 1200 + 64 * (len(chunks) - 1)


def test_chunk_ids_are_one_based_and_variant_tagged() -> None:
    fake = FakeTokenizer()
    chunks = chunk_text("capcom-001", _paragraphs(20, 40), "A'", fake)
    assert len(chunks) >= 2
    assert [chunk.chunk_id for chunk in chunks] == [
        f"capcom-001::A'::{index}" for index in range(1, len(chunks) + 1)
    ]


def test_identical_input_gives_identical_output() -> None:
    text = _paragraphs(25, 35)
    first = chunk_text("doc:d", text, "A", FakeTokenizer())
    second = chunk_text("doc:d", text, "A", FakeTokenizer())
    assert first == second


def test_no_decode_tokenizer_falls_back_to_deterministic_halving() -> None:
    body = _one_long_sentence(1200)
    chunks = chunk_text("doc:fallback", body, "A0", OpaqueTokenizer())
    assert len(chunks) >= 3
    for chunk in chunks:
        assert chunk.token_count <= 508
    again = chunk_text("doc:fallback", body, "A0", OpaqueTokenizer())
    assert again == chunks


def test_unknown_variant_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown chunk variant"):
        chunk_text("doc:x", "hello world.", "B", FakeTokenizer())


def test_empty_and_blank_text_produce_no_chunks() -> None:
    fake = FakeTokenizer()
    assert chunk_text("doc:e", "", "A", fake) == []
    assert chunk_text("doc:e", " \n\n ", "A", fake) == []


def test_split_sentences_ja_ideographic_stops() -> None:
    assert split_sentences("アニメは終わった。新しい季が始まる。") == [
        "アニメは終わった。",
        "新しい季が始まる。",
    ]


def test_split_sentences_zh_marks() -> None:
    assert split_sentences("新版上线了！销量增长200%。更多内容即将推出。") == [  # noqa: RUF001
        "新版上线了！",  # noqa: RUF001
        "销量增长200%。",
        "更多内容即将推出。",
    ]


def test_split_sentences_en_period_but_not_decimals() -> None:
    assert split_sentences("Sales grew 3.5% in Q4. The sequel follows.") == [
        "Sales grew 3.5% in Q4.",
        "The sequel follows.",
    ]
    assert split_sentences("It costs 3.5. Then more.") == ["It costs 3.5.", "Then more."]


def test_split_sentences_ascii_marks_and_ellipsis_runs() -> None:
    assert split_sentences("Really? Yes! Done… at last.") == [
        "Really?",
        "Yes!",
        "Done…",  # the ellipsis is a terminator, like the fullwidth marks
        "at last.",
    ]
    assert split_sentences("そうですか……なるほど。") == ["そうですか……", "なるほど。"]


def test_split_sentences_newlines_and_blank_lines() -> None:
    assert split_sentences("line one\nline two\r\n\r\nline three") == [
        "line one",
        "line two",
        "line three",
    ]


def test_split_sentences_keeps_closing_quote_with_its_sentence() -> None:
    assert split_sentences("「終わった。」次の話へ。") == ["「終わった。」", "次の話へ。"]


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("ACGN_EMBED_SLOW") != "1",
    reason="set ACGN_EMBED_SLOW=1 (plus HF_HUB_OFFLINE=1 and the local HF cache) to run",
)
def test_real_e5_budget_chunk_encodes_to_exactly_512_ids() -> None:
    """The pre-registered RUNBOOK §3.4 acceptance, on the real e5 tokenizer (offline).

    Invocation (see the module docstring):
        ACGN_EMBED_SLOW=1 HF_HUB_OFFLINE=1 \
        HF_HOME=$HOME/LocalDev/cityu/hf-cache/2026-2027_SemA/cs5481-acgn-chronicle \
        pytest -q tests/test_chunk_a.py -k e5
    """
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")
    counter = HFCounter(tokenizer)
    assert len(tokenizer(PASSAGE_PREFIX).input_ids) == 4  # the RUNBOOK §3.4 measurement
    budget = budget_for(CHUNK_VARIANTS["A"].max_tokens, 512, PASSAGE_PREFIX, counter)
    assert budget == 508
    bodies = {
        "ja": "アニメ" * 800,
        "zh": "动画制作委员会宣布新作" * 600,
        "en": _one_long_sentence(900),
    }
    for language, body in bodies.items():
        chunks = chunk_text(f"doc:e5-{language}", body, "A0", counter, 512, PASSAGE_PREFIX)
        full = [chunk for chunk in chunks if chunk.token_count == budget]
        assert full, f"{language}: no chunk reached the budget: {[c.token_count for c in chunks]}"
        encoded = counter.ids(PASSAGE_PREFIX + full[0].text)
        assert len(encoded) == 512, f"{language}: {len(encoded)} ids for a budget-sized chunk"
