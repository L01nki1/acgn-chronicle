"""A failed translation lane must be visible, not just a `used: false` on every row.

The delivery round exposed this the expensive way: one collaborator's translator client lacked a
provider session header, so every Chinese question fell back to its original text. The archive
said `used: false` honestly, but nothing distinguished "translation unavailable here" from "the
translator never worked", and `meta.query_transform` still claimed `translate-zh`.
"""

from __future__ import annotations

import dataclasses

import pytest

from acgn_rag.core.retrieve.translate_query import (
    TRANSLATION_MISS_REASONS,
    TranslatedQuery,
    TranslationAttempt,
    attempt_translation,
    translate_query,
)


class _Completion:
    def __init__(self, text: str, finish_reason: str = "stop") -> None:
        self.text = text
        self.finish_reason = finish_reason


class _Client:
    def __init__(self, result=None) -> None:
        self.result = result
        self.calls = 0

    def complete(self, system: str, user: str, *, max_tokens: int):
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_success_reports_ok_and_the_phrasing() -> None:
    client = _Client(_Completion("When was the new chapter announced?"))

    attempt = attempt_translation("新章节什么时候公开？", client)

    assert attempt.reason == "ok"
    assert attempt.used
    assert attempt.translated is not None
    assert attempt.translated.english == "When was the new chapter announced?"


def test_a_provider_failure_is_recorded_with_its_exception_text() -> None:
    """The case that actually happened: a transport error, swallowed, leaving no trace."""
    client = _Client(RuntimeError("InternalServerError: Error code: 530"))

    attempt = attempt_translation("新章节什么时候公开？", client)

    assert attempt.translated is None
    assert attempt.reason == "provider_error"
    assert "530" in (attempt.detail or "")
    assert "RuntimeError" in (attempt.detail or "")


def test_an_empty_completion_is_distinct_from_a_dead_provider() -> None:
    client = _Client(_Completion("", finish_reason="length"))

    attempt = attempt_translation("新章节什么时候公开？", client)

    assert attempt.reason == "empty_translation"
    assert "length" in (attempt.detail or "")


def test_an_empty_question_is_its_own_reason_and_costs_no_call() -> None:
    client = _Client(_Completion("unused"))

    attempt = attempt_translation("   ", client)

    assert attempt.reason == "empty_question"
    assert client.calls == 0


def test_every_reason_is_declared() -> None:
    """The vocabulary is closed, so an archive can group by it without surprises."""
    seen = set()
    for client, question in (
        (_Client(_Completion("ok line")), "问题"),
        (_Client(RuntimeError("boom")), "问题"),
        (_Client(_Completion("")), "问题"),
        (_Client(_Completion("x")), ""),
    ):
        seen.add(attempt_translation(question, client).reason)

    assert seen == set(TRANSLATION_MISS_REASONS)


def test_the_wrapper_keeps_its_old_behaviour_exactly() -> None:
    """`translate_query` must be untouched for callers that only want the text.

    This is what makes the change safe to land without rerunning the evaluation: on the success
    path the returned object and the retrieval text are byte-identical to before, and the only
    new information appears on paths that previously returned `None`.
    """
    client = _Client(_Completion("When was it announced?"))
    attempt = attempt_translation("问题", client)
    assert attempt.translated is not None
    assert translate_query("问题", client) == attempt.translated

    for failing in (_Client(RuntimeError("boom")), _Client(_Completion(""))):
        assert translate_query("问题", failing) is None


def test_attempt_is_frozen_and_reports_use() -> None:
    translated = TranslatedQuery(original="问", english="Q")
    attempt = TranslationAttempt(translated, "ok")

    assert attempt.used
    assert TranslationAttempt(None, "provider_error").used is False
    with pytest.raises(dataclasses.FrozenInstanceError):
        attempt.reason = "changed"  # type: ignore[misc]
