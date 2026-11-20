"""Observable behaviour of the query-translation lane's reply cleaning.

The lane decides only the retrieval text (the answerer always receives the original question),
but a malformed reply still costs a retrieval miss, so the parsing is pinned here.
"""

from __future__ import annotations

import pytest

from acgn_rag.core.retrieve.translate_query import (
    TRANSLATE_SYSTEM_PROMPT,
    TranslatedQuery,
    clean_translation,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # A plain one-line reply is used as-is.
        ("Nana Mizuki voice actor role", "Nana Mizuki voice actor role"),
        # A fenced reply is unwrapped to the text between the fences. Taking the last split
        # segment made this empty, silently losing the translation.
        ("```\nNana Mizuki voice actor role\n```", "Nana Mizuki voice actor role"),
        ("```Nana Mizuki role```", "Nana Mizuki role"),
        # Prose after the closing fence must not become the query.
        ("```\nNana Mizuki role\n```\nHere is your query.", "Nana Mizuki role"),
        # A truncated reply with no closing fence keeps its body.
        ("```Nana Mizuki role", "Nana Mizuki role"),
        # Surrounding quotes and backticks are stripped.
        ('"Nana Mizuki role"', "Nana Mizuki role"),
        ("'Nana Mizuki role'", "Nana Mizuki role"),
        # Leading blank lines are skipped before the first meaningful line.
        ("\n\nNana Mizuki role\n", "Nana Mizuki role"),
        # Nothing usable stays empty, so the caller falls back to the original question.
        ("", ""),
        ("```\n```", ""),
    ],
)
def test_clean_translation_keeps_the_first_meaningful_line(raw: str, expected: str) -> None:
    assert clean_translation(raw) == expected


def test_translated_query_reports_whether_anything_changed() -> None:
    assert TranslatedQuery(original="水樹奈々の役", english="Nana Mizuki role").changed is True
    assert (
        TranslatedQuery(original="Nana Mizuki role", english="  Nana   Mizuki Role ").changed
        is False
    )


def test_translation_prompt_requires_romanisation_but_keeps_latin_names() -> None:
    """The two rules a measured failure produced: romanise kana, copy Latin-script names."""
    assert "romanised" in TRANSLATE_SYSTEM_PROMPT
    assert "copied exactly" in TRANSLATE_SYSTEM_PROMPT
    assert "normal word order" in TRANSLATE_SYSTEM_PROMPT
