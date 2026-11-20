"""Run identity must reflect which prompt rules were actually in effect.

Both prompt rules are switchable and default-on, so a run with one disabled would otherwise
record the same `answerer_version` as a run with it enabled and the two would compare as
identical. B raised this on #116: an A/B arm that clears a default-on flag is not a control.
"""

from __future__ import annotations

import pytest

from acgn_rag.core.ground import generate
from acgn_rag.core.ground.generate import ANSWERER_VERSION


@pytest.fixture(autouse=True)
def _clear_switches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ACGN_ANSWER_QUOTE_DISCIPLINE", raising=False)
    monkeypatch.delenv("ACGN_ANSWER_FOCUS", raising=False)


def test_default_records_the_plain_prompt_version() -> None:
    """Both rules on is the adopted default, and it keeps the documented v1 slug."""
    assert generate.answerer_version() == ANSWERER_VERSION


def test_disabling_focus_changes_the_recorded_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACGN_ANSWER_FOCUS", "0")

    assert generate.answerer_version() == f"{ANSWERER_VERSION}+no-focus"


def test_disabling_quote_discipline_changes_the_recorded_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ACGN_ANSWER_QUOTE_DISCIPLINE", "0")

    assert generate.answerer_version() == f"{ANSWERER_VERSION}+no-quote-discipline"


def test_every_combination_is_distinct(monkeypatch: pytest.MonkeyPatch) -> None:
    """Four switch states, four identities: no two runs can be confused for each other."""
    seen: dict[str, str] = {}
    for quote in ("1", "0"):
        for focus in ("1", "0"):
            monkeypatch.setenv("ACGN_ANSWER_QUOTE_DISCIPLINE", quote)
            monkeypatch.setenv("ACGN_ANSWER_FOCUS", focus)
            seen[f"quote={quote},focus={focus}"] = generate.answerer_version()

    assert len(set(seen.values())) == 4, seen


def test_identity_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two calls in one process agree, so the recorded value is not order-dependent."""
    assert generate.answerer_version() == generate.answerer_version()

    monkeypatch.setenv("ACGN_ANSWER_FOCUS", "0")
    assert generate.answerer_version() == generate.answerer_version()
