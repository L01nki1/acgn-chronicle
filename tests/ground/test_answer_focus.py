"""The answer-focus rule: appended only, off by default, and it composes with quote discipline.

A human labelling pass found answers that state the right fact but bury it in a longer sentence
or demote it to a parenthetical. The rule that addresses it must stay append-only so the
existing prompt contract keeps holding, and it must stay off until an A/B adopts it.
"""

from __future__ import annotations

import pytest

from acgn_rag.core.ground import generate
from acgn_rag.core.ground.policy import FOCUS_RULE, QUOTE_DISCIPLINE_RULE


@pytest.fixture(autouse=True)
def _clear_switches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ACGN_ANSWER_QUOTE_DISCIPLINE", raising=False)
    monkeypatch.delenv("ACGN_ANSWER_FOCUS", raising=False)


def test_answer_focus_is_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Adoption is a measurement, not a judgement call: the default prompt must not change."""
    assert FOCUS_RULE not in generate.system_prompt()

    monkeypatch.setenv("ACGN_ANSWER_FOCUS", "1")
    assert (
        generate.system_prompt()
        == generate.SYSTEM_PROMPT + " " + QUOTE_DISCIPLINE_RULE + " " + FOCUS_RULE
    )


def test_focus_composes_with_quote_discipline_in_a_fixed_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ACGN_ANSWER_QUOTE_DISCIPLINE", "0")
    monkeypatch.setenv("ACGN_ANSWER_FOCUS", "1")

    prompt = generate.system_prompt()

    assert prompt == generate.SYSTEM_PROMPT + " " + FOCUS_RULE
    assert FOCUS_RULE not in generate.SYSTEM_PROMPT


def test_both_switches_off_reproduces_the_original_prompt_byte_for_byte(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ACGN_ANSWER_QUOTE_DISCIPLINE", "0")
    monkeypatch.setenv("ACGN_ANSWER_FOCUS", "0")

    assert generate.system_prompt() == generate.SYSTEM_PROMPT


def test_focus_rule_states_the_ordering_constraint_it_was_written_for() -> None:
    """The three failures a labeller actually observed, each named in the rule."""
    # The asked-for fact must lead, not trail.
    assert "first sentence of the first claim" in FOCUS_RULE
    # The worst case demoted the answer to a parenthetical.
    assert "parenthetical" in FOCUS_RULE
    # Unasked-for context was the bulk of the dilution.
    assert "belongs in neither claim" in FOCUS_RULE
