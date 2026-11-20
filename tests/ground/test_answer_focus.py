"""The answer-focus rule: appended only, on by default, and it composes with quote discipline.

A human labelling pass found answers that state the right fact but bury it in a longer sentence
or demote it to a parenthetical. The rule that addresses it must stay append-only so the
existing prompt contract keeps holding. It was adopted on three runs after that measurement and
remains switchable, because the default is a recorded decision rather than a judgement call.
"""

from __future__ import annotations

import pytest

from acgn_rag.core.ground import generate
from acgn_rag.core.ground.policy import FOCUS_RULE, QUOTE_DISCIPLINE_RULE


@pytest.fixture(autouse=True)
def _clear_switches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ACGN_ANSWER_QUOTE_DISCIPLINE", raising=False)
    monkeypatch.delenv("ACGN_ANSWER_FOCUS", raising=False)


def test_answer_focus_is_on_by_default() -> None:
    """Adopted after three runs: 8.31 -> 2.2-2.6 length ratio, 4/27 -> 13-15/26 leading."""
    assert (
        generate.system_prompt()
        == generate.SYSTEM_PROMPT + " " + QUOTE_DISCIPLINE_RULE + " " + FOCUS_RULE
    )


def test_focus_escape_hatch_restores_the_pre_adoption_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ACGN_ANSWER_FOCUS", "0")

    prompt = generate.system_prompt()

    assert prompt == generate.SYSTEM_PROMPT + " " + QUOTE_DISCIPLINE_RULE
    assert FOCUS_RULE not in prompt


def test_focus_composes_with_quote_discipline_in_a_fixed_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ACGN_ANSWER_QUOTE_DISCIPLINE", "0")

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


def test_focus_rule_permits_a_headline_that_is_itself_the_answer() -> None:
    """Rule (f): brevity must not read as an instruction to fragment or extend a good headline.

    The one item still refused across every focus run had a headline that was itself a complete
    answer beginning with the asked-for subject. Without this rule the brevity clauses were read
    as "do not use the headline", the model spliced body text onto it, and the resulting claim
    overran the quote budget - so the whole question abstained.
    """
    assert "headline that already answers the question IS a complete claim" in FOCUS_RULE
    assert "copy it whole and stop" in FOCUS_RULE
    # The two failure modes it names explicitly.
    assert "Never build a longer run out of a headline plus surrounding body text" in FOCUS_RULE
    assert "never let one claim exceed the quote budget" in FOCUS_RULE
    # And it must not contradict the brevity rules: it says so.
    assert "rules (d) and (e) ask for brevity, never for fragmentation" in FOCUS_RULE


def test_permission_to_copy_a_headline_carries_a_refusal_guard() -> None:
    """Rule (g): copying a whole headline must not become licence to answer a different question.

    Adding rule (f) cost one correctly-refused item: a question asked whether an editor had been
    replaced, the evidence reported a hiatus, and the model quoted the hiatus headline as if it
    answered. (f) alone reads as "a headline is enough"; (g) makes relevance distinct from
    answering and names refusal as the correct outcome.
    """
    assert "it never permits answering a question the headline does not answer" in FOCUS_RULE
    assert "Relevant is not the same as answering." in FOCUS_RULE
    # It must point at the actual escape hatch, not merely advise caution.
    assert '{"claims": []}' in FOCUS_RULE
    # And name the concrete confusion that produced the false answer.
    assert "asks whether an editor changed" in FOCUS_RULE
