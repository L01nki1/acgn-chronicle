"""W1 grounded-prompt policy text; the W2 prompt builder must include it verbatim."""

from __future__ import annotations

import os

MEDIA_SOURCE_RULE = (
    "If a source's `source_type` is `media`, extract only verifiable facts and figures; "
    "never paraphrase its commentary, reviews or speculation. If the source is `official`, "
    "official statements and facts may be restated."
)

#: C4 makes a media citation valid only when `text` is a verbatim run of the cited chunk. A
#: live run showed the model failing that check for two distinct reasons, so this instruction
#: addresses the mechanics rather than repeating "be verbatim":
#:
#:   - B-005 "Studio KAI is animating hololive's first television anime project" - the model
#:     rewrote the sentence (0.545 similarity to the closest source run). The instruction now
#:     says to COPY, and to quote the run that answers the question even when it carries
#:     surrounding words the question did not ask about.
#:   - C-005 dropped "(Darwin Jihen)" from the middle of an otherwise identical sentence
#:     (0.878 similarity). The instruction now forbids deleting anything from the middle.
#:
#: Adopted on measurement (2026-11-20): 7 delivery-set runs, 4 without and 3 with. Answerable
#: false refusals fell from 5.5/27 to 0.7/27 mean, and every "on" run was at or below the best
#: "off" run. The mechanism is visible directly: of the claims the model generated, the share
#: failing the verbatim C4 check fell from 33.1% (48/145) to 13.8% (15/109). Unanswerable items
#: stayed refused (28/28 vs 21/21) and no run produced a false answer. It changes no validation
#: rule - only what the model is asked for.
QUOTE_DISCIPLINE_RULE = (
    "For a claim citing media, `text` must be COPIED, not written: an exact, unbroken run of "
    "characters taken from the cited block's `text` or `title`. Concretely: "
    "(a) do not reword the sentence into your own phrasing, even to make it read better; "
    "(b) do not drop words from the middle of the run - parentheses, subtitles, publisher "
    "names and parenthetical romanisations stay in; "
    "(c) do not splice two passages together, and do not join the run to your own sentence; "
    "(d) it is fine to copy a run that contains extra words the question did not ask about - "
    "extra context is acceptable, a changed word is not; "
    "(e) strip only leading and trailing whitespace. "
    "A claim whose `text` is not an exact substring of the cited block is discarded by "
    "validation and the question goes unanswered, so quote what is on the page."
)

#: Answers were answering the question and then burying it. A human labelling pass of the
#: delivery set found the failure is structural, not verbal: across 27 answered items the mean
#: gold is 27 characters and the mean answer 108 — 4x — and 23 of 27 do not open with the fact
#: that was asked for. The worst case put the asked-for keyword in parentheses and made the
#: subject of the sentence something the question never asked about ("The project (under the
#: abbreviated keyword "Odeholo") is part of hololive's 10th anniversary celebration").
#:
#: The cause is the interaction of two standing rules, not a careless model: "use only the
#: evidence" plus "a media claim must be a verbatim run" makes copying a whole published
#: sentence the safe move, because a whole sentence can never fail the substring check while a
#: phrase might. Measured on the real index, 10 of 26 gold facts DO pass the verbatim check as
#: short answers ("Odeholo", "Studio KAI", "2027"), so precision is reachable — the prompt just
#: never asked for it. This paragraph asks, and states the ordering constraint explicitly.
FOCUS_RULE = (
    "Answer the question that was asked, not the paragraph it came from. "
    "(a) The first sentence of the first claim must state the asked-for fact as its subject or "
    "object - the date, the name, the number - not the surrounding event. "
    '(b) Never demote the answer to a parenthetical, a relative clause or a trailing "which also" '
    "detail: if the model's only route to a fact is a sentence whose subject is something else, "
    "quote the shortest verbatim run that still contains the fact, not the whole sentence. "
    "(c) Add a second claim only when the question asks two things. Context that the question did "
    "not ask for belongs in neither claim. "
    "(d) A three-to-six word verbatim run IS a complete answer when it answers the question. "
    '"Odeholo", "Studio KAI", "2027" and "three more cast members" are all valid claims on their '
    "own: the citation carries the verifiability, so the claim does not have to carry the context. "
    "Copying a whole sentence to look complete is worse than being brief. "
    "(e) Prefer the shortest verbatim run that answers the question exactly. Do not append "
    "publisher names, dates, background or 'also' clauses that the question did not ask for."
)


def _flag(name: str, default: bool) -> bool:
    """Read a boolean switch; an unset or blank variable takes ``default``.

    Anything that is not an explicit truthy literal is off, so a typo can never silently turn
    the adopted rule back on when the operator meant to disable it.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def answer_focus_enabled() -> bool:
    """Whether the answer-focus paragraph is appended.

    Off by default until measured: the quote-discipline precedent is that a prompt rule is
    adopted on an A/B with repeats, not on how well it reads. ``ACGN_ANSWER_FOCUS=1`` enables it.
    """
    return _flag("ACGN_ANSWER_FOCUS", default=False)


def quote_discipline_enabled() -> bool:
    """Whether the quote-discipline paragraph is appended. **On by default** (measured adoption).

    ``ACGN_ANSWER_QUOTE_DISCIPLINE=0`` is the escape hatch back to the pre-2026-11-20 prompt.
    """
    return _flag("ACGN_ANSWER_QUOTE_DISCIPLINE", default=True)


def media_rule_for_prompt(max_quote_chars: int | None = None) -> str:
    """The media-citation sentence(s) this process sends.

    With quote discipline on (the default) the copy mechanics follow the standing media rule;
    with ``ACGN_ANSWER_QUOTE_DISCIPLINE=0`` this returns exactly the pre-adoption sentence, so
    the on/off comparison remains one appended paragraph.
    """
    limit = f" of at most {max_quote_chars} characters" if max_quote_chars else ""
    base = (
        MEDIA_SOURCE_RULE
        + f" For a claim citing media, put only a verbatim source excerpt{limit} in text, "
        "without an attribution prefix. Cite only the media blocks that contain that excerpt."
    )
    if quote_discipline_enabled():
        return base + " " + QUOTE_DISCIPLINE_RULE
    return base


__all__ = [
    "MEDIA_SOURCE_RULE",
    "QUOTE_DISCIPLINE_RULE",
    "media_rule_for_prompt",
    "quote_discipline_enabled",
]
