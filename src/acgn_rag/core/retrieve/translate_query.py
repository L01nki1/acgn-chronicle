"""Query translation for cross-language retrieval.

Owner: D | Spec: REBUILD_PLAN §4.3 (reopened by measurement)

The protocol once rejected a translation lane on the strength of a 23-item run where
translate-then-retrieve scored hit@5 0.0870 against 0.2609 for direct retrieval. A later
measurement on this corpus contradicts that as a general conclusion: on the three Chinese
items the delivery set answers from English ANN pages, direct retrieval reached 0/3 (the
query landed on chuapp/nijimen pages), while an English rendering of the same question
reached 3/3 and put the gold page at rank 1. The earlier run conflated two different
failures, so this module keeps the experiment cheap and reversible rather than adopting
translation wholesale.

Design constraints:

  - Retrieval only. The translated text never reaches the answer prompt as evidence, so a
    bad translation can move a result but cannot become a citation.
  - Proper nouns survive. Titles, slugs and product names are what the index matches on, so
    the prompt asks for the English *search* phrasing and explicitly forbids translating or
    transliterating names.
  - One extra call per question, only when it can help: the caller decides, this module never
    guesses whether a question "looks Chinese".
  - Failure is not fatal. Any provider error returns None, and the caller falls back to the
    original query text, which is exactly the current behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass

TRANSLATE_SYSTEM_PROMPT = (
    "Rewrite the user's question as an English search query for an anime and games news "
    "archive. Rules: "
    "1. A name written in Latin script (series titles, game titles, studio names, publisher "
    "names, character names, voice-actor names) must be copied exactly, never translated or "
    "respun. "
    "2. A name written in Japanese or Chinese script must be romanised the way an English news "
    "site would write it - a Japanese voice actor's name becomes its standard romaji, and a "
    "Japanese title becomes its official English title when one exists. An index of English "
    "articles does not contain the kanji or kana spelling, so leaving it untranslated makes "
    "the name unfindable. "
    "3. Translate the surrounding question words into English. "
    "4. Write a short, natural English phrase in normal word order - the way an article "
    "headline or a plain English version of the question reads. Do not produce a bag of "
    "keywords, and do not end with a bare question word such as 'who' or 'when'. "
    "5. Do not answer the question, do not add facts, do not invent titles. "
    "6. Reply with the query text only."
)


@dataclass(frozen=True, slots=True)
class TranslatedQuery:
    """The original question plus the English retrieval phrasing derived from it."""

    original: str
    english: str

    @property
    def changed(self) -> bool:
        return _normalize(self.original) != _normalize(self.english)


def _normalize(text: str) -> str:
    return " ".join(text.split()).strip().lower()


#: Why a translation attempt produced no phrasing. `ok` is the only success value.
#:
#: `provider_error` is the one that matters operationally: the translator never reached the
#: model, so every question falls back to its original text and the lane measured nothing. A run
#: that reports this for every row was not measuring translation at all.
TRANSLATION_MISS_REASONS = ("ok", "provider_error", "empty_translation", "empty_question")


@dataclass(frozen=True)
class TranslationAttempt:
    """One translation attempt: the phrasing if any, plus why it is absent when it is.

    ``detail`` carries the exception text or the finish reason. It exists so a failed lane is
    diagnosable from the archive alone instead of requiring a local reproduction.
    """

    translated: TranslatedQuery | None
    reason: str
    detail: str | None = None

    @property
    def used(self) -> bool:
        return self.translated is not None


def build_translation_prompt(question: str) -> tuple[str, str]:
    return TRANSLATE_SYSTEM_PROMPT, f"Question:\n{question}"


def clean_translation(raw: str) -> str:
    """Keep the reply's first meaningful line; strip quoting, fences and list punctuation.

    A fenced reply is unwrapped to the text *between* the fences. Splitting on the fence and
    taking the last segment instead made a well-formed ```query``` reply empty (the translation
    was silently lost) while any prose after the closing fence became the query — which is how a
    stray "Here is your query." could become a retrieval query.
    """
    text = raw.strip()
    fence = "```"
    if fence in text:
        body = text.split(fence, 1)[1]
        # The closing fence may be absent (a truncated reply): then the tail is the body.
        text = body.split(fence, 1)[0].strip()
        if not text:
            text = raw.strip().strip(fence).strip()
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    return line.strip("\"“”'` ").strip()


def translate_query(
    question: str,
    client,
    *,
    max_tokens: int = 512,
) -> TranslatedQuery | None:
    """Return the English retrieval phrasing, or None when translation is unavailable.

    ``client`` is the answer-line transport (``complete(system, user, max_tokens=...)``).
    Returning None on any exception keeps a provider outage from turning into a retrieval
    outage: the caller re-uses the original question.

    The budget is not about the short query text: a reasoning model spends tokens thinking
    before it answers, and 256 was observed to end with ``finish_reason=length`` and no
    content on the longest question. 512 covers that reasoning plus the line we keep.

    This is the thin wrapper kept for callers that only want the text. Use
    :func:`attempt_translation` when the reason for a miss matters: ``None`` here conflates a
    dead provider with an empty completion, and that conflation is what let a run report
    ``used: false`` on every Chinese row without anyone noticing the translator had never
    worked.
    """
    return attempt_translation(question, client, max_tokens=max_tokens).translated


def attempt_translation(
    question: str,
    client,
    *,
    max_tokens: int = 512,
) -> TranslationAttempt:
    """Return the phrasing **and why it was or was not produced**.

    The empty question, the raising provider and the empty completion all used to become the
    same ``None``, so an archive could not distinguish "translation is not needed here" from
    "the translator is broken". They are separate outcomes here, and the exception text is
    carried through so a reviewer can see a 530 or a missing session header in the archive
    rather than having to re-derive it.
    """
    question = question.strip()
    if not question:
        return TranslationAttempt(None, "empty_question")

    system, user = build_translation_prompt(question)
    try:
        completion = client.complete(system, user, max_tokens=max_tokens)
    except Exception as exc:
        # Deliberately still not fatal: the caller re-uses the original query. The difference
        # is that the failure now leaves a trace instead of vanishing.
        return TranslationAttempt(None, "provider_error", f"{type(exc).__name__}: {exc}")

    english = clean_translation(getattr(completion, "text", "") or "")
    if not english:
        finish = getattr(completion, "finish_reason", None)
        return TranslationAttempt(None, "empty_translation", f"finish_reason={finish}")
    return TranslationAttempt(TranslatedQuery(original=question, english=english), "ok")


__all__ = [
    "TRANSLATE_SYSTEM_PROMPT",
    "TRANSLATION_MISS_REASONS",
    "TranslatedQuery",
    "TranslationAttempt",
    "attempt_translation",
    "build_translation_prompt",
    "clean_translation",
    "translate_query",
]
