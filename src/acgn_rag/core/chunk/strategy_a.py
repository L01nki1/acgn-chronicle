"""Config A chunking: CJK-aware sentence packing under a content-token budget.

Owner: F | Spec: 04 §4.3, issue #8 (A/A'/A0 are parameter variants of one implementation)
Status: implemented; strategy B (event aggregation) stays a placeholder until A has cluster ids

Rules this module promises (the tests check each one):

- Content budget (RUNBOOK §3.4): ``budget_for`` = min(config_tokens, max_seq_length -
  len(tokenizer.ids(prefix))). ``ids()`` includes the model's special tokens, so they are
  paid for exactly once: e5-base turns "passage: " into 4 ids (2 content + <s>/</s>),
  giving a budget of 508 at max_seq_length 512 - never a hard-coded constant. The
  declared config binds and the model only caps, so an 8k-context lane must not turn
  config A into an 8k chunker, or the A/A' comparison's independent variable becomes
  encoder-dependent.
- Sentence boundaries (``split_sentences``): a sentence ends after the CJK terminators
  (ideographic full stop, the fullwidth exclamation and question marks, ASCII ! and ?,
  the ellipsis), after a run of them ("……" is one terminator, not two cuts) plus any
  closing quote or bracket that directly follows (a quoted line stays with its sentence),
  and after a newline (hard boundary; blank segments are dropped). An ASCII period ends a
  sentence only when it is not a decimal point: the next character must be whitespace or
  end-of-text and the neighbouring characters must not both be digits, so "3.5" and
  "1.2億" never split while "in Q4." does. There is no abbreviation dictionary, so
  "U.S.A." may over-split; packing respects the budget either way.
- Packing: whole sentences are packed greedily, joined with a single space (whitespace
  tokenizers count the joiner for free; SP tokenizers fold it into the following token),
  and the budget check is always a fresh ``count`` of the joined text, so the joiner can
  never cause overflow.
- Oversize sentence (a single sentence longer than the whole budget): split at token
  boundaries through tokenizer.ids/decode, every piece re-verified with ``count``. Every
  piece after the first is sized for budget minus overlap, because it will be packed
  after an overlap head. Without ``decode``, a deterministic character-halving fallback
  applies.
- Overlap (variants with overlap > 0): the next chunk begins with the previous chunk's
  trailing ``overlap`` content tokens (decoded from the previous chunk's ids) and the
  packing budget counts them. If the head cannot share a chunk with the next piece, the
  head is dropped: a chunk must carry new content, and a head-only chunk would duplicate
  the previous chunk's tail verbatim.
- Deterministic: no I/O, no globals, no randomness; identical input gives identical
  output. ``chunk_id`` is ``doc_id::variant::<1-based index>``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from acgn_rag.core.chunk.variants import CHUNK_VARIANTS

# CJK sentence terminators. The two fullwidth marks are deliberate - they are what ja/zh
# news text actually ends sentences with; noqa keeps ruff's confusables list quiet.
_SENTENCE_TERMINATORS = "。！？!?…"  # noqa: RUF001
# Closing quotes/brackets that belong to the sentence that just ended (「…。」 etc.).
_CLOSERS = "」』）)\"'”’》〉】"  # noqa: RUF001


@dataclass(frozen=True)
class Chunk:
    """One packed chunk of one document under one variant; content only, no e5 prefix.

    ``token_count`` is content tokens (specials excluded) under the same tokenizer that
    produced the budget - the number RUNBOOK §3.4's acceptance encodes.
    """

    text: str
    chunk_id: str
    token_count: int


class TokenCounter(Protocol):
    """The tokenizer surface the chunker needs: an HF adapter in production, a fake in CI.

    ``count(text)`` returns CONTENT tokens, specials excluded - the number the packing
    budget governs. ``ids(text)`` returns the full id sequence the model would produce,
    specials included, so ``len(ids(prefix))`` is exactly what max_seq_length must pay
    for. Both must come from the encoder lane's tokenizer (e5 = XLM-R, never tiktoken):
    mixing tokenizers between the budget and the packing is the classic silent-truncation
    bug (RUNBOOK §3.4). Implementations place all specials at the two ends of ``ids()``
    (the HF standard); the chunker derives their number as ``len(ids) - count`` instead
    of assuming a layout. ``decode`` is optional at runtime: without it the chunker falls
    back to character halving (see ``DecodingTokenCounter`` for the exact-cut contract).
    """

    def count(self, text: str) -> int: ...

    def ids(self, text: str) -> list[int]: ...


class DecodingTokenCounter(TokenCounter, Protocol):
    """A TokenCounter that can also decode ids back to text, enabling exact token cuts.

    ``decode`` must skip model specials, so decoding a prefix or suffix slice of
    ``ids(text)`` returns the text of its content tokens only.
    """

    def decode(self, ids: Sequence[int]) -> str: ...


def budget_for(
    config_tokens: int, max_seq_length: int, prefix: str, tokenizer: TokenCounter
) -> int:
    """Content budget for one chunk: the config binds, the model only caps (RUNBOOK §3.4).

    Uses ``len(tokenizer.ids(prefix))`` - specials included - so <s>/</s> are counted
    exactly once: e5-base + "passage: " gives 4 ids, hence a budget of 508 at
    max_seq_length 512. Using the specials-free ``count`` here would under-pay by the
    specials and quietly reintroduce the truncation trap this formula exists to close.
    """
    return min(config_tokens, max_seq_length - len(tokenizer.ids(prefix)))


def split_sentences(text: str) -> list[str]:
    """Split into sentences per the module docstring; stripped, blank segments dropped."""
    sentences: list[str] = []
    start = 0
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char == "\n" or char == "\r":
            if char == "\r" and text[index + 1 : index + 2] == "\n":
                index += 1  # CRLF is one boundary
            index += 1
        elif char in _SENTENCE_TERMINATORS:
            index = _skip_run(text, index + 1, _SENTENCE_TERMINATORS)
            index = _skip_run(text, index, _CLOSERS)
        elif char == "." and _period_ends_sentence(text, index):
            index = _skip_run(text, index + 1, _CLOSERS)
        else:
            index += 1
            continue
        sentences.append(text[start:index])
        start = index
    sentences.append(text[start:])
    return [part for part in (piece.strip() for piece in sentences) if part]


def chunk_text(
    doc_id: str,
    text: str,
    variant: str,
    tokenizer: TokenCounter,
    max_seq_length: int = 512,
    passage_prefix: str = "passage: ",
) -> list[Chunk]:
    """Chunk one document under one variant: "A" (512/64), "A'" (256/0) or "A0" (512/0).

    Pure text -> chunks, no I/O: the variant only supplies (max_tokens, overlap) - never
    three pipelines (issue #8). Chunks carry content only; the e5 passage prefix is added
    at encode time and paid for by ``budget_for``.
    """
    try:
        config = CHUNK_VARIANTS[variant]
    except KeyError:
        known = ", ".join(repr(name) for name in sorted(CHUNK_VARIANTS))
        raise KeyError(f"unknown chunk variant {variant!r}; known variants: {known}") from None
    budget = budget_for(config.max_tokens, max_seq_length, passage_prefix, tokenizer)
    if budget < 1:
        raise ValueError(
            f"content budget is {budget}: max_seq_length {max_seq_length} cannot hold the "
            f"{len(tokenizer.ids(passage_prefix))}-id prefix"
        )
    if config.overlap >= budget:
        raise ValueError(
            f"variant {variant!r} overlaps {config.overlap} tokens but the budget is "
            f"{budget}: no room left for new content"
        )
    pieces: list[str] = []
    for sentence in split_sentences(text):
        if tokenizer.count(sentence) > budget:
            pieces.extend(_split_oversize(sentence, budget, config.overlap, tokenizer))
        else:
            pieces.append(sentence)
    chunks: list[Chunk] = []
    parts: list[str] = []  # pieces accumulated into the chunk currently being built
    for piece in pieces:
        if not piece:
            continue
        if not parts:
            parts = [piece]
            continue
        if tokenizer.count(" ".join([*parts, piece])) <= budget:
            parts.append(piece)
            continue
        # Every flush consumes `piece` into the next chunk's opening, so the loop always
        # makes progress.
        flushed = " ".join(parts)
        chunks.append(_emit(doc_id, variant, len(chunks) + 1, flushed, tokenizer))
        parts = _next_parts(flushed, piece, config.overlap, budget, tokenizer)
    if parts:
        chunks.append(_emit(doc_id, variant, len(chunks) + 1, " ".join(parts), tokenizer))
    return chunks


def _emit(doc_id: str, variant: str, index: int, text: str, tokenizer: TokenCounter) -> Chunk:
    return Chunk(
        text=text,
        chunk_id=f"{doc_id}::{variant}::{index}",
        token_count=tokenizer.count(text),
    )


def _next_parts(
    flushed: str, piece: str, overlap: int, budget: int, tokenizer: TokenCounter
) -> list[str]:
    """Opening of the next chunk: the overlap head when it fits alongside `piece`.

    The head is dropped when it cannot share a chunk with `piece`: a chunk must carry
    new content, and a head-only chunk would duplicate the previous chunk's tail verbatim.
    """
    if overlap <= 0 or tokenizer.count(flushed) <= overlap:
        return [piece]
    head = _trailing_text(flushed, overlap, tokenizer)
    if tokenizer.count(f"{head} {piece}") <= budget:
        return [head, piece]
    return [piece]


def _split_oversize(text: str, budget: int, overlap: int, tokenizer: TokenCounter) -> list[str]:
    """Cut one over-budget sentence into pieces that each fit; token-exact with `decode`.

    Pieces after the first are sized for budget minus overlap because they are packed
    after an overlap head (see the module docstring).
    """
    decode = _decoder(tokenizer)
    if decode is None:
        return _halve_text(text, budget, tokenizer)
    ids = tokenizer.ids(text)
    specials = len(ids) - tokenizer.count(text)
    pieces: list[str] = []
    start = 0
    capacity = budget  # the first piece may fill a whole chunk on its own
    while start < len(ids):
        size = min(capacity + specials + 1, len(ids) - start)
        piece = decode(ids[start : start + size])
        while tokenizer.count(piece) > capacity and size > 1:
            size -= 1
            piece = decode(ids[start : start + size])
        pieces.append(piece)
        start += size
        capacity = max(1, budget - overlap)
    return pieces


def _halve_text(text: str, budget: int, tokenizer: TokenCounter) -> list[str]:
    """Deterministic fallback without `decode`: halve at the character midpoint until it fits."""
    if tokenizer.count(text) <= budget:
        return [text]
    middle = len(text) // 2
    return _halve_text(text[:middle], budget, tokenizer) + _halve_text(
        text[middle:], budget, tokenizer
    )


def _trailing_text(text: str, token_count: int, tokenizer: TokenCounter) -> str:
    """The trailing `token_count` content tokens of `text`, as text again.

    With `decode`: exact for tokenizers whose ids map one-to-one onto content tokens
    (the CI fake and SP models in practice); the caller always re-measures the returned
    text, so a merge during re-encoding can only shrink what the budget sees. Without
    `decode`: the longest whitespace-word suffix whose count stays within `token_count`
    (best effort - that lane has no ids-to-text map).
    """
    decode = _decoder(tokenizer)
    if decode is not None:
        ids = tokenizer.ids(text)
        specials = len(ids) - tokenizer.count(text)
        size = min(token_count + specials, len(ids))
        piece = decode(ids[-size:])
        while tokenizer.count(piece) > token_count and size > token_count:
            size -= 1
            piece = decode(ids[-size:])
        return piece
    suffix: list[str] = []
    for word in reversed(text.split()):
        candidate = [word, *suffix]
        if tokenizer.count(" ".join(candidate)) > token_count:
            break
        suffix = candidate
    return " ".join(suffix)


def _decoder(tokenizer: TokenCounter) -> Callable[[Sequence[int]], str] | None:
    """`decode` when the tokenizer provides it; the protocol leaves it optional."""
    decode = getattr(tokenizer, "decode", None)
    return decode if callable(decode) else None


def _skip_run(text: str, index: int, chars: str) -> int:
    """Index just past the run of `chars` starting at `index`."""
    while index < len(text) and text[index] in chars:
        index += 1
    return index


def _period_ends_sentence(text: str, index: int) -> bool:
    """ASCII "." ends a sentence iff it is not a decimal point: not between two digits and
    followed by whitespace or end-of-text (so "3.5" and "1.2億" never split)."""
    previous = text[index - 1 : index]
    following = text[index + 1 : index + 2]
    if previous.isdigit() and following.isdigit():
        return False
    return following == "" or following.isspace()


__all__ = [
    "Chunk",
    "DecodingTokenCounter",
    "TokenCounter",
    "budget_for",
    "chunk_text",
    "split_sentences",
]
